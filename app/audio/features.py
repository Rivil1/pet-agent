"""声学特征提取。

对应 docs/DESIGN.md §3.6「声学特征」与「可复现性要求」。

**这是证据链三个来源中的 A（measured）**——从音频直接计算，可复算。

设计要点：
1. 固定重采样率（``TARGET_SR``），保证同一段音频得到同一组特征。
2. F0 用 pyin，并按文献做法剔除 voiced 置信度 < 0.15 的帧。
3. ``f0_slope`` 按 f0 均值归一化，使不同个体的音高差异不影响斜率可比性。
4. 质量评估基于 voiced 帧比例与信噪比估计；低质量时下游必须降低置信度。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas import AcousticFeatures, FeatureQuality

#: 固定重采样率。改变它会使历史特征与新特征不可比。
TARGET_SR = 16_000

#: F0 搜索范围（Hz）。家猫叫声基频大致落在该区间内。
F0_MIN = 200.0
F0_MAX = 1200.0

#: pyin voiced 置信度阈值。文献做法（SWIPE' 置信度 < 0.15 的值被忽略）。
VOICED_PROB_THRESHOLD = 0.15

#: 能量门限相对 p95 包络的比例，用于端点检测。
ENERGY_THRESHOLD_RATIO = 0.10

#: 判为一次独立叫声的最短时长（秒）。
MIN_CALL_SECONDS = 0.05

#: 间隔合并阈值（秒）。短于此间隔的两段视为**同一次叫声**。
#:
#: 必要性：谐波拍频会让单次叫声的包络出现深凹，被误判为多次独立叫声。
#: 实测在一个 0.5s 合成信号上曾得到 20 次/10s 的荒谬速率。
MIN_GAP_SECONDS = 0.08

#: 估计 ``call_rate`` 所需的最短观测窗口（秒）。
#:
#: 叫声速率是「次/10s」的归一化指标，窗口过短时归一化会放大噪声：
#: 0.5s 片段里出现 1 次叫声会被算成 20 次/10s。
#: 窗口不足时**列入 unavailable**，而不是给一个假数字。
MIN_WINDOW_FOR_CALL_RATE = 3.0

_EPS = 1e-10


class AudioTooShort(ValueError):
    """音频过短，无法提取特征。"""


# ─────────────────────────────────────────────────────────────
# 加载
# ─────────────────────────────────────────────────────────────


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """加载音频并重采样到固定采样率。

    单声道、float32、固定采样率 —— 保证特征可比与可复现。
    """
    import librosa  # 延迟导入：librosa 导入较慢，测试其他模块时不需要

    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr)


# ─────────────────────────────────────────────────────────────
# 分段（端点检测）
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segment:
    start: int  # 样点下标
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def _rms_envelope(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """短时 RMS 包络。用 stride 实现，避免依赖 librosa 的 feature 模块。"""
    if len(y) < frame_length:
        pad = frame_length - len(y)
        y = np.pad(y, (0, pad))
    n_frames = 1 + (len(y) - frame_length) // hop_length
    idx = (
        np.arange(frame_length)[None, :]
        + hop_length * np.arange(n_frames)[:, None]
    )
    frames = y[idx]
    return np.sqrt(np.mean(frames**2, axis=1) + _EPS)


def _segment_calls(
    y: np.ndarray,
    sr: int,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> tuple[list[Segment], np.ndarray, int]:
    """基于能量门限做端点检测，切分出独立叫声。

    门限取包络 p95 的固定比例 —— 用分位数而非最大值，避免单个爆音抬高门限。
    """
    env = _rms_envelope(y, frame_length, hop_length)
    if env.size == 0:
        return [], env, hop_length

    threshold = ENERGY_THRESHOLD_RATIO * float(np.percentile(env, 95))
    active = env > threshold

    segments: list[Segment] = []
    min_frames = max(1, int(MIN_CALL_SECONDS * sr / hop_length))
    start: int | None = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            if i - start >= min_frames:
                segments.append(Segment(start * hop_length, i * hop_length))
            start = None
    if start is not None and len(active) - start >= min_frames:
        segments.append(Segment(start * hop_length, len(active) * hop_length))

    return _merge_close_segments(segments, sr), env, hop_length


def _merge_close_segments(segments: list[Segment], sr: int) -> list[Segment]:
    """合并间隔极近的分段。

    谐波拍频会让单次叫声的包络出现深凹，被误判为多次独立叫声。
    真实端点检测必须做这步合并。
    """
    if len(segments) <= 1:
        return segments
    min_gap = int(MIN_GAP_SECONDS * sr)
    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.start - last.end <= min_gap:
            merged[-1] = Segment(start=last.start, end=seg.end)
        else:
            merged.append(seg)
    return merged


# ─────────────────────────────────────────────────────────────
# F0
# ─────────────────────────────────────────────────────────────


def _estimate_f0(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """pyin 基频估计。返回 (f0_valid, voiced_prob)。

    按文献做法剔除置信度低于阈值的帧。
    """
    import librosa

    f0, _voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=F0_MIN,
        fmax=F0_MAX,
        sr=sr,
        frame_length=2048,
        hop_length=256,
        center=True,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=float)
    voiced_prob = np.asarray(voiced_prob, dtype=float)

    # 双保险：置信度达标 **且** 估计值有效
    mask = (voiced_prob >= VOICED_PROB_THRESHOLD) & np.isfinite(f0) & (f0 > 0)
    return f0[mask], voiced_prob


def _f0_slope(f0: np.ndarray) -> float:
    """基频轮廓斜率。

    对（归一化时间, 归一化频率）做最小二乘线性拟合。
    按 f0 均值归一化，使斜率只反映**走向**，不受个体音高高低影响。
    """
    if f0.size < 4:
        return 0.0
    t = np.linspace(0.0, 1.0, f0.size)
    mean = float(np.mean(f0))
    if mean <= 0:
        return 0.0
    y_norm = f0 / mean
    slope = float(np.polyfit(t, y_norm, 1)[0])
    return float(np.clip(slope, -2.0, 2.0))


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────


def extract_features(y: np.ndarray, sr: int) -> AcousticFeatures:
    """从音频波形提取声学特征。

    Raises:
        AudioTooShort: 音频短于判为一次叫声所需的最小时长。
    """
    if sr != TARGET_SR:
        raise ValueError(
            f"采样率必须为 {TARGET_SR}，得到 {sr}。"
            "请用 load_audio() 加载以保证可复现性。"
        )
    if y.size < int(MIN_CALL_SECONDS * sr):
        raise AudioTooShort(f"音频长度 {y.size / sr:.3f}s 短于最小时长 {MIN_CALL_SECONDS}s")

    segments, env, hop = _segment_calls(y, sr)
    f0_valid, voiced_prob = _estimate_f0(y, sr)

    total_seconds = y.size / sr
    unavailable: list[str] = []
    window_too_short = total_seconds < MIN_WINDOW_FOR_CALL_RATE

    # ── duration：优先用分段得到的单次叫声时长；无分段时退回整段时长 ──
    if segments:
        duration = float(np.mean([s.length for s in segments]) / sr)
        n_calls = len(segments)
        if window_too_short:
            # 窗口过短，归一化后的速率是噪声；显式标记不可测，不零填充
            ici = 0.0
            call_rate = 0.0
            unavailable.extend(["call_rate", "ici_mean"])
        else:
            starts = np.array([s.start for s in segments], dtype=float) / sr
            if len(starts) > 1:
                ici = float(np.mean(np.diff(starts)))
            else:
                ici = float(max(total_seconds - duration, 0.0))
                unavailable.append("ici_mean")
            call_rate = n_calls / (total_seconds / 10.0)
    else:
        duration = float(total_seconds)
        ici = 0.0
        call_rate = 0.0
        unavailable.extend(["call_rate", "ici_mean"])

    # ── F0 统计 ──
    if f0_valid.size > 0:
        f0_mean = float(np.mean(f0_valid))
        f0_range = float(np.percentile(f0_valid, 95) - np.percentile(f0_valid, 5))
        f0_slope = _f0_slope(f0_valid)
    else:
        f0_mean = 0.0
        f0_range = 0.0
        f0_slope = 0.0
        unavailable.extend(["f0_mean", "f0_range", "f0_slope"])

    # ── 能量与粗糙度 ──
    rms_mean = float(np.mean(env)) if env.size else 0.0

    # 能量门限：分位数而非最大值，避免单个爆音抬高门限
    threshold = (
        ENERGY_THRESHOLD_RATIO * float(np.percentile(env, 95)) if env.size else 0.0
    )
    active_env = env[env > threshold] if env.size else env
    if active_env.size >= 3:
        # 粗糙度 ≈ 包络一阶差分的相对波动 —— 对应「快速幅度变化」的直觉
        roughness = float(
            np.clip(np.std(np.diff(active_env)) / (np.mean(active_env) + _EPS), 0.0, 2.0)
        )
    else:
        roughness = 0.0

    # ── 质量评估 ──
    voiced_ratio = float(f0_valid.size / max(voiced_prob.size, 1))
    snr_db = _estimate_snr_db(env, threshold)
    quality = _classify_quality(voiced_ratio, snr_db)

    return AcousticFeatures(
        duration=round(duration, 4),
        f0_mean=round(f0_mean, 2),
        f0_range=round(f0_range, 2),
        f0_slope=round(f0_slope, 4),
        call_rate=round(call_rate, 4),
        ici_mean=round(ici, 4),
        rms_mean=round(rms_mean, 6),
        roughness=round(roughness, 4),
        estimated_snr_db=round(snr_db, 2) if snr_db is not None else None,
        quality=quality,
        unavailable=sorted(set(unavailable)),
    )


def _estimate_snr_db(env: np.ndarray, threshold: float) -> float | None:
    """信噪比估计：**活跃段能量 vs 静默段能量**。

    早期实现用包络的 p95/p20 相除，但 p20 并非真正的静音（包络平滑后不会接近 0），
    会把 SNR 系统性低估。改为按门限划分活跃/静默两段。
    """
    if env.size < 10:
        return None
    active = env[env > threshold]
    noise = env[env <= threshold]
    if active.size < 3 or noise.size < 3:
        # 无静音段（整段都是叫声）→ 无法估计噪声底，不编造数字
        return None
    noise_power = float(np.mean(noise**2))
    if noise_power <= _EPS:
        return None
    snr = 10.0 * np.log10(float(np.mean(active**2)) / noise_power)
    return float(np.clip(snr, -20.0, 60.0))


def _classify_quality(voiced_ratio: float, snr_db: float | None) -> FeatureQuality:
    """特征质量分级。低质量时下游必须降低置信度（docs/DESIGN.md §5.3 约束 3）。

    注意 SNR 为 ``None`` 的语义：它表示**无法估计**（例如整段都是叫声，
    没有静音段可作噪声底），**不等于音频质量差**。

    早期实现直接把 ``snr_db is None`` 归为 POOR，那是**把估计器的局限
    记在了音频头上** —— 一段干净的完整叫声会被误判为低质量。
    正确做法：改用可测量的 ``voiced_ratio`` 判定，但**提高门槛作为补偿**，
    以弥补缺失的那个信号。
    """
    if snr_db is None:
        if voiced_ratio >= 0.50:
            return FeatureQuality.GOOD
        if voiced_ratio >= 0.20:
            return FeatureQuality.FAIR
        return FeatureQuality.POOR
    if voiced_ratio < 0.05 or snr_db < 6.0:
        return FeatureQuality.POOR
    if voiced_ratio < 0.15 or snr_db < 12.0:
        return FeatureQuality.FAIR
    return FeatureQuality.GOOD


# ─────────────────────────────────────────────────────────────
# 合成音频（测试与演示用）
# ─────────────────────────────────────────────────────────────


def synthesize_meow(
    duration: float = 0.5,
    sr: int = TARGET_SR,
    f0_start: float = 500.0,
    f0_end: float = 700.0,
    roughness: float = 0.0,
    harmonics: int = 6,
    seed: int = 0,
) -> np.ndarray:
    """合成一段类猫叫信号，**仅用于测试与离线演示**。

    不是真实猫叫的声学模型，只是让特征提取链路可在无音频素材时被验证。
    """
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.linspace(0.0, duration, n, endpoint=False)

    # 基频轮廓：线性扫频 + 轻微抖动
    f0 = np.linspace(f0_start, f0_end, n)
    if roughness > 0:
        jitter = rng.normal(0.0, roughness * f0_start * 0.05, n)
        f0 = f0 + jitter

    phase = 2.0 * np.pi * np.cumsum(f0) / sr
    y = np.zeros(n)
    for h in range(1, harmonics + 1):
        y += np.sin(h * phase) / h

    # 起落包络（避免瞬态引入虚假高频）
    env = np.sin(np.pi * t / duration) ** 0.5
    y *= env
    y *= 0.3 / (np.max(np.abs(y)) + _EPS)
    return y.astype(np.float32)
