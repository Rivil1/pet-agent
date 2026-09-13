#!/usr/bin/env python3
"""从 CatMeows 构建群体先验（群体先验的**唯一合法来源**）。

## 为什么必须用我们自己的提取器

`docs/DESIGN.md` 的先验表是「**每个特征在每个情境下的高斯参数**」，
而这些特征是由 `app/audio/features.py` 定义的：

    duration / f0_mean / f0_range / f0_slope / call_rate / ici_mean / rms_mean / roughness

拿论文里报的统计量填进去是**不可用**的 —— 论文用的提取器不同
（f0 估计算法、平滑窗长、是否剔除无声帧、单位定义都可能不一样）。
填进去之后，后验会基于一套「生产代码根本测不出那个数」的分布来算，
而**不会有任何报错**。

所以：下载原始音频 → 用我们的提取器跑一遍 → 统计。

## 三件必须诚实处理的事

1. **CatMeows 只有 3 类情境**（`brushing` / `isolation_unfamiliar_environment` /
   `waiting_for_food`），而 `ContextLabel` 有 6 类。
   缺的那三类（`greeting` / `door_attention` / `other`）**没有数据** ——
   本脚本不会为它们编造数值。

2. **留出猫（hold-out）**：`DESIGN §6.1` 要求「21 只猫里留 4 只做 hold-out，
   不参与先验统计」。否则测出来的是「拟合先验的能力」，不是泛化能力。

3. **测不出的特征**：`call_rate` / `ici_mean` 需要一段连续录音里的多次叫声，
   而 CatMeows 的片段大多是单次叫声 —— 它们在多数样本上会出现在
   `unavailable` 里。脚本会把**可测样本数**一并报出来，
   而不是悄悄用一个样本量不足的均值。

## 用法

    # 先看数据长什么样（不写文件）
    python scripts/build_priors.py --explore /path/to/catmeows/wav

    # 构建先验并留出验证集
    python scripts/build_priors.py /path/to/catmeows/wav \\
        --out data/priors/catmeows_stats.json --holdout 4

    # 只在留出猫上评估分类（E21）
    python scripts/build_priors.py /path/to/catmeows/wav --eval-only

文件名格式（CatMeows 原始数据集）::

    F_MAG01_EU_FN_FED01_106.wav
    │ │     │  └─ 录制会话
    │ │     └─ 品种：EU=欧洲短毛 / MC=缅因
    │ └─ 猫 ID  ← 留出划分按它做
    └─ 情境：F=等食 / I=隔离 / B=梳毛
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

#: 文件名里的情境首字母 → `ContextLabel`。
#:
#: ⚠️ 这是**全部**能映射的情境。CatMeows 没有 `greeting` / `door_attention` / `other`。
CONTEXT_FROM_PREFIX = {
    "F": "food_waiting",
    "I": "isolation_distress",
    "B": "affection_brushing",
}

#: 数据集的权威描述（写进先验文件的 provenance 段）。
SOURCE = {
    "dataset": "CatMeows",
    "url": "https://zenodo.org/records/4008297",
    "doi": "10.5281/zenodo.4008297",
    "license": "CC-BY-4.0",
    "citation": (
        "Ludovico, R., et al. (2020). CatMeows: A Publicly-Available Dataset of "
        "Cat Vocalizations. Zenodo."
    ),
    "mirror_used": "https://huggingface.co/datasets/zeddez/CatMeows",
}

#: 8 个特征。必须与 `app.interpreter.priors.FEATURE_ORDER` 一致 ——
#: 不一致时 `PriorTable.load()` 会报「缺少特征」，而那是加载期才发现。
FEATURE_ORDER = (
    "duration",
    "f0_mean",
    "f0_range",
    "f0_slope",
    "call_rate",
    "ici_mean",
    "rms_mean",
    "roughness",
)

_NAME_RE = re.compile(r"^([FIB])_([A-Z]+\d+)_([A-Z]{2})_")


@dataclass(frozen=True)
class Clip:
    """一条录音及其元数据。"""

    path: Path
    context: str
    cat_id: str
    breed: str


@dataclass
class Sample:
    """一条成功提取的样本。"""

    clip: Clip
    values: dict[str, float]
    unavailable: set[str]
    quality: str


@dataclass
class ExtractionReport:
    """提取阶段的账面。**每个数字都要能被解释。**"""

    total_files: int = 0
    parsed: int = 0
    unparsable: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    too_short: list[str] = field(default_factory=list)
    ok: list[Sample] = field(default_factory=list)

    def per_context(self) -> dict[str, list[Sample]]:
        out: dict[str, list[Sample]] = defaultdict(list)
        for s in self.ok:
            out[s.clip.context].append(s)
        return dict(out)

    def availability(self) -> dict[str, Counter]:
        """每个情境下，各特征「测得出」的次数。

        **这是最重要的一张表。** 一个特征只在 10% 的样本上测得出时，
        它的均值不可用 —— 而那个均值看起来和别的一样正常。
        """
        out: dict[str, Counter] = {}
        for ctx, samples in self.per_context().items():
            avail: Counter = Counter()
            for s in samples:
                for f in FEATURE_ORDER:
                    if f not in s.unavailable:
                        avail[f] += 1
            out[ctx] = avail
        return out


# =============================================================================
# 扫描与提取
# =============================================================================


def parse_name(path: Path) -> Clip | None:
    """从文件名解析元数据。解析不出返回 `None`（**不猜**）。"""
    m = _NAME_RE.match(path.stem)
    if not m:
        return None
    prefix, cat_id, breed = m.groups()
    context = CONTEXT_FROM_PREFIX.get(prefix)
    if context is None:
        return None
    return Clip(path=path, context=context, cat_id=cat_id, breed=breed)


def extract_all(directory: Path, *, verbose: bool = True) -> ExtractionReport:
    """跑一遍全量提取。"""
    import librosa

    from app.audio.features import (
        TARGET_SR,
        AudioTooShort,
        extract_features,
    )

    report = ExtractionReport()
    files = sorted(directory.glob("*.wav"))
    report.total_files = len(files)
    if not files:
        raise SystemExit(f"{directory} 下没有 .wav 文件")

    for i, path in enumerate(files, 1):
        clip = parse_name(path)
        if clip is None:
            report.unparsable.append(path.name)
            continue
        report.parsed += 1

        try:
            y, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
        except Exception as exc:  # noqa: BLE001 - 坏文件是数据集的一部分
            report.unreadable.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue

        try:
            feats = extract_features(np.asarray(y, dtype=np.float32), TARGET_SR)
        except AudioTooShort as exc:
            report.too_short.append(f"{path.name}: {exc}")
            continue

        report.ok.append(
            Sample(
                clip=clip,
                values={f: float(getattr(feats, f)) for f in FEATURE_ORDER},
                unavailable=set(feats.unavailable),
                quality=str(getattr(feats.quality, "value", feats.quality)),
            )
        )

        if verbose and i % 100 == 0:
            print(f"  … {i}/{len(files)}", file=sys.stderr)

    return report


# =============================================================================
# 留出划分
# =============================================================================


def split_cats(
    cats: Sequence[str], *, holdout: int, seed: int
) -> tuple[list[str], list[str]]:
    """按猫划分训练 / 留出。

    **按猫而不是按样本**划分。按样本划分会让同一只猫的录音同时出现在
    两边 —— 而那测的是「认不认得这只猫」，不是「能不能泛化到新猫」。
    这个区别决定了 E21 那个数字到底意味着什么。
    """
    ordered = sorted(cats)
    if holdout <= 0:
        return ordered, []
    holdout = min(holdout, max(0, len(ordered) - 1))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ordered))
    held = sorted(ordered[i] for i in idx[:holdout])
    train = sorted(c for c in ordered if c not in set(held))
    return train, held


# =============================================================================
# 统计
# =============================================================================


def gaussian_stats(values: Sequence[float]) -> tuple[float, float]:
    """均值与**样本标准差**（ddof=1）。

    用 ddof=1 而不是 0：这是样本，不是总体。n 很小时
    （某些特征可能只有几个可测样本）ddof=0 会系统性地低估离散度，
    而低估离散度会让后验**过于自信** —— 正好是这套系统要避免的。
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("空样本无法统计")
    if arr.size == 1:
        # 单样本时样本标准差无定义。返回 0 会让 `PriorTable.load()`
        # 报「std 必须为正」，于是问题在加载期暴露 —— 那是刻意的。
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def build_priors(
    report: ExtractionReport,
    *,
    train_cats: Sequence[str],
    holdout_cats: Sequence[str],
    min_samples: int,
    seed: int,
) -> tuple[dict, dict]:
    """从训练猫的样本构建先验表。返回 `(先验 payload, 构建报告)`。"""
    train_set = set(train_cats)
    per_ctx: dict[str, list[Sample]] = defaultdict(list)
    for s in report.ok:
        if s.clip.cat_id in train_set:
            per_ctx[s.clip.context].append(s)

    contexts: dict[str, dict[str, list[float]]] = {}
    build_notes: dict[str, dict[str, object]] = {}

    for ctx, samples in sorted(per_ctx.items()):
        stats: dict[str, list[float]] = {}
        notes: dict[str, object] = {
            "n_clips": len(samples),
            "n_cats": len({s.clip.cat_id for s in samples}),
            "features": {},
        }
        for feat in FEATURE_ORDER:
            vals = [s.values[feat] for s in samples if feat not in s.unavailable]
            n = len(vals)
            if n < min_samples:
                # **样本不足就不给这个特征。** 用一个 n=2 的均值会让后验
                # 看起来像有依据的 —— 而它其实只是两个数的平均。
                notes["features"][feat] = {
                    "status": "insufficient",
                    "n": n,
                    "reason": f"可测样本 {n} < 门槛 {min_samples}",
                }
                continue
            mean, std = gaussian_stats(vals)
            if std <= 0:
                notes["features"][feat] = {
                    "status": "degenerate",
                    "n": n,
                    "reason": f"标准差为 {std}（全部取值相同），无法作为高斯参数",
                }
                continue
            stats[feat] = [round(mean, 6), round(std, 6)]
            notes["features"][feat] = {
                "status": "ok",
                "n": n,
                "coverage": round(n / len(samples), 3),
            }
        contexts[ctx] = stats
        build_notes[ctx] = notes

    # ── base_rates：只在**训练猫**的样本上数，且归一化 ──
    counts = Counter()
    for s in report.ok:
        if s.clip.cat_id in train_set:
            counts[s.clip.context] += 1
    total = sum(counts.values())
    base_rates = {k: round(v / total, 6) for k, v in sorted(counts.items())}

    sha_seed = hashlib.sha256(
        json.dumps({"contexts": contexts, "base_rates": base_rates}, sort_keys=True).encode()
    ).hexdigest()

    payload = {
        "version": f"catmeows-{sha_seed[:8]}",
        "provenance": "catmeows",
        "reviewed": False,  # 未经过兽医审核 —— 群体统计不需要，但不要谎称审核过
        "is_placeholder": False,
        "source": SOURCE,
        "note": _build_note(report, train_cats, holdout_cats, min_samples, seed),
        "feature_specs": {},  # 由调用方从占位文件沿用（单位与口径说明）
        "base_rates": base_rates,
        "contexts": contexts,
        "build": {
            "seed": seed,
            "min_samples": min_samples,
            "train_cats": list(train_cats),
            "holdout_cats": list(holdout_cats),
            "per_context": build_notes,
            "extraction": {
                "extractor": "app.audio.features.extract_features",
                "target_sr": 16000,
                "total_files": report.total_files,
                "parsed": report.parsed,
                "ok": len(report.ok),
                "unreadable": len(report.unreadable),
                "too_short": len(report.too_short),
                "unparsable": len(report.unparsable),
            },
        },
    }
    return payload, build_notes


def _build_note(
    report: ExtractionReport,
    train_cats: Sequence[str],
    holdout_cats: Sequence[str],
    min_samples: int,
    seed: int,
) -> str:
    return (
        "由 scripts/build_priors.py 从 CatMeows 原始音频构建，"
        "特征用本项目的 extract_features 提取（不是论文报的统计量 —— "
        "不同提取器的口径不可互换）。"
        f"训练猫 {len(train_cats)} 只、留出 {len(holdout_cats)} 只（seed={seed}）。"
        f"备注：数据集只覆盖 3 类情境，其余 ContextLabel 无数据；"
        f"可测样本少于 {min_samples} 的特征被显式标记为 insufficient 而不是给均值。"
    )


# =============================================================================
# 留出评估（E21）
# =============================================================================


def evaluate_holdout(
    report: ExtractionReport,
    payload: dict,
    *,
    holdout_cats: Sequence[str],
    contexts: Sequence[str],
) -> dict:
    """在留出猫上算朴素贝叶斯分类的准确率与 macro-F1。

    ## 这就是 E21。它此前一直是「不可测」，因为先验是占位值。

    ## 这里刻意用**最朴素**的分类器

    不用判别模型、不做调参、不加特征工程 —— 因为要回答的问题是
    「这份先验有没有区分度」，而不是「能调到多准」。
    用一个复杂模型会把「先验好不好」和「模型好不好」混在一起。

    缺失特征**跳过**（不加惩罚、也不填 0）：填 0 会把
    「测不出」变成「测量值是 0」，而那是静默编造。
    """
    held = set(holdout_cats)
    samples = [s for s in report.ok if s.clip.cat_id in held]
    if not samples:
        return {"error": "没有留出样本"}

    ctx_stats = payload["contexts"]
    log_rates = {
        c: math.log(max(payload["base_rates"].get(c, 1e-9), 1e-9)) for c in contexts
    }

    confusion: Counter = Counter()
    skipped = 0

    for s in samples:
        actual = s.clip.context
        if actual not in ctx_stats:
            continue
        scores: dict[str, float] = {}
        usable = False
        for c in contexts:
            if c not in ctx_stats:
                continue
            score = log_rates.get(c, -20.0)
            for feat, (mean, std) in ctx_stats[c].items():
                if feat in s.unavailable:
                    continue  # 测不出就跳过 —— 不填 0
                z = (s.values[feat] - mean) / max(std, 1e-6)
                score += -0.5 * z * z - math.log(max(std, 1e-6))
                usable = True
            scores[c] = score
        if not usable or not scores:
            skipped += 1
            continue
        predicted = max(scores, key=lambda k: scores[k])
        confusion[(actual, predicted)] += 1

    labels = [c for c in contexts if c in ctx_stats]
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion.get((label, label), 0)
        pred = sum(v for (a, p), v in confusion.items() if p == label)
        sup = sum(v for (a, p), v in confusion.items() if a == label)
        prec = tp / pred if pred else 0.0
        rec = tp / sup if sup else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sup,
        }

    scored = [v["support"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = (
        sum(per_class[c]["f1"] for c in labels if per_class[c]["support"] > 0) / len(scored)
        if scored
        else 0.0
    )
    total = sum(confusion.values())
    correct = sum(v for (a, p), v in confusion.items() if a == p)
    chance = max(payload["base_rates"].values()) if payload["base_rates"] else 0.0

    return {
        "n_samples": len(samples),
        "n_scored": total,
        "n_skipped_no_features": skipped,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "majority_class_baseline": round(chance, 4),
        "per_class": per_class,
        "confusion": {
            f"{a}->{p}": v for (a, p), v in sorted(confusion.items())
        },
        "holdout_cats": list(holdout_cats),
    }


# =============================================================================
# 输出
# =============================================================================


def print_exploration(report: ExtractionReport) -> None:
    print("=" * 78)
    print("数据探查")
    print("=" * 78)
    print(f"  文件总数      {report.total_files}")
    print(f"  文件名可解析  {report.parsed}")
    print(f"  成功提取      {len(report.ok)}")
    print(f"  读不出        {len(report.unreadable)}")
    print(f"  太短          {len(report.too_short)}")
    print(f"  文件名不识别  {len(report.unparsable)}")
    if report.unreadable:
        print("\n  读不出的文件（数据集本身的问题，不是我们代码的）：")
        for n, e in report.unreadable[:5]:
            print(f"    {n}: {e[:70]}")

    print()
    cats = sorted({s.clip.cat_id for s in report.ok})
    print(f"  猫的数量      {len(cats)}")
    breeds = Counter(s.clip.breed for s in report.ok)
    print(f"  品种          {dict(breeds)}")

    print()
    print("  情境分布（成功提取的样本）")
    per_ctx = report.per_context()
    for ctx, samples in sorted(per_ctx.items()):
        n_cats = len({s.clip.cat_id for s in samples})
        print(f"    {ctx:22} {len(samples):4d} 条 / {n_cats:2d} 只猫")

    print()
    print("  特征可测率（**这一列决定哪些特征能进先验**）")
    header = f"    {'情境':22}" + "".join(f"{f[:8]:>9}" for f in FEATURE_ORDER)
    print(header)
    for ctx, avail in sorted(report.availability().items()):
        n = len(per_ctx[ctx])
        row = f"    {ctx:22}" + "".join(
            f"{avail.get(f, 0) / n * 100:8.0f}%" for f in FEATURE_ORDER
        )
        print(row)

    print()
    print("  质量分布")
    q = Counter(s.quality for s in report.ok)
    for k, v in q.most_common():
        print(f"    {k:12} {v:4d}  ({v / len(report.ok) * 100:.0f}%)")

    print()
    print("  ⚠️ CatMeows 只覆盖 3 类情境；greeting / door_attention / other 无数据")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="从 CatMeows 构建群体先验")
    ap.add_argument("directory", type=Path, help="含 CatMeows .wav 的目录")
    ap.add_argument("--out", type=Path, default=None, help="输出先验 JSON 路径")
    ap.add_argument("--holdout", type=int, default=4, help="留出的猫数（DESIGN 建议 4）")
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--min-samples", type=int, default=20, help="一个特征进先验所需的最少可测样本")
    ap.add_argument("--explore", action="store_true", help="只探查，不写文件")
    ap.add_argument("--eval-only", action="store_true", help="只跑留出评估")
    ap.add_argument("--template", type=Path, default=None, help="沿用此文件的 feature_specs")
    args = ap.parse_args(argv)

    if not args.directory.is_dir():
        print(f"目录不存在：{args.directory}", file=sys.stderr)
        return 2

    print(f"扫描 {args.directory} …", file=sys.stderr)
    report = extract_all(args.directory)
    print_exploration(report)

    if args.explore:
        return 0

    cats = sorted({s.clip.cat_id for s in report.ok})
    train_cats, holdout_cats = split_cats(cats, holdout=args.holdout, seed=args.seed)
    print(f"\n  留出划分（seed={args.seed}）")
    print(f"    训练 {len(train_cats)} 只：{', '.join(train_cats)}")
    print(f"    留出 {len(holdout_cats)} 只：{', '.join(holdout_cats)}")

    payload, build_notes = build_priors(
        report,
        train_cats=train_cats,
        holdout_cats=holdout_cats,
        min_samples=args.min_samples,
        seed=args.seed,
    )

    print("\n  各情境进入先验的特征")
    for ctx, notes in sorted(build_notes.items()):
        ok = [f for f, v in notes["features"].items() if v["status"] == "ok"]
        bad = {f: v for f, v in notes["features"].items() if v["status"] != "ok"}
        print(f"    {ctx:22} n={notes['n_clips']:4d} 猫={notes['n_cats']:2d} 可用 {len(ok)}/8")
        for f, v in bad.items():
            print(f"        ✗ {f:10} {v['reason']}")

    if args.template and args.template.is_file():
        tpl = json.loads(args.template.read_text(encoding="utf-8"))
        payload["feature_specs"] = tpl.get("feature_specs", {})

    print("\n  === 留出猫上的分类（E21）===")
    contexts = sorted(payload["contexts"])
    result = evaluate_holdout(
        report, payload, holdout_cats=holdout_cats, contexts=contexts
    )
    if "error" in result:
        print(f"    {result['error']}")
    else:
        print(f"    样本 {result['n_samples']}（计入 {result['n_scored']}，"
              f"无可用特征跳过 {result['n_skipped_no_features']}）")
        print(f"    准确率            {result['accuracy']}")
        print(f"    macro-F1          {result['macro_f1']}")
        print(f"    多数类基线        {result['majority_class_baseline']}  ← 低于它说明先验无区分度")
        for c, v in sorted(result["per_class"].items()):
            print(f"      {c:22} P={v['precision']:.3f} R={v['recall']:.3f} "
                  f"F1={v['f1']:.3f} n={v['support']}")
    payload["holdout_eval"] = result

    if args.out:
        payload["feature_specs"] = payload.get("feature_specs") or {}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False)
        args.out.write_text(text + "\n", encoding="utf-8")
        sha = hashlib.sha256((text + "\n").encode()).hexdigest()
        print(f"\n  已写入 {args.out}")
        print(f"  sha256 {sha[:16]}…")
        print(f"  版本   {payload['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
