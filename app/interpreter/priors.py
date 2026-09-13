"""先验与似然估计。

对应 docs/DESIGN.md §3.6「推理」与「冷启动：分层贝叶斯收缩」。

关键设计：
1. **群体先验来自版本化文件**，加载时算 ``sha256`` —— 可复现性要求（§3.6 §11）。
2. **个体似然走收缩**：``P(f|k,c) = λ_c·P_ind + (1−λ_c)·P_pop``，
   ``λ_c = n_c/(n_c+κ)``。新猫 ``λ_c=0`` 时完全退回群体先验。
3. **标准差也收缩**。只收缩均值是常见错误——2 个样本估出的 std 极不可靠。
4. 先验文件带 ``provenance``；占位先验会被透传到输出，**不静默当成真实统计**。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.schemas import ContextLabel

#: 收缩常数 κ。**未经验证的经验值**（docs/DESIGN.md §6.4）。
#: 语义：当个体样本数达到 κ 时，个体与群体权重各半。
DEFAULT_KAPPA = 5.0

#: 标准差下界。防止某个特征在某个情境上估出接近 0 的 std 导致密度爆炸。
STD_FLOOR_RATIO = 0.05
"""相对群体 std 的下界比例。"""

#: 特征顺序固定 —— 保证同一输入得到同一归因顺序。
FEATURE_ORDER: tuple[str, ...] = (
    "duration",
    "f0_mean",
    "f0_range",
    "f0_slope",
    "call_rate",
    "ici_mean",
    "rms_mean",
    "roughness",
)

#: 参与似然计算的特征。``f0_mean`` 在无有效 F0 帧时为 0，需排除，单独处理。
LIKELIHOOD_FEATURES: tuple[str, ...] = FEATURE_ORDER


class PriorTableError(ValueError):
    """先验文件格式非法。"""


@dataclass(frozen=True)
class GaussianStat:
    """一个特征在一个情境下的高斯参数。"""

    mean: float
    std: float

    def log_density(self, value: float) -> float:
        """标准正态对数密度。"""
        std = max(self.std, 1e-6)
        z = (value - self.mean) / std
        return float(-0.5 * z * z - np.log(std))


@dataclass(frozen=True)
class PriorTable:
    """群体先验表。"""

    version: str
    provenance: str
    is_placeholder: bool
    sha256: str
    note: str
    base_rates: dict[ContextLabel, float]
    contexts: dict[ContextLabel, dict[str, GaussianStat]]

    @property
    def contexts_ordered(self) -> tuple[ContextLabel, ...]:
        """固定顺序，保证数值可复现。"""
        return tuple(
            c for c in ContextLabel if c in self.contexts
        )

    def stat(self, context: ContextLabel, feature: str) -> GaussianStat:
        return self.contexts[context][feature]

    def log_base_rate(self, context: ContextLabel) -> float:
        """自然对数基础率。"""
        return float(np.log(max(self.base_rates.get(context, 1e-6), 1e-9)))

    @classmethod
    def load(cls, path: str | Path) -> PriorTable:
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw.decode("utf-8"))

        if "contexts" not in payload or "base_rates" not in payload:
            raise PriorTableError("先验文件缺少 contexts 或 base_rates")

        contexts: dict[ContextLabel, dict[str, GaussianStat]] = {}
        for ctx_name, stats in payload["contexts"].items():
            try:
                ctx = ContextLabel(ctx_name)
            except ValueError as exc:
                raise PriorTableError(f"未知情境标签：{ctx_name}") from exc
            parsed: dict[str, GaussianStat] = {}
            for feat, pair in stats.items():
                if len(pair) != 2:
                    raise PriorTableError(f"{ctx_name}.{feat} 应为 [mean, std]")
                mean, std = float(pair[0]), float(pair[1])
                if std <= 0:
                    raise PriorTableError(f"{ctx_name}.{feat} 的 std 必须为正")
                parsed[feat] = GaussianStat(mean=mean, std=std)
            missing = set(FEATURE_ORDER) - set(parsed)
            if missing:
                raise PriorTableError(f"{ctx_name} 缺少特征：{sorted(missing)}")
            contexts[ctx] = parsed

        base_rates = {ContextLabel(k): float(v) for k, v in payload["base_rates"].items()}
        total = sum(base_rates.values())
        if not 0.99 <= total <= 1.01:
            raise PriorTableError(f"base_rates 之和为 {total:.3f}，应为 1.0")

        return cls(
            version=str(payload.get("version", "unknown")),
            provenance=str(payload.get("provenance", "unknown")),
            is_placeholder=bool(payload.get("is_placeholder", False)),
            sha256=sha,
            note=str(payload.get("note", "")),
            base_rates=base_rates,
            contexts=contexts,
        )


# ─────────────────────────────────────────────────────────────
# 个体模型与收缩
# ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelledSample:
    """一条已被用户确认情境的叫声样本。"""

    context: ContextLabel
    features: dict[str, float]


@dataclass(frozen=True)
class IndividualModel:
    """某只猫的个体似然模型，含分层收缩。"""

    pet_id: str
    kappa: float = DEFAULT_KAPPA
    stats: dict[ContextLabel, dict[str, GaussianStat]] = None  # type: ignore[assignment]
    sample_count: int = 0
    """该猫**已确认**的样本数 n_c。"""

    def __post_init__(self) -> None:
        if self.stats is None:
            object.__setattr__(self, "stats", {})

    @property
    def lambda_c(self) -> float:
        """个体化程度 λ_c = n_c/(n_c+κ)。可对用户展示为「系统对它的了解程度」。"""
        return self.sample_count / (self.sample_count + self.kappa) if self.sample_count else 0.0

    @property
    def is_cold_start(self) -> bool:
        return self.sample_count == 0

    @classmethod
    def from_samples(
        cls,
        pet_id: str,
        samples: list[LabelledSample],
        kappa: float = DEFAULT_KAPPA,
    ) -> IndividualModel:
        """从已确认样本估计个体高斯参数。"""
        grouped: dict[ContextLabel, list[LabelledSample]] = {}
        for s in samples:
            grouped.setdefault(s.context, []).append(s)

        stats: dict[ContextLabel, dict[str, GaussianStat]] = {}
        for ctx, items in grouped.items():
            per_feature: dict[str, GaussianStat] = {}
            for feat in FEATURE_ORDER:
                values = [
                    it.features[feat]
                    for it in items
                    if feat in it.features and np.isfinite(it.features[feat])
                ]
                if len(values) >= 2:
                    per_feature[feat] = GaussianStat(
                        mean=float(np.mean(values)), std=float(np.std(values, ddof=1))
                    )
            if per_feature:
                stats[ctx] = per_feature

        return cls(pet_id=pet_id, kappa=kappa, stats=stats, sample_count=len(samples))

    def shrunk_stat(
        self, context: ContextLabel, feature: str, population: GaussianStat
    ) -> GaussianStat:
        """收缩后的似然参数。

        ``P(f|k,c) = λ_c·P_ind + (1−λ_c)·P_pop``

        **均值和标准差都收缩**。只收缩均值是常见错误：
        2 个样本估出的 std 极不可靠，会让某个特征产生虚假的强判别力。
        """
        lam = self.lambda_c
        if lam <= 0.0:
            return population

        ind = self.stats.get(context, {}).get(feature)
        if ind is None:
            return population

        mean = lam * ind.mean + (1 - lam) * population.mean
        # std 收缩后再施加下界，防止接近 0 导致密度爆炸
        std = lam * ind.std + (1 - lam) * population.std
        std = max(std, population.std * STD_FLOOR_RATIO, 1e-6)
        return GaussianStat(mean=mean, std=std)
