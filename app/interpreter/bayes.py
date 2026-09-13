"""带证据归因的朴素贝叶斯推理。

对应 docs/DESIGN.md §3.6「推理」与「LLM 与代码的分工」。

**这个模块不调用任何大模型。** 概率由代码算，这保证：
1. 可复现 —— 同一特征 + 同一先验版本 → 同一后验
2. 可归因 —— 每个特征对判断的贡献是一个可打印的数字
3. 可审计 —— 证据能由 value / reference 复算

同时它有成本红利：行为解释的主要成本只有一步「措辞化」，几乎免费。

## 数学形式

对每个候选情境 ``k``：

```
LLR_i(k) = log [ p(f_i | k, c) / p(f_i | ¬k, c) ]
logit(k) = log P(k | scene) + Σ_i LLR_i(k)
P(k)     = softmax_k( logit )
```

- ``p(f_i | k, c)`` 由**收缩后**的高斯参数给出（见 ``priors.IndividualModel``）
- ``p(f_i | ¬k, c)`` 是其余情境按基础率加权的混合密度
- ``log P(k | scene)`` 在用户给出场景时对该情境加成

### 诚实说明

one-vs-rest 的 LLR 再 softmax 归一化，是**为了可归因性而做的近似**：
绝对概率语义不精确（故本系统只用它做**排序**与**分档**，不对外声称绝对概率）。

这是刻意的取舍：`DESIGN.md` §3.6 要求「每个特征的贡献是一个可打印的数字」，
而一个连贯的多项式朴素贝叶斯无法直接给出这种分解。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from app.interpreter.priors import (
    FEATURE_ORDER,
    IndividualModel,
    PriorTable,
)
from app.schemas import (
    AcousticFeatures,
    BehaviorInterpretation,
    ContextLabel,
    EvidenceItem,
    EvidenceKind,
    EvidenceMode,
    FeatureQuality,
    IntentCandidate,
)

#: 单特征对数几率贡献的绝对值上限。
#: 防止某个极端特征单独主导判断（「一个特征说了算」是不可审计的）。
LLR_CAP = 3.0

#: 用户给出场景时，匹配情境的对数几率加成。
SCENE_LOG_ODDS_BOOST = 1.10

#: 相似样本证据生效的最小相似度。低于此值不构成证据。
SIMILARITY_FLOOR = 0.80

#: 相似样本证据的最大对数几率贡献。
SIMILARITY_MAX_LOG_ODDS = 0.95

#: 质量对**证据力度**的折扣系数。
#:
#: ⚠️ 早期实现用的是「均匀展宽似然标准差」，**那是错的**。原因：
#: 对数密度 `-0.5·z² − log σ` 中，均匀乘 c 后 `z²/c²` 项缩小，
#: 但 `log σ` 项**不受影响** —— 结果是似然退化成
#: 「偏好标准差较窄的情境」，**与观测值无关**。实测确实出现
#: 「质量越差、某个情境概率反而越高」的反常。
#:
#: 正确做法：**质量影响的是证据的力度，不是似然本身**。
#: 对 LLR 整体打折，则 logit 按比例向先验收缩 —— 单调、可解释。
EVIDENCE_DISCOUNT: dict[FeatureQuality, float] = {
    FeatureQuality.GOOD: 1.0,
    FeatureQuality.FAIR: 0.75,
    FeatureQuality.POOR: 0.50,
}

#: logit 温度。在 softmax 前除以它（>1 使分布变缓）。
#:
#: ⚠️ **未经验证的校准参数**（对应 docs/DESIGN.md §7.3 U1）。
#: 实测发现：占位先验下 logit 跨度可达 ±6，softmax 会退化成近似 one-hot，
#: 使**每一次判断都落进「高置信」档**——这与项目「不声称超出证据的确定性」的
#: 原则直接冲突。
#:
#: 正确修法是做温度校准并用 ECE 拟合（U1）；在完成校准前，
#: 先用一个保守的默认值避免过度自信。**它是待替换的权宜值，不是调好的参数。**
LOGIT_TEMPERATURE = 2.0

#: 进入证据列表的最大条数。
MAX_EVIDENCE_ITEMS = 4

#: 人类可读的特征名与单位。
_FEATURE_LABEL: dict[str, tuple[str, str]] = {
    "duration": ("叫声时长", "s"),
    "f0_mean": ("基频均值", "Hz"),
    "f0_range": ("基频跨度", "Hz"),
    "f0_slope": ("基频轮廓走向", ""),
    "call_rate": ("叫声速率", "次/10s"),
    "ici_mean": ("叫声间隔", "s"),
    "rms_mean": ("能量强度", ""),
    "roughness": ("粗糙度", ""),
}

_CONTEXT_DISPLAY: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "很可能是在等吃的",
    ContextLabel.DOOR_ATTENTION: "很可能是在关注门外动静",
    ContextLabel.AFFECTION_BRUSHING: "很可能是在求摸 / 享受亲密",
    ContextLabel.ISOLATION_DISTRESS: "可能是不安或想找人",
    ContextLabel.GREETING: "很可能是在打招呼",
    ContextLabel.OTHER: "无法归入常见情境",
}

_CONTEXT_OBSERVATION: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "观察它是否伴随蹭腿、绕食盆走动、望向存放食物的地方",
    ContextLabel.DOOR_ATTENTION: "观察它是否伴随抓门、来回踱步、贴近门口",
    ContextLabel.AFFECTION_BRUSHING: "观察它是否主动靠近、蹭人、呼噜、翻肚皮",
    ContextLabel.ISOLATION_DISTRESS: "观察它是否躲藏、拒绝互动、或持续来回走动",
    ContextLabel.GREETING: "观察它是否竖尾靠近、蹭人后离开",
    ContextLabel.OTHER: "记录下当时的场景与它的动作，有助于下次判断",
}

#: 场景描述关键词 → 情境。规则匹配，不调模型。
_SCENE_KEYWORDS: dict[ContextLabel, tuple[str, ...]] = {
    ContextLabel.DOOR_ATTENTION: ("门", "门口", "门外", "door"),
    ContextLabel.FOOD_WAITING: ("食盆", "猫粮", "罐头", "吃", "饭", "饿", "food"),
    ContextLabel.AFFECTION_BRUSHING: ("摸", "抱", "梳", "撸", "亲密", "brush"),
    ContextLabel.ISOLATION_DISTRESS: ("陌生", "独处", "隔离", "新环境", "isolat"),
    ContextLabel.GREETING: ("回家", "进门", "打招呼", "迎接", "greet"),
}


@dataclass(frozen=True)
class SimilarSample:
    """一条检索到的历史叫声样本。"""

    sample_id: str
    similarity: float
    context: ContextLabel | None
    """用户确认过的情境。``None`` 表示未确认 —— **不构成证据，不参与计算**。"""


def match_scene(scene: str | None) -> ContextLabel | None:
    """从场景描述中匹配情境。规则匹配，可复现。"""
    if not scene:
        return None
    lowered = scene.lower()
    for context, keywords in _SCENE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return context
    return None


def _feature_vector(features: AcousticFeatures) -> dict[str, float]:
    """把契约对象摊平成特征字典，**剔除不可用特征**。

    不可用包括两类：
    1. 提取阶段显式标记的（``features.unavailable``）——如窗口过短时的 ``call_rate``
    2. 推理阶段发现的异常值（非有限、或 ``f0_mean`` 为 0 表示无有效基频帧）

    第二类作为双保险保留：它能拦住**手工构造**的、未正确设置 ``unavailable`` 的对象。
    宁可不参与计算，也不拿假值去算似然。
    """
    declared = set(features.unavailable)
    values: dict[str, float] = {}
    for name in FEATURE_ORDER:
        if name in declared:
            continue
        value = float(getattr(features, name))
        if not np.isfinite(value):
            continue
        # f0_mean 为 0 表示「无有效 F0 帧」，不是「基频为 0」——必须剔除而非当真值
        if name == "f0_mean" and value <= 0.0:
            continue
        values[name] = value
    return values


def _evidence_discount(quality: FeatureQuality) -> float:
    return EVIDENCE_DISCOUNT.get(quality, 1.0)


def _log_densities(
    values: dict[str, float],
    contexts: tuple[ContextLabel, ...],
    prior: PriorTable,
    individual: IndividualModel | None,
) -> dict[ContextLabel, dict[str, float]]:
    """每个 (情境, 特征) 的对数密度，使用收缩后参数。

    注意：此处**不**做质量调整 —— 质量作用于证据折扣（见 ``EVIDENCE_DISCOUNT``），
    而不是展宽似然。
    """
    out: dict[ContextLabel, dict[str, float]] = {}
    for ctx in contexts:
        per_feature: dict[str, float] = {}
        for feat, value in values.items():
            pop = prior.stat(ctx, feat)
            if pop is None:
                # **该特征在这个情境下没有群体数据 → 跳过。**
                #
                # 这与「这个样本测不出该特征」是同一类事实：
                # 两者都不能用来算似然。真实数据下这很常见 ——
                # CatMeows 的 `call_rate` / `ici_mean` 可测率只有 0–4%。
                #
                # 不跳过的话只剩两条路：报错（一份部分覆盖的先验不可用），
                # 或者填一个默认值（静默编造，后验看上去完全正常）。
                continue
            stat = (
                individual.shrunk_stat(ctx, feat, pop)
                if individual is not None
                else pop
            )
            per_feature[feat] = stat.log_density(value)
        out[ctx] = per_feature
    return out


def _log_priors(
    contexts: tuple[ContextLabel, ...],
    prior: PriorTable,
    scene_context: ContextLabel | None,
) -> dict[ContextLabel, float]:
    """基础率对数 + 场景加成。"""
    out = {ctx: prior.log_base_rate(ctx) for ctx in contexts}
    if scene_context is not None and scene_context in out:
        out[scene_context] += SCENE_LOG_ODDS_BOOST
    return out


def _log_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def interpret(
    *,
    features: AcousticFeatures,
    prior: PriorTable,
    individual: IndividualModel | None = None,
    scene: str | None = None,
    similar_samples: list[SimilarSample] | None = None,
    allow_unvalidated_prior: bool = False,
) -> BehaviorInterpretation:
    """推断裂叫声最可能产生于哪种情境。

    Args:
        features: 声学特征（由 ``app.audio.features.extract_features`` 产出）。
        prior: 群体先验表。**必须已验证**（见下）。
        individual: 该猫的个体模型。为 ``None`` 或 ``sample_count=0`` 时退回纯群体先验。
        scene: 用户补充的场景描述。
        similar_samples: 检索到的历史叫声。**只有已确认情境的样本参与计算。**
        allow_unvalidated_prior: 跳过「先验已验证」的检查。
            **只给消融实验与数学层测试用**，生产路径不得传。

    Raises:
        ValueError: 先验未通过验证（占位 / 未做留出评估 / 实测不如基线）。

    ## 为什么把门禁放在这里，而不是只放在路由层

    路由层（`choose_mode`）已经会因此降级。但**本函数是数学层**，
    而“先验能不能用来产出后验概率”是一个**不变量**，不是路由的偏好：

    - 任何绕过路由直接调本函数的路径（新调用方、脚本、将来的重构）
      都会拿到一堆看起来很正常、实际上不具区分度的后验概率；
    - CatMeows 实测结果正是这个情形：留出 macro-F1 0.364
      低于多数类基线 0.506 —— 那些概率**不是证据**。

    把检查放进本函数，不变量就是**结构性的**：不依赖调用方是否记得先问路由。
    这与「存储层强制要求 tenant_id」是同一个思路（见 `app/store/base.py`）。
    """
    if not prior.is_validated and not allow_unvalidated_prior:
        margin = prior.discrimination_margin
        if prior.is_placeholder:
            detail = "含占位数据（数字是编的）"
        elif margin is None:
            detail = "未做过留出评估，无法确认有区分度"
        else:
            detail = (
                f"实测不具区分度：留出 macro-F1 {prior.holdout_macro_f1:.3f} "
                f"≤ 多数类基线 {prior.holdout_majority_baseline:.3f}"
            )
        raise ValueError(
            f"先验 {prior.version} {detail}，不得用来产出后验概率。\n"
            f"请用 `interpret_meow`（它会选择合适的降级模式），"
            f"或在消融/数学层测试里显式传 `allow_unvalidated_prior=True`。"
        )

    contexts = prior.contexts_ordered
    if len(contexts) < 2:
        raise ValueError("先验表至少需要 2 个情境才能做比较")

    values = _feature_vector(features)
    if not values:
        raise ValueError("没有任何可用的声学特征")

    discount = _evidence_discount(features.quality)
    densities = _log_densities(values, contexts, prior, individual)
    log_priors = _log_priors(contexts, prior, match_scene(scene))

    # ── one-vs-rest 对数似然比 ──
    # llr[ctx][feat] = log p(f|ctx) − log Σ_{j≠ctx} w_j·p(f|j)
    contributions: dict[ContextLabel, dict[str, float]] = {}
    for ctx in contexts:
        others = [c for c in contexts if c is not ctx]
        others_weight = np.array([np.exp(log_priors[c]) for c in others], dtype=float)
        if others_weight.sum() <= 0:
            others_weight = np.ones(len(others), dtype=float)
        others_weight = others_weight / others_weight.sum()

        per_feature: dict[str, float] = {}
        for feat in values:
            log_p_others = float(
                np.log(
                    sum(
                        w * np.exp(densities[c][feat])
                        for w, c in zip(others_weight, others)
                    )
                    + 1e-300
                )
            )
            raw = densities[ctx][feat] - log_p_others
            per_feature[feat] = float(np.clip(raw, -LLR_CAP, LLR_CAP)) * discount
        contributions[ctx] = per_feature

    # ── 相似样本证据（先验调整） ──
    sample_adjustment: dict[ContextLabel, float] = dict.fromkeys(contexts, 0.0)
    used_samples: list[SimilarSample] = []
    for sample in similar_samples or []:
        if sample.context is None or sample.similarity < SIMILARITY_FLOOR:
            continue
        if sample.context not in sample_adjustment:
            continue
        scaled = (sample.similarity - SIMILARITY_FLOOR) / (1.0 - SIMILARITY_FLOOR)
        sample_adjustment[sample.context] += SIMILARITY_MAX_LOG_ODDS * float(
            np.clip(scaled, 0.0, 1.0)
        )
        used_samples.append(sample)

    # ── 汇总 → softmax ──
    logits = np.array(
        [
            log_priors[ctx] + sum(contributions[ctx].values()) + sample_adjustment[ctx]
            for ctx in contexts
        ],
        dtype=float,
    )
    # 温度只在归一化时施加；log_odds 保持原始值，使「贡献之和 == logit」依然成立（可归因）
    posteriors = _log_softmax(logits / LOGIT_TEMPERATURE)

    candidates = [
        IntentCandidate(
            context=ctx,
            posterior=round(float(p), 6),
            display=_CONTEXT_DISPLAY[ctx],
            log_odds=round(float(logit), 4),
        )
        for ctx, p, logit in zip(contexts, posteriors, logits)
    ]
    candidates.sort(key=lambda c: c.posterior or 0.0, reverse=True)

    evidence = _build_evidence(
        candidates=candidates,
        contributions=contributions,
        values=values,
        prior=prior,
        individual=individual,
        scene=scene,
        used_samples=used_samples,
    )

    return BehaviorInterpretation(
        evidence_mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
        acoustic_features=features,
        candidates=candidates,
        evidence=evidence,
        individualization=round(individual.lambda_c, 4) if individual else 0.0,
        sample_count=individual.sample_count if individual else 0,
        suggested_observation=_suggested_observation(candidates, values),
        limitations=_limitations(prior, individual, features, values),
        prior_version=prior.version,
    )


# ─────────────────────────────────────────────────────────────
# 证据构建
# ─────────────────────────────────────────────────────────────


def _build_evidence(
    *,
    candidates: list[IntentCandidate],
    contributions: dict[ContextLabel, dict[str, float]],
    values: dict[str, float],
    prior: PriorTable,
    individual: IndividualModel | None,
    scene: str | None,
    used_samples: list[SimilarSample],
) -> list[EvidenceItem]:
    """构建证据列表。

    **预算必须给 PRIOR / RETRIEVED 预留槽位**。
    早期实现对 MEASURED 先填满到上限、再插入 PRIOR 与追加 RETRIEVED，
    最后 `[:MAX]` 截断——结果是**场景证据与历史样本证据被静默丢弃**，
    而它们恰恰是用户最容易理解的证据类型。
    """
    scene_ctx = match_scene(scene)
    # 同时取回已收窄的 context（``Sample.context`` 是 ``ContextLabel | None``，
    # 只做列表过滤不会让类型收窄传播到后续使用点）
    retrieved: list[tuple[SimilarSample, ContextLabel]] = [
        (s, s.context) for s in used_samples if s.context is not None
    ][:2]
    reserved = (1 if scene_ctx is not None else 0) + len(retrieved)
    measured_budget = max(1, MAX_EVIDENCE_ITEMS - reserved)

    items: list[EvidenceItem] = []
    top_contexts = [c.context for c in candidates[:2]]

    ranked: list[tuple[float, ContextLabel, str]] = []
    for ctx in top_contexts:
        for feat, contrib in contributions[ctx].items():
            ranked.append((abs(contrib), ctx, feat))
    ranked.sort(key=lambda x: x[0], reverse=True)

    seen_features: set[str] = set()
    for _mag, ctx, feat in ranked:
        if len(items) >= measured_budget:
            break
        if feat in seen_features:
            continue
        seen_features.add(feat)
        items.append(
            _measured_evidence(
                feat, ctx, contributions[ctx][feat], values[feat], prior, individual
            )
        )

    if scene_ctx is not None:
        items.insert(
            0,
            EvidenceItem(
                kind=EvidenceKind.PRIOR,
                statement=f"你提到当时的情况是「{scene}」，这与「{_CONTEXT_DISPLAY[scene_ctx]}」一致",
                source=f"scene_prior:{scene_ctx.value}",
                value=None,
                reference=None,
                log_odds_contribution=SCENE_LOG_ODDS_BOOST,
            ),
        )

    for sample, ctx in retrieved:
        scaled = (sample.similarity - SIMILARITY_FLOOR) / (1.0 - SIMILARITY_FLOOR)
        contrib = SIMILARITY_MAX_LOG_ODDS * float(np.clip(scaled, 0.0, 1.0))
        items.append(
            EvidenceItem(
                kind=EvidenceKind.RETRIEVED,
                statement=(
                    f"与它过去一次被确认属于「{_CONTEXT_DISPLAY[ctx]}」"
                    f"的叫声相似（相似度 {sample.similarity:.2f}）"
                ),
                source=f"meow_sample:{sample.sample_id}",
                value=round(sample.similarity, 4),
                reference=None,
                log_odds_contribution=round(contrib, 4),
            )
        )

    return items[:MAX_EVIDENCE_ITEMS]


def _measured_evidence(
    feat: str,
    ctx: ContextLabel,
    contribution: float,
    value: float,
    prior: PriorTable,
    individual: IndividualModel | None,
) -> EvidenceItem:
    label, unit = _FEATURE_LABEL.get(feat, (feat, ""))
    pop = prior.stat(ctx, feat)
    if pop is None:
        # **内部不变量**：本函数只被 `contributions` 里出现过的特征调用，
        # 而那些特征是 `_log_densities` 跳过了缺数据项之后剩下的。
        # 所以走到这里说明上游的过滤被改坏了 —— 报错而不是静默降级。
        raise AssertionError(
            f"特征 {feat!r} 在情境 {ctx.value!r} 下无群体数据，"
            f"但它出现在了 contributions 里 —— 上游过滤有 bug"
        )
    stat = individual.shrunk_stat(ctx, feat, pop) if individual else pop

    direction = "高于" if value > stat.mean else "低于"
    unit_suffix = f" {unit}" if unit else ""
    return EvidenceItem(
        kind=EvidenceKind.MEASURED,
        statement=(
            f"{label} {_fmt(value)}{unit_suffix}，"
            f"{direction}该情境的参考值 {_fmt(stat.mean)}{unit_suffix}"
        ),
        source=f"acoustic:{feat}",
        value=round(value, 4),
        reference=round(stat.mean, 4),
        log_odds_contribution=round(contribution, 4),
    )


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


# ─────────────────────────────────────────────────────────────
# 措辞（模板实现）
# ─────────────────────────────────────────────────────────────


def _suggested_observation(
    candidates: list[IntentCandidate], values: dict[str, float]
) -> str:
    """建议观察项。

    当前为**模板实现**（确定性、可离线）。LLM 层可在此之上做措辞优化，
    但**不得改变其指向的情境**。
    """
    top = candidates[0] if candidates else None
    base = _CONTEXT_OBSERVATION.get(top.context, "") if top else ""
    if "f0_mean" not in values:
        base += "；本次未能提取到有效基频，录音质量可能不佳，建议在安静环境重录"
    return base or "记录下当时的场景与它的动作，有助于下次判断"


def _limitations(
    prior: PriorTable,
    individual: IndividualModel | None,
    features: AcousticFeatures,
    values: dict[str, float],
) -> str:
    parts = [
        "仅凭叫声无法确定真实需求；若伴随持续焦躁、异常叫声或食欲改变，建议就医观察。"
    ]

    if prior.is_placeholder:
        parts.append(
            f"⚠️ 群体先验为占位数据（{prior.version}），尚未用公开数据集实测统计替换，"
            "排序仅供参考。"
        )
    if individual is None or individual.is_cold_start:
        parts.append("这是它第一次被记录，尚无个体样本，判断完全依赖群体先验。")
    elif individual.lambda_c < 0.5:
        parts.append(
            f"个体样本仅 {individual.sample_count} 个（个体化程度 {individual.lambda_c:.0%}），"
            "个体特征尚未充分体现。"
        )
    if features.quality is not FeatureQuality.GOOD:
        parts.append(f"本次录音质量评级为 {features.quality.value}，判别力已相应下调。")
    if "f0_mean" not in values:
        parts.append("未提取到有效基频（可能录音过短或噪声过大）。")
    if features.unavailable:
        parts.append(f"本次未参与判断的特征：{'、'.join(features.unavailable)}。")

    parts.append(
        "置信度尚未完成校准（详见 docs/DESIGN.md U1），请以候选排序与证据为主，"
        "不要依赖概率的绝对值。"
    )

    return " ".join(parts)
