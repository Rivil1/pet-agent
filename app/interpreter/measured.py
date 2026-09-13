"""冷启动：只有测量值，没有可用的推断模型（`EvidenceMode.MEASURED_ONLY`）。

## 它为什么存在

新用户刚记录几条时，案例推理的样本还不够
（`MIN_CASES_PER_CONTEXT` 条/情境）。此时有两条路：

| 做法 | 后果 |
| --- | --- |
| 用群体先验猜一个后验 | **先验是占位的** —— 那个概率是编的 |
| **只说测到了什么** | 诚实的少，但可核查 |

选后者。这与 B3（零填充是静默编造）是同一条原则：
**宁可显式说「我还不知道」，也不给一个看起来正常的数字。**

## 输出什么

**系统测量**（可复现）+ 用户自己提供的场景，而已。

| 输出 | 内容 |
| --- | --- |
| `evidence` | 测量值；有**个体基线**时才给对比，否则只报数值 |
| `candidates` | **仅场景驱动**：用户说在门口，才列出「门口相关」这一条 |
| `suggested_observation` | 邀请记录，而不是给结论 |

**没有任何关于猫的判断。**

## 与 B 方案的一致性

用户选了「测量值 + 场景假设」作为冷启动形态（见 `docs/DESIGN.md` §3.7）。
本模块就是那个形态的实现。

## 一条硬约束

`EvidenceItem.log_odds_contribution` 必须为 `None`。
填 `0.0` 会谎称「已参与计算但影响为零」——而事实上本模式**没有任何概率计算**。
契约（`BehaviorInterpretation._contribution_matches_mode`）会校验这一点。
"""

from __future__ import annotations

from app.interpreter.priors import FEATURE_ORDER, IndividualModel
from app.schemas import (
    AcousticFeatures,
    BehaviorInterpretation,
    ContextLabel,
    EvidenceItem,
    EvidenceKind,
    EvidenceMode,
    FeatureQuality,
    IntentCandidate,
    MeowRecord,
)

#: 场景关键词 → 情境。规则匹配，不调模型，可复现。
SCENE_KEYWORDS: dict[ContextLabel, tuple[str, ...]] = {
    ContextLabel.DOOR_ATTENTION: ("门", "门口", "门外", "door"),
    ContextLabel.FOOD_WAITING: ("食盆", "猫粮", "罐头", "吃", "饭", "饿", "food"),
    ContextLabel.AFFECTION_BRUSHING: ("摸", "抱", "梳", "撸", "亲密", "brush"),
    ContextLabel.ISOLATION_DISTRESS: ("陌生", "独处", "隔离", "新环境", "isolat"),
    ContextLabel.GREETING: ("回家", "进门", "打招呼", "迎接", "greet"),
}

#: 场景只是**你的描述**，不是系统判断 —— 所以它列出的是「常见解释」，
#: 措辞里不出现「可能」「很可能」这类概率暗示（本模式没有概率）。
_CONTEXT_HYPOTHESIS: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "饭点相关（饿了、习惯性索食）",
    ContextLabel.DOOR_ATTENTION: "门口相关（想出去、想进来、要人陪）",
    ContextLabel.AFFECTION_BRUSHING: "亲密相关（求摸、想被注意）",
    ContextLabel.ISOLATION_DISTRESS: "不安相关（找人、陌生环境）",
    ContextLabel.GREETING: "打招呼相关（迎接、确认你在）",
    ContextLabel.OTHER: "不属于常见情境",
}

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


def match_scene(scene: str | None) -> ContextLabel | None:
    """从场景描述里匹配情境。规则匹配，可复现。"""
    if not scene:
        return None
    for ctx, keywords in SCENE_KEYWORDS.items():
        if any(kw in scene for kw in keywords):
            return ctx
    return None


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def interpret_measured_only(
    *,
    features: AcousticFeatures,
    scene: str | None = None,
    individual: IndividualModel | None = None,
    records: list[MeowRecord] | None = None,
) -> BehaviorInterpretation:
    """只输出测量值与场景假设。

    Args:
        features: 本次叫声的声学特征。
        scene: 用户提供的场景描述（可选）。
        individual: 个体模型。**有它才给对比** —— 参考系必须是这只猫自己的，
            不能用占位先验（那会让「高于参考值」这句话建立在编造的数值上）。
        records: 已有的叫声记录，仅用于告诉用户「还差多少条」。

    Returns:
        `BehaviorInterpretation`，`evidence_mode=MEASURED_ONLY`，
        **无 posterior / 无 log_odds / 无 matched_count / 无 similar_cases**。
    """
    values = _usable_values(features)

    evidence = _build_measured_evidence(features, values, individual)
    scene_ctx = match_scene(scene)

    candidates: list[IntentCandidate] = []
    if scene_ctx is not None:
        candidates = [
            IntentCandidate(
                context=scene_ctx,
                display=f"你提到的情况是「{scene}」，与之相关的是：{_CONTEXT_HYPOTHESIS[scene_ctx]}",
                posterior=None,
                log_odds=None,
                matched_count=0,
            )
        ]

    return BehaviorInterpretation(
        evidence_mode=EvidenceMode.MEASURED_ONLY,
        acoustic_features=features,
        candidates=candidates,
        evidence=evidence,
        # 以下四项在本模式下必须为空/零 —— 契约会校验
        similar_cases=[],
        case_total=0,
        individualization=round(individual.lambda_c, 4) if individual else 0.0,
        sample_count=individual.sample_count if individual else 0,
        suggested_observation=_suggested_observation(
            scene_ctx, records or [], individual
        ),
        limitations=_limitations(features, values, records or [], individual),
        prior_version=None,
    )


def _usable_values(features: AcousticFeatures) -> dict[str, float]:
    """可用特征值。与另外两个模式同一口径。"""
    import math

    declared = set(features.unavailable)
    out: dict[str, float] = {}
    for name in FEATURE_ORDER:
        if name in declared:
            continue
        value = float(getattr(features, name))
        if not math.isfinite(value):
            continue
        if name == "f0_mean" and value <= 0.0:
            continue
        out[name] = value
    return out


def _build_measured_evidence(
    features: AcousticFeatures,
    values: dict[str, float],
    individual: IndividualModel | None,
) -> list[EvidenceItem]:
    """构建测量证据。

    **参考系的规则**（这是本函数唯一容易写错的地方）：

    | 有该猫的个体基线？ | 输出 |
    | --- | --- |
    | 是 | 「比它自己平时高/低 X」← 可核查 |
    | 否 | **只报数值**，不给比较 |

    **绝不退回群体先验。** 占位先验的均值是编的，
    用它当参考系会让「高于参考值」这句话建立在假数字上 ——
    而那是最容易被当成事实的一类表述。
    """
    items: list[EvidenceItem] = []
    scale = _EVIDENCE_SCALE

    for feat in FEATURE_ORDER:
        if feat not in values:
            continue
        label, unit = _FEATURE_LABEL.get(feat, (feat, ""))
        suffix = f" {unit}" if unit else ""
        value = values[feat]

        stat = _individual_stat(individual, feat)
        if stat is None:
            statement = f"{label} {_fmt(value)}{suffix}（本次测量值）"
            reference = None
        else:
            direction = "高于" if value > stat.mean else "低于"
            statement = (
                f"{label} {_fmt(value)}{suffix}，"
                f"{direction}它自己平时的 {_fmt(stat.mean)}{suffix}"
            )
            reference = round(stat.mean, 4)

        items.append(
            EvidenceItem(
                kind=EvidenceKind.MEASURED,
                statement=statement,
                source=f"acoustic:{feat}",
                value=round(value, 4),
                reference=reference,
                # **必须为 None** —— 本模式没有任何概率计算
                log_odds_contribution=None,
            )
        )

    if len(items) > scale:
        items = items[:scale]
    return items


#: 证据列表最多展示几项（测量值太多会淹没重点）。
_EVIDENCE_SCALE = 6


def _individual_stat(individual: IndividualModel | None, feat: str):
    """取该猫在**任一情境**下该特征的基线。

    这里刻意不按情境筛选：冷启动阶段样本本来就少，
    按情境筛几乎必然取不到值。取「该特征的总体个体基线」是更可用的近似。

    ⚠️ 这与案例推理不同 —— 那里必须比同情境，因为那里在做情境判断。
    """
    if individual is None or individual.is_cold_start:
        return None
    candidates = [
        per_feature[feat]
        for per_feature in individual.stats.values()
        if feat in per_feature
    ]
    if not candidates:
        return None
    # 多情境取均值：冷启动阶段不做情境区分
    from app.interpreter.priors import GaussianStat

    return GaussianStat(
        mean=sum(c.mean for c in candidates) / len(candidates),
        std=sum(c.std for c in candidates) / len(candidates),
    )


def _suggested_observation(
    scene_ctx: ContextLabel | None,
    records: list[MeowRecord],
    individual: IndividualModel | None,
) -> str:
    """建议观察项。

    冷启动时它的主语是「**你**」，不是「猫」——
    系统还不知道这只猫，所以它请求信息，而不是给出判断。
    """
    from app.interpreter.case_based import MIN_CASES_PER_CONTEXT

    n = len(records)
    remaining = max(0, MIN_CASES_PER_CONTEXT - n)

    parts: list[str] = []
    if scene_ctx is not None:
        parts.append(_OBSERVATION[scene_ctx])
    else:
        parts.append("记录下当时的场景（在门口 / 饭点 / 半夜…）与它的动作")

    if remaining:
        parts.append(
            f"再记录 {remaining} 次，我就能拿它跟自己的历史对比了"
        )
    return "；".join(parts)


_OBSERVATION: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "观察它是否伴随蹭腿、绕食盆走动、望向存放食物的地方",
    ContextLabel.DOOR_ATTENTION: "观察它是否伴随抓门、来回踱步、贴近门口",
    ContextLabel.AFFECTION_BRUSHING: "观察它是否主动靠近、蹭人、呼噜、翻肚皮",
    ContextLabel.ISOLATION_DISTRESS: "观察它是否躲藏、拒绝互动、或持续来回走动",
    ContextLabel.GREETING: "观察它是否竖尾靠近、蹭人后离开",
    ContextLabel.OTHER: "记录下当时的场景与它的动作，有助于下次判断",
}


def _limitations(
    features: AcousticFeatures,
    values: dict[str, float],
    records: list[MeowRecord],
    individual: IndividualModel | None,
) -> str:
    parts = [
        "仅凭叫声无法确定真实需求；若伴随持续焦躁、异常叫声或食欲改变，建议就医观察。",
        "**本次不给出概率判断**：这只猫的历史记录还不够，"
        "而群体先验尚未用实测数据建立（见 docs/DESIGN.md §7.3 U3）。",
    ]
    if individual is None or individual.is_cold_start:
        parts.append("还没有它自己的基线，因此测量值没有对比对象。")
    if records:
        parts.append(f"目前已记录 {len(records)} 条。")
    if features.quality is not FeatureQuality.GOOD:
        parts.append(f"本次录音质量评级为 {features.quality.value}，判别力已相应下调。")
    if "f0_mean" not in values:
        parts.append("未提取到有效基频（可能录音过短或噪声过大）。")
    if features.unavailable:
        parts.append(f"本次未参与判断的特征：{'、'.join(features.unavailable)}。")
    parts.append("测量值可复现，你可以自己核对；上面没有关于「它想干什么」的判断。")
    return " ".join(parts)
