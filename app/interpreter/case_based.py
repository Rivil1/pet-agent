"""案例推理：在这只猫**自己**的标注历史上做 k-NN。

对应决策 D31：推断模型由参数化贝叶斯改为案例推理。

## 为什么换掉参数化贝叶斯

```
P(f|k,c) = λ_c·P_ind + (1−λ_c)·P_pop
```

去掉群体先验后 `λ_c` **没有可收缩的对象**，公式直接失效。
而这并不坏 —— **参数化高斯本来就不适合 n=3**（当初加 std 收缩，
正是因为 3 个样本估不准方差）。

| | 参数化贝叶斯 | k-NN 案例推理 |
| --- | --- | --- |
| n=1 能工作吗 | ❌ 估不出方差 | ✅ |
| 输出形态 | `0.41`（编的） | **「3 次里 2 次在门口」**（真的） |
| 可解释性 | 需复算对数几率 | **直接指着历史说** |

## 它绕开了文献里那个致命缺陷

群体先验的 95.94% 不可采信，因为**跨个体泛化未验证**（21 只猫、交叉验证是否按个体分组未说明）。

案例推理只拿这只猫与**它自己**比 —— **不存在跨个体泛化问题**。

## 相似度：只比「两边都测到」的特征

这是本模块最容易写错的地方。若某个特征在一侧缺失就当成 0 或跳过比较，
缺失会静默变成「相同」—— 于是一条只采集了 2 个特征的记录会显得异常相似。

所以：**逐特征取交集**；交集特征太少时**该记录不参与匹配**（而不是降低标准）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.interpreter.priors import FEATURE_ORDER
from app.schemas import (
    AcousticFeatures,
    BehaviorInterpretation,
    CaseMatch,
    ContextLabel,
    EvidenceItem,
    EvidenceKind,
    EvidenceMode,
    IntentCandidate,
    MeowRecord,
)

#: 每个特征的「有意义的差异尺度」。用于把不同量纲的特征放到同一尺度上比较。
#:
#: ⚠️ **这是领域知识，不是从数据学的。**
#:
#: 正因为不能从数据学（我们没有标注数据 —— 那正是换掉群体先验的原因），
#: 它必须是一组**写死的、可审计的**数值，而不是估出来的。
#: 代价是它粗糙；收益是它不需要任何外部数据集。
#:
#: 取值思路：两个叫声在这个特征上差多少，你会说「这明显不一样」。
#: 如 `duration` 0.3s —— 0.4s 与 0.7s 是两种叫声；0.4s 与 0.45s 差不多。
FEATURE_SCALE: dict[str, float] = {
    "duration": 0.30,      # s
    "f0_mean": 80.0,       # Hz
    "f0_range": 70.0,      # Hz
    "f0_slope": 0.25,      # 归一化
    "call_rate": 2.0,      # 次/10s
    "ici_mean": 0.30,      # s
    "rms_mean": 0.08,      # 归一化
    "roughness": 0.12,     # 归一化
}

#: 参与匹配所需的最少共同特征数。
#:
#: 低于此值时相似度不可信（1–2 个特征比出来的「很像」没有意义），
#: **该记录不参与匹配** —— 而不是降低标准硬算。
MIN_COMMON_FEATURES = 4

#: 相似度下限。低于此值的案例不算「相似」，不进入计数。
SIMILARITY_FLOOR = 0.72

#: **单个情境**至少要有这么多相似案例，才给出该情境的计数。
#:
#: 这是 D31 的核心：阈值语义由「总样本数」改为「单个情境的相似案例数」。
#: 思想借自 Hermes 的技能提取（「同一模式重复出现 3 次才值得沉淀」）。
#:
#: 为什么按情境而不是按总数：总样本 8 条也可能分散在 6 个情境里，
#: 每处只有 1–2 条 —— 「3 次里 2 次」就成了噪声。
MIN_CASES_PER_CONTEXT = 3

#: 展示给用户的最相似案例条数。
MAX_SIMILAR_CASES = 3


@dataclass(frozen=True)
class MatchedCase:
    """一条匹配到的历史案例。"""

    record: MeowRecord
    similarity: float
    common_features: tuple[str, ...]
    """参与比较的特征。**可审计**：能看出这个相似度是基于什么算出来的。"""


def _usable(features: AcousticFeatures) -> dict[str, float]:
    """取出可用特征值，剔除显式标记不可用的与非法值。

    与 `app.interpreter.bayes._feature_vector` 同一口径 ——
    两个模式对「什么算可用特征」的判定必须一致，否则同一段音频
    在冷启动与案例推理下会看到不同的特征集。
    """
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


def similarity_between(
    a: AcousticFeatures, b: AcousticFeatures
) -> tuple[float | None, tuple[str, ...]]:
    """两条叫声的相似度。

    Returns:
        ``(相似度, 参与比较的特征名)``。**共同特征不足时相似度为 ``None``** ——
        表示「无法比较」，而不是「不相似」。这两者在小样本下差别很大：
        「不相似」会稀释分母，而事实是我们根本不知道。
    """
    va, vb = _usable(a), _usable(b)
    common = tuple(f for f in FEATURE_ORDER if f in va and f in vb)
    if len(common) < MIN_COMMON_FEATURES:
        return None, common

    # 各特征按领域尺度归一后取均方距离 → 映射到 (0, 1] 的相似度
    total = 0.0
    for f in common:
        z = (va[f] - vb[f]) / FEATURE_SCALE[f]
        total += z * z
    distance = math.sqrt(total / len(common))
    return 1.0 / (1.0 + distance), common


def match_cases(
    features: AcousticFeatures,
    records: list[MeowRecord],
    *,
    floor: float = SIMILARITY_FLOOR,
    limit: int | None = None,
) -> list[MatchedCase]:
    """在记录集合里找相似案例，按相似度降序。

    **只使用已确认的记录**（`is_confirmed`）——
    未确认的样本不构成证据（与 `SimilarSample.context is None` 不参与计算同理）。
    """
    matches: list[MatchedCase] = []
    for rec in records:
        if not rec.is_confirmed:
            continue
        sim, common = similarity_between(features, rec.features)
        if sim is None or sim < floor:
            continue
        matches.append(
            MatchedCase(record=rec, similarity=sim, common_features=common)
        )

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:limit] if limit is not None else matches


def count_by_context(matches: list[MatchedCase]) -> dict[ContextLabel, int]:
    """各情境的相似案例条数。**这就是输出给用户的「3 次」里的分子。**"""
    counts: dict[ContextLabel, int] = {}
    for m in matches:
        ctx = m.record.context
        counts[ctx] = counts.get(ctx, 0) + 1
    return counts


def interpret_case_based(
    *,
    features: AcousticFeatures,
    records: list[MeowRecord],
) -> BehaviorInterpretation:
    """按这只猫自己的标注案例给出解释。

    **不输出后验概率**，输出的是**计数**：

    > 「它这样叫过 3 次，最像这次的那次是你开了门它出去了」
    > 「3 次里有 2 次在门口」

    计数是可机械核对的事实；而 3 个样本算出的「67%」是噪声。
    """
    matches = match_cases(features, records)
    if not matches:
        raise ValueError(
            "没有相似案例时不应进入 case_based 模式 —— "
            "调用方应先判断冷启动（见 app/interpreter/router.py）。"
        )

    counts = count_by_context(matches)
    total = len(matches)

    # 只有达到门限的情境才给出计数
    qualified = {c: n for c, n in counts.items() if n >= MIN_CASES_PER_CONTEXT}
    if not qualified:
        raise ValueError(
            f"没有任何情境达到 {MIN_CASES_PER_CONTEXT} 个相似案例 —— "
            "调用方应先判断冷启动。"
        )

    ordered = sorted(qualified.items(), key=lambda kv: (-kv[1], kv[0].value))
    candidates = [
        IntentCandidate(
            context=ctx,
            display=(
                f"它这样叫过 {total} 次，其中 {n} 次是「{_CONTEXT_SHORT[ctx]}」"
            ),
            matched_count=n,
            posterior=None,  # 契约强制：case_based 禁止数值后验
            log_odds=None,
        )
        for ctx, n in ordered
    ]

    return BehaviorInterpretation(
        evidence_mode=EvidenceMode.CASE_BASED,
        acoustic_features=features,
        candidates=candidates,
        evidence=_build_case_evidence(matches, qualified, total),
        similar_cases=[
            CaseMatch(
                record_id=m.record.record_id or "",
                similarity=round(m.similarity, 4),
                context=m.record.context,
                actions=list(m.record.actions),
                resolution=m.record.resolution,
                recorded_at=m.record.recorded_at,
            )
            for m in matches[:MAX_SIMILAR_CASES]
        ],
        case_total=total,
        individualization=0.0,
        sample_count=len(records),
        suggested_observation=_observation_for(candidates, matches),
        limitations=_limitations(total, qualified, features),
        prior_version=None,  # 案例推理不使用群体先验
    )


#: 情境的短名，用在「其中 N 次是『门口』」这样的句子里。
_CONTEXT_SHORT: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "饭点",
    ContextLabel.DOOR_ATTENTION: "门口",
    ContextLabel.AFFECTION_BRUSHING: "求摸或梳毛",
    ContextLabel.ISOLATION_DISTRESS: "独处不安",
    ContextLabel.GREETING: "打招呼",
    ContextLabel.OTHER: "其他",
}


def _build_case_evidence(
    matches: list[MatchedCase],
    qualified: dict[ContextLabel, int],
    total: int,
) -> list[EvidenceItem]:
    """构建证据列表。

    **RETRIEVED 项的 `log_odds_contribution` 必须为 None** ——
    case_based 模式下没有任何概率计算（契约会校验这一点）。
    填 0.0 会谎称「已参与计算但影响为零」。
    """
    items: list[EvidenceItem] = []

    for m in matches[:MAX_SIMILAR_CASES]:
        parts = [f"最像这次的那次（相似度 {m.similarity:.2f}）"]
        if m.record.resolution:
            parts.append(f"你记的结果是「{m.record.resolution}」")
        else:
            parts.append("你没有记录结果")
        items.append(
            EvidenceItem(
                kind=EvidenceKind.RETRIEVED,
                statement="；".join(parts),
                source=f"meow_record:{m.record.record_id}",
                value=round(m.similarity, 4),
                reference=None,
                log_odds_contribution=None,
            )
        )

    for ctx, n in sorted(qualified.items(), key=lambda kv: (-kv[1], kv[0].value)):
        items.append(
            EvidenceItem(
                kind=EvidenceKind.RETRIEVED,
                statement=(
                    f"相似的历史里，{total} 次有 {n} 次是「{_CONTEXT_SHORT[ctx]}」"
                ),
                source=f"case_count:{ctx.value}",
                value=float(n),
                reference=float(total),
                log_odds_contribution=None,
            )
        )

    return items


def _observation_for(
    candidates: list[IntentCandidate], matches: list[MatchedCase]
) -> str:
    """建议观察项。

    优先复述**主人自己记过的结果** —— 那比任何系统建议都有用，
    因为它是主人自己的经验被调回来了。
    """
    with_resolution = [m for m in matches if m.record.resolution]
    if with_resolution:
        best = with_resolution[0]
        return (
            f"上次最像的这次，你的处理是「{best.record.resolution}」。"
            "可以看看这次是否也是同样的需求。"
        )
    top = candidates[0].context if candidates else None
    return _OBSERVATION[top] if top else "记录下当时的场景与它的动作，有助于下次判断"


_OBSERVATION: dict[ContextLabel, str] = {
    ContextLabel.FOOD_WAITING: "观察它是否伴随蹭腿、绕食盆走动、望向存放食物的地方",
    ContextLabel.DOOR_ATTENTION: "观察它是否伴随抓门、来回踱步、贴近门口",
    ContextLabel.AFFECTION_BRUSHING: "观察它是否主动靠近、蹭人、呼噜、翻肚皮",
    ContextLabel.ISOLATION_DISTRESS: "观察它是否躲藏、拒绝互动、或持续来回走动",
    ContextLabel.GREETING: "观察它是否竖尾靠近、蹭人后离开",
    ContextLabel.OTHER: "记录下当时的场景与它的动作，有助于下次判断",
}


def _limitations(
    total: int,
    qualified: dict[ContextLabel, int],
    features: AcousticFeatures,
) -> str:
    parts = [
        "仅凭叫声无法确定真实需求；若伴随持续焦躁、异常叫声或食欲改变，建议就医观察。",
        f"本次依据的是**你自己记录**的 {total} 条相似历史，不是群体统计"
        f"（相似度按特征距离计算，阈值 {SIMILARITY_FLOOR}）。",
    ]
    dropped = total - sum(qualified.values())
    if dropped > 0:
        parts.append(
            f"另有 {dropped} 条相似案例所属情境的样本不足 "
            f"{MIN_CASES_PER_CONTEXT} 条，未给出计数 —— "
            "样本太少时「3 次里 1 次」没有意义。"
        )
    no_resolution = 0
    if features.quality.value != "good":
        parts.append(f"本次录音质量评级为 {features.quality.value}，判别力已相应下调。")
    if no_resolution:
        parts.append(f"有 {no_resolution} 条案例没有记录结果。")
    if features.unavailable:
        parts.append(f"本次未参与判断的特征：{'、'.join(features.unavailable)}。")
    parts.append(
        "相似度由特征距离计算，**不是**对「它想干什么」的判断。"
        "请以历史案例的具体内容为准。"
    )
    return " ".join(parts)


def describe_matches(matches: list[MatchedCase]) -> list[dict[str, Any]]:
    """调试/评测用：把匹配结果摊平成可打印的形状。"""
    return [
        {
            "record_id": m.record.record_id,
            "context": m.record.context.value,
            "similarity": round(m.similarity, 4),
            "common_features": len(m.common_features),
            "resolution": m.record.resolution,
        }
        for m in matches
    ]
