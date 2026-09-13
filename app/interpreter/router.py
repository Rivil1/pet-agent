"""行为解释的入口：**按可用证据选择模式**。

## 为什么需要一个路由层

项目有四种证据模式，而调用方（图节点）不该自己判断该用哪种：

| 模式 | 前提 | 输出 |
| --- | --- | --- |
| `ACOUSTIC_PLUS_HISTORY` | 群体先验**已实测** | 后验概率 |
| `CASE_BASED` | 该情境有 ≥ `MIN_CASES_PER_CONTEXT` 个相似案例 | **计数** |
| `MEASURED_ONLY` | 有音频，但上面两条都不满足 | 仅测量值 |
| `TEXT_ONLY` | 无音频 | 需更多信息 |

若让调用方自己判断，判断逻辑会散落各处，且**某处漏判会静默产生一个基于占位先验的后验** ——
那正是 D31 要防的事。

## 选择顺序是 fail-closed 的

```
先验可用？ ──是──► ACOUSTIC_PLUS_HISTORY
    │否
    ▼
案例够多？ ──是──► CASE_BASED
    │否
    ▼
            MEASURED_ONLY   ← 诚实的少，但不会编
```

**默认往下走，而不是往上走。** 占位先验永远不产生后验概率 ——
这不是靠调用方自觉，而是 `PriorTable.is_placeholder` 这一个标志位决定的。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.interpreter import bayes
from app.interpreter.case_based import (
    MIN_CASES_PER_CONTEXT,
    count_by_context,
    match_cases,
)
from app.interpreter.case_based import interpret_case_based as _case_based
from app.interpreter.measured import interpret_measured_only as _measured_only
from app.interpreter.priors import IndividualModel, PriorTable
from app.schemas import (
    AcousticFeatures,
    BehaviorInterpretation,
    EvidenceMode,
    MeowRecord,
)


@dataclass(frozen=True)
class ModeDecision:
    """模式选择的结果。

    **它是可观测的** —— 调用方可以记录「为什么走到这个模式」，
    而不是只看到一个结果。冷启动阶段最需要的正是这个解释。
    """

    mode: EvidenceMode
    reason: str
    matched_records: int = 0
    qualified_contexts: int = 0


def choose_mode(
    *,
    features: AcousticFeatures,
    records: list[MeowRecord] | None = None,
    prior: PriorTable | None = None,
) -> ModeDecision:
    """决定用哪个模式。**纯函数，不产生副作用。**"""
    if prior is not None and not prior.is_placeholder:
        return ModeDecision(
            mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
            reason=f"群体先验 {prior.version} 已实测（provenance={prior.provenance}）",
        )

    prior_note = (
        "群体先验为占位数据，不产生后验概率"
        if prior is not None
        else "未提供群体先验"
    )

    matches = match_cases(features, records or [])
    counts = count_by_context(matches)
    qualified = {c: n for c, n in counts.items() if n >= MIN_CASES_PER_CONTEXT}

    if qualified:
        return ModeDecision(
            mode=EvidenceMode.CASE_BASED,
            reason=(
                f"有 {len(qualified)} 个情境达到 {MIN_CASES_PER_CONTEXT} 个相似案例；"
                f"{prior_note}"
            ),
            matched_records=len(matches),
            qualified_contexts=len(qualified),
        )

    return ModeDecision(
        mode=EvidenceMode.MEASURED_ONLY,
        reason=(
            f"相似案例 {len(matches)} 条，没有任何情境达到 "
            f"{MIN_CASES_PER_CONTEXT} 条；{prior_note}"
        ),
        matched_records=len(matches),
        qualified_contexts=0,
    )


def interpret_meow(
    *,
    features: AcousticFeatures,
    records: list[MeowRecord] | None = None,
    prior: PriorTable | None = None,
    individual: IndividualModel | None = None,
    scene: str | None = None,
    similar_samples: list[bayes.SimilarSample] | None = None,
) -> tuple[BehaviorInterpretation, ModeDecision]:
    """按可用证据选择模式并解释。

    Returns:
        ``(解释结果, 模式决策)``。决策单独返回是为了让**「为什么是这个模式」可见** ——
        冷启动阶段用户最需要的正是这个解释。
    """
    decision = choose_mode(features=features, records=records, prior=prior)

    if decision.mode is EvidenceMode.ACOUSTIC_PLUS_HISTORY:
        assert prior is not None  # noqa: S101 — choose_mode 已保证
        result = bayes.interpret(
            features=features,
            prior=prior,
            individual=individual,
            scene=scene,
            similar_samples=similar_samples,
        )
        return result, decision

    if decision.mode is EvidenceMode.CASE_BASED:
        result = _case_based(features=features, records=records or [])
        return _with_scene_evidence(result, scene), decision

    result = _measured_only(
        features=features,
        scene=scene,
        individual=individual,
        records=records or [],
    )
    return _with_scene_evidence(result, scene), decision


def _with_scene_evidence(
    result: BehaviorInterpretation, scene: str | None
) -> BehaviorInterpretation:
    """把用户提供的场景作为一条 PRIOR 证据补进去。

    **两个非贝叶斯模式都要走这里** —— 否则同一个输入在不同模式下
    证据列表的形状不一致（初版就漏了 `measured_only`）。

    与贝叶斯路径的处理保持一致：`bayes._build_evidence` 也会为场景
    插入一条 PRIOR 项。用户提供的信息属于「主人记录」那一层，
    无论走到哪个模式都应该出现在证据里。

    **只补证据，不改计数。** 场景是「主人的描述」，而计数是「相似案例统计」——
    混在一起会让用户以为场景被算进了相似度，而它没有。
    """
    if not scene:
        return result

    from app.schemas import EvidenceItem, EvidenceKind

    note = EvidenceItem(
        kind=EvidenceKind.PRIOR,
        statement=f"你提到当时的情况是「{scene}」（这条不计入上面的相似案例统计）",
        source="scene_prior:user_input",
        value=None,
        reference=None,
        log_odds_contribution=None,
    )
    return result.model_copy(update={"evidence": [note, *result.evidence]})
