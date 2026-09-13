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


def _prior_verdict(prior: PriorTable | None) -> tuple[bool, str]:
    """先验能不能用来产出后验概率。返回 `(可用, 原因)`。

    ## ★ 三种「不可用」，而不是一种

    初版只看 `is_placeholder`：只要不是占位值就算「已实测」，
    于是直接进入 `ACOUSTIC_PLUS_HISTORY` 并输出后验概率。

    换成 CatMeows 真实统计后，留出猫实测结果是：

        留出 macro-F1  0.364
        多数类基线     0.506

    **它比「总是猜多数类」还差。** 六个可用特征的分离度全在 0.32–0.72，
    分布大幅重叠。

    这种情况下输出后验概率**不是证据**，是一个看起来很确定的噪声 ——
    而用户无法分辨它和真实证据的区别。所以三种都要拦：

    | 情形 | 结论 |
    |---|---|
    | 占位数据 | 不可用 —— 数字是编的 |
    | 未做过留出评估 | 不可用 —— **无法确认**有区分度 |
    | 实测不如基线 | 不可用 —— 已确认**没有**区分度 |

    第三条最关键：它把「已实测」从「可用」里拆了出来。
    """
    if prior is None:
        return False, "未提供群体先验"

    if prior.is_placeholder:
        return False, "群体先验为占位数据，不产生后验概率"

    margin = prior.discrimination_margin
    if margin is None:
        return False, (
            "群体先验未做过留出验证，无法确认有区分度 —— 不产生后验概率"
        )

    if margin <= 0.0:
        return False, (
            f"群体先验实测**不具区分度**：留出 macro-F1 "
            f"{prior.holdout_macro_f1:.3f} ≤ 多数类基线 "
            f"{prior.holdout_majority_baseline:.3f}"
            f"（留出猫 {len(prior.holdout_cats)} 只）—— 不产生后验概率"
        )

    return True, (
        f"群体先验 {prior.version} 已实测且优于基线"
        f"（macro-F1 {prior.holdout_macro_f1:.3f} > 基线 "
        f"{prior.holdout_majority_baseline:.3f}）"
    )


def choose_mode(
    *,
    features: AcousticFeatures,
    records: list[MeowRecord] | None = None,
    prior: PriorTable | None = None,
) -> ModeDecision:
    """决定用哪个模式。**纯函数，不产生副作用。**"""
    usable, prior_note = _prior_verdict(prior)
    if usable:
        return ModeDecision(
            mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
            reason=prior_note,
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
