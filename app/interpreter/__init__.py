"""行为解释器：先按可用证据选模式，再解释。

**本模块不调用大模型。** 概率与计数由代码计算，保证可复现、可归因、可审计。

## 四种模式（入口用 `interpret_meow`）

| 模式 | 前提 | 输出 |
| --- | --- | --- |
| `ACOUSTIC_PLUS_HISTORY` | 群体先验**已实测** | 后验概率 |
| `CASE_BASED` | 某情境有 ≥3 个相似案例 | **计数**（「3 次里 2 次在门口」） |
| `MEASURED_ONLY` | 有音频，但上面两条不满足 | 仅测量值 + 场景假设 |
| `TEXT_ONLY` | 无音频 | 需更多信息（由图节点构造） |

**P0 的常态是 `CASE_BASED` 与 `MEASURED_ONLY`** ——
群体先验需标注数据，我们没有（`docs/16-model-selection.md` §4），
因此用「这只猫自己的标注历史」代替「猫类群体统计」。

用法：

    from app.interpreter import interpret_meow

    result, decision = interpret_meow(
        features=feats, records=my_records, scene="它对着门叫"
    )
    print(decision.reason)   # 为什么走到这个模式

## 旧入口仍可用

`interpret()`（纯贝叶斯）保留，但它需要一个**非占位**先验；
直接在生产用占位先验会得到一个编造的概率。**新代码请用 `interpret_meow`。**
"""

from app.interpreter.bayes import SimilarSample, interpret, match_scene
from app.interpreter.case_based import (
    FEATURE_SCALE,
    MAX_SIMILAR_CASES,
    MIN_CASES_PER_CONTEXT,
    MatchedCase,
    count_by_context,
    interpret_case_based,
    match_cases,
    similarity_between,
)
from app.interpreter.measured import interpret_measured_only
from app.interpreter.priors import (
    DEFAULT_KAPPA,
    FEATURE_ORDER,
    GaussianStat,
    IndividualModel,
    LabelledSample,
    PriorTable,
    PriorTableError,
)
from app.interpreter.router import ModeDecision, choose_mode, interpret_meow

__all__ = [
    # ── 入口 ──
    "interpret_meow",
    "choose_mode",
    "ModeDecision",
    # ── 各模式实现 ──
    "interpret",
    "interpret_case_based",
    "interpret_measured_only",
    "match_scene",
    # ── 案例推理 ──
    "match_cases",
    "similarity_between",
    "count_by_context",
    "MatchedCase",
    "FEATURE_SCALE",
    "MIN_CASES_PER_CONTEXT",
    "MAX_SIMILAR_CASES",
    # ── 旧贝叶斯路径 ──
    "SimilarSample",
    "PriorTable",
    "PriorTableError",
    "IndividualModel",
    "LabelledSample",
    "GaussianStat",
    "FEATURE_ORDER",
    "DEFAULT_KAPPA",
]
