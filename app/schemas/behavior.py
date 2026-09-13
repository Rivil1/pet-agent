"""行为解释契约。

设计全文见 docs/03-behavior-interpreter.md。

本模块把两条核心约束编码进类型系统：

1. **没有声学证据就不许给数值置信度**：``evidence_mode == TEXT_ONLY`` 时，
   结构上禁止携带 ``acoustic_features`` 与数值 ``posterior``。
   否则系统会退化成「用户描述什么就顺着说什么」的谄媚模型。
2. **证据必须可归因**：每条 ``EvidenceItem`` 都要有 ``kind`` 与
   ``log_odds_contribution``（可由特征值与参考值复算）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.memory import MemorySource, MemoryStatus


class ContextLabel(str, Enum):
    """预测目标是 **情境 context**，不是 **意图 intent**。

    理由：用户能可靠标注「当时在干什么」，无法标注「猫当时想什么」。
    用不可验证的目标做监督学习，评测无从谈起（docs/03 §1）。
    """

    FOOD_WAITING = "food_waiting"
    DOOR_ATTENTION = "door_attention"
    AFFECTION_BRUSHING = "affection_brushing"
    ISOLATION_DISTRESS = "isolation_distress"
    GREETING = "greeting"
    OTHER = "other"


class FeatureQuality(str, Enum):
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    """低质量时下游必须降低置信度（docs/03 §9 L3）"""


class EvidenceMode(str, Enum):
    """证据模式。**它决定了输出什么数值，以及禁止输出什么数值。**

    | 模式 | 有音频 | 有可用模型 | 允许 posterior | 允许计数 |
    |---|---|---|---|---|
    | ACOUSTIC_PLUS_HISTORY | ✅ | 群体先验（已实测） | ✅ | ❌ |
    | CASE_BASED | ✅ | 个体标注案例 | ❌ | ✅ |
    | MEASURED_ONLY | ✅ | **无** | ❌ | ❌ |
    | TEXT_ONLY | ❌ | 无 | ❌ | ❌ |

    P0 的常态是 **CASE_BASED** 与 **MEASURED_ONLY**：
    群体先验需标注数据，我们没有（docs/16-model-selection.md §4），
    因此用「这只猫自己的标注历史」代替「猫类群体统计」。
    """

    ACOUSTIC_PLUS_HISTORY = "acoustic_plus_history"
    """群体先验已实测可用 → 输出后验概率。**当前未启用**（先验仍为占位）。"""

    CASE_BASED = "case_based"
    """按**这只猫自己**的标注案例做 k-NN 推断。

    输出的是**计数**（「3 次里 2 次在门口」）而不是概率：
    3 个样本估不出一个可信的分布，但「3 次里 2 次」是可直接核查的事实。

    为什么它绕开了群体先验的根本缺陷：只拿这只猫跟它自己比，
    **不存在跨个体泛化问题**（那正是文献 95.94% 不可采信的原因）。
    """

    MEASURED_ONLY = "measured_only"
    """冷启动：有测量、但没有可用的推断模型。

    **禁止输出数值置信度与计数** —— 样本不足时的任何比例都是噪声。
    只输出可核查的测量值 + 用户自己提供的场景。
    """

    TEXT_ONLY = "text_only"
    """降级模式：无音频或特征提取失败。
    **禁止输出数值置信度**——没有声学证据就不该有概率。"""


#: 禁止输出数值后验的模式。
#:
#: 三种模式的**理由各不相同**，但结论一致 —— 不能把推测包装成测量：
#: - ``CASE_BASED``：有真实案例，但样本量支持不了分布 → 只给计数
#: - ``MEASURED_ONLY``：根本没有推断模型
#: - ``TEXT_ONLY``：根本没有声学证据
_NO_POSTERIOR_MODES = frozenset(
    {EvidenceMode.CASE_BASED, EvidenceMode.MEASURED_ONLY, EvidenceMode.TEXT_ONLY}
)

#: 必须有声学特征的模式（即：除了「没音频」之外全部）。
_WITH_FEATURES_MODES = frozenset(
    {
        EvidenceMode.ACOUSTIC_PLUS_HISTORY,
        EvidenceMode.CASE_BASED,
        EvidenceMode.MEASURED_ONLY,
    }
)


class BehaviorAction(str, Enum):
    """主人可观察到的**动作**。自由文本无法可靠匹配，故用固定词表。

    为什么只收动作而不是意图：
    主人能可靠观察「它在抓门」，无法可靠观察「它想出去」。
    意图是推断，动作是观察 —— 用可验证的目标做监督才谈得上评测。
    """

    SCRATCH_DOOR = "scratch_door"
    PACING = "pacing"
    RUB_LEG = "rub_leg"
    TAIL_UP = "tail_up"
    NEAR_FOOD_BOWL = "near_food_bowl"
    LOOK_AT_DOOR = "look_at_door"
    HIDING = "hiding"
    PURRING = "purring"
    BELLY_UP = "belly_up"
    APPROACH_HUMAN = "approach_human"
    AVOID_CONTACT = "avoid_contact"
    GROOMING = "grooming"
    ARCHED_BACK = "arched_back"
    OTHER = "other"


_ACTION_DISPLAY: dict[BehaviorAction, str] = {
    BehaviorAction.SCRATCH_DOOR: "抓门",
    BehaviorAction.PACING: "来回走",
    BehaviorAction.RUB_LEG: "蹭腿",
    BehaviorAction.TAIL_UP: "竖尾",
    BehaviorAction.NEAR_FOOD_BOWL: "绕着食盆",
    BehaviorAction.LOOK_AT_DOOR: "望着门",
    BehaviorAction.HIDING: "躲起来",
    BehaviorAction.PURRING: "呼噜",
    BehaviorAction.BELLY_UP: "翻肚皮",
    BehaviorAction.APPROACH_HUMAN: "主动靠近",
    BehaviorAction.AVOID_CONTACT: "拒绝互动",
    BehaviorAction.GROOMING: "舔毛",
    BehaviorAction.ARCHED_BACK: "弓背炸毛",
    BehaviorAction.OTHER: "其他",
}


def action_label(action: BehaviorAction) -> str:
    """动作的中文表述。供渲染层与证据文案使用。"""
    return _ACTION_DISPLAY.get(action, action.value)


class EvidenceKind(str, Enum):
    """证据来源。

    ## 一条会变紧的约束

    早期这里写的是「**LLM 不得产生证据项**」。那个说法过于粗 ——
    它把两件事混成了一件：

    | | 谁来产 | 能否重算出同一个数 |
    | --- | --- | --- |
    | 测量 | 代码 | ✅ 能 |
    | 观察 | **可以**是多模态模型 | ❌ 不能 |

    真正要守的不是「谁说的」，而是**「能不能重算」**。
    所以现在允许模型产生 `OBSERVED` 项，但**仍禁止它伪装成 `MEASURED`** ——
    后者由 `BehaviorInterpretation` 的校验器强制（`MEASURED` 必须带 `value`）。
    """

    MEASURED = "measured"
    """由**代码**从音频/图像计算，可复现。**必须带 `value`。**"""

    RETRIEVED = "retrieved"
    """来自这只猫的历史样本（已被主人确认）。"""

    PRIOR = "prior"
    """来自场景描述或公开数据集先验。"""

    OBSERVED = "observed"
    """**由多模态模型观察得到**的描述（如「画面里有一只猫面向门」）。

    ## 为什么需要单独一栏，而不是归入 MEASURED

    多模态模型对**关系性描述**（猫 ↔ 门/人/物）是可靠的，
    但那不是测量：同一个输入跑两次可能得到不同措辞，
    而 `MEASURED` 的全部意义在于「**重算一次还是它**」。

    把两者混在一栏，会让 `value` / `reference` 的复算承诺失效 ——
    而那个承诺是整个证据链可审计性的地基。

    ## 两条硬约束

    1. `source` 必须标明**哪个模型**（如 `model:qwen3-vl-flash`）。
       模型会换、会升级，不记来源就无法回溯一个判断是怎么得出的。
    2. **不带 `log_odds_contribution`** —— 它不在概率模型里，
       给它一个对数几率贡献值是编造。

    ## 它不得用来做这两件事

    - **推断因果**：「开门让它停了」是主人的判断，不是画面里的事实
    - **判断神态/健康**：`general.demeanor` 是**红旗信号**，
      用未经校验的模型推断去驱动它是危险的（见 `docs/DESIGN.md` §5.4）
    """


class AcousticFeatures(BaseModel):
    """声学特征。提取工具：librosa + pyin（docs/DESIGN.md §3.6）。

    ``unavailable`` 列出**本次无法可靠测量**的特征。

    为什么需要它：把测不出的特征零填充是**静默编造** ——
    下游会把它当成真的「速率为 0」参与似然计算。
    宁可显式标记缺失，也不给一个假数字。
    推理侧必须跳过 ``unavailable`` 中的特征（见 ``app/interpreter/bayes.py``）。
    """

    duration: float = Field(description="叫声时长（秒）")
    f0_mean: float = Field(description="基频均值（Hz），剔除 voiced 置信度 <0.15 的帧")
    f0_range: float = Field(description="基频 P5–P95 分位跨度")
    f0_slope: float = Field(
        description="基频轮廓线性拟合斜率。**方向性含义有文献支持**："
        "正向/亲和情境倾向上升，负向倾向下降，故归因时是最强单项证据。"
    )
    call_rate: float = Field(
        description="叫声速率（次/10s，索求行为强度指标）。"
        "**需要足够长的观测窗口**；窗口过短时列入 unavailable。"
    )
    ici_mean: float = Field(
        description="相邻叫声间隔均值（秒）。只有一次叫声时无意义，列入 unavailable。"
    )
    rms_mean: float = Field(description="能量包络均值")
    roughness: float = Field(description="粗糙度（幅度调制深度）")

    estimated_snr_db: float | None = None
    quality: FeatureQuality = FeatureQuality.GOOD

    unavailable: list[str] = Field(
        default_factory=list,
        description="本次无法可靠测量的特征名。推理时**必须跳过**。",
    )


class EvidenceItem(BaseModel):
    """一条可追溯的证据。

    **这是「证据」与「讲故事」的分界线**：每条都必须能回溯到
    measured（计算）/ retrieved（检索）/ prior（场景或数据集）。
    """

    kind: EvidenceKind
    statement: str = Field(
        description="用户可读的说明，如「叫声时长 0.82s，长于它日常索食均值 0.41s」"
    )
    source: str = Field(
        description="如 'acoustic:duration' / 'meow_sample:8f2a' / 'dataset:catmeows'"
    )
    value: float | None = None
    reference: float | None = Field(
        default=None, description="对照参考值（个体基线或群体统计）"
    )
    log_odds_contribution: float | None = Field(
        default=None,
        description=(
            "该项对最终判断的对数几率贡献。"
            "**必须能由 value 与 reference 复算得到**，否则不可审计。\n\n"
            "在 ACOUSTIC_PLUS_HISTORY 下**必填**（没有归因的证据不是证据）；"
            "其余模式**必须为 None** —— 没有概率模型时填 0.0 会谎称「已参与计算但影响为零」。"
        ),
    )


class IntentCandidate(BaseModel):
    """一个候选情境。"""

    context: ContextLabel
    posterior: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="后验概率。**由评分层计算，LLM 不得写入或修改。**",
    )
    display: str = Field(description="用户可读表述，如「很可能是在等吃的」")
    log_odds: float | None = None

    matched_count: int = Field(
        default=0,
        ge=0,
        description=(
            "CASE_BASED 专用：相似历史案例中属于该情境的**条数**。\n\n"
            "这是计数不是概率 —— 3 个样本的「67%」是噪声，"
            "而「3 次里 2 次」是可直接核查的事实。其余模式必须为 0。"
        ),
    )


class CaseMatch(BaseModel):
    """一条与本次叫声相似的历史标注案例。

    这是 CASE_BASED 模式的核心产物：**把「结论」换成「先例」**。
    不说「它有 41% 可能是要出去」，而是说
    「上次它这样叫、也在门口，你开了门它就出去了」。

    后者用户能自己核查对错，前者不能 —— 而核查能力就是这套系统仅有的诚实基础。
    """

    record_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    context: ContextLabel
    actions: list[BehaviorAction] = Field(default_factory=list)
    resolution: str | None = Field(
        default=None, description="主人记录的结果，如「开门它就出去了」。**可空。**"
    )
    recorded_at: datetime | None = None


class MeowRecord(BaseModel):
    """一条**已被主人标注**的叫声记录。

    这是案例推理（k-NN）的样本，也是「学习主人的经验」这个定位的落点：
    主人提供**可观察**的情境与动作，以及**结果**（什么让它停了）——
    而不是「它当时的意图」（那不可验证，见 GLOSSARY）。

    ## 为什么 `resolution`（结果）字段最要紧

    `docs/14` 与 Hermes 的经验都指向同一件事：

    > 技能里最有价值的不是 happy path，而是**撞过又纠正的部分**。

    对应到这里就是：**记录什么让它停了，比记录它想干什么有用**。
    「开门它就出去了」是一个**可验证的结果**；「它想出去」是一个推断。

    ## 为什么只收词表内的动作

    自由文本无法可靠地做相似度匹配。「抓门」与「挠门」在字符串上是两回事，
    在行为上是一回事。所以动作用 `BehaviorAction` 固定词表。
    """

    record_id: str | None = None
    user_id: str
    pet_id: str = Field(description="多租户隔离键。检索必须按此过滤。")
    session_id: str | None = Field(
        default=None,
        description=(
            "产生这条标注的会话。**全链路追溯用**：能回答「这条案例是哪一轮录的」。\n\n"
            "它不是隔离键，也不进相似度计算。"
        ),
    )

    context: ContextLabel = Field(description="**主人当时观察到的情况**，不是意图")
    features: AcousticFeatures

    actions: list[BehaviorAction] = Field(
        default_factory=list, description="主人观察到的动作（可多选）"
    )
    resolution: str | None = Field(
        default=None,
        description="**结果**：后来什么让它停了，如「开门它就出去了」。可为空。",
    )

    recorded_at: datetime
    source: MemorySource = MemorySource.USER_OBSERVATION
    status: MemoryStatus = MemoryStatus.ACTIVE

    @model_validator(mode="after")
    def _system_inference_cannot_be_active(self) -> MeowRecord:
        """系统归纳不得作为已确认的案例参与推理（防自我强化）。

        与 `MemoryEvent` 同一条不变量：若让模型自己产的标签进入案例库，
        它会在后续推理中被当作事实引用 —— 幻觉自我强化。
        """
        if (
            self.source is MemorySource.SYSTEM_INFERENCE
            and self.status is MemoryStatus.ACTIVE
        ):
            raise ValueError(
                "SYSTEM_INFERENCE 不得以 ACTIVE 状态存在（防自我强化）。"
                "系统推测的情境只能先以 PENDING_CONFIRMATION 存在，"
                "待主人确认后把 source 改为 USER_CONFIRMATION 再转 ACTIVE。"
            )
        return self

    @property
    def is_confirmed(self) -> bool:
        """只有确认过的案例参与推理。未确认的不构成证据。"""
        return self.status is MemoryStatus.ACTIVE


class PendingInterpretation(BaseModel):
    """一次解释的存档，等待主人标注。

    ## 为什么特征必须存在服务端

    主人标注时要提供「情境 / 动作 / 结果」，但**不能提供声学特征**。

    理由：`AcousticFeatures` 的承诺是 `MEASURED` —— 由代码计算、可复现。
    若允许客户端提交特征，那个承诺立刻失效：客户端可以发任意数字，
    而案例推理的相似度会建在它们上面。

    所以流程是：

    ```
    POST /v1/interpret   → 服务端提取特征 + 解释 → 返回 interpretation_id
    POST .../meow-records → 客户端只发「情境/动作/结果」+ id
                          → 服务端按 id 取回**它自己存的**特征
    ```

    ## 为什么不让客户端重传音频

    重传会重新提取特征，而音频解码/librosa 在不同环境下可能有微小差异 ——
    那样标注指向的特征与当时看到解释时的特征**不是同一份**。
    """

    interpretation_id: str
    user_id: str
    pet_id: str
    session_id: str | None = Field(
        default=None,
        description=(
            "产生这次解释的会话。**主人标注时凭它归属到会话**，\n"
            "也是将来解析「那它为什么这样」所需的上下文（当前未实现）。"
        ),
    )
    features: AcousticFeatures
    evidence_mode: EvidenceMode
    candidates: list[IntentCandidate] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BehaviorInterpretation(BaseModel):
    """行为解释结果。"""

    evidence_mode: EvidenceMode

    acoustic_features: AcousticFeatures | None = None
    candidates: list[IntentCandidate] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    similar_cases: list[CaseMatch] = Field(
        default_factory=list,
        description=(
            "CASE_BASED 专用：与本次最相似的几条历史案例，**含具体内容**。\n\n"
            "为什么要带内容而不只带相似度：用户能核查「上次是你开了门它出去了」，"
            "这一个事实比任何概率都有用。"
        ),
    )
    case_total: int = Field(
        default=0,
        ge=0,
        description="CASE_BASED 专用：参与统计的相似案例总数（即计数的分母）。",
    )

    individualization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="λ_c = n_c/(n_c+κ)，个体化程度。可对用户展示为「系统对它的了解程度」",
    )
    sample_count: int = Field(default=0, description="该猫已确认的样本数 n_c")

    suggested_observation: str = Field(
        description="建议观察项。**任何情况下都必填**——把不确定性交还给用户"
    )
    limitations: str = Field(
        default="仅凭叫声无法确定真实需求；若伴随持续焦躁或异常叫声，建议就医观察。"
    )

    prior_version: str | None = Field(
        default=None,
        description="先验数据版本。要求可复现：同一特征 + 同一先验版本 → 同一后验（docs/03 §11）",
    )

    # ── 结构性约束 ───────────────────────────────────────────

    @model_validator(mode="after")
    def _no_posterior_without_model(self) -> BehaviorInterpretation:
        """**硬约束**：没有可信推断模型时，禁止输出数值后验。

        三种模式的理由不同（见 ``_NO_POSTERIOR_MODES``），但后果一样：
        没有这条约束，系统会把推测包装成测量，退化为「用户说什么就顺着说什么」
        的谄媚模型。
        """
        if self.evidence_mode in _NO_POSTERIOR_MODES:
            if any(c.posterior is not None for c in self.candidates):
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 时禁止输出 posterior 数值。"
                    "没有可信的推断模型就不该有概率——否则是在把推测包装成测量。"
                )
            if any(c.log_odds is not None for c in self.candidates):
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 时禁止输出 log_odds。"
                    "对数几率同样来自那套不可信的模型。"
                )
        return self

    @model_validator(mode="after")
    def _text_only_forbids_features(self) -> BehaviorInterpretation:
        """无音频模式不得携带声学特征。"""
        if (
            self.evidence_mode is EvidenceMode.TEXT_ONLY
            and self.acoustic_features is not None
        ):
            raise ValueError("evidence_mode=text_only 时不得携带 acoustic_features")
        return self

    @model_validator(mode="after")
    def _contribution_matches_mode(self) -> BehaviorInterpretation:
        """对数几率贡献必须与模式一致。

        这把「证据必须可归因」从「字段必填」收紧为「**按模式必填**」：
        - 有概率模型的模式下，每条证据都必须带贡献值（否则证据不可审计）
        - 没有概率模型的模式下，贡献值必须为 ``None``
          —— 填 0.0 会谎称「已参与计算但影响为零」，而事实是它没参与任何计算
        """
        has_model = self.evidence_mode is EvidenceMode.ACOUSTIC_PLUS_HISTORY
        for item in self.evidence:
            # **OBSERVED 永远不参与概率模型** —— 它是模型的观察描述，
            # 不是概率计算的输入。给它一个对数几率贡献值是编造。
            needs_contribution = has_model and item.kind is not EvidenceKind.OBSERVED
            if needs_contribution and item.log_odds_contribution is None:
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 下证据 {item.source!r} "
                    "缺少 log_odds_contribution——不可归因的证据不是证据。"
                )
            if not needs_contribution and item.log_odds_contribution is not None:
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 下证据 {item.source!r} "
                    "不得携带 log_odds_contribution——本模式没有任何概率计算。"
                )
        return self

    @model_validator(mode="after")
    def _measured_evidence_has_a_number(self) -> BehaviorInterpretation:
        """**`MEASURED` 必须带 `value`。**

        这是把「模型不能伪装成测量」从一句 docstring 变成一条断言：
        测量值的定义就是「有一个可复算的数字」，没有数字的就不是测量。

        多模态模型的产出应该用 `OBSERVED`（它不需要 `value`）。
        """
        for item in self.evidence:
            if item.kind is EvidenceKind.MEASURED and item.value is None:
                raise ValueError(
                    f"MEASURED 证据 {item.source!r} 必须带 value —— "
                    "测量值的定义就是可复算的数字。"
                    "若这是多模态模型的观察描述，请用 EvidenceKind.OBSERVED。"
                )
        return self

    @model_validator(mode="after")
    def _case_data_only_in_case_based(self) -> BehaviorInterpretation:
        """计数与案例只在 CASE_BASED 下允许。

        防止把「3 次里 2 次」这类计数混进没有案例的模式，
        也防止 CASE_BASED 声称案例却没有。
        """
        if self.evidence_mode is EvidenceMode.CASE_BASED:
            if not self.similar_cases:
                raise ValueError(
                    "evidence_mode=case_based 时必须提供 similar_cases"
                    "——否则「案例推理」无案例可查。"
                )
            if self.case_total < len(self.similar_cases):
                raise ValueError(
                    f"case_total={self.case_total} 小于 similar_cases "
                    f"({len(self.similar_cases)})——分母不能小于展示的案例数。"
                )
            for c in self.candidates:
                if c.matched_count > self.case_total:
                    raise ValueError(
                        f"候选 {c.context.value} 的 matched_count={c.matched_count} "
                        f"超过 case_total={self.case_total}"
                    )
        else:
            if self.similar_cases:
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 时不得携带 similar_cases"
                )
            if self.case_total:
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 时不得携带 case_total"
                )
            if any(c.matched_count for c in self.candidates):
                raise ValueError(
                    f"evidence_mode={self.evidence_mode.value} 时不得携带 matched_count"
                )
        return self

    @model_validator(mode="after")
    def _acoustic_mode_requires_features(self) -> BehaviorInterpretation:
        """除 TEXT_ONLY 外，都必须能拿出声学特征。"""
        if (
            self.evidence_mode in _WITH_FEATURES_MODES
            and self.acoustic_features is None
        ):
            raise ValueError(
                f"evidence_mode={self.evidence_mode.value} 时必须提供 acoustic_features"
            )
        return self

    @model_validator(mode="after")
    def _posteriors_normalized(self) -> BehaviorInterpretation:
        """后验必须归一化（开放集由 other 兜底）。"""
        values = [c.posterior for c in self.candidates if c.posterior is not None]
        if values:
            total = sum(values)
            if not 0.99 <= total <= 1.01:
                raise ValueError(
                    f"posterior 之和为 {total:.3f}，应归一化到 1.0（开放集用 other 兜底）"
                )
        return self

    @property
    def top_candidate(self) -> IntentCandidate | None:
        """后验最高的候选。

        显式构造 ``list[tuple[float, IntentCandidate]]`` 而非直接 ``max(key=...)`` ——
        因为 ``posterior`` 是 ``float | None``（``text_only`` 模式下必须为 ``None``），
        直接在 key 里取会让类型与运行时都面临 ``None`` 参与比较。
        这里先过滤、再确定地比较。
        """
        scored = [(c.posterior, c) for c in self.candidates if c.posterior is not None]
        if not scored:
            return None
        return max(scored, key=lambda pair: pair[0])[1]

    @property
    def confidence_tier(self) -> str:
        """置信度分档（docs/03 §7）。决定系统是给明确建议还是只给方向。

        | 档位 | 行为 |
        |---|---|
        | high (≥0.65) | 明确候选 + 主证据 + 建议动作 |
        | medium (0.45–0.65) | 只给方向性描述，不给具体建议动作 |
        | low (<0.45) | 明说无法判断，只输出建议观察项 |
        | none | text_only 或样本不足 |
        """
        top = self.top_candidate
        if top is None or top.posterior is None:
            return "none"
        if top.posterior >= 0.65:
            return "high"
        if top.posterior >= 0.45:
            return "medium"
        return "low"


class IndividualizationProfile(BaseModel):
    """个体化进度。分层贝叶斯收缩的状态（docs/03 §3.4）。

    P(f|k,c) = λ_c · P_ind(f|k,c) + (1−λ_c) · P_pop(f|k)
    λ_c = n_c / (n_c + κ)
    """

    pet_id: str
    sample_count: int
    kappa: float = 5.0
    """收缩常数。**当前为未经验证的经验值**，待敏感性分析确定（docs/06 §3.3）。"""

    @property
    def lambda_c(self) -> float:
        return self.sample_count / (self.sample_count + self.kappa)

    @property
    def is_cold_start(self) -> bool:
        return self.sample_count == 0
