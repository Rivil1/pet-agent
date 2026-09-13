"""输入与路由契约，含**意图策略与代价矩阵**。

设计全文见 docs/09-intent-and-planning.md。

两条核心设计：

1. **意图边界按下游行为差异划分**，不按语义主题。判据：
   两个意图若在下游的动作、约束、失败处理上完全一致，就应合并。
2. **不存在一条全局置信度阈值**。不同误判方向的代价差异极大，
   因此策略是「每个意图对一条阈值」，且部分意图改用「执行 + 明确反馈」范式。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.observation import MediaKind


class AudioKind(str, Enum):
    """音频类型。**两者走完全不同的链路，不可混用。**

    用户语音 → ASR 转写为文本；
    猫咪叫声 → 声学特征提取，**不做转写**（猫叫不是语音，见 docs/03 §5）。
    """

    USER_VOICE = "user_voice"
    CAT_MEOW = "cat_meow"


class InputIntent(str, Enum):
    """意图枚举。

    划分标准是**下游行为差异**，不是语义主题（docs/01 §5.2）。

    - ``CHAT`` 与 ``MEMORY_QUERY`` 的差异是**可写 vs 只读**：
      ``MEMORY_QUERY`` 是一次**审计**——用户要求系统复述它已知的。
      审计不得产生新记录（否则审计会污染被审计对象），因此它
      **禁止把本轮推断写回记忆**，而 ``CHAT`` 允许（进 PENDING）。
      这是**动作许可差异**，不是参数差异。
    - ``RECORD_EVENT`` 的响应必须报告写入结果（写没写成），故写入必须早于响应；
      其余意图的写入是副作用，可晚于响应。
    - ``COMPOUND`` 走规划层（慢路径），与其余所有单意图的快路径不同。

    注：原文档曾声称差异是「后者有强制拒答约束」，该说法不成立——
    两者都带 ``retrieval_note``（见 docs/10-self-review.md A1）。
    """

    CHAT = "chat"
    MEMORY_QUERY = "memory_query"
    TRANSLATE_BEHAVIOR = "translate_behavior"
    PROFILE_UPDATE = "profile_update"
    RECORD_EVENT = "record_event"
    AMBIGUOUS = "ambiguous"
    """置信度低于澄清阈值。**独立分支，不回退到 CHAT。**
    猜错意图会写入错误记忆、污染长期状态，代价高于多问一句。"""

    COMPOUND = "compound"
    """复合请求，需走 planner 分解为多个子任务（慢路径）。"""


#: 走快路径的单意图集合（固定链路，无需规划）
FAST_PATH_INTENTS: frozenset[InputIntent] = frozenset(
    {
        InputIntent.CHAT,
        InputIntent.MEMORY_QUERY,
        InputIntent.TRANSLATE_BEHAVIOR,
        InputIntent.PROFILE_UPDATE,
        InputIntent.RECORD_EVENT,
    }
)


class RawInput(BaseModel):
    """一次请求的原始输入。"""

    text: str | None = None
    image_urls: list[str] = Field(default_factory=list)
    audio_url: str | None = None
    audio_kind: AudioKind | None = None
    media_kind: MediaKind | None = Field(
        default=None,
        description=(
            "``audio_url`` 指向的媒体的实际类型，供**多模态提取器**使用。\n\n"
            "为什么不从扩展名猜：一个 ``.mp4`` 的音频轨与一个 ``.m4a`` 对声学提取是一样的，\n"
            "但对多模态模型是不同的请求。让调用方声明比猜准确。\n\n"
            "为 ``None`` 时视为 ``MediaKind.AUDIO``。"
        ),
    )
    scene_description: str | None = Field(
        default=None,
        description="用户补充的场景，如「它对着门叫」。行为解释的场景先验来源。",
    )

    @model_validator(mode="after")
    def _audio_kind_required_with_audio(self) -> RawInput:
        """有音频必须声明类型，否则无法确定走 ASR 还是声学特征提取。"""
        if self.audio_url and self.audio_kind is None:
            raise ValueError(
                "提供 audio_url 时必须指定 audio_kind。"
                "用户语音走 ASR，猫咪叫声走声学特征提取，链路不同。"
            )
        return self

    @model_validator(mode="after")
    def _not_empty(self) -> RawInput:
        if not any((self.text, self.image_urls, self.audio_url)):
            raise ValueError("至少需要 text / image_urls / audio_url 之一")
        return self


class RouteSlots(BaseModel):
    """路由槽位。意图决定走哪条链路，槽位决定链路的参数。"""

    has_audio: bool = False
    has_audio_kind: AudioKind | None = None
    has_image: bool = False
    pet_id_hint: str | None = Field(
        default=None, description="从文本解析出的宠物指代，如「团团」"
    )
    scene_description: str | None = None
    time_range: str | None = Field(
        default=None,
        description="时间范围。**冲突检测的必需字段**（docs/04 §3.3），"
        "如「上个月」→ [上月1日, 上月30日]；None 表示无界（当前有效）。",
    )
    sub_intents: list[InputIntent] = Field(
        default_factory=list,
        description="COMPOUND 时解析出的子意图，作为 planner 的输入",
    )


class SessionContext(BaseModel):
    """会话层上下文。用于指代消解与省略补全（docs/09 §5）。"""

    session_id: str
    current_pet_id: str | None = Field(
        default=None, description="多宠物场景下用于解析「它」"
    )
    last_intent: InputIntent | None = None
    last_interpretation_id: str | None = Field(
        default=None, description="用于解析「那它为什么这样」中的「这样」"
    )
    recent_emotions: list[str] = Field(default_factory=list)
    turn_count: int = 0


# ─────────────────────────────────────────────────────────────
# 意图策略与代价矩阵
# ─────────────────────────────────────────────────────────────


class MemoryWritePermission(str, Enum):
    """本轮是否允许把系统推断写回长期记忆。

    这是 ``CHAT`` 与 ``MEMORY_QUERY`` 的真实差异所在——
    「审计」动作不得产生新记录，否则审计会污染被审计对象。
    """

    ALLOWED = "allowed"
    """允许（推断进 PENDING_CONFIRMATION，永不直接 ACTIVE）"""

    FORBIDDEN = "forbidden"
    """禁止。用于只读/审计类意图。"""


class IntentPolicy(BaseModel):
    """单个意图的路由策略。**每个意图独立配置**，不存在全局阈值。"""

    intent: InputIntent
    min_confidence: float = Field(
        ge=0.0, le=1.0, description="低于此值 → AMBIGUOUS（澄清）"
    )
    prefer_recall: bool = Field(
        description="倾向判入（True）还是判出（False）。由误判代价的方向决定。"
    )
    require_confirmation: bool = Field(
        default=False,
        description=(
            "是否采用「执行 + 明确反馈」范式。"
            "用于**两个方向误判代价都高**的意图——此时靠阈值无法解决，"
            "必须改变交互设计让用户能以低成本纠错。"
        ),
    )
    memory_write: MemoryWritePermission = Field(
        default=MemoryWritePermission.ALLOWED,
        description="本轮是否允许把系统推断写回记忆。审计类意图必须设为 FORBIDDEN。",
    )
    retrieval_k: int = Field(
        default=5, ge=1, le=20, description="该意图下的向量召回数量"
    )
    rejection_strict: bool = Field(
        default=False,
        description="是否采用更严格的拒答阈值（检索质量稍差即说「没有记录」）",
    )
    response_reports_write: bool = Field(
        default=False,
        description=(
            "响应是否需报告写入结果。True 时写入必须早于响应"
            "（路径：extract → writer → guard → tts）；"
            "False 时写入作为副作用置于响应之后，不阻塞主链路（docs/01 §2.2）。"
        ),
    )
    rationale: str

    @model_validator(mode="after")
    def _audit_intents_are_read_only(self) -> IntentPolicy:
        """审计类意图不得写记忆，否则审计污染被审计对象。"""
        if (
            self.intent is InputIntent.MEMORY_QUERY
            and self.memory_write is not MemoryWritePermission.FORBIDDEN
        ):
            raise ValueError(
                "MEMORY_QUERY 是一次审计（用户要求系统复述已知信息），"
                "不得把本轮推断写回记忆。"
            )
        return self


class MisrouteCost(BaseModel):
    """一次误判的代价。用于「代价加权错误率」评测与阈值调参。"""

    predicted: InputIntent
    actual: InputIntent
    cost: float = Field(ge=0.0, le=1.0)
    is_silent: bool = Field(
        default=False, description="用户是否难以察觉（静默失败代价更高）"
    )
    rationale: str


#: 误判代价矩阵（docs/09 §4）。
#:
#: 注意 RECORD_EVENT 的两个方向代价都高：
#:   - 误判成它 → 闲聊被写成记忆（污染长期状态，且很久才可能被发现）
#:   - 被他类吞掉 → 用户以为记住了实际没记（静默失败）
#: 因此它不使用纯阈值，而使用 require_confirmation=True。
MISROUTE_COSTS: tuple[MisrouteCost, ...] = (
    MisrouteCost(
        predicted=InputIntent.CHAT,
        actual=InputIntent.RECORD_EVENT,
        cost=0.9,
        is_silent=True,
        rationale="闲聊内容被写成长期记忆，污染后续所有检索结果",
    ),
    MisrouteCost(
        predicted=InputIntent.RECORD_EVENT,
        actual=InputIntent.CHAT,
        cost=0.9,
        is_silent=True,
        rationale="用户以为已记住，实际未写入；可能很久以后或永远不发现",
    ),
    MisrouteCost(
        predicted=InputIntent.CHAT,
        actual=InputIntent.PROFILE_UPDATE,
        cost=0.7,
        rationale="照片未被处理，用户会立刻发现并重传",
    ),
    MisrouteCost(
        predicted=InputIntent.PROFILE_UPDATE,
        actual=InputIntent.CHAT,
        cost=0.4,
        rationale="误入档案更新流程，但需用户确认才写入，可被拦下",
    ),
    MisrouteCost(
        predicted=InputIntent.CHAT,
        actual=InputIntent.TRANSLATE_BEHAVIOR,
        cost=0.5,
        rationale="少一次行为解释，用户会重问",
    ),
    MisrouteCost(
        predicted=InputIntent.TRANSLATE_BEHAVIOR,
        actual=InputIntent.CHAT,
        cost=0.5,
        rationale="对无音频输入走解释链路，会输出无证据的推测，由守卫拦下",
    ),
    MisrouteCost(
        predicted=InputIntent.CHAT,
        actual=InputIntent.MEMORY_QUERY,
        cost=0.3,
        rationale="少一次严格拒答约束，但答案质量影响有限",
    ),
    MisrouteCost(
        predicted=InputIntent.MEMORY_QUERY,
        actual=InputIntent.CHAT,
        cost=0.3,
        rationale="闲聊误走严格拒答模式，偏保守，无害",
    ),
    MisrouteCost(
        predicted=InputIntent.AMBIGUOUS,
        actual=InputIntent.CHAT,
        cost=0.15,
        rationale="多问一句，代价最低",
    ),
    MisrouteCost(
        predicted=InputIntent.CHAT,
        actual=InputIntent.AMBIGUOUS,
        cost=0.6,
        is_silent=True,
        rationale="该问未问——强行猜测最可能的意图，可能写入错误记忆",
    ),
)


#: 意图路由策略（docs/09 §4）。
INTENT_POLICIES: tuple[IntentPolicy, ...] = (
    IntentPolicy(
        intent=InputIntent.RECORD_EVENT,
        min_confidence=0.35,
        prefer_recall=True,
        require_confirmation=True,
        response_reports_write=True,
        rationale=(
            "两个方向误判代价都高（0.9/0.9）：误判成它会污染长期记忆，"
            "被他类吞掉是静默失败。精确率与召回率都不能牺牲时靠阈值无解，"
            "故改用「执行 + 明确反馈」：直接写入并明确回显记录内容，"
            "假阳性由用户一句话纠正，假阴性由系统反馈消除。"
            "响应需报告写入结果，因此写入必须早于响应（docs/01 §2.2）。"
        ),
    ),
    IntentPolicy(
        intent=InputIntent.TRANSLATE_BEHAVIOR,
        min_confidence=0.5,
        prefer_recall=True,
        response_reports_write=False,
        rationale="漏判代价（少一次解释）高于误判代价（守卫可拦下无证据推测）",
    ),
    IntentPolicy(
        intent=InputIntent.MEMORY_QUERY,
        min_confidence=0.5,
        prefer_recall=False,
        memory_write=MemoryWritePermission.FORBIDDEN,
        retrieval_k=8,
        rejection_strict=True,
        response_reports_write=False,
        rationale=(
            "这是一次**审计**：用户要求系统复述它已知的信息。"
            "审计不得产生新记录，否则会污染被审计对象——"
            "因此禁止写回推断，提高召回数（k=8）并采用更严格的拒答阈值。"
            "差异在「动作许可 + 参数」，不是「有无拒答约束」。"
        ),
    ),
    IntentPolicy(
        intent=InputIntent.PROFILE_UPDATE,
        min_confidence=0.6,
        prefer_recall=False,
        rationale="有图片时意图明确；无图片误判会进入需确认流程，成本可控",
    ),
    IntentPolicy(
        intent=InputIntent.COMPOUND,
        min_confidence=0.55,
        prefer_recall=True,
        rationale=(
            "漏判复合意图会丢失部分子任务（用户不会察觉少了什么），"
            "误判则多走规划层，仅增加延迟。故倾向召回。"
        ),
    ),
    IntentPolicy(
        intent=InputIntent.CHAT,
        min_confidence=0.4,
        prefer_recall=False,
        response_reports_write=False,
        rationale="兜底意图，应最后考虑，避免吞掉其他意图导致静默失败",
    ),
)


def misroute_cost(predicted: InputIntent, actual: InputIntent) -> float:
    """查代价矩阵：系统判成了 ``predicted``，而「应该」是 ``actual``。

    ⚠️ **参数语义容易读反**（实测确实会读反），举例说明：

        misroute_cost(AMBIGUOUS, CHAT)   → 0.15  多问了一句，便宜
        misroute_cost(CHAT, AMBIGUOUS)   → 0.60  该问未问，贵且用户不察觉

    即：把 ``actual`` 读作「正确类别」，把 ``predicted`` 读作「实际输出」。
    """
    if predicted is actual:
        return 0.0
    for entry in MISROUTE_COSTS:
        if entry.predicted is predicted and entry.actual is actual:
            return entry.cost
    return 0.5


def policy_for(intent: InputIntent) -> IntentPolicy:
    for p in INTENT_POLICIES:
        if p.intent is intent:
            return p
    raise KeyError(f"未配置策略的意图：{intent}")
