"""记忆契约：三层记忆 + 数据飞轮 + 冲突消解 + 防自我强化。

设计全文见 docs/04-memory.md。

本模块把两条最重要的安全约束编码进了类型系统，而不是留给 prompt 或调用方自觉：

1. **防自我强化**（§3.5）：``SYSTEM_INFERENCE`` 不得以 ``ACTIVE`` 状态存在。
   模型输出不能作为下一轮的事实输入，否则幻觉会自我强化。
2. **冲突判定需要时间范围**（§3.3）：同主体 + 极性相反 + **时间范围重叠** 三者齐备才算冲突。
   缺第三条会把「正常演变」误判为「矛盾」。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────
# 枚举
# ─────────────────────────────────────────────────────────────


class MemoryLayer(str, Enum):
    """三层记忆。访问模式不同，不可混存、不可混用检索方式。"""

    PROFILE = "profile"
    """稳定事实：外貌、名字、品种、must_keep_features、角色设定。
    **访问方式：主键直读，不做向量检索。**
    详见 docs/04-memory.md §1。"""

    EPISODE = "episode"
    """事件 / 经验：'昨天它害怕吸尘器'、'用户今天加班'。
    **访问方式：混合检索（结构化预过滤 + 向量召回）。**"""

    SESSION = "session"
    """短期会话状态：当前对话、最近情绪、当前场景、最近一次解释结果。
    **访问方式：直读，不检索。** 会话结束即失效。"""


class MemoryAccessMode(str, Enum):
    """各层的正确访问模式。用类型表达，防止误用。"""

    PRIMARY_KEY = "primary_key"
    HYBRID_SEARCH = "hybrid_search"
    DIRECT_READ = "direct_read"


LAYER_ACCESS_MODE: dict[MemoryLayer, MemoryAccessMode] = {
    MemoryLayer.PROFILE: MemoryAccessMode.PRIMARY_KEY,
    MemoryLayer.EPISODE: MemoryAccessMode.HYBRID_SEARCH,
    MemoryLayer.SESSION: MemoryAccessMode.DIRECT_READ,
}


class EventType(str, Enum):
    BEHAVIOR = "behavior"
    PREFERENCE = "preference"
    ROUTINE = "routine"
    HEALTH = "health"
    CONTEXT = "context"


class Polarity(str, Enum):
    """极性。冲突检测的核心字段之一（§3.3 条件 ②）。"""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    PENDING_CONFIRMATION = "pending_confirmation"
    """待用户确认。**不参与默认检索。**"""

    SUPERSEDED = "superseded"
    """被更新的记忆取代。保留以支持历史查询与审计，不参与默认检索。"""

    REJECTED = "rejected"


class MemorySource(str, Enum):
    USER_OBSERVATION = "user_observation"
    """用户直接陈述。"""

    USER_CORRECTION = "user_correction"
    """用户纠正系统判断。可信度最高，且应触发冲突消解。"""

    USER_CONFIRMATION = "user_confirmation"
    """用户确认了系统此前的推断。由 PENDING 转 ACTIVE 时写入此来源。"""

    MEOW_LABEL = "meow_label"
    """用户对一次叫声解释的确认标签。同时作为行为解释器的训练样本。"""

    SYSTEM_INFERENCE = "system_inference"
    """系统推断。**永远不得为 ACTIVE**（见 MemoryEvent 校验器）。"""


#: 各来源的初始置信度（§3.4）
SOURCE_TRUST: dict[MemorySource, float] = {
    MemorySource.USER_CORRECTION: 0.95,
    MemorySource.USER_CONFIRMATION: 0.90,
    MemorySource.USER_OBSERVATION: 0.90,
    MemorySource.MEOW_LABEL: 0.90,
    MemorySource.SYSTEM_INFERENCE: 0.60,
}

#: ASR 转写后抽取信息的置信度折扣。
#: 语音识别错误会被当作事实存下来且事后无法察觉，因此对时间/数量类字段降权并要求确认。
ASR_CONFIDENCE_DISCOUNT = 0.85


#: 各事件类型的检索衰减半衰期（天）。None 表示不衰减。
#: 用统一衰减率是常见错误：会让系统忘记宠物"一直喜欢"的东西。
EVENT_HALFLIFE_DAYS: dict[EventType, float | None] = {
    EventType.PREFERENCE: None,
    EventType.ROUTINE: None,
    EventType.BEHAVIOR: 30.0,
    EventType.CONTEXT: 7.0,
    EventType.HEALTH: None,  # 由 HealthDataPolicy 的保留期决定
}


class RetrievalSource(str, Enum):
    VECTOR = "vector"
    RELATIONAL = "relational"
    HYBRID = "hybrid"


class WriteAction(str, Enum):
    """写入决策的动作。每个候选记忆必须有明确去向，不可默默丢弃。"""

    WRITE = "write"
    REINFORCE = "reinforce"
    """语义重复：不新增，而是 support_count += 1。"""

    PENDING = "pending"
    SUPERSEDE = "supersede"
    REJECT = "reject"


# ─────────────────────────────────────────────────────────────
# 记忆本体
# ─────────────────────────────────────────────────────────────


#: 主体无法从内容识别时的回退前缀。
#:
#: `_guess_subject`（`app/graph/nodes.py`）认不出内容时返回
#: `misc:<哈希>`。放在契约层是因为**两个不同层次都要用它**：
#:
#: - 编排层：生成回退值
#: - 习惯层：识别「这条的主体是猜的」，并**显式报告**（`app/habits/detect.py`）
#:
#: 让习惯层去 import 编排层的常量会造成分层倒置 ——
#: 领域逻辑不该依赖图编排。
SUBJECT_FALLBACK_PREFIX = "misc:"


class MemoryEvent(BaseModel):
    """一条长期记忆。"""

    memory_id: str | None = None
    user_id: str
    pet_id: str = Field(description="多租户隔离键。所有检索必须按此过滤。")
    session_id: str | None = Field(
        default=None,
        description=(
            "产生这条记忆的会话。**审计用**（见 `ARCHITECTURE.md` §2.2 audit_log）—— "
            "它能回答「这条记忆是哪一轮对话产生的」。\n\n"
            "它**不是隔离键**，也不进检索排序：隔离仍然只靠 user_id + pet_id。"
        ),
    )

    layer: MemoryLayer = MemoryLayer.EPISODE
    event_type: EventType

    subject: str = Field(
        description=(
            "冲突检测的主体标识（§3.3 条件 ①），如 'vacuum_cleaner'、'cat_wand'。"
            "同一 subject + 相反 polarity + 时间重叠 才算冲突。"
        )
    )
    content: str = Field(description="自然语言描述，如 '最近晚上经常在门口叫'")
    polarity: Polarity = Polarity.NEUTRAL

    # ── 时间语义：冲突检测的第三个条件 ──────────────────────────
    valid_from: datetime | None = Field(
        default=None, description="陈述所描述的时间范围起。None 表示无下界。"
    )
    valid_to: datetime | None = Field(
        default=None, description="陈述所描述的时间范围止。None 表示'至今有效'。"
    )
    occurred_at: datetime | None = Field(
        default=None, description="事件实际发生时间（与陈述时间不同）"
    )

    source: MemorySource
    confidence: float = Field(ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE

    # ── 强化（§3.2）──────────────────────────────────────────
    support_count: int = Field(
        default=1,
        ge=1,
        description="该模式被重复观察到的次数。≥3 且跨度 ≥14 天是晋升到 Profile 的条件。",
    )
    last_seen_at: datetime | None = None

    # ── 取代链（§3.3）────────────────────────────────────────
    superseded_by: str | None = None
    supersedes: list[str] = Field(default_factory=list)

    dedup_key: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # ── 结构性约束 ───────────────────────────────────────────

    @model_validator(mode="after")
    def _no_active_system_inference(self) -> MemoryEvent:
        """防自我强化（§3.5 R1）。

        模型输出不能作为下一轮的事实输入。若允许 SYSTEM_INFERENCE 处于 ACTIVE，
        幻觉会在多轮对话中被反复强化为"系统已知事实"。
        """
        if (
            self.source is MemorySource.SYSTEM_INFERENCE
            and self.status is MemoryStatus.ACTIVE
        ):
            raise ValueError(
                "SYSTEM_INFERENCE 不得以 ACTIVE 状态存在（防自我强化）。"
                "只能写入 PENDING_CONFIRMATION；用户确认后 source 改为 "
                "USER_CONFIRMATION 再转 ACTIVE。"
            )
        return self

    @model_validator(mode="after")
    def _time_range_ordering(self) -> MemoryEvent:
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("valid_from 不得晚于 valid_to")
        return self

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content 不得为空")
        return v

    # ── 行为 ─────────────────────────────────────────────────

    @property
    def halflife_days(self) -> float | None:
        """该记忆在检索重排序中的衰减半衰期。"""
        return EVENT_HALFLIFE_DAYS[self.event_type]

    @property
    def is_retrievable_by_default(self) -> bool:
        """默认检索是否包含本条（§3.5 R5）。"""
        return self.status is MemoryStatus.ACTIVE

    def conflicts_with(self, other: MemoryEvent) -> bool:
        """冲突判定（§3.3）。三个条件必须同时满足。

        注意：时间范围重叠是必要条件。缺了它会把「正常演变」误判为「矛盾」，
        例如「上个月喜欢逗猫棒」与「这周不喜欢逗猫棒」并不冲突。
        """
        if self.pet_id != other.pet_id:
            return False
        if self.subject != other.subject or self.event_type != other.event_type:
            return False
        if self.polarity is Polarity.NEUTRAL or other.polarity is Polarity.NEUTRAL:
            return False
        if self.polarity is other.polarity:
            return False
        return _ranges_overlap(
            self.valid_from, self.valid_to, other.valid_from, other.valid_to
        )

    def is_eligible_for_profile_promotion(
        self, *, min_support: int = 3, min_span_days: int = 14
    ) -> bool:
        """是否可晋升到 Profile 层（§2）。"""
        if self.layer is not MemoryLayer.EPISODE:
            return False
        if self.support_count < min_support:
            return False
        anchor = self.created_at
        latest = self.last_seen_at or self.created_at
        return (latest - anchor).days >= min_span_days


def _ranges_overlap(
    a_from: datetime | None,
    a_to: datetime | None,
    b_from: datetime | None,
    b_to: datetime | None,
) -> bool:
    """半开区间重叠判定。None 表示无界。"""
    # 无界视为 (-inf, +inf)，用极值代替以简化比较
    lo_a = a_from or datetime.min.replace(tzinfo=timezone.utc)
    hi_a = a_to or datetime.max.replace(tzinfo=timezone.utc)
    lo_b = b_from or datetime.min.replace(tzinfo=timezone.utc)
    hi_b = b_to or datetime.max.replace(tzinfo=timezone.utc)
    return lo_a <= hi_b and lo_b <= hi_a


# ─────────────────────────────────────────────────────────────
# 检索与写入
# ─────────────────────────────────────────────────────────────


class ScoreBreakdown(BaseModel):
    """重排序得分的分项（§4.3）。

    保留分项以便审计「为什么这条记忆被排到前面」。
    """

    semantic_similarity: float = 0.0
    recency: float = 0.0
    support: float = 0.0
    source_trust: float = 0.0
    type_match: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.semantic_similarity
            + self.recency
            + self.support
            + self.source_trust
            + self.type_match
        )


class MemoryItem(BaseModel):
    """检索结果包装。元信息必须保留——`response_guard` 依赖它判定断言是否有据。"""

    event: MemoryEvent
    score: float = Field(ge=0.0)
    breakdown: ScoreBreakdown | None = None
    retrieval_source: RetrievalSource = RetrievalSource.HYBRID
    matched_on: str | None = None

    @model_validator(mode="after")
    def _only_retrievable(self) -> MemoryItem:
        if not self.event.is_retrievable_by_default:
            raise ValueError(
                f"status={self.event.status.value} 的记忆不得进入默认检索结果（§3.5 R5）"
            )
        return self


class MemoryWriteDecision(BaseModel):
    """数据飞轮的写入决策。每个候选都要有明确去向，不可默默丢弃。"""

    event: MemoryEvent
    action: WriteAction
    reason: str
    duplicate_of: str | None = None
    supersedes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _action_consistency(self) -> MemoryWriteDecision:
        if self.action is WriteAction.SUPERSEDE and not self.supersedes:
            raise ValueError("action=supersede 时必须指明被取代的记忆")
        if self.action is WriteAction.REINFORCE and not self.duplicate_of:
            raise ValueError("action=reinforce 时必须指明被强化的记忆")
        return self


class ConversationTurn(BaseModel):
    """注入 prompt 用的**一轮对话视图**。

    为什么不直接用持久化实体 `SessionMessage`：

    1. **会成环** —— `app/schemas/digest.py` 已经导入本模块，
       而 `session.py` 导入 `digest.py`，所以本模块不能再导入 `session.py`。
    2. **职责不同** —— 注入块只需要「谁说了什么」，不需要租户键与 ID。
       与 `DigestMessage` 之于 `SessionMessage` 是同一个做法：
       流水线拿输入视图，不拿持久化形状。
    """

    role: Literal["user", "assistant"]
    content: str
    at: datetime | None = None


class ContextInjectionBlock(BaseModel):
    """注入 prompt 的记忆块（§5）。

    四个要点：区分已确认/待确认、携带来源与置信度、显式说明"检索不到就是没有"、
    **把「已验证记录」与「最近对话」分开**。
    """

    pet_identity: str
    known_facts: list[MemoryItem] = Field(default_factory=list)
    unconfirmed: list[MemoryEvent] = Field(
        default_factory=list,
        description="PENDING 记忆单独标注。模型不得把它们当作事实使用。",
    )
    recent_turns: list[ConversationTurn] = Field(
        default_factory=list,
        description=(
            "最近若干轮对话记录。**它不是事实来源** —— "
            "其中 assistant 的内容是系统当时的回复（可能含未验证推测）。\n\n"
            "单独分区的理由：把它与 `known_facts` 混为一栏，"
            "模型上一轮的推测就会被当成「已知事实」回灌，"
            "形成与不变量 I1 同构的自我强化（只是通道从写入换成了读入）。"
        ),
    )
    retrieval_note: str = Field(
        default=(
            "以上为检索到的记录。未在其中的信息即为没有记录，不要推测或使用常识补全。"
        )
    )

    def render(self) -> str:
        """渲染为注入 prompt 的文本。"""
        import json

        payload: dict[str, object] = {
            "pet": {"identity": self.pet_identity},
            "known_facts": [
                {
                    "content": item.event.content,
                    "source": item.event.source.value,
                    "confidence": round(item.event.confidence, 2),
                    "support_count": item.event.support_count,
                    "observed_at": (item.event.occurred_at or item.event.created_at)
                    .date()
                    .isoformat(),
                }
                for item in self.known_facts
            ],
            "unconfirmed": [
                {"content": e.content, "note": "系统推断，未经用户确认"}
                for e in self.unconfirmed
            ],
            "retrieval_note": self.retrieval_note,
        }
        if self.recent_turns:
            # 只在非空时出现：空历史不应改变注入块的形状（免得下游与已有测试被无谓扰动）
            payload["recent_turns"] = [
                {"role": t.role, "content": t.content} for t in self.recent_turns
            ]
            payload["recent_turns_note"] = (
                "以上是对话记录（含助手当时的回复）。"
                "assistant 的内容是**当时的推测或措辞**，未经核实 —— "
                "不得把它当作事实再次断言，也不得据此声称「之前确认过」。"
            )
        return json.dumps(payload, ensure_ascii=False, indent=2)
