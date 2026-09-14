"""编排状态。

对应 docs/DESIGN.md §2.4。

**关键设计**：``errors`` 与 ``node_trace`` 用 ``Annotated[..., operator.add]`` 归约，
使 planner fan-out 出的并行分支能正确合并写入（LangGraph reducer 语义）。

**当前 P0 范围**：只实现快路径。``plan`` / ``current_wave`` / ``subtask_results``
等规划层字段保留在状态里但**不被使用** ——
规划层是 P3（`DESIGN.md` §7.1：预期触发率 <10%，不影响产品）。
保留字段是为了让状态契约稳定，避免后续加规划层时改动所有节点签名。
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.schemas import (
    AcousticFeatures,
    BehaviorAction,
    BehaviorInterpretation,
    GuardResult,
    InputIntent,
    MemoryEvent,
    MemoryItem,
    NodeError,
    NodeTrace,
    PetProfile,
    PetProfileDraft,
    RawInput,
    RouteSlots,
    SessionContext,
    SessionMessage,
)


class StateError(RuntimeError):
    """状态缺少必需字段。**这是编程错误，不是用户错误**。"""


class AgentState(TypedDict, total=False):
    """一次请求的完整状态。

    ⚠️ **全部字段可选（``total=False``）是刻意的**，不是偷懒。

    原因：LangGraph 的节点返回的是**局部更新**——
    每个节点只写自己改动的字段，返回值里不含 ``user_id`` / ``pet_id``。
    若把它们标为必填，类型系统就会拒绝**每一个节点**的返回值。
    类型约束必须与框架的执行模型一致。

    代价是读取时类型系统不能保证字段存在。补上这个安全性的是下面的
    ``tenant_of()`` / ``require_input()`` 访问器：它们把 ``KeyError``
    变成一个说明白了的 ``StateError``。

    **哪些字段总是存在**：``initial_state()`` 保证 ``user_id`` / ``pet_id`` /
    ``raw_input`` 在入口就被填上，且节点不会删除它们。
    """

    # ── 身份与租户（由 initial_state 保证存在，读取请走 tenant_of）──
    user_id: str
    pet_id: str
    session_id: str
    trace_id: str

    # ── 原始输入 ──
    raw_input: RawInput
    transcribed_text: str | None

    # ── 路由 ──
    intent: InputIntent
    route_confidence: float
    route_slots: RouteSlots

    #: 交互模式（`app/companion`）。**它只决定说多少证据，不决定谁在说话。**
    #:
    #: 两轨的身份都是那只猫本人 —— 所以这里不需要「角色扮演标注」：
    #: 它不是扮演，是这个产品的声音。
    interaction_mode: str

    # ── 上下文 ──
    pet_profile: PetProfile | None
    retrieved_memories: list[MemoryItem]
    pending_memories: list[MemoryEvent]
    session_context: SessionContext
    recent_turns: list[SessionMessage]
    """最近若干轮对话，**时间正序**。

    它是 `DESIGN.md` §3.5 里 Session 层的落地：Profile / Episode 回答
    「它是一隻什么样的猫」，这一项回答「我们刚才聊到哪」。

    ⚠️ **它不是事实来源**：其中 ``assistant`` 的内容是系统当时的回复，
    可能含未经验证的推测 —— 注入时必须与 ``known_facts`` 分区
    （见 `ContextInjectionBlock.recent_turns`），否则会开一条自我强化新通道。
    """

    # ── 能力节点产物 ──
    profile_draft: PetProfileDraft | None
    acoustic_features: AcousticFeatures | None
    interpretation: BehaviorInterpretation | None
    draft_response: str | None

    pending_action_suggestions: list[BehaviorAction]
    """多模态模型观察到的动作，**作为标注表单的预填候选**。

    为什么值得单独一个字段，而不是让客户端去解析证据文本：
    主人的标注负担是**产品成败关键**（见 `docs/03` L1b）。
    录完自动填上「抓门」，主人只需要确认或改一下 ——
    这直接把一次标注从「想 + 选 + 写」降到「看一眼 + 点确认」。

    ⚠️ **它只是候选。** 主人提交时仍然自己选定；
    未经确认的模型输出不得成为 `MeowRecord.actions`。
    """

    # ── 守卫与输出 ──
    guard_result: GuardResult | None
    final_response: str | None
    guard_mode: str
    """``"full"``（默认）或 ``"echo_only"``。

    ``echo_only`` 用于**回显用户原话**的场景（如「已记录：…」）。
    此时跳过「禁用词」检查 —— 用户自己说的话不应被系统改掉，
    否则「记住，它没问题」会被降级成「我这边暂时没有相关记录」，用户无法理解。
    """

    # ── 副作用 ──
    candidate_memories: list[MemoryEvent]
    written_memory_ids: list[str]

    # ── 可观测性（并行分支安全合并）──
    errors: Annotated[list[NodeError], operator.add]
    node_trace: Annotated[list[NodeTrace], operator.add]


def initial_state(
    *,
    user_id: str,
    pet_id: str,
    raw_input: RawInput,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> AgentState:
    """构造初始状态。**这是唯一的入口** —— 它保证三个必需字段一定就位。"""
    import uuid

    return AgentState(
        user_id=user_id,
        pet_id=pet_id,
        session_id=session_id or str(uuid.uuid4()),
        trace_id=trace_id or str(uuid.uuid4()),
        raw_input=raw_input,
        # 缺省陪伴轨 —— **默认值的选择是刻意的**：误判为分析轨的代价
        # （冷场、说教）高于误判为情绪轨（少给一次证据，用户会追问）。
        # 真实判定在 `understand_input` 里，那里能看到文本与音频。
        interaction_mode="companion",
        retrieved_memories=[],
        pending_memories=[],
        candidate_memories=[],
        written_memory_ids=[],
        errors=[],
        node_trace=[],
    )


# ─────────────────────────────────────────────────────────────
# 显式访问器
# ─────────────────────────────────────────────────────────────
#
# 为什么需要它们：state 是 total=False（见上面的说明），所以直接下标访问
# 在类型上不安全。但直接把下标换成 .get() 又会丢掉「必须存在」这个语义。
# 访问器保留该语义，同时把 KeyError 变成有名字的错误。


def tenant_of(state: AgentState) -> tuple[str, str]:
    """取 ``(user_id, pet_id)``。

    任何存储操作都必须带这两个字段（`ARCHITECTURE.md` §2.6 T3）。

    用 ``.get()`` 而非下标：既让类型检查放心，也顺带拦住了**空字符串**——
    空租户标识比缺失更危险，因为它会静默地匹配不到任何数据。

    Raises:
        StateError: 缺少租户标识 —— 说明调用方绕过了 ``initial_state()``。
    """
    user_id = state.get("user_id")
    pet_id = state.get("pet_id")
    if not user_id or not pet_id:
        raise StateError(
            "AgentState 缺少租户标识（user_id / pet_id）。"
            "请用 initial_state() 构造状态，它保证这两个字段就位。"
        )
    return user_id, pet_id


def require_input(state: AgentState) -> RawInput:
    """取原始输入。

    Raises:
        StateError: 缺少 ``raw_input``。
    """
    raw = state.get("raw_input")
    if raw is None:
        raise StateError("AgentState 缺少 raw_input")
    return raw
