"""编排节点。

对应 docs/DESIGN.md §2.3 的节点契约。

**P0 只实现快路径**。每个节点遵守两件事：

1. **不持有跨请求状态** —— 一切通过 ``AgentState`` 传递
2. **失败必留痕** —— 写入 ``errors``，且区分「可见」与「静默」降级（§2.5）

⚠️ **P0 的诚实边界**：
- ``understand_input`` 是**规则实现**，不是模型分类。因此它的 confidence 是
  「规则命中强度」，**未经校准**（`DESIGN.md` §7.3 U1）。阈值语义与校准后的版本不同。
- ``response_guard`` 只跑**规则层**，LLM 补漏层未实现（`ARCHITECTURE.md` §5.1）。
- 记忆**抽取**是规则实现；被动抽取（从闲聊里发现值得记的事）未实现。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from app.graph.normalize import parse_time_range
from app.graph.state import (
    AgentState,
    StateError,
    require_input,
    tenant_of,
)
from app.interpreter import PriorTable, interpret_meow
from app.llm import LLMClient
from app.memory import (
    MemoryWriter,
    build_context_block,
    pending_memories,
    retrieve,
    retrieve_with_status,
)
from app.profile import VisionAnalyzer, identify_from_photos
from app.schemas import (
    FORBIDDEN_PHRASES,
    AudioKind,
    ConversationTurn,
    ErrorKind,
    ErrorSeverity,
    EventType,
    EvidenceMode,
    GuardResult,
    InputIntent,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    NodeError,
    NodeTrace,
    Polarity,
    RawInput,
    RouteSlots,
    SessionContext,
    SessionMessage,
    Severity,
    Violation,
    ViolationType,
    policy_for,
)
from app.store.base import MemoryStore, NotFound

# ─────────────────────────────────────────────────────────────
# 规则表
# ─────────────────────────────────────────────────────────────

#: 意图关键词规则。**顺序即优先级** —— 先匹配者胜。
#: 这是 P0 的占位实现；生产应由模型分类 + 置信度校准（U1）。
_INTENT_RULES: tuple[tuple[InputIntent, float, tuple[str, ...]], ...] = (
    # 记录类优先：它的漏判代价最高（静默丢失，0.9）
    (
        InputIntent.RECORD_EVENT,
        0.80,
        ("记住", "记一下", "帮我记", "别忘了记", "记录一下"),
    ),
    (
        InputIntent.PROFILE_UPDATE,
        0.75,
        ("这是它的照片", "更新档案", "传照片", "建档案"),
    ),
    (
        InputIntent.TRANSLATE_BEHAVIOR,
        0.70,
        ("为什么叫", "一直叫", "叫声", "它叫", "为什么这样"),
    ),
    (
        InputIntent.MEMORY_QUERY,
        0.70,
        ("还记得", "它喜欢什么", "它怕什么", "它讨厌什么", "以前"),
    ),
    (InputIntent.CHAT, 0.45, ("怎么样", "在吗", "想你", "今天", "你好")),
)

#: 相关事件类型：按意图把检索范围收窄（对应 retrieval 的 ``wanted_types``）。
_INTENT_EVENT_TYPES: dict[InputIntent, frozenset[EventType]] = {
    InputIntent.MEMORY_QUERY: frozenset(
        {EventType.PREFERENCE, EventType.ROUTINE, EventType.BEHAVIOR}
    ),
    InputIntent.CHAT: frozenset({EventType.PREFERENCE, EventType.ROUTINE}),
    InputIntent.TRANSLATE_BEHAVIOR: frozenset({EventType.BEHAVIOR, EventType.ROUTINE}),
}

#: 场景描述关键词。命中即作为 ``scene_description`` 传给解释器。
_SCENE_HINTS: tuple[str, ...] = (
    "对着门",
    "在门口",
    "门外",
    "食盆",
    "要吃的",
    "饿",
    "摸它",
    "抱它",
    "梳毛",
    "陌生",
    "独处",
    "回家",
    "进门",
)

# ─────────────────────────────────────────────────────────────
# 会话记忆注入的预算
# ─────────────────────────────────────────────────────────────

#: 注入多少**条消息**。一轮 ≈ user + assistant 两条，故 20 条 ≈ 最近 10 轮。
RECENT_MESSAGE_LIMIT = 20

#: 注入历史的**字符预算**。超出时从最旧的开始丢。
#:
#: 两条限制拦的是不同的失控方式：轮数防「很久没聊后一次灌入全部历史」，
#: 字符防「一轮就写了 5000 字」。只留一条都盖不住。
RECENT_TURNS_CHARS_BUDGET = 1500


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _trace(
    node: str, *, started: datetime, decision: str | None = None, **kw
) -> NodeTrace:
    return NodeTrace(
        node=node,
        # 用 `round()` 而非 `int()`：无需显式数值转换，
        # 且无参 `round` 已经返回 int（延迟毫秒不需要截断语义）
        latency_ms=round((_now() - started).total_seconds() * 1000),
        decision=decision,
        **kw,
    )


# ─────────────────────────────────────────────────────────────
# 1. understand_input
# ─────────────────────────────────────────────────────────────


def understand_input(state: AgentState) -> AgentState:
    """意图路由 + 槽位抽取 + **时间归一化**。

    置信度低于该意图的 ``min_confidence`` → ``AMBIGUOUS``（澄清），**不猜测**。
    理由（`DESIGN.md` §5.3）：猜错意图会写入错误记忆、污染长期状态，
    代价高于多问一句。
    """
    started = _now()
    raw: RawInput = require_input(state)
    text = state.get("transcribed_text") or raw.text or ""

    # ── 槽位 ──
    scene = raw.scene_description or next((h for h in _SCENE_HINTS if h in text), None)
    time_range = parse_time_range(text)
    slots = RouteSlots(
        has_audio=bool(raw.audio_url),
        has_audio_kind=raw.audio_kind,
        has_image=bool(raw.image_urls),
        scene_description=scene,
        time_range=(
            time_range.start.isoformat() if time_range and time_range.start else None
        ),
    )

    # ── 规则路由 ──
    intent, confidence = _classify(raw, text, slots)
    policy = policy_for(intent) if intent is not InputIntent.AMBIGUOUS else None

    if policy is not None and confidence < policy.min_confidence:
        routed = InputIntent.AMBIGUOUS
        decision = f"intent={intent.value} conf={confidence:.2f} < {policy.min_confidence} → ambiguous"
    else:
        routed = intent
        decision = f"intent={routed.value} conf={confidence:.2f}"

    route_slots = slots.model_copy(update={"time_range": slots.time_range})
    return AgentState(
        intent=routed,
        route_confidence=confidence,
        route_slots=route_slots,
        node_trace=[_trace("understand_input", started=started, decision=decision)],
    )


def _classify(raw: RawInput, text: str, slots: RouteSlots) -> tuple[InputIntent, float]:
    """规则分类。

    **含猫叫音频 → 直接判为行为解释**（强信号，不需要文本佐证）。
    """
    if raw.audio_url and raw.audio_kind is AudioKind.CAT_MEOW:
        return InputIntent.TRANSLATE_BEHAVIOR, 0.95

    if raw.image_urls and not text:
        return InputIntent.PROFILE_UPDATE, 0.80

    for intent, conf, keywords in _INTENT_RULES:
        if any(kw in text for kw in keywords):
            return intent, conf

    if not text.strip():
        # 完全无法判断 → 低置信度，交由阈值转 AMBIGUOUS
        return InputIntent.CHAT, 0.20

    return InputIntent.CHAT, 0.35


def route_after_understanding(state: AgentState) -> str:
    """条件边：返回下一个节点名。"""
    intent = state.get("intent", InputIntent.AMBIGUOUS)
    if intent is InputIntent.AMBIGUOUS:
        return "clarify_ask"
    return {
        InputIntent.CHAT: "memory_retriever",
        InputIntent.MEMORY_QUERY: "memory_retriever",
        InputIntent.TRANSLATE_BEHAVIOR: "behavior_interpreter",
        InputIntent.PROFILE_UPDATE: "profile_analyzer",
        InputIntent.RECORD_EVENT: "memory_extractor",
    }.get(intent, "clarify_ask")


# ─────────────────────────────────────────────────────────────
# 2. 检索与澄清
# ─────────────────────────────────────────────────────────────


def make_context_loader(store: MemoryStore):
    """构造**入口上下文加载节点**。

    它承担 `DESIGN.md` §3.5 里「档案层主键直读」这一步。

    为什么必须是独立节点、且放在入口：档案是所有下游节点都需要的上下文
    （陪伴对话要用它的身份约束与硬特征，行为解释要用它的个体模型）。
    放在某一条分支里（如只在检索节点里加载）会让其他分支拿不到档案。

    **早期实现漏掉了这一步**，后果是 `companion_agent` 永远看到 ``pet=None``，
    直接返回兜底文案 —— 而 LLM 从未被调用。这种缺陷很隐蔽：
    系统不报错、不崩溃，只是所有回复都变成同一句「先上传照片」。
    """

    def load_context(state: AgentState) -> AgentState:
        started = _now()
        try:
            user_id, pet_id = tenant_of(state)
        except StateError as exc:
            return AgentState(
                pet_profile=None,
                errors=[
                    NodeError(
                        node="load_context",
                        kind=ErrorKind.DATA_MISSING,
                        severity=ErrorSeverity.FATAL,
                        message=str(exc),
                    )
                ],
                node_trace=[_trace("load_context", started=started, degraded=True)],
            )

        try:
            pet = store.get_pet(user_id=user_id, pet_id=pet_id)
        except NotFound:
            # 档案不存在不是错误 —— 新用户还没建档，下游会引导上传照片
            pet = None

        session_id = state.get("session_id")
        recent, dropped = _recent_turns(
            store, user_id=user_id, pet_id=pet_id, session_id=session_id
        )
        profile_note = (
            f"档案已加载（{len(pet.must_keep_features)} 项硬特征）" if pet else "无档案"
        )
        if session_id:
            history_note = f"，历史 {len(recent)} 条"
            if dropped:
                history_note += f"（截断 {dropped} 条）"
        else:
            history_note = "，无会话标识"

        return AgentState(
            pet_profile=pet,
            session_context=SessionContext(
                session_id=session_id or "",
                current_pet_id=pet_id,
                turn_count=len(recent),
            ),
            recent_turns=recent,
            node_trace=[
                _trace(
                    "load_context",
                    started=started,
                    decision=profile_note + history_note,
                )
            ],
        )

    return load_context


def _recent_turns(
    store: MemoryStore, *, user_id: str, pet_id: str, session_id: str | None
) -> tuple[list[SessionMessage], int]:
    """取最近对话并施加字符预算。

    Returns:
        ``(turns, dropped)``。**截断必须留痕**：静默截断会让「模型没提过这件事」
        与「历史被丢掉了」在用户看来完全一样（同 B23 的丢弃要留痕）。

    从**最新**往回装：只有超预算的更旧消息被丢，最新一轮一定进得去。
    """
    if not session_id:
        # 没有会话标识就不猜 —— 宁可没有历史，也不能把别的会话串进来
        return [], 0

    rows = store.list_recent_messages(
        user_id=user_id,
        pet_id=pet_id,
        session_id=session_id,
        limit=RECENT_MESSAGE_LIMIT,
    )
    kept: list[SessionMessage] = []
    used = 0
    for message in reversed(rows):
        if kept and used + len(message.content) > RECENT_TURNS_CHARS_BUDGET:
            break
        kept.append(message)
        used += len(message.content)
    kept.reverse()
    return kept, len(rows) - len(kept)


def make_memory_retriever(
    store: MemoryStore,
    embedder,
    *,
    use_memory: bool = True,
    retrieve_fn: Callable[..., Any] | None = None,
):
    """构造检索节点。依赖注入，便于单测与**消融**替换。

    Args:
        use_memory: 为 ``False`` 时**完全不检索** —— 即「只有当前轮」的基线。
            它不是一个假实现，而是一个真实存在的系统形态：
            没有记忆层的对话助手就是这么工作的。
        retrieve_fn: 检索实现。默认 `retrieve_with_status`（混合检索）。
            消融时传入朴素向量 top-k，**同一张图、只换这一处** ——
            这正是 `DESIGN §6.3` 要求的「配置开关而非两份代码」。
    """
    retrieve_impl = retrieve_fn or retrieve_with_status

    def memory_retriever(state: AgentState) -> AgentState:
        started = _now()
        user_id, pet_id = tenant_of(state)
        query = state.get("transcribed_text") or require_input(state).text or ""
        intent = state.get("intent", InputIntent.CHAT)
        k = policy_for(intent).retrieval_k if intent in InputIntent else 5

        if not use_memory:
            # 无记忆基线：仍然标注出来，否则「召回了 0 条」看起来像检索失败。
            return AgentState(
                retrieved_memories=[],
                pending_memories=[],
                node_trace=[
                    _trace(
                        "memory_retriever",
                        started=started,
                        decision="已禁用记忆检索（消融基线）",
                    )
                ],
            )

        result = retrieve_impl(
            store=store,
            embedder=embedder,
            user_id=user_id,
            pet_id=pet_id,
            query=query,
            k=k,
            wanted_types=_INTENT_EVENT_TYPES.get(intent),
        )
        items = result.items
        pending = pending_memories(store, user_id=user_id, pet_id=pet_id)

        if result.degraded:
            # **降级要显式留痕，不能表现为「召回了 0 条」。**
            #
            # 向量索引不可用时（例如只配了 MySQL、没配 Milvus），
            # 「查不到」与「没有」在用户看来完全一样 ——
            # 而系统会据此自信地说出「没有相关记录」。
            # 写进 trace 并置 degraded，用户才知道这一次是没查成。
            return AgentState(
                retrieved_memories=items,
                pending_memories=pending,
                node_trace=[
                    _trace(
                        "memory_retriever",
                        started=started,
                        decision=f"检索降级：{result.degraded_reason}",
                        degraded=True,
                    )
                ],
            )

        return AgentState(
            retrieved_memories=items,
            pending_memories=pending,
            node_trace=[
                _trace(
                    "memory_retriever",
                    started=started,
                    decision=f"召回 {len(items)} 条，待确认 {len(pending)} 条",
                )
            ],
        )

    return memory_retriever


def clarify_ask(state: AgentState) -> AgentState:
    """澄清分支。**独立分支，不回退到闲聊**（`DESIGN.md` §5.1）。

    写 ``draft_response`` 而非 ``final_response`` —— 所有面向用户的文本
    都必须过守卫（`DESIGN.md` §2.2 W3）。
    """
    started = _now()
    conf = state.get("route_confidence", 0.0)
    return AgentState(
        draft_response=(
            "我不太确定你想问什么。你是想问它最近怎么样、想记住一件事，"
            "还是想让我看看它的叫声？"
        ),
        node_trace=[
            _trace(
                "clarify_ask", started=started, decision=f"路由不确定 conf={conf:.2f}"
            )
        ],
    )


# ─────────────────────────────────────────────────────────────
# 3. companion_agent
# ─────────────────────────────────────────────────────────────

#: 系统提示。**三层信息分离规则必须显式注入** —— 这是 prompt 层面的防线，
#: 硬约束由 ``response_guard`` 提供（prompt 是建议，节点是强制）。
_SYSTEM_PROMPT = """你是用户宠物猫的陪伴助手。

规则：
1. 只使用「已知事实」中的信息回答。没有记录的事就说没有记录，不要推测或用常识补全。
2. 不要声称超出证据的确定性。推测必须标明是推测。
3. 「待确认」列表里的是系统推断，**不得当作事实使用**。
4. 不要给兽医诊断、疾病名称或用药建议。
5. 回复简短、自然、有温度。"""


def make_companion_agent(llm: LLMClient):
    """构造陪伴对话节点。"""

    def companion_agent(state: AgentState) -> AgentState:
        started = _now()
        pet = state.get("pet_profile")
        if pet is None:
            return AgentState(
                draft_response="我还没有这只猫的档案，先上传几张照片让我认识它吧。",
                node_trace=[
                    _trace("companion_agent", started=started, decision="无档案")
                ],
            )

        # 会话历史先转成**展示视图**再注入：注入块不应依赖持久化实体
        recent_turns = [
            ConversationTurn(role=m.role, content=m.content, at=m.at)
            for m in state.get("recent_turns", [])
        ]
        block = build_context_block(
            pet=pet,
            items=state.get("retrieved_memories", []),
            pending=state.get("pending_memories", []),
            recent_turns=recent_turns,
        )
        user_text = state.get("transcribed_text") or require_input(state).text or ""
        system = f"{_SYSTEM_PROMPT}\n\n{block.render()}"

        try:
            draft = llm.complete(system=system, user=user_text)
        except Exception as exc:  # noqa: BLE001 — 降级而非崩溃
            return AgentState(
                draft_response="抱歉，我现在有点忙不过来，稍后再试试？",
                errors=[
                    NodeError(
                        node="companion_agent",
                        kind=ErrorKind.UPSTREAM_ERROR,
                        severity=ErrorSeverity.DEGRADED,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                ],
                node_trace=[_trace("companion_agent", started=started, degraded=True)],
            )

        return AgentState(
            draft_response=draft,
            node_trace=[
                _trace(
                    "companion_agent",
                    started=started,
                    decision=(
                        f"召回 {len(state.get('retrieved_memories', []))} 条记忆"
                        f" + {len(recent_turns)} 条会话历史注入"
                    ),
                )
            ],
        )

    return companion_agent


# ─────────────────────────────────────────────────────────────
# 4. response_guard（规则层）
# ─────────────────────────────────────────────────────────────


def response_guard(state: AgentState) -> AgentState:
    """守卫的**规则层**（`ARCHITECTURE.md` §5.1）。默认实例 —— 生产走这个。

    只跑确定性检查：快（<10ms）、必跑。LLM 补漏层是 P1。

    当前检查项：
    1. 空/退化响应
    2. 健康禁词（`DESIGN.md` §5.4 禁止清单）
    3. 检索为空时的事实性断言（幻觉的主要来源）

    消融请用 `make_response_guard(apply=False)`。
    """
    return _guarded(state, apply=True)


def make_response_guard(*, apply: bool = True):
    """构造守卫节点。**消融用**。

    `apply=False` 时**不拦截，只照原样输出草案** —— 于是评测能看见
    「如果没有守卫，用户会收到什么」。

    ## 为什么要有这个缝隙，而不是「把守卫代码注释掉」

    注释掉代码就再也无法验证同一套代码在有/无守卫下分别是多少 ——
    而那正是「守卫有价值」这个结论的唯一证据。
    用开关跑两次，差异被限定在这一处，消融结论才可信（`DESIGN §6.3`）。

    ⚠️ **这个开关只服务于评测，绝不能接进生产配置。**
    生产的守卫是无条件跑的：它拦的是健康禁词与无据断言，
    前者涉及合规（D9），后者直接决定用户会不会被误导。
    所以这里没有读任何环境变量 —— 它只能由代码显式传入。
    """

    def node(state: AgentState) -> AgentState:
        return _guarded(state, apply=apply)

    return node


def _guarded(state: AgentState, *, apply: bool) -> AgentState:
    """守卫的真实实现。`apply=False` 时跳过拦截，直接放行草案。"""
    started = _now()
    draft = (state.get("draft_response") or "").strip()
    violations: list[Violation] = []

    if not draft:
        violations.append(
            Violation(
                type=ViolationType.OVERCERTAINTY,
                severity=Severity.CRITICAL,
                detail="响应为空",
                expected="至少给出一个可读的回复或明确的说明",
            )
        )

    for phrase in FORBIDDEN_PHRASES:
        if phrase in draft:
            # echo_only：这是**回显用户原话**，不得因用户用词而降级系统回复。
            # 否则「记住，它没问题」会被改成「我这边暂时没有相关记录」，用户无法理解。
            if state.get("guard_mode") == "echo_only":
                continue
            violations.append(
                Violation(
                    type=ViolationType.HEALTH_BOUNDARY,
                    severity=Severity.CRITICAL,
                    detail=f"出现排除性表述「{phrase}」",
                    span=phrase,
                    expected="不得声明健康或排除疾病；说明「未发现异常不等于健康」",
                )
            )

    # 检索为空时不得出现「有过记录」类断言 —— 这是幻觉最直接的来源。
    #
    # 与「会话历史注入」同步：可依据的来源现在有两个，且**强度不同** ——
    #   · 长期记忆类（「根据记录」「档案里写着」）**必须**有检索到的记忆；
    #   · 对话历史类（「你之前说过」）可由**主人自己说过的话**满足，
    #     但**不能**由 assistant 的历史回复满足 —— 那是系统自己的输出，
    #     拿它当依据正是自我强化的读入通道（与写入通道 I1 同构）。
    has_memories = bool(state.get("retrieved_memories"))
    owner_said_something = any(m.role == "user" for m in state.get("recent_turns", []))
    cue_groups: tuple[tuple[tuple[str, ...], bool], ...] = (
        (("根据记录", "档案里写着"), has_memories),
        (("你之前说过", "我记得它"), has_memories or owner_said_something),
    )
    for cues, grounded in cue_groups:
        if grounded:
            continue
        for cue in cues:
            if cue in draft:
                violations.append(
                    Violation(
                        type=ViolationType.UNTRACEABLE_CLAIM,
                        severity=Severity.CRITICAL,
                        detail=f"无据断言「{cue}」：既无检索到的记忆，主人也没这样说过",
                        span=cue,
                        expected="没有依据时应明确说「没有相关记录」",
                    )
                )

    critical = [
        v for v in violations if v.severity in (Severity.CRITICAL, Severity.MAJOR)
    ]

    if not apply:
        # 消融：不拦截。`guard_result` 仍然给出（它记录了**本该被拦下什么**），
        # 但 `final_response` 用未被改写的草案 ——
        # 这样评测能量到「违规会不会真的到达用户」。
        return AgentState(
            guard_result=GuardResult(
                passed=not critical,
                violations=violations,
                rewrite_attempts=0,
                degrade_to_conservative=False,
            ),
            final_response=draft,
            node_trace=[
                _trace(
                    "response_guard",
                    started=started,
                    decision=f"守卫已禁用（消融）：{len(critical)} 条本应拦下",
                    degraded=False,
                )
            ],
        )

    if critical:
        final = (
            "我这边暂时没有相关记录，所以不方便判断。"
            "你可以先描述一下当时的情况，我帮你记下来，之后就能参考了。"
        )
        guard = GuardResult(
            passed=False,
            violations=violations,
            rewritten=final,
            rewrite_attempts=1,
            degrade_to_conservative=True,
            degraded_notice="原回复存在无依据或越界表述，已降级为保守回答",
        )
    else:
        final = draft
        guard = GuardResult(passed=True, violations=violations)

    return AgentState(
        guard_result=guard,
        final_response=final,
        node_trace=[
            _trace(
                "response_guard",
                started=started,
                decision=guard.summary(),
                degraded=guard.degrade_to_conservative,
            )
        ],
    )


# ─────────────────────────────────────────────────────────────
# 5. 行为解释
# ─────────────────────────────────────────────────────────────


def make_behavior_interpreter(
    prior: PriorTable,
    feature_extractor,
    similar_lookup=None,
    records_lookup=None,
    media_extractor=None,
):
    """构造行为解释节点。

    ``feature_extractor`` / ``similar_lookup`` / ``records_lookup`` /
    ``media_extractor`` 注入，便于测试替换（真实实现要下载音频、跑 librosa、
    查向量库、调多模态模型）。

    ``records_lookup`` 返回该猫**已被主人标注**的叫声记录（`MeowRecord`）——
    案例推理的样本库。**它决定走到 `case_based` 还是 `measured_only`**
    （见 `app/interpreter/router.py`）。

    ``media_extractor`` 产出 **`OBSERVED` 证据**（模型对画面/音频的观察描述）。
    三条不变量：

    1. **它不参与模式选择。** 观察在解释产出**之后**追加，
       不进入 `interpret_meow` 的任何输入。
    2. **它不改变候选排序。** 它不是概率模型的输入。
    3. **提取失败必须可见。** 失败进 trace 与 `NodeError`，不静默丢弃。

    为什么用 `interpret_meow` 而不是直接 `interpret`：
    后者需要一个**非占位**的先验，否则会算出以编造数值为基础的后验。
    路由层把这个判断集中在一处，调用方不必自己办。
    """

    def behavior_interpreter(state: AgentState) -> AgentState:
        started = _now()
        raw: RawInput = require_input(state)

        # ── 先取模型观察（如果配了提取器） ──
        #
        # 放在分支之前：**声学提取失败时它仍然有用** ——
        # 原本那种情况只返回一个错误，用户什么都看不到，
        # 而模型可能已经从画面里看到了「猫面向门、前爪抬起」。
        observation, obs_error = _extract_observation(media_extractor, raw)
        suggested = list(observation.actions) if observation and observation.ok else []

        if not raw.audio_url or raw.audio_kind is not AudioKind.CAT_MEOW:
            # 无猫叫音频 → 退化为纯文本推断，**且必须标注**
            from app.schemas import (
                BehaviorInterpretation,
                ContextLabel,
                IntentCandidate,
            )

            interp = BehaviorInterpretation(
                evidence_mode=EvidenceMode.TEXT_ONLY,
                candidates=[
                    IntentCandidate(context=ContextLabel.OTHER, display="需要更多信息")
                ],
                evidence=_observed_evidence(observation),
                suggested_observation="录一段叫声，或在描述里补充它当时的动作与场景",
                limitations=(
                    "本次没有可分析的叫声音频，因此**不给出数值置信度**。"
                    "请上传一段猫叫录音以获得基于声学证据的判断。"
                ),
            )
            return AgentState(
                interpretation=interp,
                pending_action_suggestions=suggested,
                errors=[] if obs_error is None else [obs_error],
                node_trace=[
                    _trace(
                        "behavior_interpreter",
                        started=started,
                        decision=(
                            "无音频 → text_only（禁止数值置信度）"
                            + _obs_note(
                                observation, configured=media_extractor is not None
                            )
                        ),
                        degraded=True,
                    )
                ],
            )

        try:
            features = feature_extractor(raw.audio_url)
        except Exception as exc:  # noqa: BLE001 — 降级，且必须可见
            # **不再直接返回错误。** 声学特征拿不到，但模型观察可能拿得到 ——
            # 只给一个错误等于把已有的信息丢掉。
            from app.schemas import (
                BehaviorInterpretation,
                ContextLabel,
                IntentCandidate,
            )

            interp = BehaviorInterpretation(
                evidence_mode=EvidenceMode.TEXT_ONLY,
                candidates=[
                    IntentCandidate(context=ContextLabel.OTHER, display="需要更多信息")
                ],
                evidence=_observed_evidence(observation),
                suggested_observation=(
                    "这次没能分析叫声音频（格式或质量原因）。"
                    "可以描述一下它当时的动作，或换一段录音再试。"
                ),
                limitations=(
                    "**声学特征提取失败**，因此本次不给出任何声学测量值。"
                    "若上方有画面观察，那是模型的描述，不是测量。"
                ),
            )
            return AgentState(
                interpretation=interp,
                pending_action_suggestions=suggested,
                errors=[
                    NodeError(
                        node="behavior_interpreter",
                        kind=ErrorKind.DATA_MISSING,
                        severity=ErrorSeverity.DEGRADED,
                        message=f"声学特征提取失败：{type(exc).__name__}: {exc}",
                    ),
                    *([] if obs_error is None else [obs_error]),
                ],
                node_trace=[
                    _trace(
                        "behavior_interpreter",
                        started=started,
                        decision=(
                            "声学提取失败 → text_only"
                            + _obs_note(
                                observation, configured=media_extractor is not None
                            )
                        ),
                        degraded=True,
                    )
                ],
            )

        samples = similar_lookup(state) if similar_lookup else []
        records = records_lookup(state) if records_lookup else []
        interp, decision = interpret_meow(
            features=features,
            records=records,
            prior=prior,
            individual=None,  # P0：个体参数化模型未接通
            scene=state.get("route_slots", RouteSlots()).scene_description,
            similar_samples=samples,
        )

        # ── 追加模型观察 ──
        # **在解释产出之后** —— 它不参与模式选择，也不改变候选排序。
        interp = _append_observed(interp, observation)

        return AgentState(
            acoustic_features=features,
            interpretation=interp,
            pending_action_suggestions=suggested,
            errors=[] if obs_error is None else [obs_error],
            node_trace=[
                _trace(
                    "behavior_interpreter",
                    started=started,
                    decision=(
                        f"evidence_mode={interp.evidence_mode.value} "
                        f"cases={decision.matched_records} "
                        f"tier={interp.confidence_tier} "
                        # 把「为什么是这个模式」写进 trace ——
                        # 冷启动阶段用户最需要的正是这个解释
                        f"why={decision.reason}"
                        + _obs_note(observation, configured=media_extractor is not None)
                    ),
                    degraded=interp.evidence_mode
                    in (EvidenceMode.TEXT_ONLY, EvidenceMode.MEASURED_ONLY),
                )
            ],
        )

    return behavior_interpreter


def _extract_observation(media_extractor, raw: RawInput):
    """取模型观察。失败返回 ``(None, NodeError)`` —— **不抛异常**。

    提取器本身已经把上游错误变成 ``ok=False``；这里再包一层是为了防
    未预期的异常（如模型名拼错、配置缺失）拖垮整个节点。

    ⚠️ **调用方必须用 ``media_extractor is None`` 区分「未配置」与「失败」**。
    两者都返回 ``None`` —— 初版因此让 trace 里两种情况显示成同一句话，
    而那正是 B24 记下的「两件事看起来一样」。（见 B25）
    """
    if media_extractor is None or not raw.audio_url:
        return None, None

    from app.schemas import MediaKind

    try:
        obs = media_extractor.extract(
            raw.audio_url, media_kind=raw.media_kind or MediaKind.AUDIO
        )
    except Exception as exc:  # noqa: BLE001 — 降级，且必须可见
        return None, NodeError(
            node="behavior_interpreter",
            kind=ErrorKind.UPSTREAM_ERROR,
            severity=ErrorSeverity.DEGRADED,
            message=f"多模态观察失败：{type(exc).__name__}: {exc}",
        )

    if not obs.ok:
        return obs, NodeError(
            node="behavior_interpreter",
            kind=ErrorKind.UPSTREAM_ERROR,
            severity=ErrorSeverity.DEGRADED,
            message=f"多模态观察不可用：{obs.error}",
        )
    return obs, None


def _obs_note(observation, configured: bool) -> str:
    """trace 里的观察说明。

    **四种状态必须可区分**（初版只区分了两种，于是「未配置」与「失败」一样）：

    | 状态 | 含义 |
    | --- | --- |
    | `未配置` | 没注入提取器 —— 这是**配置问题** |
    | `失败` | 注入了但抛异常 —— 这是**上游问题** |
    | `不可用` | 模型说这段媒体看不出东西 —— 这是**正确行为** |
    | `N动作/M物体` | 成功 |

    三者的排查方向完全不同。写成一样等于没有这个字段。
    """
    if not configured:
        return " obs=未配置"
    if observation is None:
        return " obs=失败"
    if not observation.ok:
        return " obs=不可用"
    return f" obs={len(observation.actions)}动作/{len(observation.scene_objects)}物体"


def _observed_evidence(observation) -> list:
    """把模型观察转成 `OBSERVED` 证据项。

    **`log_odds_contribution` 必须为 `None`** —— 它不在概率模型里。
    契约会校验这一点（见 `BehaviorInterpretation._contribution_matches_mode`）。
    """
    if observation is None or not observation.is_usable:
        return []

    from app.schemas import EvidenceItem, EvidenceKind

    return [
        EvidenceItem(
            kind=EvidenceKind.OBSERVED,
            statement=text,
            source=f"model:{observation.source_model}",
            value=None,
            reference=None,
            log_odds_contribution=None,
        )
        for text in observation.evidence_statements()
    ]


def _append_observed(interp, observation):
    """把观察追加到已产出的解释上。**不改动其他任何字段。**

    刻意只做一件事：追加证据。
    若在这里顺手重排候选或调整模式，未经校验的模型输出就会影响判断 ——
    而那正是 `OBSERVED` 与 `MEASURED` 分开的意义。
    """
    extra = _observed_evidence(observation)
    if not extra:
        return interp
    return interp.model_copy(update={"evidence": [*interp.evidence, *extra]})


def render_interpretation(state: AgentState) -> AgentState:
    """把结构化解释渲染为文本。**不调用模型** —— 模板实现，确定性。"""
    started = _now()
    interp = state.get("interpretation")
    if interp is None:
        return AgentState(
            draft_response="我暂时无法解析这段叫声。",
            node_trace=[
                _trace("render_interpretation", started=started, degraded=True)
            ],
        )

    lines: list[str] = []
    top = interp.top_candidate
    if top is not None and top.posterior is not None:
        lines.append(
            f"{top.display}（置信度 {top.posterior:.0%}，共 {len(interp.candidates)} 个候选）"
        )
    elif top is not None:
        lines.append(f"{top.display}（未给出数值置信度）")
    else:
        lines.append(interp.candidates[0].display if interp.candidates else "无法判断")

    if interp.evidence:
        lines.append("\n依据：")
        for e in interp.evidence:
            # 贡献值只在有概率模型的模式下存在（见 EvidenceItem 契约）。
            # 无模型时只陈述测量值——不补一个 0.00，那会谎称「已参与计算」。
            if e.log_odds_contribution is None:
                lines.append(f"  · {e.statement}")
            else:
                sign = "+" if e.log_odds_contribution >= 0 else ""
                lines.append(
                    f"  · {e.statement}（{sign}{e.log_odds_contribution:.2f}）"
                )

    if interp.suggested_observation:
        lines.append(f"\n{interp.suggested_observation}")
    if interp.limitations:
        lines.append(f"\n{interp.limitations}")

    return AgentState(
        draft_response="\n".join(lines),
        node_trace=[
            _trace("render_interpretation", started=started, decision="模板渲染")
        ],
    )


# ─────────────────────────────────────────────────────────────
# 6. 记忆抽取与写入
# ─────────────────────────────────────────────────────────────

#: 记录类指令前缀。剥掉它们剩下的才是要记的内容。
_RECORD_PREFIXES = (
    "请记住",
    "记住",
    "帮我记一下",
    "帮我记",
    "记一下",
    "记录一下",
    "别忘了记",
)


def extract_candidates(state: AgentState) -> AgentState:
    """从本轮对话抽取候选记忆。

    **P0 只处理显式记录请求**（``RECORD_EVENT``）。被动抽取（从闲聊里
    发现值得记的事）未实现 —— 它需要模型抽取 + 更完善的准入策略。

    这是务实的选择：显式记录覆盖了「数据飞轮」最核心的入口，
    且行为确定、可测试。
    """
    started = _now()
    if state.get("intent") is not InputIntent.RECORD_EVENT:
        return AgentState(
            candidate_memories=[],
            node_trace=[
                _trace("memory_extractor", started=started, decision="非记录意图，跳过")
            ],
        )

    text = (state.get("transcribed_text") or require_input(state).text or "").strip()
    content = text
    for prefix in _RECORD_PREFIXES:
        if content.startswith(prefix):
            content = content[len(prefix) :].strip("：:，,。 ")
            break

    if not content:
        return AgentState(
            candidate_memories=[],
            node_trace=[
                _trace("memory_extractor", started=started, decision="剥掉前缀后为空")
            ],
        )

    user_id, pet_id = tenant_of(state)
    slots = state.get("route_slots", RouteSlots())
    from datetime import datetime as _dt

    valid_from = None
    if slots.time_range:
        try:
            valid_from = _dt.fromisoformat(slots.time_range)
        except ValueError:
            valid_from = None

    candidate = MemoryEvent(
        user_id=user_id,
        pet_id=pet_id,
        # 追溯用：这条记忆是哪一轮对话产生的（不是隔离键，也不进检索排序）
        session_id=state.get("session_id"),
        event_type=EventType.PREFERENCE,
        subject=_guess_subject(content),
        content=content,
        polarity=_guess_polarity(content),
        valid_from=valid_from,
        source=MemorySource.USER_OBSERVATION,
        confidence=0.9,
        status=MemoryStatus.ACTIVE,
    )

    return AgentState(
        candidate_memories=[candidate],
        node_trace=[
            _trace(
                "memory_extractor",
                started=started,
                decision=f"抽出 1 条候选：{content[:20]}",
            )
        ],
    )


#: 主体识别规则：从内容里提取「关于什么」的键。用于冲突检测的「同主体」判定。
_SUBJECT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("vacuum", ("吸尘器", "吸尘")),
    ("cat_wand", ("逗猫棒", "逗猫")),
    ("scratching_post", ("猫抓板", "抓板")),
    ("carrier", ("猫包", "航空箱", "笼子")),
    ("litter_box", ("猫砂盆", "砂盆", "猫砂")),
    ("food", ("猫粮", "罐头", "粮")),
    ("water", ("喝水", "水碗", "水盆")),
    ("window", ("窗台", "窗户", "窗边")),
    ("stranger", ("陌生人", "生人")),
)

_NEGATIVE_CUES = ("怕", "不喜欢", "讨厌", "不敢", "躲", "害怕", "抗拒", "拒绝")


def _guess_subject(content: str) -> str:
    """猜主体。未识别到则用内容的哈希前缀 —— 保证「不同内容不会误判为同主体」。"""
    for subject, keywords in _SUBJECT_HINTS:
        if any(kw in content for kw in keywords):
            return subject
    import hashlib

    return "misc:" + hashlib.blake2b(content.encode("utf-8"), digest_size=4).hexdigest()


def _guess_polarity(content: str) -> Polarity:
    if any(cue in content for cue in _NEGATIVE_CUES):
        return Polarity.NEGATIVE
    if any(cue in content for cue in ("喜欢", "爱", "最喜欢", "开心")):
        return Polarity.POSITIVE
    return Polarity.NEUTRAL


def make_memory_writer(store: MemoryStore, embedder):
    """构造记忆写入节点。**这是唯一的写入点**（`DESIGN.md` §2.2 W2）。"""
    writer = MemoryWriter(store=store, embedder=embedder)

    def memory_writer(state: AgentState) -> AgentState:
        started = _now()
        candidates = state.get("candidate_memories", [])
        if not candidates:
            return AgentState(
                written_memory_ids=[],
                node_trace=[
                    _trace("memory_writer", started=started, decision="无候选")
                ],
            )

        decisions = [writer.decide(c) for c in candidates]
        result = writer.apply(decisions)
        summary = "；".join(f"{d.action.value}({d.reason[:24]})" for d in decisions)

        return AgentState(
            written_memory_ids=result.written_memory_ids,
            errors=[
                NodeError(
                    node="memory_writer",
                    kind=ErrorKind.INVALID_OUTPUT,
                    severity=ErrorSeverity.SILENT,
                    message=f"{action}: {reason}",
                )
                for action, reason in result.skipped
            ],
            node_trace=[_trace("memory_writer", started=started, decision=summary)],
        )

    return memory_writer


def record_acknowledge(state: AgentState) -> AgentState:
    """记录类意图的「执行 + 明确反馈」（`DESIGN.md` §5.3）。

    这条路径的响应**必须报告写入结果** —— 因为该意图的两个方向误判代价都高，
    靠阈值无解，所以改用「执行 + 明确反馈」让用户能以极低成本纠错。
    """
    started = _now()
    written = state.get("written_memory_ids", [])
    candidates = state.get("candidate_memories", [])

    if written and candidates:
        content = candidates[0].content
        text = f"已记录：{content}。记错了请直接告诉我。"
    else:
        text = "我这次没能记下来，可以再说一次吗？"

    return AgentState(
        draft_response=text,
        guard_mode="echo_only",
        node_trace=[
            _trace(
                "record_acknowledge",
                started=started,
                decision=f"写入 {len(written)} 条",
            )
        ],
    )


# ─────────────────────────────────────────────────────────────
# 7. 档案分析
# ─────────────────────────────────────────────────────────────


def make_profile_analyzer(analyzer: VisionAnalyzer):
    """构造档案分析节点。"""

    def profile_analyzer(state: AgentState) -> AgentState:
        started = _now()
        urls = require_input(state).image_urls
        try:
            draft = identify_from_photos(urls, analyzer)
        except Exception as exc:  # noqa: BLE001 — 全失败必须显式报错，**不编造特征**
            return AgentState(
                errors=[
                    NodeError(
                        node="profile_analyzer",
                        kind=ErrorKind.DATA_MISSING,
                        severity=ErrorSeverity.FATAL,
                        message=str(exc),
                    )
                ],
                draft_response=f"档案建立失败：{exc}",
                node_trace=[_trace("profile_analyzer", started=started, degraded=True)],
            )

        note = draft.coverage_note or f"基于 {draft.analyzed_photo_count} 张照片"
        return AgentState(
            profile_draft=draft,
            draft_response=(
                f"我从照片里提取到这些稳定特征：{'、'.join(draft.must_keep_features) or '无'}。"
                f"（{note}）确认无误吗？"
            ),
            node_trace=[
                _trace(
                    "profile_analyzer",
                    started=started,
                    decision=f"稳定 {len(draft.must_keep_features)} 项，不稳定 {len(draft.observed_but_unstable)} 项",
                )
            ],
        )

    return profile_analyzer
