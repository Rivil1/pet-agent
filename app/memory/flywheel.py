"""记忆飞轮：准入 → 去重（强化）→ 冲突消解 → 置信度路由。

对应 docs/DESIGN.md §3.5「写入：数据飞轮的决策管线」。

**这是纯代码模块，不调用大模型。** 模型只负责「抽取候选」，
准入、去重、冲突、路由全部由代码决定 —— 因为它们需要可复现、可审计、可测试。

四条设计原则的实现位置：

| 原则 | 实现 |
|---|---|
| 判据是「未来会不会被再次检索」 | ``judge_value`` |
| 去重是**强化**，不是重复插入 | ``_dedup_decision``（唯一索引 + 语义相似） |
| 冲突需三条件齐备，**时间范围重叠是关键** | 委托契约层 ``MemoryEvent.conflicts_with`` |
| **模型输出不得成为下一轮的事实输入** | ``route_by_confidence`` 强制 SYSTEM_INFERENCE → PENDING |
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.llm.base import Embedder, cosine
from app.schemas import (
    ASR_CONFIDENCE_DISCOUNT,
    SOURCE_TRUST,
    EventType,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    MemoryWriteDecision,
    WriteAction,
)
from app.store.base import DuplicateMemory, MemoryStore

#: 语义去重阈值。超过则视为同一件事，转为强化。
DEDUP_TAU = 0.92

#: 晋升到 Profile 层的最小出现次数。
PROMOTION_MIN_SUPPORT = 3

#: 晋升到 Profile 层的最小时间跨度（天）。
PROMOTION_MIN_SPAN_DAYS = 14

#: 未被确认的 PENDING 记忆超过此天数自动拒绝。
PENDING_TTL_DAYS = 14

#: 内容最短长度。过短的句子几乎不可能是可检索的记忆。
MIN_CONTENT_CHARS = 4

#: 「含数量/时间」的识别模式。
#:
#: ⚠️ **两个已踩过的坑**：
#:
#: 1. 早期实现逐个字符检查汉字数字（一二三…）。汉语里「一」「十」在日常用语中
#:    极常见：「**一**开就跑」「**一直**这样」都不是数量，却会被误判 →
#:    所有这类记忆都进 PENDING，永远不会生效。
#: 2. 即使改成「数字 + 单位」，**单位选择不当仍会误报**：
#:    「十**分**黏人」会被当成「10 分钟」，「一**点**」会被当成「1 点钟」。
#:
#: 因此单位表只保留**歧义小**的：
#: - 中文数字后允许单字单位（次、天、个…）——这些字单独出现时几乎总是量词
#: - **剔除 `分` 与 `点`**（「十分」「一点」「有点」歧义太大）
#: - 时间点靠阿拉伯数字捕获（「早上 7 点」→ 匹配 `7`）
_QUANTITY_PATTERN = re.compile(
    r"[0-9０-９]+"
    r"|(?:[一二三四五六七八九十百千万两]+)"
    r"(?:分钟|小时|公斤|毫升|次|天|周|月|年|个|颗|粒|勺|克|斤|度|碗|遍)"
)

#: 纯语气词/无信息内容。不记 —— 记了只会稀释检索。
_FILLER = frozenset(
    {"好的", "嗯", "哦", "谢谢", "哈哈", "收到", "测试", "在吗", "你好", "ok", "okay"}
)

#: 疑问句不记。用户的问题不是关于宠物的事实。
_QUESTION_MARKS = ("？", "?")


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class WritePolicy:
    """写入策略。集中配置，便于评测时扫描参数敏感性（DESIGN.md §6.4）。"""

    dedup_tau: float = DEDUP_TAU
    promotion_min_support: int = PROMOTION_MIN_SUPPORT
    promotion_min_span_days: int = PROMOTION_MIN_SPAN_DAYS
    pending_ttl_days: int = PENDING_TTL_DAYS
    min_content_chars: int = MIN_CONTENT_CHARS


# ─────────────────────────────────────────────────────────────
# 1. 准入：价值判定
# ─────────────────────────────────────────────────────────────


def judge_value(event: MemoryEvent, policy: WritePolicy = WritePolicy()) -> tuple[bool, str]:
    """这条信息值得长期保存吗？

    判据（DESIGN.md §3.5）：**未来会不会被再次检索**。
    不会 → 不记。记忆的价值在于被检索，不在于被存储。
    """
    content = event.content.strip()

    # 先查无信息量：能说清是「废话」时，不应用「过短」这个粗理由
    if content.lower() in _FILLER:
        return False, "无信息量内容（语气词/寒暄）"

    if len(content) < policy.min_content_chars:
        return False, f"内容过短（{len(content)} 字），不构成可检索的记忆"

    if content.endswith(_QUESTION_MARKS):
        return False, "疑问句不是关于宠物的事实"

    if event.event_type is EventType.HEALTH:
        # 健康记录另走 health_records 表，不进通用记忆，避免检索时混入
        return False, "健康信号应写入 health_records，不进入通用记忆"

    return True, "可作为长期记忆"


# ─────────────────────────────────────────────────────────────
# 2. 去重键
# ─────────────────────────────────────────────────────────────


def make_dedup_key(event: MemoryEvent) -> str:
    """去重键 = (event_type, subject, 归一化内容)。

    归一化只做大小写与空白 —— **不做语义归一化**，
    因为语义层面的重复由向量相似度处理（见 ``_dedup_decision``）。
    两者互补：键精确、向量模糊。
    """
    normalized = " ".join(event.content.strip().lower().split())
    return f"{event.event_type.value}:{event.subject}:{normalized}"


# ─────────────────────────────────────────────────────────────
# 3. 置信度路由
# ─────────────────────────────────────────────────────────────


def route_by_confidence(
    event: MemoryEvent, policy: WritePolicy = WritePolicy()
) -> tuple[MemoryStatus, str]:
    """按来源可信度决定落库状态。

    **这里实现防自我强化（不变量 I1）**：
    ``SYSTEM_INFERENCE`` 的可信度只有 0.60，必然落入 PENDING，
    结构上无法成为 ``ACTIVE``。
    """
    trust = SOURCE_TRUST[event.source]

    if event.source is MemorySource.SYSTEM_INFERENCE:
        return (
            MemoryStatus.PENDING_CONFIRMATION,
            f"系统推断（可信度 {trust:.2f}）只能待确认，不得作为事实",
        )

    # ASR 转写后的内容：语音识别错误会被当作事实存下且事后无法察觉，
    # 因此对**含数字/时间**的内容降权并要求确认（DESIGN.md §3.4）。
    if event.source is MemorySource.USER_OBSERVATION and _looks_like_quantity(
        event.content
    ):
        discounted = trust * ASR_CONFIDENCE_DISCOUNT
        return (
            MemoryStatus.PENDING_CONFIRMATION,
            f"内容含数量/时间（可能来自语音转写，可信度 {discounted:.2f}），需用户确认",
        )

    if trust >= 0.85:
        return MemoryStatus.ACTIVE, f"来源可信度 {trust:.2f}，直接生效"
    if trust >= 0.5:
        return MemoryStatus.PENDING_CONFIRMATION, f"来源可信度 {trust:.2f}，需确认"
    return MemoryStatus.REJECTED, f"来源可信度 {trust:.2f} 过低"


def _looks_like_quantity(text: str) -> bool:
    """内容是否包含**数量或时间表达**。

    只有这类内容才需要用户确认 —— 因为语音转写把「三点」听成「三只」这种错误
    无法事后察觉。

    **不要用「含汉字数字」做判断**：汉语里「一」「十」等在日常用语中极常见，
    逐个字符检查会把「一开就跑」误判为数量表达。
    """
    return _QUANTITY_PATTERN.search(text) is not None


# ─────────────────────────────────────────────────────────────
# 4. 决策管线
# ─────────────────────────────────────────────────────────────


@dataclass
class MemoryWriter:
    """记忆写入器。**唯一写入点**（DESIGN.md §2.2 W1/W2）。"""

    store: MemoryStore
    embedder: Embedder
    policy: WritePolicy = WritePolicy()

    def decide(self, event: MemoryEvent) -> MemoryWriteDecision:
        """对单个候选记忆做完整决策。**不写入。**"""
        ok, reason = judge_value(event, self.policy)
        if not ok:
            return MemoryWriteDecision(event=event, action=WriteAction.REJECT, reason=reason)

        # ── 去重（强化） ──
        decision = self._dedup_decision(event)
        if decision is not None:
            return decision

        # ── 冲突消解 ──
        conflicts = self.store.find_conflicts(
            user_id=event.user_id, pet_id=event.pet_id, candidate=event
        )
        if conflicts:
            return self._conflict_decision(event, conflicts)

        # ── 置信度路由 ──
        status, route_reason = route_by_confidence(event, self.policy)
        if status is MemoryStatus.REJECTED:
            return MemoryWriteDecision(
                event=event, action=WriteAction.REJECT, reason=route_reason
            )
        if status is MemoryStatus.PENDING_CONFIRMATION:
            return MemoryWriteDecision(
                event=self._with_status(event, status),
                action=WriteAction.PENDING,
                reason=route_reason,
            )
        return MemoryWriteDecision(
            event=self._with_status(event, MemoryStatus.ACTIVE),
            action=WriteAction.WRITE,
            reason=route_reason,
        )

    # ── 去重 ─────────────────────────────────────────────

    def _dedup_decision(self, event: MemoryEvent) -> MemoryWriteDecision | None:
        """精确键命中 或 语义相似度超阈值 → **强化**，而非新增。"""
        key = make_dedup_key(event)
        existing = self.store.find_by_dedup_key(
            user_id=event.user_id, pet_id=event.pet_id, dedup_key=key
        )
        if existing is not None:
            return MemoryWriteDecision(
                event=self._reinforced(event, existing),
                action=WriteAction.REINFORCE,
                reason=f"与已有记忆内容一致（精确键命中），转为强化，support_count → {existing.support_count + 1}",
                duplicate_of=existing.memory_id,
            )

        # 语义去重
        vec = self.embedder.embed(event.content)
        for item in self.store.search_memories(
            user_id=event.user_id, pet_id=event.pet_id, query_vector=vec, limit=10
        ):
            sim = cosine(vec, self.embedder.embed(item.event.content))
            if sim >= self.policy.dedup_tau:
                return MemoryWriteDecision(
                    event=self._reinforced(event, item.event),
                    action=WriteAction.REINFORCE,
                    reason=(
                        f"与已有记忆语义相似（{sim:.3f} ≥ {self.policy.dedup_tau}），"
                        f"转为强化，support_count → {item.event.support_count + 1}"
                    ),
                    duplicate_of=item.event.memory_id,
                )
        return None

    @staticmethod
    def _reinforced(new: MemoryEvent, existing: MemoryEvent) -> MemoryEvent:
        """强化：把计数与时间带到新事件上，由存储层更新既有记录。"""
        return new.model_copy(
            update={
                "memory_id": existing.memory_id,
                "support_count": existing.support_count + 1,
                "last_seen_at": _now(),
                "created_at": existing.created_at,
                "status": existing.status,
                "confidence": min(1.0, max(existing.confidence, new.confidence) + 0.02),
            }
        )

    # ── 冲突 ─────────────────────────────────────────────

    def _conflict_decision(
        self, event: MemoryEvent, conflicts: list[MemoryEvent]
    ) -> MemoryWriteDecision:
        """冲突消解：**保留取代链，不删除历史**。

        新信息来自更可信来源 → 新记忆 ACTIVE，旧的 SUPERSEDED。
        新信息来自**更低**可信度来源 → **不覆盖**，新记忆进 PENDING。
        """
        new_trust = SOURCE_TRUST[event.source]
        strongest = max(conflicts, key=lambda c: SOURCE_TRUST[c.source])
        old_trust = SOURCE_TRUST[strongest.source]

        old_ids = [c.memory_id for c in conflicts if c.memory_id]

        if new_trust < old_trust:
            return MemoryWriteDecision(
                event=self._with_status(event, MemoryStatus.PENDING_CONFIRMATION),
                action=WriteAction.PENDING,
                reason=(
                    f"与 {len(conflicts)} 条现有记忆冲突，但新来源可信度更低"
                    f"（{new_trust:.2f} < {old_trust:.2f}），不覆盖，待用户确认"
                ),
            )

        return MemoryWriteDecision(
            event=self._with_status(event, MemoryStatus.ACTIVE).model_copy(
                update={"supersedes": old_ids}
            ),
            action=WriteAction.SUPERSEDE,
            reason=(
                f"与 {len(conflicts)} 条现有记忆冲突且来源更可信"
                f"（{new_trust:.2f} ≥ {old_trust:.2f}），建立取代链"
            ),
            supersedes=old_ids,
        )

    @staticmethod
    def _with_status(event: MemoryEvent, status: MemoryStatus) -> MemoryEvent:
        return event.model_copy(update={"status": status})

    # ── 落地 ─────────────────────────────────────────────

    def apply(self, decisions: list[MemoryWriteDecision]) -> AppliedResult:
        """执行决策。**这是唯一的写入点。**"""
        written: list[str] = []
        skipped: list[tuple[str, str]] = []

        for d in decisions:
            ev = d.event
            if ev.dedup_key is None:
                ev = ev.model_copy(update={"dedup_key": make_dedup_key(ev)})

            try:
                if d.action is WriteAction.WRITE:
                    stored = self.store.insert_memory(ev, vector=self.embedder.embed(ev.content))
                    written.append(stored.memory_id or "")

                elif d.action is WriteAction.REINFORCE:
                    self.store.update_memory(ev, vector=self.embedder.embed(ev.content))
                    written.append(ev.memory_id or "")

                elif d.action is WriteAction.SUPERSEDE:
                    now = _now()
                    for old_id in d.supersedes:
                        old = self.store.get_memory(
                            user_id=ev.user_id, pet_id=ev.pet_id, memory_id=old_id
                        )
                        self.store.update_memory(
                            old.model_copy(
                                update={
                                    "status": MemoryStatus.SUPERSEDED,
                                    "valid_to": old.valid_to or now,
                                }
                            )
                        )
                    stored = self.store.insert_memory(ev, vector=self.embedder.embed(ev.content))
                    written.append(stored.memory_id or "")

                elif d.action is WriteAction.PENDING:
                    stored = self.store.insert_memory(ev, vector=self.embedder.embed(ev.content))
                    written.append(stored.memory_id or "")

                else:  # REJECT
                    skipped.append((d.action.value, d.reason))

            except DuplicateMemory as exc:
                # 唯一索引冲突 = 并发写入同一 dedup_key → 转为强化路径
                skipped.append(("duplicate", str(exc)))

        return AppliedResult(written_memory_ids=written, skipped=skipped)


@dataclass(frozen=True)
class AppliedResult:
    written_memory_ids: list[str]
    skipped: list[tuple[str, str]]


# ─────────────────────────────────────────────────────────────
# 晋升与过期
# ─────────────────────────────────────────────────────────────


def is_eligible_for_promotion(
    event: MemoryEvent,
    *,
    min_support: int = PROMOTION_MIN_SUPPORT,
    min_span_days: int = PROMOTION_MIN_SPAN_DAYS,
) -> bool:
    """是否可晋升到 Profile 层（从「事件」变为「属性」）。

    条件：重复出现 ≥N 次 **且** 时间跨度 ≥M 天。
    只满足次数不满足跨度是不够的 —— 一天内说三次不代表这是稳定属性。
    """
    if event.event_type not in (EventType.PREFERENCE, EventType.ROUTINE):
        return False
    if event.support_count < min_support:
        return False
    latest = event.last_seen_at or event.created_at
    return (latest - event.created_at) >= timedelta(days=min_span_days)


def expire_pending(
    store: MemoryStore, *, user_id: str, pet_id: str, ttl_days: int = PENDING_TTL_DAYS
) -> list[str]:
    """把长期未被确认的 PENDING 记忆标记为 REJECTED（DESIGN.md §3.5 R4）。

    返回被过期的 memory_id 列表。
    """
    cutoff = _now() - timedelta(days=ttl_days)
    expired: list[str] = []
    for ev in store.list_memories(user_id=user_id, pet_id=pet_id, include_non_active=True):
        if ev.status is not MemoryStatus.PENDING_CONFIRMATION:
            continue
        if ev.created_at > cutoff:
            continue
        store.update_memory(ev.model_copy(update={"status": MemoryStatus.REJECTED}))
        if ev.memory_id:
            expired.append(ev.memory_id)
    return expired
