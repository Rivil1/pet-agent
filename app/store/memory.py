"""内存实现。P0 用；生产换 PostgreSQL + pgvector（docs/ARCHITECTURE.md §2.2）。

**它不是「简化版」，而是把架构文档里的结构性约束搬进了内存实现**，
以便这些约束可以被单元测试直接验证：

| 架构文档的约束 | 本实现的位置 |
|---|---|
| §2.6 T3 仓储层签名强制租户 | 每个方法都是 `*, user_id, pet_id` |
| §2.2 `idx_mem_dedup` 唯一索引 | `insert_memory` 抛 `DuplicateMemory` |
| §2.2 `no_active_system_inference` CHECK | `_assert_invariants`（深度防御） |
| §2.3 partial index `WHERE status='active'` | `search_memories` 只召回 active |
| §4.2 A2 归属不符返回 404 | `NotFound` 不区分「不存在」与「不属于你」 |
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.llm.base import cosine
from app.schemas import (
    MemoryEvent,
    MemoryItem,
    MemorySource,
    MemoryStatus,
    MeowRecord,
    PendingInterpretation,
    PetProfile,
    RetrievalSource,
    SessionMessage,
)
from app.store.base import DuplicateMemory, InvariantViolation, NotFound


def _tenant_of(obj: object) -> tuple[str, str]:
    """取 `(user_id, pet_id)`。级联删除与租户过滤都用它。

    写成函数而不是各处 `obj.user_id, obj.pet_id`：
    多一个取值路径，就多一个写错字段名的地方，
    而写错的后果是**删掉了别人的数据**。
    """
    return (getattr(obj, "user_id", ""), getattr(obj, "pet_id", ""))


@dataclass
class _StoredMemory:
    event: MemoryEvent
    vector: list[float] | None = None


@dataclass
class InMemoryStore:
    """内存存储。

    ⚠️ **不保证并发安全**，也不持久化。P0 用于让编排层与记忆飞轮可被测试。
    """

    _pets: dict[str, PetProfile] = field(default_factory=dict)
    _memories: dict[str, _StoredMemory] = field(default_factory=dict)
    _meow_records: dict[str, MeowRecord] = field(default_factory=dict)
    _pending: dict[str, PendingInterpretation] = field(default_factory=dict)
    _messages: dict[str, SessionMessage] = field(default_factory=dict)
    _dedup_index: dict[tuple[str, str], str] = field(default_factory=dict)

    # ── 档案 ──────────────────────────────────────────────

    def save_pet(self, pet: PetProfile) -> PetProfile:
        self._pets[pet.pet_id] = pet
        return pet

    def get_pet(self, *, user_id: str, pet_id: str) -> PetProfile:
        pet = self._pets.get(pet_id)
        # 归属不符与不存在返回同一个错误 —— 不泄露资源存在性
        if pet is None or pet.user_id != user_id:
            raise NotFound(f"pet {pet_id} 不存在")
        return pet

    def list_pets(self, *, user_id: str) -> list[PetProfile]:
        return [p for p in self._pets.values() if p.user_id == user_id]

    # ── 记忆 ──────────────────────────────────────────────

    def insert_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent:
        self._assert_invariants(event)

        # 先定型 memory_id：后续的字典键与索引都需要它是 str。
        # 用局部变量而非依赖 event.memory_id 的收窄 —— 赋值后类型收窄会丢失。
        memory_id = event.memory_id or str(uuid.uuid4())
        if event.memory_id is None:
            event = event.model_copy(update={"memory_id": memory_id})

        if event.dedup_key:
            key = (event.pet_id, event.dedup_key)
            if key in self._dedup_index:
                raise DuplicateMemory(f"dedup_key={event.dedup_key} 已存在")

        self._memories[memory_id] = _StoredMemory(event=event, vector=vector)
        if event.dedup_key:
            self._dedup_index[(event.pet_id, event.dedup_key)] = memory_id
        return event

    def update_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent:
        self._assert_invariants(event)
        if event.memory_id is None or event.memory_id not in self._memories:
            raise NotFound(f"memory {event.memory_id} 不存在")
        old = self._memories[event.memory_id]
        self._memories[event.memory_id] = _StoredMemory(
            event=event, vector=vector if vector is not None else old.vector
        )
        return event

    def get_memory(self, *, user_id: str, pet_id: str, memory_id: str) -> MemoryEvent:
        stored = self._memories.get(memory_id)
        if stored is None:
            raise NotFound(f"memory {memory_id} 不存在")
        ev = stored.event
        if ev.user_id != user_id or ev.pet_id != pet_id:
            raise NotFound(f"memory {memory_id} 不存在")
        return ev

    def find_by_dedup_key(
        self, *, user_id: str, pet_id: str, dedup_key: str
    ) -> MemoryEvent | None:
        mid = self._dedup_index.get((pet_id, dedup_key))
        if mid is None:
            return None
        ev = self._memories[mid].event
        return ev if ev.user_id == user_id else None

    def find_conflicts(
        self, *, user_id: str, pet_id: str, candidate: MemoryEvent
    ) -> list[MemoryEvent]:
        """冲突判定委托给契约层的 ``MemoryEvent.conflicts_with``。

        判定逻辑（同主体 + 反极性 + **时间范围重叠**）放在契约层，
        保证存储实现之间行为一致。
        """
        out: list[MemoryEvent] = []
        for stored in self._memories.values():
            ev = stored.event
            if ev.user_id != user_id or ev.pet_id != pet_id:
                continue
            if ev.memory_id == candidate.memory_id:
                continue
            if ev.status is not MemoryStatus.ACTIVE:
                continue
            if candidate.conflicts_with(ev):
                out.append(ev)
        return out

    def list_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        include_non_active: bool = False,
        session_id: str | None = None,
    ) -> list[MemoryEvent]:
        out = []
        for stored in self._memories.values():
            ev = stored.event
            if ev.user_id != user_id or ev.pet_id != pet_id:
                continue
            # 追溯过滤：不是隔离键，只是「哪一轮产生的」
            if session_id is not None and ev.session_id != session_id:
                continue
            if not include_non_active and not ev.is_retrievable_by_default:
                continue
            out.append(ev)
        return out

    # ── 叫声记录（案例推理的样本库） ─────────────────────

    def insert_meow_record(self, record: MeowRecord) -> MeowRecord:
        stored = record
        if not stored.record_id:
            stored = record.model_copy(
                update={"record_id": f"mr-{uuid.uuid4().hex[:12]}"}
            )
        assert stored.record_id is not None  # noqa: S101
        self._meow_records[stored.record_id] = stored
        return stored

    def list_meow_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        only_confirmed: bool = True,
        session_id: str | None = None,
    ) -> list[MeowRecord]:
        out = [
            r
            for r in self._meow_records.values()
            if r.user_id == user_id
            and r.pet_id == pet_id
            and (session_id is None or r.session_id == session_id)
            and (not only_confirmed or r.is_confirmed)
        ]
        # 按时间排序：案例推理的「最像的那次」在平局时应稳定选最早的，而非随内存顺序
        out.sort(key=lambda r: r.recorded_at)
        return out

    # ── 待标注解释（主人标注的入口） ─────────────────────

    def save_pending_interpretation(
        self, pending: PendingInterpretation
    ) -> PendingInterpretation:
        self._pending[pending.interpretation_id] = pending
        return pending

    def get_pending_interpretation(
        self, *, user_id: str, pet_id: str, interpretation_id: str
    ) -> PendingInterpretation | None:
        # 归属不符返回 None 而不是抛错 —— 与 NotFound 同理，不泄露资源存在性
        pending = self._pending.get(interpretation_id)
        if pending is None:
            return None
        if pending.user_id != user_id or pending.pet_id != pet_id:
            return None
        return pending

    # ── 会话消息（日报的输入） ─────────────────────────

    def insert_message(self, message: SessionMessage) -> SessionMessage:
        stored = message.with_id()
        assert stored.message_id is not None  # noqa: S101
        self._messages[stored.message_id] = stored
        return stored

    def list_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SessionMessage]:
        out = [
            m
            for m in self._messages.values()
            if m.user_id == user_id and m.pet_id == pet_id
        ]
        if since is not None:
            out = [m for m in out if m.at >= since]
        if until is not None:
            out = [m for m in out if m.at < until]
        out.sort(key=lambda m: m.at)
        return out

    def list_recent_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[SessionMessage]:
        """取某会话最近 ``limit`` 条消息，按时间**正序**返回。

        ``session_id=None`` 表示不按会话过滤 —— 兼容没有 session 的历史数据。
        """
        if limit <= 0:
            return []
        rows = [
            m
            for m in self._messages.values()
            if m.user_id == user_id
            and m.pet_id == pet_id
            and (session_id is None or m.session_id == session_id)
        ]
        rows.sort(key=lambda m: m.at)
        # 先取尾部（最近）再保持正序 —— 注入 prompt 时时间顺序不能反
        return rows[-limit:]

    def search_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        query_vector: list[float],
        limit: int = 20,
    ) -> list[MemoryItem]:
        """向量召回。模拟 SQL 的 partial index：**只召回 active 记忆**。

        隔离是硬过滤，不是打分项 —— 别的宠物的记忆**根本不进入候选**。
        """
        scored: list[tuple[float, MemoryEvent]] = []
        for stored in self._memories.values():
            ev = stored.event
            if ev.user_id != user_id or ev.pet_id != pet_id:
                continue
            if not ev.is_retrievable_by_default:
                continue
            if stored.vector is None:
                continue
            scored.append((cosine(query_vector, stored.vector), ev))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            MemoryItem(
                event=ev,
                score=max(0.0, sim),
                retrieval_source=RetrievalSource.VECTOR,
                matched_on="embedding",
            )
            for sim, ev in scored[:limit]
        ]

    # ── 级联硬删 ─────────────────────────────────────

    def delete_pet_data(self, *, user_id: str, pet_id: str) -> int:
        """**级联硬删**该宠物的全部数据，返回删除条数。

        与 `HealthRecordStore.delete_pet_data` 同一语义：
        软标记不满足删除要求 —— 一条标了 `deleted=True` 的记录
        仍然是泄露风险。

        与 `MySQLStore` 的同名方法必须行为一致，否则「删除」这件事
        在测试环境与生产会是两回事（而测试环境是通过的那个）。
        """
        removed = 0

        for mid in [
            k for k, v in self._memories.items() if _tenant_of(v.event) == (user_id, pet_id)
        ]:
            del self._memories[mid]
            removed += 1

        # 去重索引要同步清 —— 不清的话，删掉重记时那条 key 永远冲突，
        # 而报错是 DuplicateMemory，看起来像「重复写入」而不是「索引残留」。
        for key, mid in list(self._dedup_index.items()):
            if mid not in self._memories:
                del self._dedup_index[key]

        for rid in [
            k for k, v in self._meow_records.items() if _tenant_of(v) == (user_id, pet_id)
        ]:
            del self._meow_records[rid]
            removed += 1

        for iid in [
            k for k, v in self._pending.items() if _tenant_of(v) == (user_id, pet_id)
        ]:
            del self._pending[iid]
            removed += 1

        for mdid in [
            k for k, v in self._messages.items() if _tenant_of(v) == (user_id, pet_id)
        ]:
            del self._messages[mdid]
            removed += 1

        return removed

    # ── 不变量（深度防御） ────────────────────────────────

    @staticmethod
    def _assert_invariants(event: MemoryEvent) -> None:
        """在存储层再检查一次关键不变量。

        契约层校验的是「构造时」；存储层挡的是**绕过契约构造的写入路径**
        （反序列化、脚本、未来的新代码）。应用层挡错误，存储层挡不可能。

        生产环境对应 `ARCHITECTURE.md` §2.2 的 DB CHECK 约束。
        """
        if (
            event.source is MemorySource.SYSTEM_INFERENCE
            and event.status is MemoryStatus.ACTIVE
        ):
            raise InvariantViolation(
                "不变量 I1 被违反：SYSTEM_INFERENCE 不得以 ACTIVE 状态存储"
            )

        if event.valid_from and event.valid_to and event.valid_from > event.valid_to:
            raise InvariantViolation("valid_from 不得晚于 valid_to")
