"""记忆与档案存储抽象。

对应 docs/ARCHITECTURE.md §2（存储架构）与 §2.6（租户隔离的三个强制点）。

**本模块是实现租户隔离强制点 T3 的地方**：
所有查询方法签名强制要求 ``user_id`` 与 ``pet_id``，**且无默认值**。
调用方无法「忘记」传租户标识——这是把隔离从约定变成约束的关键。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.schemas import (
    MemoryEvent,
    MemoryItem,
    MeowRecord,
    PendingInterpretation,
    PetProfile,
    SessionMessage,
)


class StoreError(Exception):
    """存储层基础错误。"""


class NotFound(StoreError):
    """资源不存在**或不属于当前用户**。

    两者合并为同一个错误是有意的：区分它们会通过响应码泄露资源存在性
    （`ARCHITECTURE.md` §4.2 A2）。
    """


class DuplicateMemory(StoreError):
    """`(pet_id, dedup_key)` 唯一约束冲突。

    对应 `ARCHITECTURE.md` §2.2 的唯一索引：重复写入应转为**强化**。
    """


class InvariantViolation(StoreError):
    """存储层不变量被违反。

    与契约层的 Pydantic 校验不同：契约层挡的是**构造时的错误**，
    存储层挡的是**绕过契约的写入路径**（反序列化、脚本、未来的新代码）。
    """


@runtime_checkable
class MemoryStore(Protocol):
    """记忆与档案的存储接口。

    ⚠️ **所有方法强制 ``user_id`` + ``pet_id``，且为关键字参数。**
    这不是风格问题：这是让「多租户隔离」在类型层面无法被绕过。
    """

    # ── 档案 ──────────────────────────────────────────────

    def save_pet(self, pet: PetProfile) -> PetProfile: ...

    def get_pet(self, *, user_id: str, pet_id: str) -> PetProfile:
        """取档案。不存在或不属于该用户 → 抛 ``NotFound``。"""
        ...

    def list_pets(self, *, user_id: str) -> list[PetProfile]: ...

    # ── 记忆 ──────────────────────────────────────────────

    def insert_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent:
        """插入一条记忆。

        ``vector`` 为语义检索用的嵌入。**放在接口里是必须的** ——
        早期版本只在具体实现上加了它，导致抽象接口与实现不一致，
        新实现（如 PostgreSQL）很容易漏掉。

        Raises:
            DuplicateMemory: ``(pet_id, dedup_key)`` 已存在 → 调用方应转为强化。
        """
        ...

    def update_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent: ...

    def get_memory(
        self, *, user_id: str, pet_id: str, memory_id: str
    ) -> MemoryEvent: ...

    def find_by_dedup_key(
        self, *, user_id: str, pet_id: str, dedup_key: str
    ) -> MemoryEvent | None: ...

    def find_conflicts(
        self, *, user_id: str, pet_id: str, candidate: MemoryEvent
    ) -> list[MemoryEvent]:
        """找出与候选冲突的**现有**记忆（时间范围重叠 + 主体相同 + 极性相反）。"""
        ...

    def list_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        include_non_active: bool = False,
        session_id: str | None = None,
    ) -> list[MemoryEvent]:
        """``session_id`` 为**追溯过滤**（"哪一轮产生的"），不是隔离键。"""
        ...

    def insert_meow_record(self, record: MeowRecord) -> MeowRecord:
        """写入一条已被主人标注的叫声记录。

        它是案例推理（k-NN）的样本库 —— 与通用记忆分开，
        因为它带声学特征，且只能由**主人的标注**产生。
        """
        ...

    def list_meow_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        only_confirmed: bool = True,
        session_id: str | None = None,
    ) -> list[MeowRecord]:
        """按租户取叫声记录。

        ``only_confirmed=True`` 时只返回 ``status == ACTIVE`` 的 ——
        未确认的样本不构成证据（与 `SimilarSample.context is None` 同理）。

        ``session_id`` 为**追溯过滤**，不影响相似度计算。
        """
        ...

    # ── 待标注解释（主人标注的入口） ────────────────────

    def save_pending_interpretation(
        self, pending: PendingInterpretation
    ) -> PendingInterpretation:
        """存档一次解释，等主人标注。

        **特征只存在服务端** —— 客户端标注时只发情境/动作/结果，
        服务端按 id 取回自己存的特征。否则 `MEASURED` 的承诺会失效。
        """
        ...

    def get_pending_interpretation(
        self, *, user_id: str, pet_id: str, interpretation_id: str
    ) -> PendingInterpretation | None:
        """取回待标注解释。归属不符时返回 ``None``（与 `NotFound` 同理，不泄露存在性）。"""
        ...

    # ── 会话消息（日报的输入） ────────────────────────

    def insert_message(self, message: SessionMessage) -> SessionMessage: ...

    def list_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SessionMessage]:
        """按时间范围取消息。**日报的输入。**"""
        ...

    def list_recent_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[SessionMessage]:
        """取**最近若干条**消息，按**时间正序**返回。**会话记忆注入的输入。**

        与 ``list_messages`` 的分工是刻意的：

        | 方法 | 用途 | 取法 |
        | --- | --- | --- |
        | ``list_messages`` | **日报** | 时间窗口（整天）|
        | ``list_recent_messages`` | **当前对话** | 某会话最近 N 条 |

        排序与截断方向相反，合成一个方法会让两边都变脆。

        ``session_id=None`` 表示**不按会话过滤**（兼容没有 session 的历史数据）。
        """
        ...

    def search_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        query_vector: list[float],
        limit: int = 20,
    ) -> list[MemoryItem]:
        """向量召回。**必须按 ``pet_id`` 过滤**——这是隔离的正确性底线。"""
        ...
