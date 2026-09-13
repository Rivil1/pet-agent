"""向量索引抽象。

## 这个模块存在的唯一理由：让「租户过滤」无法被实现者忘记

语义召回的隔离要求在**检索时**加硬过滤（只召回这只猫的记忆），
而不是「先全部召回再用应用代码筛一遍」。后者有两个致命问题：

1. **`limit` 语义被破坏**。若先取 top-20 再筛掉别人的，
   实际返回的可能只有 3 条 —— 而调用方以为自己拿到了 20 条候选。
2. **越权数据已经进入了进程内存**。筛掉只是不返回，
   但它已经被读出来了，那一刻隔离就已经破了。

所以 `search` 的签名**强制要求** `user_id` 与 `pet_id`，且都是 keyword-only、
**没有默认值**。调用方无法「忘记」传 —— 这是把隔离从约定变成约束
（与 `MemoryStore` 的签名同源，见 `app/store/base.py` 的 T3 说明）。

## 为什么还要一个显式的「不可用」实现

按项目的 fail-closed 取向（D45）：向量索引连不上时，
**不能静默返回空列表**。空的检索结果与「这只猫确实没有相关记忆」
在调用方看来完全一样，而那会导致系统自信地说出
「没有相关记录」—— 用户无法分辨这是真的没有，还是后端挂了。

`UnavailableVectorIndex` 会抛异常，把「不可用」变成一个显式状态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class VectorIndexError(Exception):
    """向量索引的基础错误。"""


class VectorIndexUnavailable(VectorIndexError):
    """索引不可用（连不上 / 未配置 / 维度不符）。

    **刻意是一个异常而不是空结果** —— 详见模块 docstring。
    """


@dataclass(frozen=True)
class VectorEntry:
    """一条待索引的记忆向量。

    带 `user_id` / `pet_id` 是为了让**索引本身**能承载隔离字段。
    两者都不是可选的：没有它们，检索侧的过滤表达式就无从写起。
    """

    memory_id: str
    user_id: str
    pet_id: str
    vector: list[float]


@dataclass(frozen=True)
class ScoredMemory:
    """一次检索命中。"""

    memory_id: str
    score: float


@runtime_checkable
class VectorIndex(Protocol):
    """向量索引接口。

    ⚠️ **所有检索方法强制 `user_id` + `pet_id`，且为关键字参数、无默认值。**
    这不是风格问题：它让「跨宠物召回」在类型层面无法被写出来。
    """

    def upsert(self, entries: Sequence[VectorEntry]) -> int:
        """写入或覆盖向量。返回写入条数。

        **幂等**：同一 `memory_id` 重复写入应覆盖而非追加。
        这条保证是「Milvus 丢了可重建」的前提 ——
        回填会重复跑，不幂等的话索引里会出现重复条目，
        而重复条目会让同一个 `memory_id` 在 top-k 里占多个位置。
        """
        ...

    def search(
        self,
        *,
        user_id: str,
        pet_id: str,
        vector: list[float],
        limit: int = 20,
    ) -> list[ScoredMemory]:
        """按 `pet_id` 硬过滤的向量召回。

        Raises:
            VectorIndexUnavailable: 索引不可用。**不要改成返回空列表。**
        """
        ...

    def delete(self, memory_ids: Sequence[str]) -> int:
        """按 id 删除。返回删除条数。不存在的 id 不算错误。"""
        ...

    def count(self) -> int:
        """索引中的条目总数。用于「索引是否落后于 MySQL」的健全性检查。"""
        ...

    def describe(self) -> dict[str, str]:
        """给 `/healthz` 与日志用。**不含凭据。**"""
        ...


# ─────────────────────────────────────────────────────────────
# 内存实现（测试与无 Milvus 时的降级目标）
# ─────────────────────────────────────────────────────────────


class InMemoryVectorIndex:
    """内存向量索引。

    **不是「假实现」**：它真的做暴力余弦检索，结果与 Milvus 在语义上一致
    （只是在规模上不成立）。所以它既能支撑单元测试，
    也能在没有 Milvus 时让整条链路跑通。

    它**不满足生产要求**（无持久化、无并发保护），但把接口固定下来，
    使调用方不必等真实索引就位就能被测试覆盖。
    """

    def __init__(self, *, dim: int = 1024) -> None:
        self._dim = dim
        self._rows: dict[str, VectorEntry] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def upsert(self, entries: Sequence[VectorEntry]) -> int:
        written = 0
        for entry in entries:
            if len(entry.vector) != self._dim:
                raise VectorIndexError(
                    f"向量维度不符：期望 {self._dim}，得到 {len(entry.vector)}"
                    f"（memory_id={entry.memory_id}）"
                )
            self._rows[entry.memory_id] = entry
            written += 1
        return written

    def search(
        self,
        *,
        user_id: str,
        pet_id: str,
        vector: list[float],
        limit: int = 20,
    ) -> list[ScoredMemory]:
        if limit <= 0:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(f"查询向量维度不符：期望 {self._dim}，得到 {len(vector)}")

        scored: list[ScoredMemory] = []
        for entry in self._rows.values():
            # 隔离是**硬过滤**：不属于这只猫的条目根本不进入打分。
            if entry.user_id != user_id or entry.pet_id != pet_id:
                continue
            scored.append(ScoredMemory(entry.memory_id, _cosine(vector, entry.vector)))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:limit]

    def delete(self, memory_ids: Sequence[str]) -> int:
        removed = 0
        for mid in memory_ids:
            if self._rows.pop(mid, None) is not None:
                removed += 1
        return removed

    def count(self) -> int:
        return len(self._rows)

    def describe(self) -> dict[str, str]:
        return {
            "kind": "memory",
            "dim": str(self._dim),
            "count": str(self.count()),
        }


class UnavailableVectorIndex:
    """显式不可用的索引。

    每个方法都抛 `VectorIndexUnavailable`，且**消息里带原因** ——
    「为什么不可用」直接决定要不要降级、以及要不要告诉用户。
    """

    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def reason(self) -> str:
        return self._reason

    def upsert(self, entries: Sequence[VectorEntry]) -> int:
        raise VectorIndexUnavailable(self._reason)

    def search(
        self,
        *,
        user_id: str,
        pet_id: str,
        vector: list[float],
        limit: int = 20,
    ) -> list[ScoredMemory]:
        raise VectorIndexUnavailable(self._reason)

    def delete(self, memory_ids: Sequence[str]) -> int:
        raise VectorIndexUnavailable(self._reason)

    def count(self) -> int:
        raise VectorIndexUnavailable(self._reason)

    def describe(self) -> dict[str, str]:
        return {"kind": "unavailable", "reason": self._reason}


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。

    与 `app/llm/base.py` 的实现保持同一语义（零向量返回 0），
    但**不复用**它：这里是索引层，不该依赖 LLM 层的客户端封装。
    """
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))
