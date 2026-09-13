"""模型与向量化抽象。

对应 docs/ARCHITECTURE.md §9（技术选型）与 §8.3（``MOCK_PROVIDER``）。

**为什么要有 mock**：`DESIGN.md` §6.5 要求「无 API Key / 无网络时也能跑通全部评测」。
若所有测试都依赖真实 API，评测就不可复现，也不可在 CI 里跑。

设计约束：
1. ``Embedder`` 返回**确定性**向量 —— 同一文本必须得到同一向量（可复现性要求）。
2. ``HashEmbedder`` 不是假实现，它是一个真实的 hashing vectorizer，
   只是语义能力弱。它在 P0 承担「可离线」的职责，生产换 DashScope。
3. ``MockLLM`` 只做**结构化占位**，不做「假装理解」——它返回可预测的结果，
   使上层逻辑（记忆飞轮、守卫）可以被单元测试覆盖。
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

#: 哈希向量器维度。与 ``ARCHITECTURE.md`` §2.3 的记忆语义检索维度对齐。
DEFAULT_EMBED_DIM = 1024


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。"""

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """确定性 hashing vectorizer。

    **不是语义模型**，只做字符 n-gram 哈希。能力弱但有一个关键性质：
    **完全确定、无网络依赖**，使检索链路与评测可离线复现。

    生产环境替换为 DashScope ``text-embedding-v3``（维度 1024，已核实）。
    """

    def __init__(self, dim: int = DEFAULT_EMBED_DIM, ngram: int = 2) -> None:
        self._dim = dim
        self._ngram = ngram

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        normalized = " ".join(text.lower().split())
        n = self._ngram
        grams = (
            [normalized[i : i + n] for i in range(len(normalized) - n + 1)]
            if len(normalized) >= n
            else [normalized]
        )
        for gram in grams:
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。假定两向量已归一化时退化为点积。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


# ─────────────────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """文本生成。"""

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str: ...


class MockLLM:
    """确定性 LLM 占位。

    设计原则：**不假装理解**。它按调用方给出的 ``routes`` 关键词表返回可预测结果。

    存在意义：让记忆飞轮、守卫、编排等**不含模型逻辑的部分**可以被单元测试覆盖。
    真正的语义能力由生产实现（DashScope）提供，不在这里模拟。
    """

    def __init__(self, default: str = "", routes: dict[str, str] | None = None) -> None:
        self._default = default
        self._routes = routes or {}
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        for keyword, response in self._routes.items():
            if keyword in user:
                return response
        return self._default

    @property
    def call_count(self) -> int:
        return len(self.calls)
