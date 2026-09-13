"""记忆检索：混合召回 → 重排序 → 多样性裁剪 → 上下文注入。

对应 docs/DESIGN.md §3.5「检索：混合检索管线」与 §「上下文注入契约」。

三个容易做错的点，本模块显式处理：

1. **时近衰减必须按事件类型区分**。统一衰减率会让系统忘掉猫「一直喜欢」的东西，
   却记住上周三的琐事。半衰期取自契约层的 ``EVENT_HALFLIFE_DAYS``。
2. **隔离是硬过滤，不是打分项**。别的宠物的记忆**根本不进入候选**，
   而不是「打低分」——后者会泄漏到 top-K。
3. **上下文注入必须显式声明「检索不到就是没有」**，否则模型会用常识补全。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

from app.llm.base import Embedder, cosine
from app.schemas import (
    SOURCE_TRUST,
    ContextInjectionBlock,
    ConversationTurn,
    EventType,
    MemoryEvent,
    MemoryItem,
    MemoryStatus,
    PetProfile,
    RetrievalSource,
    ScoreBreakdown,
)
from app.store.base import MemoryStore
from app.store.vectors import VectorIndexUnavailable


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RetrievalWeights:
    """重排序权重。集中配置，便于消融实验（DESIGN.md §6.3）。"""

    semantic: float = 1.0
    recency: float = 0.30
    support: float = 0.20
    source_trust: float = 0.20
    type_match: float = 0.20


#: MMR 的 λ。越大越偏向相关性，越小越偏向多样性。
MMR_LAMBDA = 0.7

#: support 分量的饱和点。log(1+5) 归一化 —— 出现 5 次即接近满分。
SUPPORT_SATURATION = 5


def recency_score(event: MemoryEvent, *, now: datetime | None = None) -> float:
    """时近性得分，**按事件类型使用不同半衰期**。

    ``halflife_days is None`` 表示不衰减（偏好 / 习惯），返回 1.0。
    """
    hl = event.halflife_days
    if hl is None:
        return 1.0
    anchor = event.occurred_at or event.created_at
    age_days = max(0.0, ((now or _now()) - anchor).total_seconds() / 86400.0)
    # `0.5 ** float` 已经是 float，无需再包一层 `float()`
    return 0.5 ** (age_days / hl)


def support_score(event: MemoryEvent) -> float:
    """重复次数的饱和得分。用 log 压缩，避免高频事件压过一切。"""
    return min(1.0, math.log1p(event.support_count) / math.log1p(SUPPORT_SATURATION))


def score_item(
    item: MemoryItem,
    *,
    embedder: Embedder,
    query_vector: list[float],
    wanted_types: frozenset[EventType] | None,
    weights: RetrievalWeights = RetrievalWeights(),
    now: datetime | None = None,
) -> MemoryItem:
    """重排序打分。

    注意：``ScoreBreakdown`` 的字段存的是**已加权**的分量，
    这样契约里的 ``total``（各字段之和）就是最终得分，无需改契约。
    """
    ev = item.event
    semantic = cosine(query_vector, embedder.embed(ev.content))
    breakdown = ScoreBreakdown(
        semantic_similarity=semantic * weights.semantic,
        recency=recency_score(ev, now=now) * weights.recency,
        support=support_score(ev) * weights.support,
        source_trust=SOURCE_TRUST[ev.source] * weights.source_trust,
        type_match=(
            weights.type_match
            if (wanted_types is not None and ev.event_type in wanted_types)
            else 0.0
        ),
    )
    return item.model_copy(
        update={
            "score": breakdown.total,
            "breakdown": breakdown,
            "retrieval_source": RetrievalSource.HYBRID,
        }
    )


def retrieve(
    *,
    store: MemoryStore,
    embedder: Embedder,
    user_id: str,
    pet_id: str,
    query: str,
    k: int = 5,
    wanted_types: frozenset[EventType] | None = None,
    weights: RetrievalWeights = RetrievalWeights(),
    recall_limit: int = 20,
    now: datetime | None = None,
) -> list[MemoryItem]:
    """完整检索管线。**只返回命中**。

    Args:
        k: 最终返回条数。
        wanted_types: 路由推断出的相关事件类型。为 ``None`` 表示不做类型加权。
        recall_limit: 向量召回条数（重排序前的候选规模）。

    需要知道「是否降级」时用 `retrieve_with_status` ——
    本函数把降级压成了空列表，而空列表无法区分
    「确实没有」与「根本没查成」。
    """
    return retrieve_with_status(
        store=store,
        embedder=embedder,
        user_id=user_id,
        pet_id=pet_id,
        query=query,
        k=k,
        wanted_types=wanted_types,
        weights=weights,
        recall_limit=recall_limit,
        now=now,
    ).items


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的完整结果：命中 + **降级原因**。

    ## 为什么需要一个类型，而不是只返回 `list`

    向量索引不可用时，`[]`（空命中）与「这只猫确实没有相关记忆」
    在调用方看来完全一样。而两者的用户可见行为应当不同：

    | 情况 | 正确的行为 |
    |---|---|
    | 确实没有记忆 | 「我还不知道它这件事」 |
    | 索引挂了 | 「检索暂时不可用」—— 不能让用户以为系统查过了 |

    把两者压成同一个 `[]`，系统会自信地对一个它压根没查过的问题作答。
    所以降级必须是一个**显式字段**，由节点写进 trace（与 D40/D45 同一取向）。
    """

    items: list[MemoryItem]
    degraded_reason: str | None = None

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None


def retrieve_with_status(
    *,
    store: MemoryStore,
    embedder: Embedder,
    user_id: str,
    pet_id: str,
    query: str,
    k: int = 5,
    wanted_types: frozenset[EventType] | None = None,
    weights: RetrievalWeights = RetrievalWeights(),
    recall_limit: int = 20,
    now: datetime | None = None,
) -> RetrievalResult:
    """与 `retrieve` 同逻辑，但把**降级原因**一并返回。

    节点走这个入口；只想拿结果的调用方仍可用 `retrieve()`。
    """
    query_vector = embedder.embed(query)

    # ① 结构化预过滤（在 store 内部完成：pet_id + status=active）
    try:
        recalled = store.search_memories(
            user_id=user_id, pet_id=pet_id, query_vector=query_vector, limit=recall_limit
        )
    except VectorIndexUnavailable as exc:
        # **降级而不是抛。**
        #
        # 后端可能没配 Milvus（纯 MySQL 部署），或 Milvus 临时不可达。
        # 共同点是「查不到」不等于「没有」。把原话带出去，
        # 让 trace 与用户提示能说清楚发生了什么。
        return RetrievalResult(items=[], degraded_reason=str(exc))

    if not recalled:
        return RetrievalResult(items=[])

    # ② 重排序
    scored = [
        score_item(
            item,
            embedder=embedder,
            query_vector=query_vector,
            wanted_types=wanted_types,
            weights=weights,
            now=now,
        )
        for item in recalled
    ]
    scored.sort(key=lambda i: i.score, reverse=True)

    # ③ 多样性裁剪
    return RetrievalResult(items=mmr_select(scored, embedder=embedder, n=k))


def mmr_select(
    items: list[MemoryItem],
    *,
    embedder: Embedder,
    n: int,
    lambda_: float = MMR_LAMBDA,
) -> list[MemoryItem]:
    """Maximal Marginal Relevance 多样性裁剪。

    避免 5 条召回全是同一件事的变体（如「怕吸尘器」「怕吹风机」「怕理发器」
    高度相似，占满上下文却只提供一条信息）。
    """
    if len(items) <= n:
        return items

    vectors = {id(it): embedder.embed(it.event.content) for it in items}
    selected: list[MemoryItem] = []
    remaining = list(items)

    while remaining and len(selected) < n:
        if not selected:
            best = max(remaining, key=lambda i: i.score)
        else:

            def mmr(item: MemoryItem) -> float:
                redundancy = max(
                    cosine(vectors[id(item)], vectors[id(s)]) for s in selected
                )
                return lambda_ * item.score - (1 - lambda_) * redundancy

            best = max(remaining, key=mmr)
        selected.append(best)
        remaining.remove(best)

    return selected


# ─────────────────────────────────────────────────────────────
# 上下文注入
# ─────────────────────────────────────────────────────────────


def build_context_block(
    *,
    pet: PetProfile,
    items: list[MemoryItem],
    pending: list[MemoryEvent] | None = None,
    recent_turns: list[ConversationTurn] | None = None,
) -> ContextInjectionBlock:
    """构建注入 prompt 的记忆块。

    四个要点（DESIGN.md §3.5）：
    1. 区分「已确认」与「待确认」—— 模型不得把系统猜测当事实
    2. 携带 ``source`` 与 ``confidence`` —— 否则守卫无法判定断言是否有据
    3. 显式说明「检索不到就是没有」—— 否则模型用常识补全（幻觉主要来源）
    4. **把「最近对话」单独分区** —— assistant 的历史回复是推测，不是事实

    ``recent_turns`` 是**展示视图**（`ConversationTurn`）而不是持久化实体，
    理由见该类型的 docstring。
    """
    return ContextInjectionBlock(
        pet_identity=pet.identity_block(),
        known_facts=items,
        unconfirmed=pending or [],
        recent_turns=recent_turns or [],
        retrieval_note=(
            "以上为检索到的记录。未在其中的信息即为没有记录，"
            "不要推测或使用常识补全。"
            + (
                "（本次未检索到任何记忆，请明确告知用户没有相关记录。）"
                if not items
                else ""
            )
        ),
    )


def pending_memories(
    store: MemoryStore, *, user_id: str, pet_id: str, limit: int = 3
) -> list[MemoryEvent]:
    """取待确认记忆。**它们不进 known_facts**，单独标注。"""
    out = [
        ev
        for ev in store.list_memories(
            user_id=user_id, pet_id=pet_id, include_non_active=True
        )
        if ev.status is MemoryStatus.PENDING_CONFIRMATION
    ]
    out.sort(key=lambda e: e.created_at, reverse=True)
    return out[:limit]
