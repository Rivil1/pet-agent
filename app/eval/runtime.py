"""评测运行时：**按配置装配可执行的被测系统**。

## 设计要点：被测系统是真的那个

这里不实现一套「评测专用的简化管线」。被测的是 `build_graph` 装配出来的
**同一张图** —— 唯一差别是消融开关。理由：

评测若跑在一条与生产不同的路径上，它测出的数字就不指向生产的行为。
而「为了好测而另写一套」是最常见的做法，也是最容易自欺的做法：
它会很自然地绕开那些真正难测、也真正容易出问题的地方。

## 三个消融开关的落点

| 开关 | 在图上动的地方 |
|---|---|
| `use_memory=False` | `memory_retriever` 节点直接返回空 |
| `retrieval_mode=naive` | `memory_retriever` 换成朴素向量 top-k |
| `apply_guard=False` | `response_guard` 节点不拦截，只放行草案 |

三处都在 `build_graph` 的参数里，**同一个函数、同一张图**。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from app.eval.config import EvalConfig
from app.graph import build_graph, initial_state
from app.interpreter import PriorTable
from app.llm import HashEmbedder, LLMClient, MockLLM
from app.memory.retrieval import (
    RetrievalResult,
    RetrievalWeights,
    mmr_select,
    retrieve_with_status,
    score_item,
)
from app.profile import VisualObservation
from app.schemas import PetProfile, RawInput, Species, VisualProfile
from app.store import InMemoryStore

__all__ = [
    "Runtime",
    "TurnOutcome",
    "OverconfidentLLM",
    "naive_retrieve",
    "make_runtime",
]

#: 先验表路径。评测要用的是生产同一份 —— 换一份会让数字不指向生产。
_PRIOR_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "priors"
    / "catmeows_stats.json"
)


# =============================================================================
# 朴素检索（消融用）
# =============================================================================


def naive_retrieve(
    *,
    store: Any,
    embedder: Any,
    user_id: str,
    pet_id: str,
    query: str,
    k: int = 5,
    wanted_types: Any = None,
    weights: RetrievalWeights = RetrievalWeights(),
    recall_limit: int = 20,
    now: Any = None,
) -> RetrievalResult:
    """朴素向量 top-k：**去掉重排序与 MMR 多样性裁剪**。

    这是「纯 RAG」的常见形态，也是混合检索要对比的基线。

    ## 它为什么值得被单独实现，而不是「把权重调成 0」

    权重调 0 保留了重排序的**形状**（仍然按 score 排序、仍然做 MMR），
    只是各项贡献变成常量 —— 那测的是「排序键失效」，
    不是「没有重排序」。而这里要回答的问题是：
    「加这一层结构化重排序，到底带来了什么」。

    所以它直接返回向量相似度排序的前 k 条，不做任何后处理。

    注意力：仍需**保留降级语义** —— 消融基线也必须在向量索引不可用时
    如实报告，否则两组数字的差异里会混入「一边降级了」这个无关因素。
    """
    from app.store.vectors import VectorIndexUnavailable

    try:
        query_vector = embedder.embed(query)
        recalled = store.search_memories(
            user_id=user_id, pet_id=pet_id, query_vector=query_vector, limit=recall_limit
        )
    except VectorIndexUnavailable as exc:
        return RetrievalResult(items=[], degraded_reason=str(exc))

    # 只按向量分排序，截断到 k —— 没有重排序、没有 MMR
    ranked = sorted(recalled, key=lambda it: it.score, reverse=True)
    return RetrievalResult(items=ranked[:k])


# =============================================================================
# 一次交互的结果
# =============================================================================


@dataclass
class TurnOutcome:
    """驱动一次对话之后的全部可观测产出。"""

    final_response: str = ""
    draft_response: str = ""
    intent: str = "unknown"
    route_confidence: float = 0.0
    traced_nodes: tuple[str, ...] = ()
    degraded: bool = False
    degraded_notice: str | None = None
    retrieved_count: int = 0
    written_memory_ids: tuple[str, ...] = ()
    #: 守卫拦下了什么。**即使关掉守卫也仍然记录** ——
    #: 那正是「守卫的价值」这个数字的来源。
    guard_violations: tuple[str, ...] = ()
    guard_blocked: bool = False
    guard_skipped: bool = False
    interpretation: Any = None
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def response_was_rewritten(self) -> bool:
        return self.guard_blocked and self.final_response != self.draft_response


# =============================================================================
# 运行时
# =============================================================================


class _StubVision:
    """视觉替身：不调网络。评测不测视觉质量（那需要兽医标注，属 B 层）。

    明确写出来而不是用 `None`：调用方需要知道「视觉没被测」，
    而不是以为它默默通过了。
    """

    def analyze(self, image_url: str) -> VisualObservation:
        return VisualObservation(image_url=image_url)


class OverconfidentLLM:
    """**会编造的 LLM —— 故障注入。**

    ## 它模拟的不是「一个坏模型」

    它模拟的是**真实模型最常见的失败模式**：在没有依据时仍然给出
    听起来确定的回答。这是 LLM 的默认行为 —— 它有问必答，
    而「不知道」需要额外的机制去保证。

    ## 为什么评测必须包含它

    `MockLLM` 输出固定文案，**永远不会**产生需要守卫拦下的内容。
    用它在 `apply_guard=False` 下跑，结果与基线完全一致 ——
    于是报告会得出「守卫没有价值」这个**反向的**结论。

    故障注入把「守卫到底拦住了什么」变成一个可测量的数字：
    同一个会编造的 LLM，开守卫 vs 关守卫，无据陈述率的差值就是答案。

    这也是 `docs/06-roadmap.md` 里那句「**用一个实验证明某个约束是必要的**」
    的具体做法。
    """

    #: 这句话同时命中两类违规：
    #: 「根据记录」要求有检索到的记忆（此处没有），
    #: 「它很健康」是健康禁词。
    UNGROUNDED = "根据记录，它很健康，而且特别怕吸尘器。"

    def __init__(self, default: str | None = None) -> None:
        self._default = default or self.UNGROUNDED
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        return self._default

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _make_llm(profile: str) -> LLMClient:
    """按剖面选 LLM。

    `neutral`（默认）用 `MockLLM` —— `DESIGN §6.5` 要求评测离线可跑。
    `overconfident` 用故障注入替身，让守卫的有效性变得可测。
    """
    if profile == "overconfident":
        return OverconfidentLLM()  # type: ignore[return-value]
    return MockLLM(default="喵。")


@dataclass
class Runtime:
    """按配置装配好的可执行环境。

    **结构上**满足 `EvalContext` 协议，但**不继承它** ——
    协议里的 `config` 是 property，继承会让它变成类属性，
    而 dataclass 会把它当成「有默认值的字段」，于是所有无默认值的
    字段都报 `non-default argument follows default argument`。
    Python 的 Protocol 本来就是结构化类型，继承不是必需的。

    **每个场景拿到自己的 store**（`new_store()`），
    否则前一个场景写下的记忆会污染后一个 —— 而那种污染是静默的：
    后一个场景的数字会莫名其妙地偏离，而没有任何报错。
    """

    config: EvalConfig
    prior: PriorTable
    embedder: Any = field(default_factory=HashEmbedder)
    llm: LLMClient | None = None

    def __post_init__(self) -> None:
        self._graphs: dict[int, Any] = {}
        self._tenant_seq = 0
        #: 本次运行的唯一标识。
        #:
        #: **必需，否则跨运行会累积。** 租户计数器每次运行从 0 开始，
        #: 于是第二次跑评测时 `eval-baseline-1` 里还留着第一次的数据 ——
        #: 表现为「冲突检出 3 条（期望 1）」「记忆 29 条（期望 1）」这类
        #: **指向错误**的失败。
        self._run_id = uuid4().hex[:8]
        #: 本次运行创建过的租户，收尾时清掉。
        self._created: list[tuple[str, str]] = []
        self._stores: list[Any] = []
        # LLM 在装配时定型：配置在运行期不可变（frozen dataclass），
        # 所以这里没有"两份配置共用同一个 LLM"的风险。
        if self.llm is None:
            self.llm = _make_llm(self.config.llm_profile)

    def new_tenant(self) -> tuple[str, str]:
        """下一个隔离的 `(user_id, pet_id)` 命名空间。

        ## 为什么需要它（这是评测跑出来的一个真缺陷）

        内存后端下每个场景都拿一个**全新的 store**，自然互不干扰。
        但 MySQL 后端下所有场景共享同一张表 —— 而它们都用
        `user_id="u1", pet_id="p1"`，于是：

        - `memory-dedup` 看到 29 条记忆（历次运行 + 其他场景的）
        - `memory-conflict` 检出 6 条冲突

        而这两条都会以「去重没生效」「冲突误判」的形式报出来 ——
        **指向错误的地方**。

        ## 为什么不是「每个场景前清库」

        清库对**真实数据库是破坏性的**。而这个评测本来就允许
        指向线上库（那是验证「两套实现行为一致」的唯一办法）。
        用独立租户既不破坏数据，又顺带验证了真实的多租户隔离。
        """
        self._tenant_seq += 1
        n = self._tenant_seq
        # ⚠️ **`pet_id` 也必须带上 run_id，不能只让 `user_id` 唯一。**
        #
        # `pets.pet_id` 是**主键**，而 `save_pet` 走 `ON DUPLICATE KEY UPDATE`。
        # 所以两次运行都用 `pet-1` 时，第二次的写入不会新建一行 ——
        # 它会静默地把上一次那一行更新掉，而 `UPDATE` 子句里**不含 user_id**，
        # 于是行的归属还是旧租户。
        #
        # 结果：`save_pet` 报成功，紧接着 `get_pet` 报「不存在」。
        # 那个报错指向「写入没生效」，而真正的原因是**跨运行主键撞车**。
        tenant = (f"eval-{self._run_id}-{n}", f"pet-{self._run_id}-{n}")
        self._created.append(tenant)
        return tenant

    def cleanup(self) -> int:
        """删掉本次运行写下的全部数据。返回删除条数。

        只在有 `delete_pet_data` 的后端上做事（MySQL 有，内存不需要）。
        **失败不抛出** —— 收尾失败不该把一次成功的评测变成失败，
        但它会打印出来（静默失败的清理会慢慢把库塞满）。
        """
        import logging

        log = logging.getLogger(__name__)
        removed = 0
        for store in self._stores:
            deleter = getattr(store, "delete_pet_data", None)
            if not callable(deleter):
                continue
            for user_id, pet_id in self._created:
                try:
                    removed += int(deleter(user_id=user_id, pet_id=pet_id) or 0)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "清理评测数据失败 user=%s pet=%s：%s", user_id, pet_id, exc
                    )
        return removed

    # ── EvalContext 协议 ──────────────────────────────────

    def new_store(self) -> Any:
        if self.config.store_backend == "memory":
            return InMemoryStore()
        store = self._mysql_store()
        # 只有真实后端需要收尾清理（内存随进程消失）
        self._stores.append(store)
        return store

    def _mysql_store(self) -> Any:
        """MySQL 后端。未配置时**抛错而不是静默退回内存**。

        静默退回会让 `mysql-store` 这一组「消融」其实跑的还是内存 ——
        两组数字一样，于是得出「后端不影响正确性」这个**错误结论**。
        """
        from app.store.factory import build_store_from_env

        bundle = build_store_from_env(dim=getattr(self.embedder, "dim", 1024))
        if bundle.backend != "mysql":
            raise RuntimeError(
                "配置要求 mysql 后端，但环境变量未提供 MYSQL_*。"
                "不能静默退回内存 —— 那会让本次对照失去意义。"
            )
        return bundle.store

    def new_pet(
        self, store: Any, *, user_id: str, pet_id: str, name: str
    ) -> PetProfile:
        pet = PetProfile(
            pet_id=pet_id,
            user_id=user_id,
            name=name,
            species=Species.CAT,
            breed="英短",
            visual=VisualProfile(fur_color="橘白", fur_length="短毛", eye_color="黄"),
            must_keep_features=["橘白短毛", "圆脸", "胸口一块白毛"],
            identity_prompt="This is the same real cat.",
        )
        store.save_pet(pet)
        return pet

    # ── 驱动 ──────────────────────────────────────────────

    def graph_for(self, store: Any) -> Any:
        """按配置装配图。同一个 store 复用同一张图（编译有开销）。"""
        key = id(store)
        if key not in self._graphs:
            self._graphs[key] = build_graph(
                store=store,
                embedder=self.embedder,
                llm=self.llm,
                prior=self.prior,
                feature_extractor=lambda url: None,
                vision=_StubVision(),
                use_memory=self.config.use_memory,
                retrieve_fn=naive_retrieve if self.config.retrieval_mode == "naive" else None,
                apply_guard=self.config.apply_guard,
            )
        return self._graphs[key]

    def converse(
        self,
        store: Any,
        *,
        user_id: str,
        pet_id: str,
        text: str,
        session_id: str = "eval-session",
        trace_id: str = "eval-trace",
    ) -> TurnOutcome:
        """跑一轮对话，收集全部可观测产出。"""
        graph = self.graph_for(store)
        state = initial_state(
            user_id=user_id,
            pet_id=pet_id,
            raw_input=RawInput(text=text),
            session_id=session_id,
            trace_id=trace_id,
        )
        state["transcribed_text"] = text

        started = time.perf_counter()
        result = graph.invoke(state, config={"configurable": {"thread_id": session_id}})
        elapsed_ms = (time.perf_counter() - started) * 1000

        traces = list(result.get("node_trace", []))
        guard = result.get("guard_result")

        return TurnOutcome(
            final_response=result.get("final_response") or "",
            draft_response=result.get("draft_response") or "",
            intent=(
                result.get("intent").value
                if hasattr(result.get("intent"), "value")
                else str(result.get("intent", "unknown"))
            ),
            route_confidence=float(result.get("route_confidence", 0.0) or 0.0),
            traced_nodes=tuple(t.node for t in traces),
            degraded=any(t.degraded for t in traces),
            degraded_notice=next(
                (t.decision for t in traces if t.degraded), None
            ),
            retrieved_count=len(result.get("retrieved_memories", [])),
            written_memory_ids=tuple(result.get("written_memory_ids", []) or ()),
            guard_violations=tuple(
                v.detail for v in getattr(guard, "violations", []) or []
            ),
            guard_blocked=bool(
                guard and not guard.passed and guard.degrade_to_conservative
            ),
            guard_skipped=not self.config.apply_guard,
            interpretation=result.get("interpretation"),
            latency_ms=elapsed_ms,
            raw=dict(result),
        )

    def converse_many(
        self,
        store: Any,
        *,
        user_id: str,
        pet_id: str,
        texts: Sequence[str],
        session_id: str = "eval-session",
    ) -> list[TurnOutcome]:
        """连续跑多轮。**同一 session** —— 会话记忆注入会起作用，这是有意的。"""
        return [
            self.converse(
                store,
                user_id=user_id,
                pet_id=pet_id,
                text=t,
                session_id=session_id,
            )
            for t in texts
        ]


def make_runtime(config: EvalConfig) -> Runtime:
    """装配一个运行时。**离线可用** —— 用 HashEmbedder + MockLLM。

    为什么评测默认不接真模型：`DESIGN §6.5` 要求「无 API Key / 无网络时
    也能跑通全部评测」。而且 A 层指标（确定性那些）本来就不依赖模型质量 ——
    接真模型只会让数字随供应商抖动，让「这次比上次好」变成噪声。
    """
    return Runtime(config=config, prior=PriorTable.load(_PRIOR_PATH))
