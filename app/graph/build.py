"""图组装。

对应 docs/DESIGN.md §2.2「编排图」。

**P0 只组装快路径**（单意图固定链路）：

```
START → load_context → understand_input
          ├─ memory_retriever → companion_agent ─┐
          ├─ behavior_interpreter → render ──────┤
          ├─ memory_extractor → writer → ack ────┤→ response_guard → END
          ├─ profile_analyzer ───────────────────┤
          └─ clarify_ask ────────────────────────┘
```

``load_context`` 承担档案层的主键直读。**它必须在入口**，因为档案是所有
下游分支都需要的上下文 —— 放在某一条分支里会让其他分支拿不到。

**慢路径（planner / fan-out / wave 调度）未组装** —— 规划层是 P3
（`DESIGN.md` §7.1：预期触发率 <10%，不影响产品）。状态字段已预留，
加规划层时不需要改动现有节点签名。

**写入顺序**遵循 `DESIGN.md` §2.2：

| 意图 | 顺序 | 理由 |
|---|---|---|
| `RECORD_EVENT` | extractor → writer → ack → guard | 响应要报告写入结果 |
| 其余 | … → guard（副作用在 P0 未异步化） | 写入是副作用 |

> P0 未实现异步 outbox：`memory_writer` 同步执行，省掉 worker 与队列
> （`ARCHITECTURE.md` §3.2 的 P0 决策）。
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any

from app.graph.nodes import (
    clarify_ask,
    extract_candidates,
    make_behavior_interpreter,
    make_companion_agent,
    make_context_loader,
    make_memory_retriever,
    make_memory_writer,
    make_profile_analyzer,
    record_acknowledge,
    render_interpretation,
    response_guard,
    route_after_understanding,
    understand_input,
)
from app.graph.state import AgentState
from app.interpreter import PriorTable
from app.llm import Embedder, LLMClient
from app.profile import VisionAnalyzer
from app.store.base import MemoryStore
from langgraph.graph import END, START, StateGraph

#: 条件边的目标映射。**必须与 ``route_after_understanding`` 的返回值一致** ——
#: 缺一个键会在运行期抛错，所以这里集中定义并由测试覆盖。
#:
#: 键类型写 ``Hashable`` 而非 ``str``：LangGraph 的 ``path_map`` 参数声明为
#: ``dict[Hashable, str]``，而 ``dict`` 的键是**不变的**（invariant），
#: ``dict[str, str]`` 并不能赋给它。
ROUTE_TARGETS: dict[Hashable, str] = {
    "clarify_ask": "clarify_ask",
    "memory_retriever": "memory_retriever",
    "behavior_interpreter": "behavior_interpreter",
    "profile_analyzer": "profile_analyzer",
    "memory_extractor": "memory_extractor",
}


def build_graph(
    *,
    store: MemoryStore,
    embedder: Embedder,
    llm: LLMClient,
    prior: PriorTable,
    feature_extractor: Callable[[str], Any],
    vision: VisionAnalyzer,
    similar_lookup: Callable[[AgentState], list] | None = None,
    records_lookup: Callable[[AgentState], list] | None = None,
    media_extractor: Any | None = None,
):
    """组装并编译图。

    所有外部依赖**注入**，不在此处实例化 —— 这样测试可以传入 mock，
    离线跑通整条链路（`DESIGN.md` §6.5 可复现要求）。

    ``records_lookup`` 提供该猫**已被主人标注**的叫声记录（`MeowRecord`），
    即案例推理的样本库。**不传时行为解释只能走到 `measured_only`** ——
    因为案例推理需要样本，而样本只能来自主人的记录。

    ``media_extractor`` 产出 **`OBSERVED` 证据**（多模态模型对画面/音频的观察）。
    它**不参与模式选择，也不改变候选排序** —— 只在解释产出后追加。
    """
    builder = StateGraph(AgentState)

    builder.add_node("load_context", make_context_loader(store))
    builder.add_node("understand_input", understand_input)
    builder.add_node("memory_retriever", make_memory_retriever(store, embedder))
    builder.add_node("companion_agent", make_companion_agent(llm))
    builder.add_node(
        "behavior_interpreter",
        make_behavior_interpreter(
            prior, feature_extractor, similar_lookup, records_lookup, media_extractor
        ),
    )
    builder.add_node("render_interpretation", render_interpretation)
    builder.add_node("memory_extractor", extract_candidates)
    builder.add_node("memory_writer", make_memory_writer(store, embedder))
    builder.add_node("record_acknowledge", record_acknowledge)
    builder.add_node("profile_analyzer", make_profile_analyzer(vision))
    builder.add_node("clarify_ask", clarify_ask)
    builder.add_node("response_guard", response_guard)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "understand_input")
    builder.add_conditional_edges(
        "understand_input", route_after_understanding, ROUTE_TARGETS
    )

    # 快路径各分支
    builder.add_edge("memory_retriever", "companion_agent")
    builder.add_edge("behavior_interpreter", "render_interpretation")

    # 写入顺序：extractor → writer → ack，然后统一进守卫
    builder.add_edge("memory_extractor", "memory_writer")
    builder.add_edge("memory_writer", "record_acknowledge")

    # 所有面向用户的文本都必须过守卫（W3）
    for node in (
        "companion_agent",
        "render_interpretation",
        "record_acknowledge",
        "profile_analyzer",
        "clarify_ask",
    ):
        builder.add_edge(node, "response_guard")

    builder.add_edge("response_guard", END)

    compiled = builder.compile()

    # 编译期自检：每个声明的路由目标都必须真的是一个节点。
    # 缺少这类检查时，「加了边但忘了加节点」会让**整张图无法编译**，
    # 而这个错误直到第一次构建图才会暴露。
    node_names = set(builder.nodes)
    for route, target in ROUTE_TARGETS.items():
        if target not in node_names:
            raise ValueError(
                f"ROUTE_TARGETS['{route}'] 指向不存在的节点 '{target}'。"
                "routes 与 nodes 必须保持一致。"
            )

    return compiled
