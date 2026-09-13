"""LangSmith 观测接入：把一次请求变成可回放的调用树。

## 为什么需要它

`AgentState.node_trace` 只活在**单个响应**里 —— 请求结束就没了。
「系统为什么这么答」需要能**跨请求回放**：哪个节点花了多久、检索命中了什么、
守卫为什么降级。LangGraph 的每个节点天然是一个 span，接到 LangSmith 就有调用树。

## traceId 与 LangSmith run id 是**同一个标识**，不是映射表

LangChain 的 `RunnableConfig.run_id` 要求 UUID。而本项目的 `trace_id`
本来就由 `uuid4()` 生成，所以：

| `trace_id` 形态 | `run_id` |
| --- | --- |
| 合法 UUID（默认路径） | **直接使用** —— 二者字面相同 |
| 客户端自定义的非 UUID 串 | `uuid5(namespace, trace_id)` —— 仍是它的**确定性函数** |

两种情况下 `trace_id` 都同时写进 `metadata`，可在 LangSmith 里直接检索。
**不引入第二张 ID 表**：那会引入一个必然漂移的对应关系。

## 失败姿态：可见，但不致命

观测**不是正确性路径**。LangSmith 不可达时一次对话仍应正常完成，
所以这里全部 `fail-soft`。但也**不能静默** —— 启用状态与错误通过
`describe()` 暴露到 `/healthz`，否则「以为在采数据、其实没有」正是
本项目反复记为失败模式的形状（B22 / B25）。

## 为什么显式读环境变量，而不是依赖 SDK 自动开启

LangChain 支持 `LANGSMITH_TRACING` 之类的全局隐式开关。这里不走那条路：

1. 隐式开关是**进程全局**的，测试无法只关一个用例；
2. 本项目其它外部依赖（`app/llm/providers.py`）都是显式读 env 的，
   观测保持一致；
3. 显式构造 tracer 才能把 `run_id` 钉成我们的 `trace_id`。
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

#: 显式开关。缺省时按「有没有 key」推断。
ENV_TRACING = "PET_AGENT_TRACING"

#: key / project / endpoint 的候选变量名（本项目命名优先，兼容 LangSmith 与旧 LangChain 名）。
_ENV_KEYS = (
    "PET_AGENT_LANGSMITH_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
)
_ENV_PROJECTS = (
    "PET_AGENT_LANGSMITH_PROJECT",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_PROJECT",
)
_ENV_ENDPOINTS = (
    "PET_AGENT_LANGSMITH_ENDPOINT",
    "LANGSMITH_ENDPOINT",
    "LANGCHAIN_ENDPOINT",
)

DEFAULT_PROJECT = "pet-agent"

#: `uuid5` 派生用的命名空间。**写死且不可变** ——
#: 改了它，同一 `trace_id` 会指向另一个 run，历史关联全部失效。
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "pet-agent.trace")

#: 关的取值。**显式关必须能压过「有 key」** —— 否则 CI 里设了 key 就会真的外发。
_OFF = {"0", "false", "no", "off"}
_ON = {"1", "true", "yes", "on"}


def _env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = _env(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class TracingConfig:
    """观测配置。**密钥不进 `repr`** —— 它会被写进日志。"""

    enabled: bool
    project: str = DEFAULT_PROJECT
    endpoint: str | None = None
    api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls) -> TracingConfig:
        """读环境变量。

        启用判定：显式 `PET_AGENT_TRACING` 优先；未设时「有 key 就启用」。
        """
        api_key = _first_env(_ENV_KEYS)
        flag = (_env(ENV_TRACING) or "").lower()

        if flag in _OFF:
            enabled = False
        elif flag in _ON:
            enabled = True
        else:
            enabled = api_key is not None

        return cls(
            enabled=enabled,
            project=_first_env(_ENV_PROJECTS) or DEFAULT_PROJECT,
            endpoint=_first_env(_ENV_ENDPOINTS),
            api_key=api_key,
        )

    @property
    def usable(self) -> bool:
        """真的能采吗。**开了但没有 key 是配置错误**，不应当被当成"已启用"。"""
        return self.enabled and bool(self.api_key)

    def describe(self) -> dict[str, str]:
        """给 `/healthz` 与日志用。**不含密钥。**"""
        return {
            "langsmith_enabled": str(self.enabled).lower(),
            "langsmith_usable": str(self.usable).lower(),
            "langsmith_project": self.project,
            "langsmith_endpoint": self.endpoint or "(默认)",
        }


def resolve_run_id(trace_id: str) -> uuid.UUID:
    """把 `trace_id` 映射成 LangSmith 的 `run_id`。

    **优先原样使用** —— `trace_id` 默认就是 `uuid4()`，此时二者是同一个标识。
    非 UUID 的客户端自定义 id 才走 `uuid5` 确定性派生（不是随机，也不是映射表）。
    """
    try:
        return uuid.UUID(trace_id)
    except (ValueError, AttributeError, TypeError):
        return uuid.uuid5(_NAMESPACE, str(trace_id))


def build_tracer(config: TracingConfig) -> tuple[Any | None, str | None]:
    """构造 tracer。

    Returns:
        ``(tracer, error)``。未启用时返回 ``(None, None)``；
        启用但构造失败时返回 ``(None, 原因)`` —— **调用方据此把失败暴露出去**，
        而不是让「以为在采数据、其实没有」静默发生。
    """
    if not config.usable:
        if config.enabled and not config.api_key:
            return None, f"{ENV_TRACING} 已开启但缺少 LangSmith API key"
        return None, None
    try:
        # 延迟导入：不开启观测时不该付导入成本，也不该硬依赖它们。
        from langchain_core.tracers import LangChainTracer
        from langsmith import Client

        client = Client(api_url=config.endpoint, api_key=config.api_key)
        return LangChainTracer(project_name=config.project, client=client), None
    except Exception as exc:  # noqa: BLE001 — 观测失败不得让对话失败
        return None, f"LangSmith tracer 构造失败：{type(exc).__name__}: {exc}"


def run_config(
    *,
    trace_id: str,
    session_id: str,
    user_id: str,
    pet_id: str,
    tracer: Any | None = None,
    run_name: str = "pet-agent.turn",
) -> dict[str, Any]:
    """构造 `graph.invoke` 的 config。

    `run_id` 与 `trace_id` 同源（见模块 docstring）；`metadata` 里同时保留
    `trace_id` / `session_id`，于是「全链路可追溯」在 LangSmith 侧也成立 ——
    按 session 或 trace 都能筛出同一条链。
    """
    config: dict[str, Any] = {
        "run_id": resolve_run_id(trace_id),
        "run_name": run_name,
        "metadata": {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "pet_id": pet_id,
        },
        "tags": ["pet-agent"],
    }
    if tracer is not None:
        config["callbacks"] = [tracer]
    return config
