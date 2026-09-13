"""可观测性契约：节点错误与 trace。

`AgentState` 中这两个字段使用 ``Annotated[..., operator.add]`` 归约，
使并行分支（尤其是 planner 的 fan-out）的写入能正确合并。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ErrorKind(str, Enum):
    TIMEOUT = "timeout"
    UPSTREAM_ERROR = "upstream_error"
    """模型 / ASR / TTS 等服务不可用"""

    INVALID_OUTPUT = "invalid_output"
    """结构化输出不符合契约"""

    VALIDATION_FAILED = "validation_failed"
    GUARD_REJECTED = "guard_rejected"
    DATA_MISSING = "data_missing"
    DEPENDENCY_FAILED = "dependency_failed"
    """前置子任务失败导致本任务无法执行"""

    INTERNAL = "internal"


class ErrorSeverity(str, Enum):
    FATAL = "fatal"
    """整个请求失败"""

    DEGRADED = "degraded"
    """完成但能力受损，**必须用户可见**"""

    SILENT = "silent"
    """仅影响体验，用户无需感知（如 TTS 失败）"""


class NodeError(BaseModel):
    """一次节点级错误。"""

    node: str
    kind: ErrorKind
    severity: ErrorSeverity
    message: str
    task_id: str | None = Field(
        default=None, description="若由 planner 派发的子任务产生，记录 task_id"
    )
    retryable: bool = False
    detail: dict | None = None
    occurred_at: datetime = Field(default_factory=_now)

    @property
    def is_user_visible(self) -> bool:
        """正确性相关的降级必须可见；体验相关的可静默（docs/01 §6）。"""
        return self.severity in (ErrorSeverity.FATAL, ErrorSeverity.DEGRADED)


class NodeTrace(BaseModel):
    """一次节点执行的 trace。

    保留这些信息的目的：任何一个结论都能回答「当时为什么这么判断」。
    """

    node: str
    task_id: str | None = None
    wave: int | None = Field(
        default=None, description="planner 派发的子任务所属 wave，用于分析并行收益"
    )
    started_at: datetime = Field(default_factory=_now)
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    degraded: bool = False
    decision: str | None = Field(
        default=None,
        description="该节点做出的关键决策摘要，如 \"intent=compound conf=0.81\"",
    )

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)
