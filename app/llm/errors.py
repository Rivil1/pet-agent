"""上游服务错误。

对应 docs/ARCHITECTURE.md §5.3「超时、重试、降级矩阵」与 §4.3 错误模型。

**为什么要单独定义而不是直接抛 httpx 异常**：
节点需要根据「可重试 / 不可重试」「是否配额问题」做不同的降级决策，
而 httpx 的异常层级不携带这些语义。
"""

from __future__ import annotations

from enum import Enum


class UpstreamKind(str, Enum):
    """失败性质。决定是否重试以及降级方式。"""

    TIMEOUT = "timeout"
    """超时。可重试。"""

    RATE_LIMITED = "rate_limited"
    """限流/配额。可重试（退避更久）。"""

    AUTH = "auth"
    """鉴权失败。**不可重试** —— 重试只会浪费配额。"""

    BAD_REQUEST = "bad_request"
    """请求本身非法。**不可重试** —— 重试结果一样。"""

    SERVER_ERROR = "server_error"
    """上游 5xx。可重试。"""

    MALFORMED_RESPONSE = "malformed_response"
    """响应结构不符合预期（如模型没返回合法 JSON）。可重试一次。"""

    NETWORK = "network"
    """网络层失败。可重试。"""


#: 可重试的错误性质。
RETRYABLE: frozenset[UpstreamKind] = frozenset(
    {
        UpstreamKind.TIMEOUT,
        UpstreamKind.RATE_LIMITED,
        UpstreamKind.SERVER_ERROR,
        UpstreamKind.NETWORK,
    }
)
"""注意 ``MALFORMED_RESPONSE`` **不在此列**：模型输出不合 JSON 时，
重试同一个 prompt 大概率还是不合 —— 那属于要改 prompt 或降级的情况。
把它标成可重试会掩盖真实问题。"""


class UpstreamError(RuntimeError):
    """上游服务调用失败。"""

    def __init__(
        self,
        kind: UpstreamKind,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code
        self.attempts = attempts

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    def __str__(self) -> str:
        parts = [f"[{self.kind.value}]", super().__str__()]
        if self.status_code is not None:
            parts.append(f"(HTTP {self.status_code})")
        if self.attempts > 1:
            parts.append(f"after {self.attempts} attempts")
        return " ".join(parts)


class MissingCredential(RuntimeError):
    """缺少 API Key。**这是配置错误，不是上游故障** —— 不应被当成可降级的运行时错误。"""
