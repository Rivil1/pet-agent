"""共享 HTTP 传输层：超时、重试、错误翻译。

对应 docs/ARCHITECTURE.md §5.3。

三条实现准则：

1. **只重试可重试的错误**（见 ``errors.RETRYABLE``）。
   对「模型没返回合法 JSON」重试同一个 prompt 是徒劳的 —— 那属于要改 prompt 或降级。
2. **重试上限为 1（LLM）/ 2（其他）**。生成类调用重试会产生不同结果，
   无限重试会同时放大成本与不确定性。
3. **不记录 API Key 与媒体 URL**（`ARCHITECTURE.md` §6.1 L1）。
   错误信息只带状态码与上游返回的简短文本，不带请求体。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.llm.errors import UpstreamError, UpstreamKind

#: 各状态码到错误性质的映射。
_STATUS_TO_KIND: dict[int, UpstreamKind] = {
    400: UpstreamKind.BAD_REQUEST,
    401: UpstreamKind.AUTH,
    403: UpstreamKind.AUTH,
    404: UpstreamKind.BAD_REQUEST,
    422: UpstreamKind.BAD_REQUEST,
    429: UpstreamKind.RATE_LIMITED,
}


def _classify_status(status: int) -> UpstreamKind:
    if status in _STATUS_TO_KIND:
        return _STATUS_TO_KIND[status]
    if status >= 500:
        return UpstreamKind.SERVER_ERROR
    return UpstreamKind.BAD_REQUEST


def _short_error_text(response: httpx.Response) -> str:
    """截断上游错误信息。

    带上一点上游原文便于排错，但**截断**以避免把整段响应（可能含敏感内容）写进日志。
    """
    try:
        payload = response.json()
    except (ValueError, json.JSONDecodeError):
        return (response.text or "")[:200]

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:200]
        if "message" in payload:
            return str(payload["message"])[:200]
    return str(payload)[:200]


def post_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
    max_attempts: int = 1,
    backoff_s: float = 0.8,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """POST JSON 并返回解析后的响应。

    Args:
        max_attempts: **总尝试次数**（1 表示不重试）。
        transport: 测试注入用（``httpx.MockTransport``）。

    Raises:
        UpstreamError: 任意失败。携带 ``kind`` / ``retryable`` 供调用方决策。
    """
    last: UpstreamError | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_s, transport=transport) as client:
                response = client.post(url, json=payload, headers=headers)

            if response.is_success:
                try:
                    data = response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise UpstreamError(
                        UpstreamKind.MALFORMED_RESPONSE,
                        f"响应不是合法 JSON：{exc}",
                        status_code=response.status_code,
                        attempts=attempt,
                    ) from exc
                if not isinstance(data, dict):
                    raise UpstreamError(
                        UpstreamKind.MALFORMED_RESPONSE,
                        f"响应顶层不是对象，而是 {type(data).__name__}",
                        status_code=response.status_code,
                        attempts=attempt,
                    )
                return data

            last = UpstreamError(
                _classify_status(response.status_code),
                _short_error_text(response),
                status_code=response.status_code,
                attempts=attempt,
            )

        except httpx.TimeoutException as exc:
            last = UpstreamError(
                UpstreamKind.TIMEOUT, f"请求超时（{timeout_s}s）", attempts=attempt
            )
            del exc
        except httpx.TransportError as exc:
            # 带上异常类型而不带 URL —— URL 可能含签名参数
            last = UpstreamError(
                UpstreamKind.NETWORK,
                f"网络层失败：{type(exc).__name__}",
                attempts=attempt,
            )

        if last is not None and (not last.retryable or attempt >= max_attempts):
            raise last
        if attempt < max_attempts:
            # 指数退避。同步实现 —— 调用方在 FastAPI 的线程池里运行。
            time.sleep(backoff_s * (2 ** (attempt - 1)))

    # 循环必然在内部 return 或 raise；这行只为让类型检查满意
    raise last or UpstreamError(UpstreamKind.NETWORK, "未知失败")


def bearer_headers(api_key: str) -> dict[str, str]:
    """构造鉴权头。

    **这是唯一接触 API Key 的地方** —— 便于审计「密钥有没有被写进日志」。
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
