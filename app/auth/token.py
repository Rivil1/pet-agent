"""鉴权：HMAC 签名 token。

对应 docs/ARCHITECTURE.md §4.2「P0 最小可行鉴权（**解决 U6**）」。

**为什么需要它**：`DESIGN.md` 决策 D5 是「多宠物 + 多用户全链路隔离」，
而隔离的前提是 ``user_id`` **不可伪造**。若 ``user_id`` 由客户端自报，
隔离主张就不成立 —— 任何人都能读别人的数据。
所以这不是「以后再说」的功能，是隔离主张的**前置条件**。

**为什么不用 JWT / OAuth**：本项目不需要跨服务、不需要第三方登录、
不需要刷新令牌。HMAC 签名 token 覆盖了「不可伪造」这个唯一需求。

token 格式（两段，点号分隔）::

    base64url(user_id.exp_epoch_seconds) . hmac_sha256_hex(secret)

签名覆盖第一段全文，因此 ``user_id`` 与 ``exp`` 都无法被篡改。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

#: 默认有效期（秒）。开发用 30 天。
DEFAULT_TTL_SECONDS = 30 * 24 * 3600


class AuthError(Exception):
    """鉴权失败基类。**对外一律返回 401，不区分原因**。"""


class InvalidToken(AuthError):
    """签名错误、格式错误、或已过期。"""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).hexdigest()


def issue_token(
    user_id: str, *, secret: str, ttl_seconds: int = DEFAULT_TTL_SECONDS, now: int | None = None
) -> str:
    """签发 token。

    Raises:
        ValueError: ``user_id`` 为空，或 ``ttl_seconds`` 非正。
    """
    if not user_id or not user_id.strip():
        raise ValueError("user_id 不得为空")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds 必须为正")

    exp = int(now if now is not None else time.time()) + ttl_seconds
    payload = _b64e(f"{user_id.strip()}.{exp}".encode("utf-8"))
    return f"{payload}.{_sign(payload, secret)}"


def verify_token(token: str, *, secret: str, now: int | None = None) -> str:
    """校验 token，返回 ``user_id``。

    校验顺序（**先签名后过期**）：签名不过就直接拒，避免用未验证的数据做任何判断。

    Raises:
        InvalidToken: 任意校验失败。**不区分原因** —— 区分会泄露信息给攻击者。
    """
    if not token or "." not in token:
        raise InvalidToken("token 格式非法")

    payload, _, signature = token.rpartition(".")
    if not payload or not signature:
        raise InvalidToken("token 格式非法")

    # 恒定时间比较，避免时序侧信道
    if not hmac.compare_digest(_sign(payload, secret), signature):
        raise InvalidToken("签名校验失败")

    try:
        decoded = _b64d(payload).decode("utf-8")
        user_id, _, exp_text = decoded.rpartition(".")
        exp = int(exp_text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidToken("token 载荷非法") from exc

    if not user_id:
        raise InvalidToken("token 载荷缺少 user_id")

    if exp < int(now if now is not None else time.time()):
        raise InvalidToken("token 已过期")

    return user_id


# ─────────────────────────────────────────────────────────────
# FastAPI 集成
# ─────────────────────────────────────────────────────────────


def bearer_token(authorization: str | None) -> str:
    """从 ``Authorization`` 头中取出 Bearer token。

    Raises:
        InvalidToken: 头缺失或格式不对。
    """
    if not authorization:
        raise InvalidToken("缺少 Authorization 头")
    scheme, _, credential = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credential.strip():
        raise InvalidToken("Authorization 头格式应为 'Bearer <token>'")
    return credential.strip()
