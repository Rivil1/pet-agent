"""会话消息契约。

## 为什么需要它

`app/digest.summarize_day()` 需要**当天的对话消息**才能做总结。
在此之前只有 `MemoryEvent`（提取后的事实），而那是**总结的产物**，
不能拿来当总结的输入 —— 循环了。

所以补上原始消息的存储。

## 与 `DigestMessage` 的关系

`DigestMessage` 是**摘要流水线的输入视图**（只有角色与内容），
而 `SessionMessage` 是**持久化实体**（带租户、时间与 ID）。

`as_digest_message()` 负责两者的转换 ——
这样摘要流水线不需要知道存储层的形状。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.digest import DigestMessage


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionMessage(BaseModel):
    """一条会话消息。"""

    message_id: str | None = None
    user_id: str
    pet_id: str = Field(description="多租户隔离键。检索必须按此过滤。")
    session_id: str | None = Field(
        default=None,
        description=(
            "会话标识。**同一轮对话的所有消息共享它** —— "
            "会话记忆注入按它取回最近若干轮（`DESIGN.md` §3.5 的 Session 层）。\n\n"
            "可空只是为了兼容历史数据；新写入由 API 层保证一定带值。"
        ),
    )
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    at: datetime = Field(default_factory=_now)

    def with_id(self) -> SessionMessage:
        if self.message_id:
            return self
        return self.model_copy(update={"message_id": f"msg-{uuid.uuid4().hex[:12]}"})

    def as_digest_message(self) -> DigestMessage:
        """转成摘要流水线的输入视图。"""
        return DigestMessage(role=self.role, content=self.content, at=self.at)
