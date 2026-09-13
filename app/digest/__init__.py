"""每日对话总结：提取事实与记忆。

## 这个功能补的是「单轮看不到的东西」

单轮提取（`memory_extractor`）只看当前这一句。每日总结看**一整天**，
因此它能发现单轮原理上发现不了的三类信息：

| 类型 | 例 | 为什么单轮做不到 |
|---|---|---|
| **聚合模式** | 「今天它叫了 6 次」 | 需要跨轮计数 |
| **跨轮矛盾** | 「早上你说它精神好，晚上说没精神」 | 需要同时看到两条 |
| **反复关切** | 「你今天提到吸尘器 3 次」 | 需要频次统计 |

## 流程

```
消息 ──► aggregate（代码）──► DigestStats           ← 计数是事实，由代码产生
                              │
                              ▼
                         LLM 提取候选
                              │
                              ▼
              parse_candidates（**quote 机械校验**）
                    │                    │
                 通过                  未通过
                    │                    │
                    ▼                    ▼
            DigestCandidate        RejectedCandidate
                    │                    │
                    ▼                    └──► 保留在 summary.rejected（**编造现形处**）
            MemoryWriter.decide
                    │
                    ▼
            MemoryWriter.apply  ──► 落库（唯一写入点）
```

## 三个刻意的设计决定

1. **不持久化 `DailySummary` 本身。**
   它是对记忆的一个**视图**。把视图也存进记忆，就有了第二个真相来源，
   两者会漂移。要留档由调用方决定，不由本模块偷偷写。

2. **不因一条坏候选而失败。**
   LLM 返回的枚举值可能是 ``"health"``（不存在）。逐条隔离后，
   坏的那条进 ``rejected``，其余照常——否则一次格式抖动会丢掉一整天的总结。

3. **空对话不调模型。**
   没有消息就是没事发生。此时调用模型只会烧钱并可能让它编点东西出来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, time, timezone
from typing import Any, Protocol

from app.digest.aggregate import aggregate
from app.digest.extract import SYSTEM_PROMPT, build_user_prompt, parse_candidates
from app.llm.base import LLMClient
from app.memory.flywheel import AppliedResult, MemoryWriter
from app.schemas.digest import (
    DailySummary,
    DigestCandidate,
    DigestMessage,
    ExtractionSource,
)
from app.schemas.health import HealthRecord
from app.schemas.memory import EventType, MemoryEvent, MemorySource, MemoryStatus


class HealthSink(Protocol):
    """健康记录的接收方。

    这里用 Protocol 而不是直接依赖 `app.health.HealthWriter`，
    目的是让「对话总结」不绑定到健康模块的具体实现 ——
    它只需要知道「有一个地方能接住健康信号」。
    """

    def admit_digest_candidates(
        self,
        candidates: list[DigestCandidate],
        *,
        user_id: str,
        pet_id: str,
        day: date_type,
        at: datetime | None = None,
    ) -> Any: ...


#: 来源层 → 记忆来源。
#:
#: 这个映射决定了**能否直接生效**：
#: - ``OWNER_RECORD``（主人直述）→ ``USER_OBSERVATION`` → 可 ACTIVE
#: - ``AI_INFERENCE``（系统归纳）→ ``SYSTEM_INFERENCE`` → **只能 PENDING_CONFIRMATION**
#:
#: 第二条由 `app/schemas/memory.py` 的契约强制，不靠这里自觉。
_SOURCE_MAP: dict[ExtractionSource, MemorySource] = {
    ExtractionSource.OWNER_RECORD: MemorySource.USER_OBSERVATION,
    ExtractionSource.AI_INFERENCE: MemorySource.SYSTEM_INFERENCE,
}

#: 来源层 → **初始状态**。
#:
#: 初版实现无条件写 ``ACTIVE``，结果 **构造 MemoryEvent 时就直接抛异常**：
#: 契约规定 ``SYSTEM_INFERENCE`` 不得以 ACTIVE 存在（防自我强化）。
#: 也就是说，一条 AI 归纳的候选会让**整天的总结崩掉**。
#:
#: 契约在这里做对了它该做的事 —— 它在写入之前就把错误拦住了。
#: 但这个案例值得记下：**「事后由路由层修正」的想法是错的**，
#: 因为非法组合在**构造**那一刻就不存在，根本活不到路由层。
_STATUS_MAP: dict[ExtractionSource, MemoryStatus] = {
    ExtractionSource.OWNER_RECORD: MemoryStatus.ACTIVE,
    ExtractionSource.AI_INFERENCE: MemoryStatus.PENDING_CONFIRMATION,
}


@dataclass(frozen=True)
class DailyDigestResult:
    """一天总结的完整结果。"""

    summary: DailySummary
    applied: AppliedResult
    skipped_reason: str | None = None
    """未调用模型时的原因（如「当天没有对话」）。**非 None 时 ``applied`` 为空。**"""

    health_records: list[HealthRecord] = field(default_factory=list)
    """**已落到 health_records 表的健康记录。**

    它们**不在** ``applied`` 里 —— 健康数据另走一张表（数据策略不同）。
    之前这个列表不存在，于是健康信号「显示了但没落地」。
    """

    @property
    def rejected_count(self) -> int:
        return len(self.summary.rejected)


def summarize_day(
    *,
    day: date_type,
    messages: list[DigestMessage],
    writer: MemoryWriter,
    llm: LLMClient,
    user_id: str,
    pet_id: str,
    existing_memories: list[MemoryEvent] | None = None,
    health_sink: HealthSink | None = None,
) -> DailyDigestResult:
    """总结一天的对话，提取并写入记忆。

    Args:
        day: **本地日**。日界由调用方决定——本函数不猜时区。
        messages: 当天的会话消息。
        writer: 记忆写入器（唯一写入点）。
        llm: 文本模型。
        user_id / pet_id: 多租户隔离键。
        existing_memories: 当天已写入的记忆，用于避免重复提取。
        health_sink: 健康记录接收方。**不传时健康信号无处可去，
            缺口会在 ``notes`` 里显式报出（不静默）。**

    Returns:
        结果对象。空对话时 ``skipped_reason`` 非空且不调用模型。
    """
    known = existing_memories or []
    stats = aggregate(day=day, messages=messages, existing_memory_count=len(known))

    if stats.is_empty:
        return _skipped(day, stats, "当天没有对话，无需总结（未调用模型）")

    prompt = build_user_prompt(
        stats=stats,
        messages=messages,
        existing_memories=[m.content for m in known],
    )
    raw = llm.complete(system=SYSTEM_PROMPT, user=prompt, temperature=0.0)

    candidates, rejected = parse_candidates(raw, messages=messages)

    summary = DailySummary(
        date=day,
        stats=stats,
        candidates=candidates,
        rejected=rejected,
    )
    if rejected:
        summary.notes.append(
            f"{len(rejected)} 条候选未通过原文校验，已拒绝且**未写入记忆**。"
            "拒绝原因见 rejected 字段。"
        )
    if not candidates:
        summary.notes.append("本次没有提取到任何可写入的记忆。")

    if not candidates:
        return DailyDigestResult(
            summary=summary,
            applied=AppliedResult(written_memory_ids=[], skipped=[]),
        )

    occurred_at = _occurred_at(day, messages)
    events = [
        _to_event(
            c,
            day=day,
            occurred_at=occurred_at,
            user_id=user_id,
            pet_id=pet_id,
        )
        for c in candidates
    ]

    # 仍走飞轮：准入 → 去重（幂等）→ 冲突消解 → 置信度路由
    decisions = [writer.decide(ev) for ev in events]
    applied = writer.apply(decisions)

    for content, reason in applied.skipped:
        summary.notes.append(f"未写入：{content}（{reason}）")

    # ── 健康信号的去向必须可见 ──
    #
    # 飞轮会拒绝 HEALTH 事件进通用记忆（原因：健康信号另走 health_records，
    # 避免检索时混入）。这是对的。
    #
    # 但**光拒绝不够** —— 必须有接收方，否则健康信号就是「显示了但没落地」。
    # 本参数就是那个接收方；未提供时缺口必须在 notes 里显式报出。
    health_candidates = [c for c in candidates if c.event_type is EventType.HEALTH]
    health_records: list[HealthRecord] = []

    if health_candidates:
        if health_sink is None:
            summary.notes.append(
                f"{len(health_candidates)} 条健康记录**未写入通用记忆**"
                "（健康信号另走 health_records 表，这是设计使然）。"
                "⚠️ **但本次未提供 health_sink** —— "
                "这些记录只出现在日报的健康层，不会进入健康模块。"
            )
        else:
            admission = health_sink.admit_digest_candidates(
                health_candidates,
                user_id=user_id,
                pet_id=pet_id,
                day=day,
                at=_occurred_at(day, messages),
            )
            health_records = list(admission.records)
            summary.notes.append(
                f"{len(health_records)} 条健康记录已写入 health_records"
                "（不进通用记忆，避免检索时混入）。"
            )
            for content, reason in admission.rejected:
                summary.notes.append(f"健康记录未写入：{content}（{reason}）")

    return DailyDigestResult(
        summary=summary,
        applied=applied,
        health_records=health_records,
    )


def _to_event(
    candidate: DigestCandidate,
    *,
    day: date_type,
    occurred_at: datetime,
    user_id: str,
    pet_id: str,
) -> MemoryEvent:
    """把候选转成记忆事件。

    ``layer`` 固定为 ``EPISODE``：**一条当天说的话不足以成为稳定事实。**
    「事实」的晋升由飞轮的 ``is_eligible_for_promotion``（重复出现 + 时间跨度）
    负责，不由总结器自封——今天说「它怕吸尘器」很可能只是今天的事。

    ``status`` **必须在这里就定对**，不能留给路由层：
    ``SYSTEM_INFERENCE + ACTIVE`` 是非法组合，在**构造**时就会被契约拒绝，
    根本活不到路由层。
    """
    return MemoryEvent(
        user_id=user_id,
        pet_id=pet_id,
        event_type=candidate.event_type,
        subject=candidate.subject,
        content=candidate.content,
        polarity=candidate.polarity,
        source=_SOURCE_MAP[candidate.source_layer],
        confidence=candidate.confidence,
        status=_STATUS_MAP[candidate.source_layer],
        occurred_at=occurred_at,
    )


def _occurred_at(day: date_type, messages: list[DigestMessage]) -> datetime:
    """事件时间。

    优先用当天最后一条消息的时间（那才是「事情发生」的时刻）；
    没有时间戳时退回当天 23:59 UTC。

    ``day`` 是**本地日标签**，调用方负责日界；这里只保证时间戳自洽。
    """
    stamps = [m.at for m in messages if m.at is not None]
    if stamps:
        latest = max(stamps)
        if latest.tzinfo is None:
            return latest.replace(tzinfo=timezone.utc)
        return latest
    return datetime.combine(day, time(23, 59, 59), tzinfo=timezone.utc)


def _skipped(day: date_type, stats: object, reason: str) -> DailyDigestResult:
    from app.schemas.digest import DigestStats

    assert isinstance(stats, DigestStats)  # noqa: S101 — 内部调用，类型自证
    summary = DailySummary(date=day, stats=stats, notes=[reason])
    return DailyDigestResult(
        summary=summary,
        applied=AppliedResult(written_memory_ids=[], skipped=[]),
        skipped_reason=reason,
    )
