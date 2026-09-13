"""从记忆事件检测习惯。**纯代码聚合，不经过任何模型。**

## 为什么这里没有 LLM

`DESIGN.md` D7：**置信度由检索/评分层计算，LLM 不产生数字。**

习惯的全部输出都是数字（次数、天数、时段占比、规律性）。
让模型去「总结这只猫有什么习惯」，它会给出读起来很自然、
但无法核对的说法 —— 而用户没有任何办法分辨
「它每天六点叫你」是从 20 条记录数出来的，还是编的。

所以：**代码数，模型只在最后措辞**（见 `app/habits/answer.py`）。

## 两处刻意的区分

### 1. 「发生时间」与「记录时间」

| 用途 | 用哪个 | 为什么 |
|---|---|---|
| 天数 / 跨度 | `occurred_at`，缺失时回退 `created_at` | 在 N 天里各记过一次，就是 N 天的证据；用记录时间不改变这个事实 |
| **时段分布** | **只用 `occurred_at`** | 记录时间不是发生时间。把「晚上补记」当成「晚上发生」，等于往时间统计里**编造**数据 |

所以时段分布经常比总次数少 —— 那是对的，并且会体现在 `limitations` 里。

### 2. `support_count` 是次数，不是天数

一条记忆被强化 5 次时 `support_count=5`，但它只有一个时间戳。
所以 `observations=5` 而 `distinct_days=1` —— 这两个数**不能互相推导**，
混为一谈会让「说过三次」被当成「连续三天」。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from app.habits.model import (
    MIN_DAYS_FOR_HABIT,
    MIN_OBSERVATIONS_FOR_HABIT,
    MIN_SPAN_DAYS_FOR_TREND,
    Habit,
    HabitReport,
    HabitStrength,
    HabitTrend,
)
from app.schemas import (
    SUBJECT_FALLBACK_PREFIX,
    EventType,
    MemoryEvent,
    MemoryStatus,
)
from app.schemas.timeofday import TimeOfDay, bucket_of

__all__ = ["detect_habits", "HABIT_EVENT_TYPES"]


#: 哪些事件类型可以构成习惯。
#:
#: - `routine`（惯例）：**最典型的习惯** —— 六点叫人起床、固定时间讨食
#: - `preference`（偏好）：稳定的喜好也是习惯的一种（爱吃三文鱼）
#: - `behavior`（行为）：反复出现的行为模式
#:
#: **刻意不含 `context` 与 `health`**：
#: - `context` 是「当时的情形」，是情境不是习惯
#: - `health` 走 `health_records` 表（数据策略不同，见 `app/health/store.py`）。
#:   把健康信号混进「习惯」会让「它经常吐」看起来像一种生活方式。
HABIT_EVENT_TYPES: frozenset[EventType] = frozenset(
    {EventType.ROUTINE, EventType.PREFERENCE, EventType.BEHAVIOR}
)

#: 参与聚合的记忆状态。
#:
#: `SUPERSEDED` 的不能算（它已经被新事实取代），
#: `REJECTED` 的更不能算（主人明确否定了）。
#: `PENDING_CONFIRMATION` **算** —— 它是「已观察到但还没确认」，
#: 而习惯检测关心的是「发生过几次」，不是「确认过几次」。
#: （这与「未确认的样本不构成案例推理证据」是两件事：
#: 那里要的是**标签可信**，这里要的只是**事件发生过**。）
_COUNTED_STATUS: frozenset[MemoryStatus] = frozenset(
    {MemoryStatus.ACTIVE, MemoryStatus.PENDING_CONFIRMATION}
)


def _anchor_time(event: MemoryEvent) -> datetime | None:
    """用于**日期/跨度**的时间。缺 `occurred_at` 时回退 `created_at`。"""
    return event.occurred_at or event.created_at


def _timing_time(event: MemoryEvent) -> datetime | None:
    """用于**时段分布**的时间。**只接受 `occurred_at`。**

    回退到 `created_at` 会认认真真地算出一个错误的时段分布：
    主人晚上补记「它今天早上六点叫我」，那条就被算成「晚上」——
    而报告里看不出任何异常。
    """
    return event.occurred_at


def _normalized_entropy(counts: Sequence[int]) -> float:
    """归一化熵（0–1）。0 = 全部集中在同一类，1 = 完全均匀。

    样本数 ≤ 1 时返回 0（无信息，视为完全集中）——
    调用方会因为它没有时间信息而根本不计算规律性。
    """
    total = sum(counts)
    if total <= 1:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(len(probs))
    return h / h_max if h_max > 0 else 0.0


def _utc(dt: datetime) -> datetime:
    """统一成 aware UTC，便于比较。"""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _evaluate_strength(
    *, observations: int, distinct_days: int, span_days: int
) -> tuple[HabitStrength, list[str]]:
    """判定成熟度，并**逐条说明缺什么**。

    返回的 `limitations` 是给用户看的 —— 「观察到 5 次但只分布在 2 天」
    比一个笼统的「还不够」有用得多。
    """
    missing: list[str] = []
    if observations < MIN_OBSERVATIONS_FOR_HABIT:
        missing.append(
            f"只观察到 {observations} 次，不足 {MIN_OBSERVATIONS_FOR_HABIT} 次"
        )
    if distinct_days < MIN_DAYS_FOR_HABIT:
        missing.append(
            f"只在 {distinct_days} 天出现过，不足 {MIN_DAYS_FOR_HABIT} 天 —— "
            f"单日多次不算习惯，习惯的必要条件是跨天重复"
        )

    if not missing:
        return HabitStrength.ESTABLISHED, []

    # 有重复迹象（次数够或天数够其一）→ emerging；两个都不够 → insufficient
    if observations >= 2 and (observations >= MIN_OBSERVATIONS_FOR_HABIT or distinct_days >= 2):
        return HabitStrength.EMERGING, missing
    return HabitStrength.INSUFFICIENT, missing


def _evaluate_trend(
    dated: Sequence[tuple[datetime, int]], *, span_days: int, observations: int
) -> tuple[HabitTrend, list[str]]:
    """判断趋势。**窗口不足时返回 `UNKNOWN`，不猜方向。**

    `dated` 是 `(锚点时间, support_count)` 的序列。
    """
    notes: list[str] = []
    if span_days < MIN_SPAN_DAYS_FOR_TREND:
        notes.append(
            f"观察跨度只有 {span_days} 天（不足 {MIN_SPAN_DAYS_FOR_TREND} 天），"
            f"不判断趋势 —— 窗口太短时「最近变多了」只是在描述波动"
        )
        return HabitTrend.UNKNOWN, notes

    if not dated:
        return HabitTrend.UNKNOWN, ["没有可用的时间信息，无法判断趋势"]

    times = [_utc(t) for t, _ in dated]
    mid = min(times) + (max(times) - min(times)) / 2

    early = sum(n for t, n in dated if _utc(t) < mid)
    late = sum(n for t, n in dated if _utc(t) >= mid)

    # 半数比较，而非「多一次就算上升」：后者在样本少时噪声极大
    if late > early * 1.5:
        return HabitTrend.RISING, notes
    if early > late * 1.5:
        return HabitTrend.FALLING, notes
    return HabitTrend.STEADY, notes


def detect_habits(
    events: Iterable[MemoryEvent],
    *,
    now: datetime | None = None,
    min_observations: int = MIN_OBSERVATIONS_FOR_HABIT,
) -> HabitReport:
    """把记忆事件聚合成习惯。

    Args:
        events: 该宠物（同一租户）的全部记忆事件。**调用方负责租户过滤** ——
            本函数不做租户检查，因为它拿到的是一个已经过滤好的序列；
            在这里再过滤一次会给人「不过滤也安全」的错觉。
        now: 当前时间（测试注入用）。
        min_observations: 覆盖最少观察次数阈值。

    纯函数：同样的输入与 `now` 一定产出同样的结果。
    """
    del now  # 刻意不用「现在」：习惯全部由**已记录的事件**得出，
    # 引入「现在」会让同一个数据集在不同时刻算出不同结果。
    # 需要「最近」的概念时，用事件自身的时间跨度表达。

    groups: dict[tuple[str, EventType], list[MemoryEvent]] = defaultdict(list)
    skip_reasons: Counter[str] = Counter()
    considered = 0

    for event in events:
        considered += 1
        if event.event_type not in HABIT_EVENT_TYPES:
            skip_reasons[f"事件类型不构成习惯（{event.event_type.value}）"] += 1
            continue
        if event.status not in _COUNTED_STATUS:
            skip_reasons[f"状态不计入（{event.status.value}）"] += 1
            continue
        subject = (event.subject or "").strip()
        if not subject:
            skip_reasons["没有主体（subject），无法归组"] += 1
            continue
        groups[(subject, event.event_type)].append(event)

    habits: list[Habit] = []
    global_notes: list[str] = []
    events_without_occurred = 0
    ungrouped_events = 0

    for (subject, event_type), group in groups.items():
        observations = sum(max(0, e.support_count) for e in group)

        # ── 锚点时间（可回退到 created_at）──
        anchors = [t for e in group if (t := _anchor_time(e)) is not None]
        if not anchors:
            # 理论上不可能（created_at 必填），但不做假设
            skip_reasons["没有任何时间信息"] += len(group)
            continue
        anchors_utc = [_utc(t) for t in anchors]
        first_seen, last_seen = min(anchors_utc), max(anchors_utc)
        span_days = (last_seen - first_seen).days
        distinct_days = len({t.date() for t in anchors_utc})

        # ── 时段分布：**只用 occurred_at** ──
        timing = [t for e in group if (t := _timing_time(e)) is not None]
        untimed = len(group) - len(timing)
        histogram: dict[TimeOfDay, int] = defaultdict(int)
        for t in timing:
            histogram[bucket_of(_utc(t))] += 1

        concentration = 0.0
        dominant: TimeOfDay | None = None
        regularity = 0.0
        if histogram:
            total_timed = sum(histogram.values())
            dominant, top = max(histogram.items(), key=lambda kv: kv[1])
            concentration = top / total_timed
            regularity = 1.0 - _normalized_entropy(list(histogram.values()))

        strength, limitations = _evaluate_strength(
            observations=observations,
            distinct_days=distinct_days,
            span_days=span_days,
        )
        # ⚠️ **主体是猜不出时的回退值时，要说出来。**
        #
        # `_guess_subject` 认不出内容时返回 `misc:<哈希>`。哈希不同
        # 意味着**内容相近的两条也不会归为一组** —— 于是同一件事
        # 分三次说，会在报告里变成三条「记录还太少」。
        #
        # 那不是「记录真的不够」，而是「系统没认出它们是同一件事」——
        # 两者对用户的含义完全不同，不能都不说。
        if subject.startswith(SUBJECT_FALLBACK_PREFIX):
            ungrouped_events += len(group)
            limitations.append(
                "这条记录的主体无法从内容识别，它不会与其他措辞不同的记录"
                "归为一组 —— 所以同一个习惯换个说法说多次，也不会被累加"
            )

        if observations < min_observations and observations >= min_observations:
            limitations = list(limitations)

        if untimed:
            # 时段那一段会自己说明（见下面 histogram 分支）——
            # 这里只累计全局计数，不重复添加同一条限制。
            events_without_occurred += untimed

        trend, trend_notes = _evaluate_trend(
            [(t, max(0, e.support_count)) for e in group if (t := _anchor_time(e))],
            span_days=span_days,
            observations=observations,
        )
        limitations.extend(trend_notes)

        if not histogram:
            limitations.append(
                "没有任何记录带 occurred_at，所以无法给出时段分布 —— "
                "记录时间不是发生时间，用它算时段会编造数据"
            )
        else:
            # 有直方图但**部分**观察没参与时，才需要单说一句。
            # 全部都没参与的情况下，上面那句已经说完了 ——
            # 两条限制说同一件事会让用户以为存在两个不同的问题。
            if untimed:
                events_without_occurred += untimed
                limitations.append(
                    f"有 {untimed} 条观察没有 occurred_at，未参与时段统计"
                    f"（记录时间不是发生时间，用它算时段会编造数据）"
                )
            if concentration < 0.5:
                limitations.append(
                    f"时间很分散（最集中的时段只占 {concentration * 100:.0f}%），"
                    f"不能说「它一般在{_display(dominant)}做这件事」"
                )

        # 代表表述：取**原文**里最长的（信息最多），不重写
        representative = max((e.content for e in group), key=len)
        evidence = tuple(
            e.memory_id for e in group if e.memory_id is not None
        )

        habits.append(
            Habit(
                subject=subject,
                content=representative,
                event_type=event_type,
                observations=observations,
                distinct_days=distinct_days,
                first_seen=first_seen,
                last_seen=last_seen,
                span_days=span_days,
                time_histogram=dict(histogram),
                dominant_time=dominant,
                time_concentration=round(concentration, 4),
                regularity=round(regularity, 4),
                strength=strength,
                trend=trend,
                evidence_ids=evidence,
                limitations=tuple(limitations),
            )
        )

    if events_without_occurred:
        global_notes.append(
            f"{events_without_occurred} 条记录没有 occurred_at，"
            f"它们参与次数与天数统计，但**不参与时段分布**"
        )

    if ungrouped_events:
        global_notes.append(
            f"{ungrouped_events} 条记录的主体无法从内容识别（多为非常见说法）。"
            f"它们**不会互相归组** —— 因此同一件事用不同措辞分多次说，"
            f"也不会被累加成习惯。这是当前主体识别的能力边界"
        )

    # 排序：成熟的在前，其次次数多的，最后按主体名（保证可复现）
    habits.sort(
        key=lambda h: (
            -{HabitStrength.ESTABLISHED: 2, HabitStrength.EMERGING: 1}.get(h.strength, 0),
            -h.observations,
            h.subject,
        )
    )

    return HabitReport(
        habits=tuple(habits),
        considered_events=considered,
        skipped_events=sum(skip_reasons.values()),
        skip_reasons=dict(skip_reasons),
        limitations=tuple(global_notes),
    )


def _display(slot: TimeOfDay | None) -> str:
    from app.schemas.timeofday import time_display

    return time_display(slot) if slot is not None else "该时段"


def active_since(hours: int, *, now: datetime) -> datetime:
    """工具：`now` 往前 N 小时。用于「最近」类查询。

    放在这里而不是让调用方各写各的：时间窗口的边界算错会让
    「最近一周」变成「最近 8 天」，而报告里看不出来。
    """
    return now - timedelta(hours=hours)
