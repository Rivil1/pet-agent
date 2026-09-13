"""时段划分：**全项目唯一的边界定义**。

## 为什么必须只有一处

「凌晨 5 点算早上还是夜里」这类问题，如果两个模块各写一份判断，
就会在某个边界上悄悄不一致 —— 而那种不一致是静默的：
故事说「早上六点叫醒你」，习惯统计说「主要在夜里」，
两个输出都看起来正常，用户不知道该信哪个。

所以边界只在这里定义一次，其他地方一律引用。

## 「时间未知」由各调用方自己决定，不在这里兜底

这个模块**不接受** `None`，因为「不知道该归哪个时段」的处理是**按场景不同**的：

| 调用方 | 时间未知时 | 为什么 |
|---|---|---|
| 故事分段 | 归入 `EVENING` | 它要的是一天的**叙事分段**，缺一段就不完整；晚上是一天的中位，且不猜具体时段 |
| 习惯统计 | **排除该条** | 它要的是**时间分布的统计量**。把未知塞进某个桶，等于往统计里掺假数据 |

如果在公共层给一个默认值，两个场景就都被迫接受同一个错误的取舍。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

__all__ = ["TimeOfDay", "HOUR_BUCKETS", "bucket_of", "time_display"]


class TimeOfDay(str, Enum):
    """一天中的时段。**顺序即时间顺序**（用于报告里稳定排序）。"""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


#: 时段边界。**改这里就是改全项目的口径。**
#:
#: 覆盖 5:00–22:59；其余（23:00–4:59）归 `NIGHT`。
#: 没有把边界做成配置项：它是一个**共同约定**，而不是一台需要调的参数 ——
#: 可配置会让「昨天算早上、今天算夜里」这种事变成可能。
HOUR_BUCKETS: tuple[tuple[range, TimeOfDay], ...] = (
    (range(5, 12), TimeOfDay.MORNING),
    (range(12, 18), TimeOfDay.AFTERNOON),
    (range(18, 23), TimeOfDay.EVENING),
)

#: 夜间的小时范围（写出来是为了让「没被上面覆盖的那些」可见）
NIGHT_HOURS = range(23, 24)
NIGHT_HOURS_EARLY = range(0, 5)


_TIME_DISPLAY: dict[TimeOfDay, str] = {
    TimeOfDay.MORNING: "早上",
    TimeOfDay.AFTERNOON: "下午",
    TimeOfDay.EVENING: "晚上",
    TimeOfDay.NIGHT: "夜里",
}


def bucket_of(moment: datetime) -> TimeOfDay:
    """`datetime` → 时段。**不接受 `None`**（见模块 docstring）。"""
    hour = moment.hour
    for hours, slot in HOUR_BUCKETS:
        if hour in hours:
            return slot
    return TimeOfDay.NIGHT


def time_display(at: TimeOfDay) -> str:
    return _TIME_DISPLAY[at]
