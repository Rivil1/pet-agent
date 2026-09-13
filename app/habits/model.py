"""习惯的数据模型。

## 什么是「习惯」（以及为什么它可验证）

`docs/DESIGN.md` D29 把知识对象定为**主人的经验**而不是猫的心思，理由是
「可验证」。习惯正好落在可验证那一侧 —— 它不是「它想要什么」，
而是「**这件事发生过多少次、在什么时候发生、有多规律**」。

这三个问题都可以由代码从已存储的记忆事件里**重算**出来，
所以习惯是这个系统里少有的、可以声称绝对数值的东西。

## 每一个数字都必须可复算（D7：LLM 不产生数字）

`Habit` 的每个字段都能追溯到具体的记忆事件：

| 字段 | 来源 |
|---|---|
| `observations` | 各事件的 `support_count` 之和 |
| `distinct_days` | 事件 `occurred_at` 的**日期去重计数** |
| `time_histogram` | 事件 `occurred_at` 按时段分桶 |
| `first_seen` / `last_seen` | `occurred_at` 的最小/最大值 |
| `regularity` | 由 `time_histogram` 算出的归一化集中度 |

`evidence_ids` 让用户可以逐条核对 —— 一个不能用「哪几条记录支撑它」回答的
习惯，就不该被显示为习惯。

## 「还不构成习惯」必须是一个显式状态，不是一个低分

观察 2 次就说「弱习惯」暗示它**是**一个习惯。所以强度是一个三值枚举，
而不是 0–1 的分数：`INSUFFICIENT` 表示「还不够下结论」，
而它附带**为什么**（缺什么）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from app.schemas import EventType
from app.schemas.timeofday import TimeOfDay

__all__ = [
    "HabitStrength",
    "HabitTrend",
    "Habit",
    "HabitReport",
    "MIN_OBSERVATIONS_FOR_HABIT",
    "MIN_DAYS_FOR_HABIT",
    "MIN_SPAN_DAYS_FOR_TREND",
]


# ─────────────────────────────────────────────────────────────
# 阈值：**每一个都要能说出理由**，否则就是拍脑袋的经验值
# ─────────────────────────────────────────────────────────────

#: 至少观察到这么多次，才谈得上「习惯」。
#:
#: 与 `app/memory/flywheel.py` 的 `PROMOTION_MIN_SUPPORT` 对齐 ——
#: 那一条定的也是 3，理由相同：两次是巧合，三次才勉强算重复。
#: **刻意复用同一个数字**：两处用不同阈值会让「已晋升为长期事实」
#: 与「已形成习惯」这两个说法在一个用户看来互相矛盾。
MIN_OBSERVATIONS_FOR_HABIT = 3

#: 至少要出现在这么多**不同的天**上。
#:
#: 单日出现 5 次是「那天发生了 5 次」，不是习惯。
#: 习惯的必要条件是**跨天重复** —— 这是它与「一次事件」的分界。
MIN_DAYS_FOR_HABIT = 3

#: 判断趋势所需的最小时间跨度（天）。
#:
#: 跨度太短时「最近变多了」只是在描述波动。宁可说 `UNKNOWN`，
#: 也不要给出一个用户会当真的方向。
MIN_SPAN_DAYS_FOR_TREND = 14


class HabitStrength(str, Enum):
    """习惯的成熟度。**三值，不是分数。**"""

    #: 观察充分（次数、天数、跨度都够）→ 可以称为习惯
    ESTABLISHED = "established"
    #: 有重复迹象但还不够下结论
    EMERGING = "emerging"
    #: 观察太少 —— **不是一个弱习惯，而是还不能称之为习惯**
    INSUFFICIENT = "insufficient"


class HabitTrend(str, Enum):
    """趋势。`UNKNOWN` 是一等公民，不是一个兜底。"""

    RISING = "rising"
    FALLING = "falling"
    STEADY = "steady"
    #: 观察窗口太短或样本太少 → **不判断方向**
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Habit:
    """一条习惯。

    frozen：它是**某一次聚合的结果**，不是一个会被就地修改的实体。
    需要更新的习惯会让 `detect_habits()` 重新算出一个新的对象 ——
    那样「报告里的数字」与「用来算它的数据」之间永远是对齐的。
    """

    #: 归一化后的主体（分组键）
    subject: str
    #: 代表性表述 —— 取**出现次数最多**的那条原文，不重写
    content: str
    event_type: EventType

    # ── 可复算的计数 ──
    observations: int
    distinct_days: int
    first_seen: datetime
    last_seen: datetime
    span_days: int

    # ── 时间分布 ──
    time_histogram: dict[TimeOfDay, int] = field(default_factory=dict)
    #: 出现最多的时段。全无时间信息时为 `None`
    dominant_time: TimeOfDay | None = None
    #: 主导时段占比（0–1）。**低占比意味「时间很分散」**，
    #: 这时说「它一般早上叫」是错的。
    time_concentration: float = 0.0
    #: 时间规律性 = 1 − 归一化熵（0–1）。1 = 每天都在同一时段
    regularity: float = 0.0

    strength: HabitStrength = HabitStrength.INSUFFICIENT
    trend: HabitTrend = HabitTrend.UNKNOWN

    #: 支撑这条习惯的记忆 id。**用户可以逐条核对。**
    evidence_ids: tuple[str, ...] = ()
    #: 为什么不能声称更多。**空的 limitations 才是可疑的** ——
    #: 任何聚合都有边界，写不出来通常意味着没想到。
    limitations: tuple[str, ...] = ()

    @property
    def is_habit(self) -> bool:
        """够不够称为「习惯」。**只有这一个判据，不分散到调用方。**"""
        return self.strength is HabitStrength.ESTABLISHED

    @property
    def observations_per_day(self) -> float:
        """平均每天观察到几次。用于跨习惯比较**强度**。

        `+1` 是因为 `span_days` 是**端点差**：从第 1 天到第 1 天跨度是 0，
        但那实际覆盖 1 天。不加 1 会让短跨度的习惯强度虚高，
        而虚高的方向恰好是「把一次事件说成习惯」。
        """
        if self.span_days <= 0:
            # 不需要 `float(...)`：Python 的数值塔里 int 可直接当 float 用
            return self.observations
        return self.observations / (self.span_days + 1)

    def describe_counts(self) -> str:
        """给报告/日志用的计数摘要。**只有数字，不含解释。**"""
        return (
            f"{self.observations} 次 / {self.distinct_days} 天 / "
            f"跨度 {self.span_days} 天"
        )


@dataclass(frozen=True)
class HabitReport:
    """一次习惯聚合的完整产出。

    带上 `considered` / `skipped` 是为了回答「为什么只有这几条习惯」——
    没有它们，用户只看到 2 条，无法知道系统看过多少、排除了什么。
    """

    habits: tuple[Habit, ...] = ()
    #: 参与聚合的记忆事件总数
    considered_events: int = 0
    #: 被排除的事件数
    skipped_events: int = 0
    #: 排除原因 → 条数。**分类而不是一句话**，便于核对。
    skip_reasons: dict[str, int] = field(default_factory=dict)
    #: 全局限制说明（例如「多数记录没有时间戳」）
    limitations: tuple[str, ...] = ()

    def established(self) -> tuple[Habit, ...]:
        return tuple(h for h in self.habits if h.is_habit)

    def emerging(self) -> tuple[Habit, ...]:
        return tuple(h for h in self.habits if h.strength is HabitStrength.EMERGING)
