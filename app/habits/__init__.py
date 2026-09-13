"""习惯：从**主人记录的事件**里聚合出这只猫的稳定模式。

## 为什么习惯值得单独一层

`DESIGN.md` 把「持久属性（喜好 / 习惯 / 禁忌 / 性格）」归到 Profile 层，
但它们在此之前只有一个晋升计数（`support_count >= 3` 且跨度 >= 14 天），
**没有任何聚合视图** —— 系统知道「有 3 条关于吸尘器的记忆」，
但回答不了「它有什么习惯」。

## 为什么它可验证（而「猫的心思」不可验证）

习惯回答的是三个可数的问题：**发生过几次、在什么时候、有多规律**。
三个都能由代码从已存储的事件重算，所以这是本项目里少有的、
可以声称绝对数值的能力（与行为解释的谨慎形成对照）。

## 边界

| 做 | 不做 |
|---|---|
| 数出「观察到 12 次、跨 9 天、早上占 83%」 | 断言「它想让你起床」 |
| 数据不够时说「还不能称之为习惯，缺 X」 | 给一个 0.3 的「弱习惯」分数 |
| 时间分散时说「不能说它一般早上叫」 | 挑一个最多的时段当结论 |

详见 `app/habits/model.py` 与 `app/habits/detect.py`。
"""

from app.habits.detect import HABIT_EVENT_TYPES, detect_habits
from app.habits.model import (
    MIN_DAYS_FOR_HABIT,
    MIN_OBSERVATIONS_FOR_HABIT,
    MIN_SPAN_DAYS_FOR_TREND,
    Habit,
    HabitReport,
    HabitStrength,
    HabitTrend,
)

__all__ = [
    "Habit",
    "HabitReport",
    "HabitStrength",
    "HabitTrend",
    "HABIT_EVENT_TYPES",
    "detect_habits",
    "MIN_OBSERVATIONS_FOR_HABIT",
    "MIN_DAYS_FOR_HABIT",
    "MIN_SPAN_DAYS_FOR_TREND",
]
