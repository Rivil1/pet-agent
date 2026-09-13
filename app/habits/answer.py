"""把习惯聚合渲染成用户能读的文本。

## 唯一的硬约束：**数字只能来自代码**

`DESIGN.md` D7 与 D42：模型不得推断因果、不得产生数字。

习惯的每一个数字（次数 / 天数 / 占比 / 规律性）都已经由
`app/habits/detect.py` 数出来了。本模块只做两件事：

1. **按模板措辞** —— 把数字放进句子里
2. **拒绝超出数据的问题** —— 问的事情没有记录时，说「没有记录」，
   而不是从已有数字里推一个听起来合理的答案

所以这里**没有** `llm.complete(...)`。不是因为不该用模型润色，
而是因为一旦模型能改数字，那些数字就不再可复算 ——
而可复算正是习惯这一层唯一的价值来源。

## 「不能说」比「说得漂亮」重要

三种情况必须在文本里明确说出来：

| 情况 | 文本必须说 |
|---|---|
| 观察不足 | 「还不能称之为习惯」，并给出**缺什么** |
| 时间分散 | 「不能说它一般…」，并给出集中度 |
| 时间信息缺失 | 「无法给出时段分布」，并说明为什么（记录时间 ≠ 发生时间） |
"""

from __future__ import annotations

from typing import Iterable, Sequence

from app.habits.model import (
    MIN_DAYS_FOR_HABIT,
    Habit,
    HabitReport,
    HabitStrength,
    HabitTrend,
)
from app.schemas.timeofday import time_display

__all__ = [
    "render_habit_report",
    "render_habit",
    "habit_facts",
    "answer_habit_question",
    "TREND_DISPLAY",
]


TREND_DISPLAY: dict[HabitTrend, str] = {
    HabitTrend.RISING: "最近变多了",
    HabitTrend.FALLING: "最近变少了",
    HabitTrend.STEADY: "一直比较稳定",
    HabitTrend.UNKNOWN: "还看不出趋势",
}

_STRENGTH_DISPLAY: dict[HabitStrength, str] = {
    HabitStrength.ESTABLISHED: "已形成",
    HabitStrength.EMERGING: "刚有苗头",
    HabitStrength.INSUFFICIENT: "记录太少",
}


def render_habit(habit: Habit, *, index: int | None = None) -> str:
    """渲染一条习惯。**所有数字都直接来自 `habit`，不做任何换算。**"""
    prefix = f"{index}. " if index is not None else ""
    lines: list[str] = [f"{prefix}{habit.content}"]

    # ── 计数（可复算）──
    counts = f"      观察到 {habit.describe_counts()}"
    lines.append(counts)

    # ── 时段（只在有数据时给）──
    if habit.dominant_time is not None and habit.time_concentration >= 0.5:
        lines.append(
            f"      时间上集中在{time_display(habit.dominant_time)}"
            f"（占 {habit.time_concentration * 100:.0f}%）"
        )
    elif habit.time_histogram:
        dist = "、".join(
            f"{time_display(k)}{v}次"
            for k, v in sorted(
                habit.time_histogram.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        lines.append(f"      时间分布较散：{dist}")

    # ── 趋势（只有窗口够长时才有意义）──
    lines.append(f"      {TREND_DISPLAY[habit.trend]}")

    # ── 成熟度：不够就明说，不给一个「弱习惯」的分数 ──
    if habit.strength is not HabitStrength.ESTABLISHED:
        lines.append(f"      ⚠️ {_STRENGTH_DISPLAY[habit.strength]}：还不能称之为习惯")
        for note in habit.limitations:
            lines.append(f"         · {note}")

    # ── 可核对 ──
    if habit.evidence_ids:
        lines.append(f"      记录 {len(habit.evidence_ids)} 条，可逐条核对")

    return "\n".join(lines)


def render_habit_report(report: HabitReport, *, limit: int = 8) -> str:
    """渲染整份报告。"""
    if not report.habits:
        return (
            f"还没有足够的记录来判断习惯。\n"
            f"（看过 {report.considered_events} 条记录，"
            f"其中 {report.skipped_events} 条不参与习惯统计）"
        )

    parts: list[str] = []
    established = report.established()
    emerging = report.emerging()
    insufficient = tuple(
        h for h in report.habits if h.strength is HabitStrength.INSUFFICIENT
    )

    if established:
        parts.append(f"**已形成的习惯（{len(established)} 条）**\n")
        for i, h in enumerate(established[:limit], 1):
            parts.append(render_habit(h, index=i))
    else:
        parts.append(
            "**还没有形成可确认的习惯** —— "
            "下面的记录有重复迹象，但还没达到「跨天重复」的门槛。\n"
        )

    if emerging:
        parts.append(f"\n**刚有苗头（{len(emerging)} 条）**\n")
        for h in emerging[:limit]:
            parts.append(render_habit(h))

    # ⚠️ **观察不足的也要列出来。**
    #
    # 初版只显示 established + emerging，于是只观察过 1 次的那些
    # **从报告里完全消失** —— 用户看不到任何东西，也无法分辨
    # 「系统没见过这条记录」与「见过了但不够下结论」。
    # 后者是一个**应该被告诉**的事实。
    if insufficient:
        parts.append(f"\n**记录还太少（{len(insufficient)} 条，不足以判断）**\n")
        for h in insufficient[:limit]:
            parts.append(f"  · {h.content}（{h.observations} 次 / {h.distinct_days} 天）")

    # ── 排除说明：让「为什么只有这几条」可回答 ──
    if report.skip_reasons:
        parts.append("\n**未参与统计的记录**")
        for reason, n in sorted(report.skip_reasons.items(), key=lambda kv: -kv[1]):
            parts.append(f"  · {reason}：{n} 条")

    if report.limitations:
        parts.append("\n**说明**")
        for note in report.limitations:
            parts.append(f"  · {note}")

    return "\n".join(parts)


def habit_facts(report: HabitReport, *, limit: int = 10) -> list[str]:
    """把已形成的习惯摊平成一串**可核对的事实句**。

    用途：喂给下游（故事 / 日报 / LLM 措辞）。每一句都只包含
    代码算出的数字，模型即使引用也不会说错 —— 它没有可编造的空间。

    只输出 `ESTABLISHED` 的：把「刚有苗头」的也混进去，
    会让下游把两件事当成同一件，而它们的证据强度差着一个数量级。
    """
    facts: list[str] = []
    for h in report.established()[:limit]:
        when = (
            f"，多在{time_display(h.dominant_time)}"
            if h.dominant_time is not None and h.time_concentration >= 0.5
            else ""
        )
        facts.append(
            f"{h.content}（观察到 {h.observations} 次、跨 {h.distinct_days} 天{when}）"
        )
    return facts


# ─────────────────────────────────────────────────────────────
# 问答：只回答能从已算出的数字里回答的问题
# ─────────────────────────────────────────────────────────────

#: 问题关键词 → 命中的习惯主体/内容必须含有的词。
#:
#: 用关键词表而不是语义匹配：习惯数量是个位数，
#: 而**错配的代价**是把 A 习惯的答案说成 B 的 —— 用户无法察觉。
#: 关键词匹配错了至少是「答不出来」，而不是「答错」。
_QUESTION_TOPICS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("吃", "饭", "食", "罐头", "喂"), ("吃", "饭", "食", "罐头", "粮")),
    (("叫", "吵", "喵", "唤"), ("叫", "喵", "唤", "吵")),
    (("睡", "觉", "躺"), ("睡", "觉", "躺", "窝")),
    (("玩", "玩具", "逗"), ("玩", "玩具", "逗")),
    (("躲", "怕", "害怕", "紧张"), ("躲", "怕", "害怕", "紧张", "藏")),
    (("上厕所", "猫砂", "尿", "排便"), ("砂", "尿", "便", "厕所")),
)


def answer_habit_question(report: HabitReport, question: str) -> str | None:
    """回答一个习惯问题。

    Returns:
        回答文本；**无法从已有数据回答时返回 `None`**（调用方据此走「没有记录」）。

    ## 为什么可能返回 `None` 而不是硬答

    习惯这一层只有「次数 / 天数 / 时段」三类事实。
    问「它为什么这么做」时，**数据里没有答案** ——
    那种问题属于行为解释（有它自己的一套证据链），不属于习惯。

    返回 `None` 让调用方明确地说「没有相关记录」，
    与 `response_guard` 的 fail-closed 取向一致。
    """
    q = question.strip()
    if not q:
        return None

    established = report.established()

    # ── 因果类问题：**必须显式拒绝，不能拿观察记录充数** ──
    #
    # 「它为什么怕吸尘器」与「它怕吸尘器吗」看起来只差一个字，
    # 但前者要的是**原因**，而习惯数据里只有次数/天数/时段。
    #
    # 初版没区分这两者：问到「为什么」时它会返回「观察到 2 次」——
    # 那是一个**看起来像答案的东西**，而用户会以为系统回答了问题。
    # 这与 D42（模型不得推断因果）同源：因果必须由主人提供。
    if any(k in q for k in ("为什么", "为啥", "原因", "怎么会", "咋回事")):
        topic = _match_topic(report, q)
        head = "习惯记录里没有原因。"
        if topic:
            # 有相关观察就一并给出 —— 那部分是**有据的**
            return (
                f"{head}我能说的是观察到的情况：\n"
                + "\n".join(render_habit(h) for h in topic[:2])
                + "\n\n（原因需要你自己补充 —— 系统不会推断因果）"
            )
        return f"{head}系统不会推断因果，这件事需要你自己补充。"

    # ── 总览类问题 ──
    if any(k in q for k in ("什么习惯", "哪些习惯", "有什么习惯", "习惯是")):
        if not established:
            return (
                f"目前还没有已确认的习惯。"
                f"（看过 {report.considered_events} 条记录，"
                f"还没有任何一类达到「跨 {MIN_DAYS_FOR_HABIT} 天重复」的门槛）"
            )
        return "已形成的习惯：\n" + "\n".join(habit_facts(report))

    # ── 趋势类问题 ──
    if any(k in q for k in ("最近", "变多", "变少", "越来越", "比上个月", "比以前")):
        known = [h for h in established if h.trend is not HabitTrend.UNKNOWN]
        if not known:
            return (
                "还判断不了趋势 —— 观察窗口太短。"
                "趋势需要至少两周的记录，否则「最近变多了」只是在描述波动。"
            )
        return "\n".join(
            f"{h.content}：{TREND_DISPLAY[h.trend]}"
            f"（{h.describe_counts()}）"
            for h in known
        )

    # ── 主题类问题 ──
    hits = _match_topic(report, q)
    if hits is None:
        return None
    if not hits:
        # 关键词对上了但没记录 → 明确说没有，**不从别的习惯推**
        return None
    return "\n".join(render_habit(h) for h in hits[:3])


def _match_topic(report: HabitReport, question: str) -> list[Habit] | None:
    """按关键词找相关习惯。

    Returns:
        - `None`：问题不属于任何已知主题
        - `[]`：主题对上了但**没有记录**（调用方应说「没有」）
        - 非空列表：命中的习惯

    三态是必需的：把「主题不对」与「主题对但没记录」压成一个空列表，
    调用方就无法区分「这个问题我不管」与「这件事我没记录」——
    而后者才是应该对用户说的。
    """
    for q_words, match_words in _QUESTION_TOPICS:
        if not any(w in question for w in q_words):
            continue
        return [
            h
            for h in report.habits  # 含 emerging：用户可以追问「有苗头吗」
            if any(w in h.content for w in match_words)
            or any(w in h.subject for w in match_words)
        ]
    return None
