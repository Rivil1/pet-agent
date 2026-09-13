"""习惯检测与问答。

## 本文件重点覆盖四处「容易静默出错」的地方

1. **「发生时间」与「记录时间」不能混用**
   把「晚上补记」当成「晚上发生」，会算出一个**看起来完全正常的**
   错误时段分布。所以时段统计只能用 `occurred_at`；
   而天数/跨度可以回退 `created_at`（在 N 天各记过一次就是 N 天证据）。

2. **`support_count` 是次数，不是天数**
   一条记忆被强化 5 次时 `support_count=5`，但它只有一个时间戳。
   混为一谈会把「说过三次」当成「连续三天」。

3. **「还不够」必须是显式状态，不是一个低分**
   观察 2 次说「弱习惯」暗示它**是**一个习惯。

4. **因果问题必须显式拒绝**
   「它为什么怕吸尘器」与「它怕吸尘器吗」只差一个字，
   但前者要的是**原因**，而习惯数据里只有次数/天数/时段。
   拿观察记录回答因果问题，是一个**看起来像答案的东西**。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.habits import (
    HABIT_EVENT_TYPES,
    HabitStrength,
    HabitTrend,
    detect_habits,
)
from app.habits.answer import (
    answer_habit_question,
    habit_facts,
    render_habit_report,
)
from app.habits.model import (
    MIN_DAYS_FOR_HABIT,
    MIN_OBSERVATIONS_FOR_HABIT,
    MIN_SPAN_DAYS_FOR_TREND,
)
from app.schemas import (
    EventType,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    Polarity,
)

NOW = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)


def make_event(
    *,
    subject: str = "wake_up",
    content: str = "六点叫我起床",
    days_ago: int = 0,
    hour: int = 6,
    event_type: EventType = EventType.ROUTINE,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    support_count: int = 1,
    occurred_at: datetime | None = ...,
    created_at: datetime | None = None,
) -> MemoryEvent:
    """造一条记忆。

    `occurred_at=...`（默认）表示「按 days_ago/hour 推出来」；
    显式传 `None` 表示**这条没有发生时间**（用于测时段排除）。
    """
    anchor = (NOW - timedelta(days=days_ago)).replace(hour=hour)
    return MemoryEvent(
        user_id="u1",
        pet_id="p1",
        subject=subject,
        content=content,
        event_type=event_type,
        polarity=Polarity.NEUTRAL,
        source=MemorySource.USER_OBSERVATION,
        status=status,
        confidence=0.9,
        support_count=support_count,
        occurred_at=anchor if occurred_at is ... else occurred_at,
        created_at=created_at or anchor,
    )


def series(subject: str, content: str, days: list[int], hour: int = 6) -> list[MemoryEvent]:
    return [make_event(subject=subject, content=content, days_ago=d, hour=hour) for d in days]


# =============================================================================
# 基本计数
# =============================================================================


class TestCounts:
    def test_observations_and_days(self):
        events = series("wake_up", "六点叫我", [20, 15, 10, 5])
        report = detect_habits(events)
        h = report.habits[0]

        assert h.observations == 4
        assert h.distinct_days == 4
        assert h.span_days == 15
        assert h.strength is HabitStrength.ESTABLISHED

    def test_support_count_is_observations_not_days(self):
        """**次数 ≠ 天数。** 一条强化 5 次的记忆只有 1 个时间戳。

        混为一谈会把「说过三次」当成「连续三天」——
        而习惯的必要条件恰恰是**跨天重复**。
        """
        events = [make_event(support_count=5)]
        h = detect_habits(events).habits[0]

        assert h.observations == 5
        assert h.distinct_days == 1, "只有一个时间戳 → 只有一天"
        assert h.strength is not HabitStrength.ESTABLISHED, (
            "单日出现 5 次不是习惯"
        )

    def test_multiple_events_same_day_counts_once(self):
        events = [
            make_event(days_ago=3, hour=6),
            make_event(days_ago=3, hour=8),
            make_event(days_ago=3, hour=10),
        ]
        h = detect_habits(events).habits[0]
        assert h.observations == 3
        assert h.distinct_days == 1, "同一天 3 次 → 1 天"


# =============================================================================
# 时间分布：**本文件最重要的一处区分**
# =============================================================================


class TestTiming:
    def test_histogram_uses_occurred_at(self):
        events = series("wake_up", "六点叫我", [10, 8, 6, 4], hour=6)
        h = detect_habits(events).habits[0]
        assert h.dominant_time is not None
        assert h.dominant_time.value == "morning"
        assert h.time_concentration == 1.0

    def test_events_without_occurred_at_excluded_from_timing(self):
        """**时段统计不得回退到 `created_at`。**

        主人晚上补记「它今天早上六点叫我」，那条不能被算成「晚上」——
        那会认认真真地算出一个错误的时段分布，而报告里看不出异常。
        """
        events = [
            make_event(days_ago=5, hour=6, occurred_at=None),   # 无发生时间
            make_event(days_ago=4, hour=6, occurred_at=None),
            make_event(days_ago=3, hour=6),
            make_event(days_ago=2, hour=6),
            make_event(days_ago=1, hour=6),
        ]
        h = detect_habits(events).habits[0]

        assert sum(h.time_histogram.values()) == 3, (
            "只有带 occurred_at 的 3 条进时段统计"
        )
        assert h.distinct_days == 5, "但天数统计包含全部 5 条"
        assert any("occurred_at" in n for n in h.limitations), (
            "必须说明有多少条因缺发生时间而未进时段统计"
        )

    def test_no_timing_data_yields_explicit_limitation(self):
        events = [make_event(days_ago=d, occurred_at=None) for d in (5, 4, 3)]
        h = detect_habits(events).habits[0]

        assert h.time_histogram == {}
        assert h.dominant_time is None
        assert h.time_concentration == 0.0
        assert any("无法给出时段分布" in n for n in h.limitations)

    def test_scattered_timing_blocks_a_confident_claim(self):
        """时间分散时**不能说**「它一般早上做这件事」。"""
        events = [
            make_event(days_ago=6, hour=6),
            make_event(days_ago=5, hour=13),
            make_event(days_ago=4, hour=19),
            make_event(days_ago=3, hour=23),
            make_event(days_ago=2, hour=6),
            make_event(days_ago=1, hour=14),
        ]
        h = detect_habits(events).habits[0]
        assert h.time_concentration < 0.5
        assert any("时间很分散" in n for n in h.limitations)

    def test_regularity_is_high_for_concentrated_timing(self):
        concentrated = series("a", "内容", [10, 8, 6, 4, 2], hour=6)
        scattered = [
            make_event(subject="a", content="内容", days_ago=d, hour=h)
            for d, h in ((10, 6), (8, 13), (6, 19), (4, 23), (2, 6))
        ]
        assert (
            detect_habits(concentrated).habits[0].regularity
            > detect_habits(scattered).habits[0].regularity
        )


# =============================================================================
# 成熟度：三值，不是分数
# =============================================================================


class TestStrength:
    def test_established_needs_observations_and_days(self):
        events = series("wake_up", "六点叫我", [10, 8, 6])
        h = detect_habits(events).habits[0]
        assert h.observations >= MIN_OBSERVATIONS_FOR_HABIT
        assert h.distinct_days >= MIN_DAYS_FOR_HABIT
        assert h.strength is HabitStrength.ESTABLISHED
        assert h.is_habit is True

    def test_insufficient_is_not_a_weak_habit(self):
        """**「还不够」是一个状态，不是一个低分。**"""
        events = [make_event(days_ago=1)]
        h = detect_habits(events).habits[0]
        assert h.strength is HabitStrength.INSUFFICIENT
        assert h.is_habit is False
        assert h.limitations, "必须说明缺什么"

    def test_limitations_name_what_is_missing(self):
        events = [make_event(days_ago=d, support_count=2) for d in (1, 0)]
        h = detect_habits(events).habits[0]
        joined = " ".join(h.limitations)
        assert "天" in joined, f"应指出天数不足：{h.limitations}"

    def test_established_has_no_strength_limitations(self):
        events = series("wake_up", "六点叫我", [20, 15, 10, 5])
        h = detect_habits(events).habits[0]
        assert h.strength is HabitStrength.ESTABLISHED
        # 趋势可能仍有说明（窗口不足），但不能有「还不能称之为习惯」类
        assert not any("不足" in n and "次" in n for n in h.limitations)


# =============================================================================
# 趋势：窗口不足时不猜方向
# =============================================================================


class TestTrend:
    def test_short_window_is_unknown(self):
        """跨度不足 → `UNKNOWN`，而不是「最近变多了」。

        窗口太短时「最近变多」只是在描述波动，而用户会当真。
        """
        events = series("wake_up", "六点叫我", [5, 3, 1])
        h = detect_habits(events).habits[0]
        assert h.span_days < MIN_SPAN_DAYS_FOR_TREND
        assert h.trend is HabitTrend.UNKNOWN
        assert any("不判断趋势" in n for n in h.limitations)

    def test_rising_trend(self):
        events = series("wake_up", "六点叫我", [40, 24]) + series(
            "wake_up", "六点叫我", [8, 6, 4, 2]
        )
        h = detect_habits(events).habits[0]
        assert h.span_days >= MIN_SPAN_DAYS_FOR_TREND
        assert h.trend is HabitTrend.RISING

    def test_falling_trend(self):
        events = series("wake_up", "六点叫我", [40, 38, 36, 34]) + series(
            "wake_up", "六点叫我", [4, 2]
        )
        h = detect_habits(events).habits[0]
        assert h.trend is HabitTrend.FALLING

    def test_steady_trend(self):
        events = series("wake_up", "六点叫我", [40, 32, 24, 16, 8, 1])
        h = detect_habits(events).habits[0]
        assert h.trend is HabitTrend.STEADY


# =============================================================================
# 排除与可追溯
# =============================================================================


class TestFiltering:
    def test_non_habit_event_types_excluded(self):
        """`context` / `health` 不构成习惯。

        把健康信号混进「习惯」，会让「它经常吐」看起来像一种生活方式。
        """
        events = [
            make_event(subject="ctx", event_type=EventType.CONTEXT, days_ago=d)
            for d in (3, 2, 1)
        ]
        report = detect_habits(events)
        assert report.habits == ()
        assert report.skipped_events == 3
        assert any("事件类型" in r for r in report.skip_reasons)

    def test_superseded_and_rejected_excluded(self):
        events = [
            make_event(days_ago=5, status=MemoryStatus.SUPERSEDED),
            make_event(days_ago=4, status=MemoryStatus.REJECTED),
            make_event(days_ago=3),
        ]
        report = detect_habits(events)
        assert report.habits[0].observations == 1, "只有 ACTIVE 那条算"

    def test_pending_confirmation_is_counted(self):
        """未确认的**算** —— 习惯关心「发生过几次」，不是「确认过几次」。

        与「未确认的样本不构成案例推理证据」是两件事：
        那里要的是**标签可信**，这里要的是**事件发生过**。
        """
        events = [
            make_event(days_ago=d, status=MemoryStatus.PENDING_CONFIRMATION)
            for d in (5, 4, 3)
        ]
        report = detect_habits(events)
        assert report.habits[0].observations == 3

    def test_missing_subject_excluded_with_reason(self):
        events = [make_event(subject="", days_ago=d) for d in (3, 2, 1)]
        report = detect_habits(events)
        assert report.habits == ()
        assert any("主体" in r for r in report.skip_reasons)

    def test_evidence_ids_are_traceable(self):
        """每条习惯都要能说出「哪几条记录支撑它」。"""
        events = series("wake_up", "六点叫我", [10, 8, 6])
        stored = []
        for i, e in enumerate(events):
            stored.append(e.model_copy(update={"memory_id": f"m{i}"}))
        h = detect_habits(stored).habits[0]
        assert set(h.evidence_ids) == {"m0", "m1", "m2"}

    def test_skip_reasons_are_categorised(self):
        events = [
            make_event(subject="a", event_type=EventType.CONTEXT, days_ago=3),
            make_event(subject="", days_ago=2),
            make_event(subject="b", days_ago=1),
        ]
        report = detect_habits(events)
        assert len(report.skip_reasons) >= 2
        assert sum(report.skip_reasons.values()) == report.skipped_events


# =============================================================================
# 确定性
# =============================================================================


class TestDeterminism:
    def test_same_input_same_output(self):
        events = series("wake_up", "六点叫我", [10, 8, 6])
        a = detect_habits(events)
        b = detect_habits(events)
        assert [h.subject for h in a.habits] == [h.subject for h in b.habits]
        assert [h.observations for h in a.habits] == [h.observations for h in b.habits]

    def test_now_parameter_does_not_change_results(self):
        """**刻意不受「现在」影响。**

        引入「现在」会让同一份数据在不同时刻算出不同结果，
        而报告里的数字就不再可复现了。
        """
        events = series("wake_up", "六点叫我", [10, 8, 6])
        a = detect_habits(events, now=NOW)
        b = detect_habits(events, now=NOW + timedelta(days=365))
        assert a.habits[0].observations == b.habits[0].observations
        assert a.habits[0].strength is b.habits[0].strength

    def test_ordering_is_stable(self):
        events = series("zzz", "后一个", [10, 8, 6]) + series("aaa", "前一个", [10, 8, 6])
        report = detect_habits(events)
        # 同分时按主体名，保证可复现
        assert [h.subject for h in report.habits] == ["aaa", "zzz"]


# =============================================================================
# 渲染
# =============================================================================


class TestRendering:
    def test_report_groups_by_strength(self):
        """三档必须**分组显示**，而不是混在一起。

        `wake_up` 3 次 3 天 → ESTABLISHED
        `food`   2 次 2 天 → EMERGING
        `vacuum` 1 次 1 天 → INSUFFICIENT
        """
        events = (
            series("wake_up", "六点叫我", [20, 15, 10])
            + series("food", "爱吃罐头", [8, 4], hour=18)
            + series("vacuum", "怕吸尘器", [1], hour=14)
        )
        text = render_habit_report(detect_habits(events))
        assert "已形成的习惯" in text
        assert "刚有苗头" in text
        assert "记录还太少" in text

    def test_insufficient_habit_is_visible_not_dropped(self, ):
        """**观察不足的习惯必须出现在报告里，不能消失。**

        初版只渲染 established + emerging，于是只观察过 1 次的那些
        完全不见 —— 用户无法分辨「系统没见过这条记录」
        与「见过了但不够下结论」，而后者是一个**应该被告诉**的事实。
        """
        events = [make_event(content="它偶尔会抓沙发", days_ago=1)]
        text = render_habit_report(detect_habits(events))
        assert "它偶尔会抓沙发" in text, "记录本身必须可见"
        assert "不足以判断" in text or "还不能称之为习惯" in text

    def test_facts_only_include_established(self):
        """喂给下游的事实句**只含已形成的习惯**。

        「刚有苗头」的证据强度差一个数量级，混进去会让下游
        把两件事当成同一件。
        """
        events = series("wake_up", "六点叫我", [20, 15, 10]) + series(
            "vacuum", "怕吸尘器", [3]
        )
        report = detect_habits(events)
        facts = habit_facts(report)
        assert len(facts) == len(report.established())
        assert any("六点叫我" in f for f in facts)
        assert not any("怕吸尘器" in f for f in facts)

    def test_empty_report_says_so_with_counts(self):
        text = render_habit_report(detect_habits([]))
        assert "还没有足够" in text


# =============================================================================
# 问答
# =============================================================================


class TestQuestionAnswering:
    @staticmethod
    def _report():
        events = series("wake_up", "每天早上六点会来叫我起床", [20, 15, 12, 9, 6, 3]) + (
            series("food", "爱吃三文鱼罐头", [18, 10, 4], hour=18)
        )
        return detect_habits(events)

    def test_overview_question(self):
        answer = answer_habit_question(self._report(), "它有什么习惯")
        assert answer is not None
        assert "叫我起床" in answer

    def test_topic_question(self):
        answer = answer_habit_question(self._report(), "它一般几点吃饭")
        assert answer is not None
        assert "三文鱼" in answer

    def test_causal_question_is_refused_explicitly(self):
        """**「为什么」必须显式拒绝，不能拿观察记录充数。**

        「它为什么怕吸尘器」与「它怕吸尘器吗」只差一个字，
        但前者要的是原因，而习惯数据里只有次数/天数/时段。
        返回观察记录会让用户以为系统回答了问题。
        """
        answer = answer_habit_question(self._report(), "它为什么六点叫我起床")
        assert answer is not None
        assert "没有原因" in answer or "不推断因果" in answer, (
            f"应明确说明原因不在数据里，实际：{answer[:80]}"
        )

    def test_unknown_topic_returns_none(self):
        """不属于任何已知主题 → 调用方据此说「没有记录」。"""
        assert answer_habit_question(self._report(), "今天天气如何") is None

    def test_known_topic_without_records_returns_none(self):
        """主题对上了但没有记录 → **不从别的习惯推**。"""
        assert answer_habit_question(self._report(), "它喜欢什么玩具") is None

    def test_empty_question_returns_none(self):
        assert answer_habit_question(self._report(), "   ") is None

    def test_trend_question_without_window_admits_it(self):
        report = detect_habits(series("wake_up", "六点叫我", [5, 3, 1]))
        answer = answer_habit_question(report, "它最近有什么变化吗")
        assert answer is not None
        assert "判断不了趋势" in answer or "窗口太短" in answer

    def test_overview_with_no_habits_says_so(self):
        report = detect_habits([make_event(days_ago=1)])
        answer = answer_habit_question(report, "它有什么习惯")
        assert answer is not None
        assert "还没有" in answer


# =============================================================================
# 与真实输入路径的接缝：**主体识别决定归组**
# =============================================================================


class TestSubjectGrouping:
    """习惯能不能形成，取决于「同一件事的不同说法」有没有归到同一主体。

    ## 为什么这组测试重要

    习惯层自己的测试可以直接造 `subject="wake_up"` 的事件，于是全部通过 ——
    而真实对话路径里 `subject` 是由 `_guess_subject` 从内容猜的。
    如果它对措辞敏感，**同一个习惯换个说法就永远累加不起来**，
    而习惯层会把它报告成「记录还太少」—— 指向一个错误的原因。
    """

    @pytest.mark.parametrize(
        "content",
        [
            "它每天早上六点叫我起床",
            "它六点准时叫我",
            "今天六点又把我叫醒了",
        ],
    )
    def test_wake_up_paraphrases_share_subject(self, content: str):
        from app.graph.nodes import _guess_subject

        assert _guess_subject(content) == "wake_up", (
            f"{content!r} 未归到 wake_up —— 同一习惯换说法就不会被累加"
        )

    def test_paraphrases_group_into_one_habit(self):
        """**端到端：三种说法 → 一条习惯。**"""
        from app.graph.nodes import _guess_subject

        events = [
            make_event(
                subject=_guess_subject(c),
                content=c,
                days_ago=d,
                hour=6,
            )
            for c, d in [
                ("它每天早上六点叫我起床", 20),
                ("它六点准时叫我", 16),
                ("今天六点又把我叫醒了", 12),
            ]
        ]
        report = detect_habits(events)

        assert len(report.habits) == 1, (
            f"应归为 1 条习惯，实际 {len(report.habits)} 条："
            f"{[h.subject for h in report.habits]}"
        )
        assert report.habits[0].observations == 3
        assert report.habits[0].strength is HabitStrength.ESTABLISHED

    def test_unknown_content_reports_the_limitation_not_a_wrong_reason(self):
        """主体猜不出时，报告要**说明是识别失败**，而不是「记录太少」。

        两者对用户的含义完全不同：前者是系统能力边界，
        后者是「你再多记几次」—— 而用户可能已经记了很多次。
        """
        from app.schemas import SUBJECT_FALLBACK_PREFIX

        events = [
            make_event(
                subject=f"{SUBJECT_FALLBACK_PREFIX}deadbeef",
                content="某种无法归类的说法",
                days_ago=3,
            )
        ]
        report = detect_habits(events)

        assert any("主体" in n for n in report.limitations), (
            f"必须说明主体识别失败，实际：{report.limitations}"
        )
        assert any(
            "主体无法" in n for h in report.habits for n in h.limitations
        )

    def test_distinct_subjects_do_not_merge(self):
        """反向保证：不同主体**不能**被合并。

        误合并会让系统声称一个不存在的习惯，比漏掉更糟。
        """
        events = series("vacuum", "怕吸尘器", [10, 8, 6]) + series(
            "food", "爱吃罐头", [10, 8, 6], hour=18
        )
        report = detect_habits(events)
        assert len(report.habits) == 2
