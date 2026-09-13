"""「宠物的一天」测试。

**测的重点是三条禁令能否被机械强制，而不是故事写得好不好。**

娱乐形态有三类风险（`docs/14` §3），每一类都必须有代码级的门：

| 禁令 | 风险 | 本文件对应的断言 |
| --- | --- | --- |
| 1 | 娱乐化健康信号 | 健康信号强制进 L3、L3 禁止 playful、免责声明不得覆盖健康内容 |
| 2 | 系统性固化行为解读 | 系统归纳不进娱乐层、中性比例可断言 |
| 3 | 鼓励打扰猫 | 不索取新照片（只用已上传的） |

外加一条本形态特有的风险：**漫画格无法标注不确定性** ——
所以每个节拍必须能指回已校验的事实（`fact_refs`）。
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from app.digest.aggregate import aggregate
from app.schemas import (
    DISCLAIMER_ENTERTAINMENT,
    ENTERTAINABLE_EVENT_TYPES,
    ENTERTAINABLE_SOURCES,
    DailyStory,
    DigestCandidate,
    DigestMessage,
    EventType,
    ExtractionSource,
    InfoLayer,
    Polarity,
    StoryBeat,
    StoryTimeOfDay,
    StoryTone,
    find_physiology_words,
)
from app.story import compose_story, render_beat_for_panel, render_story

DAY = date(2026, 3, 14)


def _cand(
    content: str,
    *,
    event_type: EventType = EventType.BEHAVIOR,
    source_layer: ExtractionSource = ExtractionSource.OWNER_RECORD,
    subject: str = "window",
) -> DigestCandidate:
    return DigestCandidate(
        event_type=event_type,
        subject=subject,
        content=content,
        quote=content,
        source_layer=source_layer,
        confidence=0.8,
        polarity=Polarity.NEUTRAL,
    )


@pytest.fixture()
def stats():
    return aggregate(day=DAY, messages=[])


def _story(cands, stats, messages=None) -> DailyStory:
    from app.schemas import DailySummary

    summary = DailySummary(date=DAY, stats=stats, candidates=cands)
    return compose_story(summary, messages=messages)


# ═══════════════════════════════════════════════════════════════
# 1. 禁令 1 —— 娱乐化健康信号
# ═══════════════════════════════════════════════════════════════


class TestBanOneHealthNotEntertained:
    """**禁令 1：凡是被红旗覆盖的信号，一律禁止进入娱乐层。**

    文档里的反例：

    > 猫砂盆异常 → 「本喵今天好喜欢这个猫砂盆，反复去了好几次呢～」
    > **一次急诊信号被变成了笑话。**
    """

    def test_health_event_is_routed_to_L3(self, stats):
        story = _story(
            [_cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit")],
            stats,
        )
        assert len(story.beats) == 1
        assert story.beats[0].info_layer is InfoLayer.HEALTH

    def test_health_beat_cannot_be_playful(self):
        """**核心断言**：健康信号不得带娱乐调性。

        用户会觉得「系统都说可爱，那应该没事」。
        """
        with pytest.raises(ValueError, match="serious"):
            StoryBeat(
                at=StoryTimeOfDay.EVENING,
                info_layer=InfoLayer.HEALTH,
                tone=StoryTone.PLAYFUL,
                line="今天我有点不对劲：它吐了两次",
                fact_refs=[0],
            )

    def test_health_beat_requires_traceability(self):
        """健康内容必须可追溯到具体记录，否则它是个无法核查的健康断言。"""
        with pytest.raises(ValueError, match="fact_refs"):
            StoryBeat(
                at=StoryTimeOfDay.EVENING,
                info_layer=InfoLayer.HEALTH,
                tone=StoryTone.SERIOUS,
                line="今天我不舒服",
            )

    def test_disclaimer_does_not_cover_health_content(self, stats):
        """**「仅供娱乐」不能盖在健康内容上。**

        它传达的是「这不用太当真」——而那正是禁令 1 想防的。
        """
        story = _story(
            [_cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit")],
            stats,
        )
        text = render_story(story)
        assert story.health_notice is not None
        assert "这不是诊断" in text
        # 当天只有健康内容 → 不应出现娱乐免责声明
        assert DISCLAIMER_ENTERTAINMENT not in text

    def test_both_notices_when_mixed_content(self, stats):
        """混合内容时两条声明都要在，且各自覆盖自己那部分。"""
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit"),
            ],
            stats,
        )
        text = render_story(story)
        assert DISCLAIMER_ENTERTAINMENT in text
        assert "这不是诊断" in text

    def test_health_notice_must_match_content(self, stats):
        """双向检查：有健康节拍必须有提示；无健康节拍不得有提示。"""
        with pytest.raises(ValueError, match="必须提供 health_notice"):
            DailyStory(
                date=DAY,
                title="t",
                beats=[
                    StoryBeat(
                        at=StoryTimeOfDay.EVENING,
                        info_layer=InfoLayer.HEALTH,
                        tone=StoryTone.SERIOUS,
                        line="它吐了两次",
                        fact_refs=[0],
                    )
                ],
            )
        with pytest.raises(ValueError, match="不得提供 health_notice"):
            DailyStory(
                date=DAY,
                title="t",
                beats=[
                    StoryBeat(at=StoryTimeOfDay.EVENING, line="本喵心情不错")
                ],
                health_notice="注意健康",
            )

    def test_health_content_is_not_dropped(self, stats):
        """健康信号进了 L3，**不是被丢掉** —— 丢掉会让用户永远不知道。"""
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit"),
            ],
            stats,
        )
        assert story.has_health_content
        assert len(story.beats) == 2


# ═══════════════════════════════════════════════════════════════
# 2. 拟人化边界 —— 同一个词在不同层
# ═══════════════════════════════════════════════════════════════


class TestRoleplayOverreach:
    """**判据不是「像不像猫说的话」，而是「用户会不会据此改变照顾行为」。**"""

    @pytest.mark.parametrize(
        "line",
        ["本喵饿了", "本喵肚子疼", "本喵想出去", "本喵不喜欢这个猫粮"],
    )
    def test_entertainment_layer_rejects_physiology(self, line):
        with pytest.raises(ValueError, match="ROLEPLAY_OVERREACH"):
            StoryBeat(
                at=StoryTimeOfDay.EVENING,
                info_layer=InfoLayer.ENTERTAINMENT,
                tone=StoryTone.PLAYFUL,
                line=line,
                fact_refs=[0],
            )

    @pytest.mark.parametrize(
        "line",
        ["本喵今天心情不错", "本喵想你啦", "本喵在窗台发呆"],
    )
    def test_entertainment_layer_allows_emotion(self, line):
        """情绪与关系可以拟人化 —— 它们不影响照料决策。"""
        beat = StoryBeat(
            at=StoryTimeOfDay.EVENING,
            info_layer=InfoLayer.ENTERTAINMENT,
            tone=StoryTone.PLAYFUL,
            line=line,
            fact_refs=[0],
        )
        assert beat.line == line

    def test_same_word_is_legal_in_health_layer(self):
        """**同一个生理词在 L3 合法。**

        诊断之外、带严肃语气、且可追溯的硥状描述正是健康层该说的话。
        层不同，规则不同 —— 这是本设计最要紧的一条区分。
        """
        beat = StoryBeat(
            at=StoryTimeOfDay.EVENING,
            info_layer=InfoLayer.HEALTH,
            tone=StoryTone.SERIOUS,
            line="今天我有点难受：它吐了两次",
            fact_refs=[0],
        )
        assert "难受" in beat.line

    def test_physiology_detector_finds_words(self):
        assert find_physiology_words("本喵有点饿了") == ["饿了", "饿"]
        assert find_physiology_words("本喵心情很好") == []


# ═══════════════════════════════════════════════════════════════
# 3. 禁令 2 —— 系统性固化行为解读
# ═══════════════════════════════════════════════════════════════


class TestBanTwoSystematicFraming:
    """**禁令 2：禁止把有真实生理含义的行为，系统性固定映射到同一娱乐叙事。**"""

    def test_ai_inference_does_not_enter_entertainment(self, stats):
        """系统归纳不进入娱乐层。

        把**推测**反复渲染成故事会系统性固化解读 —— 这正是禁令 2 针对的机制。
        """
        story = _story(
            [
                _cand(
                    "它可能是对声音敏感",
                    source_layer=ExtractionSource.AI_INFERENCE,
                )
            ],
            stats,
        )
        assert story.beats == []
        assert len(story.excluded) == 1
        assert "禁令 2" in story.excluded[0].reason

    def test_owner_record_enters(self, stats):
        story = _story([_cand("它在窗台睡了一下午")], stats)
        assert len(story.beats) == 1

    def test_health_event_not_in_entertainable_types(self):
        """健康事件不在可娱乐白名单里 —— 双重保险（分类层已拦一次）。"""
        assert EventType.HEALTH not in ENTERTAINABLE_EVENT_TYPES
        assert ExtractionSource.AI_INFERENCE not in ENTERTAINABLE_SOURCES

    def test_neutral_ratio_is_assertable(self, stats):
        """禁令 2 要求「必须有相当比例的中性/事实性呈现」。

        这个属性让该要求**可被测试断言**，而不只是写在文档里。
        """
        story = _story(
            [_cand("它在窗台睡了一下午"), _cand("它蹭了蹭我的腿")], stats
        )
        assert story.neutral_ratio == 1.0
        assert story.tone_mix == {StoryTone.NEUTRAL: 2}

    def test_tone_mix_is_recorded(self, stats):
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit"),
            ],
            stats,
        )
        assert story.tone_mix[StoryTone.SERIOUS] == 1
        assert story.tone_mix[StoryTone.NEUTRAL] == 1


# ═══════════════════════════════════════════════════════════════
# 4. 可追溯性 —— 漫画格无法标注不确定性
# ═══════════════════════════════════════════════════════════════


class TestTraceability:
    """**一个画格是一个确定的陈述，你没有地方写「可能」。**

    所以每个节拍必须能指回已校验的事实。
    """

    def test_every_beat_has_fact_refs(self, stats):
        story = _story(
            [_cand("它在窗台睡了一下午"), _cand("它蹭了蹭我的腿")], stats
        )
        for beat in story.beats:
            assert beat.fact_refs, f"节拍「{beat.line}」没有事实引用"

    def test_fact_refs_point_to_real_indices(self, stats):
        story = _story(
            [_cand("它在窗台睡了一下午"), _cand("它蹭了蹭我的腿")], stats
        )
        assert story.beats[0].fact_refs == [0]
        assert story.beats[1].fact_refs == [1]

    def test_negative_ref_is_rejected(self):
        with pytest.raises(ValueError, match="负下标"):
            DailyStory(
                date=DAY,
                title="t",
                beats=[
                    StoryBeat(
                        at=StoryTimeOfDay.EVENING, line="本喵很好", fact_refs=[-1]
                    )
                ],
            )

    def test_ranking_uses_candidate_index(self, stats):
        """混合内容时下标必须仍指向正确的候选（排除项不占用下标）。"""
        story = _story(
            [
                _cand("它可能是对声音敏感", source_layer=ExtractionSource.AI_INFERENCE),
                _cand("它在窗台睡了一下午"),
            ],
            stats,
        )
        assert story.beats[0].fact_refs == [1]
        assert story.beats[0].line == "它在窗台睡了一下午"


# ═══════════════════════════════════════════════════════════════
# 5. 漫画/写真扩展位
# ═══════════════════════════════════════════════════════════════


class TestComicExtensionSlots:
    """本期不生成图像，但结构必须天然支持。"""

    def test_beat_exposes_photo_slot(self, stats):
        story = _story([_cand("它在窗台睡了一下午")], stats)
        assert hasattr(story.beats[0], "photo_ref")
        assert story.beats[0].photo_ref is None

    def test_panel_style_is_derived_from_layer(self, stats):
        """**视觉样式由信息层决定，不由调用方自由选择。**

        A3 要求 L1 不得使用 L3 的视觉样式（避免「看起来像健康提示」）。
        反过来也成立：健康内容不得穿上娱乐层的样式。
        """
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit"),
            ],
            stats,
        )
        panels = [render_beat_for_panel(b) for b in story.beats]
        styles = {p["style"] for p in panels}
        assert styles == {"entertainment", "health"}

    def test_panel_carries_caption_and_time(self, stats):
        story = _story([_cand("它在窗台睡了一下午")], stats)
        panel = render_beat_for_panel(story.beats[0])
        assert panel["caption"] == "它在窗台睡了一下午"
        assert panel["time_of_day"] == "evening"


# ═══════════════════════════════════════════════════════════════
# 6. 不静默丢弃
# ═══════════════════════════════════════════════════════════════


class TestNoSilentDrop:
    """与 `DailySummary.rejected` 同一条原则：**不让任何东西静默消失**。"""

    def test_excluded_records_reason(self, stats):
        story = _story(
            [_cand("它可能是对声音敏感", source_layer=ExtractionSource.AI_INFERENCE)],
            stats,
        )
        assert len(story.excluded) == 1
        assert story.excluded[0].reason

    def test_excluded_surfaced_in_render(self, stats):
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它可能是对声音敏感", source_layer=ExtractionSource.AI_INFERENCE),
            ],
            stats,
        )
        text = render_story(story)
        assert "未进入故事" in text

    def test_one_bad_beat_does_not_kill_the_day(self, stats):
        """逐条隔离：一条坏节拍不能让当天的故事全丢。

        与 `app/digest` 逐条隔离坏候选是同一条原则。
        """
        story = _story([_cand("它在窗台睡了一下午"), _cand("它蹭了蹭我的腿")], stats)
        assert len(story.beats) == 2


# ═══════════════════════════════════════════════════════════════
# 7. 渲染与可复现
# ═══════════════════════════════════════════════════════════════


class TestRender:
    def test_health_section_comes_first(self, stats):
        """A2：L3 可抢占并压住 L1/L2 —— 先看到需要严肃对待的内容。"""
        story = _story(
            [
                _cand("它在窗台睡了一下午"),
                _cand("它今天吐了两次", event_type=EventType.HEALTH, subject="vomit"),
            ],
            stats,
        )
        text = render_story(story)
        assert text.index("需要留意") < text.index("它在窗台睡了一下午")

    def test_empty_story_renders(self, stats):
        story = _story([], stats)
        assert "还没有可讲述的内容" in render_story(story)

    def test_time_buckets_from_message_timestamps(self, stats):
        msgs = [
            DigestMessage(
                role="user",
                content="它早上在窗台趴着",
                at=datetime(2026, 3, 14, 8, tzinfo=timezone.utc),
            ),
            DigestMessage(
                role="user",
                content="晚上它蹭我的手",
                at=datetime(2026, 3, 14, 21, tzinfo=timezone.utc),
            ),
        ]
        story = _story(
            [_cand("它早上在窗台趴着"), _cand("晚上它蹭我的手")], stats, msgs
        )
        slots = [b.at for b in story.beats]
        assert slots == [StoryTimeOfDay.MORNING, StoryTimeOfDay.EVENING]

    def test_time_buckets_match_by_quote_not_content(self, stats):
        """**回归测试**：时间分桶必须用 quote 匹配，不能用 content。

        初版用 content 精确匹配，而 content 是**改写后**的句子
        （「它喜欢在窗台晒太阳」），原文是「它早上在窗台趴着晒太阳」——
        一条也匹配不上，所有节拍静默退回默认的 EVENING。

        掩盖它的原因：单测 fixture 里 content 与原文相同。
        """
        from app.schemas import DigestCandidate, Polarity

        msgs = [
            DigestMessage(
                role="user",
                content="它早上在窗台趴着晒太阳",
                at=datetime(2026, 3, 14, 8, tzinfo=timezone.utc),
            ),
            DigestMessage(
                role="user",
                content="晚上我梳毛的时候它一直呼噜",
                at=datetime(2026, 3, 14, 21, tzinfo=timezone.utc),
            ),
        ]
        # content 是改写的；quote 才是原文片段
        cands = [
            DigestCandidate(
                event_type=EventType.PREFERENCE,
                subject="window",
                content="它喜欢在窗台晒太阳",  # ← 原文里没有这句
                quote="它早上在窗台趴着晒太阳",  # ← 原文里有
                source_layer=ExtractionSource.OWNER_RECORD,
                confidence=0.8,
                polarity=Polarity.POSITIVE,
            ),
            DigestCandidate(
                event_type=EventType.BEHAVIOR,
                subject="brush",
                content="它梳毛时很放松",
                quote="晚上我梳毛的时候它一直呼噜",
                source_layer=ExtractionSource.OWNER_RECORD,
                confidence=0.8,
                polarity=Polarity.POSITIVE,
            ),
        ]
        story = _story(cands, stats, msgs)
        slots = [b.at for b in story.beats]
        assert slots == [StoryTimeOfDay.MORNING, StoryTimeOfDay.EVENING], (
            f"时间分桶失效，得到 {slots}"
        )

    def test_reproducible(self, stats):
        """同一份总结 → 同一个故事（DESIGN.md §6.5 可复现要求）。"""
        cands = [_cand("它在窗台睡了一下午"), _cand("它蹭了蹭我的腿")]
        a = render_story(_story(cands, stats))
        b = render_story(_story(cands, stats))
        assert a == b
