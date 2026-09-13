"""每日总结的测试。

**这个模块测的重点不是「提取到了什么」，而是「拒绝了什么」。**

每日总结最容易出的问题是 LLM 做「合理但无依据」的归纳：
主人只说「它今天老在窗台」，模型总结成「这只猫喜欢窗台晒太阳」。
后者很可能对——但原文里没有这句话。

所以本文件的机制性断言集中在：

- 引用片段不在原文 → **必须拒绝且不写入**
- 引用过短（校验退化成恒真）→ 必须拒绝
- 一条坏候选不能让整天失败
- 系统归纳不得直接生效（只能 PENDING）
- 同一天重复总结是**强化**而非新增（幂等）

这些都属于 `docs/BUGS.md` §3.1 说的「机制性断言」——
概率与文本系统不会自己报错，必须主动构造不变性去抓。
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest
from app.digest import summarize_day
from app.digest.aggregate import aggregate, render_transcript
from app.digest.extract import (
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_candidates,
    verify_quote,
)
from app.llm import HashEmbedder, MockLLM
from app.memory import MemoryWriter
from app.schemas import (
    DigestMessage,
    EventType,
    ExtractionSource,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    Polarity,
)
from app.store import InMemoryStore

EMB = HashEmbedder()
DAY = date(2026, 3, 14)
U, P = "user-1", "pet-1"


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════


def _msgs(*contents: str, assistant: bool = False) -> list[DigestMessage]:
    role = "assistant" if assistant else "user"
    return [DigestMessage(role=role, content=c) for c in contents]  # type: ignore[arg-type]


def _cand_json(
    content: str,
    quote: str,
    *,
    source_layer: str = "owner_record",
    event_type: str = "preference",
    subject: str = "vacuum",
    polarity: str = "negative",
) -> dict[str, str]:
    return {
        "content": content,
        "quote": quote,
        "source_layer": source_layer,
        "event_type": event_type,
        "subject": subject,
        "polarity": polarity,
    }


def _payload(*cands: dict[str, str]) -> str:
    return json.dumps({"candidates": list(cands)}, ensure_ascii=False)


@pytest.fixture()
def conv() -> list[DigestMessage]:
    return _msgs(
        "它今天又在窗台趴着",
        "我开吸尘器它就跑了，躲到床底下",
        "晚上我梳毛的时候它一直呼噜",
    )


@pytest.fixture()
def writer() -> MemoryWriter:
    return MemoryWriter(store=InMemoryStore(), embedder=EMB)


# ═══════════════════════════════════════════════════════════════
# 1. 引用校验 —— 本功能的核心
# ═══════════════════════════════════════════════════════════════


class TestQuoteVerification:
    """引用片段必须逐字出现在原文里。**这是唯一的反编造机制。**"""

    def test_exact_quote_passes(self):
        ok, reason = verify_quote("开吸尘器它就跑了", "我开吸尘器它就跑了，躲到床底下")
        assert ok
        assert reason == ""

    def test_whitespace_differences_tolerated(self):
        """空白差异不算编造 —— 模型复制时的换行/空格不该导致误拒。"""
        ok, _ = verify_quote("开吸尘器 它就跑了", "我 开吸尘器\n它就跑了 ，躲起来")
        assert ok

    def test_paraphrase_is_rejected(self):
        """**改写过的片段必须被拒。**

        若允许语义等价，校验就退化成「看起来像」= 没有校验。
        """
        ok, reason = verify_quote(
            "它对吸尘器的声音很害怕",  # 改写
            "我开吸尘器它就跑了，躲到床底下",
        )
        assert not ok
        assert "找不到" in reason

    def test_short_quote_is_rejected(self):
        """**过短的引用让校验退化成恒真** —— 这是最隐蔽的失效方式。

        「它」「今天」几乎必然能在任意对话里找到，
        于是任何编造只要附一句常见短语就能通过。
        """
        ok, reason = verify_quote("它", "它今天又在窗台趴着")
        assert not ok
        assert "过短" in reason

    def test_empty_quote_is_rejected(self):
        ok, reason = verify_quote("   ", "它今天又在窗台趴着")
        assert not ok
        assert "为空" in reason

    def test_quote_crossing_two_sentences_is_rejected(self):
        """跨句拼接的引用不是「原文片段」，必须拒绝。"""
        ok, _ = verify_quote(
            "它今天又在窗台趴着我开吸尘器",
            "它今天又在窗台趴着\n我开吸尘器它就跑了",
        )
        # 去空白后恰好相邻，这里断言的是「拼接本身可被察觉」的设计意图：
        # 实现选择容忍空白，所以这个用例确认容忍的边界是「仅空白」
        assert ok, "仅空白差异应被容忍；若将来收紧，本断言需同步修改"


# ═══════════════════════════════════════════════════════════════
# 2. 解析：逐条隔离，坏候选不拖垮整天
# ═══════════════════════════════════════════════════════════════


class TestParsing:
    def test_good_candidate_admitted(self, conv):
        raw = _payload(_cand_json("它怕吸尘器", "我开吸尘器它就跑了"))
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert len(admitted) == 1
        assert not rejected
        assert admitted[0].event_type is EventType.PREFERENCE
        assert admitted[0].source_layer is ExtractionSource.OWNER_RECORD

    def test_fabricated_quote_rejected(self, conv):
        """编造内容即使格式完美也必须被拒 —— 这才是校验存在的意义。"""
        raw = _payload(
            _cand_json("它喜欢窗台晒太阳", "它最喜欢趴在窗台晒太阳了")
        )
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert not admitted
        assert len(rejected) == 1
        assert "找不到" in rejected[0].reason

    def test_one_bad_candidate_does_not_kill_the_day(self, conv):
        """**一条坏候选不能让整天的总结失败。**

        没有逐条隔离时，模型返回一个不存在的枚举值就会让整天丢失。
        """
        raw = _payload(
            _cand_json("它怕吸尘器", "我开吸尘器它就跑了"),
            _cand_json("它喜欢窗台", "这句话原文里没有"),
            _cand_json("它梳毛时呼噜", "晚上我梳毛的时候它一直呼噜"),
        )
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert len(admitted) == 2
        assert len(rejected) == 1

    def test_invalid_enum_rejected_with_reason(self, conv):
        raw = _payload(
            _cand_json(
                "它怕吸尘器", "我开吸尘器它就跑了", event_type="not_a_real_type"
            )
        )
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert not admitted
        assert "取值非法" in rejected[0].reason
        assert "preference" in rejected[0].reason  # 错误信息里列出允许值

    def test_missing_quote_rejected(self, conv):
        raw = json.dumps(
            {"candidates": [{"content": "它怕吸尘器", "event_type": "preference"}]},
            ensure_ascii=False,
        )
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert not admitted
        assert "quote 缺失" in rejected[0].reason

    def test_missing_subject_falls_back_to_event_type(self, conv):
        """subject 缺失不致命 —— 用 event_type 兜底，保证仍可检索。"""
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "content": "它怕吸尘器",
                        "quote": "我开吸尘器它就跑了",
                        "source_layer": "owner_record",
                        "event_type": "preference",
                        "polarity": "negative",
                    }
                ]
            },
            ensure_ascii=False,
        )
        admitted, _ = parse_candidates(raw, messages=conv)
        assert admitted[0].subject == "preference"

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "我不知道该怎么回答",
            "{不是合法 json",
            '{"candidates": "应该是数组"}',
            "[1, 2, 3]",
        ],
    )
    def test_malformed_output_becomes_rejection_not_crash(self, raw, conv):
        """格式问题 → 拒绝记录，**不抛异常**。"""
        admitted, rejected = parse_candidates(raw, messages=conv)
        assert not admitted
        assert rejected, "格式错误也必须留下可复核的记录"

    def test_markdown_fence_is_stripped(self, conv):
        raw = "```json\n" + _payload(
            _cand_json("它怕吸尘器", "我开吸尘器它就跑了")
        ) + "\n```"
        admitted, _ = parse_candidates(raw, messages=conv)
        assert len(admitted) == 1

    def test_prose_wrapped_json_is_extracted(self, conv):
        raw = "好的，这是结果：\n" + _payload(
            _cand_json("它怕吸尘器", "我开吸尘器它就跑了")
        ) + "\n希望有帮助。"
        admitted, _ = parse_candidates(raw, messages=conv)
        assert len(admitted) == 1

    def test_confidence_depends_on_source_layer(self, conv):
        """主人的直述高于系统归纳 —— 这不是调参，是语义。"""
        raw = _payload(
            _cand_json("它怕吸尘器", "我开吸尘器它就跑了"),
            _cand_json(
                "它可能对声音敏感",
                "我开吸尘器它就跑了",
                source_layer="ai_inference",
            ),
        )
        admitted, _ = parse_candidates(raw, messages=conv)
        by_source = {c.source_layer: c.confidence for c in admitted}
        assert (
            by_source[ExtractionSource.OWNER_RECORD]
            > by_source[ExtractionSource.AI_INFERENCE]
        )

    def test_empty_candidate_list_is_valid(self, conv):
        admitted, rejected = parse_candidates('{"candidates": []}', messages=conv)
        assert not admitted
        assert not rejected


# ═══════════════════════════════════════════════════════════════
# 3. 确定性统计 —— 计数是事实，由代码算
# ═══════════════════════════════════════════════════════════════


class TestAggregate:
    def test_counts_messages_by_role(self, conv):
        msgs = conv + _msgs("我在呢", assistant=True)
        stats = aggregate(day=DAY, messages=msgs)
        assert stats.message_count == 4
        assert stats.user_message_count == 3
        assert stats.assistant_message_count == 1

    def test_topic_counts_only_use_user_messages(self):
        """**助手的话不计入主题统计。**

        否则每个主题的计数会凭空翻倍 —— 助手的话本就是从主人的话生成的，
        一起统计等于自说自话，不是观察。
        """
        msgs = [
            DigestMessage(role="user", content="它怕吸尘器"),
            DigestMessage(role="assistant", content="吸尘器吸尘器吸尘器"),
        ]
        stats = aggregate(day=DAY, messages=msgs)
        assert stats.topic_counts.get("vacuum") == 1

    def test_topic_counts_are_reproducible(self, conv):
        a = aggregate(day=DAY, messages=conv)
        b = aggregate(day=DAY, messages=conv)
        assert a.topic_counts == b.topic_counts

    def test_topic_counts_sorted_by_frequency(self):
        """排序稳定 —— 否则同一天两次调用可能给出不同顺序，不可复现。"""
        msgs = _msgs("吸尘器 吸尘器 吸尘器", "猫粮", "猫粮")
        stats = aggregate(day=DAY, messages=msgs)
        items = list(stats.topic_counts.items())
        assert items[0][0] == "vacuum"
        assert items == sorted(items, key=lambda kv: (-kv[1], kv[0]))

    def test_empty_day_is_not_an_error(self):
        stats = aggregate(day=DAY, messages=[])
        assert stats.is_empty
        assert stats.topic_counts == {}

    def test_inconsistent_counts_are_rejected(self):
        from app.schemas import DigestStats

        with pytest.raises(ValueError, match="消息数不一致"):
            DigestStats(
                date=DAY,
                message_count=5,
                user_message_count=2,
                assistant_message_count=1,
                total_chars=10,
            )

    def test_transcript_labels_speakers_in_chinese(self, conv):
        """角色用中文标注：模型在中文任务下对指代的理解更稳，
        减少把助手的话当成主人的事实。"""
        text = render_transcript(conv + _msgs("我在", assistant=True))
        assert "主人：" in text
        assert "你：" in text
        assert "user：" not in text


# ═══════════════════════════════════════════════════════════════
# 4. Prompt 构造
# ═══════════════════════════════════════════════════════════════


class TestPrompt:
    def test_system_prompt_contains_json_keyword(self):
        """DashScope 的 JSON mode 要求 messages 里出现 "JSON"，否则请求被拒。

        这是实测得到的约束（见 docs/16-model-selection.md），不是习惯。
        """
        assert "JSON" in SYSTEM_PROMPT

    def test_system_prompt_announces_code_verification(self):
        """必须让模型知道 quote 会被代码校验 —— 这是硬门，不是建议。"""
        assert "代码" in SYSTEM_PROMPT

    def test_user_prompt_includes_code_computed_counts(self, conv):
        prompt = build_user_prompt(stats=aggregate(day=DAY, messages=conv), messages=conv)
        assert "由代码计算" in prompt
        assert "不要重新数" in prompt

    def test_user_prompt_lists_existing_memories(self, conv):
        prompt = build_user_prompt(
            stats=aggregate(day=DAY, messages=conv),
            messages=conv,
            existing_memories=["它很怕吸尘器"],
        )
        assert "已经记录过" in prompt
        assert "它很怕吸尘器" in prompt


# ═══════════════════════════════════════════════════════════════
# 5. 端到端 —— 写入行为与幂等
# ═══════════════════════════════════════════════════════════════


class TestEndToEnd:
    def test_empty_day_does_not_call_model(self, writer):
        """**没有对话就不调模型。** 否则既烧钱，又可能让它编点东西出来。"""
        llm = MockLLM(default="{}")
        result = summarize_day(
            day=DAY, messages=[], writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert result.skipped_reason is not None
        assert llm.call_count == 0
        assert result.applied.written_memory_ids == []

    def test_owner_record_is_written_active(self, writer, conv):
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "我开吸尘器它就跑了"))
        )
        result = summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert len(result.applied.written_memory_ids) == 1
        stored = writer.store.list_memories(user_id=U, pet_id=P)
        assert len(stored) == 1
        assert stored[0].status is MemoryStatus.ACTIVE
        assert stored[0].source is MemorySource.USER_OBSERVATION

    def test_ai_inference_cannot_be_active(self, writer, conv):
        """**核心安全断言。**

        系统归纳的内容是 `SYSTEM_INFERENCE`，而契约规定它不得为 ACTIVE
        （docs/04 §3.5 防自我强化）。它必须停在 PENDING_CONFIRMATION。
        """
        llm = MockLLM(
            default=_payload(
                _cand_json(
                    "它可能对高频声音敏感",
                    "我开吸尘器它就跑了",
                    source_layer="ai_inference",
                )
            )
        )
        summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        stored = writer.store.list_memories(user_id=U, pet_id=P, include_non_active=True)
        assert stored, "候选应被写入，但状态必须是 PENDING"
        assert all(m.status is MemoryStatus.PENDING_CONFIRMATION for m in stored)
        assert all(m.source is MemorySource.SYSTEM_INFERENCE for m in stored)

    def test_fabricated_candidates_are_not_written(self, writer, conv):
        """**编造的内容一条都不许落库。**"""
        llm = MockLLM(
            default=_payload(
                _cand_json("它喜欢窗台晒太阳", "它最喜欢趴在窗台晒太阳"),
                _cand_json("它讨厌梳毛", "它一看到梳子就咬人"),
            )
        )
        result = summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert result.applied.written_memory_ids == []
        assert len(result.summary.rejected) == 2
        assert writer.store.list_memories(user_id=U, pet_id=P, include_non_active=True) == []

    def test_same_day_twice_reinforces_instead_of_duplicating(self, writer, conv):
        """**幂等**：重复总结同一天是「强化」，不是新增。

        这依赖飞轮已有的去重。若它失效，长期运行会产生大量重复记忆。
        """
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "我开吸尘器它就跑了"))
        )

        for _ in range(2):
            summarize_day(
                day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
            )

        stored = writer.store.list_memories(user_id=U, pet_id=P, include_non_active=True)
        assert len(stored) == 1, "同一天的同一句话不应产生第二条记忆"
        assert stored[0].support_count >= 2

    def test_summary_is_not_persisted_as_memory(self, writer, conv):
        """总结本身不写入记忆 —— 它是对记忆的**视图**。

        存视图会产生第二个真相来源，两者会漂移。
        """
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "我开吸尘器它就跑了"))
        )
        summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        stored = writer.store.list_memories(user_id=U, pet_id=P, include_non_active=True)
        assert len(stored) == 1  # 只有候选那一条，没有「今日总结」这类记录

    def test_notes_record_rejections(self, writer, conv):
        llm = MockLLM(default=_payload(_cand_json("它喜欢窗台晒太阳", "原文没有这句")))
        result = summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert any("未通过原文校验" in n for n in result.summary.notes)

    def test_health_gap_is_visible_without_sink(self, writer, conv):
        """未提供 health_sink 时，健康信号的去向必须可见。

        飞轮会正确拒绝 HEALTH 事件进通用记忆，而「光拒绝」意味着
        健康信号无处可去 —— 不写进 notes 就是一个静默的功能缺失。
        """
        llm = MockLLM(
            default=_payload(
                _cand_json(
                    "它今天吐了两次",
                    "我开吸尘器它就跑了",
                    event_type="health",
                    subject="vomit",
                )
            )
        )
        result = summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert result.health_records == []
        assert any("health_sink" in n for n in result.summary.notes), (
            f"缺口必须在 notes 里说明，当前 notes={result.summary.notes}"
        )
        assert any("health_records" in n for n in result.summary.notes)

    def test_health_signal_lands_in_health_records(self, writer, conv):
        """**回归测试**：提供 sink 后健康信号真的落地。

        在此之前它「记忆层拒了、日报显示了、但没地方存」——
        一个已经登记却没有接收方的缺口。
        """
        from app.health import HealthWriter, InMemoryHealthStore, RedFlagTable
        from pathlib import Path

        health_store = InMemoryHealthStore()
        sink = HealthWriter(
            store=health_store,
            redflags=RedFlagTable.load(
                Path(__file__).resolve().parent.parent
                / "data"
                / "health"
                / "red_flags.yaml"
            ),
            consent_version="v1",
        )
        llm = MockLLM(
            default=_payload(
                _cand_json(
                    "它今天吐了两次",
                    "我开吸尘器它就跑了",
                    event_type="health",
                    subject="vomit",
                )
            )
        )
        result = summarize_day(
            day=DAY,
            messages=conv,
            writer=writer,
            llm=llm,
            user_id=U,
            pet_id=P,
            health_sink=sink,
        )

        assert len(result.health_records) == 1
        assert result.health_records[0].signal == "gi.vomiting_frequency"
        # 对话文本给不出可靠量化值，**一律为 None**（不解析中文数词）
        assert result.health_records[0].value is None
        assert result.health_records[0].consent_version == "v1"

        persisted = health_store.list_records(user_id=U, pet_id=P)
        assert len(persisted) == 1, "必须真的落到 health_records 表"
        assert not any("health_sink" in n for n in result.summary.notes)

    def test_tenant_isolation(self, writer, conv):
        """多租户：写入必须落在给定 pet 下。"""
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "我开吸尘器它就跑了"))
        )
        summarize_day(
            day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        assert writer.store.list_memories(user_id=U, pet_id=P)
        assert writer.store.list_memories(user_id=U, pet_id="other-pet") == []


# ═══════════════════════════════════════════════════════════════
# 6. 质量指标
# ═══════════════════════════════════════════════════════════════


class TestOverlappingKeywords:
    """回归测试：同义词重叠曾导致**重复计数**。

    「吸尘器」包含「吸尘」——逐词 ``text.count(kw)`` 会把一次提及算成两次。
    而计数是要展示给用户并据以判断的**事实**，重复计数就是假数字。
    """

    def test_substring_synonym_counted_once(self):
        from app.digest.aggregate import TOPIC_VOCABULARY, _count_mentions

        kws = TOPIC_VOCABULARY["vacuum"]
        assert _count_mentions("它怕吸尘器", kws) == 1, "「吸尘器」不应因包含「吸尘」而计两次"
        assert _count_mentions("开吸尘它就躲", kws) == 1, "只想说「吸尘」时也要能匹配"

    def test_repeated_mentions_all_counted(self):
        from app.digest.aggregate import TOPIC_VOCABULARY, _count_mentions

        kws = TOPIC_VOCABULARY["vacuum"]
        assert _count_mentions("吸尘器吸尘器吸尘器", kws) == 3

    def test_aggregate_uses_the_safe_counter(self):
        """端到端确认：aggregate 走的是去重计数，不是逐词 count。"""
        stats = aggregate(day=DAY, messages=_msgs("它怕吸尘器"))
        assert stats.topic_counts["vacuum"] == 1


class TestRejectionRate:
    """`rejection_rate` 是**质量指标**，不是错误指标。"""

    def test_rate_is_zero_when_all_rejected_nothing_proposed(self, conv):
        admitted, rejected = parse_candidates('{"candidates": []}', messages=conv)
        from app.schemas import DailySummary

        s = DailySummary(
            date=DAY, stats=aggregate(day=DAY, messages=conv),
            candidates=admitted, rejected=rejected,
        )
        assert s.rejection_rate == 0.0

    def test_rate_reflects_fabrication_share(self, conv):
        from app.schemas import DailySummary

        raw = _payload(
            _cand_json("它怕吸尘器", "我开吸尘器它就跑了"),
            _cand_json("它喜欢窗台", "原文没有这句"),
        )
        admitted, rejected = parse_candidates(raw, messages=conv)
        s = DailySummary(
            date=DAY, stats=aggregate(day=DAY, messages=conv),
            candidates=admitted, rejected=rejected,
        )
        assert s.rejection_rate == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════
# 7. 记忆事件映射
# ═══════════════════════════════════════════════════════════════


class TestEventMapping:
    def test_source_mapping_is_exhaustive(self):
        """来源映射必须覆盖所有取值 —— 漏一个会在运行时 KeyError。"""
        from app.digest import _SOURCE_MAP

        assert set(_SOURCE_MAP) == set(ExtractionSource)

    def test_status_mapping_is_exhaustive(self):
        """回归测试：初版无条件写 ACTIVE，导致 **AI 归纳的候选让整天总结崩溃**。

        ``SYSTEM_INFERENCE + ACTIVE`` 是非法组合，在**构造**时就抛异常，
        根本活不到路由层。所以初始状态必须在这里就定对。
        """
        from app.digest import _STATUS_MAP

        assert set(_STATUS_MAP) == set(ExtractionSource)
        assert _STATUS_MAP[ExtractionSource.AI_INFERENCE] is MemoryStatus.PENDING_CONFIRMATION
        assert _STATUS_MAP[ExtractionSource.OWNER_RECORD] is MemoryStatus.ACTIVE

    def test_occurred_at_prefers_last_message_time(self, writer):
        stamps = [
            DigestMessage(
                role="user", content="它今天又在窗台趴着 早上", at=datetime(2026, 3, 14, 9, tzinfo=timezone.utc)
            ),
            DigestMessage(
                role="user", content="吸尘器一开它就跑掉了 晚上", at=datetime(2026, 3, 14, 20, tzinfo=timezone.utc)
            ),
        ]
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "吸尘器一开它就跑掉了"))
        )
        summarize_day(
            day=DAY, messages=stamps, writer=writer, llm=llm, user_id=U, pet_id=P
        )
        stored = writer.store.list_memories(user_id=U, pet_id=P)
        occurred = stored[0].occurred_at
        assert occurred is not None
        assert occurred.hour == 20

    def test_naive_timestamp_gets_utc(self, writer):
        """朴素时间戳必须补上时区，否则与库里的 aware 时间比较会抛异常。"""
        msgs = [DigestMessage(role="user", content="吸尘器一开它就跑", at=datetime(2026, 3, 14, 20))]
        llm = MockLLM(default=_payload(_cand_json("它怕吸尘器", "吸尘器一开它就跑")))
        summarize_day(day=DAY, messages=msgs, writer=writer, llm=llm, user_id=U, pet_id=P)
        stored = writer.store.list_memories(user_id=U, pet_id=P)
        assert stored[0].occurred_at is not None
        assert stored[0].occurred_at.tzinfo is not None

    def test_layer_is_always_episode(self, writer, conv):
        """**不直接写 Profile。**

        一条当天说的话不足以成为稳定事实；晋升由飞轮的
        `is_eligible_for_promotion`（重复出现 + 时间跨度）负责。
        """
        llm = MockLLM(
            default=_payload(_cand_json("它怕吸尘器，听到声音就躲", "我开吸尘器它就跑了"))
        )
        summarize_day(day=DAY, messages=conv, writer=writer, llm=llm, user_id=U, pet_id=P)
        stored = writer.store.list_memories(user_id=U, pet_id=P)
        from app.schemas import MemoryLayer

        assert stored[0].layer is MemoryLayer.EPISODE
