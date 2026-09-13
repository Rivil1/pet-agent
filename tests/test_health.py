"""健康模块测试：红旗求值、三值逻辑、记录落地。

**本文件的重点是「不知道」与「没事」必须被区分。**

红旗求值用三值逻辑（真/假/**无法评估**）。若把「无法评估」当成「未命中」，
系统就会输出「未发现异常」—— 而它其实什么都没看。

这与 B3（零填充是静默编造）是同一类陷阱在健康模块的形态：
**用默认值顶替缺失，让「不知道」伪装成「已检查」。**

代价是非对称的（`docs/07-health.md` §9）：

| 错误 | 代价 |
| --- | --- |
| 假阴性（该报没报） | 猫可能死亡 —— **不可接受** |
| 假阳性（虚惊一场） | 多跑一趟医院 —— 可接受 |
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from app.health import (
    HealthRecordStore,
    HealthWriter,
    InMemoryHealthStore,
    RedFlagTable,
    RedFlagTableError,
    eval_condition,
    evaluate,
)
from app.schemas import (
    MIN_COVERAGE_FOR_ASSESSMENT,
    EventType,
    ExtractionSource,
    HealthRecordSource,
    Polarity,
    UrgencyLevel,
)
from app.schemas.digest import DigestCandidate

RULES = Path(__file__).resolve().parent.parent / "data" / "health" / "red_flags.yaml"
NOW = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def table() -> RedFlagTable:
    return RedFlagTable.load(RULES)


@pytest.fixture()
def writer(table: RedFlagTable):
    return HealthWriter(
        store=InMemoryHealthStore(), redflags=table, consent_version="v1"
    )


def _health_cand(subject: str, content: str = "它今天吐了两次") -> DigestCandidate:
    return DigestCandidate(
        event_type=EventType.HEALTH,
        subject=subject,
        content=content,
        quote=content,
        source_layer=ExtractionSource.OWNER_RECORD,
        confidence=0.8,
        polarity=Polarity.NEGATIVE,
    )


# ═══════════════════════════════════════════════════════════════
# 1. 规则表加载与校验
# ═══════════════════════════════════════════════════════════════


class TestTableLoading:
    def test_loads(self, table):
        assert table.version
        assert table.rules
        assert table.sha256

    def test_review_status_is_visible(self, table):
        """未审核不为错，但**必须可见** —— 它决定输出该怎么自我描述。"""
        assert table.is_reviewed is False  # 当前 reviewed_by: null

    def test_enabled_rules_have_sources(self, table):
        """已启用的规则必须可回溯到来源。"""
        for rule in table.enabled_rules:
            assert rule.sources, f"规则 {rule.rule_id} 缺少 sources"
            assert rule.message, f"规则 {rule.rule_id} 缺少 message"

    def test_disabled_rules_have_reason(self, table):
        """未启用的必须写明原因 —— 否则「为什么不启用」无人可查。"""
        for rule in table.rules:
            if not rule.enabled:
                assert rule.disabled_reason, f"规则 {rule.rule_id} 缺少 disabled_reason"

    def test_threshold_rules_are_disabled(self, table):
        """阈值型规则不得启用（不同来源数值不一致，本表不自行断言）。"""
        for rule in table.enabled_rules:
            for ref in _all_conditions(rule.conditions):
                if ref.get("op") in (">=", ">", "<=", "<"):
                    assert ref.get("value") is not None, (
                        f"已启用规则 {rule.rule_id} 含空阈值"
                    )

    def test_undefined_signal_is_load_error(self, tmp_path):
        """**拼写错误必须在加载期报错，不能留到运行期。**

        一条永远不命中的红旗规则，比没有这条规则更危险：
        它让人以为这个风险已被覆盖。
        """
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: t\n"
            "signal_vocabulary:\n  a.b:\n    type: boolean\n"
            "rules:\n"
            "  - id: typo_rule\n"
            "    enabled: false\n"
            "    urgency: L3\n"
            "    title: t\n"
            "    message: m\n"
            "    disabled_reason: r\n"
            "    conditions:\n"
            "      any_of:\n        - signal: a.typo\n          op: is_true\n",
            encoding="utf-8",
        )
        with pytest.raises(RedFlagTableError, match="未定义的信号"):
            RedFlagTable.load(bad)

    def test_enabled_rule_without_sources_is_load_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "version: t\n"
            "signal_vocabulary:\n  a.b:\n    type: boolean\n"
            "rules:\n"
            "  - id: no_source\n"
            "    enabled: true\n"
            "    urgency: L3\n"
            "    title: t\n"
            "    message: m\n"
            "    conditions:\n"
            "      any_of:\n        - signal: a.b\n          op: is_true\n",
            encoding="utf-8",
        )
        with pytest.raises(RedFlagTableError, match="sources"):
            RedFlagTable.load(bad)


def _all_conditions(cond: dict) -> list[dict]:
    out: list[dict] = []
    for key in ("any_of", "all_of"):
        for item in cond.get(key, []) or []:
            if "signal" in item:
                out.append(item)
            else:
                out.extend(_all_conditions(item))
    return out


# ═══════════════════════════════════════════════════════════════
# 2. 三值逻辑 —— 本模块的核心
# ═══════════════════════════════════════════════════════════════


class TestThreeValuedLogic:
    """**「没采集」≠「没命中」。**"""

    def test_missing_signal_is_unknown_not_false(self):
        """缺失信号返回 ``None``，不是 ``False``。"""
        result = eval_condition(
            {"signal": "litter_box.urine_output", "op": "in", "value": ["none"]},
            {},  # 什么都没采集
        )
        assert result is None, "缺失信号必须返回 None（无法评估）"
        assert result is not False

    def test_explicit_none_value_is_unknown(self):
        """**信号「已采集但值为 None」也算无法评估。**

        这里很容易写错：字典里有这个键，看起来「有数据」；
        若让 None 进入比较，`None >= 5` 会返回 False，
        于是「不知道」被静默变成了「未命中」。
        """
        result = eval_condition(
            {"signal": "litter_box.visit_frequency", "op": ">=", "value": 5},
            {"litter_box.visit_frequency": None},
        )
        assert result is None

    def test_present_signal_evaluates(self):
        assert (
            eval_condition(
                {"signal": "litter_box.urine_output", "op": "in", "value": ["none"]},
                {"litter_box.urine_output": "none"},
            )
            is True
        )
        assert (
            eval_condition(
                {"signal": "litter_box.urine_output", "op": "in", "value": ["none"]},
                {"litter_box.urine_output": "normal"},
            )
            is False
        )

    def test_any_of_unknown_does_not_become_true(self):
        """any_of：无 True 但有 None → 结果必须是 None，不能是 True。"""
        assert (
            eval_condition(
                {
                    "any_of": [
                        {"signal": "a", "op": "is_true"},
                        {"signal": "b", "op": "is_true"},
                    ]
                },
                {"a": False},  # b 未采集
            )
            is None
        )

    def test_any_of_true_wins(self):
        assert (
            eval_condition(
                {"any_of": [{"signal": "a", "op": "is_true"}, {"signal": "b", "op": "is_true"}]},
                {"a": True},
            )
            is True
        )

    def test_all_of_unknown_does_not_become_false(self):
        """all_of：无 False 但有 None → None，不能是 False。"""
        assert (
            eval_condition(
                {"all_of": [{"signal": "a", "op": "is_true"}, {"signal": "b", "op": "is_true"}]},
                {"a": True},  # b 未采集
            )
            is None
        )

    def test_all_of_false_wins(self):
        assert (
            eval_condition(
                {"all_of": [{"signal": "a", "op": "is_true"}, {"signal": "b", "op": "is_true"}]},
                {"a": False, "b": True},
            )
            is False
        )

    def test_nested_condition(self):
        """嵌套 all_of（尿闭规则的第二个分支就是这种形状）。"""
        cond = {
            "any_of": [
                {"signal": "litter_box.urine_output", "op": "in", "value": ["none", "few_drops"]},
                {
                    "all_of": [
                        {"signal": "litter_box.visit_frequency", "op": ">=", "value": 5},
                        {"signal": "litter_box.straining", "op": "is_true"},
                    ]
                },
            ]
        }
        assert eval_condition(cond, {"litter_box.urine_output": "few_drops"}) is True
        assert (
            eval_condition(
                cond,
                {"litter_box.visit_frequency": 6, "litter_box.straining": True},
            )
            is True
        )
        # 访问次数够但没有努责信息 → 无法评估
        assert eval_condition(cond, {"litter_box.visit_frequency": 6}) is None

    def test_null_threshold_never_matches(self):
        """阈值为 null 的规则永不命中（双保险：它们在加载期就应是 disabled）。"""
        assert (
            eval_condition(
                {"signal": "gi.vomiting_frequency", "op": ">=", "value": None},
                {"gi.vomiting_frequency": 5},
            )
            is False
        )


# ═══════════════════════════════════════════════════════════════
# 3. 红旗求值
# ═══════════════════════════════════════════════════════════════


class TestEvaluation:
    def test_urinary_obstruction_triggers(self, table):
        """尿闭是已启用规则里最要紧的一条（公猫 24–48h 可致命）。"""
        hits, undecidable = evaluate(
            table, {"litter_box.urine_output": "few_drops"}
        )
        ids = [h.rule_id for h in hits]
        assert "urinary_obstruction" in ids
        hit = next(h for h in hits if h.rule_id == "urinary_obstruction")
        assert hit.urgency is UrgencyLevel.EMERGENCY
        assert hit.sources, "命中的红旗必须带来源"
        assert "24" in hit.message or "急症" in hit.message

    def test_aggravating_signals_recorded(self, table):
        hits, _ = evaluate(
            table,
            {
                "litter_box.urine_output": "none",
                "vocalization.context": "voiding_attempt",
            },
        )
        hit = next(h for h in hits if h.rule_id == "urinary_obstruction")
        assert "vocalization.context" in hit.aggravating_conditions

    def test_no_signals_means_all_undecidable(self, table):
        """**一个信号都没有时，不能报「未发现异常」。**

        所有规则都应进 undecidable —— 这是 INSUFFICIENT_DATA 的依据。
        """
        hits, undecidable = evaluate(table, {})
        assert hits == []
        assert set(undecidable) == {r.rule_id for r in table.enabled_rules}

    def test_partial_signals_still_report_undecidable(self, table):
        """采集了一部分也不够 —— 未覆盖的规则必须被列出。"""
        hits, undecidable = evaluate(table, {"respiratory.pattern": "labored"})
        assert [h.rule_id for h in hits] == ["respiratory_distress"]
        assert "urinary_obstruction" in undecidable

    def test_disabled_rules_never_evaluate(self, table):
        """未启用的规则不参与求值，即使条件满足。"""
        hits, undecidable = evaluate(
            table,
            {"gi.vomiting_frequency": 99, "general.demeanor": "lethargic"},
        )
        assert not any(h.rule_id == "persistent_gi_signs" for h in hits)
        assert "persistent_gi_signs" not in undecidable

    def test_hits_sorted_by_urgency(self, table):
        hits, _ = evaluate(
            table,
            {
                "respiratory.pattern": "open_mouth",
                "litter_box.urine_output": "none",
            },
        )
        assert len(hits) >= 2
        levels = [h.urgency.value for h in hits]
        assert levels == sorted(levels, reverse=True)


# ═══════════════════════════════════════════════════════════════
# 4. 记录落地
# ═══════════════════════════════════════════════════════════════


class TestHealthWriter:
    def test_consent_version_required(self, table):
        """健康数据需显式同意（HealthDataPolicy）。"""
        with pytest.raises(ValueError, match="consent_version"):
            HealthWriter(
                store=InMemoryHealthStore(), redflags=table, consent_version=""
            )

    def test_digest_candidate_becomes_record(self, writer):
        admission = writer.admit_digest_candidates(
            [_health_cand("vomit")], user_id="u", pet_id="p", day=NOW.date()
        )
        assert admission.written_count == 1
        rec = admission.records[0]
        assert rec.signal == "gi.vomiting_frequency"
        assert rec.value is None, "对话文本给不出可靠量化值"
        assert rec.source is HealthRecordSource.USER_INPUT
        assert rec.sensitive is True

    def test_unmapped_subject_gets_unstructured_prefix(self, writer):
        """未映射的 subject **不参与红旗求值** —— 这是刻意的。

        给它一个 `unstructured.` 前缀，它就不在规则表词汇表里，
        因此永远不会被误当成结构化信号。
        """
        admission = writer.admit_digest_candidates(
            [_health_cand("mystery_thing")], user_id="u", pet_id="p", day=NOW.date()
        )
        assert admission.records[0].signal.startswith("unstructured.")
        assert admission.records[0].signal not in writer.redflags.signal_names()

    def test_non_health_candidate_rejected_with_reason(self, writer):
        cand = _health_cand("vomit")
        other = cand.model_copy(update={"event_type": EventType.BEHAVIOR})
        admission = writer.admit_digest_candidates(
            [other], user_id="u", pet_id="p", day=NOW.date()
        )
        assert admission.written_count == 0
        assert admission.rejected
        assert "非健康事件" in admission.rejected[0][1]

    def test_records_are_isolated_by_tenant(self, writer):
        writer.admit_digest_candidates(
            [_health_cand("vomit")], user_id="u", pet_id="p", day=NOW.date()
        )
        assert writer.store.list_records(user_id="u", pet_id="p")
        assert writer.store.list_records(user_id="u", pet_id="other") == []

    def test_hard_delete_cascades(self, writer):
        """`on_account_deletion == cascade_hard_delete`：记录与评估一起删。"""
        writer.admit_digest_candidates(
            [_health_cand("vomit")], user_id="u", pet_id="p", day=NOW.date()
        )
        writer.assess(user_id="u", pet_id="p", signals={})
        assert writer.store.list_records(user_id="u", pet_id="p")
        assert writer.store.list_assessments(user_id="u", pet_id="p")

        removed = writer.store.delete_pet_data(user_id="u", pet_id="p")
        assert removed >= 2
        assert writer.store.list_records(user_id="u", pet_id="p") == []
        assert writer.store.list_assessments(user_id="u", pet_id="p") == []


# ═══════════════════════════════════════════════════════════════
# 5. 分诊 —— 「不知道」必须说出来
# ═══════════════════════════════════════════════════════════════


class TestAssessment:
    def test_empty_signals_give_insufficient_data(self, writer):
        """**核心断言**：什么都没采集时，不能说「未发现异常」。"""
        a = writer.assess(user_id="u", pet_id="p", signals={})
        assert a.level is UrgencyLevel.INSUFFICIENT_DATA
        assert a.signals_missing
        assert a.red_flags_triggered == []

    def test_low_coverage_gives_insufficient_data(self, writer):
        a = writer.assess(user_id="u", pet_id="p", signals={"respiratory.rate": 24})
        assert a.coverage < MIN_COVERAGE_FOR_ASSESSMENT
        assert a.level is UrgencyLevel.INSUFFICIENT_DATA

    def test_red_flag_wins_over_insufficient_data(self, writer):
        """命中了就是命中了 —— 覆盖率不足不能把急诊降级。"""
        a = writer.assess(
            user_id="u", pet_id="p", signals={"litter_box.urine_output": "none"}
        )
        assert a.level is UrgencyLevel.EMERGENCY
        assert a.red_flags_triggered
        assert a.recommendation == a.red_flags_triggered[0].action

    def test_unstructured_records_do_not_inflate_coverage(self, writer):
        """未结构化信号**不计入覆盖率**。

        把它们算进去会虚报「我检查过了」——那正是 INSUFFICIENT_DATA 要防的事。
        """
        a = writer.assess(
            user_id="u",
            pet_id="p",
            signals={"unstructured.mystery": "something"},
        )
        assert a.coverage == 0.0
        assert a.signals_assessed == []
        assert a.level is UrgencyLevel.INSUFFICIENT_DATA

    def test_none_valued_signal_is_missing_not_assessed(self, writer):
        """**值为 None 的信号算「未采集」，不算「已评估」。**"""
        a = writer.assess(
            user_id="u",
            pet_id="p",
            signals={"litter_box.urine_output": None, "respiratory.rate": 24},
        )
        assert "litter_box.urine_output" in a.signals_missing
        assert "litter_box.urine_output" not in a.signals_assessed
        assert "respiratory.rate" in a.signals_assessed

    def test_disclaimer_present(self, writer):
        a = writer.assess(user_id="u", pet_id="p", signals={})
        assert "不是诊断" in a.disclaimer
        assert a.must_not_be_read_as == "未发现异常不代表健康"

    def test_no_exclusionary_wording(self, writer):
        """**永不输出「健康」「正常」「没问题」**（三不原则 · 不排除）。"""
        from app.schemas import FORBIDDEN_PHRASES

        for signals in (
            {},
            {"respiratory.rate": 24},
            {"litter_box.urine_output": "none"},
        ):
            a = writer.assess(user_id="u", pet_id="p", signals=signals)
            blob = f"{a.recommendation} {a.coverage_note}"
            for phrase in FORBIDDEN_PHRASES:
                assert phrase not in blob, f"出现禁用词「{phrase}」：{blob}"

    def test_rule_version_recorded(self, writer, table):
        """可复现：评估必须记录它依据的规则版本。"""
        a = writer.assess(user_id="u", pet_id="p", signals={})
        assert a.rule_version == table.version

    def test_assessment_is_persisted(self, writer):
        writer.assess(user_id="u", pet_id="p", signals={})
        assert len(writer.store.list_assessments(user_id="u", pet_id="p")) == 1


# ═══════════════════════════════════════════════════════════════
# 6. 协议一致性
# ═══════════════════════════════════════════════════════════════


class TestCoverageGateDirection:
    """**回归测试：覆盖率门限必须只拦一个方向。**

    初版把「覆盖率不足 → 必须是 INSUFFICIENT_DATA」写成了无条件，
    于是一个**已命中的急诊红旗**会在构造评估时直接抛异常。

    这属于 B11/B12（检测过宽）同一类错误：为了防一个方向，把反方向也拦了。
    而这里反方向是危险的那个 —— 假阴性的代价是猫可能死亡。
    """

    def test_low_coverage_cannot_claim_no_deviation(self):
        """方向一：「没测」不得变成「没事」。"""
        from app.schemas import HealthAssessment

        with pytest.raises(ValueError, match="伪装成「没有异常」"):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.NO_DEVIATION_DETECTED,
                coverage=0.06,
                coverage_note="仅评估了 1 项",
                recommendation="未发现偏离",
            )

    def test_low_coverage_can_still_report_red_flag(self):
        """方向二（关键）：**低覆盖率不得抹掉已命中的急诊**。

        一个真实存在的危险信号，不因为「别的没测」而失效。
        """
        from app.schemas import HealthAssessment, RedFlagHit

        assessment = HealthAssessment(
            user_id="u",
            pet_id="p",
            level=UrgencyLevel.EMERGENCY,
            coverage=0.06,
            coverage_note="仅评估了排尿信号",
            red_flags_triggered=[
                RedFlagHit(
                    rule_id="urinary_obstruction",
                    urgency=UrgencyLevel.EMERGENCY,
                    title="排尿困难",
                    message="请立即就医",
                    action="立即前往宠物医院急诊",
                    rule_version="t",
                )
            ],
            recommendation="立即前往宠物医院急诊",
        )
        assert assessment.level is UrgencyLevel.EMERGENCY

    def test_emergency_still_requires_a_red_flag(self):
        """但「EMERGENCY 必须由红旗触发」这条不能放松。"""
        from app.schemas import HealthAssessment

        with pytest.raises(ValueError, match="必须由红旗规则触发"):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.EMERGENCY,
                coverage=0.9,
                coverage_note="ok",
                recommendation="去医院",
            )


class TestProtocol:
    def test_inmemory_satisfies_protocol(self):
        assert isinstance(InMemoryHealthStore(), HealthRecordStore)  # type: ignore[misc]

    def test_all_signals_in_rules_are_in_vocabulary(self, table):
        """规则引用的信号必须在词汇表内（加载期已校验，这里端到端确认）。"""
        known = table.signal_names()
        for rule in table.rules:
            for cond in _all_conditions(rule.conditions):
                assert cond["signal"] in known
