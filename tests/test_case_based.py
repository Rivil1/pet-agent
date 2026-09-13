"""案例推理、冷启动与模式路由的测试。

## 本文件守护的核心主张

**「占位先验永不产生后验概率」** —— 这是 D31 的 fail-closed 保证。

它不是靠调用方自觉，而是 `PriorTable.is_placeholder` 一个标志位决定的。
在它之前，图的 `behavior_interpreter` 节点直接用占位先验算后验，
于是输出里的 `0.41` 是一个**编造的数字**。

## 其余机制性断言

| 断言 | 抓的是什么 |
| --- | --- |
| 相似度只比共同特征 | 缺失特征被当成「相同」→ 相似度虚高 |
| 共同特征不足返回 None | 「无法比较」与「不相似」必须分开 |
| 计数阈值按**情境**而非总数 | 总样本 8 条分散在 6 个情境 → 「3 次里 2 次」是噪声 |
| 未确认记录不参与 | 未确认的样本不构成证据 |
| MEASURED_ONLY 无概率字段 | 没有模型就不该有数字 |
| 参考系不用群体先验 | 「高于参考值」不能建立在编造的均值上 |
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from app.audio.features import TARGET_SR, extract_features, synthesize_meow
from app.interpreter import (
    FEATURE_SCALE,
    MIN_CASES_PER_CONTEXT,
    choose_mode,
    interpret_meow,
    match_cases,
    similarity_between,
)
from app.interpreter.case_based import MIN_COMMON_FEATURES, SIMILARITY_FLOOR
from app.interpreter.measured import match_scene
from app.interpreter.priors import PriorTable
from app.schemas import (
    AcousticFeatures,
    BehaviorAction,
    ContextLabel,
    EvidenceMode,
    EvidenceKind,
    FeatureQuality,
    MemorySource,
    MemoryStatus,
    MeowRecord,
)

PRIOR_PATH = "data/priors/catmeows_stats.json"
NOW = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────


def feats(
    *, f0_start: float = 500.0, f0_end: float = 700.0, dur: float = 0.7, rough: float = 0.0
) -> AcousticFeatures:
    """合成一段叫声并提取特征。**离线、可复现。**"""
    return extract_features(
        synthesize_meow(
            duration=dur, sr=TARGET_SR, f0_start=f0_start, f0_end=f0_end, roughness=rough
        ),
        TARGET_SR,
    )


def rec(
    ctx: ContextLabel,
    *,
    resolution: str | None = None,
    rid: str = "r",
    f: AcousticFeatures | None = None,
    source: MemorySource = MemorySource.USER_OBSERVATION,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    days_ago: int = 0,
) -> MeowRecord:
    return MeowRecord(
        record_id=rid,
        user_id="u",
        pet_id="p",
        context=ctx,
        features=f or feats(),
        actions=[BehaviorAction.SCRATCH_DOOR],
        resolution=resolution,
        recorded_at=NOW - timedelta(days=days_ago),
        source=source,
        status=status,
    )


@pytest.fixture(scope="module")
def placeholder_prior() -> PriorTable:
    """真实的占位先验（`is_placeholder: true`）。**路由必须因它降级。**"""
    table = PriorTable.load(PRIOR_PATH)
    assert table.is_placeholder, "本文件假设先验仍为占位数据"
    return table


# ═══════════════════════════════════════════════════════════════
# 1. 相似度
# ═══════════════════════════════════════════════════════════════


class TestSimilarity:
    def test_identical_features_are_maximally_similar(self):
        f = feats()
        sim, common = similarity_between(f, f)
        assert sim == pytest.approx(1.0)
        assert len(common) >= MIN_COMMON_FEATURES

    def test_similar_meows_score_high(self):
        sim, _ = similarity_between(feats(f0_start=500), feats(f0_start=505))
        assert sim is not None and sim > 0.9

    def test_very_different_meows_score_low(self):
        sim, _ = similarity_between(
            feats(f0_start=300, dur=0.5), feats(f0_start=900, dur=1.2, rough=0.5)
        )
        assert sim is not None and sim < 0.8

    def test_insufficient_common_features_returns_none(self):
        """**关键断言**：共同特征不足时返回 ``None``（无法比较），不是低相似度。

        若返回一个低分，「无法比较」会被当成「不相似」参与分母 ——
        而事实是我们根本不知道。这两者在计数输出里差别很大。
        """
        a = feats()
        # 把绝大多数特征标记为不可用
        thin = a.model_copy(
            update={
                "unavailable": [
                    "duration",
                    "f0_range",
                    "f0_slope",
                    "call_rate",
                    "ici_mean",
                    "rms_mean",
                ]
            }
        )
        sim, common = similarity_between(thin, a)
        assert sim is None
        assert len(common) < MIN_COMMON_FEATURES

    def test_unavailable_features_are_excluded_from_comparison(self):
        """不可用特征不参与比较 —— 否则缺失会被当成「相同」而虚高相似度。"""
        a = feats()
        b = a.model_copy(update={"duration": 99.0, "unavailable": ["duration"]})
        sim, common = similarity_between(a, b)
        assert sim == pytest.approx(1.0), "被标记不可用的特征不应影响相似度"
        assert "duration" not in common

    def test_every_feature_has_a_scale(self):
        """每个特征都必须有归一尺度 —— 漏一个会在运行时 KeyError。"""
        from app.interpreter.case_based import FEATURE_ORDER

        assert set(FEATURE_ORDER) == set(FEATURE_SCALE)
        assert all(v > 0 for v in FEATURE_SCALE.values())

    def test_scale_matters_for_cross_dimension_comparison(self):
        """不同量纲必须可比：基频差 80Hz 与时长差 0.3s 应视为**同等显著**。"""
        a = feats(f0_start=500, f0_end=700)
        b = a.model_copy(
            update={"f0_mean": a.f0_mean + FEATURE_SCALE["f0_mean"]}
        )
        c = a.model_copy(
            update={"duration": a.duration + FEATURE_SCALE["duration"]}
        )
        sim_b, _ = similarity_between(a, b)
        sim_c, _ = similarity_between(a, c)
        assert sim_b == pytest.approx(sim_c, abs=1e-6)


# ═══════════════════════════════════════════════════════════════
# 2. 匹配与计数
# ═══════════════════════════════════════════════════════════════


class TestMatching:
    def test_unconfirmed_records_do_not_participate(self):
        """未确认的样本不构成证据（与 `SimilarSample.context is None` 同理）。"""
        pending = rec(
            ContextLabel.DOOR_ATTENTION,
            rid="pending",
            status=MemoryStatus.PENDING_CONFIRMATION,
        )
        assert match_cases(feats(), [pending]) == []

    def test_system_inference_cannot_be_active(self):
        """与 `MemoryEvent` 同一条不变量：防自我强化。"""
        with pytest.raises(ValueError, match="SYSTEM_INFERENCE"):
            rec(
                ContextLabel.DOOR_ATTENTION,
                source=MemorySource.SYSTEM_INFERENCE,
                status=MemoryStatus.ACTIVE,
            )

    def test_below_floor_is_not_a_match(self):
        far = rec(
            ContextLabel.DOOR_ATTENTION,
            f=feats(f0_start=300, f0_end=400, dur=1.2, rough=0.6),
        )
        matches = match_cases(feats(f0_start=900, f0_end=950), [far])
        assert all(m.similarity >= SIMILARITY_FLOOR for m in matches)

    def test_matches_sorted_by_similarity(self):
        recs = [
            rec(ContextLabel.DOOR_ATTENTION, rid="a", f=feats(f0_start=500)),
            rec(ContextLabel.DOOR_ATTENTION, rid="b", f=feats(f0_start=520)),
            rec(ContextLabel.DOOR_ATTENTION, rid="c", f=feats(f0_start=560)),
        ]
        matches = match_cases(feats(f0_start=498), recs)
        sims = [m.similarity for m in matches]
        assert sims == sorted(sims, reverse=True)

    def test_common_features_are_recorded(self):
        """参与比较的特征必须可审计 —— 能看出相似度基于什么算出来的。"""
        matches = match_cases(feats(), [rec(ContextLabel.GREETING, rid="a")])
        assert matches
        assert matches[0].common_features


# ═══════════════════════════════════════════════════════════════
# 3. 阈值语义：按情境，不按总数（D31）
# ═══════════════════════════════════════════════════════════════


class TestThresholdSemantics:
    def test_threshold_is_per_context_not_total(self):
        """**D31 的核心**：8 条总样本分散在 6 个情境时，不应给出任何计数。

        总样本 8 条看起来不少，但每个情境只有 1–2 条 ——
        「3 次里 2 次」就成了噪声。
        """
        recs = [
            rec(ContextLabel.DOOR_ATTENTION, rid="d1"),
            rec(ContextLabel.GREETING, rid="g1"),
            rec(ContextLabel.FOOD_WAITING, rid="f1"),
            rec(ContextLabel.OTHER, rid="o1"),
            rec(ContextLabel.AFFECTION_BRUSHING, rid="a1"),
            rec(ContextLabel.DOOR_ATTENTION, rid="d2"),
        ]
        f = feats()
        matches = match_cases(f, recs)
        assert len(matches) >= 6, "总样本足以匹配"

        # 但没有任何情境达到门限
        from app.interpreter.case_based import count_by_context

        counts = count_by_context(matches)
        assert max(counts.values()) < MIN_CASES_PER_CONTEXT

        _, decision = interpret_meow(features=f, records=recs)
        assert decision.mode is EvidenceMode.MEASURED_ONLY, (
            "总样本够但单情境不够时，必须走冷启动"
        )

    def test_enough_in_one_context_qualifies(self):
        recs = [
            rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)
        ]
        _, decision = interpret_meow(features=feats(), records=recs)
        assert decision.mode is EvidenceMode.CASE_BASED
        assert decision.qualified_contexts == 1

    def test_case_based_candidates_ordered_by_count(self):
        recs = [
            rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(4)
        ] + [rec(ContextLabel.GREETING, rid=f"g{i}") for i in range(3)]
        result, decision = interpret_meow(features=feats(), records=recs)
        assert decision.mode is EvidenceMode.CASE_BASED
        counts = [c.matched_count for c in result.candidates]
        assert counts == sorted(counts, reverse=True)
        assert result.candidates[0].context is ContextLabel.DOOR_ATTENTION


# ═══════════════════════════════════════════════════════════════
# 4. 契约：计数模式禁止概率
# ═══════════════════════════════════════════════════════════════


class TestCaseBasedContract:
    def test_no_posterior_no_log_odds(self):
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert all(c.posterior is None for c in result.candidates)
        assert all(c.log_odds is None for c in result.candidates)

    def test_no_log_odds_contribution(self):
        """**填 0.0 会谎称「已参与计算但影响为零」** —— 而本模式没有任何概率计算。"""
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert all(e.log_odds_contribution is None for e in result.evidence)

    def test_similar_cases_carry_content(self):
        """案例必须带**具体内容** —— 用户才能核查「上次是你开了门它出去了」。"""
        recs = [
            rec(ContextLabel.DOOR_ATTENTION, resolution="开门它就出去了", rid=f"d{i}")
            for i in range(3)
        ]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert result.similar_cases
        assert any(c.resolution == "开门它就出去了" for c in result.similar_cases)
        assert all(c.actions == [BehaviorAction.SCRATCH_DOOR] for c in result.similar_cases)

    def test_case_total_is_the_denominator(self):
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert result.case_total == len(match_cases(feats(), recs))
        assert all(c.matched_count <= result.case_total for c in result.candidates)

    def test_no_prior_version(self):
        """案例推理不使用群体先验 —— 报告一个 prior_version 会让人以为用了。"""
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert result.prior_version is None

    def test_reproducible(self):
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        a, _ = interpret_meow(features=feats(), records=recs)
        b, _ = interpret_meow(features=feats(), records=recs)
        assert a.model_dump() == b.model_dump()


# ═══════════════════════════════════════════════════════════════
# 5. 冷启动
# ═══════════════════════════════════════════════════════════════


class TestMeasuredOnly:
    def test_no_probability_fields(self):
        """没有模型就不该有数字。契约也会校验这一点。"""
        result, decision = interpret_meow(features=feats(), records=[])
        assert decision.mode is EvidenceMode.MEASURED_ONLY
        assert all(c.posterior is None for c in result.candidates)
        assert all(c.log_odds is None for c in result.candidates)
        assert all(c.matched_count == 0 for c in result.candidates)
        assert result.similar_cases == []
        assert result.case_total == 0
        assert all(e.log_odds_contribution is None for e in result.evidence)

    def test_measurements_are_reported(self):
        """测量值必须报出来 —— 它是本模式唯一的实质内容。"""
        result, _ = interpret_meow(features=feats(), records=[])
        measured = [e for e in result.evidence if e.kind is EvidenceKind.MEASURED]
        assert measured
        assert all(e.value is not None for e in measured)

    def test_no_reference_without_individual_baseline(self):
        """**没有个体基线时不给比较。**

        绝不退回群体先验 —— 占位先验的均值是编的，
        用它当参考系会让「高于参考值」建立在假数字上。
        """
        result, _ = interpret_meow(features=feats(), records=[], individual=None)
        measured = [e for e in result.evidence if e.kind is EvidenceKind.MEASURED]
        assert all(e.reference is None for e in measured)
        assert all("本次测量值" in e.statement for e in measured)

    def test_scene_produces_candidate_but_no_probability(self):
        result, _ = interpret_meow(
            features=feats(), records=[], scene="它在门口叫"
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].context is ContextLabel.DOOR_ATTENTION
        assert result.candidates[0].posterior is None

    def test_no_scene_means_no_candidates(self):
        """**没有场景就不猜。** 这是「不编造」在候选层的体现。"""
        result, _ = interpret_meow(features=feats(), records=[])
        assert result.candidates == []

    def test_scene_evidence_added_in_both_modes(self):
        """场景证据在两个非贝叶斯模式下**形状一致**。

        初版只在 case_based 里加了场景证据，measured_only 漏了 ——
        同一输入在不同模式下证据列表不一致。
        """
        recs = [rec(ContextLabel.GREETING, rid=f"g{i}") for i in range(3)]

        cold, _ = interpret_meow(features=feats(), records=[], scene="它在门口叫")
        warm, _ = interpret_meow(features=feats(), records=recs, scene="它在门口叫")

        for result in (cold, warm):
            priors = [e for e in result.evidence if e.kind is EvidenceKind.PRIOR]
            assert priors, "场景应产生 PRIOR 证据"
            assert "门口" in priors[0].statement

    def test_observation_invites_recording(self):
        """冷启动时建议的主语是「**你**」—— 系统请求信息，而不是给判断。"""
        result, _ = interpret_meow(features=feats(), records=[])
        assert "再记录" in result.suggested_observation

    def test_limitations_state_why_no_probability(self):
        result, _ = interpret_meow(features=feats(), records=[])
        assert "不给出概率判断" in result.limitations

    def test_scene_matching_is_rule_based(self):
        assert match_scene("它对着门叫") is ContextLabel.DOOR_ATTENTION
        assert match_scene("该喂饭了") is ContextLabel.FOOD_WAITING
        assert match_scene("") is None
        assert match_scene(None) is None
        assert match_scene("完全无关的描述") is None


# ═══════════════════════════════════════════════════════════════
# 6. 路由：fail-closed
# ═══════════════════════════════════════════════════════════════


class TestRouter:
    def test_placeholder_prior_never_produces_posterior(self, placeholder_prior):
        """**本文件最重要的一条断言。**

        占位先验下，即使有先验对象、也有足够的案例，
        也**不得**产生任何后验概率。
        """
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        result, decision = interpret_meow(
            features=feats(), records=recs, prior=placeholder_prior
        )
        assert decision.mode is not EvidenceMode.ACOUSTIC_PLUS_HISTORY
        assert all(c.posterior is None for c in result.candidates)
        assert "占位数据" in decision.reason

    def test_placeholder_prior_without_records_goes_cold(self, placeholder_prior):
        _, decision = interpret_meow(
            features=feats(), records=[], prior=placeholder_prior
        )
        assert decision.mode is EvidenceMode.MEASURED_ONLY

    def test_real_prior_uses_bayesian_path(self, placeholder_prior):
        """先验一旦标记为已实测，就切回贝叶斯路径（后验可用）。"""
        import dataclasses

        real = dataclasses.replace(
            placeholder_prior, is_placeholder=False, provenance="catmeows"
        )
        decision = choose_mode(features=feats(), records=[], prior=real)
        assert decision.mode is EvidenceMode.ACOUSTIC_PLUS_HISTORY

    def test_no_prior_goes_cold_not_bayesian(self):
        decision = choose_mode(features=feats(), records=[])
        assert decision.mode is EvidenceMode.MEASURED_ONLY
        assert "未提供群体先验" in decision.reason

    def test_decision_is_observable(self, placeholder_prior):
        """决策理由必须可读 —— 冷启动阶段用户最需要的正是这个解释。"""
        _, decision = interpret_meow(
            features=feats(), records=[], prior=placeholder_prior
        )
        assert decision.reason
        assert decision.mode.value in decision.reason or "相似案例" in decision.reason

    def test_choice_order_is_fail_closed(self, placeholder_prior):
        """顺序必须是「先验 → 案例 → 测量」，**默认往下走**。"""
        # 先验可用 → 贝叶斯
        import dataclasses

        real = dataclasses.replace(placeholder_prior, is_placeholder=False)
        assert (
            choose_mode(features=feats(), records=[], prior=real).mode
            is EvidenceMode.ACOUSTIC_PLUS_HISTORY
        )
        # 先验不可用 + 案例够 → 案例
        recs = [rec(ContextLabel.DOOR_ATTENTION, rid=f"d{i}") for i in range(3)]
        assert (
            choose_mode(features=feats(), records=recs, prior=placeholder_prior).mode
            is EvidenceMode.CASE_BASED
        )
        # 都不满足 → 测量
        assert (
            choose_mode(features=feats(), records=[], prior=placeholder_prior).mode
            is EvidenceMode.MEASURED_ONLY
        )


# ═══════════════════════════════════════════════════════════════
# 7. 边界
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_quality_is_acknowledged(self):
        from app.schemas import FeatureQuality as Q

        f = feats().model_copy(update={"quality": Q.POOR})
        result, _ = interpret_meow(features=f, records=[])
        assert "poor" in result.limitations

    def test_no_valid_f0_is_acknowledged(self):
        f = feats().model_copy(update={"f0_mean": 0.0})
        result, _ = interpret_meow(features=f, records=[])
        assert "未提取到有效基频" in result.limitations

    def test_unavailable_features_are_listed(self):
        f = feats().model_copy(
            update={"unavailable": ["call_rate"], "call_rate": 99.0}
        )
        result, _ = interpret_meow(features=f, records=[])
        # 被标记不可用的特征不应出现为测量证据
        assert not any(e.source == "acoustic:call_rate" for e in result.evidence)

    def test_case_evidence_mentions_missing_resolution(self):
        """案例没记结果时要说出来 —— 否则用户以为系统漏看了。"""
        recs = [rec(ContextLabel.DOOR_ATTENTION, resolution=None, rid=f"d{i}") for i in range(3)]
        result, _ = interpret_meow(features=feats(), records=recs)
        assert any("没有记录结果" in e.statement for e in result.evidence)
