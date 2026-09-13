"""行为解释器测试。

覆盖 docs/DESIGN.md §3.6 的核心主张：
- 概率可复现（同一输入 → 同一后验）
- 证据可归因（每个特征的贡献是一个数字）
- 冷启动退化为群体先验（λ_c=0）
- 个体样本增多 → 个体化程度上升
- 占位先验被**透传**而非静默当成真实统计
- 未确认的历史样本**不构成证据**
- 低质量音频 → 判别力下降
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from app.audio.features import (
    TARGET_SR,
    AudioTooShort,
    extract_features,
    synthesize_meow,
)
from app.interpreter import (
    DEFAULT_KAPPA,
    FEATURE_ORDER,
    IndividualModel,
    LabelledSample,
    PriorTable,
    SimilarSample,
    interpret,
    match_scene,
)
from app.schemas import ContextLabel, EvidenceKind, FeatureQuality

PRIOR_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "priors" / "catmeows_stats.json"
)


# ─────────────────────────────────────────────────────────────
# 取值 helper
# ─────────────────────────────────────────────────────────────
#
# 为什么需要：``posterior`` 与 ``log_odds`` 都是 Optional（``text_only`` 模式下
# 必须为 None）。直接在断言里比较会让类型检查报错，而且**若行为变了**，
# 测试会在比较时抛 ``TypeError`` —— 那是个没有信息量的失败。
# 显式断言非 None，失败信息就直接指向「哪个情境不该是 None」。


def posterior_of(result, ctx: ContextLabel) -> float:
    """取某情境的后验概率，断言非 ``None``。"""
    value = next((c.posterior for c in result.candidates if c.context is ctx), None)
    assert value is not None, f"{ctx.value} 的 posterior 不应为 None"
    return value


def logit_of(result, ctx: ContextLabel) -> float:
    """取某情境的 logit，断言非 ``None``。"""
    value = next((c.log_odds for c in result.candidates if c.context is ctx), None)
    assert value is not None, f"{ctx.value} 的 log_odds 不应为 None"
    return value


@pytest.fixture(scope="module")
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture(scope="module")
def meow():
    """一段合成「类猫叫」，仅用于让链路可跑 —— 不是真实猫叫的声学模型。"""
    y = synthesize_meow(duration=0.5, f0_start=500.0, f0_end=700.0)
    return extract_features(y, TARGET_SR)


# ═══════════════════════════════════════════════════════════════
# 先验加载
# ═══════════════════════════════════════════════════════════════


class TestPriorTable:
    def test_loads(self, prior: PriorTable):
        assert prior.version
        assert len(prior.contexts) >= 2
        assert abs(sum(prior.base_rates.values()) - 1.0) < 1e-6

    def test_sha256_computed(self, prior: PriorTable):
        """可复现性要求：先验版本可被校验。"""
        assert len(prior.sha256) == 64
        assert all(c in "0123456789abcdef" for c in prior.sha256)

    def test_placeholder_is_flagged(self, prior: PriorTable):
        """U3：群体先验尚未构建。**必须被标记，不得静默当成真实统计。**"""
        assert prior.is_placeholder is True
        assert prior.provenance == "placeholder"
        assert prior.note

    def test_all_features_present_per_context(self, prior: PriorTable):
        for ctx in prior.contexts_ordered:
            for feat in FEATURE_ORDER:
                stat = prior.stat(ctx, feat)
                assert stat.std > 0

    def test_context_order_is_deterministic(self, prior: PriorTable):
        assert prior.contexts_ordered == prior.contexts_ordered

    def test_rejects_bad_std(self, tmp_path):
        import json

        bad = {
            "version": "t",
            "base_rates": {"greeting": 0.5, "other": 0.5},
            "contexts": {
                "greeting": {f: [1.0, 0.0] for f in FEATURE_ORDER},
                "other": {f: [1.0, 1.0] for f in FEATURE_ORDER},
            },
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="std 必须为正"):
            PriorTable.load(p)

    def test_rejects_missing_feature(self, tmp_path):
        import json

        bad = {
            "version": "t",
            "base_rates": {"greeting": 0.5, "other": 0.5},
            "contexts": {
                "greeting": {"duration": [1.0, 1.0]},
                "other": {f: [1.0, 1.0] for f in FEATURE_ORDER},
            },
        }
        p = tmp_path / "bad2.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="缺少特征"):
            PriorTable.load(p)


# ═══════════════════════════════════════════════════════════════
# 声学特征
# ═══════════════════════════════════════════════════════════════


class TestAcousticFeatures:
    def test_extracts_from_synthesized(self, meow):
        assert meow.duration > 0
        assert meow.f0_mean > 0
        assert meow.rms_mean > 0
        assert meow.quality in tuple(FeatureQuality)

    def test_f0_recovery_within_tolerance(self):
        """合成信号的基频应能被 pyin 大致恢复。"""
        y = synthesize_meow(duration=0.6, f0_start=600.0, f0_end=600.0, harmonics=8)
        feats = extract_features(y, TARGET_SR)
        assert 400.0 < feats.f0_mean < 850.0

    def test_rising_contour_gives_positive_slope(self):
        y = synthesize_meow(duration=0.6, f0_start=420.0, f0_end=780.0)
        assert extract_features(y, TARGET_SR).f0_slope > 0

    def test_falling_contour_gives_negative_slope(self):
        y = synthesize_meow(duration=0.6, f0_start=780.0, f0_end=420.0)
        assert extract_features(y, TARGET_SR).f0_slope < 0

    def test_rejects_wrong_sample_rate(self):
        y = synthesize_meow(sr=8000)
        with pytest.raises(ValueError, match="采样率必须"):
            extract_features(y, 8000)

    def test_rejects_too_short(self):
        with pytest.raises(AudioTooShort):
            extract_features(np.zeros(10, dtype=np.float32), TARGET_SR)

    def test_deterministic(self):
        """可复现性要求：同一段音频 → 同一组特征。"""
        y = synthesize_meow(seed=7)
        a = extract_features(y, TARGET_SR)
        b = extract_features(y, TARGET_SR)
        assert a == b

    def test_short_clip_cannot_estimate_call_rate(self):
        """短窗口下叫声速率无法可靠估计——必须标记 unavailable，不得零填充。

        0.5s 里出现 1 次叫声，若归一化到「次/10s」会得到 20 次/10s 的荒谬值。
        """
        feats = extract_features(synthesize_meow(duration=0.5), TARGET_SR)
        assert "call_rate" in feats.unavailable
        assert "ici_mean" in feats.unavailable

    def test_long_clip_can_estimate_call_rate(self):
        """足够长的窗口下，叫声速率应可测且数量级合理。"""
        feats = extract_features(synthesize_meow(duration=4.0), TARGET_SR)
        assert "call_rate" not in feats.unavailable
        assert 0.0 < feats.call_rate < 10.0

    def test_gap_merging_prevents_over_segmentation(self):
        """谐波拍频造成的包络凹陷不得被误判为多次独立叫声。

        实测：修复前在 0.5s 合成信号上曾得到 20 次/10s。
        """
        feats = extract_features(synthesize_meow(duration=4.0), TARGET_SR)
        assert feats.call_rate < 10.0, (
            f"叫声速率 {feats.call_rate} 明显过高，分段可能过度"
        )

    def test_unavailable_excludes_feature_from_inference(self, prior, meow):
        """unavailable 中的特征必须被推理跳过。"""
        marked = meow.model_copy(
            update={"unavailable": ["call_rate"], "call_rate": 999.0}
        )
        r_marked = interpret(features=marked, prior=prior)
        r_wild = interpret(
            features=meow.model_copy(update={"call_rate": 999.0}), prior=prior
        )
        # 标记为不可用后，999 这个荒谬值不应影响结果
        assert [c.posterior for c in r_marked.candidates] != [
            c.posterior for c in r_wild.candidates
        ]
        assert "call_rate" in r_marked.limitations

    def test_silence_is_poor_quality(self):
        """静音不应被当作有效叫声，也不应编造基频。"""
        y = np.zeros(TARGET_SR, dtype=np.float32)
        feats = extract_features(y, TARGET_SR)
        assert feats.f0_mean == 0.0
        assert "f0_mean" in feats.unavailable
        assert feats.quality is FeatureQuality.POOR

    def test_missing_snr_is_not_bad_quality(self):
        """**无法估计 SNR ≠ 音频质量差**。

        回归：早期实现把 ``snr_db is None`` 直接归为 POOR，
        于是「整段都是叫声、没有静音段」的**干净录音**被判成低质量——
        那把**估计器的局限**记在了音频头上。
        """
        from app.audio.features import _classify_quality

        assert _classify_quality(0.70, None) is FeatureQuality.GOOD
        assert _classify_quality(0.30, None) is FeatureQuality.FAIR
        assert _classify_quality(0.10, None) is FeatureQuality.POOR

    def test_snr_estimation_needs_actual_silence(self):
        """无静音段时应返回 None（不可估计），而不是编一个数字。"""
        feats = extract_features(synthesize_meow(duration=4.0), TARGET_SR)
        # 合成信号的包络近似「整段都是叫声」，噪声底不可估
        assert feats.estimated_snr_db is None
        assert feats.quality is not FeatureQuality.POOR


# ═══════════════════════════════════════════════════════════════
# 贝叶斯推理
# ═══════════════════════════════════════════════════════════════


class TestInterpretation:
    def test_posteriors_normalized(self, prior, meow):
        r = interpret(features=meow, prior=prior)
        total = sum(c.posterior for c in r.candidates if c.posterior is not None)
        # 契约层把 posterior 舍入到 6 位小数，6 个情境累计误差上限 3e-6
        assert abs(total - 1.0) < 1e-4

    def test_candidates_sorted_desc(self, prior, meow):
        r = interpret(features=meow, prior=prior)
        posts = [c.posterior for c in r.candidates if c.posterior is not None]
        assert len(posts) == len(r.candidates), "声学模式下每个候选都应有后验"
        assert posts == sorted(posts, reverse=True)

    def test_deterministic(self, prior, meow):
        """同一输入 → 同一输出。这是「证据」成立的前提。"""
        a = interpret(features=meow, prior=prior)
        b = interpret(features=meow, prior=prior)
        assert [c.posterior for c in a.candidates] == [
            c.posterior for c in b.candidates
        ]
        assert [e.log_odds_contribution for e in a.evidence] == [
            e.log_odds_contribution for e in b.evidence
        ]

    def test_evidence_non_empty_and_capped(self, prior, meow):
        r = interpret(features=meow, prior=prior)
        assert r.evidence
        measured = [e for e in r.evidence if e.kind is EvidenceKind.MEASURED]
        assert measured, "至少应有一条可复算的测量证据"
        for e in measured:
            # 有概率模型的模式下贡献值必填（契约已保证），这里解开 Optional
            assert e.log_odds_contribution is not None
            assert abs(e.log_odds_contribution) <= 3.0 + 1e-9

    def test_measured_evidence_is_recomputable(self, prior, meow):
        """证据的可追溯性：MEASURED 项必须带 value 与 reference，可复算。"""
        r = interpret(features=meow, prior=prior)
        measured = [e for e in r.evidence if e.kind is EvidenceKind.MEASURED]
        assert measured
        for e in measured:
            assert e.value is not None
            assert e.reference is not None
            assert e.source.startswith("acoustic:")

    def test_prior_version_recorded(self, prior, meow):
        """可复现性要求：输出必须带先验版本。"""
        r = interpret(features=meow, prior=prior)
        assert r.prior_version == prior.version

    def test_mandatory_limitations_and_observation(self, prior, meow):
        """docs/DESIGN.md §3.6：任何情况下都必须输出这两项。"""
        r = interpret(features=meow, prior=prior)
        assert r.limitations
        assert r.suggested_observation

    def test_acoustic_mode_requires_features(self, prior, meow):
        r = interpret(features=meow, prior=prior)
        assert r.acoustic_features is not None


class TestScenePrior:
    def test_match_scene_rules(self):
        assert match_scene("它对着门叫") is ContextLabel.DOOR_ATTENTION
        assert match_scene("在食盆旁边叫") is ContextLabel.FOOD_WAITING
        assert match_scene("unknown situation") is None
        assert match_scene(None) is None

    def test_scene_boosts_matching_context(self, prior, meow):
        without = interpret(features=meow, prior=prior)
        with_scene = interpret(features=meow, prior=prior, scene="它对着门叫")

        assert posterior_of(with_scene, ContextLabel.DOOR_ATTENTION) > posterior_of(
            without, ContextLabel.DOOR_ATTENTION
        )

    def test_scene_appears_in_evidence(self, prior, meow):
        r = interpret(features=meow, prior=prior, scene="它对着门叫")
        assert any(e.kind is EvidenceKind.PRIOR for e in r.evidence)


class TestIndividualization:
    def test_cold_start_uses_population_only(self, prior, meow):
        cold = IndividualModel.from_samples("p1", [])
        assert cold.is_cold_start
        assert cold.lambda_c == 0.0

        r = interpret(features=meow, prior=prior, individual=cold)
        assert r.sample_count == 0
        assert r.individualization == 0.0
        assert "第一次被记录" in r.limitations

    def test_lambda_grows_with_samples(self):
        samples = [
            LabelledSample(
                context=ContextLabel.FOOD_WAITING,
                features=dict.fromkeys(FEATURE_ORDER, 1.0),
            )
            for _ in range(5)
        ]
        m = IndividualModel.from_samples("p1", samples)
        assert m.lambda_c == pytest.approx(5 / (5 + DEFAULT_KAPPA))
        assert not m.is_cold_start

    def test_individual_evidence_shifts_posterior(self, prior, meow):
        """个体样本应能把判断拉向它自己的模式。"""
        base = interpret(features=meow, prior=prior)
        target = ContextLabel.ISOLATION_DISTRESS

        # 构造一批在该情境下与本次特征完全一致的样本
        template = {f: float(getattr(meow, f)) for f in FEATURE_ORDER}
        samples = [
            LabelledSample(context=target, features=dict(template)) for _ in range(30)
        ]
        warm_model = IndividualModel.from_samples("p1", samples)
        warm = interpret(features=meow, prior=prior, individual=warm_model)

        assert posterior_of(warm, target) > posterior_of(base, target)
        assert warm.individualization > 0.5

    def test_shrunk_std_has_floor(self, prior: PriorTable):
        """std 收缩必须有下界，否则 2 个样本会产生虚假的强判别力。"""
        template = dict.fromkeys(FEATURE_ORDER, 1.0)
        samples = [
            LabelledSample(context=ContextLabel.GREETING, features=dict(template))
            for _ in range(3)
        ]
        m = IndividualModel.from_samples("p1", samples)
        pop = prior.stat(ContextLabel.GREETING, "duration")
        stat = m.shrunk_stat(ContextLabel.GREETING, "duration", pop)
        assert stat.std > 0
        assert np.isfinite(stat.log_density(1.0))


class TestSimilarSamples:
    def test_unconfirmed_samples_do_not_count(self, prior, meow):
        """未确认情境的历史样本**不构成证据**。"""
        base = interpret(features=meow, prior=prior)
        with_unconfirmed = interpret(
            features=meow,
            prior=prior,
            similar_samples=[
                SimilarSample(sample_id="s1", similarity=0.99, context=None)
            ],
        )
        assert [c.posterior for c in base.candidates] == [
            c.posterior for c in with_unconfirmed.candidates
        ]
        assert not any(
            e.kind is EvidenceKind.RETRIEVED for e in with_unconfirmed.evidence
        )

    def test_low_similarity_ignored(self, prior, meow):
        r = interpret(
            features=meow,
            prior=prior,
            similar_samples=[
                SimilarSample(
                    sample_id="s1", similarity=0.5, context=ContextLabel.GREETING
                )
            ],
        )
        assert not any(e.kind is EvidenceKind.RETRIEVED for e in r.evidence)

    def test_confirmed_similar_sample_contributes(self, prior, meow):
        no_sample = interpret(features=meow, prior=prior)
        with_sample = interpret(
            features=meow,
            prior=prior,
            similar_samples=[
                SimilarSample(
                    sample_id="s1", similarity=0.95, context=ContextLabel.GREETING
                )
            ],
        )

        # 断言在 logit 而非 posterior 上：后验经 softmax + 温度后可能小到 6 位小数下为 0，
        # 而 logit 的增量恰好等于样本调整量，是对机制的精确断言。
        assert logit_of(with_sample, ContextLabel.GREETING) > logit_of(
            no_sample, ContextLabel.GREETING
        )
        assert any(e.kind is EvidenceKind.RETRIEVED for e in with_sample.evidence)


class TestQualityDegradation:
    def _tv_from_prior(self, r, prior):
        """后验与先验分布的总变差距离。越大 = 判别越强。"""
        total = sum(prior.base_rates.values())
        return 0.5 * sum(
            abs((c.posterior or 0.0) - prior.base_rates[c.context] / total)
            for c in r.candidates
        )

    @pytest.mark.parametrize("bad", [FeatureQuality.FAIR, FeatureQuality.POOR])
    def test_lower_quality_is_less_confident(self, prior, meow, bad):
        """低质量音频 → 似然展宽 → 后验更靠近先验（判别力下降）。

        控制变量：**同一组特征值**，只改 ``quality``。
        度量用与先验的总变差距离——直接比 logit 是不对的，
        因为 logit 里含不受质量影响的先验项。
        """
        good = interpret(
            features=meow.model_copy(update={"quality": FeatureQuality.GOOD}),
            prior=prior,
        )
        degraded = interpret(
            features=meow.model_copy(update={"quality": bad}), prior=prior
        )
        assert (
            self._tv_from_prior(degraded, prior)
            <= self._tv_from_prior(good, prior) + 1e-9
        )

    def test_posterior_is_not_degenerate(self, prior, meow):
        """后验不得退化成 one-hot。

        占位先验下 logit 跨度可达 ±6，不加温度会让每次判断都落进「高置信」档——
        这与项目「不声称超出证据的确定性」的原则直接冲突。
        """
        r = interpret(features=meow, prior=prior)
        posts = [c.posterior or 0.0 for c in r.candidates]
        assert max(posts) < 0.99, "后验过度集中，说明未做有效的置信度校准"
        assert sum(1 for p in posts if p > 0.01) >= 2, "只有一个候选有非零概率"

    def test_quality_noted_in_limitations(self, prior, meow):
        degraded = meow.model_copy(update={"quality": FeatureQuality.POOR})
        r = interpret(features=degraded, prior=prior)
        assert "录音质量" in r.limitations

    def test_uncalibrated_confidence_is_disclosed(self, prior, meow):
        """置信度未校准必须在输出里声明（U1）。"""
        r = interpret(features=meow, prior=prior)
        assert "校准" in r.limitations

    def test_placeholder_prior_surfaced_in_limitations(self, prior, meow):
        """占位先验必须出现在输出限制中，不得静默使用。"""
        assert prior.is_placeholder
        r = interpret(features=meow, prior=prior)
        assert "占位" in r.limitations

    def test_missing_f0_is_flagged(self, prior, meow):
        no_f0 = meow.model_copy(update={"f0_mean": 0.0})
        r = interpret(features=no_f0, prior=prior)
        assert "基频" in r.limitations


class TestRobustness:
    def test_rejects_single_context_table(self, meow):
        from app.interpreter import PriorTable as PT

        table = PT(
            version="t",
            provenance="test",
            is_placeholder=False,
            sha256="0" * 64,
            note="",
            base_rates={ContextLabel.OTHER: 1.0},
            contexts={
                ContextLabel.OTHER: {
                    f: __import__(
                        "app.interpreter.priors", fromlist=["GaussianStat"]
                    ).GaussianStat(1.0, 1.0)
                    for f in FEATURE_ORDER
                }
            },
        )
        with pytest.raises(ValueError, match="至少需要 2 个情境"):
            interpret(features=meow, prior=table)

    def test_evidence_budget_reserves_slots(self, prior, meow):
        """PRIOR / RETRIEVED 证据不得被 MEASURED 挤掉。"""
        r = interpret(
            features=meow,
            prior=prior,
            scene="它对着门叫",
            similar_samples=[
                SimilarSample(
                    sample_id="s1", similarity=0.95, context=ContextLabel.GREETING
                )
            ],
        )
        kinds = [e.kind for e in r.evidence]
        assert EvidenceKind.PRIOR in kinds, "场景证据被截断了"
        assert EvidenceKind.RETRIEVED in kinds, "历史样本证据被截断了"
        assert len(r.evidence) <= 4

    def test_confidence_tier_present(self, prior, meow):
        r = interpret(features=meow, prior=prior)
        assert r.confidence_tier in {"high", "medium", "low", "none"}
        assert r.top_candidate is not None
