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


@pytest.fixture()
def prior(prior_factory) -> PriorTable:
    """**测试自己的先验，不读生产数据文件。**

    初版读 `data/priors/catmeows_stats.json`。那个文件后来被真实统计
    替换掉之后，十几个测试一起变红 —— 而它们测的代码一行没改。
    那种耦合会让「数据更新」和「代码回归」在结果里长得一样，
    而两者的处理方式完全不同。
    """
    return prior_factory()


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

    def test_placeholder_is_flagged(self, prior_factory):
        """占位先验**必须被标记**，不得静默当成真实统计。

        ⚠️ 用工厂造一个占位先验，而不是读生产文件 ——
        初版断言生产文件是占位的，而那个文件后来被真实统计替换掉了，
        测试就红了，而它测的代码一行没改。
        """
        placeholder = prior_factory(is_placeholder=True)
        assert placeholder.is_placeholder is True
        assert placeholder.provenance == "placeholder"
        assert placeholder.is_validated is False, "占位数据不可能通过验证"

    def test_unvalidated_real_prior_is_not_usable_for_posterior(self, prior_factory):
        """**真实的数字≠可用的数字。**

        这是 CatMeows 实测结果所对应的关键区分：
        换成真实统计后 `is_placeholder=False`，
        但留出 macro-F1（0.364）低于多数类基线（0.506）——
        它在没见过的猫上比「总是猜多数类」还差。

        所以「是不是编的」与「能不能用」必须是两个判据。
        """
        real_but_useless = prior_factory(
            is_placeholder=False, macro_f1=0.364, majority_baseline=0.506
        )
        assert real_but_useless.is_placeholder is False
        assert real_but_useless.is_validated is False
        assert real_but_useless.discrimination_margin < 0

    def test_production_prior_file_loads(self):
        """生产先验文件（不论内容）必须能加载。

        上面几个测试测的是「各种先验的行为」；
        这一条只测「当前那份文件格式合法」——
        两者分开，文件内容更新就不会连带打断行为测试。
        """
        table = PriorTable.load(PRIOR_PATH)
        assert table.version
        assert table.contexts, "先验至少要有一个情境"
        assert abs(sum(table.base_rates.values()) - 1.0) < 1e-3
        # 不论真假，都必须能就「能不能产出后验概率」给出明确结论
        assert isinstance(table.is_validated, bool)

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

    def test_partial_feature_coverage_is_allowed(self, tmp_path):
        """**缺特征不再算格式错误** —— 它是「没有数据」，是合法状态。

        ## 为什么这条断言被反过来了

        初版是 `test_rejects_missing_feature`：缺一个特征就报错。
        那时先验是占位值、8 个特征都写了，所以这条约束从未被真实数据碰过。

        换成 CatMeows 真实统计后：`call_rate` / `ici_mean` 的可测率
        只有 **0–4%**（每条录音都是单次叫声，没有「间隔」可言）。
        它们**没有数据**，不是「忘了写」。

        把「没有数据」当格式错误会逼着人填一个假数字才能加载 ——
        而那正是这个项目一直在防的事。

        所以：缺特征 → 允许，推理时跳过（与 `AcousticFeatures.unavailable` 同类）；
        **写错特征名** → 仍然报错（那会让一个特征静默地永不生效）。
        """
        import json

        partial = {
            "version": "t",
            "base_rates": {"greeting": 0.5, "other": 0.5},
            "contexts": {
                # greeting 只有 3 个特征；other 齐全
                "greeting": {
                    "duration": [1.0, 0.5],
                    "f0_mean": [500.0, 100.0],
                    "rms_mean": [0.1, 0.05],
                },
                "other": {f: [1.0, 1.0] for f in FEATURE_ORDER},
            },
        }
        p = tmp_path / "partial.json"
        p.write_text(json.dumps(partial), encoding="utf-8")

        table = PriorTable.load(p)
        # 有数据的特征照常可取
        assert table.stat(ContextLabel.GREETING, "duration") is not None
        # 没数据的返回 None —— 而不是抛错、也不是编一个值
        assert table.stat(ContextLabel.GREETING, "call_rate") is None
        assert "call_rate" in table.missing_features(ContextLabel.GREETING)
        # 齐全的情境不受影响
        assert table.missing_features(ContextLabel.OTHER) == ()

    def test_rejects_unknown_feature_name(self, tmp_path):
        """**写错特征名仍然要报错。**

        它与「没有数据」是两回事：`f0_men` 这种拼错会让那个特征
        静默地永不生效，而后验看上去完全正常。
        """
        import json

        bad = {
            "version": "t",
            "base_rates": {"greeting": 1.0},
            "contexts": {"greeting": {"f0_men": [1.0, 1.0]}},
        }
        p = tmp_path / "unknown.json"
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="未知特征"):
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

    def test_placeholder_prior_degrades_via_router(self, prior_factory, meow):
        """占位先验**必须走降级路径**，不得静默使用。

        ⚠️ 要调 `interpret_meow`（路由层）而不是 `interpret`（数学层）——
        降级是路由的职责。数学层现在会**直接报错**（不让不具区分度的先验
        产出后验），所以用它测降级是测错了函数。
        """
        from app.interpreter import interpret_meow

        placeholder = prior_factory(is_placeholder=True)
        r, decision = interpret_meow(features=meow, prior=placeholder)
        assert r.evidence_mode.value != "acoustic_plus_history"
        assert "占位" in decision.reason

    def test_non_discriminative_prior_degrades_via_router(self, prior_factory, meow):
        """**实测但不具区分度**的先验也要降级。

        这是 CatMeows 换成真实统计后的新情形：数字是真的，
        但留出 macro-F1（0.364）低于多数类基线（0.506）——
        它在没见过的猫上比「总是猜多数类」还差。
        那种先验产生的后验概率不是证据，用户必须被告知。
        """
        from app.interpreter import interpret_meow

        weak = prior_factory(macro_f1=0.364, majority_baseline=0.506)
        r, decision = interpret_meow(features=meow, prior=weak)
        assert r.evidence_mode.value != "acoustic_plus_history", (
            "不具区分度的先验不得进入贝叶斯路径"
        )
        assert "不具区分度" in decision.reason
        assert "0.364" in decision.reason and "0.506" in decision.reason

    def test_bayes_layer_refuses_unvalidated_prior(self, prior_factory, meow):
        """**数学层自己不接受未验证的先验。**

        路由会降级，但「先验能不能产出后验概率」是**不变量**而不是路由的偏好：
        任何绕过路由直接调 `interpret` 的路径（新调用方、脚本、重构）
        都会拿到一堆看起来正常、实际不具区分度的概率。

        把检查放进数学层，不变量就是结构性的 ——
        与「存储层强制要求 tenant_id」同一个思路。
        """
        weak = prior_factory(macro_f1=0.30, majority_baseline=0.50)
        with pytest.raises(ValueError, match="不得用来产出后验概率"):
            interpret(features=meow, prior=weak)

        # 消融 / 数学层测试可以显式跳过，但那是一个**要写出来的决定**
        r = interpret(features=meow, prior=weak, allow_unvalidated_prior=True)
        assert r.evidence_mode.value == "acoustic_plus_history"

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
            # 显式给验证信息，否则先被「未验证」那条拦住 ——
            # 而本测试要测的是「情境数不足」这个**更晚**的检查。
            holdout_macro_f1=0.9,
            holdout_majority_baseline=0.5,
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
