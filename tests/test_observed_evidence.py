"""多模态观察接入行为解释的测试。

## 本文件守护的核心不变量

> **`OBSERVED` 不参与模式选择，也不改变候选排序。**

未经校验的模型输出若影响判断，`OBSERVED` 与 `MEASURED` 的分栏就白做了。
所以下面有一组测试**对比「有提取器」与「没提取器」两次运行的判断结果必须一致** ——
只有证据列表变长，判断不能变。

## 另外三件必须成立的事

| # | 要求 | 理由 |
| --- | --- | --- |
| 1 | 提取失败**可见** | 否则无法区分「没配提取器」与「配了但失败」 |
| 2 | 没有提取器时**不报错** | 它是可选依赖 |
| 3 | 声学提取失败时**仍给观察** | 只返回一个错误等于把已有信息丢掉 |
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.graph import build_graph, initial_state, tenant_of
from app.interpreter import PriorTable
from app.llm import HashEmbedder, MockLLM
from app.profile import VisualObservation
from app.schemas import (
    AcousticFeatures,
    AudioKind,
    BehaviorAction,
    BehaviorInterpretation,
    ContextLabel,
    EvidenceKind,
    MediaKind,
    ModelObservation,
    RawInput,
    Species,
)
from app.store import InMemoryStore

PRIOR_PATH = "data/priors/catmeows_stats.json"
NOW = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────
# 替身
# ─────────────────────────────────────────────────────────────


class _StubVision:
    def analyze(self, image_url: str) -> VisualObservation:
        return VisualObservation(image_url=image_url)


def _synthetic_features() -> AcousticFeatures:
    """固定的特征值 —— **不用合成音频**，让测试快且确定。"""
    return AcousticFeatures(
        duration=0.64,
        f0_mean=599.0,
        f0_range=180.0,
        f0_slope=0.12,
        call_rate=3.0,
        ici_mean=0.5,
        rms_mean=0.2,
        roughness=0.18,
    )


def _extractor(url: str) -> AcousticFeatures:
    return _synthetic_features()


class _StubMediaExtractor:
    """返回预设观察。"""

    def __init__(self, observation: ModelObservation | None = None, raises: bool = False):
        self._obs = observation
        self._raises = raises
        self.calls: list[tuple[str, MediaKind]] = []

    def extract(self, media_url: str, *, media_kind: MediaKind) -> ModelObservation:
        self.calls.append((media_url, media_kind))
        if self._raises:
            raise RuntimeError("上游炸了")
        if self._obs is not None:
            return self._obs
        return ModelObservation(
            media_url=media_url,
            media_kind=media_kind,
            source_model="stub-omni",
            actions=[BehaviorAction.SCRATCH_DOOR, BehaviorAction.LOOK_AT_DOOR],
            scene_objects=["door", "human"],
            described_signs=["前爪抬起靠近门框"],
        )


def _graph(store: InMemoryStore, prior: PriorTable, media_extractor=None):
    return build_graph(
        store=store,
        embedder=HashEmbedder(),
        llm=MockLLM(default="团团挺想你的。"),
        prior=prior,
        feature_extractor=_extractor,
        vision=_StubVision(),
        # 用访问器而不是 state["..."] —— TypedDict 的键访问在类型层不安全，
        # tenant_of 把 KeyError 变成说明白了的 StateError（见 app/graph/state.py）
        records_lookup=lambda state: store.list_meow_records(
            user_id=tenant_of(state)[0], pet_id=tenant_of(state)[1]
        ),
        media_extractor=media_extractor,
    )


def _invoke(graph, **raw_kw):
    raw = RawInput(
        audio_url="http://x/meow.mp4",
        audio_kind=AudioKind.CAT_MEOW,
        media_kind=MediaKind.VIDEO,
        **raw_kw,
    )
    return graph.invoke(
        initial_state(user_id="u", pet_id="p", raw_input=raw)
    )


@pytest.fixture(scope="module")
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture()
def store() -> InMemoryStore:
    return InMemoryStore()


# ═══════════════════════════════════════════════════════════════
# 1. 观察进入证据
# ═══════════════════════════════════════════════════════════════


class TestObservationEntersEvidence:
    def test_observed_evidence_is_appended(self, store, prior):
        result = _invoke(_graph(store, prior, _StubMediaExtractor()))
        interp: BehaviorInterpretation = result["interpretation"]
        observed = [e for e in interp.evidence if e.kind is EvidenceKind.OBSERVED]
        assert observed, "模型观察必须出现在证据里"
        assert any("抓门" in e.statement for e in observed)
        assert any("画面里" in e.statement for e in observed)

    def test_source_names_the_model(self, store, prior):
        """**必须标明哪个模型** —— 模型会换、会升级，不记来源就无法回溯。"""
        result = _invoke(_graph(store, prior, _StubMediaExtractor()))
        interp: BehaviorInterpretation = result["interpretation"]
        observed = [e for e in interp.evidence if e.kind is EvidenceKind.OBSERVED]
        assert all(e.source.startswith("model:") for e in observed)
        assert all("stub-omni" in e.source for e in observed)

    def test_observed_has_no_log_odds_contribution(self, store, prior):
        """它不在概率模型里 —— 给一个对数几率贡献值是编造（契约也会拦）。"""
        result = _invoke(_graph(store, prior, _StubMediaExtractor()))
        interp: BehaviorInterpretation = result["interpretation"]
        observed = [e for e in interp.evidence if e.kind is EvidenceKind.OBSERVED]
        assert all(e.log_odds_contribution is None for e in observed)

    def test_measured_evidence_is_untouched(self, store, prior):
        """追加观察不得影响已有的测量证据。"""
        result = _invoke(_graph(store, prior, _StubMediaExtractor()))
        interp: BehaviorInterpretation = result["interpretation"]
        measured = [e for e in interp.evidence if e.kind is EvidenceKind.MEASURED]
        assert measured
        assert all(e.value is not None for e in measured)


# ═══════════════════════════════════════════════════════════════
# 2. 不变量：观察不影响判断
# ═══════════════════════════════════════════════════════════════


class TestObservationDoesNotAffectJudgement:
    """**本文件最重要的一组。**

    `OBSERVED` 若影响模式选择或候选排序，`MEASURED` / `OBSERVED` 的分栏
    就白做了 —— 未经校验的模型输出会驱动判断。
    """

    def _judgement(self, interp: BehaviorInterpretation) -> tuple:
        return (
            interp.evidence_mode,
            [(c.context, c.posterior, c.matched_count) for c in interp.candidates],
        )

    def test_mode_is_unchanged(self, store, prior):
        without = _invoke(_graph(store, prior, None))["interpretation"]
        with_obs = _invoke(_graph(store, prior, _StubMediaExtractor()))["interpretation"]
        assert with_obs.evidence_mode is without.evidence_mode

    def test_candidate_order_is_unchanged(self, store, prior):
        without = _invoke(_graph(store, prior, None))["interpretation"]
        with_obs = _invoke(_graph(store, prior, _StubMediaExtractor()))["interpretation"]
        assert self._judgement(with_obs) == self._judgement(without)

    def test_only_evidence_grows(self, store, prior):
        """**只有证据列表变长**，其他判断字段一个都不能变。"""
        without = _invoke(_graph(store, prior, None))["interpretation"]
        with_obs = _invoke(_graph(store, prior, _StubMediaExtractor()))["interpretation"]

        assert len(with_obs.evidence) > len(without.evidence)
        assert with_obs.suggested_observation == without.suggested_observation
        assert with_obs.confidence_tier == without.confidence_tier
        assert with_obs.case_total == without.case_total

    def test_observation_does_not_create_candidates(self, store, prior):
        """观察**不得**凭空造出候选 —— 候选来自声学与场景，不来自画面描述。"""
        without = _invoke(_graph(store, prior, None))["interpretation"]
        with_obs = _invoke(_graph(store, prior, _StubMediaExtractor()))["interpretation"]
        assert len(with_obs.candidates) == len(without.candidates)


# ═══════════════════════════════════════════════════════════════
# 3. 失败必须可见
# ═══════════════════════════════════════════════════════════════


class TestFailuresAreVisible:
    def test_extraction_failure_records_an_error(self, store, prior):
        """失败进 `NodeError` —— 不能静默。

        否则无法区分「没配提取器」与「配了但失败」，
        而后者意味着系统少了一整层信息。
        """
        result = _invoke(_graph(store, prior, _StubMediaExtractor(raises=True)))
        errors = result.get("errors", [])
        assert any("多模态观察失败" in e.message for e in errors)

    def test_unusable_observation_records_an_error(self, store, prior):
        obs = ModelObservation(
            media_url="http://x/meow.mp4",
            media_kind=MediaKind.VIDEO,
            source_model="stub",
            ok=False,
            error="模型判断这段媒体不足以观察",
        )
        result = _invoke(_graph(store, prior, _StubMediaExtractor(obs)))
        assert any("不可用" in e.message for e in result.get("errors", []))
        # 且不得追加任何 OBSERVED 证据
        interp = result["interpretation"]
        assert not [e for e in interp.evidence if e.kind is EvidenceKind.OBSERVED]
        trace = " ".join(t.decision or "" for t in result["node_trace"])
        assert "obs=不可用" in trace, f"「模型说看不出」应与「抛异常」区分：{trace}"

    def test_trace_distinguishes_not_configured_from_failed(self, store, prior):
        """trace 必须能区分「没配」与「配了但失败」——**两者含义完全不同**。"""
        none_result = _invoke(_graph(store, prior, None))
        fail_result = _invoke(_graph(store, prior, _StubMediaExtractor(raises=True)))

        none_trace = " ".join(t.decision or "" for t in none_result["node_trace"])
        fail_trace = " ".join(t.decision or "" for t in fail_result["node_trace"])
        assert "obs=未配置" in none_trace, none_trace
        assert "obs=失败" in fail_trace, fail_trace
        assert "obs=未配置" not in fail_trace, "「失败」不得显示成「未配置」"

    def test_observation_is_optional(self, store, prior):
        """**不传提取器不得报错** —— 它是可选依赖。"""
        result = _invoke(_graph(store, prior, None))
        assert result["interpretation"] is not None
        assert not result.get("errors")


# ═══════════════════════════════════════════════════════════════
# 4. 声学失败时观察仍可用
# ═══════════════════════════════════════════════════════════════


class TestObservationSurvivesAudioFailure:
    """原本那种情况**只返回一个错误**，用户什么都看不到 ——
    而模型可能已经从画面里看到了有用的东西。"""

    def _graph_with_failing_audio(self, store, prior, media_extractor):
        def boom(url: str):
            raise ValueError("不支持的音频格式")

        return build_graph(
            store=store,
            embedder=HashEmbedder(),
            llm=MockLLM(default="x"),
            prior=prior,
            feature_extractor=boom,
            vision=_StubVision(),
            media_extractor=media_extractor,
        )

    def test_still_returns_an_interpretation(self, store, prior):
        result = _invoke(
            self._graph_with_failing_audio(store, prior, _StubMediaExtractor())
        )
        interp = result["interpretation"]
        assert interp is not None, "声学失败不应导致完全无输出"
        assert interp.evidence_mode.value == "text_only"
        # 不得携带声学特征（TEXT_ONLY 的硬约束）
        assert interp.acoustic_features is None

    def test_observation_is_preserved(self, store, prior):
        result = _invoke(
            self._graph_with_failing_audio(store, prior, _StubMediaExtractor())
        )
        interp = result["interpretation"]
        observed = [e for e in interp.evidence if e.kind is EvidenceKind.OBSERVED]
        assert observed, "声学失败时更不该丢掉画面观察"
        assert any("抓门" in e.statement for e in observed)

    def test_failure_is_still_recorded(self, store, prior):
        """保住观察不等于把失败藏起来 —— 错误仍然要写进 errors。"""
        result = _invoke(
            self._graph_with_failing_audio(store, prior, _StubMediaExtractor())
        )
        assert any("声学特征提取失败" in e.message for e in result.get("errors", []))

    def test_limitations_state_which_is_which(self, store, prior):
        """limitations 必须说清「有观察但没有测量」—— 两者的可信度不同。"""
        result = _invoke(
            self._graph_with_failing_audio(store, prior, _StubMediaExtractor())
        )
        lim = result["interpretation"].limitations
        assert "声学特征提取失败" in lim
        assert "不是测量" in lim


# ═══════════════════════════════════════════════════════════════
# 5. 预填候选（降低标注负担）
# ═══════════════════════════════════════════════════════════════


class TestActionSuggestions:
    def test_observed_actions_become_suggestions(self, store, prior):
        """录完自动填上「抓门」，主人只需确认 —— 直接降低标注负担。"""
        result = _invoke(_graph(store, prior, _StubMediaExtractor()))
        suggested = result["pending_action_suggestions"]
        assert BehaviorAction.SCRATCH_DOOR in suggested
        assert BehaviorAction.LOOK_AT_DOOR in suggested

    def test_no_suggestions_without_extractor(self, store, prior):
        result = _invoke(_graph(store, prior, None))
        assert result["pending_action_suggestions"] == []

    def test_no_suggestions_on_failure(self, store, prior):
        result = _invoke(_graph(store, prior, _StubMediaExtractor(raises=True)))
        assert result["pending_action_suggestions"] == []

    def test_suggestions_are_candidates_not_writes(self, store, prior):
        """**它只是候选。** 未经确认的模型输出不得成为 `MeowRecord.actions`。"""
        _invoke(_graph(store, prior, _StubMediaExtractor()))
        assert store.list_meow_records(user_id="u", pet_id="p") == []


# ═══════════════════════════════════════════════════════════════
# 6. media_kind 传递
# ═══════════════════════════════════════════════════════════════


class TestMediaKindPlumbing:
    def test_video_kind_is_passed_through(self, store, prior):
        """媒体类型必须传到提取器 —— `.mp4` 与 `.m4a` 对模型是不同的请求。"""
        extractor = _StubMediaExtractor()
        _invoke(_graph(store, prior, extractor))
        assert extractor.calls
        url, kind = extractor.calls[0]
        assert url == "http://x/meow.mp4"
        assert kind is MediaKind.VIDEO

    def test_defaults_to_audio_when_unspecified(self, store, prior):
        extractor = _StubMediaExtractor()
        raw = RawInput(
            audio_url="http://x/meow.m4a", audio_kind=AudioKind.CAT_MEOW
        )  # 不声明 media_kind
        graph = _graph(store, prior, extractor)
        graph.invoke(initial_state(user_id="u", pet_id="p", raw_input=raw))
        assert extractor.calls[0][1] is MediaKind.AUDIO
