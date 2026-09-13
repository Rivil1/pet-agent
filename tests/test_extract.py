"""多模态事实提取的测试。

## 本文件守护的分界

**模型做观察，代码做测量。** 分界不是「谁说的」，而是**「能不能重算」**。

| | 产出 | 可复现 |
| --- | --- | --- |
| `app/audio/features.py` | `AcousticFeatures` | ✅ |
| `app/extract/multimodal.py` | `ModelObservation` | ❌ |

所以本文件有两组断言：

1. **提取器**：不能编造（推断性表述必须被拒、词表外的值必须被丢）
2. **契约**：`MEASURED` 必须带 `value` —— 模型不能伪装成测量
"""

from __future__ import annotations

import json

import httpx
import pytest
from app.extract import MultimodalExtractor, parse_model_observation
from app.llm import CapabilityConfig
from app.schemas import (
    FORBIDDEN_INFERENCE_PHRASES,
    SCENE_OBJECTS,
    AcousticFeatures,
    BehaviorAction,
    BehaviorInterpretation,
    EvidenceItem,
    EvidenceKind,
    EvidenceMode,
    IntentCandidate,
    MediaKind,
    ModelObservation,
    find_forbidden_inference,
)

KEY = "sk-test-key"
BASE = "https://example.test/v1"


def _config() -> CapabilityConfig:
    return CapabilityConfig(
        api_key=KEY,
        base_url=BASE,
        model="test-omni",
        path="/chat/completions",
    )


def _payload(
    *,
    actions: list[str] | None = None,
    scene_objects: list[str] | None = None,
    described_signs: list[str] | None = None,
    usable: bool = True,
) -> str:
    return json.dumps(
        {
            "usable": usable,
            "actions": actions or [],
            "scene_objects": scene_objects or [],
            "described_signs": described_signs or [],
        },
        ensure_ascii=False,
    )


def _extractor(handler, *, model: str = "qwen3-omni-flash") -> MultimodalExtractor:
    return MultimodalExtractor(
        config=_config(),
        model=model,
        transport=httpx.MockTransport(handler),
    )


def _ok_handler(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": _payload(
                                actions=["scratch_door"],
                                scene_objects=["door", "human"],
                                described_signs=["前爪抬起靠近门框"],
                            )
                        }
                    }
                ]
            },
        )

    return handler


# ═══════════════════════════════════════════════════════════════
# 1. 正常提取
# ═══════════════════════════════════════════════════════════════


class TestExtraction:
    def test_extracts_actions_and_scene(self):
        captured: dict = {}
        obs = _extractor(_ok_handler(captured)).extract(
            "https://cdn.example/clip.mp4", media_kind=MediaKind.VIDEO
        )
        assert obs.ok
        assert obs.actions == [BehaviorAction.SCRATCH_DOOR]
        assert obs.scene_objects == ["door", "human"]
        assert obs.described_signs == ["前爪抬起靠近门框"]

    def test_request_shape(self):
        captured: dict = {}
        _extractor(_ok_handler(captured)).extract(
            "https://cdn.example/clip.mp4", media_kind=MediaKind.VIDEO
        )
        assert captured["url"] == f"{BASE}/chat/completions"
        assert captured["auth"] == f"Bearer {KEY}"
        assert captured["body"]["model"] == "qwen3-omni-flash"
        assert captured["body"]["temperature"] == 0.0
        content = captured["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        # 三种媒体都用 video_url 传（实测接口形状，见设计文档）
        assert content[1]["type"] == "video_url"
        assert content[1]["video_url"]["url"] == "https://cdn.example/clip.mp4"

    def test_prompt_lists_the_vocabularies(self):
        captured: dict = {}
        _extractor(_ok_handler(captured)).extract(
            "https://x/a.mp4", media_kind=MediaKind.VIDEO
        )
        prompt = captured["body"]["messages"][0]["content"][0]["text"]
        # 固定词表必须注入 prompt，否则模型给不出可校验的枚举值
        assert "scratch_door" in prompt
        assert "door" in prompt
        # 三条禁令必须在 prompt 里明说
        assert "看得见" in prompt
        assert "不要推断原因" in prompt

    def test_source_model_is_recorded(self):
        """模型会换、会升级。不记来源就无法回溯一个判断是怎么得出的。"""
        obs = _extractor(_ok_handler({}), model="qwen3-vl-flash").extract(
            "https://x/a.mp4", media_kind=MediaKind.VIDEO
        )
        assert obs.source_model == "qwen3-vl-flash"

    def test_audio_uses_the_same_interface(self):
        """音频与视频走同一接口 —— 模型负责解码。"""
        captured: dict = {}
        obs = _extractor(_ok_handler(captured)).extract(
            "https://x/meow.m4a", media_kind=MediaKind.AUDIO
        )
        assert obs.ok
        assert captured["body"]["messages"][0]["content"][1]["type"] == "video_url"

    def test_deterministic_parse(self):
        """同输入 → 同输出。

        ⚠️ `observed_at` **必须显式传入** —— 它的默认值是 `now()`，
        不传的话两次解析的时间戳必然不同（这不是解析不确定，
        而是时间戳本来就不一样）。
        """
        from datetime import datetime, timezone

        raw = _payload(actions=["pacing"], scene_objects=["door"])
        at = datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)
        a = parse_model_observation(
            raw, media_url="u", media_kind=MediaKind.VIDEO, model="m", observed_at=at
        )
        b = parse_model_observation(
            raw, media_url="u", media_kind=MediaKind.VIDEO, model="m", observed_at=at
        )
        assert a.model_dump() == b.model_dump()


# ═══════════════════════════════════════════════════════════════
# 2. 反编造：推断性表述必须被拒
# ═══════════════════════════════════════════════════════════════


class TestNoInference:
    """**prompt 是请求，不是保证。**

    模型的「不要编造」和模型的「编造」来自同一组权重，
    所以词表校验是必需的（第二道防线）。
    """

    @pytest.mark.parametrize(
        "sign",
        [
            "它看起来很焦虑",  # 情感推断
            "它想出去",  # 意图推断
            "因为它饿了",  # 因果断言
            "它可能生病了",  # 健康结论
            "精神不振",  # 健康结论
        ],
    )
    def test_inference_phrase_is_rejected(self, sign):
        with pytest.raises(ValueError, match="推断性表述"):
            ModelObservation(
                media_url="u",
                media_kind=MediaKind.VIDEO,
                source_model="m",
                described_signs=[sign],
            )

    def test_neutral_observation_is_accepted(self):
        obs = ModelObservation(
            media_url="u",
            media_kind=MediaKind.VIDEO,
            source_model="m",
            described_signs=["前爪抬起靠近门框", "耳朵朝向门口"],
        )
        assert obs.ok

    def test_forbidden_list_covers_three_categories(self):
        """词表必须同时覆盖三类：情感、因果、健康。"""
        joined = " ".join(FORBIDDEN_INFERENCE_PHRASES)
        assert "焦虑" in joined  # 情感
        assert "因为" in joined  # 因果
        assert "生病" in joined  # 健康

    def test_detector_is_reusable(self):
        # 用集合比较，不断言顺序 —— 词表顺序是实现细节
        assert set(find_forbidden_inference("因为它饿了")) == {"因为", "饿了"}
        assert find_forbidden_inference("前爪抬起") == []

    def test_extractor_reports_rejection_as_failure_not_crash(self):
        """模型返回推断性内容时，结果是 `ok=False`，**不是异常**。

        与「单段媒体失败」走同一条路径 —— 调用方不必区分两种失败表达。
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": _payload(described_signs=["它看起来很焦虑"])
                            }
                        }
                    ]
                },
            )

        obs = _extractor(handler).extract("https://x/a.mp4", media_kind=MediaKind.VIDEO)
        assert not obs.ok
        assert obs.error and "推断性表述" in obs.error
        assert obs.described_signs == []


# ═══════════════════════════════════════════════════════════════
# 3. 词表边界
# ═══════════════════════════════════════════════════════════════


class TestVocabulary:
    def test_unknown_action_is_dropped_not_mapped(self):
        """词表外的动作**丢弃 rather than 映射** —— 映射意味着我们替模型猜了它的意思。"""
        raw = _payload(actions=["scratch_door", "doing_something_weird"])
        obs = parse_model_observation(
            raw, media_url="u", media_kind=MediaKind.VIDEO, model="m"
        )
        assert obs.actions == [BehaviorAction.SCRATCH_DOOR]

    def test_unknown_scene_object_is_dropped_and_recorded(self):
        """词表外的场景物体**丢弃并记入 `ignored_terms`**。

        与动作的处理保持一致，且**不静默消失** ——
        高频出现词表外的值说明 prompt 或词表要改，
        静默丢弃会让这个信号永远看不到。
        """
        raw = _payload(scene_objects=["door", "spaceship"])
        obs = parse_model_observation(
            raw, media_url="u", media_kind=MediaKind.VIDEO, model="m"
        )
        assert obs.scene_objects == ["door"]
        assert obs.ignored_terms == ["spaceship"]

    def test_scene_object_vocabulary_is_fixed(self):
        assert "door" in SCENE_OBJECTS
        assert "food_bowl" in SCENE_OBJECTS
        assert len(set(SCENE_OBJECTS)) == len(SCENE_OBJECTS)

    def test_no_demeanor_field_in_contract(self):
        """**契约里没有 demeanor 字段** —— 这是刻意的。

        `general.demeanor` 是红旗信号。用未经校验的模型推断驱动它是危险的：
        漏报的代价是猫可能死亡（假阴性不可接受）。
        **神态由系统「问」，不由系统「判断」。**
        """
        fields = set(ModelObservation.model_fields)
        assert "demeanor" not in fields
        assert "mood" not in fields
        assert "emotion" not in fields

    def test_no_resolution_field(self):
        """**不产出 `resolution`** —— 「开门让它停了」是因果，不是画面里的事实。"""
        assert "resolution" not in set(ModelObservation.model_fields)

    def test_no_confidence_field(self):
        """**不给 confidence** —— 模型自报置信度未校准（U1）。

        给一个未校准的数字比没有数字更糟：没有数字时人会保留怀疑，有数字时不会。
        """
        assert "confidence" not in set(ModelObservation.model_fields)


# ═══════════════════════════════════════════════════════════════
# 4. 失败路径
# ═══════════════════════════════════════════════════════════════


class TestFailurePaths:
    def test_usable_false_means_no_observation(self):
        raw = _payload(usable=False)
        obs = parse_model_observation(
            raw, media_url="u", media_kind=MediaKind.VIDEO, model="m"
        )
        assert not obs.ok
        assert obs.error and "usable=false" in obs.error

    def test_failed_observation_cannot_carry_facts(self):
        """失败时携带事实会让降级路径静默产出内容。"""
        with pytest.raises(ValueError, match="不得携带"):
            ModelObservation(
                media_url="u",
                media_kind=MediaKind.VIDEO,
                source_model="m",
                ok=False,
                described_signs=["前爪抬起"],
            )

    @pytest.mark.parametrize("status", [429, 500, 401])
    def test_upstream_error_becomes_failed_observation(self, status):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": "boom"})

        obs = _extractor(handler).extract("https://x/a.mp4", media_kind=MediaKind.VIDEO)
        assert not obs.ok
        assert obs.error

    @pytest.mark.parametrize("raw", ["", "   ", "不是 JSON", "[1,2,3]", "{}"])
    def test_unparsable_output_becomes_failed_observation(self, raw):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

        obs = _extractor(handler).extract("https://x/a.mp4", media_kind=MediaKind.VIDEO)
        # "{}" 是合法 JSON 但无字段 → 成功但无内容
        if raw == "{}":
            assert obs.ok and not obs.is_usable
        else:
            assert not obs.ok

    def test_is_usable_requires_content(self):
        empty = ModelObservation(
            media_url="u", media_kind=MediaKind.VIDEO, source_model="m"
        )
        assert empty.ok and not empty.is_usable


# ═══════════════════════════════════════════════════════════════
# 5. 契约：模型不能伪装成测量
# ═══════════════════════════════════════════════════════════════


def _features() -> AcousticFeatures:
    return AcousticFeatures(
        duration=0.6,
        f0_mean=520.0,
        f0_range=120.0,
        f0_slope=0.1,
        call_rate=3.0,
        ici_mean=0.5,
        rms_mean=0.2,
        roughness=0.2,
    )


class TestModelCannotMasqueradeAsMeasurement:
    """**`MEASURED` 必须带 `value`。**

    这是把「模型不能伪装成测量」从 docstring 变成断言：
    测量值的定义就是「有一个可复算的数字」。
    """

    def test_measured_without_value_is_rejected(self):
        with pytest.raises(ValueError, match="必须带 value"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.MEASURED_ONLY,
                acoustic_features=_features(),
                evidence=[
                    EvidenceItem(
                        kind=EvidenceKind.MEASURED,
                        statement="叫声时长 0.6s",
                        source="acoustic:duration",
                        value=None,  # ← 没有数字
                    )
                ],
                suggested_observation="观察",
            )

    def test_observed_without_value_is_fine(self):
        """`OBSERVED` 不需要 `value` —— 它本来就不是测量。"""
        obs = ModelObservation(
            media_url="u",
            media_kind=MediaKind.VIDEO,
            source_model="qwen3-omni-flash",
            actions=[BehaviorAction.SCRATCH_DOOR],
        )
        result = BehaviorInterpretation(
            evidence_mode=EvidenceMode.MEASURED_ONLY,
            acoustic_features=_features(),
            evidence=[
                EvidenceItem(
                    kind=EvidenceKind.OBSERVED,
                    statement=obs.evidence_statements()[0],
                    source=f"model:{obs.source_model}",
                    value=None,
                )
            ],
            suggested_observation="观察",
        )
        assert result.evidence[0].kind is EvidenceKind.OBSERVED

    def test_observed_never_carries_log_odds(self, prior_factory):
        """`OBSERVED` 不在概率模型里 —— 给它对数几率贡献值是编造。

        即使身处 `acoustic_plus_history` 模式，它也必须为 None。
        """
        # 用工厂造一个**已验证**的先验。
        # 初版读生产文件再把 is_placeholder 改成 False —— 而门禁现在
        # 还要求「实测优于基线」，那个文件不满足，于是这里会报错。
        # 本测试要测的是 OBSERVED 证据的形状，不是先验门禁。
        prior = prior_factory(macro_f1=0.80, majority_baseline=0.50)
        from app.interpreter import interpret

        # 正常测量证据带贡献值
        base = interpret(features=_features(), prior=prior)
        assert all(e.log_odds_contribution is not None for e in base.evidence)

        # 混入一条 OBSERVED 且带贡献值 → 必须被拒
        with pytest.raises(ValueError, match="不得携带 log_odds_contribution"):
            behavior_result = base.model_copy(
                update={
                    "evidence": [
                        *base.evidence,
                        EvidenceItem(
                            kind=EvidenceKind.OBSERVED,
                            statement="画面里观察到：抓门",
                            source="model:m",
                            log_odds_contribution=0.5,
                        ),
                    ]
                }
            )
            BehaviorInterpretation.model_validate(behavior_result.model_dump())

    def test_observed_is_accepted_alongside_measured(self, prior_factory):
        """两者可以在同一条解释里共存 —— 只是各自承担不同的可验证性承诺。"""
        # 用工厂造一个**已验证**的先验。
        # 初版读生产文件再把 is_placeholder 改成 False —— 而门禁现在
        # 还要求「实测优于基线」，那个文件不满足，于是这里会报错。
        # 本测试要测的是 OBSERVED 证据的形状，不是先验门禁。
        prior = prior_factory(macro_f1=0.80, majority_baseline=0.50)
        from app.interpreter import interpret

        base = interpret(features=_features(), prior=prior)
        merged = BehaviorInterpretation.model_validate(
            base.model_copy(
                update={
                    "evidence": [
                        *base.evidence,
                        EvidenceItem(
                            kind=EvidenceKind.OBSERVED,
                            statement="画面里观察到：抓门",
                            source="model:qwen3-omni-flash",
                            value=None,
                            log_odds_contribution=None,
                        ),
                    ]
                }
            ).model_dump()
        )
        kinds = {e.kind for e in merged.evidence}
        assert EvidenceKind.MEASURED in kinds
        assert EvidenceKind.OBSERVED in kinds


# ═══════════════════════════════════════════════════════════════
# 6. 展示措辞
# ═══════════════════════════════════════════════════════════════


class TestStatementWording:
    def test_statements_describe_not_conclude(self):
        """措辞必须停留在观察层 —— 判断权留给主人。"""
        obs = ModelObservation(
            media_url="u",
            media_kind=MediaKind.VIDEO,
            source_model="m",
            actions=[BehaviorAction.SCRATCH_DOOR, BehaviorAction.LOOK_AT_DOOR],
            scene_objects=["door", "human"],
            described_signs=["前爪抬起靠近门框"],
        )
        statements = obs.evidence_statements()
        blob = " ".join(statements)
        # 不得出现推断性表述
        assert find_forbidden_inference(blob) == []
        # 必须说清这是「画面里」看到的，而不是系统的判断
        assert "画面里" in blob
        assert "抓门" in blob
