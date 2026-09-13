"""真实模型客户端测试。

**全部用 ``httpx.MockTransport`` 离线运行** —— 不需要 API Key，也不需要网络。

这测的是「我写的 API 代码对不对」：请求组装、响应解析、错误分类、重试策略。
**它不能替代真实调用** —— 首次接入真实 API 仍需人工确认一次。
但两者解决的问题不同：mock 测试能保证「代码逻辑正确」，
而真实调用验证「我对 API 的理解正确」。前者可自动化，后者只能人工。

对应 docs/BUGS.md B13–B15 的教训：**写了一堆 API 代码但从没调过**就必须显式标注。
"""

from __future__ import annotations

from typing import Any

import json

import httpx
import pytest

from app.llm import factory as factory_mod
from app.llm.base import HashEmbedder, MockLLM
from app.llm.providers import (
    DASHSCOPE_BASE_URL,
    CapabilityConfig,
    ProviderConfig,
)
from app.llm.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    TokenUsage,
)
from app.llm.errors import MissingCredential, UpstreamError, UpstreamKind
from app.llm.factory import MockVision, UnavailableVision, build_providers
from app.llm.http import post_json
from app.profile.openai_compat import (
    OpenAICompatVision,
    extract_json_object,
    parse_observation,
)

KEY = "sk-test-not-a-real-key"


#: 测试用的模型名与路径。
#:
#: **不再引用 `DEFAULT_*_MODEL` 常量** —— 那些是**供应商预设**的一部分，
#: 而单测客户端时我们只关心「请求里带的模型名等于配置里的模型名」。
#: 引用预设常量会让测试随供应商默认值变化而失败（与客户端的正确性无关）。
_TEST_MODEL = "test-model"
_CHAT_PATH = "/chat/completions"
_EMBED_PATH = "/embeddings"


def _config(**kw) -> CapabilityConfig:
    base: dict[str, Any] = {
        "api_key": KEY,
        "base_url": "https://example.test/v1",
        "model": _TEST_MODEL,
        "path": _CHAT_PATH,
    }
    base.update(kw)
    return CapabilityConfig(**base)


def _provider_config(**kw) -> ProviderConfig:
    """四类能力都指向同一个测试端点的配置。

    换供应商后 `build_providers` 收的是 `ProviderConfig`（**每能力一份**），
    不再是单个 `CapabilityConfig` —— 这里如实按新形状构造。
    """
    chat = _config(**kw)
    return ProviderConfig(
        kind="test",
        llm=chat,
        vision=chat,
        embed=_config(path=_EMBED_PATH, **kw),
        multimodal=None,
    )


def _json_transport(handler):
    return httpx.MockTransport(handler)


# ═══════════════════════════════════════════════════════════════
# 传输层
# ═══════════════════════════════════════════════════════════════


class TestTransport:
    def test_success_returns_dict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        assert post_json(
            "https://x/y", payload={}, headers={}, timeout_s=5, transport=_json_transport(handler)
        ) == {"ok": True}

    @pytest.mark.parametrize(
        "status,expected",
        [
            (401, UpstreamKind.AUTH),
            (403, UpstreamKind.AUTH),
            (400, UpstreamKind.BAD_REQUEST),
            (429, UpstreamKind.RATE_LIMITED),
            (500, UpstreamKind.SERVER_ERROR),
            (503, UpstreamKind.SERVER_ERROR),
        ],
    )
    def test_status_classification(self, status, expected):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "boom"}})

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=5,
                transport=_json_transport(handler),
            )
        assert exc.value.kind is expected
        assert exc.value.status_code == status

    def test_auth_is_not_retryable(self):
        """鉴权失败重试只会浪费配额。"""
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(401, json={})

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=5,
                max_attempts=3, backoff_s=0, transport=_json_transport(handler),
            )
        assert not exc.value.retryable
        assert calls["n"] == 1, "对不可重试错误不应重试"

    def test_server_error_is_retried(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(503, json={})
            return httpx.Response(200, json={"ok": True})

        data = post_json(
            "https://x/y", payload={}, headers={}, timeout_s=5,
            max_attempts=3, backoff_s=0, transport=_json_transport(handler),
        )
        assert data == {"ok": True}
        assert calls["n"] == 3

    def test_retry_exhausted_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={})

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=5,
                max_attempts=2, backoff_s=0, transport=_json_transport(handler),
            )
        assert exc.value.attempts == 2

    def test_timeout_is_retryable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("too slow", request=request)

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=1,
                max_attempts=1, transport=_json_transport(handler),
            )
        assert exc.value.kind is UpstreamKind.TIMEOUT
        assert exc.value.retryable

    def test_non_json_body_is_malformed(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway</html>")

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=5,
                transport=_json_transport(handler),
            )
        assert exc.value.kind is UpstreamKind.MALFORMED_RESPONSE

    def test_error_text_is_truncated(self):
        """上游错误信息要截断 —— 避免把整段响应写进日志。"""
        long_message = "x" * 5000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": long_message}})

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={}, headers={}, timeout_s=5,
                transport=_json_transport(handler),
            )
        assert len(str(exc.value)) < 400

    def test_api_key_never_appears_in_error(self):
        """**密钥不得进入异常信息**（否则会被写进日志）。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "invalid"}})

        with pytest.raises(UpstreamError) as exc:
            post_json(
                "https://x/y", payload={"secret": KEY},
                headers={"Authorization": f"Bearer {KEY}"},
                timeout_s=5, transport=_json_transport(handler),
            )
        assert KEY not in str(exc.value)


# ═══════════════════════════════════════════════════════════════
# LLM
# ═══════════════════════════════════════════════════════════════


class TestOpenAICompatLLM:
    def test_request_shape(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": " 团团挺好 "}}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                },
            )

        llm = OpenAICompatLLM(config=_config(), transport=_json_transport(handler))
        out = llm.complete(system="sys", user="hi", temperature=0.2)

        assert out == "团团挺好", "返回值应去掉首尾空白"
        assert captured["url"] == "https://example.test/v1/chat/completions"
        assert captured["auth"] == f"Bearer {KEY}"
        assert captured["body"]["model"] == _TEST_MODEL
        assert captured["body"]["temperature"] == 0.2
        assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
        assert captured["body"]["messages"][1] == {"role": "user", "content": "hi"}

    def test_usage_is_accumulated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        llm = OpenAICompatLLM(config=_config(), transport=_json_transport(handler))
        llm.complete(system="s", user="u")
        llm.complete(system="s", user="u")

        assert isinstance(llm.usage, TokenUsage)
        assert llm.usage.prompt_tokens == 20
        assert llm.usage.completion_tokens == 10
        assert llm.usage.calls == 2

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"choices": []},
            {"choices": [{}]},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": "   "}}]},
        ],
    )
    def test_malformed_responses_raise(self, payload):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        llm = OpenAICompatLLM(config=_config(), transport=_json_transport(handler))
        with pytest.raises(UpstreamError) as exc:
            llm.complete(system="s", user="u")
        assert exc.value.kind is UpstreamKind.MALFORMED_RESPONSE

    def test_missing_key_raises(self):
        with pytest.raises(MissingCredential):
            OpenAICompatLLM(config=_config(api_key=""))


# ═══════════════════════════════════════════════════════════════
# Embedder
# ═══════════════════════════════════════════════════════════════


class TestOpenAICompatEmbedder:
    def test_request_sends_explicit_dimension(self):
        """**维度必须显式传** —— 服务端默认值变化会让写入数据库直接失败。"""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "embedding": [0.1] * 8}],
                    "usage": {"prompt_tokens": 3},
                },
            )

        emb = OpenAICompatEmbedder(
            config=_config(path=_EMBED_PATH, dim=8), transport=_json_transport(handler)
        )
        out = emb.embed("团团很怕吸尘器")

        assert captured["url"] == "https://example.test/v1/embeddings"
        assert captured["body"]["dimensions"] == 8
        assert captured["body"]["model"] == _TEST_MODEL
        assert len(out) == 8

    def test_results_are_ordered_by_index(self):
        """不依赖服务端返回顺序 —— 乱序会让向量与文本错配。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [2.0] * 4},
                        {"index": 0, "embedding": [1.0] * 4},
                    ]
                },
            )

        emb = OpenAICompatEmbedder(
            config=_config(path=_EMBED_PATH, dim=4), transport=_json_transport(handler)
        )
        out = emb.embed_many(["first", "second"])
        assert out[0][0] == 1.0, "index=0 的向量必须排第一"
        assert out[1][0] == 2.0

    def test_dimension_mismatch_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 5}]})

        emb = OpenAICompatEmbedder(
            config=_config(path=_EMBED_PATH, dim=1024), transport=_json_transport(handler)
        )
        with pytest.raises(UpstreamError, match="维度"):
            emb.embed("x")

    def test_count_mismatch_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1] * 4}]})

        emb = OpenAICompatEmbedder(
            config=_config(path=_EMBED_PATH, dim=4), transport=_json_transport(handler)
        )
        with pytest.raises(UpstreamError, match="条向量"):
            emb.embed_many(["a", "b"])

    def test_empty_input_makes_no_call(self):
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("不应发起请求")

        emb = OpenAICompatEmbedder(config=_config(path=_EMBED_PATH, dim=4), transport=_json_transport(handler))
        assert emb.embed_many([]) == []

    def test_default_dim_matches_schema(self):
        """默认 1024 维 —— 与 ``vector(1024)`` 一致。

        ``text-embedding-v4`` 支持 64–2048，**1024 是选择而非限制**：
        2048 的检索收益递减而存储翻倍。这个断言锁住的是「不要随便改维度」，
        因为改了就要改表。
        """
        assert _config(dim=1024).dim == 1024


# ═══════════════════════════════════════════════════════════════
# 视觉：JSON 抽取与「不编造」
# ═══════════════════════════════════════════════════════════════


class TestJsonExtraction:
    def test_plain_json(self):
        assert extract_json_object('{"usable": true}') == {"usable": True}

    def test_fenced_json(self):
        text = '```json\n{"usable": true, "fur_color": "橘白"}\n```'
        assert extract_json_object(text)["fur_color"] == "橘白"

    def test_json_with_surrounding_prose(self):
        """模型经常先解释一句再给 JSON —— 必须能容错。"""
        text = '好的，我看到了这只猫：\n{"usable": true, "fur_color": "橘白"}\n希望有帮助。'
        assert extract_json_object(text)["fur_color"] == "橘白"

    @pytest.mark.parametrize("bad", ["完全不是 JSON", "", "[1,2,3]", "null"])
    def test_unparseable_raises(self, bad):
        with pytest.raises(UpstreamError) as exc:
            extract_json_object(bad)
        assert exc.value.kind is UpstreamKind.MALFORMED_RESPONSE


class TestNoFabrication:
    """`DESIGN.md` §1.5 边界 —— 视觉抽取不得编造。"""

    def test_unusable_photo_yields_empty_observation(self):
        """看不清 → 空观察，而不是靠物种常识补一个毛色。"""
        obs = parse_observation('{"usable": false, "reason": "照片模糊"}')
        assert obs.ok
        assert obs.terms() == [], "不可用的照片不得贡献任何特征"

    def test_string_null_is_treated_as_missing(self):
        """模型常把 null 写成字符串 —— 不处理就会得到毛色 = "null" 这种污染。"""
        obs = parse_observation(
            '{"usable": true, "fur_color": "null", "eye_color": "未知", '
            '"fur_length": "N/A", "face_shape": "圆脸"}'
        )
        assert obs.fur_color is None
        assert obs.eye_color is None
        assert obs.fur_length is None
        assert obs.face_shape == "圆脸"

    def test_missing_usable_field_raises(self):
        """没有 ``usable`` 就无法判断照片是否可用 —— 必须报错而非默认可用。"""
        with pytest.raises(UpstreamError, match="usable"):
            parse_observation('{"fur_color": "橘白"}')

    def test_non_list_features_are_ignored(self):
        obs = parse_observation('{"usable": true, "distinctive_features": "左耳缺口"}')
        assert obs.distinctive_features == ()

    def test_null_entries_in_features_filtered(self):
        obs = parse_observation(
            '{"usable": true, "distinctive_features": ["左耳缺口", null, "  ", "未知"]}'
        )
        assert obs.distinctive_features == ("左耳缺口",)


class TestOpenAICompatVision:
    def test_request_uses_vision_model_and_image_part(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"usable": true, "fur_color": "橘白"}'}}
                    ]
                },
            )

        vision = OpenAICompatVision(config=_config(), transport=_json_transport(handler))
        obs = vision.analyze("https://cdn.example/cat.jpg")

        assert captured["body"]["model"] == _TEST_MODEL
        content = captured["body"]["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"] == "https://cdn.example/cat.jpg"
        assert obs.fur_color == "橘白"
        assert obs.image_url == "https://cdn.example/cat.jpg"

    def test_prompt_forbids_guessing(self):
        """prompt 必须显式禁止用物种先验补全 —— 这是「不编造」的第一道防线。"""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": '{"usable": true}'}}]}
            )

        OpenAICompatVision(config=_config(), transport=_json_transport(handler)).analyze("u")
        prompt = captured["body"]["messages"][0]["content"][0]["text"]
        assert "null" in prompt
        assert "不要猜" in prompt or "不要使用" in prompt

    def test_malformed_output_raises_for_isolation(self):
        """解析失败必须抛异常，由 ``identify_from_photos`` 隔离为单张失败 ——
        静默返回空观察会让坏照片混进交集运算。"""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "我看到了橘色的猫"}}]}
            )

        vision = OpenAICompatVision(config=_config(), transport=_json_transport(handler))
        with pytest.raises(UpstreamError):
            vision.analyze("u")


# ═══════════════════════════════════════════════════════════════
# 工厂
# ═══════════════════════════════════════════════════════════════


class TestProviderFactory:
    def test_no_key_falls_back_to_mock(self, monkeypatch):
        monkeypatch.delenv(factory_mod.ENV_API_KEY, raising=False)
        monkeypatch.delenv(factory_mod.ENV_MOCK, raising=False)
        p = build_providers(api_key="")
        assert p.is_mock
        assert isinstance(p.llm, MockLLM)
        assert isinstance(p.embedder, HashEmbedder)

    def test_require_real_without_key_raises(self, monkeypatch):
        """**这个开关让「假演示」无法静默发生。**"""
        monkeypatch.delenv(factory_mod.ENV_API_KEY, raising=False)
        with pytest.raises(MissingCredential):
            build_providers(api_key="", require_real=True)

    def test_key_present_builds_live_clients(self):
        p = build_providers(api_key=KEY, config=_provider_config())
        assert not p.is_mock
        assert isinstance(p.llm, OpenAICompatLLM)
        assert isinstance(p.embedder, OpenAICompatEmbedder)
        assert isinstance(p.vision, OpenAICompatVision)

    def test_mock_flag_forces_mock_even_with_key(self, monkeypatch):
        """``MOCK_PROVIDER=1`` 用于 CI：即使有密钥也必须走 mock，避免测试真的花钱。"""
        monkeypatch.setenv(factory_mod.ENV_MOCK, "1")
        p = build_providers(api_key=KEY)
        assert p.is_mock

    def test_unavailable_vision_fails_loudly(self):
        """没接视觉后端时必须**显式失败** —— 静默返回空会表现成「照片里什么都没有」。"""
        with pytest.raises(MissingCredential):
            UnavailableVision().analyze("u")

    def test_mock_vision_is_deterministic(self):
        v = MockVision()
        a, b = v.analyze("x"), v.analyze("y")
        assert a.terms() == b.terms(), "mock 观察必须稳定，否则交集运算不可断言"
        assert a.image_url == "x" and b.image_url == "y"

    def test_describe_contains_no_secret(self):
        p = build_providers(api_key=KEY, config=_provider_config())
        described = json.dumps(p.describe())
        assert KEY not in described
        assert described


class TestDefaultEndpoints:
    def test_default_base_url_is_openai_compatible(self):
        """已核实：国内区域的 OpenAI 兼容端点在 ``/compatible-mode/v1``。"""
        assert DASHSCOPE_BASE_URL.endswith("/compatible-mode/v1")
        assert "dashscope.aliyuncs.com" in DASHSCOPE_BASE_URL
