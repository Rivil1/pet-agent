"""provider 模式推断的测试。

## 为什么这组测试重要

真实模型需要 `DASHSCOPE_API_KEY`，而**没配密钥时系统会退到 mock 而不报错**。
所以「现在跑的是真模型还是占位」这个判断本身就是一道安全机制。

**而安全机制自己也会错。** 初版用 `all(isinstance(p, _MOCK_TYPES))` 两分，
两个方向都错：

| 输入 | 初版 | 正确 |
| --- | --- | --- |
| MockLLM + 自定义桩 | `live` ← **危险** | `unknown` |
| 三个自定义桩 | `live` ← **危险** | `unknown` |

> 报错方向很重要：**把 mock 报成 live，比把 live 报成 mock 危险得多。**
> 前者让一次假演示看起来像真演示；后者只是多一个警告。

（见 `BUGS.md` B26）
"""

from __future__ import annotations

import pytest
from app.llm import (
    UNTRUSTED_MODES,
    HashEmbedder,
    MockLLM,
    describe_providers,
    infer_provider_mode,
)
from app.llm.openai_compat import CapabilityConfig, OpenAICompatEmbedder, OpenAICompatLLM
from app.llm.factory import MockVision
from app.profile.openai_compat import OpenAICompatVision


class _CustomStub:
    """未知实现：既不是已知 mock，也不是已知真实。"""


def _config() -> CapabilityConfig:
    return CapabilityConfig(
        api_key="sk-test",
        base_url="https://example.test/v1",
        model="test-model",
        path="/chat/completions",
    )


@pytest.fixture()
def live_triple():
    cfg = _config()
    return (
        OpenAICompatLLM(config=cfg),
        OpenAICompatEmbedder(config=cfg),
        OpenAICompatVision(config=cfg),
    )


class TestThreeStateInference:
    def test_all_mocks_is_mock(self):
        assert infer_provider_mode(MockLLM(), HashEmbedder(), MockVision()) == "mock"

    def test_all_real_is_live(self, live_triple):
        assert infer_provider_mode(*live_triple) == "live"

    def test_mixed_is_unknown(self):
        """**初版在这里报 `live`** —— 声称有真实推理，实际大部分是占位。"""
        assert (
            infer_provider_mode(MockLLM(), HashEmbedder(), _CustomStub()) == "unknown"
        )

    def test_unknown_implementations_are_unknown(self):
        """**未知实现不等于真实实现。**

        初版的逻辑是「不是 mock 就算 live」，于是三个测试桩也会报 `live`。
        声称 `live` 需要有**正面证据**（在已知真实实现的名单里）。
        """
        assert (
            infer_provider_mode(_CustomStub(), _CustomStub(), _CustomStub())
            == "unknown"
        )

    @pytest.mark.parametrize(
        "triple",
        [
            ("mock", "mock", "unknown"),
            ("live", "live", "unknown"),
            ("mock", "live", "live"),
        ],
    )
    def test_any_mixture_is_unknown(self, triple):
        pool = {
            "mock": MockLLM,
            "live": lambda: OpenAICompatLLM(config=_config()),
            "unknown": _CustomStub,
        }
        assert infer_provider_mode(*(pool[k]() for k in triple)) == "unknown"

    def test_describe_includes_types_for_diagnosis(self):
        """光有 mode 不够 —— 出问题时要知道**具体哪个**不是真实实现。"""
        info = describe_providers(MockLLM(), HashEmbedder(), _CustomStub())
        assert info["mode"] == "unknown"
        assert info["llm"] == "MockLLM"
        assert info["vision"] == "_CustomStub"


class TestUntrustedModes:
    def test_unknown_is_untrusted(self):
        """**`unknown` 必须与 `mock` 同等处理。**

        无法确认是真模型时，应当告诉调用方别当真，
        而不是默默让它以为一切正常。
        """
        assert "mock" in UNTRUSTED_MODES
        assert "unknown" in UNTRUSTED_MODES
        assert "live" not in UNTRUSTED_MODES


class TestBuildProviders:
    def test_no_key_gives_mock(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("MOCK_PROVIDER", raising=False)
        from app.llm import build_providers

        assert build_providers().is_mock is True

    def test_key_gives_live(self):
        from app.llm import build_providers

        p = build_providers(api_key="sk-test")
        assert p.is_mock is False
        assert p.describe()["mode"] == "live"

    def test_mock_flag_forces_mock_even_with_key(self, monkeypatch):
        from app.llm import build_providers

        monkeypatch.setenv("MOCK_PROVIDER", "1")
        p = build_providers(api_key="sk-test")
        assert p.is_mock is True
        assert p.describe()["mode"] == "mock"

    def test_missing_key_raises_when_real_required(self):
        from app.llm import MissingCredential, build_providers

        with pytest.raises(MissingCredential):
            build_providers(api_key="", require_real=True)

    def test_describe_never_leaks_the_key(self):
        """描述会进日志与 `/healthz` —— **不得含密钥**。"""
        from app.llm import build_providers

        secret = "sk-super-secret-value-12345"
        described = str(build_providers(api_key=secret).describe())
        assert secret not in described
        assert "sk-" not in described
