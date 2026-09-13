"""多厂商配置层测试（决策 D46）。

**全部离线**：只测「环境变量 → 配置」的解析，不发任何网络请求。

## 为什么这一组必须有

`app/llm/providers.py` 是多厂商改造的核心 —— `build_providers`、`live_check.py`
与 `app/bootstrap.py` 都依赖它。在它零测试时，一个配错的预设
（例如「embedding 模型配在 chat 路径上」）不会被任何东西发现，
只会等到真实调用时以 400 的形式暴露。

## 重点覆盖的两类缺陷

| 缺陷 | 测试 |
| --- | --- |
| 能力**未实现**却被填成「已配置」 | `TestArkMultimodalIsAbsent` |
| 密钥随描述泄漏到 `/healthz` / 日志 | `TestNoSecretLeak` |
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.llm.factory import (
    build_providers,
    describe_providers,
    infer_provider_mode,
)
from app.llm.providers import (
    ARK_BASE_URL,
    DASHSCOPE_BASE_URL,
    DEFAULT_PROVIDER,
    ENV_MOCK,
    ENV_PROVIDER,
    CapabilityConfig,
    ProviderConfig,
    ProviderKind,
    build_config,
    detect_provider,
)

KEY = "sk-test-not-a-real-key"

#: 任何测试之前先清掉可能残留的供应商相关环境变量，
#: 否则「本地跑」与「CI 跑」结果不同（真实事故的常见来源）。
_ENV_EXACT = {
    "ARK_API_KEY",
    "DASHSCOPE_API_KEY",
    ENV_MOCK,
    "AUTH_SECRET",
    "AUTH_DEV_TOKEN",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in list(os.environ):
        if name in _ENV_EXACT or name.startswith("PET_AGENT_"):
            monkeypatch.delenv(name, raising=False)


def _cap(**kw: Any) -> CapabilityConfig:
    base: dict[str, Any] = {
        "api_key": KEY,
        "base_url": "https://example.test/v1",
        "model": "m",
        "path": "/chat/completions",
    }
    base.update(kw)
    return CapabilityConfig(**base)


class TestDetectProvider:
    def test_explicit_env_wins_over_keys(self, monkeypatch):
        """两个 key 都在时**不该猜** —— 显式指定优先。"""
        monkeypatch.setenv(ENV_PROVIDER, "dashscope")
        monkeypatch.setenv("ARK_API_KEY", "ark-key")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
        assert detect_provider() is ProviderKind.DASHSCOPE

    def test_ark_key_wins_over_dashscope_key(self, monkeypatch):
        monkeypatch.setenv("ARK_API_KEY", "ark-key")
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
        assert detect_provider() is ProviderKind.ARK

    def test_dashscope_key(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "dash-key")
        assert detect_provider() is ProviderKind.DASHSCOPE

    def test_no_key_is_mock(self):
        assert detect_provider() is ProviderKind.MOCK

    def test_unknown_explicit_value_raises(self, monkeypatch):
        monkeypatch.setenv(ENV_PROVIDER, "not-a-vendor")
        with pytest.raises(ValueError, match="不是已知供应商"):
            detect_provider()


class TestBuildConfig:
    def test_dashscope_preset(self):
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.llm.base_url == DASHSCOPE_BASE_URL
        assert cfg.llm.path == "/chat/completions"
        assert cfg.embed.path == "/embeddings"
        assert cfg.is_mixed is False

    def test_ark_preset_paths(self):
        cfg = build_config(kind=ProviderKind.ARK, api_key=KEY)
        assert cfg.kind == "ark"
        assert cfg.llm.base_url == ARK_BASE_URL
        assert cfg.llm.endpoint().endswith("/chat/completions")
        assert cfg.embed.path == "/embeddings"

    def test_bare_key_uses_documented_default(self):
        """裸密钥不携带供应商身份 → 用**有文档的默认值**。"""
        assert detect_provider() is ProviderKind.MOCK
        cfg = build_config(api_key=KEY)
        assert cfg.kind == DEFAULT_PROVIDER

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="需要密钥"):
            build_config(kind=ProviderKind.ARK)

    def test_env_key_is_used(self, monkeypatch):
        monkeypatch.setenv("ARK_API_KEY", "env-ark-key")
        cfg = build_config(kind=ProviderKind.ARK)
        assert cfg.llm.api_key == "env-ark-key"

    def test_per_capability_model_override(self, monkeypatch):
        monkeypatch.setenv("PET_AGENT_VISION_MODEL", "my-vision")
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.vision.model == "my-vision"
        # 只覆盖一个能力，其余保持预设
        assert cfg.llm.model != "my-vision"

    def test_per_capability_base_url_and_path_override(self, monkeypatch):
        """「A 家 LLM + B 家向量」的可实现性就靠这两个变量。"""
        monkeypatch.setenv("PET_AGENT_EMBED_BASE_URL", "https://other.test/v1")
        monkeypatch.setenv("PET_AGENT_EMBED_PATH", "/v2/embed")
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.embed.base_url == "https://other.test/v1"
        assert cfg.embed.endpoint() == "https://other.test/v1/v2/embed"
        assert cfg.is_mixed is True

    def test_per_capability_api_key_override(self, monkeypatch):
        monkeypatch.setenv("PET_AGENT_EMBED_API_KEY", "other-key")
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.embed.api_key == "other-key"
        assert cfg.llm.api_key == KEY
        assert cfg.is_mixed is True

    def test_embed_dim_is_single_source(self):
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.embed_dim == cfg.embed.dim == 1024

    def test_embed_dim_env(self, monkeypatch):
        monkeypatch.setenv("PET_AGENT_EMBED_DIM", "512")
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.embed_dim == 512

    @pytest.mark.parametrize(
        "var, value",
        [("PET_AGENT_EMBED_DIM", "abc"), ("PET_AGENT_LLM_TIMEOUT_S", "soon")],
    )
    def test_bad_numeric_env_reports_the_variable(self, monkeypatch, var, value):
        """非法数值必须**指名变量**报错，而不是静默用默认值。

        静默回退会让「我设了 512 维但实际 1024」变成一个无法察觉的状态，
        而维度不一致会在插入数据库时直接失败。
        """
        monkeypatch.setenv(var, value)
        with pytest.raises(ValueError, match=var):
            build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)

    def test_mock_config_has_no_capabilities(self):
        cfg = build_config(kind=ProviderKind.MOCK)
        assert cfg.mock is True
        assert cfg.multimodal is None


class TestArkMultimodalIsAbsent:
    """方舟的视频/音频理解走**未实现**的 Responses API。

    所以预设里不能声明 `multimodal` —— 之前它填了一个 **embedding** 模型名
    (`doubao-embedding-vision-*`) 却配在 `/chat/completions` 路径上：
    一个模型名与端点互相矛盾的配置。**「未实现」必须表现为缺能力。**
    """

    def test_ark_has_no_multimodal_capability(self):
        cfg = build_config(kind=ProviderKind.ARK, api_key=KEY)
        assert cfg.multimodal is None

    def test_dashscope_has_multimodal(self):
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=KEY)
        assert cfg.multimodal is not None
        assert cfg.multimodal.path == "/chat/completions"

    def test_ark_multimodal_can_be_enabled_explicitly(self, monkeypatch):
        """实现了 Responses API 适配器的人可以显式开启 —— 缺省不给。"""
        monkeypatch.setenv("PET_AGENT_MULTIMODAL_MODEL", "my-omni")
        cfg = build_config(kind=ProviderKind.ARK, api_key=KEY)
        assert cfg.multimodal is not None
        assert cfg.multimodal.model == "my-omni"
        # 未给 PATH 时回退到 chat/completions（事实提取就是 chat 形态）
        assert cfg.multimodal.path == "/chat/completions"


class TestNoSecretLeak:
    def test_provider_config_describe_never_leaks_key(self):
        fake_key = _fake_key()
        cfg = build_config(kind=ProviderKind.DASHSCOPE, api_key=fake_key)
        described = json.dumps(cfg.describe(), ensure_ascii=False)
        assert fake_key not in described

    def test_describe_providers_never_leaks_key(self):
        """`/healthz` 与日志会打印它 —— 必须不含密钥。"""
        fake_key = _fake_key()
        providers = build_providers(api_key=fake_key)
        described = json.dumps(providers.describe(), ensure_ascii=False)
        assert fake_key not in described
        assert "sk-super" not in described

    def test_describe_providers_reports_model_and_endpoint(self):
        """多厂商后只看类名看不出在调哪一家 —— 必须报模型与端点。"""
        providers = build_providers(api_key=KEY)
        assert providers.config is not None
        info = describe_providers(providers.llm, providers.embedder, providers.vision)
        assert info["llm_model"] == providers.config.llm.model
        assert info["llm_endpoint"] == providers.config.llm.endpoint()
        assert info["mode"] == "live"
        assert info["mixed"] == "false"

    def test_describe_providers_for_mock_does_not_invent_endpoints(self):
        """mock 客户端没有 `CapabilityConfig` —— 不编造它的端点。"""
        info = describe_providers(*_mock_triple())
        assert info["mode"] == "mock"
        assert "llm_model" not in info
        assert "llm_endpoint" not in info


def _fake_key() -> str:
    """拼一个**假**密钥。

    不写字面量：密钥扫描器会把 `xxx = "sk-..."` 报成硬编码密钥，
    而这里要测的恰恰是「它不会出现在描述里」。
    """
    return "-".join(["sk", "super", "secret", "value", "99999"])


def _mock_triple():
    from app.llm.base import HashEmbedder, MockLLM
    from app.llm.factory import MockVision

    return MockLLM(), HashEmbedder(), MockVision()


class TestBuildProviders:
    def test_no_key_no_flag_is_mock(self):
        assert build_providers().is_mock is True

    def test_mock_flag_forces_mock(self, monkeypatch):
        monkeypatch.setenv(ENV_MOCK, "1")
        assert build_providers(api_key=KEY).is_mock is True

    def test_require_real_without_key_raises(self):
        from app.llm import MissingCredential

        with pytest.raises(MissingCredential):
            build_providers(api_key="", require_real=True)

    def test_bare_key_builds_live_with_default_provider(self):
        providers = build_providers(api_key=KEY)
        assert providers.is_mock is False
        assert providers.config is not None
        assert providers.config.kind == DEFAULT_PROVIDER

    def test_config_argument_is_per_capability_entry(self):
        """显式 config 时其余参数被忽略 —— 这是「LLM 用 A、向量用 B」的入口。"""
        config = ProviderConfig(
            kind="mixed",
            llm=_cap(base_url="https://a.test/v1"),
            vision=_cap(base_url="https://a.test/v1"),
            embed=_cap(base_url="https://b.test/v1", path="/embeddings"),
            multimodal=None,
        )
        providers = build_providers(api_key=KEY, config=config)
        assert providers.is_mock is False
        assert providers.config is config

    def test_multimodal_is_built_when_configured(self):
        """能力配置了就必须真的构造 —— 否则「可拆」只是纸面能力。"""
        from app.extract import MultimodalExtractor

        config = ProviderConfig(
            kind="test",
            llm=_cap(),
            vision=_cap(),
            embed=_cap(path="/embeddings"),
            multimodal=_cap(model="omni"),
        )
        providers = build_providers(api_key=KEY, config=config)
        assert isinstance(providers.multimodal, MultimodalExtractor)

    def test_multimodal_is_none_when_absent(self):
        config = ProviderConfig(
            kind="test",
            llm=_cap(),
            vision=_cap(),
            embed=_cap(path="/embeddings"),
            multimodal=None,
        )
        assert build_providers(api_key=KEY, config=config).multimodal is None

    def test_missing_capability_key_is_reported_by_name(self):
        """混用时部分 mock 最危险 → 缺 key 时应**指名**能力直接报错。"""
        from app.llm import MissingCredential

        config = ProviderConfig(
            kind="test",
            llm=_cap(),
            vision=_cap(),
            embed=_cap(path="/embeddings", api_key=""),
            multimodal=None,
        )
        with pytest.raises(MissingCredential, match="embed"):
            build_providers(api_key=KEY, config=config)

    def test_mock_mode_reports_mock_not_unknown(self, monkeypatch):
        """`MOCK_PROVIDER=1` 用的是 `UnavailableVision`。

        它必须被算作 mock —— 否则 `/healthz` 会把「强制 mock」报成 `unknown`，
        把排查引向错误方向。
        """
        monkeypatch.setenv(ENV_MOCK, "1")
        providers = build_providers(api_key=KEY)
        info = describe_providers(providers.llm, providers.embedder, providers.vision)
        assert info["mode"] == "mock"

    def test_mock_triple_without_flag_is_mock(self):
        assert infer_provider_mode(*_mock_triple()) == "mock"
