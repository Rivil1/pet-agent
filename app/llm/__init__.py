"""模型与向量化抽象。

**两层**：

| 层 | 内容 | 用途 |
|---|---|---|
| 协议 + mock | ``Embedder`` / ``LLMClient`` / ``HashEmbedder`` / ``MockLLM`` | 离线可跑的测试与评测 |
| 真实实现 | ``OpenAICompatLLM`` / ``OpenAICompatEmbedder`` / ``OpenAICompatVision`` | 生产 |

用 ``build_providers(...)`` 按环境选择，它带 ``is_mock`` 标志 ——
**避免一次真实演示静默退化成假演示**。
"""

from app.llm.base import (
    DEFAULT_EMBED_DIM,
    Embedder,
    HashEmbedder,
    LLMClient,
    MockLLM,
    cosine,
)
from app.llm.openai_compat import (
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    TokenUsage,
)
from app.llm.providers import (
    ARK_BASE_URL,
    DASHSCOPE_BASE_URL,
    ENV_PROVIDER,
    CapabilityConfig,
    ProviderConfig,
    ProviderKind,
    build_config,
    detect_provider,
)
from app.llm.errors import MissingCredential, UpstreamError, UpstreamKind
from app.llm.factory import (
    UNTRUSTED_MODES,
    describe_providers,
    infer_provider_mode,
    ENV_API_KEY,
    ENV_MOCK,
    MockVision,
    Providers,
    UnavailableVision,
    build_providers,
)

__all__ = [
    # 协议与 mock
    "Embedder",
    "HashEmbedder",
    "LLMClient",
    "MockLLM",
    "cosine",
    "DEFAULT_EMBED_DIM",
    # 真实实现
    "OpenAICompatLLM",
    "OpenAICompatEmbedder",
    # ── 多厂商配置 ──
    "ProviderKind",
    "ProviderConfig",
    "CapabilityConfig",
    "build_config",
    "detect_provider",
    "ARK_BASE_URL",
    "DASHSCOPE_BASE_URL",
    "ENV_PROVIDER",
    "TokenUsage",
    # 错误
    "UpstreamError",
    "UpstreamKind",
    "MissingCredential",
    # 工厂
    "Providers",
    "build_providers",
    "describe_providers",
    "infer_provider_mode",
    "UNTRUSTED_MODES",
    "MockVision",
    "UnavailableVision",
    "ENV_API_KEY",
    "ENV_MOCK",
]
