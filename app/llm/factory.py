"""Provider 工厂：按配置选择真实模型或 mock。

对应 docs/ARCHITECTURE.md §8.3（``MOCK_PROVIDER`` 开关）。

**为什么需要它**：`DESIGN.md` §6.5 要求「无 API Key / 无网络时也能跑通全部评测」。
所以默认行为是「有 Key 就用真的，没有就用 mock 并**明确告知**」——
而不是启动就崩，也不是**静默**退回 mock。

最后一点很重要：静默退回 mock 会让一次真实演示悄悄变成假演示。
所以 mock 模式会带一个 ``is_mock`` 标志，并可由 API 暴露在响应里。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from app.llm.base import Embedder, HashEmbedder, LLMClient, MockLLM
from app.llm.errors import MissingCredential
from app.llm.openai_compat import OpenAICompatEmbedder, OpenAICompatLLM
from app.llm.providers import (
    ENV_API_KEY,
    ENV_MOCK,
    ENV_PROVIDER,
    ProviderConfig,
    ProviderKind,
    build_config,
    detect_provider,
)
from app.profile.analyzer import VisionAnalyzer, VisualObservation
from app.profile.openai_compat import OpenAICompatVision

__all__ = ["ENV_API_KEY", "ENV_MOCK", "ENV_PROVIDER", "Providers", "build_providers"]

#: mock 模式下返回的固定观察 —— 用于离线跑通档案链路。
#: 刻意不在不同照片间变化，以便 ``intersect_observations`` 产生稳定的
#: ``must_keep_features``，让离线测试可断言。
_MOCK_OBSERVATION = VisualObservation(
    image_url="",
    fur_color="橘白",
    fur_length="短毛",
    eye_color="黄色",
    body_shape="偏圆",
    face_shape="圆脸",
    distinctive_features=("胸口一块白毛",),
)


class MockVision:
    """确定性视觉分析替身。

    **它不是「假装理解」**，而是返回固定观察，让上游的多图取交集逻辑
    可被离线测试覆盖。真实语义能力由 ``OpenAICompatVision`` 提供。
    """

    def __init__(self, observation: VisualObservation | None = None) -> None:
        self._observation = observation or _MOCK_OBSERVATION
        self.calls: list[str] = []

    def analyze(self, image_url: str) -> VisualObservation:
        self.calls.append(image_url)
        # ``VisualObservation`` 是 frozen dataclass —— 用 ``replace`` 而非 ``model_copy``
        return replace(self._observation, image_url=image_url)


class UnavailableVision:
    """没有可用视觉后端时的替身。

    **显式失败而非静默返回空** —— 否则「没接视觉模型」会表现成
    「照片里什么都看不到」，那是在伪造一个看起来正常的结论。
    """

    def analyze(self, image_url: str) -> VisualObservation:
        raise MissingCredential(
            "未配置视觉后端（缺少 DASHSCOPE_API_KEY）。"
            "档案功能需要视觉模型，或显式使用 MockVision 做离线测试。"
        )


#: 已知的 mock 实现。
#:
#: 为什么用推断而不是加一个 `is_mock` 参数：
#: **调用方会忘记传的参数**正是 B22 的失败模式（工厂加了参数但上游没传 →
#: 能力静静地不存在）。推断不会被忘记。
#:
#: `UnavailableVision` 也在内：它是**显式缺失**的占位（一调就报 MissingCredential），
#: 不是未知实现。若把它算作 `unknown`，`MOCK_PROVIDER=1` 下 `/healthz`
#: 会报 `unknown` 而不是 `mock` —— 标签错了，排查时会被引到错误方向。
_MOCK_TYPES: tuple[type, ...] = (
    MockLLM,
    HashEmbedder,
    MockVision,
    UnavailableVision,
)

#: 已知的**真实**实现。
#:
#: 为什么需要白名单，而不是「不是 mock 就算 live」：
#: 后者会把**未知实现**（测试桩、第三方适配器、写错的对象）也算成真实推理 ——
#: 而那是一个危险方向的假阳性。
#:
#: > **声称 `live` 需要有正面证据**，不能从「它不是已知的 mock」推出来。
#: > 这与整个项目的 fail-closed 取向一致：无法确认时宁可说不知道。
_LIVE_TYPES: tuple[type, ...] = (
    OpenAICompatLLM,
    OpenAICompatEmbedder,
    OpenAICompatVision,
)


def infer_provider_mode(llm: object, embedder: object, vision: object) -> str:
    """推断 provider 模式。**三种状态，不是两种。**

    | 返回 | 条件 |
    | --- | --- |
    | `mock` | 三个全是**已知**占位实现 |
    | `live` | 三个全是**已知**真实实现 |
    | **`unknown`** | 其余一切（混合 / 未知实现） |

    ## 为什么必须有 `unknown`，而且两边都要白名单

    初版用 `all(isinstance(p, _MOCK_TYPES))` 两分，两个方向都错：

    | 输入 | 初版 | 正确 | 为什么 |
    | --- | --- | --- | --- |
    | MockLLM + 自定义桩 | `live` | **`unknown`** | 声称有真实推理，实际大部分是占位 |
    | 三个自定义桩 | `live` | **`unknown`** | 未知实现不等于真实实现 |

    > 安全检查报错的方向很重要：**把 mock 报成 live，比把 live 报成 mock 危险得多。**
    > 前者让一次假演示看起来像真演示；后者只是多一个警告。
    """
    modes = {_classify_type(p) for p in (llm, embedder, vision)}
    if modes == {"mock"}:
        return "mock"
    if modes == {"live"}:
        return "live"
    return "unknown"


def _classify_type(obj: object) -> str:
    if isinstance(obj, _MOCK_TYPES):
        return "mock"
    if isinstance(obj, _LIVE_TYPES):
        return "live"
    return "unknown"


def describe_providers(llm: object, embedder: object, vision: object) -> dict[str, str]:
    """给 API / 日志用的描述。**不含密钥。**

    除类名外还带**模型名与端点** —— 换供应商后最需要能核对的就是它们。
    类名现在是中性的（`OpenAICompatLLM`），只看它看不出在调哪一家；
    多厂商（D46）之后这变成一个真实的观测盲区。

    `mixed` 表示三项是否指向了不同的端点 —— 混用时只报 `mode` 会让人
    以为三项同源。
    """
    info: dict[str, str] = {
        "mode": infer_provider_mode(llm, embedder, vision),
        "llm": type(llm).__name__,
        "embedder": type(embedder).__name__,
        "vision": type(vision).__name__,
    }

    endpoints: list[tuple[str, str]] = []
    for field, obj in (("llm", llm), ("embedder", embedder), ("vision", vision)):
        config = getattr(obj, "config", None)
        model = getattr(config, "model", None)
        endpoint = getattr(config, "endpoint", None)
        if not isinstance(model, str) or not callable(endpoint):
            # mock 与自定义桩没有 `CapabilityConfig` —— 不编造它们的端点
            continue
        info[f"{field}_model"] = model
        info[f"{field}_endpoint"] = str(endpoint())
        endpoints.append(
            (str(getattr(config, "base_url", "")), str(getattr(config, "api_key", "")))
        )

    # 密钥只用于**比较**是否同源，绝不写进返回值
    info["mixed"] = str(len(set(endpoints)) > 1).lower()
    return info


#: 哪些模式下的输出**不得当作真实推理**。
#:
#: `unknown` 也在内 —— 无法确认是真模型时，应当与 mock 同等对待。
UNTRUSTED_MODES: frozenset[str] = frozenset({"mock", "unknown"})


@dataclass(frozen=True)
class Providers:
    """一组可互相替换的 provider。"""

    llm: LLMClient
    embedder: Embedder
    vision: VisionAnalyzer
    is_mock: bool
    config: ProviderConfig | None = None
    multimodal: object | None = None
    """多模态事实提取器（若配置了）。**不参与 `is_mock` 判定** —— 它是可选能力。"""

    def describe(self) -> dict[str, str]:
        """给健康检查/日志用的描述。**不含密钥。**

        包含**模型名与端点** —— 换供应商后最需要能核对的就是它们。
        只报 `mode` 与类名不够：类名现在是中性的（`OpenAICompatLLM`），
        看不出在调哪一家。
        """
        out = {
            "mode": "mock" if self.is_mock else "live",
            "llm": type(self.llm).__name__,
            "embedder": type(self.embedder).__name__,
            "vision": type(self.vision).__name__,
        }
        if self.config is not None:
            out.update({f"cfg.{k}": v for k, v in self.config.describe().items()})
        out["multimodal_extractor"] = (
            type(self.multimodal).__name__ if self.multimodal else "(未配置)"
        )
        return out

    def require(self, capability: str) -> object:
        """取一个能力，缺失时给出可操作的错误。"""
        value = getattr(self, capability, None)
        if value is None:
            raise MissingCredential(
                f"能力 {capability!r} 未配置。请在 ProviderConfig 里提供它。"
            )
        return value


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def build_providers(
    *,
    api_key: str | None = None,
    config: ProviderConfig | None = None,
    provider: ProviderKind | str | None = None,
    require_real: bool = False,
    transport: object | None = None,
) -> Providers:
    """构造 provider 集合。

    Args:
        api_key: 显式密钥（覆盖所有能力）。
        config: 完整配置。**给出它时其余参数被忽略** —— 这是「按能力拆」的入口：
            你可以让 LLM 用方舟、向量用 DashScope。
        provider: 供应商。为 ``None`` 时按环境变量推断。
        require_real: 为 ``True`` 时缺少密钥直接报错；
            为 ``False``（默认）时退回 mock。**这个开关让「假演示」无法静默发生。**
        transport: 测试注入（``httpx.MockTransport``）。

    Raises:
        MissingCredential: ``require_real=True`` 但没有密钥。
        ValueError: 显式指定的供应商名非法，或有密钥但配置不完整。
    """
    want_mock = _env_flag(ENV_MOCK)

    if config is None:
        kind = _resolve_kind(provider)
        if kind is ProviderKind.MOCK and api_key:
            # 显式给了密钥但没说明供应商，且环境变量也认不出 ——
            # 用一个**有文档的默认值**（裸密钥本身不携带供应商身份）。
            from app.llm.providers import DEFAULT_PROVIDER

            kind = ProviderKind(DEFAULT_PROVIDER)
        missing = kind is ProviderKind.MOCK
        if require_real and missing:
            raise MissingCredential(
                f"require_real=True 但未提供任何供应商密钥。\n"
                f"可用环境变量：ARK_API_KEY（火山方舟）/ DASHSCOPE_API_KEY，"
                f"或显式设置 {ENV_PROVIDER}。"
            )
        if missing or want_mock:
            return _mock_providers(unavailable_vision=want_mock)
        config = build_config(kind=kind, api_key=api_key)

    if config.mock:
        return _mock_providers(unavailable_vision=want_mock)

    _require_keys(config)
    return Providers(
        llm=OpenAICompatLLM(config=config.llm, transport=transport),
        embedder=OpenAICompatEmbedder(config=config.embed, transport=transport),
        vision=OpenAICompatVision(config=config.vision, transport=transport),
        is_mock=False,
        config=config,
        multimodal=_build_multimodal(config, transport),
    )


def _build_multimodal(
    config: ProviderConfig, transport: object | None
) -> object | None:
    """构造多模态事实提取器。**未配置该能力时返回 `None`。**

    这一步以前缺失，于是 `Providers.multimodal` 恒为 `None` ——
    「每能力可拆」里那个 `multimodal` 能力配了也不生效。

    返回 `None` 而不是空实现：调用方用 `is None` 区分
    「**未配置**」与「**配置了但失败**」，两者给用户的提示不同
    （见 `app/graph/nodes.py` 的 `configured` 判断）。
    """
    if config.multimodal is None:
        return None
    # 延迟导入：`app.extract` 依赖 `app.llm.providers`，
    # 在 `app.llm` 包初始化期间顶层导入会形成脆弱的导入顺序依赖。
    from app.extract import MultimodalExtractor

    return MultimodalExtractor(config=config.multimodal, transport=transport)


def _resolve_kind(provider: ProviderKind | str | None) -> ProviderKind:
    if provider is None:
        return detect_provider()
    if isinstance(provider, ProviderKind):
        return provider
    try:
        return ProviderKind(provider.lower())
    except ValueError as exc:
        allowed = " / ".join(k.value for k in ProviderKind)
        raise ValueError(f"未知供应商 {provider!r}（允许：{allowed}）") from exc


def _require_keys(config: ProviderConfig) -> None:
    """每个被启用的能力都必须有密钥。

    **不在这里「回退到 mock」** —— 混用时部分 mock 是最危险的状态：
    它会让报告出来的 mode 变成 `unknown`（见 B26），
    而用户以为自己配好了。宁可启动时直接报错。
    """
    caps = {"llm": config.llm, "vision": config.vision, "embed": config.embed}
    if config.multimodal is not None:
        caps["multimodal"] = config.multimodal
    missing = [name for name, cap in caps.items() if not cap.api_key.strip()]
    if missing:
        raise MissingCredential(
            f"以下能力缺少 API Key：{'、'.join(missing)}。"
            "请设置对应的 `PET_AGENT_<能力>_API_KEY`，"
            f"或为供应商设置 {config.kind} 的主密钥。"
        )


def _mock_providers(*, unavailable_vision: bool) -> Providers:
    return Providers(
        llm=MockLLM(default="团团看起来挺好的。"),
        embedder=HashEmbedder(),
        vision=UnavailableVision() if unavailable_vision else MockVision(),
        is_mock=True,
        config=build_config(kind=ProviderKind.MOCK),
    )
