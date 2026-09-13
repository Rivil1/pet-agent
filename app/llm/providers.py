"""供应商配置：**每个能力可以指向不同的供应商**。

## 为什么必须能拆

四类能力（LLM / 视觉 / 向量 / 多模态事实提取）在各家的覆盖不同：

| | LLM | 视觉 | 向量 | 多模态事实提取 |
| --- | --- | --- | --- | --- |
| DashScope | ✅ | ✅ | ✅ | ✅ `qwen3-omni-flash` |
| **火山方舟** | ✅ | ✅ | ✅ | ❌ **未实现**（见下）|
| DeepSeek | ✅ | △ 实验性 | ❌ **没有** | ❌ |

**绑在一条配置上，就没法用「A 家的 LLM + B 家的向量」** ——
而 DeepSeek 恰恰是必须混用的那一家。

（曾把四个能力塞进一个 `DashScopeConfig`，所以混用做不到。
**已由决策 D46 解掉**：每个能力各持一份 `CapabilityConfig`。）

## 火山方舟的接口差异（已查官方文档）

| | DashScope | 火山方舟 |
| --- | --- | --- |
| base_url | `dashscope.aliyuncs.com/compatible-mode/v1` | `ark.cn-beijing.volces.com/api/v3` |
| 对话 / 事实提取路径 | `/chat/completions` | `/chat/completions`（同）|
| 文本向量路径 | `/embeddings` | `/embeddings`（同）|
| 环境变量 | `DASHSCOPE_API_KEY` | `ARK_API_KEY` |

所以 `path` 必须是**每个能力各自的字段**，不能全局常量。

> 方舟另有一套**多模态向量**接口（`/embeddings/multimodal`），
> 但本项目**不用**它 —— 我们的 `multimodal` 能力是 **chat 形态的事实提取**
> （见 `app/extract/multimodal.py`），不是 embedding。两者不可互换。

## ⚠️ 未实现：火山方舟的视频/音频理解

方舟的视频与音频理解走 **Responses API**（`POST /api/v3/responses`），
其内容块形状（`input_video` 等）与 OpenAI 的 `chat/completions` **不同**，
而且官方文档把它与 Files API 上传串在一起。

**我没有验证过那套形状，所以不实现它** ——
按命名规则猜接口形状正是 `qwen3-omni-flash` 那个模型名的来历，
而它至今未经验证（见 `docs/16-model-selection.md` 的 V2）。

所以方舟预设里**根本不声明 `multimodal_model`** —— 该能力为 `None`，
调用方按「未配置」处理。

> 曾经这里填的是 `doubao-embedding-vision-*`（一个 **embedding** 模型），
> 却配在 `/chat/completions` 路径上 —— 一个模型名与端点互相矛盾的配置，
> 调下去只会 400。**「能力未实现」必须表现为缺能力，
> 不能表现为填一个看起来像配置的东西。**

需要视频/音频理解时有两个选择：

1. 用 DashScope 的多模态通道（`multimodal` 能力单独指向 DashScope）
2. 按官方文档实现 Responses API 适配器后再接

**这就是「每能力可拆」的实际价值**：一项没实现不影响其余三项。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ProviderKind(str, Enum):
    DASHSCOPE = "dashscope"
    ARK = "ark"
    MOCK = "mock"


# ─────────────────────────────────────────────────────────────
# 预设：base_url / 路径 / 建议模型名
# ─────────────────────────────────────────────────────────────

#: 火山方舟官方 OpenAI 兼容端点。**已核对官方文档。**
ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: 各预设的路径与建议模型名。
#:
#: ⚠️ **模型名需要你按自己账号可用的列表核对** —— 方舟的模型名带日期后缀
#: （如 `-260215`），且账号可能只开了部分模型。全部可用环境变量覆盖。
_PRESETS: dict[ProviderKind, dict[str, str]] = {
    ProviderKind.DASHSCOPE: {
        "base_url": DASHSCOPE_BASE_URL,
        "llm_model": "qwen-plus",
        "vision_model": "qwen3-vl-flash",
        "embed_model": "text-embedding-v4",
        "multimodal_model": "qwen3-omni-flash",
        "chat_path": "/chat/completions",
        "embed_path": "/embeddings",
        "multimodal_path": "/chat/completions",
        "env_key": "DASHSCOPE_API_KEY",
    },
    ProviderKind.ARK: {
        "base_url": ARK_BASE_URL,
        # 方舟模型名带日期后缀。这三个是按官方模型列表填的**建议值**，
        # 请用 `PET_AGENT_LLM_MODEL` 等环境变量覆盖成你账号里可用的名字。
        "llm_model": "doubao-seed-2-0-mini-260428",
        "vision_model": "doubao-seed-2-0-mini-260428",
        "embed_model": "doubao-embedding-vision-251215",
        # ⚠️ **故意没有 `multimodal_model`** ——
        # 方舟的视频/音频理解走未实现的 Responses API（见模块 docstring）。
        # 填一个 embedding 模型名会让「未实现」伪装成「已配置」。
        # 若你已实现 Responses API 适配器，用
        # `PET_AGENT_MULTIMODAL_MODEL`（+ `_PATH` / `_BASE_URL`）显式开启。
        "chat_path": "/chat/completions",
        "embed_path": "/embeddings",
        "env_key": "ARK_API_KEY",
    },
}


@dataclass(frozen=True)
class CapabilityConfig:
    """**单个能力**的连接配置。

    四个能力各持一份 —— 这样「LLM 用方舟、向量用 DashScope」才可能。
    """

    api_key: str
    base_url: str
    model: str
    path: str
    timeout_s: float = 30.0
    dim: int = field(
        default=1024,
        metadata={
            "doc": (
                "向量维度。**仅 `EMBED` 能力使用**，其余能力忽略它。\n\n"
                "放在这里而不是全局，因为它是**能力的属性**："
                "同一套系统里视觉与向量可能来自不同供应商，维度只属于向量那一个。"
            )
        },
    )

    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.path}"


@dataclass(frozen=True)
class ProviderConfig:
    """一套 provider 配置。"""

    kind: str
    llm: CapabilityConfig
    vision: CapabilityConfig
    embed: CapabilityConfig
    multimodal: CapabilityConfig | None = None
    mock: bool = False

    @property
    def embed_dim(self) -> int:
        """向量维度。**单一来源是 `embed.dim`** —— 不在两处各存一份。

        两处各存一份就会出现「改了 A 忘了改 B」，而维度不一致的后果是
        **插入数据库时直接失败**（`vector(1024)` 不匹配）。
        """
        return self.embed.dim

    def describe(self) -> dict[str, str]:
        """给 `/healthz` 与日志用。**不含密钥。**"""
        out = {
            "kind": self.kind,
            "llm": self.llm.model,
            "vision": self.vision.model,
            "embed": self.embed.model,
            "embed_dim": str(self.embed_dim),
        }
        out["multimodal"] = self.multimodal.model if self.multimodal else "(未配置)"
        # 混用时要能看出来 —— 只看 kind 会以为四项同一家
        out["mixed"] = str(self.is_mixed).lower()
        return out

    @property
    def is_mixed(self) -> bool:
        """四类能力是否指向不同的端点。"""
        caps = [self.llm, self.vision, self.embed]
        if self.multimodal is not None:
            caps.append(self.multimodal)
        return len({(c.base_url, c.api_key) for c in caps}) > 1


# ─────────────────────────────────────────────────────────────
# 从环境变量构造
# ─────────────────────────────────────────────────────────────

#: 显式传入 `api_key` 但没说明供应商时用的默认值。
#:
#: **这是一个有文档的默认，不是猜测。** 一个裸密钥本身不携带供应商身份 ——
#: DashScope 的 key 在方舟上会被拒（401），反之亦然。
#:
#: 默认改为火山方舟（ARK）：用户说「pet项目模型现在都先用火山引擎」。
#: 用 DashScope 时请显式传 `provider="dashscope"`，或设 `DASHSCOPE_API_KEY`。
DEFAULT_PROVIDER = "ark"

ENV_PROVIDER = "PET_AGENT_PROVIDER"
ENV_API_KEY = "PET_AGENT_API_KEY"
ENV_MOCK = "MOCK_PROVIDER"

#: 每套配置**总是**具备的能力。
_CORE_CAPABILITIES = ("LLM", "VISION", "EMBED")

#: 全部能力（含可选的 `MULTIMODAL`）——
#: 每个都可用 `PET_AGENT_<能力>_MODEL` / `_BASE_URL` / `_PATH` / `_API_KEY` 覆盖。
_CAPABILITIES = (*_CORE_CAPABILITIES, "MULTIMODAL")


def _env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def detect_provider() -> ProviderKind:
    """按可用的环境变量推断供应商。

    顺序：显式指定 → 有 ARK key → 有 DashScope key → mock。
    **显式指定优先** —— 两个 key 都在时不该猜。
    """
    explicit = _env(ENV_PROVIDER)
    if explicit:
        try:
            return ProviderKind(explicit.lower())
        except ValueError as exc:
            allowed = " / ".join(k.value for k in ProviderKind)
            raise ValueError(
                f"{ENV_PROVIDER}={explicit!r} 不是已知供应商（允许：{allowed}）"
            ) from exc

    if _env("ARK_API_KEY"):
        return ProviderKind.ARK
    if _env("DASHSCOPE_API_KEY"):
        return ProviderKind.DASHSCOPE
    return ProviderKind.MOCK


def build_config(
    *,
    kind: ProviderKind | None = None,
    api_key: str | None = None,
) -> ProviderConfig:
    """构造配置。

    Args:
        kind: 供应商。为 ``None`` 时按环境变量推断。
        api_key: 显式密钥（会覆盖所有能力的 key）。

    Raises:
        ValueError: 供应商需要密钥但没有。

    ``api_key`` 给了但供应商认不出时，用**有文档的默认供应商** ——
    裸密钥本身不携带供应商身份（DashScope 的 key 在方舟上会被 401）。
    这与 `build_providers` 同一套规则；此前两处不一致，
    导致直接调 `build_config(api_key=...)` 会**静默丢掉密钥**返回 mock 配置。
    """
    explicit_kind = kind is not None
    resolved = kind or detect_provider()
    if not explicit_kind and resolved is ProviderKind.MOCK and api_key:
        resolved = ProviderKind(DEFAULT_PROVIDER)

    if resolved is ProviderKind.MOCK:
        return _mock_config()

    preset = _PRESETS[resolved]
    key = (api_key or _env(preset["env_key"]) or _env(ENV_API_KEY) or "").strip()
    if not key:
        raise ValueError(
            f"供应商 {resolved.value} 需要密钥。"
            f"请设置 {preset['env_key']}（或 {ENV_API_KEY}）。"
        )

    caps = {
        name: _capability(resolved, name, key, api_key_override=api_key is not None)
        for name in _CORE_CAPABILITIES
    }
    # 多模态事实提取是**可选**能力：方舟未实现，所以预设里不声明它。
    # 未声明时给 `None`，调用方按「未配置」走降级路径。
    multimodal = (
        _capability(resolved, "MULTIMODAL", key, api_key_override=api_key is not None)
        if _declares(resolved, "MULTIMODAL")
        else None
    )
    return ProviderConfig(
        kind=resolved.value,
        llm=caps["LLM"],
        vision=caps["VISION"],
        embed=caps["EMBED"],
        multimodal=multimodal,
        mock=False,
    )


def _declares(kind: ProviderKind, name: str) -> bool:
    """该供应商是否声明了这项能力（或用户用环境变量显式开启）。

    **未声明 ≠ 空实现。** 方舟的视频/音频理解走未实现的 Responses API，
    所以方舟预设里没有 `multimodal_model` —— 能力为 `None`，
    调用方按「未配置」处理（见 `app/graph/nodes.py` 的 `configured` 判断）。

    若在这里回退到某个模型名，「未实现」就会变成「配置了但一调就 400」。
    """
    if f"{name.lower()}_model" in _PRESETS[kind]:
        return True
    return _env(f"PET_AGENT_{name}_MODEL") is not None


def _capability(
    kind: ProviderKind, name: str, key: str, *, api_key_override: bool
) -> CapabilityConfig:
    preset = _PRESETS[kind]
    lower = name.lower()

    base_url = _env(f"PET_AGENT_{name}_BASE_URL") or preset["base_url"]
    path_key = (
        "multimodal_path"
        if name == "MULTIMODAL"
        else ("chat_path" if name in ("LLM", "VISION") else "embed_path")
    )
    # 能力未声明时（如方舟的 MULTIMODAL）允许只给 `_MODEL` 就开启 ——
    # 路径回退到 chat/completions（事实提取就是 chat 形态）。
    path = _env(f"PET_AGENT_{name}_PATH") or preset.get(path_key) or preset["chat_path"]
    model = _env(f"PET_AGENT_{name}_MODEL") or preset.get(f"{lower}_model")
    if not model:
        raise ValueError(
            f"供应商 {kind.value} 未声明 {name} 的模型名，"
            f"请用 PET_AGENT_{name}_MODEL 指定"
        )
    cap_key = _env(f"PET_AGENT_{name}_API_KEY") or key

    return CapabilityConfig(
        api_key=cap_key,
        base_url=base_url,
        model=model,
        path=path,
        timeout_s=_float_env(f"PET_AGENT_{name}_TIMEOUT_S", 30.0),
        dim=_embed_dim() if name == "EMBED" else 1024,
    )


def _float_env(name: str, default: float) -> float:
    """读一个浮点环境变量。**非法值报错而不是静默用默认值。**

    静默回退会让「我设了 60s 但实际跑 30s」变成一个无法察觉的状态，
    而超时长度直接决定一个慢请求是成功还是降级。
    """
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} 不是数字") from exc


def _embed_dim() -> int:
    raw = _env("PET_AGENT_EMBED_DIM")
    if raw is None:
        return 1024
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"PET_AGENT_EMBED_DIM={raw!r} 不是整数") from exc


def _mock_config() -> ProviderConfig:
    """mock 配置。

    **仍然填真实的 URL 与路径** —— 这样 `describe()` 报出来的端点是有意义的，
    而 mock 与否由 `mock` 字段表示，不靠「URL 是空的」来暗示。
    """
    placeholder = CapabilityConfig(
        api_key="", base_url="", model="(mock)", path="", timeout_s=5.0
    )
    return ProviderConfig(
        kind=ProviderKind.MOCK.value,
        llm=placeholder,
        vision=placeholder,
        embed=placeholder,
        multimodal=None,
        mock=True,
    )
