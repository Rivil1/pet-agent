"""OpenAI 兼容的模型客户端。

**这个模块里没有任何供应商专属的东西** —— 端点、模型名、超时全部来自
[`CapabilityConfig`][app.llm.providers.CapabilityConfig]。

## 为什么改名

它原来叫 `OpenAICompatLLM` / `OpenAICompatEmbedder`。而接入火山方舟之后，
那个名字就开始说谎了 —— 同一个类会被用来调 `ark.cn-beijing.volces.com`。

> **名字与行为不符是 bug 的温床**，而且是最难发现的那类：
> 读代码的人会按名字推断行为，然后被误导。
> （本项目已记过多次同类问题：B2 / B20 / B24 / B25。）

## 供应商差异放在配置里

`CapabilityConfig.path` 是**每个能力各自的字段**，因为不同供应商的路径不同 ——
火山方舟的多模态向量在 `/embeddings/multimodal`，而 DashScope 在 `/embeddings`。

`CapabilityConfig.endpoint()` 负责拼接，客户端不关心具体路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.errors import MissingCredential, UpstreamError, UpstreamKind
from app.llm.http import bearer_headers, post_json
from app.llm.providers import CapabilityConfig


class TokenUsage:
    """token 消耗。用于 `ARCHITECTURE.md` §6.2 的 ``llm_tokens_total`` 指标。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, *, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1


def _as_int(value: object) -> int:
    """宽容地解析上游给的整数。**token 统计不该因上游格式怪而变成 500。**

    它不影响正确性判断（用量只用于成本观测），所以坏值降级为 0 而不是报错。
    """
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _row_index(row: object) -> int:
    """取向量条目的 `index`，非字典或非数值时退到 0。"""
    if not isinstance(row, dict):
        return 0
    try:
        return int(row.get("index", 0))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _usage_of(payload: dict) -> tuple[int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0
    return _as_int(usage.get("prompt_tokens")), _as_int(usage.get("completion_tokens"))


def _require_key(api_key: str | None, *, endpoint: str = "") -> str:
    if not api_key or not api_key.strip():
        where = f"（端点 {endpoint}）" if endpoint else ""
        # 不写死供应商：同一个类会被用来调 DashScope 与方舟。
        raise MissingCredential(
            f"缺少 API Key{where}。请在环境变量中配置"
            "（PET_AGENT_<能力>_API_KEY，或供应商主密钥），"
            "或改用 mock provider（MOCK_PROVIDER=1）。"
        )
    return api_key.strip()


# ─────────────────────────────────────────────────────────────
# 文本生成
# ─────────────────────────────────────────────────────────────


@dataclass
class OpenAICompatLLM:
    """文本生成客户端。实现 ``app.llm.base.LLMClient`` 协议。"""

    config: CapabilityConfig
    usage: TokenUsage = field(default_factory=TokenUsage)
    transport: object | None = field(default=None, repr=False)
    max_attempts: int = field(
        default=1,
        metadata={
            "doc": (
                "最大尝试次数。**默认 1（不重试）** —— 重试只对可重试类失败有意义，"
                "而 `MALFORMED_RESPONSE` 刻意不在可重试集合里："
                "同一个 prompt 再问一次不会让模型输出合法 JSON。"
            )
        },
    )

    def __post_init__(self) -> None:
        _require_key(self.config.api_key, endpoint=self.config.endpoint())

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        """生成一段文本。

        Raises:
            UpstreamError: 上游失败。携带 ``retryable`` 供节点决定是否降级。
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        data = post_json(
            self.config.endpoint(),
            payload=payload,
            headers=bearer_headers(self.config.api_key),
            timeout_s=self.config.timeout_s,
            max_attempts=self.max_attempts,
            transport=self.transport,  # type: ignore[arg-type]
        )
        prompt_tokens, completion_tokens = _usage_of(data)
        # 必须用关键字调用：``TokenUsage.add`` 是关键字专用参数，
        # ``add(*tuple)`` 会直接抛 TypeError（曾经就是这么错的）。
        self.usage.add(prompt=prompt_tokens, completion=completion_tokens)
        return _first_message_content(data)


def _first_message_content(data: dict) -> str:
    """从 chat/completions 响应里取首个 choice 的正文。"""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "响应中没有 choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "choices[0] 不是对象")
    message = first.get("message")
    if not isinstance(message, dict):
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "choices[0].message 缺失")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "模型返回了空内容")
    return content.strip()


# ─────────────────────────────────────────────────────────────
# 文本向量化
# ─────────────────────────────────────────────────────────────


@dataclass
class OpenAICompatEmbedder:
    """文本向量化客户端。实现 ``app.llm.base.Embedder`` 协议。"""

    config: CapabilityConfig
    usage: TokenUsage = field(default_factory=TokenUsage)
    transport: object | None = field(default=None, repr=False)
    max_attempts: int = field(
        default=1,
        metadata={
            "doc": (
                "最大尝试次数。**默认 1（不重试）** —— 重试只对可重试类失败有意义，"
                "而 `MALFORMED_RESPONSE` 刻意不在可重试集合里："
                "同一个 prompt 再问一次不会让模型输出合法 JSON。"
            )
        },
    )

    def __post_init__(self) -> None:
        _require_key(self.config.api_key)

    @property
    def dim(self) -> int:
        return self.config.dim

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。

        **显式传 ``dimensions``**：服务端默认值可能变化，
        而向量维度必须与数据库列定义严格一致（``vector(1024)``），
        静默变维会让插入直接失败。
        """
        if not texts:
            return []
        payload = {
            "model": self.config.model,
            "input": texts,
            "dimensions": self.config.dim,
            "encoding_format": "float",
        }
        data = post_json(
            self.config.endpoint(),
            payload=payload,
            headers=bearer_headers(self.config.api_key),
            timeout_s=self.config.timeout_s,
            max_attempts=self.max_attempts,
            transport=self.transport,  # type: ignore[arg-type]
        )
        prompt_tokens, _ = _usage_of(data)
        self.usage.add(prompt=prompt_tokens, completion=0)

        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise UpstreamError(
                UpstreamKind.MALFORMED_RESPONSE,
                f"返回 {len(rows) if isinstance(rows, list) else '非列表'} 条向量，"
                f"但请求了 {len(texts)} 条",
            )

        # 按 index 排序，不依赖服务端返回顺序
        ordered = sorted(rows, key=_row_index)
        vectors: list[list[float]] = []
        for row in ordered:
            if not isinstance(row, dict):
                raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "向量条目不是对象")
            emb = row.get("embedding")
            if not isinstance(emb, list) or len(emb) != self.config.dim:
                got = len(emb) if isinstance(emb, list) else "非列表"
                raise UpstreamError(
                    UpstreamKind.MALFORMED_RESPONSE,
                    f"向量维度为 {got}，期望 {self.config.dim}",
                )
            try:
                vector = [float(x) for x in emb]
            except (TypeError, ValueError) as exc:
                raise UpstreamError(
                    UpstreamKind.MALFORMED_RESPONSE, f"向量含非数值元素：{exc}"
                ) from exc
            vectors.append(vector)
        return vectors
