"""OpenAI 兼容的视觉分析：照片 → ``VisualObservation``。

与 `app/llm/openai_compat.py` 同一思路 —— **端点与模型名全部来自配置**，
模块里没有供应商专属的东西。

（原名 `app/profile/dashscope.py`。接入火山方舟后那个名字开始说谎，
而名字与行为不符是 bug 的温床。）
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace

from app.llm.errors import UpstreamError, UpstreamKind
from app.llm.http import bearer_headers, post_json
from app.llm.providers import CapabilityConfig
from app.profile.analyzer import VisualObservation

#: 分析 prompt。
#:
#: **每一句都在防一种编造方式**，改动前请先想清楚防的是什么。
_VISION_PROMPT = """你在为一只真实的猫建立外貌档案。这张照片将作为它长期的身份锚点，
所以准确比完整重要。

严格按以下规则回答：

1. 先判断「这张照片是否足以判断这只猫的外观」。如果猫太小、模糊、逆光、
   被遮挡、只是局部特写、或有其他猫，就把 usable 设为 false。
2. 只描述**照片里实际看得见的**。不要使用「橘猫通常……」「短毛猫一般……」这类物种先验。
3. **看不清的字段一律填 null**，不要猜。宁可留空，也不要给一个可能错的答案。
4. distinctive_features 只列**明显且具体**的特征（如「左耳有小缺口」「胸口一块白毛」）。
   不要写「很可爱」「毛茸茸」这类无法用于核对的描述。没有就留空数组。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块：

{
  "usable": true,
  "fur_color": "毛色描述或 null",
  "fur_length": "短毛/中长毛/长毛，或 null",
  "eye_color": "眼睛颜色或 null",
  "body_shape": "体型描述或 null",
  "face_shape": "脸型描述或 null",
  "distinctive_features": ["明显的个体特征"]
}"""

#: 匹配 ```json ... ``` 或 ``` ... ``` 围栏。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)




@dataclass
class OpenAICompatVision:
    """视觉分析客户端。实现 ``VisionAnalyzer`` 协议。"""

    config: CapabilityConfig
    transport: object | None = field(default=None, repr=False)
    max_attempts: int = field(default=1, repr=False)

    def analyze(self, image_url: str) -> VisualObservation:
        """分析一张照片。

        Raises:
            UpstreamError: 调用失败，或模型返回的内容无法解析为约定的 JSON。
                调用方（``identify_from_photos``）会把它隔离为「这一张失败」，
                而**不会**让一张坏照片静默污染交集。
        """
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        data = post_json(
            self.config.endpoint(),
            payload=payload,
            headers=bearer_headers(self.config.api_key),
            timeout_s=self.config.timeout_s,
            max_attempts=1,
            transport=self.transport,  # type: ignore[arg-type]
        )
        raw = _content_of(data)
        parsed = parse_observation(raw)
        # ``VisualObservation`` 是 **frozen dataclass**，不是 Pydantic 模型 ——
        # 没有 ``model_copy``。用 ``dataclasses.replace``。
        # 把它当成 Pydantic 模型用会让每一次真实视觉调用都崩。
        return replace(parsed, image_url=image_url)


def _content_of(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "视觉响应没有 choices")
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise UpstreamError(UpstreamKind.MALFORMED_RESPONSE, "视觉模型返回空内容")
    return content.strip()


def extract_json_object(text: str) -> dict:
    """从模型输出里抽出 JSON 对象。

    容忍三种常见形态：纯 JSON、``\\`\\`\\`json`` 围栏、以及前后夹带解释文字。

    Raises:
        UpstreamError: 抽不出合法 JSON 对象。
    """
    candidates: list[str] = []
    fenced = _FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    candidates.append(text.strip())

    # 兜底：取第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise UpstreamError(
        UpstreamKind.MALFORMED_RESPONSE,
        f"模型输出无法解析为 JSON 对象（前 120 字符：{text[:120]!r}）",
    )


def _clean_str(value: object) -> str | None:
    """把模型给的字符串规整化。

    ``"null"`` / ``"none"`` / ``"unknown"`` / ``"未知"`` 这些**字符串形式的空值**
    要当成 ``None`` —— 模型经常把 null 写成字符串，若不处理，
    就会得到毛色 = "null" 这种污染档案的特征。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.lower() in {"null", "none", "n/a", "unknown", "unclear", "不可见", "未知", "不确定"}:
        return None
    return cleaned


def parse_observation(raw: str) -> VisualObservation:
    """把模型输出解析为 ``VisualObservation``。

    若模型自报 ``usable=false``，返回一个**空观察** —— 它仍算「分析成功」，
    只是没有提供任何特征。这样它在交集运算里既不加支持也不加反对，
    且会通过 ``terms()`` 为空来体现「这张照片没有可用信息」。

    Raises:
        UpstreamError: JSON 结构不符合约定（缺 ``usable`` 字段等）。
    """
    data = extract_json_object(raw)

    if "usable" not in data:
        raise UpstreamError(
            UpstreamKind.MALFORMED_RESPONSE,
            "模型输出缺少必需的 usable 字段，无法判断这张照片是否可用",
        )

    usable = bool(data.get("usable"))
    if not usable:
        return VisualObservation(image_url="")

    features = data.get("distinctive_features")
    if features is None:
        distinctive: tuple[str, ...] = ()
    elif isinstance(features, list):
        distinctive = tuple(
            c for c in (_clean_str(f) for f in features) if c is not None
        )
    else:
        # 模型给了个字符串而不是数组 —— 不猜它想表达什么，直接当作无
        distinctive = ()

    return VisualObservation(
        image_url="",
        fur_color=_clean_str(data.get("fur_color")),
        fur_length=_clean_str(data.get("fur_length")),
        eye_color=_clean_str(data.get("eye_color")),
        body_shape=_clean_str(data.get("body_shape")),
        face_shape=_clean_str(data.get("face_shape")),
        distinctive_features=distinctive,
    )
