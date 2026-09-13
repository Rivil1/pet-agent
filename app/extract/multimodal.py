"""多模态模型的事实提取。

对应决策：**先用多模态模型跑通，后续在真实调用中优化。**

## 为什么这个决定是对的（对 P0 而言）

我原本提的那条管线（解封装 → 人声过滤 → 时间戳对齐 → 抽帧）是**在没有任何
真实数据的情况下设计出来的**。它的每个环节都有人为设定的阈值与判据
（人声判据尤其容易写宽或写窄），而那些参数**只能靠真实样本调**。

所以先让模型跑，拿到真实的输入输出分布，再决定哪一段值得写成代码 ——
这是与「先建群体先验再写推理」相反的、更务实的顺序。

## 但这个决定有一个不能丢的约束

**模型产出的是 `OBSERVED`，不是 `MEASURED`。**

- `MEASURED` 承诺「重算一次还是它」→ 只有代码能兑现
- `OBSERVED` 是模型的描述 → 每次措辞可能不同，但**来源可记**

把两者混为一栏，会让 `value`/`reference` 的复算承诺失效，
而那是整个证据链可审计性的地基。

契约层已经强制：`MEASURED` 必须带 `value`（见 `app/schemas/behavior.py`）。

## 三道防线

| # | 防线 | 位置 |
| --- | --- | --- |
| 1 | prompt 要求「只描述看得见的」 | 本模块 |
| 2 | **词表校验**：推断性表述 → 拒绝 | `ModelObservation` 契约 |
| 3 | 结构化输出：动作取固定词表而非自由文本 | `BehaviorAction` |

**prompt 是请求，不是保证** —— 模型的「不要编造」和模型的「编造」
来自同一组权重。所以第 2 道是必需的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.llm.errors import UpstreamError
from app.llm.http import bearer_headers, post_json
from app.llm.providers import CapabilityConfig
from app.profile.openai_compat import _content_of, extract_json_object
from app.schemas.behavior import BehaviorAction
from app.schemas.observation import (
    SCENE_OBJECTS,
    MediaKind,
    ModelObservation,
)

#: 提取 prompt。
#:
#: 每一句都在防一种具体的编造方式，改动前先想清楚防的是什么：
#:
#: 1. 「只描述看得见的」→ 防情感/意图推断
#: 2. 「不要推断原因」→ 防因果断言（`resolution` 必须由主人提供）
#: 3. 「看不清就留空」→ 防零填充式的补全
#: 4. 固定词表 → 让输出可校验、可用于后续匹配
_EXTRACT_PROMPT = """你在观察一段关于一只家猫的媒体（照片 / 视频 / 录音）。

**你只负责描述「看得见、听得见」的事实。你不负责理解它。**

严格按以下规则回答：

1. **只描述可直接观察到的。** 不写「它很焦虑」「它想出去」这类推断 ——
   你没有依据，而这类判断会误导它的主人。
2. **不要推断原因。** 不写「因为它饿了」。因果由主人判断，不是你的任务。
3. **看不清、听不清就留空。** 宁可少写，也不要给一个可能错的答案。
   猜错的动作会让主人记录下错误的信息。
4. `scene_objects` 只能从下列词表里选：{objects}
5. `actions` 只能从下列词表里选（英文枚举值）：{actions}
6. `described_signs` 只写**具体的、可直接看到的**迹象
   （如「前爪抬起靠近门」「耳朵向后」），不写「很可爱」这类无法核对的描述。

只输出一个 JSON 对象，不要任何解释文字、不要 markdown 代码块：

{{
  "usable": true,
  "actions": ["scratch_door"],
  "scene_objects": ["door"],
  "described_signs": ["前爪抬起靠近门框"]
}}"""


def _render_prompt() -> str:
    return _EXTRACT_PROMPT.format(
        objects=", ".join(SCENE_OBJECTS),
        actions=", ".join(a.value for a in BehaviorAction),
    )


@dataclass
class MultimodalExtractor:
    """多模态事实提取器。

    Args:
        config: **该能力自己的**连接配置。复用 `CapabilityConfig`（每能力一份）
            —— 不新增一套密钥与超时管理。
        model: 使用的多模态模型名。默认取 `config.model`
            （传进来的 `config` 就是 `MULTIMODAL` 能力那份）。
        transport: 注入用（测试传 `httpx.MockTransport`）。
    """

    config: CapabilityConfig
    model: str | None = None
    transport: object | None = field(default=None, repr=False)
    max_attempts: int = field(default=1, repr=False)

    def extract(self, media_url: str, *, media_kind: MediaKind) -> ModelObservation:
        """对一段媒体做事实提取。

        **失败不抛异常，而是返回 `ok=False` 的观察对象** ——
        与 `OpenAICompatVision` 抛 `UpstreamError` 的做法不同。

        区别的理由：图片档案是**一批**照片取交集，一张失败不影响其余；
        而这里是**单段媒体**，失败就是这次没有观察结果。
        返回 `ok=False` 让调用方可以用同一套「逐条隔离」的逻辑处理，
        而不必区分「异常」与「无结果」两种失败表达。
        """
        model = self.model or self.config.model
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _render_prompt()},
                        self._media_part(media_url, media_kind),
                    ],
                }
            ],
            "temperature": 0.0,
        }

        try:
            data = post_json(
                self.config.endpoint(),
                payload=payload,
                headers=bearer_headers(self.config.api_key),
                timeout_s=self.config.timeout_s,
                max_attempts=self.max_attempts,
                transport=self.transport,  # type: ignore[arg-type]
            )
            # ⚠️ `_content_of` 也必须在 try 里 —— 它会在「模型返回空内容」时
            # 抛 UpstreamError。初版把它放在 try 外，于是一个空响应
            # 会让提取器**抛异常而不是返回 ok=False**，
            # 调用方拿不到统一的失败表达。
            content = _content_of(data)
        except UpstreamError as exc:
            return self._failed(
                media_url, media_kind, model, f"{exc.kind.value}: {exc}"
            )

        try:
            parsed = parse_model_observation(
                content, media_url=media_url, media_kind=media_kind, model=model
            )
        except ValueError as exc:
            # 解析失败**不抛** —— 让调用方能用同一条路径处理所有失败
            return self._failed(media_url, media_kind, model, str(exc))
        return parsed

    @staticmethod
    def _media_part(media_url: str, media_kind: MediaKind) -> dict[str, Any]:
        """构造媒体内容块。

        ## 这个形状是供应商相关的

        | 供应商 | 图 | 视频 / 音频 |
        | --- | --- | --- |
        | DashScope（OpenAI 兼容模式） | `image_url` | `video_url`（**未实测**） |
        | 火山方舟 | `image_url` | **`/responses` 端点 + `input_video`**（形状不同，未实现） |

        DashScope 的 OpenAI 兼容模式里三种媒体都用 `video_url` 传 ——
        这是按文档写的，**尚未对真实 API 验证**（见 `docs/16` 的 V2 清单）。

        火山方舟的视频/音频理解走 Responses API，其内容块形状与这里不同，
        **本模块不实现它** —— 按命名规则猜接口形状正是 `qwen3-omni-flash`
        那个模型名的来历，而它至今未经验证。

        想用方舟做视频理解时有两个选择：
        1. `MULTIMODAL` 能力单独指向 DashScope（配置支持按能力拆）
        2. 按官方文档实现 Responses API 适配器后再接
        """
        return {"type": "video_url", "video_url": {"url": media_url}}

    @staticmethod
    def _failed(
        media_url: str, media_kind: MediaKind, model: str, error: str
    ) -> ModelObservation:
        return ModelObservation(
            media_url=media_url,
            media_kind=media_kind,
            source_model=model,
            ok=False,
            error=error,
        )


def parse_model_observation(
    raw: str,
    *,
    media_url: str,
    media_kind: MediaKind,
    model: str,
    observed_at: datetime | None = None,
) -> ModelObservation:
    """解析模型输出。

    **逐字段容错，但不猜。** 词表外的值被丢弃并记入 `ignored_terms` ——
    既不静默消失，也不因一个坏词而让整条观察失败。

    Args:
        observed_at: 观察时间。**显式传入以保证可复现**；
            不传时取当前时间（这会让两次解析的 `observed_at` 不同）。
    """
    payload, err = _load_json(raw)
    if err:
        raise ValueError(err)

    if isinstance(payload.get("usable"), bool) and not payload["usable"]:
        # 模型自己判断这段媒体不足以观察 —— 这是**正确行为**，不是错误。
        # 与图片档案的 `usable=false` 同一语义。
        # 用 `isinstance(..., bool)` 而非 `is False`：后者与字面量做身份比较，
        # 且会把缺失/非布尔值当作可用，语义更清楚。
        return ModelObservation(
            media_url=media_url,
            media_kind=media_kind,
            source_model=model,
            ok=False,
            error="模型判断这段媒体不足以观察（usable=false）",
            observed_at=observed_at or datetime.now(timezone.utc),
        )

    actions: list[BehaviorAction] = []
    ignored: list[str] = []

    for raw_action in _as_list(payload.get("actions")):
        try:
            actions.append(BehaviorAction(raw_action))
        except ValueError:
            # 词表外的动作**丢弃而不是映射** —— 映射意味着我们替模型猜了它的意思
            ignored.append(raw_action)

    scene_objects: list[str] = []
    for obj in _as_list(payload.get("scene_objects")):
        if obj in SCENE_OBJECTS:
            scene_objects.append(obj)
        else:
            ignored.append(obj)

    described_signs = [
        s.strip() for s in _as_list(payload.get("described_signs")) if s.strip()
    ]

    # 交给契约校验（推断性表述、失败时携带事实）
    # 契约会抛 ValueError，由调用方转成 ok=False
    return ModelObservation(
        media_url=media_url,
        media_kind=media_kind,
        source_model=model,
        ok=True,
        actions=actions,
        scene_objects=scene_objects,
        described_signs=described_signs,
        ignored_terms=ignored,
        observed_at=observed_at or datetime.now(timezone.utc),
    )


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if isinstance(v, (str, int, float))]
    return []


def _load_json(raw: str) -> tuple[dict[str, Any], str]:
    """从模型输出里取出 JSON 对象。复用视觉模块的实现，保持解析行为一致。"""
    if not raw.strip():
        return {}, "模型返回空内容"
    try:
        payload = extract_json_object(raw)
    except Exception as exc:  # noqa: BLE001 — 解析失败要变成可展示的原因
        return {}, f"JSON 解析失败：{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return {}, "JSON 顶层不是对象"
    return payload, ""


def dump_observation(obs: ModelObservation) -> str:
    """调试用。"""
    return json.dumps(
        {
            "ok": obs.ok,
            "model": obs.source_model,
            "actions": [a.value for a in obs.actions],
            "scene_objects": obs.scene_objects,
            "described_signs": obs.described_signs,
            "error": obs.error,
        },
        ensure_ascii=False,
        indent=2,
    )
