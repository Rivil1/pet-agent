"""多模态模型的事实观察契约。

## 定位：模型做「关系性描述」，不做「理解」

用户能观察「它在抓门」，不能观察「它想出去」（见 `DESIGN.md` 边界 4）。
多模态模型在**关系性描述**（猫 ↔ 门 / 人 / 食盆）上接近可靠，
在**情感与意图推断**上不可靠 —— 后者正是 B21 与文献证据指向的同一件事。

所以本模块的字段被刻意**限制在可观察范围**：

| 允许 | 不允许 |
| --- | --- |
| `actions`：抓门 / 来回走 / 竖尾（固定词表） | 「它很焦虑」「它想出去」 |
| `scene_objects`：门 / 食盆 / 人 / 窗台 | 「因为它饿了」 |
| `described_signs`：可见的身体迹象 | 任何疾病、诊断、因果 |

## 三条刻意的缺席，每一个都对应一处已识别风险

### 1. **没有 `confidence` 字段**

模型的「置信度」是自报的，未校准（U1）。给一个未校准的数字，
用户会当真 —— 而它比没有数字更糟：**没有数字时人会保留怀疑，有数字时不会。**

### 2. **不产出 `resolution`（结果）**

「我开了门，它出去了」是**因果**，不是画面里的事实。
20 秒片段里的时间相邻不等于因果。`resolution` 必须由主人提供。

### 3. **不产出 `demeanor`（神态）**

`general.demeanor` 在 `data/health/red_flags.yaml` 里是**红旗信号**
（值域 `normal / lethargic / agitated`）。

用未经校验的模型推断去驱动一个健康信号，后果是：
- 误报 → 虚惊（可接受）
- **漏报 → 红旗不触发**（不可接受，假阴性代价是猫可能死亡）

所以：**神态由系统「问」，不由系统「判断」。**
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.behavior import BehaviorAction


class MediaKind(str, Enum):
    """媒体类型。音频与视频走同一个提取接口 —— 模型负责解码。"""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


#: 允许进入 `scene_objects` 的词表。
#:
#: **固定词表而非自由文本**：自由文本无法做后续匹配，
#: 而这里的用途正是「把画面里的东西映射到记录卡的情境选项」。
SCENE_OBJECTS: tuple[str, ...] = (
    "door",
    "food_bowl",
    "water_bowl",
    "litter_box",
    "window",
    "sofa",
    "bed",
    "human",
    "other_cat",
    "toy",
    "carrier",
    "scratching_post",
)

#: 明确禁止出现在任何字段里的表述。
#:
#: 三类：情感推断、因果断言、健康结论。
#: **这是词表级的第二道防线** —— 第一道是 prompt，第二道是这里，
#: 第三道是人工复核。prompt 是请求，不是保证。
FORBIDDEN_INFERENCE_PHRASES: tuple[str, ...] = (
    # ── 情感 / 意图推断（模型不可靠，见模块 docstring） ──
    "焦虑",
    "害怕",
    "生气",
    "开心",
    "难过",
    "想出去",
    "想进来",
    "想吃饭",
    "饿了",
    "不舒服",
    # ── 因果断言（必须由主人提供） ──
    "因为",
    "所以它",
    "导致",
    "说明它",
    # ── 健康结论（属红旗模块，不属观察） ──
    "生病",
    "疼痛",
    "异常",
    "精神不振",
    "萎靡",
    "发烧",
)


class ModelObservation(BaseModel):
    """多模态模型对一段媒体的事实观察。"""

    media_url: str
    media_kind: MediaKind
    source_model: str = Field(
        description=(
            "产出这条观察的模型标识，如 `qwen3-vl-flash`。\n\n"
            "**必填**：模型会换、会升级。不记来源就无法回溯一个判断是怎么得出的，"
            "也无法在换模型后比较质量。"
        )
    )

    ok: bool = Field(
        default=True,
        description="本次观察是否可用。``False`` 表示失败——它**不参与**任何后续计算。",
    )
    error: str | None = None

    actions: list[BehaviorAction] = Field(
        default_factory=list,
        description="画面/音频里可观察到的**动作**，取自固定词表。",
    )
    scene_objects: list[str] = Field(
        default_factory=list,
        description="画面里出现的物体，取自 `SCENE_OBJECTS`。",
    )
    described_signs: list[str] = Field(
        default_factory=list,
        description=(
            "其他**可直接看到**的迹象（如「前爪抬起靠近门」）。\n\n"
            "不得包含情感推断、因果断言、健康结论 —— 由词表校验。"
        ),
    )

    ignored_terms: list[str] = Field(
        default_factory=list,
        description=(
            "模型给出但不在词表内、因而**未被采用**的值。\n\n"
            "为什么要留这一栏，而不是直接丢掉：**不让任何东西静默消失**。\n"
            "高频出现词表外的值，说明 prompt 或词表需要改 ——\n"
            "静默丢弃会让这个信号永远看不到。"
        ),
    )

    observed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description=(
            "观察发生的时间。\n\n"
            "⚠️ 它**不是**从模型输出推导的，而是由调用方（或默认的当前时间）给定 ——\n"
            "因此解析的确定性断言应当排除它。把它留在模型里是为了让每条观察\n"
            "可回溯到「什么时候看到的」。"
        ),
    )

    @model_validator(mode="after")
    def _no_inference_in_descriptions(self) -> ModelObservation:
        """**不允许推断性表述混进观察字段。**

        这是第二道防线。第一道是 prompt 里的「只描述看得见的」，
        但 **prompt 是请求，不是保证** —— 模型的「不要编造」和
        模型的「编造」来自同一组权重。
        """
        blob = " ".join(self.described_signs)
        hits = [p for p in FORBIDDEN_INFERENCE_PHRASES if p in blob]
        if hits:
            raise ValueError(
                f"观察描述里出现推断性表述：{'、'.join(hits)}。\n"
                "模型只能描述**看得见的**：情感推断不可靠、因果必须主人提供、"
                "健康结论属红旗模块。"
            )
        return self

    @model_validator(mode="after")
    def _scene_objects_are_from_vocabulary(self) -> ModelObservation:
        """场景物体必须来自固定词表 —— 自由文本无法用于后续匹配。"""
        bad = [o for o in self.scene_objects if o not in SCENE_OBJECTS]
        if bad:
            raise ValueError(
                f"scene_objects 含词表外的值：{'、'.join(bad)}。"
                f"允许：{'、'.join(SCENE_OBJECTS)}"
            )
        return self

    @model_validator(mode="after")
    def _failed_observation_carries_no_facts(self) -> ModelObservation:
        """失败时不得携带任何事实 —— 否则降级路径会静默产出内容。"""
        if not self.ok and (self.actions or self.scene_objects or self.described_signs):
            raise ValueError("ok=False 时不得携带 actions / scene_objects / described_signs")
        return self

    @property
    def is_usable(self) -> bool:
        return self.ok and bool(
            self.actions or self.scene_objects or self.described_signs
        )

    def evidence_statements(self) -> list[str]:
        """转成可展示的证据陈述。

        **措辞刻意保守**：不写「它想出去」，只写「画面里：猫面向门，前爪抬起」。
        用户读到的是**观察**，判断权还在他手上。
        """
        from app.schemas.behavior import action_label

        out: list[str] = []
        if self.actions:
            labels = "、".join(action_label(a) for a in self.actions)
            out.append(f"画面里观察到：{labels}")
        if self.scene_objects:
            out.append(f"画面里出现的物体：{'、'.join(self.scene_objects)}")
        out.extend(f"可见迹象：{s}" for s in self.described_signs)
        return out


def find_forbidden_inference(text: str) -> list[str]:
    """检查一段文本里是否含推断性表述。供 prompt 输出解析与测试使用。"""
    return [p for p in FORBIDDEN_INFERENCE_PHRASES if p in text]
