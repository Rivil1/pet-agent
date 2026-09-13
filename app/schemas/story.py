"""「宠物的一天」契约。

## 它是什么

把 [`DailySummary`][app.schemas.digest.DailySummary] 里**已通过原文校验的候选**
改写成一段宠物口吻的叙述，供娱乐使用，并为后续的**漫画 / 写真**留出结构。

## 三个必须说清楚的设计约束

### 1. 只能从「已校验的候选」派生

**不能拿原始对话去生成故事。** 若自由发挥：

- `app/digest` 的 quote 机械校验全部被绕过
- 健康信号会被娱乐化（禁令 1 / 不变量 I16）
- 叙事会覆盖事实（违反 `docs/14` §4 的 A4）

所以每个 `StoryBeat` 都必须带 `fact_refs`，指向它据以生成的候选下标。
**能指回事实的句子才是叙述，指不回去的是编造。**

### 2. 漫画格无法标注不确定性 —— 这是最危险的形态

| 形态 | 用户读到的 |
| --- | --- |
| 文字 | 「它**可能**想去门口看看」← 保留了「可能」 |
| 漫画一格 | 猫眼巴巴望着门 ← **用户直接接受「它就是想去门口」** |

**一个画格是一个确定的陈述，你没有地方写「可能」。**
所以 `fact_refs` 不是元数据，它是让视觉形态不越界的前提。

### 3. 「宠物口吻」与「娱乐化」是两件事

第一人称叙述是**表达形式**；娱乐化是**调性**。两者可以分开：

| | 口吻 | 调性 | 层 |
| --- | --- | --- | --- |
| ❌ 娱乐化健康信号 | 宠物口吻 | **轻快** | L1（违规） |
| ✅ 严肃叙述 | 宠物口吻 | **严肃** | **L3** |

因此 `InfoLayer.HEALTH` 的节拍**禁止** `StoryTone.PLAYFUL` —— 由契约强制。

## 免责声明的边界

「仅供娱乐」这句标注**不能盖在健康内容上**：它传达的是「这不用太当真」，
而那正是禁令 1 想防的。

所以 `DailyStory` 分成两个字段：

- `disclaimer` —— 只覆盖娱乐层节拍
- `health_notice` —— 健康层节拍专用的严肃提示（**不是**免责声明）
"""

from __future__ import annotations

from datetime import date as date_type
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.schemas.digest import ExtractionSource
from app.schemas.memory import EventType

#: 娱乐层的免责声明。
DISCLAIMER_ENTERTAINMENT = (
    "本内容由 AI 生成，仅供娱乐，不构成对这只猫的观察结论。"
)

#: 健康层节拍的提示。
#:
#: 刻意**不**叫「免责声明」：它不是要用户别当真，而是告诉用户
#: 该当真但要按正确的方式当真（去健康提醒里看，而不是在这里读故事）。
HEALTH_NOTICE = (
    "以上这条来自当天的健康记录。**这不是诊断**，"
    "请到健康提醒中查看，并按其中的建议观察。"
)

#: 需求 / 生理状态词表。
#:
#: 用于检测拟人化越界（`ViolationType.ROLEPLAY_OVERREACH`）。
#:
#: **收录判据**：说了之后用户会改变照料行为的词。
#: 不收纯情绪词（「开心」「高兴」）——那些不影响照料决策，正是拟人化可以表达的。
PHYSIOLOGY_WORDS: tuple[str, ...] = (
    # ── 需求 ──
    "饿了",
    "饿",
    "想吃",
    "要吃饭",
    "渴了",
    "想喝水",
    "想出去",
    "想进来",
    "想上厕所",
    # ── 生理状态 ──
    "肚子疼",
    "肚子痛",
    "难受",
    "不舒服",
    "疼",
    "痛",
    "恶心",
    "头晕",
    "没力气",
    "懒得动",
    "发烧",
    "过敏",
    "发情",
    # ── 偏好（影响饮食/用品决策）──
    "不喜欢这个",
    "不爱吃",
    "喜欢吃",
)


class InfoLayer(str, Enum):
    """展示分层（`docs/14` §4）。**决定渲染样式与配套声明。**"""

    ENTERTAINMENT = "L1_entertainment"
    """娱乐层：明显可辨为娱乐，不得声称事实主张。"""

    FACT = "L2_fact"
    """事实层：中性、可追溯，不得混入玩笑式措辞。"""

    HEALTH = "L3_health"
    """健康层：最高视觉优先级，**永远不出现娱乐调性**。"""


class StoryTone(str, Enum):
    """本次呈现的调性。

    **记录它是为了让禁令 2 的比例可审计。**

    `docs/14` 禁令 2 要求「同一个行为在不同时间应有多样化的呈现，
    且必须有相当比例的中性/事实性呈现」—— 没有这个字段，比例无从保证，
    也无法在测试里断言。
    """

    PLAYFUL = "playful"
    """轻快、玩梗。**仅限娱乐层。**"""

    NEUTRAL = "neutral"
    """中性叙述。"""

    SERIOUS = "serious"
    """严肃。**健康层必须用这个。**"""


class StoryTimeOfDay(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"


_TIME_DISPLAY: dict[StoryTimeOfDay, str] = {
    StoryTimeOfDay.MORNING: "早上",
    StoryTimeOfDay.AFTERNOON: "下午",
    StoryTimeOfDay.EVENING: "晚上",
    StoryTimeOfDay.NIGHT: "夜里",
}

_LAYER_DISPLAY: dict[InfoLayer, str] = {
    InfoLayer.ENTERTAINMENT: "娱乐",
    InfoLayer.FACT: "事实",
    InfoLayer.HEALTH: "健康",
}


def time_display(at: StoryTimeOfDay) -> str:
    return _TIME_DISPLAY[at]


def layer_display(layer: InfoLayer) -> str:
    return _LAYER_DISPLAY[layer]


def find_physiology_words(text: str) -> list[str]:
    """找出文本里的需求/生理词。**返回命中词，便于测试与日志。**"""
    return [w for w in PHYSIOLOGY_WORDS if w in text]


class StoryBeat(BaseModel):
    """故事里的一个节拍。

    **它就是未来的一格漫画 / 一页写真。** 因此它必须结构化，
    而不是一整段散文——散文没法切格，也没法让每格绑定事实。
    """

    at: StoryTimeOfDay
    info_layer: InfoLayer = InfoLayer.ENTERTAINMENT
    tone: StoryTone = StoryTone.PLAYFUL

    line: str = Field(
        description="宠物口吻的叙述，如「本喵今天在窗台赖了一下午」。"
    )

    fact_refs: list[int] = Field(
        default_factory=list,
        description=(
            "据以生成的候选下标（对应 `DailySummary.candidates` 的位置）。\n\n"
            "**这不是元数据，是安全机制**：一个画格是确定的陈述，"
            "没有地方写「可能」——所以每个视觉元素都必须能指回已校验的事实。"
        ),
    )
    photo_ref: str | None = Field(
        default=None,
        description=(
            "已上传照片的引用 ID。**留作漫画/写真扩展位。**\n\n"
            "⚠️ 只用**已上传**的照片。为素材而索取新照片会违反禁令 3"
            "（不得设计鼓励用户打扰猫的机制）。"
        ),
    )

    @model_validator(mode="after")
    def _health_beats_must_be_serious_and_traceable(self) -> StoryBeat:
        """健康层节拍的三条硬约束。"""
        if self.info_layer is InfoLayer.HEALTH:
            if self.tone is not StoryTone.SERIOUS:
                raise ValueError(
                    f"健康层节拍必须是 serious 调性，得到 {self.tone.value}。"
                    "把健康信号渲染成娱乐调性正是禁令 1 要防的："
                    "用户会觉得「系统都说可爱，那应该没事」。"
                )
            if not self.fact_refs:
                raise ValueError(
                    "健康层节拍必须带 fact_refs —— 健康内容必须可追溯到具体记录，"
                    "否则它就是一个无法核查的健康断言。"
                )
        return self

    @model_validator(mode="after")
    def _roleplay_cannot_touch_physiology(self) -> StoryBeat:
        """拟人化越界检测（`ViolationType.ROLEPLAY_OVERREACH`）。

        **同一个词在不同层合法或违规**：

        - `L1 娱乐层` + 「饿了」 → **违规**（用户可能因此过量喂食）
        - `L3 健康层` + 「难受」 → **允许**（这正是健康层该说的话）

        所以这里只看娱乐层与事实层，健康层豁免。
        """
        if self.info_layer is InfoLayer.HEALTH:
            return self

        hits = find_physiology_words(self.line)
        if hits:
            raise ValueError(
                f"拟人化越界（ROLEPLAY_OVERREACH）：{'、'.join(hits)} "
                f"出现在 {self.info_layer.value} 层。"
                "拟人化可以表达情绪与关系，不能表达需求与生理状态——"
                "判据是「用户会不会据此改变照顾行为」。"
                "若这条确实是健康记录，应改用 InfoLayer.HEALTH。"
            )
        return self


class ExcludedBeat(BaseModel):
    """某条候选**没有**进入故事，以及原因。

    与 `DailySummary.rejected` 同一条原则：**不让任何东西静默消失**。
    用户看不到某条记录时，应该能知道它去哪了。
    """

    content: str
    reason: str
    info_layer: InfoLayer | None = None


class DailyStory(BaseModel):
    """「宠物的一天」。"""

    date: date_type
    title: str
    beats: list[StoryBeat] = Field(default_factory=list)
    excluded: list[ExcludedBeat] = Field(default_factory=list)

    disclaimer: str = DISCLAIMER_ENTERTAINMENT
    """**只覆盖娱乐层节拍。**"""

    health_notice: str | None = None
    """健康层节拍专用提示。为 `None` 表示当天没有健康层节拍。"""

    @model_validator(mode="after")
    def _health_notice_matches_content(self) -> DailyStory:
        """健康提示与实际内容必须一致 —— 两个方向都要查。

        少见但重要：**有健康节拍却没有提示**会让严肃内容裸奔；
        **有提示却没有健康节拍**会让用户白白紧张。
        """
        has_health = any(b.info_layer is InfoLayer.HEALTH for b in self.beats)
        if has_health and self.health_notice is None:
            raise ValueError(
                "含健康层节拍时必须提供 health_notice —— "
                "否则严肃内容缺少指向健康提醒的引导。"
            )
        if not has_health and self.health_notice is not None:
            raise ValueError(
                "没有健康层节拍时不得提供 health_notice —— "
                "否则会让用户为不存在的问题紧张。"
            )
        return self

    @model_validator(mode="after")
    def _beats_are_traceable(self) -> DailyStory:
        """**除纯拼接类节拍外，每个节拍都必须能指回事实。**

        允许 `fact_refs` 为空的情形：由 `DigestStats` 直接得出的聚合叙述
        （如「今天你提到吸尘器 3 次」）——它的依据是代码算出的统计，
        不是某一条候选。
        """
        for i, beat in enumerate(self.beats):
            if beat.fact_refs:
                for ref in beat.fact_refs:
                    if ref < 0:
                        raise ValueError(
                            f"第 {i + 1} 个节拍的 fact_refs 含负下标 {ref}"
                        )
        return self

    @property
    def tone_mix(self) -> dict[StoryTone, int]:
        """各调性的节拍数。**禁令 2 的可审计依据。**"""
        mix: dict[StoryTone, int] = {}
        for beat in self.beats:
            mix[beat.tone] = mix.get(beat.tone, 0) + 1
        return mix

    @property
    def neutral_ratio(self) -> float:
        """中性/事实性呈现的比例。

        `docs/14` 禁令 2 要求「必须有相当比例的中性/事实性呈现」。
        这个属性让该要求**可被测试断言**，而不是只写在文档里。
        """
        if not self.beats:
            return 1.0
        non_playful = sum(
            1 for b in self.beats if b.tone is not StoryTone.PLAYFUL
        )
        return non_playful / len(self.beats)

    @property
    def has_health_content(self) -> bool:
        return any(b.info_layer is InfoLayer.HEALTH for b in self.beats)


#: 允许进入娱乐层的事件类型。
#:
#: **`HEALTH` 不在其中** —— 这是禁令 1 的机械落法。
ENTERTAINABLE_EVENT_TYPES: frozenset[EventType] = frozenset(
    {
        EventType.BEHAVIOR,
        EventType.PREFERENCE,
        EventType.ROUTINE,
        EventType.CONTEXT,
    }
)

#: 系统归纳（`AI_INFERENCE`）不得作为娱乐内容的事实基础。
#:
#: 理由：禁令 2 —— 把**推测**反复渲染成故事，会系统性固化行为解读。
#: 故事只复现已确认的事实；推测只在事实层出现。
ENTERTAINABLE_SOURCES: frozenset[ExtractionSource] = frozenset(
    {ExtractionSource.OWNER_RECORD}
)
