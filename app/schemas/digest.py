"""每日总结契约。

## 与单轮提取的区别不是「时机」，而是「视角」

| | 单轮提取（`memory_extractor`） | 每日总结 |
|---|---|---|
| 视角 | 这一句说了什么 | **这一整天发生了什么** |
| 能发现 | 显式陈述 | **聚合模式 / 跨轮矛盾 / 反复出现的关切** |
| 时长 | 每轮 | 每天一次 |
| 主要风险 | 漏掉隐含信息 | **LLM 把推测洗成事实** |

只有每日总结能发现的东西：

- 「今天它叫了 6 次」—— 需要聚合，单轮看不到
- 「今天你提到吸尘器 3 次」—— 跨轮次
- 「早上你说它精神好，晚上说它没精神」—— **跨轮矛盾**

## 本模块的核心是「拒绝」，不是「提取」

每条候选必须附 ``quote``（据称支撑它的原文片段），
由**代码**校验该片段确实出现在当天的对话里。校验不过的一律进
[`RejectedCandidate`][app.schemas.digest.RejectedCandidate]，**不写入长期记忆**。

> 最有价值的输出不是「提取到了什么」，而是「**拒绝了什么**」——
> 它让编造变得可见，而不是静默写进长期记忆。
>
> 这与 B2（静默丢弃）、B3（静默编造）是同一类防守：
> **编造最危险的地方是它不报错。**

**为什么不能靠 prompt 解决**：prompt 是请求，不是保证。
模型的「不要编造」和模型的「编造」来自同一组权重。
所以必须有一个**机械可验证**的门：引用片段在不在原文里，是可以算的。
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.memory import EventType, MemoryLayer, Polarity

#: 引用片段的最短长度。
#:
#: 为什么需要下界：1–3 个字的片段（「它」「今天」）几乎必然能在原文里找到，
#: 于是「引用校验」会退化成恒真——任何编造只要附一句常见短语就能通过。
QUOTE_MIN_CHARS = 4


class ExtractionSource(str, Enum):
    """这条候选是**主人说的**还是**系统归纳的**。

    它决定了落库状态，不是元数据：

    | 来源 | `MemorySource` | 能否直接 ACTIVE |
    |---|---|---|
    | `OWNER_RECORD` | `USER_OBSERVATION` | ✅ |
    | `AI_INFERENCE` | `SYSTEM_INFERENCE` | ❌ **只能 PENDING_CONFIRMATION** |

    注意这里**没有** `SYSTEM_MEASUREMENT`（四层信息分层里的第一层）：
    本功能读的是**文本对话**，拿不到声学/视觉测量。
    测量值由各自的提取器产生，不经过对话总结这条路径。
    """

    OWNER_RECORD = "owner_record"
    """主人直接说的事实，或他做过的事。"""

    AI_INFERENCE = "ai_inference"
    """需要推断才能得出的（如跨轮归纳）。**这类会被要求主人确认。**"""


class DigestMessage(BaseModel):
    """一条会话消息。"""

    role: Literal["user", "assistant"]
    content: str
    at: datetime | None = None


class DigestStats(BaseModel):
    """**由代码计算的确定性统计，不经过 LLM。**

    为什么必须由代码算：一旦让 LLM 去数「今天提到几次吸尘器」，
    这个数字就不可核查了。**计数是事实，事实由代码产生。**

    这与案例推理里「输出计数而非概率」是同一条原则：
    能被机械核对的东西，不要交给概率模型。
    """

    date: date_type
    message_count: int = Field(ge=0)
    user_message_count: int = Field(ge=0)
    assistant_message_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)

    topic_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "主题词当天的出现次数。**固定词表匹配，不是主题模型** —— "
            "它可复现、可审计，但**发现不了词表外的话题**。"
        ),
    )
    existing_memory_count: int = Field(
        default=0,
        ge=0,
        description="当天已由单轮提取写入的记忆数。用于判断「这天还值不值得总结」。",
    )

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> DigestStats:
        if self.user_message_count + self.assistant_message_count != self.message_count:
            raise ValueError(
                f"消息数不一致：user={self.user_message_count} + "
                f"assistant={self.assistant_message_count} != "
                f"message_count={self.message_count}"
            )
        return self

    @property
    def is_empty(self) -> bool:
        return self.message_count == 0


class DigestCandidate(BaseModel):
    """一条通过原文校验的候选记忆。"""

    layer: MemoryLayer = MemoryLayer.EPISODE
    """**默认 Episode，不直接写 Profile。**

    一条当天说的话不足以成为「稳定事实」。事实的晋升由飞轮的
    `is_eligible_for_promotion`（重复出现 + 时间跨度）负责，
    不由总结器自封。

    这与 docs/03 的「不声称超出证据的确定性」同源：
    今天说「它怕吸尘器」可能只是今天的事。
    """

    event_type: EventType
    subject: str = Field(description="归一化主语，如 'vacuum' / 'window'")
    content: str = Field(description="一条可独立检索的记忆")
    quote: str = Field(
        description=(
            "**据称支撑这条的原文片段。必须逐字出现在当天对话里。**\n\n"
            "为什么要求逐字而不允许改写：改写后的片段无法机械核验，"
            "校验就退化成「看起来像」——那等于没有校验。"
        )
    )
    source_layer: ExtractionSource
    confidence: float = Field(ge=0.0, le=1.0, default=0.6)
    polarity: Polarity = Polarity.NEUTRAL


class RejectedCandidate(BaseModel):
    """**被拒绝的候选。这个列表是本功能最重要的产物。**

    它让「LLM 编造」变成可见记录，而不是静默写进长期记忆。

    **不做静默过滤的理由**：若把编造候选直接丢掉，
    我们既不知道自己被骗了多少，也无法据此调 prompt。
    留下拒绝记录，这个比率就成了一个可监控的质量指标。
    """

    content: str = Field(description="被拒绝的内容。**保留原文以便人工复核。**")
    reason: str = Field(description="拒绝原因（机器可读的中文说明）")
    quote: str | None = None
    source_layer: ExtractionSource | None = None


class DailySummary(BaseModel):
    """一天的总结结果。"""

    date: date_type
    stats: DigestStats
    candidates: list[DigestCandidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)

    notes: list[str] = Field(
        default_factory=list,
        description="降级说明、跳过原因等。**任何未发生的事都要在这里可见。**",
    )

    @property
    def rejection_rate(self) -> float:
        """被拒绝的比例。

        **这是一个质量指标，不是错误指标。**
        它显著升高意味着 prompt 或模型需要调整——
        若它恒为 0，反而更值得怀疑：说明校验可能失效了。
        """
        total = len(self.candidates) + len(self.rejected)
        return len(self.rejected) / total if total else 0.0
