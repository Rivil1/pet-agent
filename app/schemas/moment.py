"""瞬间（Moment）：日记的最小单位。

## 为什么它是 P0（`docs/11` §2.1）

> **当前设计的 `RECORD_EVENT` 门槛太高** —— 它要求用户「说一件事」。
> S1 需要的是**更低门槛的动作**。

关于场景 S1（想猫时）的关键判断是：

> **用户想猫时是情绪状态**，不想打字、不想被问问题。
> 每多一个输入字段，转化率就掉一截。

所以这一层的设计原则是**反工具化**的：

| 不做 | 为什么 |
|---|---|
| 不强制填描述 | 想猫的时候不想打字 |
| 不要求选分类 | 分类由系统抽，不该问用户 |
| 不弹确认框 | 「记录」应当是一次点击就完成的事 |

这三条与「档案必须经用户确认」（`DESIGN §2.3`）**不矛盾**：
档案是**身份锚点**，写错了会污染后续所有校验，所以要确认；
瞬间是**流水记录**，写错了再记一条就是了，纠错成本远低于确认成本。

## 场景标签怎么来（以及它的局限）

`docs/11` 说「系统自己抽取」。本实现用**关键词**抽取，不用模型：

- 确定、可复现、不花钱
- 用户写了「在窗台晒太阳」就抽得出；没写就只能归「其他」

**没写就不猜** —— 模型从照片猜场景需要额外验证（那是 §13 FGS 那一类
「通用 VLM 零样本做专项判断未经文献验证」的问题），
而一个猜错的场景标签会让时间线看起来比实际更可信。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator

__all__ = ["MomentScene", "Moment", "extract_scene", "SCENE_DISPLAY"]


class MomentScene(str, Enum):
    """场景标签。**固定词表** —— 自由文本无法稳定聚合。

    词表来自 `docs/11` §2.1（睡觉 / 玩耍 / 进食 / 窗台 / 与人互动 / 其他）。
    """

    SLEEPING = "sleeping"
    PLAYING = "playing"
    EATING = "eating"
    WINDOW = "window"
    WITH_HUMAN = "with_human"
    GROOMING = "grooming"
    OTHER = "other"


SCENE_DISPLAY: dict[MomentScene, str] = {
    MomentScene.SLEEPING: "睡觉",
    MomentScene.PLAYING: "玩耍",
    MomentScene.EATING: "进食",
    MomentScene.WINDOW: "窗台",
    MomentScene.WITH_HUMAN: "和人待着",
    MomentScene.GROOMING: "舔毛",
    MomentScene.OTHER: "其他",
}

#: 关键词 → 场景。**顺序即优先级**（更具体的在前）。
#:
#: 为什么用关键词而不是模型：见模块 docstring。这里要的是
#: 「用户写了就能抽出来，没写就不猜」，而模型做不到「不猜」。
_SCENE_HINTS: tuple[tuple[MomentScene, tuple[str, ...]], ...] = (
    (MomentScene.EATING, ("吃", "饭", "粮", "罐头", "喝", "舔碗", "进食")),
    (MomentScene.SLEEPING, ("睡", "打盹", "眯", "窝着", "瘫", "呼呼")),
    (MomentScene.PLAYING, ("玩", "逗猫棒", "追", "扑", "球", "疯跑", "抓")),
    (MomentScene.WINDOW, ("窗台", "窗边", "窗户", "看外面", "晒太阳")),
    (MomentScene.GROOMING, ("舔毛", "理毛", "洗脸", "梳毛")),
    (MomentScene.WITH_HUMAN, ("蹭", "抱", "腿上", "身边", "陪我", "和我", "怀里")),
)


def extract_scene(text: str | None) -> MomentScene:
    """从可选的一句话里抽场景标签。

    **没写就不猜** —— 返回 `OTHER`，而不是从别的线索硬推一个。
    """
    if not text:
        return MomentScene.OTHER
    body = text.strip()
    if not body:
        return MomentScene.OTHER
    for scene, keywords in _SCENE_HINTS:
        if any(kw in body for kw in keywords):
            return scene
    return MomentScene.OTHER


class Moment(BaseModel):
    """一条瞬间记录。

    ## 与 `MemoryEvent` 的分工

    | | `MemoryEvent` | `Moment` |
    |---|---|---|
    | 是什么 | **可检索的事实**（「它怕吸尘器」） | **一次流水**（「今天 15:20 它在窗台」） |
    | 进检索吗 | 是（向量 + 重排） | **否** —— 它按时间线走 |
    | 谁产生 | 抽取 + 准入判定 | 用户一次点击 |

    分开是刻意的：把每张照片都塞进记忆检索，会让「它怕吸尘器」这类
    稳定事实被日常流水淹没 —— 而那正是记忆层要防的事。
    """

    moment_id: str | None = None
    user_id: str
    pet_id: str
    session_id: str | None = None

    #: 照片 / 短视频地址。**当前必填** —— 没有媒体的「瞬间」就是记忆，不是瞬间。
    media_url: str = Field(min_length=1)

    #: 用户**可选**写的一句话。为空是正常的，不是缺失。
    note: str | None = None

    #: 系统抽取的场景标签（**没写就不猜**，见 `extract_scene`）
    scene: MomentScene = MomentScene.OTHER

    #: 记录时刻。默认「现在」—— 瞬间的意义就在于「刚刚发生」。
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("note")
    @classmethod
    def _blank_note_is_none(cls, value: str | None) -> str | None:
        """空白字符串归一成 `None`。

        不归一的话 `""` 与 `None` 会变成两种「没写」，
        而时间线渲染、聚合、导出都要各判一次。
        """
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @property
    def scene_display(self) -> str:
        return SCENE_DISPLAY.get(self.scene, self.scene.value)
