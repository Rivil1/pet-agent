"""对话分轨：**同一个事实标准，两种表达策略**。

## 为什么必须分轨（`docs/11-companion-design.md` §0.3）

文档的诊断是：

> **这是两个产品被焊在一起**
>
> ```
> A. 高可信度行为/健康分析工具   → 现有设计非常契合 ✅
> B. 情感陪伴产品                 → 几乎没覆盖 ❌
> ```
>
> 用 A 的严谨度要求 B，结果 B 的体验被 A 的约束杀死。

而根源**不是诚实性做多了**，是**把两类需求用同一套严谨度处理了**：

| 用户在做的事 | 真实需求 | 应有严谨度 |
|---|---|---|
| 「团团今天怎么不理我」 | **求安慰** | 温度优先 |
| 「它刚才为什么一直叫」 | **求答案** | 证据优先 |

把第二种的严谨度套到第一种上，就会得到「暂时没有相关记录哦」——
**在陪伴语境下是冷场的**。

## 关键：两轨都诚实，区别在表达策略

> 情绪轨不是「可以编」，而是「**只讲已知的正面事实，且不主动补免责声明**」。

所以两个模式共用同一条底线（不得编造事实、扮演必须标注），
差别只在**要不要主动暴露不确定性**：

| 检查项 | COMPANION | ANALYSIS |
|---|---|---|
| 禁止编造事实 | ✅ | ✅ |
| 过度确定性 | 放宽（不说概率是刻意的） | ✅ 严格 |
| 必须给 `limitations` | ❌ 不要求 | ✅ 要求 |
| 必须给替代解释 | ❌ 不要求 | ✅ 要求 |

## 判定用规则，不用 LLM 自报置信度

§5.2 明确要求规则特征优先 —— 这恰好规避了「LLM 置信度未校准」
（`docs/10-self-review.md` D1）。规则表可枚举、可测试、可复现。

**默认 COMPANION，倾向 ANALYSIS 需要明确信号。**

理由（文档原话）：陪伴产品里
**误判为分析轨的代价（冷场、说教）高于误判为情绪轨**（少给一次证据，用户会追问）。
"""

from __future__ import annotations

from enum import Enum

from app.schemas import AudioKind, RawInput

__all__ = [
    "InteractionMode",
    "decide_mode",
    "ANALYSIS_CUES",
    "COMPANION_CUES",
]


class InteractionMode(str, Enum):
    """交互模式。"""

    #: 情绪轨：温度优先。只讲已知事实，温暖措辞，**不主动**暴露不确定性
    COMPANION = "companion"
    #: 分析轨：证据优先。完整呈现证据链、置信度与局限
    ANALYSIS = "analysis"


#: 疑问词 → 分析轨。
#:
#: 这些词表明用户在**要一个答案**，而不是在倾诉。
#: 「为什么」「是不是」问的是因果与判断，那些必须有证据支撑才敢说。
ANALYSIS_CUES: tuple[str, ...] = (
    "为什么",
    "为啥",
    "怎么回事",
    "怎么会",
    "是不是",
    "会不会",
    "是不是有",
    "什么原因",
    "正常吗",
    "有问题吗",
    "严重吗",
    "要不要",
    "该不该",
    "怎么办",
    "如何",
    "多少",
    "几天",
    "几次",
)

#: 情绪词 → 陪伴轨。
#:
#: 这些词表明用户在**表达感受**。对感受的回应是「接住它」，
#: 而不是给一个带置信度的判断 —— 后者在陪伴语境里是冷场的。
COMPANION_CUES: tuple[str, ...] = (
    "想你",
    "想它",
    "难过",
    "担心",
    "好想",
    "舍不得",
    "心疼",
    "好可爱",
    "太可爱",
    "喜欢",
    "爱你",
    "好久没",
    "好久不见",
    "有点想",
    "今天好",
    "心情",
    "陪着",
    "陪我",
    "孤单",
    "寂寞",
    "不开心",
    "烦",
    "累",
)

#: 纯问候 → 陪伴轨。没有明确诉求时不该摆出分析架势。
GREETING_CUES: tuple[str, ...] = (
    "你好",
    "嗨",
    "在吗",
    "早",
    "晚安",
    "哈喽",
    "hi",
    "hello",
)


def decide_mode(
    *,
    raw: RawInput,
    text: str,
    recent_turn_count: int = 0,
) -> tuple[InteractionMode, str]:
    """判定交互模式。**纯函数。**

    Args:
        raw: 原始输入（用于识别音频/图片）。
        text: 文本内容。
        recent_turn_count: 本会话已有的轮数。**短会话高频 → 陪伴轨** ——
            刚打开就聊几句日常，那不是在查资料。

    Returns:
        `(模式, 原因)`。**原因必须返回** —— 冷启动阶段最需要的就是
        「为什么这次是分析轨」，否则用户会觉得系统突然开始说教。

    ## 判定顺序（**先分析后陪伴**）

    分析信号优先检查，因为它的线索更具体（明确的疑问词），
    而情绪词可能出现在一个求证句里（「它这样叫是不是想我了」——
    有「想」但也有「是不是」，这是**求答案**）。

    反过来先查情绪词的话，上面那句会被判成陪伴轨，
    而用户实际在问一个可以给证据的问题。
    """
    # ── 1) 猫叫音频：强信号，直接分析轨 ──
    if raw.audio_url and raw.audio_kind is AudioKind.CAT_MEOW:
        return (
            InteractionMode.ANALYSIS,
            "上传了猫叫音频 —— 声学解释是一条证据链，需要完整呈现",
        )

    body = (text or "").strip()
    if not body:
        # 无文本（例如纯图片）→ 陪伴轨。没有诉求时不该摆分析架势。
        return InteractionMode.COMPANION, "没有文字诉求，按陪伴处理"

    # ── 2) 明确疑问 → 分析轨 ──
    hit = [c for c in ANALYSIS_CUES if c in body]
    if hit:
        return (
            InteractionMode.ANALYSIS,
            f"出现疑问词 {hit[:2]} —— 用户在要一个可给证据的答案",
        )

    # ── 3) 情绪表达 → 陪伴轨 ──
    hit = [c for c in COMPANION_CUES if c in body]
    if hit:
        return (
            InteractionMode.COMPANION,
            f"出现情绪表达 {hit[:2]} —— 回应感受，不是给判断",
        )

    # ── 4) 纯问候 / 短会话高频 → 陪伴轨 ──
    if any(c in body.lower() for c in GREETING_CUES):
        return InteractionMode.COMPANION, "问候语，按陪伴处理"

    if recent_turn_count > 0 and recent_turn_count < 4 and len(body) <= 12:
        return (
            InteractionMode.COMPANION,
            f"会话刚开始（{recent_turn_count} 轮）且话很短，按陪伴处理",
        )

    # ── 5) 默认陪伴轨 ──
    # **默认值的选择是刻意的**：误判为分析轨的代价（冷场、说教）
    # 高于误判为情绪轨（少给一次证据，用户会追问）。
    return InteractionMode.COMPANION, "没有分析信号，默认陪伴轨"


def guard_policy(mode: InteractionMode) -> dict[str, bool]:
    """该模式下守卫要开哪些检查（§5.3 的表）。**集中在这里，不散落。**"""
    if mode is InteractionMode.ANALYSIS:
        return {
            "forbid_fabrication": True,
            "strict_certainty": True,
            "require_limitations": True,
            "require_alternatives": True,
            "require_roleplay_notice": True,
        }
    return {
        "forbid_fabrication": True,  # **两轨共用同一条底线**
        "strict_certainty": False,
        "require_limitations": False,
        "require_alternatives": False,
        "require_roleplay_notice": True,
    }


# ─────────────────────────────────────────────────────────────
# 对外导出
# ─────────────────────────────────────────────────────────────

from app.companion.prompts import pet_prompt  # noqa: E402

__all__ = [
    "InteractionMode",
    "decide_mode",
    "guard_policy",
    "ANALYSIS_CUES",
    "COMPANION_CUES",
    "GREETING_CUES",
    "pet_prompt",
]
