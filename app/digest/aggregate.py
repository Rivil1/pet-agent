"""确定性统计。**这一层完全不调用模型。**

## 为什么统计必须由代码算

「今天你提到吸尘器 3 次」是一个**可以与原文核对的事实**。
一旦让 LLM 去数，这个数字就变成了模型的输出——不可核查、不可复现、
同一天跑两次可能得到两个答案。

这与案例推理里「输出计数而非概率」是同一条原则：

> **能被机械核对的东西，不要交给概率模型。**

## 主题词表的诚实边界

``TOPIC_VOCABULARY`` 是**固定词表匹配**，不是主题模型。因此：

- ✅ 可复现（同一输入 → 同一统计）
- ✅ 可审计（词表就在下面，谁都能看）
- ❌ **发现不了词表外的话题**

这是刻意的取舍。真实的主题发现需要聚类或 LLM，两者都不可机械核对；
而这里的统计是要**展示给用户并对用户负责**的，所以宁可窄而准。
"""

from __future__ import annotations

from datetime import date as date_type

from app.schemas.digest import DigestMessage, DigestStats

#: 主题词表：归一化的 subject → 该 subject 的同义词/说法。
#:
#: **只收与宠物照料相关的、可能反复出现的概念。**
#: 不收情绪词、语气词、寒暄词——它们会淹没真正的信号。
TOPIC_VOCABULARY: dict[str, tuple[str, ...]] = {
    "vacuum": ("吸尘器", "吸尘", "扫地机", "扫地机器人"),
    "food": ("猫粮", "罐头", "吃饭", "喂", "食盆", "饿", "吃"),
    "water": ("喝水", "水碗", "饮水机"),
    "litter": ("猫砂", "猫砂盆", "上厕所", "拉屎", "尿"),
    "vomit": ("吐", "呕吐", "反胃"),
    "door": ("门口", "门外", "抓门", "开门"),
    "window": ("窗台", "窗边", "窗户"),
    "brush": ("梳毛", "梳子", "掉毛"),
    "play": ("逗猫棒", "玩具", "玩"),
    "sleep": ("睡", "打呼", "窝"),
    "night": ("半夜", "凌晨", "夜里", "晚上"),
    "vet": ("医院", "看医生", "兽医", "疫苗", "驱虫"),
    "med": ("吃药", "喂药", "药"),
    "weight": ("体重", "胖", "瘦", "称重"),
    "scratch": ("抓沙发", "抓家具", "猫抓板"),
    "litter_box_avoid": ("乱尿", "不在猫砂盆"),
}

#: 统计主题时只累加这些角色的消息。
#:
#: **只用 user 消息**：助手说的话是从主人的话里生成的，
#: 一起统计会让每个主题的计数凭空翻倍——那是自说自话，不是观察。
_TOPIC_ROLES = frozenset({"user"})


def _count_mentions(text: str, keywords: tuple[str, ...]) -> int:
    """统计关键词出现次数，**不重复计数重叠的同义词**。

    为什么不能逐词 ``text.count(kw)``：「吸尘器」包含「吸尘」，
    于是「它怕吸尘器」会被算成 2 次。
    **而计数是要展示给用户并据以判断的事实，重复计数就是假数字。**

    做法：从左到右贪心取**最长匹配**，匹配到就跳过该长度。
    """
    ordered = sorted(keywords, key=len, reverse=True)
    count = 0
    i = 0
    while i < len(text):
        for kw in ordered:
            if text.startswith(kw, i):
                count += 1
                i += len(kw)
                break
        else:
            i += 1
    return count


def aggregate(
    *,
    day: date_type,
    messages: list[DigestMessage],
    existing_memory_count: int = 0,
) -> DigestStats:
    """计算当天的确定性统计。

    Args:
        day: 本地日。
        messages: 当天的会话消息（顺序无关）。
        existing_memory_count: 当天已由单轮提取写入的记忆数。

    Returns:
        统计结果。空输入也返回合法对象（全 0），**不抛异常** ——
        「今天没有对话」是正常状态，不是错误。
    """
    user_msgs = [m for m in messages if m.role in _TOPIC_ROLES]
    assistant_msgs = [m for m in messages if m.role == "assistant"]

    topic_counts: dict[str, int] = {}
    for msg in user_msgs:
        for subject, keywords in TOPIC_VOCABULARY.items():
            hits = _count_mentions(msg.content, keywords)
            if hits:
                topic_counts[subject] = topic_counts.get(subject, 0) + hits

    return DigestStats(
        date=day,
        message_count=len(messages),
        user_message_count=len(user_msgs),
        assistant_message_count=len(assistant_msgs),
        total_chars=sum(len(m.content) for m in messages),
        # 只保留命中过的，且按次数降序 —— 保证输出稳定可复现
        topic_counts=dict(sorted(topic_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        existing_memory_count=existing_memory_count,
    )


def render_transcript(messages: list[DigestMessage]) -> str:
    """把消息拼成给模型看的转写文本。

    角色标注用中文而非 ``user``/``assistant``：模型在中文任务下
    对「主人」「你」的指代理解更稳，减少把助手的话当成主人的事实。
    """
    lines: list[str] = []
    for msg in messages:
        speaker = "主人" if msg.role == "user" else "你"
        lines.append(f"{speaker}：{msg.content}")
    return "\n".join(lines)
