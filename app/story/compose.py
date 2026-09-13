"""把每日总结改写成「宠物的一天」。

## 顺序很重要：**先分类，再叙述**

```
DailySummary.candidates
   │
   ├─ ① 分类（代码，不是 LLM）
   │     ├─ event_type == HEALTH     → L3 健康层
   │     ├─ source == AI_INFERENCE   → 不进娱乐层（禁令 2）
   │     └─ 其余                     → L1 娱乐层
   │
   ├─ ② 渲染（模板，不调模型）
   │     └─ 每个节拍绑定 fact_refs
   │
   └─ ③ 组装声明（按层分开）
         ├─ disclaimer    ← 只盖 L1
         └─ health_notice ← 只盖 L3
```

**分类必须在渲染之前，而且必须由代码做。**

如果把「这条算不算健康内容」交给 LLM 判断，就会退化成
「模型今天心情好就不当回事」——而健康信号的漏判是不可逆的。

## 为什么不让 LLM 写故事

1. **无依据的编造**：故事越生动，越容易加进原文没有的细节
2. **无法追溯**：LLM 写出的句子没法机械映射回 `fact_refs`
3. **可复现性**：同一份总结应生成同一个故事（`DESIGN.md` §6.5）

因此本期用**模板渲染**：确定性、可离线、每个节拍的依据都可追。
将来若要接 LLM 润色，**必须保持 `fact_refs` 不变**——
润色可以改措辞，不能改指向。
"""

from __future__ import annotations

from datetime import date as date_type
from datetime import datetime

from app.schemas.digest import DailySummary, DigestCandidate, DigestMessage
from app.schemas.memory import EventType
from app.schemas.timeofday import bucket_of
from app.schemas.story import (
    ENTERTAINABLE_EVENT_TYPES,
    ENTERTAINABLE_SOURCES,
    DailyStory,
    ExcludedBeat,
    InfoLayer,
    StoryBeat,
    StoryTimeOfDay,
    StoryTone,
    find_physiology_words,
)

def _time_of_day(occurred_at: datetime | None) -> StoryTimeOfDay:
    """时段划分。边界定义在 `app/schemas/timeofday.py`（全项目唯一）。

    无时间戳时归入 `EVENING` —— 那是**故事分段**的取舍：
    它要的是一天的叙事分段，缺一段就不完整，而晚上是一天的中位且不猜具体时段。

    （习惯统计的取舍不同：它**排除**时间未知的条目，
    因为把未知塞进某个桶等于往统计里掺假数据。见 `app/habits/detect.py`。）
    """
    if occurred_at is None:
        return StoryTimeOfDay.EVENING
    return bucket_of(occurred_at)


def _classify(
    candidate: DigestCandidate,
) -> tuple[InfoLayer | None, str]:
    """决定一条候选进哪一层。

    Returns:
        ``(层, 原因)``。层为 ``None`` 时表示不进故事，原因为排除理由。

    **这是本模块最重要的函数。** 分类错了，后面所有守卫都白费。
    """
    # ── 健康信号：强制进 L3，且不允许被排除 ──
    if candidate.event_type is EventType.HEALTH:
        return InfoLayer.HEALTH, "健康记录，进 L3 并附健康提示"

    # ── 系统归纳不进娱乐层（禁令 2）──
    if candidate.source_layer not in ENTERTAINABLE_SOURCES:
        return None, (
            "系统归纳的内容不进娱乐层。把推测反复渲染成故事会"
            "系统性固化行为解读（docs/14 禁令 2）——它应在事实层呈现。"
        )

    # ── 事件类型白名单 ──
    if candidate.event_type not in ENTERTAINABLE_EVENT_TYPES:
        return None, f"事件类型 {candidate.event_type.value} 不进入娱乐层"

    return InfoLayer.ENTERTAINMENT, "可娱乐化"


def _render_line(candidate: DigestCandidate, layer: InfoLayer) -> str:
    """把一条候选改写成宠物口吻。

    **模板，不调模型。** 理由见模块 docstring 的三条。

    注意健康层用**直述**而不是拟人化玩梗：它要给的是可核查的硥状，
    不是「本喵好可怜」。宠物口吻在这里只是人称，不是调性。
    """
    if layer is InfoLayer.HEALTH:
        return f"今天我有点不对劲：{candidate.content}"
    return candidate.content


def compose_story(
    summary: DailySummary,
    *,
    messages: list[DigestMessage] | None = None,
) -> DailyStory:
    """把每日总结组装成「宠物的一天」。

    Args:
        summary: 已通过 quote 校验的每日总结。
        messages: 当天的消息，仅用于取时间戳切分早/午/晚/夜。

    Returns:
        `DailyStory`。**同一份 summary 必然生成同一个 story**（可复现）。
    """
    stamps = messages or []

    beats: list[StoryBeat] = []
    excluded: list[ExcludedBeat] = []

    for idx, candidate in enumerate(summary.candidates):
        layer, reason = _classify(candidate)

        if layer is None:
            excluded.append(
                ExcludedBeat(content=candidate.content, reason=reason)
            )
            continue

        line = _render_line(candidate, layer)
        beat = _make_beat(
            candidate=candidate,
            layer=layer,
            line=line,
            index=idx,
            at=_time_of_day(_stamp_for(candidate, stamps)),
        )
        if beat is None:
            excluded.append(
                ExcludedBeat(
                    content=candidate.content,
                    reason="渲染后触发守卫，已排除",
                    info_layer=layer,
                )
            )
            continue
        beats.append(beat)

    beats.sort(key=lambda b: list(StoryTimeOfDay).index(b.at))

    return DailyStory(
        date=summary.date,
        title=f"{summary.date.month} 月 {summary.date.day} 日 · 它的一天",
        beats=beats,
        excluded=excluded,
        health_notice=(
            "以上这条来自当天的健康记录。**这不是诊断**，"
            "请到健康提醒中查看，并按其中的建议观察。"
            if any(b.info_layer is InfoLayer.HEALTH for b in beats)
            else None
        ),
    )


def _make_beat(
    *,
    candidate: DigestCandidate,
    layer: InfoLayer,
    line: str,
    index: int,
    at: StoryTimeOfDay,
) -> StoryBeat | None:
    """构造节拍。**构造失败返回 None 而不是抛异常。**

    为什么不让异常冒出去：契约会拒绝「娱乐层含生理词」等组合，
    而这类拒绝应该是**逐条隔离**的——一条坏节拍不能让当天的故事全丢。
    这与 `app/digest` 逐条隔离坏候选是同一条原则。
    """
    tone = StoryTone.SERIOUS if layer is InfoLayer.HEALTH else StoryTone.NEUTRAL
    try:
        return StoryBeat(
            at=at,
            info_layer=layer,
            tone=tone,
            line=line,
            fact_refs=[index],
        )
    except ValueError:
        # 兜底：把生理词换成中性表述再试一次。
        # 注意只兜底**娱乐层**——健康层不该走到这里（它的词是允许的）
        cleaned = _strip_physiology(line)
        if cleaned == line:
            return None
        try:
            return StoryBeat(
                at=at,
                info_layer=layer,
                tone=tone,
                line=cleaned,
                fact_refs=[index],
            )
        except ValueError:
            return None


def _stamp_for(
    candidate: DigestCandidate, messages: list[DigestMessage]
) -> datetime | None:
    """用 **quote** 找这条候选对应的时间戳。

    ## 为什么必须用 quote 而不是 content

    ``content`` 是**改写后**的句子（「它喜欢在窗台晒太阳」），
    而 ``quote`` 是**逐字复制**的原文片段（「它早上在窗台趴着晒太阳」）。

    初版实现用 ``content`` 精确匹配，结果**一条也匹配不上**，
    所有节拍都退回默认的 ``EVENING`` —— 时间分桶静默失效。

    这个 bug 只在端到端跑的时候才看得见：单测里我用的是不改写的
    fixture，content 与原文相同，所以掩盖了它。

    > **quote 是全文里唯一按构造就能在对话中找到的字段** ——
    > 因为校验已经强制了这一点。用它做定位是免费的。
    """
    needle = _normalize(candidate.quote)
    if not needle:
        return None
    best: datetime | None = None
    for msg in messages:
        if msg.at is None:
            continue
        if needle in _normalize(msg.content):
            if best is None or msg.at < best:
                best = msg.at
    return best


def _normalize(text: str) -> str:
    """去空白后比较 —— 与 `app/digest` 的校验保持同一口径。"""
    return "".join(text.split())


def _strip_physiology(line: str) -> str:
    """去掉生理词，改成中性表述。

    这是**兜底**而不是首选做法：首选是在分类阶段就把健康内容
    分到 L3。走到这里说明分类漏了一条，值得记录。
    """
    cleaned = line
    for word in find_physiology_words(line):
        cleaned = cleaned.replace(word, "有点不一样")
    return cleaned


def summarize_to_story(
    *,
    day: date_type,
    candidates: list[DigestCandidate],
    stats: object | None = None,
    messages: list[DigestMessage] | None = None,
) -> DailyStory:
    """便利入口：直接由候选列表生成故事（跳过 `DailySummary` 构造）。

    主要给测试与调用方使用；生产路径应走 `compose_story`。
    """
    from app.schemas.digest import DailySummary, DigestStats

    if not isinstance(stats, DigestStats):
        raise TypeError("需要 DigestStats 才能确定标题与日期")
    summary = DailySummary(date=day, stats=stats, candidates=candidates)
    return compose_story(summary, messages=messages)
