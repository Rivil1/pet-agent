"""把 `DailyStory` 渲染成文字。

**这一层不新增任何内容** —— 它只负责排版。所有事实来自 `StoryBeat.line`，
而每个节拍都带 `fact_refs`（可追溯）。渲染层若"顺手润色"，
追溯链就断了。

## 声明按层分开

`docs/14` 禁令 1 的机械落法：**「仅供娱乐」不能盖在健康内容上。**
它传达的是「这不用太当真」，而那正是禁令 1 想防的。

所以渲染时：

- 娱乐/事实节拍 → 段落后附 `disclaimer`
- 健康节拍 → 单独分区，附 `health_notice`（严肃提示，不是免责）
"""

from __future__ import annotations

from app.schemas.story import DailyStory, InfoLayer, StoryBeat, time_display


def render_story(story: DailyStory, *, digest_notes: list[str] | None = None) -> str:
    """渲染成可读文本。

    呈现顺序：**健康层在最前**。

    这与 `docs/14` §4 的 A2 一致（L3 可抢占并压住 L1/L2）——
    让用户先看到需要严肃对待的内容，再看可爱部分。

    Args:
        digest_notes: 摘要流水线的说明。

            **无故事内容时必须传** —— 否则用户看到「今天还没有可讲述的内容」，
            而实际上可能是「有对话但提取失败」。
            两件事看起来一样，而它们的含义完全不同。
    """
    lines: list[str] = [story.title, ""]

    health = [b for b in story.beats if b.info_layer is InfoLayer.HEALTH]
    others = [b for b in story.beats if b.info_layer is not InfoLayer.HEALTH]

    if health:
        lines.append("【需要留意】")
        for beat in health:
            # 健康层也标时间：只说「今天吐了两次」而不说什么时候，
            # 对「该不该现在去观察」这个判断没有帮助
            lines.append(f"  · {time_display(beat.at)}：{beat.line}")
        if story.health_notice:
            lines.append(f"  {story.health_notice}")
        lines.append("")

    if others:
        for beat in others:
            lines.append(f"{time_display(beat.at)}  {beat.line}")
        lines.append("")

    if not story.beats:
        lines.append("（今天还没有可讲述的内容）")
        # 区分「没说话」与「说了但没提取出来」——
        # 后者需要用户知道，因为它可能是系统的问题而不是当天真的没事
        if digest_notes:
            lines.append("")
            for note in digest_notes:
                lines.append(f"ⓘ {note}")
        lines.append("")

    # 免责声明只覆盖娱乐内容；若当天只有健康内容则不出现
    if any(b.info_layer is not InfoLayer.HEALTH for b in story.beats):
        lines.append(story.disclaimer)

    if story.excluded:
        # **不静默丢弃**：用户看不到某条记录时，应能知道它去哪了
        lines.append("")
        lines.append(f"（另有 {len(story.excluded)} 条记录未进入故事）")

    return "\n".join(lines).rstrip() + "\n"


def render_beat_for_panel(beat: StoryBeat) -> dict[str, str | None]:
    """把一个节拍渲染成**漫画格 / 写真页**所需的最小结构。

    **这不是图像生成**，只是把已有字段映射成下游需要的形状。
    留在这里是为了让「结构天然支持漫画/写真」这件事可验证 ——
    将来接图像模型时，改的是这一层，不是 `StoryBeat`。

    Returns:
        含 ``caption`` / ``photo_ref`` / ``style`` 的字典。

        ``style`` 由 `info_layer` 决定，**不由调用方自由选择** ——
        A3 要求 L1 不得使用 L3 的视觉样式，反之亦然。
    """
    style = {
        InfoLayer.ENTERTAINMENT: "entertainment",
        InfoLayer.FACT: "factual",
        InfoLayer.HEALTH: "health",
    }[beat.info_layer]
    return {
        "caption": beat.line,
        "photo_ref": beat.photo_ref,
        "style": style,
        "time_of_day": beat.at.value,
    }
