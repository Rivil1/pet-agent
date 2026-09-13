"""「宠物的一天」：把每日总结改写成宠物口吻的叙述。

**日报是娱乐产物，但它的安全边界必须由代码守，不能靠 prompt。**

详见 [app.schemas.story][app.schemas.story] 的模块 docstring。

## 用法

```python
from app.digest import summarize_day
from app.story import compose_story, render_story

result = summarize_day(day=..., messages=..., writer=..., llm=..., ...)
story = compose_story(result.summary, messages=messages)
print(render_story(story))
```

## 三层防线

| # | 防线 | 位置 | 违反的后果 |
| --- | --- | --- | --- |
| 1 | 只能从**已校验候选**派生 | `compose_story` 的输入类型 | 编造绕过 quote 校验 |
| 2 | **健康信号强制进 L3** 且禁止 playful | `_classify` + `StoryBeat` 契约 | 警示被娱乐调性消解 |
| 3 | 娱乐层**禁止生理词** | `StoryBeat` 契约 | 用户据拟人化台词改变照顾行为 |
"""

from app.story.compose import compose_story, summarize_to_story
from app.story.render import render_beat_for_panel, render_story

__all__ = [
    "compose_story",
    "summarize_to_story",
    "render_story",
    "render_beat_for_panel",
]
