"""从对话中提取候选记忆。**重点是拒绝，不是提取。**

## 校验流程

```
LLM 输出 JSON
   ↓
逐条解析（结构非法 → 拒绝，不抛异常）
   ↓
quote 机械校验（quote 必须逐字出现在当天对话里）
   ↓
枚举值校验（event_type / polarity / source_layer 非法 → 拒绝）
   ↓
通过 → DigestCandidate ；不通过 → RejectedCandidate（**保留原因**）
```

## 为什么一条坏候选不能让整天失败

LLM 返回的枚举值可能是 ``"health"``（不存在）或漏字段。
如果这里抛异常，**一整天的总结会因为一条坏候选而全部丢失**。

所以逐条隔离：坏的那条进 ``rejected``，好的照常通过。
这也让「模型今天坏了多少条」变成一个可观测数字。

## 与 B2/B3 的关系

B2 是静默丢弃，B3 是静默编造。这里的每一步失败都**必须留下记录**：
结构错误、引用对不上、枚举非法——全部进 ``RejectedCandidate`` 并带原因。
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, TypeVar

from app.digest.aggregate import render_transcript
from app.schemas.digest import (
    QUOTE_MIN_CHARS,
    DigestCandidate,
    DigestMessage,
    DigestStats,
    ExtractionSource,
    RejectedCandidate,
)
from app.schemas.memory import EventType, Polarity

#: 系统 prompt。
#:
#: 三个刻意的设计：
#:
#: 1. **含 "JSON" 字样** —— DashScope 的 JSON mode 要求 messages 里出现该词，
#:    否则请求会被拒。这是实测得到的约束，不是习惯。
#: 2. **明确要求逐字复制** —— 改写后的片段无法机械核验。
#: 3. **明说 quote 会被代码校验** —— 让模型知道这是硬门，而不是建议。
SYSTEM_PROMPT = """你是宠物陪伴系统的对话整理器。你的任务是从**当天**的对话里，\
提取值得长期记住的信息。

## 硬性规则

1. 每条都必须附 `quote`：**从对话里逐字复制的连续片段**。
   不得改写、不得拼接、不得跨句组合。
   **系统会用代码检查该片段是否真的出现在对话里，检查不通过的内容会被丢弃。**

2. 对话里没有明确说过的，不要写。宁可少提取，不可编造。
   你做出的归纳（如「它喜欢窗台」）如果没有原句支撑，就不要提交。

3. 只提取与这只宠物有关、且**未来还用得上**的信息。
   不要提取：纯问候、寒暄、疑问句、你自己说的话。

4. 区分两类来源：
   - `owner_record`：主人直接说的事实，或他做过的事
   - `ai_inference`：需要你归纳才能得出的（**这类会被要求主人确认后才生效**）
   拿不准时，用 `ai_inference`。

5. `content` 要写成能独立检索的一句完整话（含主语）。
   写「怕吸尘器」不好，写「它怕吸尘器，听到声音就躲」好。

## 输出格式

严格 JSON，不要 markdown 代码块，不要任何解释文字：

{"candidates": [{"content": "...", "quote": "...", "source_layer": "owner_record", \
"event_type": "preference", "subject": "vacuum", "polarity": "negative"}]}

字段取值：
- `source_layer`: owner_record | ai_inference
- `event_type`: episode | behavior | preference | routine | health | context
- `polarity`: positive | negative | neutral
- `subject`: 英文小写短词，如 vacuum / food / window / litter

若无值得记住的内容，返回 `{"candidates": []}`。"""


def build_user_prompt(
    *,
    stats: DigestStats,
    messages: list[DigestMessage],
    existing_memories: list[str] | None = None,
) -> str:
    """构造用户 prompt。

    **把已写入的记忆显式列出来**，避免重复提取同一件事：
    重复提取会被飞轮去重成「强化」，看起来没坏处，
    但它会稀释 ``rejection_rate`` 这个质量指标，也会浪费 token。
    """
    parts: list[str] = [f"日期：{stats.date.isoformat()}"]

    if stats.topic_counts:
        rendered = "、".join(f"{k} 出现 {v} 次" for k, v in stats.topic_counts.items())
        # 注意：这里给的是**代码算出来的计数**，不是让模型去数
        parts.append(f"系统统计（由代码计算，不要重新数）：{rendered}")

    if existing_memories:
        listed = "\n".join(f"  - {m}" for m in existing_memories[:20])
        parts.append(f"以下内容**已经记录过**，不要重复提取：\n{listed}")

    parts.append(f"对话转写：\n{render_transcript(messages)}")
    parts.append("请按规则提取。记住：每条的 quote 都会被代码校验。")
    return "\n\n".join(parts)


def _normalize(text: str) -> str:
    """去掉所有空白后比较。

    只去空白，**不做同义词/标点归一化**：
    归一化越强，校验越松，最终会松到「看起来像就算过」——那等于没有校验。
    """
    return "".join(text.split())


def verify_quote(
    quote: str, haystack: str, *, min_chars: int = QUOTE_MIN_CHARS
) -> tuple[bool, str]:
    """校验引用片段确实出现在原文里。

    Returns:
        ``(通过?, 失败原因)``。通过时原因为空串。

    两道门，缺一不可：

    1. **长度下界** —— 1–3 字的片段（「它」「今天」）几乎必然能找到，
       会让校验退化成恒真。任何编造只要附一句常见短语就能通过。
    2. **子串匹配** —— 逐字出现。改写过的片段会失败，**这是刻意的**：
       若允许改写，就无法机械核验。
    """
    normalized_quote = _normalize(quote)
    if not normalized_quote:
        return False, "缺少引用片段（quote 为空）"
    if len(normalized_quote) < min_chars:
        return False, (
            f"引用片段过短（{len(normalized_quote)} 字 < {min_chars}），"
            "无法作为依据"
        )
    if normalized_quote not in _normalize(haystack):
        return False, "引用片段在当天对话中找不到——疑似编造"
    return True, ""


_E = TypeVar("_E", bound=Enum)


def _try_enum(value: Any, enum_cls: type[_E], field: str) -> tuple[_E | None, str]:
    """把字符串转成枚举。失败时返回原因而不是抛异常。

    为何不直接抛：模型可能返回不存在的取值（如 `event_type="health"`）。
    抛异常会让**一整天的总结因一条坏候选而全部丢失**。
    """
    if not isinstance(value, str):
        return None, f"字段 {field} 类型非法（期望字符串，得到 {type(value).__name__}）"
    try:
        return enum_cls(value), ""
    except ValueError:
        allowed = "/".join(str(m.value) for m in enum_cls)
        return None, f"字段 {field} 取值非法：{value!r}（允许：{allowed}）"


def parse_candidates(
    raw: str, *, messages: list[DigestMessage]
) -> tuple[list[DigestCandidate], list[RejectedCandidate]]:
    """解析模型输出，逐条校验。

    **任何一条失败都不影响其余条目** —— 整天总结不能因为一条坏候选而丢失。
    """
    haystack = "\n".join(m.content for m in messages)

    payload, parse_error = _load_json(raw)
    if parse_error:
        return [], [RejectedCandidate(content=_truncate(raw), reason=parse_error)]

    items = payload.get("candidates")
    if not isinstance(items, list):
        return [], [
            RejectedCandidate(
                content=_truncate(raw),
                reason="返回结构非法：缺少 candidates 数组",
            )
        ]

    admitted: list[DigestCandidate] = []
    rejected: list[RejectedCandidate] = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            rejected.append(
                RejectedCandidate(
                    content=f"<第 {idx + 1} 条，非对象>",
                    reason=f"候选类型非法：{type(item).__name__}",
                )
            )
            continue
        candidate, reason = _build_one(item, haystack=haystack)
        if candidate is not None:
            admitted.append(candidate)
        else:
            rejected.append(
                RejectedCandidate(
                    content=_truncate(str(item.get("content", "<无 content>"))),
                    reason=reason,
                    quote=item.get("quote") if isinstance(item.get("quote"), str) else None,
                    source_layer=_optional_source(item.get("source_layer")),
                )
            )

    return admitted, rejected


def _build_one(
    item: dict[str, Any], *, haystack: str
) -> tuple[DigestCandidate | None, str]:
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "content 缺失或为空"

    quote = item.get("quote")
    if not isinstance(quote, str):
        return None, "quote 缺失——无依据的候选不予采纳"

    ok, reason = verify_quote(quote, haystack)
    if not ok:
        return None, reason

    # 逐个判 None 而不是判 err 字符串：
    # 这样类型检查器能确定非 None，而不用在后续每个使用点断言
    event_type, err = _try_enum(item.get("event_type"), EventType, "event_type")
    if event_type is None:
        return None, err
    polarity, err = _try_enum(item.get("polarity", "neutral"), Polarity, "polarity")
    if polarity is None:
        return None, err
    source_layer, err = _try_enum(
        item.get("source_layer"), ExtractionSource, "source_layer"
    )
    if source_layer is None:
        return None, err

    subject = item.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        # subject 缺失不致命：用 event_type 兜底，保证仍可检索
        subject = event_type.value

    return (
        DigestCandidate(
            event_type=event_type,
            subject=subject.strip().lower()[:64],
            content=content.strip(),
            quote=quote.strip(),
            source_layer=source_layer,
            confidence=_confidence_for(source_layer),
            polarity=polarity,
        ),
        "",
    )


def _confidence_for(source_layer: ExtractionSource) -> float:
    """来源决定置信度。

    主人的直述高于系统归纳 —— 这不是调参，是语义：
    前者是观察，后者是推断（见 DESIGN.md 边界 3 的四层信息分层）。
    """
    return {
        ExtractionSource.OWNER_RECORD: 0.80,
        ExtractionSource.AI_INFERENCE: 0.45,
    }[source_layer]


def _optional_source(value: Any) -> ExtractionSource | None:
    try:
        return ExtractionSource(value)
    except (ValueError, TypeError):
        return None


def _load_json(raw: str) -> tuple[dict[str, Any], str]:
    """从模型输出里取出 JSON 对象。

    模型可能返回：纯 JSON / ```json 围栏 / 前后带解释文字。
    三种都要能处理——否则一次格式抖动就会丢掉整天的总结。

    **但处理方式必须是「提取」而非「修复」**：只做围栏剥离与首尾偏移，
    绝不尝试补全残缺 JSON。补全意味着我们替模型编了它没说的内容。
    """
    text = raw.strip()
    if not text:
        return {}, "模型返回空内容"

    # 剥离 markdown 围栏
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # 退一步：取第一个 { 到最后一个 }（模型前面加了话）
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return {}, "返回内容不是合法 JSON，也找不到 JSON 对象"
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return {}, f"JSON 解析失败：{exc.msg}"

    if not isinstance(payload, dict):
        return {}, f"JSON 顶层不是对象（得到 {type(payload).__name__}）"
    return payload, ""


def _truncate(text: str, limit: int = 200) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "…"
