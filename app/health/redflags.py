"""红旗规则：加载与求值。

对应 `docs/07-health.md` §4，规则表在 `data/health/red_flags.yaml`。

## 为什么是「数据」而不是代码

规则表的设计目的是**让执业兽医可以审核与维护，而不需要阅读或修改任何程序代码**。
`reviewed_by: null` 表示尚未经兽医审核，未审核的阈值型规则一律 `enabled: false`。

## 三值逻辑：本模块最重要的设计

求值结果是 ``True`` / ``False`` / **``None``（无法评估）** 三者之一，不是布尔。

为什么必须如此：**「信号缺失」与「未命中」是完全不同的事。**

```
litter_box.urine_output = none   →  命中（危险）
litter_box.urine_output = normal →  未命中
litter_box.urine_output 没采集到  →  **无法评估** ← 不能当成「未命中」
```

若把「无法评估」当成「未命中」，系统就会输出「未发现异常」——
而它其实什么都没看。**这正是 `UrgencyLevel.INSUFFICIENT_DATA` 存在的理由**，
也是 B3（零填充是静默编造）在健康模块的对应形态。

因此：

- ``any_of``：任一为 True → True；否则任一无值 → **None**；否则 False
- ``all_of``：任一为 False → False；否则任一无值 → **None**；否则 True
- **规则只在结果为 True 时命中** —— None 不触发（假阴性交给 `INSUFFICIENT_DATA` 显式表达）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.schemas.health import RedFlagHit, UrgencyLevel

#: 三值逻辑的取值。``None`` 表示「无法评估」。
Tri = bool | None


class RedFlagTableError(ValueError):
    """规则表格式非法。**加载期报错，不留到运行期。**"""


@dataclass(frozen=True)
class RedFlagRule:
    """一条红旗规则。"""

    rule_id: str
    enabled: bool
    urgency: UrgencyLevel
    title: str
    message: str
    action: str
    conditions: dict[str, Any]
    species: tuple[str, ...] = ()
    sex_bias: str | None = None
    aggravating: tuple[dict[str, Any], ...] = ()
    sources: tuple[str, ...] = ()
    disabled_reason: str | None = None


@dataclass(frozen=True)
class RedFlagTable:
    """规则表。带版本与 `sha256`，用于可复现性（同一次评估可回溯到确切版本）。"""

    version: str
    reviewed_by: str | None
    reviewed_at: str | None
    vocabulary: dict[str, dict[str, Any]]
    rules: tuple[RedFlagRule, ...]
    sha256: str

    @property
    def enabled_rules(self) -> tuple[RedFlagRule, ...]:
        return tuple(r for r in self.rules if r.enabled)

    @property
    def is_reviewed(self) -> bool:
        """是否已经过兽医审核。**未审核不为错，但必须可见。**"""
        return bool(self.reviewed_by)

    def signal_names(self) -> frozenset[str]:
        return frozenset(self.vocabulary)

    @classmethod
    def load(cls, path: str | Path) -> RedFlagTable:
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        payload = yaml.safe_load(raw.decode("utf-8"))

        if not isinstance(payload, dict):
            raise RedFlagTableError("规则表顶层不是映射")
        if "signal_vocabulary" not in payload or "rules" not in payload:
            raise RedFlagTableError("规则表缺少 signal_vocabulary 或 rules")

        vocabulary: dict[str, dict[str, Any]] = payload["signal_vocabulary"]
        known = set(vocabulary)

        rules: list[RedFlagRule] = []
        for rawrule in payload["rules"]:
            if not isinstance(rawrule, dict):
                raise RedFlagTableError(f"规则项不是映射：{rawrule!r}")
            grid = rawrule.get("id")
            if not grid:
                raise RedFlagTableError("规则缺少 id")

            # ── 校验：规则只能引用词汇表里定义过的信号 ──
            # yaml 里写明了理由：避免拼写错误导致规则**静默失效**。
            # 一条永远不命中的红旗规则，比没有这条规则更危险。
            for ref in _referenced_signals(rawrule.get("conditions", {})):
                if ref not in known:
                    raise RedFlagTableError(
                        f"规则 {grid!r} 引用了未定义的信号 {ref!r}；"
                        "请在 signal_vocabulary 中先定义（拼写错误会让规则静默失效）"
                    )
            for agg in rawrule.get("aggravating", []) or []:
                ref = agg.get("signal")
                if ref and ref not in known:
                    raise RedFlagTableError(
                        f"规则 {grid!r} 的 aggravating 引用了未定义的信号 {ref!r}"
                    )

            if rawrule.get("enabled") and not rawrule.get("message"):
                raise RedFlagTableError(f"已启用的规则 {grid!r} 缺少 message")
            if rawrule.get("enabled") and not rawrule.get("sources"):
                raise RedFlagTableError(
                    f"已启用的规则 {grid!r} 缺少 sources —— "
                    "启用的规则必须可回溯到来源"
                )
            if not rawrule.get("enabled") and not rawrule.get("disabled_reason"):
                raise RedFlagTableError(
                    f"未启用的规则 {grid!r} 必须写明 disabled_reason —— "
                    "否则「为什么不启用」会变成无人可查的悬案"
                )

            rules.append(
                RedFlagRule(
                    rule_id=str(grid),
                    enabled=bool(rawrule.get("enabled", False)),
                    urgency=UrgencyLevel(str(rawrule.get("urgency", "L3"))),
                    title=str(rawrule.get("title", "")),
                    message=str(rawrule.get("message", "")).strip(),
                    action=str(rawrule.get("action", "")),
                    conditions=rawrule.get("conditions", {}),
                    species=tuple(rawrule.get("species", []) or ()),
                    sex_bias=rawrule.get("sex_bias"),
                    aggravating=tuple(rawrule.get("aggravating", []) or ()),
                    sources=tuple(rawrule.get("sources", []) or ()),
                    disabled_reason=rawrule.get("disabled_reason"),
                )
            )

        return cls(
            version=str(payload.get("version", "unknown")),
            reviewed_by=payload.get("reviewed_by"),
            reviewed_at=str(payload.get("reviewed_at")) if payload.get("reviewed_at") else None,
            vocabulary=vocabulary,
            rules=tuple(rules),
            sha256=sha,
        )


def _referenced_signals(conditions: dict[str, Any]) -> list[str]:
    """递归收集条件里引用的所有信号名。"""
    out: list[str] = []
    for key in ("any_of", "all_of"):
        for item in conditions.get(key, []) or []:
            if "signal" in item:
                out.append(str(item["signal"]))
            else:
                out.extend(_referenced_signals(item))
    return out


# ─────────────────────────────────────────────────────────────
# 求值（三值逻辑）
# ─────────────────────────────────────────────────────────────


def _compare(op: str, actual: Any, expected: Any) -> bool:
    if op == "in":
        return actual in (expected or [])
    if op == "equals":
        return actual == expected
    if op == "is_true":
        return actual is True
    if op == "is_false":
        return actual is False
    if op in (">=", ">", "<=", "<"):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        if not isinstance(expected, (int, float)) or isinstance(expected, bool):
            # 阈值是 null（待执业兽医确认）→ 永不命中。
            # 这类规则在加载期就该是 enabled: false，这里是双保险。
            return False
        return {
            ">=": actual >= expected,
            ">": actual > expected,
            "<=": actual <= expected,
            "<": actual < expected,
        }[op]
    raise RedFlagTableError(f"未知的比较运算符：{op!r}")


def eval_condition(cond: dict[str, Any], signals: dict[str, Any]) -> Tri:
    """求值单个条件。返回 ``True``/``False``/``None``（无法评估）。"""
    if "any_of" in cond:
        results = [eval_condition(c, signals) for c in cond["any_of"]]
        if any(r is True for r in results):
            return True
        return None if any(r is None for r in results) else False

    if "all_of" in cond:
        results = [eval_condition(c, signals) for c in cond["all_of"]]
        if any(r is False for r in results):
            return False
        return None if any(r is None for r in results) else True

    name = cond.get("signal")
    if not name:
        raise RedFlagTableError(f"条件既无 any_of/all_of 也无 signal：{cond!r}")

    if name not in signals:
        return None  # **无法评估，不是未命中**

    actual = signals[name]
    if actual is None:
        # **信号「已采集但值为 None」也算无法评估。**
        #
        # 这是一个很容易写错的地方：字典里有这个键，看起来「有数据」，
        # 但值是 None —— 若让 None 进入比较，`None >= 5` 会返回 False，
        # 于是「不知道」被静默地变成了「未命中」。
        # 这正是 UrgencyLevel.INSUFFICIENT_DATA 要防的事。
        return None

    return _compare(str(cond.get("op", "equals")), actual, cond.get("value"))


def evaluate(
    table: RedFlagTable, signals: dict[str, Any]
) -> tuple[list[RedFlagHit], list[str]]:
    """对一组信号求值。

    Args:
        table: 规则表（只求值 ``enabled`` 的规则）。
        signals: 已采集的信号值。**未出现的信号视为「无法评估」。**

    Returns:
        ``(命中的红旗, 因信号缺失而无法评估的规则 id)``。

        **第二个返回值必须被使用** —— 它决定了
        ``UrgencyLevel.INSUFFICIENT_DATA`` 是否要给出。
        丢掉它就等于把「没看」说成「没事」。
    """
    hits: list[RedFlagHit] = []
    undecidable: list[str] = []

    for rule in table.enabled_rules:
        result = eval_condition(rule.conditions, signals)

        if result is None:
            undecidable.append(rule.rule_id)
            continue
        if result is False:
            continue

        matched = _matched_signals(rule.conditions, signals)
        aggravated = [
            str(a["signal"])
            for a in rule.aggravating
            if a.get("signal") in signals
            and signals[a["signal"]] is not None
            and _compare(
                str(a.get("op", "equals")), signals[a["signal"]], a.get("value")
            )
        ]

        hits.append(
            RedFlagHit(
                rule_id=rule.rule_id,
                urgency=rule.urgency,
                title=rule.title,
                message=rule.message,
                action=rule.action,
                matched_conditions=matched,
                aggravating_conditions=aggravated,
                rule_version=table.version,
                sources=list(rule.sources),
            )
        )

    hits.sort(key=lambda h: h.urgency.value, reverse=True)
    return hits, undecidable


def _matched_signals(conditions: dict[str, Any], signals: dict[str, Any]) -> list[str]:
    """列出**确实为真**的信号名，用于向用户解释「为什么触发」。"""
    out: list[str] = []
    for key in ("any_of", "all_of"):
        for item in conditions.get(key, []) or []:
            if "signal" in item:
                if eval_condition(item, signals) is True:
                    out.append(str(item["signal"]))
            else:
                out.extend(_matched_signals(item, signals))
    return out


def dump_json(obj: Any) -> str:
    """调试用：把结果打成可读 JSON。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
