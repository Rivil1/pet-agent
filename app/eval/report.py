"""报告渲染。

对应 voice-eval 的 `EvalTaskResult`（维度分 + 低分归因 + 优化建议），
但多了一节 **「本报告不能证明什么」**。

## 为什么必须有一节讲「不能证明什么」

`docs/06-roadmap.md` 对这类报告的判断是：

> 「我设计了六类评测」 → 面试官的问题是「**你怎么知道有效？**」

一份只列绿色勾号的报告与一份 PPT 没有区别。真正让数字可信的是
**边界被写清楚**：

- A 层是确定性的 → 可以声称绝对数值
- B 层缺数据 → 明说测不了，而不是给个占位数字
- C 层是主观的 → 只报相对比较，不报绝对值

把这三件事分开写在报告里，读的人才有可能正确使用这些数字。
不写，数字就会以同一种可信度被引用 —— 而那是最坏的结果，
因为它比「没有数字」更危险：它有数字的样子。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from app.eval.runner import Comparison, layer_summary
from app.eval.types import Layer, SuiteResult, grade

__all__ = ["render_markdown", "render_json", "write_report"]


_LAYER_TITLE = {
    Layer.A: "A 层 · 客观可判定",
    Layer.B: "B 层 · 有参考标准",
    Layer.C: "C 层 · 需主观判断",
}

_LAYER_CLAIM = {
    Layer.A: "判定标准是代码可重算的事实 → **可以声称绝对数值**",
    Layer.B: "依赖人工标注或真实先验 → 需说明污染与一致性，本报告只给可测项",
    Layer.C: "依赖主观判断 → **只应作相对比较**，不应引用绝对值",
}

#: 这些指标「降低」才是改进。渲染时标出来，免得读的人把它读反。
_LOWER_IS_BETTER = (
    "无据陈述率",
    "泄漏",
    "误判",
    "静默错误",
    "该问未问",
    "非法组合",
    "变化数",
    "text_only 数值泄漏",
)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _num(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{value:.3f}"


def render_markdown(
    suites: Sequence[SuiteResult],
    comparisons: Sequence[Comparison] = (),
    *,
    generated_at: datetime | None = None,
) -> str:
    """渲染完整报告。"""
    ts = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    primary = suites[0]

    out: list[str] = []
    out.append("# pet-agent 评测报告")
    out.append("")
    out.append(f"> 生成于 {ts} · 种子 {primary.seed} · 配置 `{primary.config_name}`")
    out.append("")

    out.append(_section_headline(primary))
    out.append(_section_layers(primary))
    out.append(_section_scenes(primary))
    out.append(_section_blocked(primary))

    if comparisons:
        for cmp in comparisons:
            out.append(_section_comparison(cmp))

    if len(suites) > 1:
        out.append(_section_all_configs(suites))

    out.append(_section_limits(primary))
    out.append(_section_howto())

    return "\n".join(out)


# =============================================================================
# 各节
# =============================================================================


def _section_headline(suite: SuiteResult) -> str:
    total = len(suite.results)
    failed = len(suite.failed)
    errored = len(suite.errored)
    blocked = sum(len(r.outcome.blocked) for r in suite.results)
    checks = [c for r in suite.results for c in r.outcome.checks]
    passed_checks = sum(1 for c in checks if c.passed)

    out = ["## 1. 总览", ""]
    out.append("| | |")
    out.append("|---|---|")
    out.append(f"| 场景 | {total} 个（失败 {failed}，崩溃 {errored}） |")
    out.append(f"| 断言 | {passed_checks}/{len(checks)} 通过 |")
    out.append(f"| **不可测项** | **{blocked} 项**（见 §4，不是「跳过」而是「测不了」） |")
    out.append("")

    if errored:
        out.append("**崩溃的场景**（一个场景崩溃不应带走整轮，但必须点名）：")
        out.append("")
        for r in suite.errored:
            first = (r.error or "").splitlines()[0]
            out.append(f"- `{r.scene.scene_id}` —— {first}")
        out.append("")

    return "\n".join(out)


def _section_layers(suite: SuiteResult) -> str:
    summary = layer_summary(suite)
    out = ["## 2. 按可判定性分层", ""]
    out.append("> `docs/DESIGN.md` §6.1：**不要对所有评测项都声称可靠性。**")
    out.append("")

    for layer, stats in summary.items():
        out.append(f"### {_LAYER_TITLE[layer]}")
        out.append("")
        out.append(f"{_LAYER_CLAIM[layer]}")
        out.append("")
        out.append("| 场景 | 已评分 | 平均通过率 | 失败 | 不可测 |")
        out.append("|---|---|---|---|---|")
        out.append(
            f"| {int(stats['场景数'])} | {int(stats['已评分场景数'])} "
            f"| {_pct(stats['平均通过率'])} | {int(stats['失败场景数'])} "
            f"| {int(stats['不可测项数'])} |"
        )
        out.append("")

    return "\n".join(out)


def _section_scenes(suite: SuiteResult) -> str:
    out = ["## 3. 逐场景结果", ""]

    for r in suite.results:
        scene = r.scene
        items = "/".join(scene.items)
        if r.error:
            badge = "💥 崩溃"
        elif r.ok and r.outcome.checks:
            badge = "✅ 通过"
        elif not r.outcome.checks:
            badge = "⛔ 未测"
        else:
            badge = "❌ 未通过"

        out.append(f"### `{scene.scene_id}` · {scene.name}")
        out.append("")
        out.append(f"`{items}` · {badge} · {scene.question}")
        out.append("")

        if r.error:
            out.append("```")
            out.append((r.error or "").strip())
            out.append("```")
            out.append("")
            continue

        o = r.outcome

        if o.metrics:
            out.append("| 指标 | 值 |")
            out.append("|---|---|")
            for name, value in o.metrics.items():
                flag = ""
                if any(k in name for k in _LOWER_IS_BETTER):
                    flag = " ↓好"
                elif "recall" in name or "MRR" in name or "F1" in name:
                    flag = " ↑好"
                out.append(f"| {name} | `{_num(value)}`{flag} |")
            out.append("")

        if o.checks:
            out.append("**断言**")
            out.append("")
            for c in o.checks:
                mark = "✅" if c.passed else "❌"
                line = f"- {mark} {c.name}"
                if not c.passed and c.detail:
                    line += f"\n  - {c.detail}"
                out.append(line)
            out.append("")

        if o.notes:
            out.append("<details><summary>限制与说明</summary>")
            out.append("")
            for n in o.notes:
                out.append(f"- {n}")
            out.append("")
            out.append("</details>")
            out.append("")

        if o.details:
            out.append("<details><summary>明细</summary>")
            out.append("")
            out.append("```json")
            out.append(json.dumps(o.details[:8], ensure_ascii=False, indent=2))
            out.append("```")
            out.append("")
            out.append("</details>")
            out.append("")

    return "\n".join(out)


def _section_blocked(suite: SuiteResult) -> str:
    items = suite.all_blocked()
    out = ["## 4. 不可测项（为什么测不了）", ""]

    if not items:
        out.append("本次运行没有不可测项。")
        out.append("")
        return "\n".join(out)

    out.append(
        "> 这些项**没有产出数字**。给一个用占位数据算出来的数字比没有数字更糟 —— "
        "它有数字的样子，会被引用。"
    )
    out.append("")

    for scene_id, b in items:
        out.append(f"- **{b.item}**（`{scene_id}`）")
        out.append(f"  - 原因：{b.reason}")
        out.append(f"  - 需要：{b.needs}")
    out.append("")
    return "\n".join(out)


def _section_comparison(cmp: Comparison) -> str:
    exp = cmp.experiment
    out = [f"## 5. 消融对比：`{cmp.baseline.config_name}` → `{exp.config_name}`", ""]
    out.append(f"> {cmp.config_note}")
    out.append("")

    # ⚠️ **先判「实验组是不是压根没跑成」。**
    #
    # 初版直接进入指标对比，于是当实验组的场景全部崩溃时
    # （比如 mysql-store 组没配 MYSQL_*），报告写的是
    # 「所有指标完全一致。这本身是一个结论」——
    # 而真相是它一行都没跑。
    # 把「没跑」读成「跑出相同结果」是最坏的一种误读：
    # 它会得出「后端不影响正确性」这个**反向的**结论。
    if exp.errored and len(exp.errored) == len(exp.results):
        out.append(
            f"**本次对比无效：实验组的 {len(exp.errored)} 个场景全部崩溃。**"
        )
        out.append("")
        first = (exp.errored[0].error or "").splitlines()[0]
        out.append(f"首个错误：`{first}`")
        out.append("")
        out.append(
            "这不是「两组结果相同」，而是「实验组没跑起来」。"
            "两者的区别很关键 —— 前者会得出「该机制无贡献」的反向结论。"
        )
        out.append("")
        return "\n".join(out)

    if exp.errored:
        out.append(
            f"⚠️ 实验组有 {len(exp.errored)} 个场景崩溃，"
            "下面的对比**不含**那些场景。"
        )
        out.append("")

    changed = cmp.changed()
    if not changed:
        out.append(
            "**所有指标完全一致。** 这本身是一个结论：该机制在当前评测集上"
            "没有可测量的贡献 —— 比「我们做了防护」这句话有价值，也更诚实。"
        )
        out.append("")
        out.append(
            "不过要先确认它**真的跑过了** —— 「没跑」与「跑出相同结果」都会"
            "表现为无差异，而前者可能是环境问题。"
        )
        out.append("")
    else:
        out.append("| 场景 | 指标 | 基线 | 实验组 | 变化 | 方向 |")
        out.append("|---|---|---|---|---|---|")
        for d in cmp.deltas:
            if d.mark == "=":
                continue
            rel = "—" if d.rel is None else f"{d.rel * 100:+.1f}%"
            lower_better = any(k in d.metric for k in _LOWER_IS_BETTER)
            # 只说方向，不判断好坏：让代码去猜「哪个方向是好」
            # 迟早会把某个指标读反
            note = "（越低越好）" if lower_better else ""
            out.append(
                f"| `{d.scene_id}` | {d.metric} | `{_num(d.baseline)}` "
                f"| `{_num(d.experiment)}` | {rel} | {d.mark} {note} |"
            )
        out.append("")

        top = cmp.biggest(3)
        if top:
            out.append("**变化最大的三项**")
            out.append("")
            for d in top:
                out.append(
                    f"- `{d.scene_id}.{d.metric}`：`{_num(d.baseline)}` → "
                    f"`{_num(d.experiment)}`（{d.mark}）"
                )
            out.append("")

    if cmp.only_in:
        out.append("**只在一侧出现的指标**（说明某个配置下场景没产出它）")
        out.append("")
        for scene_id, metric, where in cmp.only_in:
            out.append(f"- `{scene_id}.{metric}` —— 只在 `{where}` 出现")
        out.append("")

    return "\n".join(out)


def _section_all_configs(suites: Sequence[SuiteResult]) -> str:
    out = ["## 6. 各配置总览", ""]
    out.append("| 配置 | 场景 | 失败 | 崩溃 | 断言通过率 | 不可测 |")
    out.append("|---|---|---|---|---|---|")
    for s in suites:
        checks = [c for r in s.results for c in r.outcome.checks]
        rate = (
            sum(1 for c in checks if c.passed) / len(checks) if checks else 0.0
        )
        blocked = sum(len(r.outcome.blocked) for r in s.results)
        out.append(
            f"| `{s.config_name}` | {len(s.results)} | {len(s.failed)} "
            f"| {len(s.errored)} | {_pct(rate)} | {blocked} |"
        )
    out.append("")
    return "\n".join(out)


def _section_limits(suite: SuiteResult) -> str:
    """**本报告不能证明什么。** 这一节是报告可信度的来源。"""
    out = ["## 7. 本报告不能证明什么", ""]

    out.append("### 用 MockLLM 跑出来的部分，不测模型质量")
    out.append("")
    out.append(
        "默认配置用 `MockLLM` + `HashEmbedder`（`DESIGN §6.5` 的可复现要求）。"
        "所以："
    )
    out.append("")
    out.append(
        "- **措辞质量、角色一致性、建议有用性**（C 层）完全没被测 —— "
        "mock 输出的是固定文案"
    )
    out.append(
        "- **语义检索能力**没被测 —— `HashEmbedder` 是词面匹配，"
        "「换个说法也能召回」需要真嵌入模型"
    )
    out.append(
        "- 延迟数字**不含真实模型延迟**，只是编排层自身开销"
    )
    out.append("")

    out.append("### A 层的绝对数值可信，但只覆盖 A 层")
    out.append("")
    out.append(
        "A 层的判定标准是代码可重算的事实（条数、泄漏数、是否命中禁词），"
        "所以那些数字可以当作绝对值引用。"
    )
    out.append("")
    out.append(
        "但 A 层**不覆盖**「解释得准不准」——那需要带标注的测试集与 hold-out 猫。"
    )
    out.append("")

    out.append("### 评测集是自造的")
    out.append("")
    out.append(
        "评测用例由本项目自己编写，因此**效度没有外部论证**。"
        "`DESIGN.md` §6.1 的 G1 缺口依然存在：它们能测出回归，"
        "但不能证明系统在真实用户输入上的表现。"
    )
    out.append("")

    if suite.all_blocked():
        out.append("### 有测不了的项")
        out.append("")
        out.append(
            f"本次运行有 **{len(suite.all_blocked())}** 项不可测，"
            "逐条原因见 §4。"
        )
        out.append("")

    return "\n".join(out)


def _section_howto() -> str:
    return "\n".join(
        [
            "## 8. 复现",
            "",
            "```bash",
            "make eval                 # 基线，输出 reports/eval.md",
            "make ablation             # 基线与全部消融组对比",
            "python -m app.eval --list # 看有哪些场景与配置",
            "python -m app.eval --config no-guard --only honesty,routing",
            "```",
            "",
            "评测**离线可跑**，不需要 API Key 或网络。",
            "",
        ]
    )


# =============================================================================
# 输出
# =============================================================================


def render_json(suites: Sequence[SuiteResult]) -> str:
    """机器可读形式。CI 里用于与历史结果比对。"""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suites": [
            {
                "config": s.config_name,
                "notes": list(s.config_notes),
                "seed": s.seed,
                "layers": {
                    layer.value: stats for layer, stats in layer_summary(s).items()
                },
                "scenes": [
                    {
                        "scene_id": r.scene.scene_id,
                        "name": r.scene.name,
                        "layer": r.scene.layer.value,
                        "items": list(r.scene.items),
                        "error": r.error,
                        "score": r.score,
                        "grade": grade(r.score),
                        "metrics": r.outcome.metrics,
                        "checks": [
                            {"name": c.name, "passed": c.passed, "detail": c.detail}
                            for c in r.outcome.checks
                        ],
                        "blocked": [
                            {"item": b.item, "reason": b.reason, "needs": b.needs}
                            for b in r.outcome.blocked
                        ],
                        "notes": list(r.outcome.notes),
                        "samples": r.outcome.samples,
                    }
                    for r in s.results
                ],
            }
            for s in suites
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_report(
    suites: Sequence[SuiteResult],
    comparisons: Sequence[Comparison] = (),
    *,
    out_dir: str | Path = "reports",
    stem: str = "eval",
) -> tuple[Path, Path]:
    """写出 Markdown + JSON。返回两个路径。"""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    md_path = directory / f"{stem}.md"
    json_path = directory / f"{stem}.json"

    md_path.write_text(
        render_markdown(suites, comparisons), encoding="utf-8"
    )
    json_path.write_text(render_json(suites), encoding="utf-8")
    return md_path, json_path
