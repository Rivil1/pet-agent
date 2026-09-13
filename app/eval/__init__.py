"""评测：**产出数字，而不只是通过/失败**。

## 与 `tests/` 的分工

| | `tests/` | `app/eval/` |
|---|---|---|
| 问题 | 「代码是否按我说的做」 | 「它做得有多好」 |
| 输出 | 通过 / 失败 | **数字 + 基线差值** |
| 断言 | 硬约束 | 指标 |
| 失败含义 | 有 bug | 有改进空间 |

两者都需要：`tests/` 保证不退化，`app/eval/` 证明有效。

## 为什么这件事是当前最大的缺口

`docs/06-roadmap.md` 的核心判断：

> 这个项目现在是**一份很好的设计文档**，但还不是**一个能证明自己有效的系统**。
> 差距全在「可证明性」上。

`docs/DESIGN.md` §6 早已把评测体系设计完整（A/B/C 三层可判定性、E1–E34、
基线消融、参数敏感性），但 `app/eval/` 目录一直是空的 ——
所以「效果怎么样」这个问题一直没有数字可答。

## 快速开始

```bash
python -m app.eval                 # 基线，写入 reports/eval.md
python -m app.eval --ablation      # 基线与全部消融组
python -m app.eval --list          # 看有哪些场景与配置
python -m app.eval --only honesty  # 只跑一个场景
```

## 设计上的三条硬约束

1. **分层呈现。** A 层（确定性）可以声称绝对数值；C 层（主观）只应作相对比较。
   混在一起等于给后者披上前者的可信度（`DESIGN §6.1`）。
2. **必须有基线。** 没有差值的数字不说明任何事（`DESIGN §6.3`）。
3. **不可测的要说明，不能跳过。** 跳过会让报告少一行，读的人以为都测过了。
"""

from app.eval.config import ABLATIONS, BASELINE, PRESETS, EvalConfig
from app.eval.report import render_json, render_markdown, write_report
from app.eval.runner import Comparison, compare, layer_summary, run_all_configs, run_suite
from app.eval.runtime import Runtime, make_runtime
from app.eval.scenes import ALL_SCENES, SCENES_BY_ID
from app.eval.types import (
    Blocked,
    Check,
    Layer,
    Scene,
    SceneOutcome,
    SceneResult,
    SuiteResult,
)

__all__ = [
    # 配置
    "EvalConfig",
    "BASELINE",
    "ABLATIONS",
    "PRESETS",
    # 场景
    "Scene",
    "ALL_SCENES",
    "SCENES_BY_ID",
    # 类型
    "Layer",
    "Check",
    "Blocked",
    "SceneOutcome",
    "SceneResult",
    "SuiteResult",
    # 执行
    "Runtime",
    "make_runtime",
    "run_suite",
    "run_all_configs",
    "compare",
    "Comparison",
    "layer_summary",
    # 报告
    "render_markdown",
    "render_json",
    "write_report",
]
