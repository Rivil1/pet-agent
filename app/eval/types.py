"""评测的类型与数据结构。

## 与「测试」的分工

| | 测试（`tests/`） | 评测（本模块） |
|---|---|---|
| 回答的问题 | 「代码是否按我说的做」 | 「它做得**有多好**」 |
| 输出 | 通过 / 失败 | **数字 + 基线对比** |
| 断言 | 硬约束（不变量、隔离） | 指标（recall、F1、ECE） |
| 失败的含义 | 有 bug | 有改进空间 |

两者都需要，但不能互相替代。`docs/06-roadmap.md` 的核心判断是：

> 这个项目现在是**一份很好的设计文档**，但还不是**一个能证明自己有效的系统**。
> 差距全在「可证明性」上。

所以本模块的目标不是「多跑几个断言」，而是**产出可以写进 README 的数字**，
并且每个数字都带一个基线 —— 没有基线的数字没有意义。

## 结构参照（voice-eval 的评测平台模型）

```
场景 Scene  ──►  任务 Task  ──►  逐项明细 Item  ──►  任务结果 Result
（可复用装置）    （一次运行）      （一条测量）        （维度分 + 低分归因）
```

pet-agent 把它落成：

```
Scene（一个可测装置）  ──►  Suite（一次配置下的全量运行）
        │                            │
        ├─ metrics: {名: 值}          ├─ 各场景结果
        ├─ checks:  硬断言            ├─ 维度汇总（A/B/C 层）
        ├─ notes:   限制说明          └─ 与基线的差值
        └─ blocked: 本次测不了的原因
```

## 三层可判定性（DESIGN §6.1）

`Layer` 不是标签，是**声称可靠性时的边界**：

| 层 | 判定方式 | 允许声称 |
|---|---|---|
| `A` | 确定性：比对数据库 / 结构 | ✅ 绝对数值 |
| `B` | 有参考标准（人工标注 / 兽医） | ◐ 需说明污染与一致性 |
| `C` | 主观 | ⚠️ **只报告相对比较**，不给绝对值 |

把 C 层的分数当绝对值报出去，是这套系统最容易被质疑的地方 ——
而它恰恰是「LLM-as-judge 给了 8.5 分」这类说法的常见形态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Protocol

__all__ = [
    "Layer",
    "Check",
    "Blocked",
    "SceneOutcome",
    "Scene",
    "SceneResult",
    "SuiteResult",
    "EvalContext",
]


class Layer(str, Enum):
    """可判定性分层。**决定这个指标能不能被当作绝对值引用。**"""

    A = "A"  # 客观可判定：确定性比对，可声称可靠性
    B = "B"  # 有参考标准：需说明污染与标注一致性
    C = "C"  # 需主观判断：只报告相对比较


@dataclass(frozen=True)
class Check:
    """一条硬断言。

    与 `metrics` 的分工：`metrics` 是「多好」，`checks` 是「对不对」。
    一条 check 失败就是 bug，而一个 metric 偏低只是改进空间。
    """

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        mark = "✅" if self.passed else "❌"
        return f"{mark} {self.name}" + (f" —— {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class Blocked:
    """**本次测不了的项，以及为什么。**

    ## 为什么需要这个类型，而不是「跳过」

    静默跳过的后果是：报告里少一行，而读报告的人以为「都测过了」。
    本项目已有的一个真实例子：`data/priors/catmeows_stats.json`
    里全部是占位值（`is_placeholder: true`），群体先验尚未构建 ——
    于是 E21（声学情境分类 macro-F1）**根本没法诚实地测**。

    在这种情况下，正确的行为是写清楚「为什么测不了、缺什么」，
    而不是给一个用占位数据算出来的数字。
    后者比没有数字更糟：它有数字的样子，会被引用、被传播。
    """

    item: str
    reason: str
    needs: str

    def __str__(self) -> str:
        return f"⛔ {self.item}：{self.reason}（需要：{self.needs}）"


@dataclass
class SceneOutcome:
    """一个场景跑完之后的原始产出。"""

    metrics: dict[str, float] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    blocked: list[Blocked] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    samples: int = 0
    #: 明细行。参照 voice-eval 的 `EvalDialogRoundScore`：
    #: 只有总分而没有逐条明细时，低分无法归因，也就无法改进。
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


class EvalContext(Protocol):
    """场景拿到的运行环境。

    场景**不直接构造 store / embedder / llm** —— 它们由 runner 按
    当前配置（含消融开关）装配后注入。这样同一个场景能在
    「基线」与「实验组」两种配置下跑，而场景代码一行不用改。
    """

    @property
    def config(self) -> Any: ...

    def new_store(self) -> Any:
        """一个新的空存储（每个场景独立，避免互相污染）。"""
        ...

    @property
    def embedder(self) -> Any: ...

    @property
    def llm(self) -> Any: ...

    def new_pet(self, store: Any, *, user_id: str, pet_id: str, name: str) -> Any:
        """造一只猫。返回 PetProfile。"""
        ...


@dataclass(frozen=True)
class Scene:
    """一个可测装置。

    `items` 是 `docs/DESIGN.md` §6.2 里的 E 编号（E1/E6/...）——
    有了它，报告里的每个数字都能追回设计文档的那一条，
    而不是一堆看不出对应关系的指标名。
    """

    scene_id: str
    name: str
    layer: Layer
    items: tuple[str, ...]
    question: str
    run: Callable[[EvalContext], SceneOutcome]
    #: 需要在哪个配置下才有意义（消融用）。None 表示任何配置都跑。
    requires: tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"{self.scene_id}（{'/'.join(self.items)}）"


@dataclass
class SceneResult:
    """场景 + 它这次的产出。"""

    scene: Scene
    outcome: SceneOutcome
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.outcome.failed_checks

    @property
    def score(self) -> float | None:
        """0–1 的通过率。**失败返回 0，跳过返回 None。**

        跳过（无 check 也无 metric）与失败必须可区分：
        把「没测」算成 0 分会让整体分虚低，而算成 100% 会让它虚高。
        """
        checks = self.outcome.checks
        if checks:
            return sum(1 for c in checks if c.passed) / len(checks)
        return None


@dataclass
class SuiteResult:
    """一次配置下的全量运行结果。"""

    config_name: str
    config_notes: tuple[str, ...]
    results: list[SceneResult]
    seed: int

    @property
    def failed(self) -> list[SceneResult]:
        return [r for r in self.results if not r.ok]

    @property
    def errored(self) -> list[SceneResult]:
        return [r for r in self.results if r.error]

    def by_layer(self, layer: Layer) -> list[SceneResult]:
        return [r for r in self.results if r.scene.layer is layer]

    def metric(self, name: str) -> float | None:
        """取某个指标值（跨场景唯一时才返回）。

        同名指标出现在多个场景里意味着命名冲突，那时返回 None 而不是
        随便挑一个 —— 「指标名撞车」会让报告里的数字对不上任何一处代码。
        """
        found = [r.outcome.metrics[name] for r in self.results if name in r.outcome.metrics]
        if len(found) == 1:
            return found[0]
        return None

    def all_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.results:
            for k, v in r.outcome.metrics.items():
                out[f"{r.scene.scene_id}.{k}"] = v
        return out

    def all_blocked(self) -> list[tuple[str, Blocked]]:
        return [
            (r.scene.scene_id, b) for r in self.results for b in r.outcome.blocked
        ]


#: 报告里一行的分档。参照 voice-eval 的 `lowScoreReason`：
#: 只给分数而不给「为什么低」，报告就无法驱动改进。
def grade(score: float | None) -> Literal["优秀", "良好", "偏低", "差", "未测"]:
    if score is None:
        return "未测"
    if score >= 0.95:
        return "优秀"
    if score >= 0.8:
        return "良好"
    if score >= 0.5:
        return "偏低"
    return "差"
