"""评测配置 —— 基线与被消融的对象。

## 为什么消融必须是**配置开关**而不是另一份代码

`docs/DESIGN.md` §6.3 明确要求：

> **消融须用配置开关**（`ABLATION_NO_GUARD=1`），同一套代码跑不同配置，
> 而不是维护两份代码 —— 这样消融才是可复现的。

维护两份代码的消融是不可信的：你无法确定两份代码的差异**只有**那一处。
配置开关则保证了差异被限定在一个可以逐行审阅的地方。

## 每个开关都要说清「它证明了什么」

一个开关不是为了「多跑一组数」。它要回答一个具体的主张：

| 开关 | 它要证明的主张 |
|---|---|
| `retrieval_mode=naive` | 混合检索比朴素向量 top-k 好 |
| `use_memory=False` | 记忆系统真的有用（而不是聊胜于无） |
| `apply_guard=False` | 守卫真的拦住了东西（而不是摆设） |
| `store_backend=memory` | 存储后端不影响正确性（只影响持久性） |

**如果消融跑出来两组数字一样，那也是一个结论** ——
它说明那个机制在当前评测集上没有可测量的贡献。
这比「我们做了防护」这句话有价值得多，也更诚实。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

__all__ = [
    "EvalConfig",
    "BASELINE",
    "ABLATIONS",
    "PRESETS",
    "COMPARISON_PAIRS",
]

RetrievalMode = Literal["hybrid", "naive"]
StoreBackend = Literal["memory", "mysql"]
LLMProfile = Literal["neutral", "overconfident"]


@dataclass(frozen=True)
class EvalConfig:
    """一次评测运行的配置。

    frozen 是刻意的：配置在运行期被改动会让「这份报告对应哪套配置」
    变得说不清，而那是消融结论成立的前提。
    """

    name: str = "baseline"

    #: `hybrid` = 向量召回 + 预过滤 + 重排序 + MMR 多样性裁剪
    #: `naive`  = 纯向量 top-k（无重排序、无 MMR）
    retrieval_mode: RetrievalMode = "hybrid"

    #: 关闭时完全不检索（相当于「无记忆」基线）
    use_memory: bool = True

    #: 关闭时**不应用守卫**，直接暴露未被拦下的输出。
    #: 这测量的正是「守卫拦住了多少本该拦下的东西」。
    apply_guard: bool = True

    #: **故障注入**：让 LLM 输出无依据的确定回答。
    #:
    #: 为什么需要它：`MockLLM` 输出固定文案（「喵。」），
    #: 它**永远不会**产生需要守卫拦下的内容 —— 于是
    #: `apply_guard=False` 跑出来与基线完全一致，
    #: 而那会得出「守卫没有价值」这个**反向的**结论。
    #:
    #: `overconfident` 模拟的是真实模型最常见的失败模式：
    #: 在没有依据时仍然给出听起来确定的回答。
    #: 于是「有/无守卫」的差值才真正度量了守卫的价值。
    llm_profile: LLMProfile = "neutral"

    #: 存储后端。用于验证「正确性不依赖后端」——
    #: 两个后端跑出不同结果说明有实现漂移。
    store_backend: StoreBackend = "memory"

    #: 固定随机种子。**可复现的前提**：没有它，两次运行的数字不可比，
    #: 而「这次比上次好」就成了噪声。
    seed: int = 20260913

    #: 报告里显式声明这套配置动了什么。
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        if not self.notes:
            return f"{self.name}（未改动）"
        return f"{self.name}：" + "；".join(self.notes)

    @property
    def is_baseline(self) -> bool:
        return (
            self.retrieval_mode == "hybrid"
            and self.use_memory
            and self.apply_guard
        )


#: 基线：全部机制开启。**所有对比都以它为分母。**
BASELINE = EvalConfig(name="baseline", notes=("全部机制开启",))


def _ablations() -> dict[str, EvalConfig]:
    return {
        "naive-retrieval": replace(
            BASELINE,
            name="naive-retrieval",
            retrieval_mode="naive",
            notes=(
                "去掉重排序与 MMR 多样性裁剪，改为纯向量 top-k",
                "预期：recall 可能不变，但 MRR 下降（相关项被噪声挤到后面）",
            ),
        ),
        "no-memory": replace(
            BASELINE,
            name="no-memory",
            use_memory=False,
            notes=(
                "完全不检索记忆，相当于「只有当前轮」的对话系统",
                "预期：记忆类问题的无据陈述率显著上升",
            ),
        ),
        "no-guard": replace(
            BASELINE,
            name="no-guard",
            apply_guard=False,
            notes=(
                "不应用输出守卫，直接采用模型的原始输出",
                "预期：诚实性违规率 > 0 —— 这个数字就是守卫的价值",
            ),
        ),
        "mysql-store": replace(
            BASELINE,
            name="mysql-store",
            store_backend="mysql",
            notes=(
                "换用 MySQL 存储后端（需 MYSQL_* 已配置，否则整组崩溃）",
                "预期：与 memory 后端**完全一致** —— 不一致说明有实现漂移",
            ),
        ),
        # ── 守卫的价值：一对故障注入对照 ──
        #
        # 为什么需要**两个**配置：单一组的数字不能说明守卫有没有用。
        # 只有「同一个会编造的 LLM」在开/关守卫两种情况下的差值，
        # 才是守卫的贡献。
        "faulty-llm-guarded": replace(
            BASELINE,
            name="faulty-llm-guarded",
            llm_profile="overconfident",
            notes=(
                "故障注入：LLM 输出无依据的确定回答；**守卫开启**",
                "预期：无据陈述率仍为 0（守卫把它改写成保守回答）",
            ),
        ),
        "faulty-llm-unguarded": replace(
            BASELINE,
            name="faulty-llm-unguarded",
            llm_profile="overconfident",
            apply_guard=False,
            notes=(
                "故障注入：同一个会编造的 LLM；**守卫关闭**",
                "预期：无据陈述率 > 0 —— 这个数字就是守卫的价值",
            ),
        ),
    }


#: 报告里值得单独拿出来对比的配对。
#:
#: 不是「基线与每个实验组」—— 有些对比只有在特定两组之间才有意义。
#: 例如守卫的价值不在 `baseline → no-guard`（中性 LLM 下守卫从不触发，
#: 两组都是 0），而在 `faulty-llm-guarded → faulty-llm-unguarded`。
COMPARISON_PAIRS: tuple[tuple[str, str], ...] = (
    ("baseline", "naive-retrieval"),
    ("baseline", "no-memory"),
    ("baseline", "no-guard"),
    ("baseline", "mysql-store"),
    ("faulty-llm-guarded", "faulty-llm-unguarded"),
)


ABLATIONS: dict[str, EvalConfig] = _ablations()

#: 名字 → 配置。CLI 用 `--config` 引用。
PRESETS: dict[str, EvalConfig] = {"baseline": BASELINE, **ABLATIONS}
