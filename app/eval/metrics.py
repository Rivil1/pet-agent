"""指标计算。

## 为什么每个指标都带「为什么是这个指标」

一个指标名本身不解释任何东西。`MRR = 0.79` 读起来像成绩单，
但它回答的问题是「用户要的那条记忆排在第几位」——
而这一句才是能驱动改动的信息。

所以每个函数都写清楚它在回答什么、以及**它在什么情况下会误导**。

## 刻意不实现的指标

- **准确率（accuracy）** 单独出现时几乎总是误导：类别不平衡下它会被多数类淹没
  （本项目路由的 `chat` 占大头，全猜 chat 也能有不错的准确率）。
  所以这里只给 per-class F1 与**代价加权错误率**。
- **BLEU / ROUGE** 这类文本重叠指标：本项目的输出是「该不该说这句话」，
  不是「说得像不像参考答案」。用它们会把「诚实地说不知道」判为低分。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Sequence

__all__ = [
    "recall_at_k",
    "mrr",
    "precision_at_k",
    "ConfusionMatrix",
    "cost_weighted_error_rate",
    "expected_calibration_error",
    "percentile",
    "mean",
    "distribution",
]


# =============================================================================
# 检索
# =============================================================================


def recall_at_k(
    retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int
) -> float:
    """召回率：**该召回的有没有被召回**。

    这是检索质量的第一问。它不看排序，只看「在不在前 k 里」。

    误导场景：`k` 越大越容易满分。所以报告里必须连 `k` 一起给，
    且对比时 `k` 必须一致 —— 否则「把 k 从 5 调到 20」
    看起来像一次巨大的质量提升。
    """
    rel = set(relevant)
    if not rel:
        # 没有相关项时召回率无定义。返回 1.0 会让「空查询」看起来完美，
        # 而它其实什么都没证明。
        return 1.0
    if k <= 0:
        return 0.0
    hit = len(rel & set(list(retrieved)[:k]))
    return hit / len(rel)


def precision_at_k(retrieved: Sequence[Hashable], relevant: Iterable[Hashable], k: int) -> float:
    """前 k 里有多少是相关的。**与召回率是一对**：只有召回率会被「多返回」刷高。"""
    if k <= 0:
        return 0.0
    top = list(retrieved)[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for x in top if x in rel) / len(top)


def mrr(retrieved: Sequence[Hashable], relevant: Iterable[Hashable]) -> float:
    """MRR：**排序质量** —— 第一条相关结果排在第几位。

    召回率只看「在不在」，MRR 看「排得靠不靠前」。
    两者一起才能区分两种退化：
    - 召回率掉 → 根本没检索到
    - 召回率不变但 MRR 掉 → 检索到了但排到了后面（上下文被噪声占满）

    误导场景：多相关项时只看第一条。这是定义使然，不是缺陷 ——
    但要知道它衡量的是「第一条相关」而不是「所有相关项的排序」。
    """
    rel = set(relevant)
    if not rel:
        return 1.0
    for rank, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


# =============================================================================
# 分类（意图路由）
# =============================================================================


@dataclass(frozen=True)
class ConfusionMatrix:
    """混淆矩阵 + 由它派生的指标。

    **为什么要看 per-class F1 而不是总准确率**：本项目的路由里
    `chat` 是多数类。一个把所有输入都判成 `chat` 的退化路由器
    能有不错的准确率，而在小类（`record_event`、`profile_update`）上全错 ——
    而恰恰是那些小类错了代价最高（`MISROUTE_COSTS` 里 0.9 的两条
    都在 `chat` 与 `record_event` 之间）。

    macro-F1 给小类与多数类**同样的权重**，所以它不会被多数类淹没。
    """

    labels: tuple[str, ...]
    #: counts[actual][predicted]
    counts: dict[tuple[str, str], int]

    def support(self, label: str) -> int:
        return sum(v for (a, _p), v in self.counts.items() if a == label)

    def predicted(self, label: str) -> int:
        return sum(v for (_a, p), v in self.counts.items() if p == label)

    def true_positive(self, label: str) -> int:
        return self.counts.get((label, label), 0)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def accuracy(self) -> float:
        if not self.total:
            return 0.0
        correct = sum(v for (a, p), v in self.counts.items() if a == p)
        return correct / self.total

    def precision(self, label: str) -> float:
        pred = self.predicted(label)
        return self.true_positive(label) / pred if pred else 0.0

    def recall(self, label: str) -> float:
        sup = self.support(label)
        return self.true_positive(label) / sup if sup else 0.0

    def f1(self, label: str) -> float:
        p, r = self.precision(label), self.recall(label)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def macro_f1(self) -> float:
        """只对**实际出现过**的类别取平均。

        把零样本的类别也算进去会让 macro-F1 无缘无故变低，
        而那个低分不指向任何可修的问题。
        """
        labels = [l for l in self.labels if self.support(l) > 0]
        if not labels:
            return 0.0
        return sum(self.f1(l) for l in labels) / len(labels)

    @property
    def per_class(self) -> dict[str, dict[str, float]]:
        return {
            l: {
                "precision": self.precision(l),
                "recall": self.recall(l),
                "f1": self.f1(l),
                "support": float(self.support(l)),
            }
            for l in self.labels
            if self.support(l) > 0
        }

    def worst_class(self) -> tuple[str, float] | None:
        """F1 最低的类别。**报告里要显式列出它** —— 平均分掩盖的就是它。"""
        per = self.per_class
        if not per:
            return None
        label = min(per, key=lambda k: per[k]["f1"])
        return label, per[label]["f1"]

    def render(self) -> str:
        """渲染成文本表。报告里直接贴，便于一眼看出错在哪。"""
        labels = [l for l in self.labels if self.support(l) > 0 or self.predicted(l) > 0]
        if not labels:
            return "(空)"
        width = max(len(l) for l in labels) + 2
        head = "实际\\预测".ljust(width) + "".join(l.rjust(10) for l in labels) + "召回".rjust(9)
        lines = [head, "-" * len(head)]
        for a in labels:
            row = a.ljust(width)
            for p in labels:
                n = self.counts.get((a, p), 0)
                row += (f"{n}" if n else "·").rjust(10)
            row += f"{self.recall(a):.2f}".rjust(9)
            lines.append(row)
        lines.append("-" * len(head))
        row = "精确率".ljust(width)
        for p in labels:
            row += f"{self.precision(p):.2f}".rjust(10)
        lines.append(row)
        return "\n".join(lines)


def build_confusion(
    pairs: Iterable[tuple[str, str]], labels: Sequence[str]
) -> ConfusionMatrix:
    """`(actual, predicted)` 序列 → 混淆矩阵。"""
    counts = Counter(pairs)
    return ConfusionMatrix(labels=tuple(labels), counts=dict(counts))


def cost_weighted_error_rate(
    pairs: Iterable[tuple[str, str]],
    costs: Iterable[Any],
) -> tuple[float, list[tuple[str, str, float, bool]], list[tuple[str, str]]]:
    """**代价加权错误率** —— 本项目路由的真实业务指标。

    ## 为什么不用「准确率」

    误判的代价**极不对称**（`MISROUTE_COSTS` 从 0.15 到 0.9）：

    | 误判 | 代价 | 是否静默 |
    |---|---|---|
    | `chat` ↔ `record_event` | 0.9 | **静默** |
    | `chat` → `profile_update` | 0.7 | 会被发现 |
    | `profile_update` → `chat` | 0.4 | 会被拦下 |

    把 `chat` 判成 `record_event`（闲聊被写成长期记忆）与
    把 `profile_update` 判成 `chat`（照片没处理，用户立刻重传）
    在准确率里是**同等的错**，在业务上差了 2 倍多 ——
    而且前者是静默的：用户以为记住了，实际污染了后续所有检索。

    ## ⚠️ 矩阵里没有的转移**不是**错误

    初版把未知转移默认成 `cost=1.0`，于是「X → ambiguous」
    被算成最贵的错。而矩阵里**只有** `ambiguous ← chat` 一条 ——
    这是刻意的：**反问是安全兜底，不构成误判**。

    把「找不到条目」当成「代价很高」，会让一个保守的路由器
    看起来比一个乱猜的路由器还差 —— 而那正好把设计意图反过来读。

    所以未定义转移单独返回，由调用方决定怎么呈现。
    它本身也是一条发现：矩阵没覆盖「过度澄清」这种代价形态。

    Returns:
        `(平均每次交互的代价, [(actual, predicted, cost, is_silent), ...], [未定义转移, ...])`

        **分母是样本数，而不是「该类的最坏情况代价」。**

        初版用后者，结果每一类内部的错误都被归一成 1.0 ——
        于是「闲聊被写成长期记忆」（代价 0.9）与「照片没处理」
        （代价 0.4）算出**同一个数**。那等于把要度量的东西又抹平了。

        用「每次交互的平均代价」的好处：它是**可加的**，
        可直接解读为「平均每一次交互付多少误判代价」，
        而且不同类之间可比。
    """
    cost_map: dict[tuple[str, str], tuple[float, bool]] = {}
    for c in costs:
        a = getattr(c, "actual", None)
        p = getattr(c, "predicted", None)
        if a is None or p is None:
            continue
        av = getattr(a, "value", a)
        pv = getattr(p, "value", p)
        cost_map[(av, pv)] = (
            float(getattr(c, "cost", 0.0)),
            bool(getattr(c, "is_silent", False)),
        )

    total_cost = 0.0
    n = 0
    errors: list[tuple[str, str, float, bool]] = []
    undefined: list[tuple[str, str]] = []

    for actual, predicted in pairs:
        n += 1
        if actual == predicted:
            continue
        entry = cost_map.get((actual, predicted))
        if entry is None:
            undefined.append((actual, predicted))
            continue
        cost, silent = entry
        total_cost += cost
        errors.append((actual, predicted, cost, silent))

    if n <= 0:
        return 0.0, errors, undefined
    return total_cost / n, errors, undefined


# =============================================================================
# 校准
# =============================================================================


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = 10
) -> tuple[float, list[tuple[float, float, int]]]:
    """ECE：**说「70% 可能」的时候，是不是真的七成对**。

    ## 为什么它对这套系统特别重要

    行为解释会输出「等吃的 62%」这样的数字。如果实际只有 30% 命中，
    那个 62% 就是**编造的精确** —— 比不给数字更糟，因为它看起来有依据。

    `docs/DESIGN.md` §6.2 的 E23 把它列为待测项，而它一直没被测过。

    Returns:
        `(ECE, [(bins 的平均置信度, 该桶准确率, 样本数), ...])`
    """
    n = len(confidences)
    if n == 0 or n != len(correct):
        return 0.0, []

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for conf, ok in zip(confidences, correct):
        idx = min(bins - 1, max(0, int(conf * bins)))
        buckets[idx].append((conf, ok))

    ece = 0.0
    rows: list[tuple[float, float, int]] = []
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - avg_conf)
        rows.append((avg_conf, acc, len(bucket)))
    return ece, rows


# =============================================================================
# 分布
# =============================================================================


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数。

    p95 延迟用 `max()` 代替会失真：一次冷启动会把 p95 拉成极端值，
    而它不代表用户的常态体验。
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"分位数必须在 0–100，得到 {p}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def distribution(values: Sequence[float]) -> dict[str, float]:
    """给报告用的分布摘要。**p50 与 p95 一起给** —— 只看均值会掩盖长尾。"""
    if not values:
        return {"n": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "n": float(len(values)),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": float(max(values)),
    }
