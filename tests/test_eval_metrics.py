"""评测指标的正确性。

## 为什么指标函数要单独测

指标算错**不会报错**。它会给出一个看起来合理的数字，被写进报告、
被引用、被用来做决策 —— 而没有任何环节能发现它是错的。

尤其是这三处，本文件重点覆盖：

1. **`macro-F1` 对零样本类的处理**：把从未出现的类算进去会无端拉低分数，
   而那个低分不指向任何可修的问题。
2. **`cost_weighted_error_rate` 对未定义转移的处理**：
   把它们当「最贵的错」会让一个保守的路由器看起来比乱猜的还差 ——
   正好把设计意图读反。
3. **`ECE` 的分桶**：分桶错了会让「过度自信」看起来像「校准良好」。
"""

from __future__ import annotations

import pytest

from app.eval.metrics import (
    ConfusionMatrix,
    build_confusion,
    cost_weighted_error_rate,
    distribution,
    expected_calibration_error,
    mean,
    mrr,
    percentile,
    precision_at_k,
    recall_at_k,
)


class TestRetrieval:
    def test_recall_counts_relevant_in_top_k(self):
        assert recall_at_k(["a", "b", "c"], ["a", "c"], k=3) == 1.0
        assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5

    def test_recall_respects_k(self):
        """`k` 越大越容易满分 —— 所以报告里必须连 k 一起给。"""
        assert recall_at_k(["x", "a"], ["a"], k=1) == 0.0
        assert recall_at_k(["x", "a"], ["a"], k=2) == 1.0

    def test_recall_with_no_relevant_items_is_one(self):
        """无相关项时召回率无定义。

        返回 1.0 会让「空查询」看起来完美，但返回 0.0 会让
        「这个查询本来就没有正确答案」看起来像检索失败。
        选 1.0 并把这类样本从报告的分母里排除是调用方的责任。
        """
        assert recall_at_k(["a"], [], k=5) == 1.0

    def test_recall_k_zero(self):
        assert recall_at_k(["a"], ["a"], k=0) == 0.0

    def test_precision_penalises_padding(self):
        """只有召回率会被「多返回」刷高 —— 精确率不会。"""
        assert recall_at_k(["a", "x", "y", "z"], ["a"], k=4) == 1.0
        assert precision_at_k(["a", "x", "y", "z"], ["a"], k=4) == 0.25

    def test_mrr_uses_first_relevant_rank(self):
        assert mrr(["x", "a", "b"], ["a", "b"]) == pytest.approx(0.5)
        assert mrr(["a", "x"], ["a"]) == 1.0
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_mrr_distinguishes_ranking_from_recall(self):
        """**召回率不变而 MRR 下降** = 检索到了但被噪声挤到后面。

        这正是 `naive-retrieval` 消融组要暴露的失效模式，
        所以两个指标必须能分开动。
        """
        relevant = ["a"]
        assert recall_at_k(["x", "y", "a"], relevant, k=3) == 1.0
        assert mrr(["x", "y", "a"], relevant) == pytest.approx(1 / 3)
        assert mrr(["a", "y", "x"], relevant) == 1.0


class TestConfusionMatrix:
    def test_per_class_f1(self):
        m = build_confusion(
            [("a", "a"), ("a", "b"), ("b", "b")], labels=["a", "b"]
        )
        # a: 1 TP, 1 FP(b→a=0, a→b=1 所以 predicted a 只有 1) → precision 1.0
        assert m.precision("a") == 1.0
        assert m.recall("a") == 0.5
        assert m.f1("a") == pytest.approx(2 / 3)

    def test_macro_f1_ignores_absent_labels(self):
        """零样本类不参与平均。

        把从未出现的类算进去（F1=0）会无端拉低 macro-F1，
        而那个低分**不指向任何可修的问题** —— 读的人会去查一个
        根本不存在的缺陷。
        """
        only_a = build_confusion([("a", "a"), ("a", "a")], labels=["a", "b", "c"])
        assert only_a.per_class.keys() == {"a"}
        assert only_a.macro_f1 == 1.0, "只有 a 且全对 → macro-F1 应为 1.0"

    def test_accuracy_hides_minority_failure(self):
        """**准确率会掩盖小类全错。** 这条测试就是把它固定成事实。"""
        pairs = [("chat", "chat")] * 90 + [("record_event", "chat")] * 10
        m = build_confusion(pairs, labels=["chat", "record_event"])
        assert m.accuracy == pytest.approx(0.9), "准确率看起来很好"
        assert m.recall("record_event") == 0.0, "而小类全错"
        assert m.macro_f1 < 0.55, "macro-F1 才暴露了问题"

    def test_worst_class_points_at_the_problem(self):
        pairs = [("chat", "chat")] * 10 + [("record_event", "chat")] * 4
        m = build_confusion(pairs, labels=["chat", "record_event"])
        worst = m.worst_class()
        assert worst is not None
        assert worst[0] == "record_event"
        assert worst[1] == 0.0

    def test_render_is_readable(self):
        m = build_confusion([("a", "a"), ("b", "a")], labels=["a", "b"])
        text = m.render()
        assert "实际\\预测" in text
        assert "精确率" in text

    def test_empty_matrix(self):
        m = build_confusion([], labels=["a"])
        assert m.accuracy == 0.0
        assert m.macro_f1 == 0.0
        assert m.worst_class() is None


class TestCostWeightedError:
    """**本项目路由的核心指标。**"""

    @staticmethod
    def _costs():
        from app.schemas import MISROUTE_COSTS

        return MISROUTE_COSTS

    def test_correct_predictions_cost_nothing(self):
        rate, errors, undefined = cost_weighted_error_rate(
            [("chat", "chat"), ("record_event", "record_event")], self._costs()
        )
        assert rate == 0.0
        assert errors == []
        assert undefined == []

    def test_undefined_transition_is_not_an_error(self):
        """**这是本文件最重要的一条断言。**

        代价矩阵里**没有**「X → ambiguous」的条目，那是刻意的：
        反问是安全兜底，不构成误判。

        初版把未定义转移默认成 `cost=1.0`，于是「宁可多问一句」
        的设计意图被读成了「最严重的错误」—— 一个保守的路由器
        会比一个乱猜的路由器得分更差。
        """
        rate, errors, undefined = cost_weighted_error_rate(
            [("record_event", "ambiguous")], self._costs()
        )
        assert errors == [], "→ambiguous 不应被算成有代价的错误"
        assert undefined == [("record_event", "ambiguous")], "但要单独报出来"
        assert rate == 0.0, "未定义转移不贡献加权错误率"

    def test_defined_error_is_weighted(self):
        """`chat ↔ record_event` 代价 0.9，且是**静默**的。"""
        rate, errors, undefined = cost_weighted_error_rate(
            [("record_event", "chat")], self._costs()
        )
        assert len(errors) == 1
        actual, predicted, cost, silent = errors[0]
        assert (actual, predicted) == ("record_event", "chat")
        assert cost == pytest.approx(0.9)
        assert silent is True, "这条错是静默的 —— 用户以为记住了，实际没写"
        assert rate == pytest.approx(0.9)

    def test_asymmetric_costs_matter(self):
        """同一个「错」，代价可以差 2 倍多 —— 这就是不用准确率的原因。"""
        high, _, _ = cost_weighted_error_rate(
            [("record_event", "chat")], self._costs()
        )
        low, _, _ = cost_weighted_error_rate(
            [("profile_update", "chat")], self._costs()
        )
        assert high > low

    def test_silent_errors_are_separable(self):
        _, errors, _ = cost_weighted_error_rate(
            [("record_event", "chat"), ("chat", "profile_update")], self._costs()
        )
        silent = [e for e in errors if e[3]]
        assert len(silent) == 1, "只有 chat↔record_event 那一对是静默的"


class TestCalibration:
    def test_perfect_calibration_is_zero(self):
        """说 70% 就真的七成对 → ECE = 0。"""
        ece, rows = expected_calibration_error(
            [0.9] * 10 + [0.1] * 10, [True] * 9 + [False] + [True] + [False] * 9
        )
        assert ece < 0.15, f"ECE={ece} 应接近 0"

    def test_overconfidence_shows_up(self):
        """**说 90% 但只有一半对** → ECE 高。

        这条对应 E23：行为解释会输出「等吃的 62%」，
        如果实际只有 30% 命中，那个 62% 就是**编造的精确** ——
        比不给数字更糟，因为它看起来有依据。
        """
        ece, _ = expected_calibration_error([0.9] * 10, [True] * 5 + [False] * 5)
        assert ece == pytest.approx(0.4, abs=0.05), "置信 0.9 实际 0.5 → ECE≈0.4"

    def test_underconfidence_shows_up(self):
        ece, _ = expected_calibration_error([0.1] * 10, [True] * 5 + [False] * 5)
        assert ece == pytest.approx(0.4, abs=0.05)

    def test_empty_input(self):
        assert expected_calibration_error([], []) == (0.0, [])

    def test_mismatched_lengths_is_zero_not_crash(self):
        """长度不匹配是调用方的 bug，但不该让整轮评测崩掉。"""
        assert expected_calibration_error([0.5], []) == (0.0, [])


class TestDistribution:
    def test_percentile_linear_interpolation(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3
        assert percentile([1, 2, 3, 4, 5], 0) == 1
        assert percentile([1, 2, 3, 4, 5], 100) == 5
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_percentile_single_value(self):
        assert percentile([7], 95) == 7

    def test_percentile_empty(self):
        assert percentile([], 95) == 0.0

    def test_percentile_rejects_bad_p(self):
        with pytest.raises(ValueError):
            percentile([1, 2], 150)

    def test_p95_differs_from_max_on_long_tail(self):
        """**只看均值（或只看 max）都会误导。**

        p95 与 max 分开报：max 会被一次冷启动拉到极端，
        而它不代表用户的常态体验。
        """
        values = [10.0] * 95 + [500.0] * 5
        d = distribution(values)
        assert d["p50"] == pytest.approx(10.0)
        assert d["p95"] < 500.0, "p95 不应等于 max"
        assert d["max"] == 500.0

    def test_mean_of_empty(self):
        assert mean([]) == 0.0

    def test_distribution_of_empty(self):
        d = distribution([])
        assert d["n"] == 0
        assert d["p95"] == 0.0
