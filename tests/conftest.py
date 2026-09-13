"""pytest 配置：把项目根加入 import 路径，使 `import app.schemas` 可用。

用法：
    /root/.venvs/pet-agent/bin/python -m pytest        # 在项目根执行
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 观测（LangSmith）在测试中**强制关闭**：
#
# 1. 测试必须离线、可复现；
# 2. `TracingConfig.from_env` 的缺省规则是「有 key 就启用」——
#    若跑测的机器上恰好配了 LangSmith key，测试就会真的外发数据。
#    那条路径不会让断言失败，所以是静默的。
# 用**赋值而不是 setdefault**：显式关闭必须能压过环境变量。
os.environ["PET_AGENT_TRACING"] = "0"


# ─────────────────────────────────────────────────────────────
# 先验夹具：**测试构造自己的先验，不读生产数据文件**
# ─────────────────────────────────────────────────────────────
#
# ## 为什么必须解耦
#
# 初版测试直接 `PriorTable.load("data/priors/catmeows_stats.json")`，
# 并断言它「是占位数据」。而那个文件后来被**真实统计替换掉了** ——
# 于是十几个测试一起变红，而它们测的代码一行没改。
#
# 更糟的是：那种耦合会让「数据更新」与「代码回归」在测试结果里长得一样，
# 而两者的处理方式完全不同。
#
# 所以：测试需要什么先验就造什么先验。


def _make_prior(
    *,
    version: str = "test-v1",
    is_placeholder: bool = False,
    contexts: tuple[str, ...] | None = None,
    macro_f1: float | None = 0.80,
    majority_baseline: float | None = 0.50,
    features: tuple[str, ...] | None = None,
    separation: float = 0.15,
) -> Any:
    """构造一个先验表（供 `prior_factory` 夹具调用）。

    Args:
        is_placeholder: 数字是否是编的。
        macro_f1 / majority_baseline: 留出实测结果。
            传 `None` 表示**没做过留出验证** —— 那时 `is_validated` 为 False，
            系统不产出后验概率（fail-closed）。
        features: 每个情境下有哪些特征。传子集可以测「缺特征被跳过」。
        separation: 各情境均值的间距。**默认刻意很小**（0.15）——
            真实先验的分离度就是 0.32–0.72／标准差，后验接近于先验。
            间距太大会让后验直接塌缩到 0.9999，
            那时「置信度校准」「场景加成」这类断言就全都没意义了。

    ⚠️ **默认覆盖全部 6 个 `ContextLabel`。**
    只给 3 个的话，任何引用 `door_attention` / `greeting` 的测试
    都会拿到 `stat() is None` —— 而那是「没有数据」的合法状态，
    不是 bug。测试想测的通常是别的，所以默认给全。
    """
    from app.interpreter.priors import FEATURE_ORDER, GaussianStat, PriorTable
    from app.schemas import ContextLabel

    if contexts is None:
        contexts = tuple(c.value for c in ContextLabel)
    if features is None:
        features = FEATURE_ORDER

    n = len(contexts)
    # base_rates 必须**精确**和为 1.0 —— 用等差权重再补舍入误差
    weights = [(i + 1) / (n * (n + 1) / 2) for i in range(n)]
    base_rates: dict[Any, float] = {}
    for i, c in enumerate(contexts[:-1]):
        base_rates[ContextLabel(c)] = round(weights[i], 6)
    base_rates[ContextLabel(contexts[-1])] = round(
        1.0 - sum(base_rates.values()), 6
    )

    table: dict[ContextLabel, dict[str, GaussianStat]] = {}
    for i, ctx in enumerate(contexts):
        # 各情境均值**必须不同**（否则先验无信息量，测试自己变成假的），
        # 但间距要小（否则后验退化，测不了校准）。
        table[ContextLabel(ctx)] = {
            f: GaussianStat(mean=1.0 + i * separation, std=1.0) for f in features
        }

    return PriorTable(
        version=version,
        provenance="placeholder" if is_placeholder else "catmeows",
        is_placeholder=is_placeholder,
        sha256="0" * 64,
        note="测试夹具",
        base_rates=base_rates,
        contexts=table,
        holdout_macro_f1=macro_f1,
        holdout_majority_baseline=majority_baseline,
        holdout_cats=("CAT01", "CAT02"),
    )


@pytest.fixture()
def prior_factory() -> Callable[..., Any]:
    """先验表工厂。

    用法::

        def test_x(prior_factory):
            prior = prior_factory(is_placeholder=True)
            prior = prior_factory(macro_f1=0.30, majority_baseline=0.50)  # 不具区分度
            prior = prior_factory(macro_f1=None)                          # 未验证
            prior = prior_factory(features=("duration",))                # 缺特征
    """
    return _make_prior
