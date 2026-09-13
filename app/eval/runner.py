"""评测执行器：**跑场景、收结果、算基线差值**。

对应 voice-eval 的 `EvalTask`（一次运行）与 `EvalTaskResult`（汇总）。

## 三条执行纪律

1. **场景之间必须隔离。** 每个场景拿自己的 store；
   否则前一个场景写下的记忆会污染后一个，而那种污染是静默的 ——
   数字莫名其妙地偏离，没有任何报错。

2. **场景崩溃不能带走整轮评测。** 一个场景抛异常时记为 `error` 并继续，
   否则「有一个场景挂了」会表现成「评测跑不出来」。

3. **基线是必需的。** `run_suite` 之外必须能跑 `compare`
   —— 没有差值的数字不说明任何事。
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from app.eval.config import BASELINE, EvalConfig
from app.eval.runtime import Runtime, make_runtime
from app.eval.scenes import ALL_SCENES
from app.eval.types import Layer, Scene, SceneResult, SuiteResult

__all__ = ["run_suite", "run_all_configs", "MetricDelta", "compare"]


def run_suite(
    config: EvalConfig = BASELINE,
    *,
    scenes: Sequence[Scene] | None = None,
    only: Iterable[str] | None = None,
    verbose: bool = False,
) -> SuiteResult:
    """跑一轮评测。

    Args:
        config: 消融配置。同一次运行内不可变。
        scenes: 场景集合。缺省用 `ALL_SCENES`。
        only: 只跑这些 scene_id（调试用）。
        verbose: 打印每个场景的进度。
    """
    selected = list(scenes if scenes is not None else ALL_SCENES)
    if only is not None:
        wanted = set(only)
        selected = [s for s in selected if s.scene_id in wanted]

    runtime: Runtime = make_runtime(config)
    results: list[SceneResult] = []

    try:
        for scene in selected:
            if verbose:
                print(f"  ▸ {scene.scene_id} …", end="", flush=True)
            try:
                # 每个场景独立运行时上下文（store 由场景自己 new）
                outcome = scene.run(runtime)
                result = SceneResult(scene=scene, outcome=outcome)
            except Exception as exc:  # noqa: BLE001 - 一个场景挂了不该带走整轮
                result = SceneResult(
                    scene=scene,
                    outcome=_empty_outcome(),
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                )
            results.append(result)

            if verbose:
                mark = "✅" if result.ok else ("💥" if result.error else "❌")
                print(f" {mark}")
    finally:
        # 收尾：清掉本次运行写进真实存储的数据。
        #
        # **必须在 finally 里** —— 中途抛异常时留下的数据会在下一次运行时
        # 被当成「泄漏」或「重复」，而那两个报错都指向错误的地方。
        runtime.cleanup()

    return SuiteResult(
        config_name=config.name,
        config_notes=config.notes,
        results=results,
        seed=config.seed,
    )


def _empty_outcome() -> Any:
    from app.eval.types import SceneOutcome

    return SceneOutcome()


# =============================================================================
# 基线对比
# =============================================================================


@dataclass(frozen=True)
class MetricDelta:
    """一个指标在两组配置间的差。"""

    scene_id: str
    metric: str
    baseline: float
    experiment: float

    @property
    def delta(self) -> float:
        return self.experiment - self.baseline

    @property
    def rel(self) -> float | None:
        """相对变化。基数为 0 时无定义 —— 返回 None 而不是 `inf`。

        `inf` 会在报告里显示成一个看起来像成绩的符号，而它其实只是
        「除数恰好是 0」。
        """
        if self.baseline == 0:
            return None
        return self.delta / abs(self.baseline)

    @property
    def mark(self) -> str:
        """方向标记。

        ⚠️ **「增加」不总是好**：`无据陈述率`、`泄漏条数`、`误判冲突数`
        这些指标下降才是改进。所以这里只标方向，好坏由人判断 ——
        让代码去猜「哪个方向是好」迟早会把某个指标读反。
        """
        if abs(self.delta) < 1e-9:
            return "="
        return "↑" if self.delta > 0 else "↓"


@dataclass
class Comparison:
    """基线 vs 实验组。"""

    baseline: SuiteResult
    experiment: SuiteResult
    deltas: list[MetricDelta]
    #: 只在一边出现的指标（说明某个配置下场景没产出它）
    only_in: list[tuple[str, str, str]]

    @property
    def config_note(self) -> str:
        return "；".join(self.experiment.config_notes) or "（未说明改了什么）"

    def changed(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.mark != "="]

    def biggest(self, n: int = 5) -> list[MetricDelta]:
        """变化最大的几项。**按相对幅度排**，否则大数量级指标会霸榜。"""
        scored = [d for d in self.changed() if d.rel is not None]
        scored.sort(key=lambda d: abs(d.rel or 0), reverse=True)
        return scored[:n]


def compare(baseline: SuiteResult, experiment: SuiteResult) -> Comparison:
    """两组结果求差。**按 (scene_id, metric) 配对**，不按顺序。"""
    base = baseline.all_metrics()
    exp = experiment.all_metrics()

    deltas: list[MetricDelta] = []
    only_in: list[tuple[str, str, str]] = []

    for key, b_val in base.items():
        scene_id, _, metric = key.partition(".")
        if key in exp:
            deltas.append(
                MetricDelta(
                    scene_id=scene_id,
                    metric=metric,
                    baseline=b_val,
                    experiment=exp[key],
                )
            )
        else:
            only_in.append((scene_id, metric, baseline.config_name))

    for key in exp:
        if key not in base:
            scene_id, _, metric = key.partition(".")
            only_in.append((scene_id, metric, experiment.config_name))

    deltas.sort(key=lambda d: (d.scene_id, d.metric))
    return Comparison(
        baseline=baseline, experiment=experiment, deltas=deltas, only_in=only_in
    )


def run_all_configs(
    configs: Sequence[EvalConfig] | None = None,
    *,
    scenes: Sequence[Scene] | None = None,
    verbose: bool = False,
) -> dict[str, SuiteResult]:
    """跑多组配置。返回 `{配置名: 结果}`。

    内存后端之外的配置需要额外环境（如 MySQL），
    失败时**记为错误而不中断** —— 一个可选的消融组跑不了，
    不该让整轮评测无法产出。
    """
    from app.eval.config import PRESETS

    chosen = list(configs) if configs is not None else [PRESETS["baseline"], *(
        c for name, c in PRESETS.items() if name != "baseline"
    )]

    out: dict[str, SuiteResult] = {}
    for cfg in chosen:
        if verbose:
            print(f"\n▶ 配置 {cfg.name}  —— {cfg.describe()}")
        out[cfg.name] = run_suite(cfg, scenes=scenes, verbose=verbose)
    return out


def layer_summary(suite: SuiteResult) -> dict[Layer, dict[str, float]]:
    """按可判定性分层汇总。**报告里必须分层呈现**（DESIGN §6.1）。

    把 A 层（确定性）与 C 层（主观）混在一个总分里，
    等于给后者披上前者的可信度。
    """
    summary: dict[Layer, dict[str, float]] = {}
    for layer in Layer:
        rows = suite.by_layer(layer)
        if not rows:
            continue
        scored = [r.score for r in rows if r.score is not None]
        summary[layer] = {
            "场景数": float(len(rows)),
            "已评分场景数": float(len(scored)),
            "平均通过率": sum(scored) / len(scored) if scored else 0.0,
            "失败场景数": float(sum(1 for r in rows if not r.ok)),
            "不可测项数": float(sum(len(r.outcome.blocked) for r in rows)),
        }
    return summary
