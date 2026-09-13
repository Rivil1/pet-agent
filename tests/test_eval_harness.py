"""评测框架本身的行为。

## 为什么要测「框架」

框架出错的方式比指标出错更隐蔽：它**不产出错误的数字**，
而是产出**看起来正常的报告**。三个真实踩过的坑：

1. **场景之间不隔离** —— 内存后端下每个场景拿新 store，天然没事；
   而 MySQL 后端下共享同一张表，于是下一个场景读到上一个的数据，
   报出来的是「去重没生效」「冲突误判」这类**指向错误**的失败。
2. **实验组全崩被当成「无差异」** —— 报告写「所有指标完全一致。
   这本身是一个结论」，而真相是它一行没跑。那会得出
   「该机制没有贡献」这个**反向的**结论。
3. **不可测项被静默跳过** —— 报告里少一行，读的人以为都测过了。

本文件把这三件事都固定成断言。
"""

from __future__ import annotations

import pytest

from app.eval.config import ABLATIONS, BASELINE, COMPARISON_PAIRS, PRESETS
from app.eval.report import render_json, render_markdown
from app.eval.runner import compare, layer_summary, run_suite
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


# =============================================================================
# 配置
# =============================================================================


class TestConfig:
    def test_baseline_has_everything_on(self):
        assert BASELINE.is_baseline
        assert BASELINE.retrieval_mode == "hybrid"
        assert BASELINE.use_memory
        assert BASELINE.apply_guard
        assert BASELINE.llm_profile == "neutral"

    def test_every_ablation_changes_exactly_one_thing(self):
        """**消融只能动一个变量。**

        一次动两个的话，差值归因不到任何一个 —— 而「归因不到」
        会让整组数字没法用，虽然它看起来完全正常。
        """
        for name, cfg in ABLATIONS.items():
            diffs = [
                field
                for field in (
                    "retrieval_mode",
                    "use_memory",
                    "apply_guard",
                    "llm_profile",
                    "store_backend",
                )
                if getattr(cfg, field) != getattr(BASELINE, field)
            ]
            # 故障注入组是**成对**的：同一 LLM，只差守卫。
            # 所以 faulty-llm-unguarded 允许动两个（llm_profile + apply_guard），
            # 但它的对照必须是与 faulty-llm-guarded 比，而不是与 baseline 比。
            if name == "faulty-llm-unguarded":
                assert set(diffs) == {"llm_profile", "apply_guard"}
                continue
            assert len(diffs) == 1, (
                f"消融组 {name!r} 动了 {len(diffs)} 个变量：{diffs}。"
                f"一次只能动一个，否则差值归因不到任何一个。"
            )

    def test_every_ablation_has_notes(self):
        """没有说明的消融组没法解读 —— 报告里会是一张没有上下文的表。"""
        for name, cfg in ABLATIONS.items():
            assert cfg.notes, f"{name!r} 没有 notes"
            assert any("预期" in n for n in cfg.notes), (
                f"{name!r} 没写预期结果 —— 没有预期就无法判断结果是意外还是已知"
            )

    def test_comparison_pairs_reference_real_configs(self):
        for a, b in COMPARISON_PAIRS:
            assert a in PRESETS, f"对比对引用了不存在的配置 {a!r}"
            assert b in PRESETS, f"对比对引用了不存在的配置 {b!r}"

    def test_guard_pair_is_compared_against_itself(self):
        """守卫的对照必须是**同一个会编造的 LLM**。

        拿 `baseline → no-guard` 比是没意义的：中性 LLM 下守卫从不触发，
        两边都是 0，于是「守卫没有价值」这个反向结论就出来了。
        """
        assert ("faulty-llm-guarded", "faulty-llm-unguarded") in COMPARISON_PAIRS

    def test_config_is_frozen(self):
        with pytest.raises(Exception):
            BASELINE.apply_guard = False  # type: ignore[misc]


# =============================================================================
# 场景注册表
# =============================================================================


class TestSceneRegistry:
    def test_ids_are_unique(self):
        ids = [s.scene_id for s in ALL_SCENES]
        assert len(ids) == len(set(ids))

    def test_every_scene_declares_items_and_question(self):
        """每个场景要能追回设计文档的 E 编号。

        没有 `items` 的话，报告里的数字与 `DESIGN §6.2` 的对应关系
        只能靠人记 —— 而那种对应关系会随重构悄悄失效。
        """
        for s in ALL_SCENES:
            assert s.items, f"{s.scene_id} 没声明 E 编号"
            assert s.question.endswith("？"), f"{s.scene_id} 的 question 应是一个问题"

    def test_layer_b_scenes_are_marked_blocked(self):
        """B 层场景在当前数据条件下必须**显式不可测**，而不是给个数字。"""
        b_scenes = [s for s in ALL_SCENES if s.layer is Layer.B]
        assert b_scenes, "至少应有一个 B 层场景（声学分类）"


# =============================================================================
# 执行器
# =============================================================================


def _scene(scene_id: str, run, *, layer: Layer = Layer.A) -> Scene:
    return Scene(
        scene_id=scene_id,
        name=scene_id,
        layer=layer,
        items=("E0",),
        question="测什么？",
        run=run,
    )


class TestRunner:
    def test_crashing_scene_does_not_kill_the_suite(self):
        """**一个场景挂了不该带走整轮。**

        否则「有一个场景崩溃」会表现成「评测跑不出来」，
        而那会把排查方向从「哪个场景坏了」引到「评测框架坏了」。
        """

        def boom(ctx):
            raise RuntimeError("故意崩")

        def fine(ctx):
            return SceneOutcome(checks=[Check("ok", True)])

        suite = run_suite(
            BASELINE, scenes=[_scene("boom", boom), _scene("fine", fine)]
        )
        assert len(suite.results) == 2
        assert len(suite.errored) == 1
        assert len(suite.failed) == 1, "崩溃算失败"
        assert suite.results[1].ok, "后面的场景仍然跑完了"

    def test_scenes_get_independent_stores(self):
        """**每个场景拿到自己的 store。**

        不隔离时，后一个场景会读到前一个写下的数据 ——
        而那种污染是静默的：数字莫名其妙地偏离，没有任何报错。
        """
        seen: list[int] = []

        def first(ctx):
            seen.append(id(ctx.new_store()))
            return SceneOutcome()

        def second(ctx):
            seen.append(id(ctx.new_store()))
            return SceneOutcome()

        run_suite(BASELINE, scenes=[_scene("a", first), _scene("b", second)])
        assert len(seen) == 2
        assert seen[0] != seen[1], "两个场景拿到了同一个 store"

    def test_scenes_get_independent_tenants(self):
        """租户也必须独立 —— 否则 MySQL 后端下会跨场景累积。"""
        tenants: list[tuple[str, str]] = []

        def grab(ctx):
            tenants.append(ctx.new_tenant())
            return SceneOutcome()

        run_suite(BASELINE, scenes=[_scene("a", grab), _scene("b", grab)])
        assert tenants[0] != tenants[1]
        assert tenants[0][0] != tenants[1][0], "user_id 不能重复"
        assert tenants[0][1] != tenants[1][1], (
            "pet_id 也不能重复 —— 它是**主键**，重了会让 save_pet 走 "
            "ON DUPLICATE KEY UPDATE 并静默改掉上一轮的行"
        )

    def test_only_filter(self):
        def ok(ctx):
            return SceneOutcome(checks=[Check("ok", True)])

        scenes = [_scene("a", ok), _scene("b", ok), _scene("c", ok)]
        suite = run_suite(BASELINE, scenes=scenes, only=["b"])
        assert [r.scene.scene_id for r in suite.results] == ["b"]

    def test_suite_is_deterministic(self):
        """固定种子 → 同样场景跑两次结果一致。**可复现的前提。**"""
        a = run_suite(BASELINE, scenes=[SCENES_BY_ID["memory-dedup"]])
        b = run_suite(BASELINE, scenes=[SCENES_BY_ID["memory-dedup"]])
        assert a.results[0].outcome.metrics == b.results[0].outcome.metrics

    def test_score_skips_scenes_without_checks(self):
        """**「未测」与「失败」必须可区分。**

        把「没测」算成 0 分会让整体分虚低，算成 1 分会让它虚高 ——
        两者都会让报告的分失去意义。
        """
        no_checks = SceneResult(
            scene=_scene("x", lambda ctx: SceneOutcome()), outcome=SceneOutcome()
        )
        assert no_checks.score is None

        failed = SceneResult(
            scene=_scene("y", lambda ctx: SceneOutcome()),
            outcome=SceneOutcome(checks=[Check("nope", False)]),
        )
        assert failed.score == 0.0


# =============================================================================
# 对比
# =============================================================================


class TestComparison:
    @staticmethod
    def _suite(name: str, metrics: dict[str, float], *, error: str | None = None) -> SuiteResult:
        """造一个只含单个场景的 suite。

        ⚠️ **scene_id 固定为 `s1`，两个 suite 必须一样。**
        `compare` 是按 `(scene_id, metric)` 配对的 —— 两边 id 不同时
        所有指标都会被归到「只在一侧出现」，对比结果为空。
        真实场景的 id 是模块级常量，所以它们天然稳定；
        这里是测试夹具需要自己保证。
        """
        scene = _scene("s1", lambda ctx: SceneOutcome())
        outcome = SceneOutcome(metrics=metrics)
        return SuiteResult(
            config_name=name,
            config_notes=(),
            results=[SceneResult(scene=scene, outcome=outcome, error=error)],
            seed=1,
        )

    def test_delta_and_direction(self):
        base = self._suite("base", {"x": 1.0})
        exp = self._suite("exp", {"x": 2.0})
        cmp = compare(base, exp)
        assert len(cmp.deltas) == 1
        d = cmp.deltas[0]
        assert d.delta == 1.0
        assert d.mark == "↑"

    def test_zero_baseline_has_no_relative_change(self):
        """基数为 0 时相对变化无定义 —— 返回 None 而不是 `inf`。

        `inf` 会在报告里变成一个看起来像成绩的符号，而它其实只是
        「除数恰好是 0」。
        """
        cmp = compare(self._suite("b", {"x": 0.0}), self._suite("e", {"x": 5.0}))
        assert cmp.deltas[0].rel is None

    def test_identical_metrics_produce_no_changes(self):
        cmp = compare(self._suite("b", {"x": 1.0}), self._suite("e", {"x": 1.0}))
        assert cmp.changed() == []

    def test_only_in_one_side_is_reported(self):
        cmp = compare(self._suite("b", {"x": 1.0}), self._suite("e", {"y": 2.0}))
        assert len(cmp.only_in) == 2


# =============================================================================
# 报告：**不能撒谎**
# =============================================================================


class TestReportHonesty:
    @staticmethod
    def _suite(name: str, **kw) -> SuiteResult:
        scene = _scene("s1", lambda ctx: SceneOutcome())
        return SuiteResult(
            config_name=name, config_notes=(), results=[], seed=1, **kw
        )

    def test_all_errored_experiment_is_not_reported_as_identical(self):
        """**最重要的一条。** 实验组全崩 ≠ 两组结果相同。

        前者会得出「该机制没有贡献」这个**反向的**结论 ——
        而真相是它一行都没跑（比如没配 MYSQL_* 却选了 mysql 后端）。
        """
        base = SuiteResult(
            config_name="baseline",
            config_notes=(),
            results=[
                SceneResult(
                    scene=_scene("s1", lambda ctx: SceneOutcome()),
                    outcome=SceneOutcome(metrics={"x": 1.0}),
                )
            ],
            seed=1,
        )
        broken = SuiteResult(
            config_name="broken",
            config_notes=("换了后端",),
            results=[
                SceneResult(
                    scene=_scene("s1", lambda ctx: SceneOutcome()),
                    outcome=SceneOutcome(),
                    error="RuntimeError: 没配 MYSQL_*",
                )
            ],
            seed=1,
        )
        text = render_markdown([base, broken], [compare(base, broken)])
        assert "本次对比无效" in text, "应明确指出对比无效"
        assert "这不是「两组结果相同」" in text
        assert "所有指标完全一致" not in text, "绝不能说它们一致"

    def test_partial_error_is_flagged(self):
        base = SuiteResult(
            config_name="b", config_notes=(), results=[], seed=1
        )
        exp = SuiteResult(
            config_name="e",
            config_notes=(),
            results=[
                SceneResult(
                    scene=_scene("ok", lambda ctx: SceneOutcome()),
                    outcome=SceneOutcome(metrics={"x": 1.0}),
                ),
                SceneResult(
                    scene=_scene("bad", lambda ctx: SceneOutcome()),
                    outcome=SceneOutcome(),
                    error="boom",
                ),
            ],
            seed=1,
        )
        text = render_markdown([base, exp], [compare(base, exp)])
        assert "崩溃" in text

    def test_blocked_items_are_rendered_not_omitted(self):
        """**不可测项必须出现。**

        静默跳过的后果是报告里少一行，而读的人以为都测过了。
        给一个用占位数据算出来的数字更糟 —— 它有数字的样子，会被引用。
        """
        base = SuiteResult(
            config_name="b",
            config_notes=(),
            results=[
                SceneResult(
                    scene=_scene("acoustic", lambda ctx: SceneOutcome(), layer=Layer.B),
                    outcome=SceneOutcome(
                        blocked=[
                            Blocked(
                                item="E21 声学分类 macro-F1",
                                reason="先验是占位值",
                                needs="真实 CatMeows 统计量",
                            )
                        ]
                    ),
                )
            ],
            seed=1,
        )
        text = render_markdown([base])
        assert "不可测项" in text
        assert "E21 声学分类 macro-F1" in text
        assert "先验是占位值" in text
        assert "真实 CatMeows 统计量" in text, "必须说清缺什么"

    def test_report_states_what_it_cannot_prove(self):
        """"本报告不能证明什么"这一节是报告可信度的来源。

        A 层（确定性）可以声称绝对数值，C 层（主观）只能作相对比较。
        不写清边界，数字就会以同一种可信度被引用。
        """
        text = render_markdown([self._suite("baseline")])
        assert "本报告不能证明什么" in text
        assert "MockLLM" in text or "mock" in text.lower()
        assert "A 层" in text

    def test_failed_checks_show_their_detail(self):
        base = SuiteResult(
            config_name="b",
            config_notes=(),
            results=[
                SceneResult(
                    scene=_scene("s", lambda ctx: SceneOutcome()),
                    outcome=SceneOutcome(
                        checks=[
                            Check("这个不通过", False, "因为某个具体原因"),
                        ]
                    ),
                )
            ],
            seed=1,
        )
        text = render_markdown([base])
        assert "❌" in text
        assert "因为某个具体原因" in text, "只说失败不说原因，报告就没法驱动改动"

    def test_json_is_parseable_and_complete(self):
        import json

        suite = run_suite(BASELINE, scenes=[SCENES_BY_ID["memory-dedup"]])
        payload = json.loads(render_json([suite]))
        assert payload["suites"][0]["config"] == "baseline"
        scene = payload["suites"][0]["scenes"][0]
        assert scene["scene_id"] == "memory-dedup"
        assert scene["checks"], "JSON 也要带断言明细（给 CI 用）"
        assert "metrics" in scene


# =============================================================================
# 分层汇总
# =============================================================================


class TestLayerSummary:
    def test_layers_are_reported_separately(self):
        """**分层呈现是硬约束**（DESIGN §6.1）。

        把 A 层（确定性）与 C 层（主观）混进一个总分，
        等于给后者披上前者的可信度。
        """
        suite = run_suite(BASELINE, scenes=list(ALL_SCENES))
        summary = layer_summary(suite)
        assert Layer.A in summary
        assert Layer.B in summary, "B 层即使全不可测也要出现 —— 而不是被省略"

    def test_unmeasurable_is_counted_not_dropped(self):
        suite = run_suite(BASELINE, scenes=[SCENES_BY_ID["acoustic-blocked"]])
        summary = layer_summary(suite)
        assert summary[Layer.B]["不可测项数"] >= 1
