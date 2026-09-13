"""契约层不变量测试。

这些测试**不需要大模型、不需要网络**，可挂在 CI 上。
它们验证的不是「功能是否正确」，而是「设计约束是否被结构性强制」。

对应文档：
- 记忆系统      docs/04-memory.md（防自我强化、冲突判定、检索隔离）
- 计划与依赖    docs/01-architecture.md §6、docs/09-intent-and-planning.md
- 意图策略      docs/09 §4（代价矩阵、审计只读）
- 行为解释      docs/03-behavior-interpreter.md（无证据不许给数值）
- 健康模块      docs/07-health.md（禁词、覆盖率门、红旗一致性）
- 契约自洽      data/health/red_flags.yaml（信号词汇表引用）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from app.schemas import (
    FORBIDDEN_PHRASES,
    MIN_COVERAGE_FOR_ASSESSMENT,
    AcousticFeatures,
    AudioKind,
    BehaviorInterpretation,
    ContextLabel,
    Dependency,
    DependencyKind,
    EvidenceItem,
    EvidenceKind,
    EvidenceMode,
    FeatureQuality,
    GuardResult,
    HealthAssessment,
    InputIntent,
    IntentCandidate,
    IntentPolicy,
    MemoryEvent,
    MemoryItem,
    MemorySource,
    MemoryStatus,
    MemoryWriteDecision,
    Plan,
    PlanResult,
    Polarity,
    PriorAdjustment,
    PriorOperation,
    PriorSkip,
    PriorTargetField,
    RawInput,
    Severity,
    SubTask,
    SubTaskKind,
    SubTaskResult,
    SubTaskStatus,
    UrgencyLevel,
    Violation,
    ViolationType,
    WriteAction,
    misroute_cost,
    policy_for,
)
from app.schemas.memory import EventType, RetrievalSource

NOW = datetime.now(timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════
# 记忆系统不变量（docs/04）
# ═══════════════════════════════════════════════════════════════


def _mem(**kw: Any) -> MemoryEvent:
    """构造测试用记忆。

    用**字面量**而非 ``dict(...)``，并用 ``model_validate`` 而非
    ``MemoryEvent(**base)``：后者会因为混合值类型被推断为 ``str | float``，
    触发一堆无意义的类型错误（而且不跑校验器的话会放过非法组合）。
    """
    defaults: dict[str, Any] = {
        "user_id": "u1",
        "pet_id": "p1",
        "event_type": EventType.PREFERENCE,
        "subject": "cat_wand",
        "content": "喜欢逗猫棒",
        "polarity": Polarity.POSITIVE,
        "source": MemorySource.USER_OBSERVATION,
        "confidence": 0.9,
    }
    return MemoryEvent.model_validate({**defaults, **kw})


class TestMemoryInvariants:
    """docs/04 §3.5 防自我强化 —— 本项目最重要的安全约束。"""

    def test_system_inference_cannot_be_active(self):
        """R1：模型输出不能作为下一轮的事实输入，否则幻觉会自我强化。"""
        with pytest.raises(ValueError, match="SYSTEM_INFERENCE"):
            _mem(source=MemorySource.SYSTEM_INFERENCE, status=MemoryStatus.ACTIVE)

    def test_system_inference_pending_is_the_only_path(self):
        ev = _mem(
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.PENDING_CONFIRMATION,
        )
        assert ev.status is MemoryStatus.PENDING_CONFIRMATION

    def test_non_active_memory_excluded_from_default_retrieval(self):
        pending = _mem(
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.PENDING_CONFIRMATION,
        )
        assert not pending.is_retrievable_by_default
        with pytest.raises(ValueError, match="不得进入默认检索"):
            MemoryItem(event=pending, score=0.9)

    def test_active_memory_is_retrievable(self):
        assert _mem().is_retrievable_by_default


class TestConflictDetection:
    """docs/04 §3.3 —— 三条件齐备才算冲突，缺时间重叠会把演变误判为矛盾。"""

    def test_overlapping_time_ranges_conflict(self):
        like = _mem(polarity=Polarity.POSITIVE, content="喜欢逗猫棒")
        dislike = _mem(
            polarity=Polarity.NEGATIVE,
            content="最近不喜欢",
            valid_from=NOW - timedelta(days=7),
        )
        assert like.conflicts_with(dislike)

    def test_disjoint_time_ranges_are_evolution_not_conflict(self):
        """「上个月喜欢」与「这周不喜欢」并存，不是矛盾。"""
        past = _mem(
            polarity=Polarity.POSITIVE,
            valid_from=NOW - timedelta(days=40),
            valid_to=NOW - timedelta(days=10),
        )
        recent = _mem(
            polarity=Polarity.NEGATIVE,
            valid_from=NOW - timedelta(days=7),
        )
        assert not past.conflicts_with(recent)

    @pytest.mark.parametrize(
        "kw",
        [
            {"subject": "vacuum"},  # 不同主体
            {"polarity": Polarity.POSITIVE},  # 同极性
            {"polarity": Polarity.NEUTRAL},  # 中性不参与冲突
            {"pet_id": "p2"},  # 不同宠物
        ],
    )
    def test_non_conflicting_cases(self, kw):
        assert not _mem().conflicts_with(_mem(**kw))

    def test_invalid_time_range_rejected(self):
        with pytest.raises(ValueError, match="valid_from"):
            _mem(valid_from=NOW, valid_to=NOW - timedelta(days=1))


class TestDecayPolicy:
    """docs/04 §4.3 —— 用统一衰减率会让系统忘掉猫「一直喜欢」的东西。"""

    @pytest.mark.parametrize(
        "event_type,expect_none",
        [
            (EventType.PREFERENCE, True),
            (EventType.ROUTINE, True),
            (EventType.BEHAVIOR, False),
            (EventType.CONTEXT, False),
        ],
    )
    def test_halflife_is_type_dependent(self, event_type, expect_none):
        ev = _mem(event_type=event_type)
        assert (ev.halflife_days is None) is expect_none

    def test_behavior_decays_slower_than_context(self):
        """behavior(30d) 应比 context(7d) 衰减更慢。"""
        behavior_hl = _mem(event_type=EventType.BEHAVIOR).halflife_days
        context_hl = _mem(event_type=EventType.CONTEXT).halflife_days
        # 断言非 None 后再比较：否则若两者都变成不衰减，
        # ``None > None`` 会直接抛 TypeError 而不是给出有意义的失败信息
        assert behavior_hl is not None and context_hl is not None
        assert behavior_hl > context_hl


class TestWriteDecision:
    """docs/04 §3.2 / §3.3 —— 决策必须自洽，动作不可无主语。"""

    def test_reinforce_requires_duplicate_of(self):
        with pytest.raises(ValueError, match="reinforce"):
            MemoryWriteDecision(event=_mem(), action=WriteAction.REINFORCE, reason="r")

    def test_supersede_requires_target(self):
        with pytest.raises(ValueError, match="supersede"):
            MemoryWriteDecision(event=_mem(), action=WriteAction.SUPERSEDE, reason="r")

    def test_valid_reinforce(self):
        d = MemoryWriteDecision(
            event=_mem(),
            action=WriteAction.REINFORCE,
            reason="语义重复，强化计数",
            duplicate_of="m1",
        )
        assert d.action is WriteAction.REINFORCE


# ═══════════════════════════════════════════════════════════════
# 计划、依赖与先验机制（docs/01 §6、docs/09）
# ═══════════════════════════════════════════════════════════════


def _sample_plan() -> Plan:
    return Plan(
        user_id="u",
        pet_id="p",
        subtasks=[
            SubTask(task_id="T1", kind=SubTaskKind.RECORD_EVENT),
            SubTask(task_id="T3", kind=SubTaskKind.AGGREGATE_QUERY),
            SubTask(
                task_id="T2",
                kind=SubTaskKind.INTERPRET_BEHAVIOR,
                depends_on=[
                    Dependency(
                        on_task_id="T3",
                        kind=DependencyKind.PRIOR,
                        note="聚合结果上调门口情境先验",
                    )
                ],
            ),
        ],
        waves=[["T1", "T3"], ["T2"]],
        decomposition_note="记录与聚合可并行；解释依赖聚合的先验",
    )


class TestPlanTopology:
    """docs/01 §7.2 —— 非法计划必须在构造时被拒绝，而不是进入执行。"""

    def test_valid_plan(self):
        p = _sample_plan()
        assert p.max_parallelism == 2
        assert p.serial_depth == 2
        assert p.declared_prior_refs() == ["T3->T2"]

    def test_dependency_in_same_wave_rejected(self):
        with pytest.raises(ValueError, match="拓扑序违规"):
            Plan(
                user_id="u",
                pet_id="p",
                subtasks=_sample_plan().subtasks,
                waves=[["T1", "T3", "T2"]],
                decomposition_note="x",
            )

    def test_waves_must_cover_all_tasks(self):
        with pytest.raises(ValueError, match="不一致"):
            Plan(
                user_id="u",
                pet_id="p",
                subtasks=_sample_plan().subtasks,
                waves=[["T1"], ["T2"]],
                decomposition_note="x",
            )

    def test_task_cannot_appear_twice(self):
        with pytest.raises(ValueError, match="不得出现在多个 wave"):
            Plan(
                user_id="u",
                pet_id="p",
                subtasks=_sample_plan().subtasks,
                waves=[["T1", "T3"], ["T2", "T2"]],
                decomposition_note="x",
            )

    def test_dependency_on_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="不存在的任务"):
            Plan(
                user_id="u",
                pet_id="p",
                subtasks=[
                    SubTask(
                        task_id="T1",
                        kind=SubTaskKind.RECORD_EVENT,
                        depends_on=[
                            Dependency(
                                on_task_id="NOPE",
                                kind=DependencyKind.DATA,
                                note="n",
                            )
                        ],
                    )
                ],
                waves=[["T1"]],
                decomposition_note="x",
            )


class TestPlanQueryApi:
    """docs/10-self-review.md F4 —— 文档曾引用不存在的方法。"""

    def test_task_lookup(self):
        p = _sample_plan()
        task = p.task("T2")
        assert task is not None, "T2 应存在于样例计划中"
        assert task.kind is SubTaskKind.INTERPRET_BEHAVIOR
        assert p.task("NOPE") is None

    def test_require_task_raises(self):
        with pytest.raises(KeyError):
            _sample_plan().require_task("NOPE")

    def test_wave_tasks(self):
        p = _sample_plan()
        assert len(p.wave_tasks(0)) == 2
        assert len(p.wave_tasks(1)) == 1

    def test_wave_index_out_of_range(self):
        with pytest.raises(IndexError):
            _sample_plan().wave_tasks(9)

    def test_next_wave_index_terminates(self):
        p = _sample_plan()
        assert p.next_wave_index(0) == 1
        assert p.next_wave_index(1) is None

    def test_prior_upstream_ids(self):
        assert _sample_plan().require_task("T2").prior_upstream_ids == {"T3"}

    def test_index_survives_serialization_roundtrip(self):
        """Plan 会经 LangGraph 状态传递与持久化，索引必须存活。"""
        p = _sample_plan()
        for rebuilt in (
            p.model_copy(deep=True),
            Plan.model_validate(p.model_dump()),
            Plan.model_validate_json(p.model_dump_json()),
        ):
            assert rebuilt.task("T2") is not None
            assert len(rebuilt.wave_tasks(0)) == 2


class TestPriorAdjustment:
    """docs/01 §6.2.1（自审 F1）—— 先验依赖必须有机制，不能只是名词。"""

    def _adj(self, **kw: Any) -> PriorAdjustment:
        defaults: dict[str, Any] = {
            "target_task_id": "T2",
            "target_field": PriorTargetField.SCENE_PRIOR,
            "target_key": "door_attention",
            "operation": PriorOperation.SCALE,
            "value": 1.4,
            "rationale": "本月叫声显著多于上月，持续性诉求可能性上升",
            "confidence": 0.7,
        }
        return PriorAdjustment.model_validate({**defaults, **kw})

    def test_scale_operation(self):
        assert self._adj().apply_to(0.3) == pytest.approx(0.42)

    def test_delta_operation(self):
        adj = self._adj(operation=PriorOperation.DELTA, value=0.1)
        assert adj.apply_to(0.3) == pytest.approx(0.4)

    def test_set_operation(self):
        adj = self._adj(
            operation=PriorOperation.SET,
            target_field=PriorTargetField.REJECTION_THRESHOLD,
            target_key=None,
            value=0.8,
        )
        assert adj.apply_to(0.3) == pytest.approx(0.8)

    def test_scene_prior_requires_key(self):
        with pytest.raises(ValueError, match="target_key"):
            self._adj(target_key=None)

    def test_rationale_is_mandatory(self):
        with pytest.raises(ValueError, match="依据"):
            self._adj(rationale="   ")

    def test_carried_on_subtask_result(self):
        sr = SubTaskResult(
            task_id="T3",
            kind=SubTaskKind.AGGREGATE_QUERY,
            status=SubTaskStatus.SUCCESS,
            prior_adjustments=[self._adj()],
        )
        assert len(sr.adjustments_for("T2")) == 1
        assert sr.adjustments_for("T9") == []


class TestPriorDependencyAccounting:
    """docs/01 §6.2.1 P3 —— 禁止静默丢弃先验依赖。

    这条不变量把「先验依赖」从纸面论证变成可测试的机制：
    未应用、又未说明原因的依赖会让依赖分析失去意义。
    """

    def test_fully_accounted_applied(self):
        r = PlanResult(
            declared_prior_dependencies=["T3->T2"],
            applied_prior_dependencies=["T3->T2"],
        )
        assert r.applied_prior_dependencies == ["T3->T2"]

    def test_fully_accounted_skipped_with_reason(self):
        r = PlanResult(
            declared_prior_dependencies=["T3->T2"],
            skipped_prior_dependencies=[
                PriorSkip(dependency="T3->T2", reason="T3 失败，无法产出调整")
            ],
        )
        assert r.skipped_prior_dependencies[0].reason

    def test_silent_drop_rejected(self):
        with pytest.raises(ValueError, match="未被处置"):
            PlanResult(declared_prior_dependencies=["T3->T2"])

    def test_undeclared_disposition_rejected(self):
        with pytest.raises(ValueError, match="未声明"):
            PlanResult(applied_prior_dependencies=["T3->T2"])

    def test_apply_and_skip_overlap_rejected(self):
        with pytest.raises(ValueError, match="既标记应用又标记跳过"):
            PlanResult(
                declared_prior_dependencies=["T3->T2"],
                applied_prior_dependencies=["T3->T2"],
                skipped_prior_dependencies=[PriorSkip(dependency="T3->T2", reason="x")],
            )


class TestSubTaskResult:
    """docs/01 §8 —— 正确性相关的降级必须可见。"""

    def test_degraded_requires_note(self):
        with pytest.raises(ValueError, match="降级内容"):
            SubTaskResult(
                task_id="T2",
                kind=SubTaskKind.INTERPRET_BEHAVIOR,
                status=SubTaskStatus.DEGRADED,
            )

    def test_degraded_with_note_ok(self):
        r = SubTaskResult(
            task_id="T2",
            kind=SubTaskKind.INTERPRET_BEHAVIOR,
            status=SubTaskStatus.DEGRADED,
            degraded_note="声学特征提取失败，退化为文本推断",
        )
        assert r.is_usable

    def test_partial_success_notice(self):
        r = PlanResult(
            results=[
                SubTaskResult(
                    task_id="T1",
                    kind=SubTaskKind.RECORD_EVENT,
                    status=SubTaskStatus.SUCCESS,
                ),
                SubTaskResult(
                    task_id="T2",
                    kind=SubTaskKind.INTERPRET_BEHAVIOR,
                    status=SubTaskStatus.FAILED,
                ),
            ]
        )
        assert r.is_partial_success
        assert "T2" in r.render_partial_notice()

    def test_all_success_is_not_partial(self):
        r = PlanResult(
            results=[
                SubTaskResult(
                    task_id="T1",
                    kind=SubTaskKind.RECORD_EVENT,
                    status=SubTaskStatus.SUCCESS,
                )
            ]
        )
        assert not r.is_partial_success
        assert r.render_partial_notice() == ""


# ═══════════════════════════════════════════════════════════════
# 意图策略与代价矩阵（docs/09 §4）
# ═══════════════════════════════════════════════════════════════


class TestIntentPolicies:
    def test_each_intent_has_exactly_one_policy(self):
        """防止重复定义（曾因改动引入 RECORD_EVENT 重复）。"""
        from app.schemas import INTENT_POLICIES

        intents = [p.intent for p in INTENT_POLICIES]
        assert len(intents) == len(set(intents)), f"重复策略：{intents}"

    def test_memory_query_is_read_only(self):
        """审计不得污染被审计对象 —— 结构性强制。"""
        p = policy_for(InputIntent.MEMORY_QUERY)
        assert p.memory_write.value == "forbidden"
        assert p.retrieval_k > policy_for(InputIntent.CHAT).retrieval_k
        assert p.rejection_strict

    def test_memory_query_with_write_allowed_rejected(self):
        from app.schemas import MemoryWritePermission

        with pytest.raises(ValueError, match="审计"):
            IntentPolicy(
                intent=InputIntent.MEMORY_QUERY,
                min_confidence=0.5,
                prefer_recall=False,
                memory_write=MemoryWritePermission.ALLOWED,
                rationale="故意非法",
            )

    def test_record_event_reports_write_so_write_precedes_response(self):
        p = policy_for(InputIntent.RECORD_EVENT)
        assert p.response_reports_write
        assert p.require_confirmation

    def test_asymmetric_cost_both_directions_high(self):
        """RECORD_EVENT 两个方向代价都高 —— 这是「改交互而非调阈值」的依据。"""
        into = misroute_cost(InputIntent.CHAT, InputIntent.RECORD_EVENT)
        out_of = misroute_cost(InputIntent.RECORD_EVENT, InputIntent.CHAT)
        assert into >= 0.9 and out_of >= 0.9

    def test_asking_is_cheaper_than_not_asking(self):
        """docs/09 §4 —— 宁可多问一句（0.15），也不要该问未问（0.6）。

        注意 misroute_cost(predicted, actual) 的语义：
          predicted=AMBIGUOUS, actual=CHAT  → 多问了一句（便宜）
          predicted=CHAT,      actual=AMBIGUOUS → 该问未问（贵）
        """
        over_ask = misroute_cost(InputIntent.AMBIGUOUS, InputIntent.CHAT)
        missed_ask = misroute_cost(InputIntent.CHAT, InputIntent.AMBIGUOUS)
        assert over_ask < missed_ask

    def test_missed_ask_is_silent_failure(self):
        """该问未问时用户不会察觉，所以代价更高。"""
        from app.schemas import MISROUTE_COSTS

        entry = next(
            c
            for c in MISROUTE_COSTS
            if c.predicted is InputIntent.CHAT and c.actual is InputIntent.AMBIGUOUS
        )
        assert entry.is_silent

    def test_identical_intent_costs_nothing(self):
        for i in InputIntent:
            assert misroute_cost(i, i) == 0.0


# ═══════════════════════════════════════════════════════════════
# 行为解释（docs/03）
# ═══════════════════════════════════════════════════════════════


def _features() -> AcousticFeatures:
    return AcousticFeatures(
        duration=0.82,
        f0_mean=612.4,
        f0_range=180.2,
        f0_slope=0.31,
        call_rate=4.5,
        ici_mean=0.63,
        rms_mean=0.14,
        roughness=0.22,
        quality=FeatureQuality.GOOD,
    )


class TestBehaviorInterpretation:
    """docs/03 §6 —— 没有声学证据就不许给数值置信度。"""

    def test_text_only_forbids_acoustic_features(self):
        with pytest.raises(ValueError, match="acoustic_features"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.TEXT_ONLY,
                acoustic_features=_features(),
                suggested_observation="观察",
            )

    def test_text_only_forbids_numeric_posterior(self):
        """否则系统会退化为「用户描述什么就顺着说什么」的谄媚模型。"""
        with pytest.raises(ValueError, match="posterior"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.TEXT_ONLY,
                candidates=[
                    IntentCandidate(
                        context=ContextLabel.FOOD_WAITING,
                        posterior=0.7,
                        display="可能饿了",
                    )
                ],
                suggested_observation="观察",
            )

    def test_text_only_without_numbers_is_valid(self):
        bi = BehaviorInterpretation(
            evidence_mode=EvidenceMode.TEXT_ONLY,
            candidates=[
                IntentCandidate(context=ContextLabel.FOOD_WAITING, display="可能饿了")
            ],
            suggested_observation="观察它是否蹭腿",
        )
        assert bi.confidence_tier == "none"

    def test_acoustic_mode_requires_features(self):
        with pytest.raises(ValueError, match="必须提供"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
                suggested_observation="观察",
            )

    def test_posteriors_must_be_normalized(self):
        with pytest.raises(ValueError, match="归一化"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
                acoustic_features=_features(),
                candidates=[
                    IntentCandidate(
                        context=ContextLabel.FOOD_WAITING,
                        posterior=0.5,
                        display="a",
                    ),
                    IntentCandidate(
                        context=ContextLabel.DOOR_ATTENTION,
                        posterior=0.2,
                        display="b",
                    ),
                ],
                suggested_observation="观察",
            )

    @pytest.mark.parametrize(
        "posteriors,tier",
        [
            ([0.70, 0.30], "high"),
            ([0.50, 0.30, 0.20], "medium"),
            ([0.40, 0.30, 0.30], "low"),
        ],
    )
    def test_confidence_tiers(self, posteriors, tier):
        """分档决定系统是给明确建议还是只给方向（docs/03 §7）。

        注：两候选时必有一个 ≥0.5，因此 low 档需要 ≥3 个候选。
        """
        contexts = [
            ContextLabel.FOOD_WAITING,
            ContextLabel.DOOR_ATTENTION,
            ContextLabel.OTHER,
        ]
        bi = BehaviorInterpretation(
            evidence_mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
            acoustic_features=_features(),
            candidates=[
                IntentCandidate(context=ctx, posterior=p, display=str(p))
                for ctx, p in zip(contexts, posteriors)
            ],
            suggested_observation="观察",
        )
        assert bi.confidence_tier == tier

    def test_contribution_is_required_when_there_is_a_model(self):
        """有概率模型时，每条证据必须可归因 —— 这是「证据」与「讲故事」的分界。

        注意断言位置：不再断言「字段必填」，而是断言「**在有模型的模式下必填**」。
        这比原来更强 —— 原来只要构造 EvidenceItem 就必须给值，
        而那让无模型的模式只能填 0.0，反而谎称「已参与计算但影响为零」。
        """
        bare = EvidenceItem(
            kind=EvidenceKind.MEASURED,
            statement="叫声时长 0.82s",
            source="acoustic:duration",
        )
        # 无模型时允许为 None
        assert bare.log_odds_contribution is None

        with pytest.raises(ValueError, match="log_odds_contribution"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.ACOUSTIC_PLUS_HISTORY,
                acoustic_features=_features(),
                evidence=[bare],
                suggested_observation="观察",
            )

    def test_no_contribution_when_there_is_no_model(self):
        """无模型时**不得**携带贡献值。

        填 0.0 会谎称「已参与计算但影响为零」，而事实是它没参与任何计算。
        """
        with pytest.raises(ValueError, match="不得携带 log_odds_contribution"):
            BehaviorInterpretation(
                evidence_mode=EvidenceMode.MEASURED_ONLY,
                acoustic_features=_features(),
                evidence=[
                    EvidenceItem(
                        kind=EvidenceKind.MEASURED,
                        statement="叫声时长 0.82s",
                        source="acoustic:duration",
                        log_odds_contribution=0.0,
                    )
                ],
                suggested_observation="观察",
            )

    def test_valid_evidence_item(self):
        item = EvidenceItem(
            kind=EvidenceKind.MEASURED,
            statement="叫声时长 0.82s，长于它日常索食均值 0.41s",
            source="acoustic:duration",
            value=0.82,
            reference=0.41,
            log_odds_contribution=1.34,
        )
        assert item.log_odds_contribution == pytest.approx(1.34)


class TestAudioKind:
    """docs/03 §5 —— 猫叫不做 ASR，整条链路不同，必须声明类型。"""

    def test_audio_requires_kind(self):
        with pytest.raises(ValueError, match="audio_kind"):
            RawInput(audio_url="http://x/meow.wav")

    def test_cat_meow_declared(self):
        r = RawInput(audio_url="http://x/meow.wav", audio_kind=AudioKind.CAT_MEOW)
        assert r.audio_kind is AudioKind.CAT_MEOW

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="至少需要"):
            RawInput()


# ═══════════════════════════════════════════════════════════════
# 健康模块（docs/07）
# ═══════════════════════════════════════════════════════════════


class TestHealthHonesty:
    """docs/07 §2 —— 「没检测到异常」与「没有异常」是两件事。"""

    @pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES[:4])
    def test_forbidden_phrases_rejected(self, phrase):
        with pytest.raises(ValueError, match="禁止"):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.NO_DEVIATION_DETECTED,
                coverage=0.8,
                coverage_note="本周记录 5/7 天",
                recommendation=f"它很{phrase}",
            )

    def test_insufficient_coverage_forces_level(self):
        """禁止把「没有数据」伪装成「没有异常」。"""
        with pytest.raises(ValueError, match="INSUFFICIENT_DATA"):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.NO_DEVIATION_DETECTED,
                coverage=MIN_COVERAGE_FOR_ASSESSMENT - 0.01,
                coverage_note="本周记录 1/7 天",
                recommendation="建议继续记录",
            )

    def test_insufficient_coverage_forbids_findings(self):
        from app.schemas import SignalDeviation

        with pytest.raises(ValueError):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.INSUFFICIENT_DATA,
                coverage=0.1,
                coverage_note="记录不足",
                recommendation="请补充记录",
                findings=[
                    SignalDeviation(
                        signal="meal.intake",
                        direction="decrease",
                        magnitude=2.4,
                        duration_days=3,
                        baseline_window="a/b",
                        note="n",
                    )
                ],
            )

    def test_emergency_requires_red_flag(self):
        """级别不可由模型判定，只能由红旗规则触发。"""
        with pytest.raises(ValueError, match="红旗"):
            HealthAssessment(
                user_id="u",
                pet_id="p",
                level=UrgencyLevel.EMERGENCY,
                coverage=0.8,
                coverage_note="记录完整",
                recommendation="请立即就医",
            )

    def test_valid_insufficient_data_assessment(self):
        a = HealthAssessment(
            user_id="u",
            pet_id="p",
            level=UrgencyLevel.INSUFFICIENT_DATA,
            coverage=0.1,
            coverage_note="本周有效记录 1/7 天",
            recommendation="数据不足，无法评估，请继续记录",
        )
        assert a.must_not_be_read_as == "未发现异常不代表健康"


@pytest.fixture(scope="module")
def red_flag_table():
    """红旗规则表。docs/07 §12 —— 规则是数据，必须自洽。"""
    path = PROJECT_ROOT / "data" / "health" / "red_flags.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestRedFlagTable:
    def test_loads(self, red_flag_table):
        assert red_flag_table["version"]
        assert red_flag_table["rules"]
        assert red_flag_table["reviewed_by"] is None  # 尚未经兽医审核

    def test_all_rule_signals_declared(self, red_flag_table):
        """信号拼写错误会导致规则静默失效 —— 必须校验。

        注意条件可嵌套：``any_of`` / ``all_of`` 的某些项本身是容器（无 ``signal`` 键）。
        """
        vocab = set(red_flag_table["signal_vocabulary"])

        def collect(cond: dict, out: list[str]) -> None:
            if "signal" in cond:
                out.append(cond["signal"])
            for nested_key in ("all_of", "any_of"):
                for sub in cond.get(nested_key) or []:
                    collect(sub, out)

        for rule in red_flag_table["rules"]:
            found: list[str] = []
            for key in ("any_of", "all_of"):
                for c in rule["conditions"].get(key) or []:
                    collect(c, found)
            for c in rule.get("aggravating") or []:
                collect(c, found)
            unknown = set(found) - vocab
            assert not unknown, f"{rule['id']} 引用了未定义信号：{unknown}"

    def test_only_strong_evidence_rules_enabled(self, red_flag_table):
        """docs/07 §12.3 —— 仅启用多来源兽医手册强一致的规则。"""
        enabled = {r["id"] for r in red_flag_table["rules"] if r["enabled"]}
        assert enabled == {
            "urinary_obstruction",
            "respiratory_distress",
            "neurological_emergency",
            "hemorrhage_or_toxin",
        }

    def test_disabled_rules_explain_why(self, red_flag_table):
        """不确定的阈值宁可不给，但必须说明原因。"""
        for rule in red_flag_table["rules"]:
            if not rule["enabled"]:
                assert rule.get("disabled_reason"), f"{rule['id']} 缺少 disabled_reason"

    def test_enabled_rules_have_sources(self, red_flag_table):
        for rule in red_flag_table["rules"]:
            if rule["enabled"]:
                assert rule["sources"], f"{rule['id']} 缺少来源引用"

    def test_all_rules_never_say_healthy(self, red_flag_table):
        for rule in red_flag_table["rules"]:
            assert rule["never_say_healthy"] is True

    def test_urinary_obstruction_is_emergency(self, red_flag_table):
        """尿闭 24–48 小时可致死，必须是最紧急级别。"""
        rule = next(
            r for r in red_flag_table["rules"] if r["id"] == "urinary_obstruction"
        )
        assert rule["urgency"] == "L3"
        assert rule["sex_bias"] == "male"


# ═══════════════════════════════════════════════════════════════
# 守卫（docs/01 §3.10）
# ═══════════════════════════════════════════════════════════════


class TestGuard:
    def test_no_violations_passes(self):
        assert GuardResult(passed=True).passed

    def test_critical_violation_forbids_pass(self):
        with pytest.raises(ValueError, match="不得 passed=True"):
            GuardResult(
                passed=True,
                violations=[
                    Violation(
                        type=ViolationType.OVERCERTAINTY,
                        severity=Severity.CRITICAL,
                        detail="把推测表述为结论",
                    )
                ],
            )

    def test_minor_violation_may_pass(self):
        g = GuardResult(
            passed=True,
            violations=[
                Violation(
                    type=ViolationType.OVERCERTAINTY,
                    severity=Severity.MINOR,
                    detail="措辞略绝对，可接受",
                )
            ],
        )
        assert g.passed
        assert g.worst_severity is Severity.MINOR

    def test_degrade_requires_visible_notice(self):
        with pytest.raises(ValueError, match="用户可见"):
            GuardResult(passed=False, degrade_to_conservative=True)

    def test_worst_severity(self):
        g = GuardResult(
            passed=False,
            violations=[
                Violation(
                    type=ViolationType.OVERCERTAINTY,
                    severity=Severity.MINOR,
                    detail="a",
                ),
                Violation(
                    type=ViolationType.UNTRACEABLE_CLAIM,
                    severity=Severity.CRITICAL,
                    detail="b",
                ),
            ],
        )
        assert g.worst_severity is Severity.CRITICAL


# ═══════════════════════════════════════════════════════════════
# 契约层完整性
# ═══════════════════════════════════════════════════════════════


class TestContractSurface:
    def test_all_exports_resolve(self):
        import app.schemas as S

        missing = [n for n in S.__all__ if not hasattr(S, n)]
        assert not missing, f"__all__ 声明但未导出：{missing}"

    def test_no_duplicate_exports(self):
        import app.schemas as S

        assert len(S.__all__) == len(set(S.__all__))

    def test_layer_access_modes_are_distinct(self):
        """docs/04 §1 —— 三层访问模式必须不同，这是核心设计决策。"""
        from app.schemas import LAYER_ACCESS_MODE, MemoryAccessMode, MemoryLayer

        assert LAYER_ACCESS_MODE[MemoryLayer.PROFILE] is MemoryAccessMode.PRIMARY_KEY
        assert LAYER_ACCESS_MODE[MemoryLayer.EPISODE] is MemoryAccessMode.HYBRID_SEARCH
        assert LAYER_ACCESS_MODE[MemoryLayer.SESSION] is MemoryAccessMode.DIRECT_READ

    def test_source_trust_ordering(self):
        """用户纠正 > 用户陈述 > 系统推断。"""
        from app.schemas import SOURCE_TRUST

        assert (
            SOURCE_TRUST[MemorySource.USER_CORRECTION]
            > SOURCE_TRUST[MemorySource.USER_OBSERVATION]
            > SOURCE_TRUST[MemorySource.SYSTEM_INFERENCE]
        )

    def test_retrieval_source_enum_used(self):
        item = MemoryItem(
            event=_mem(), score=0.9, retrieval_source=RetrievalSource.HYBRID
        )
        assert item.retrieval_source is RetrievalSource.HYBRID
