"""规划层契约：任务分解、依赖分析、调度与结果合并。

设计全文见 docs/09-intent-and-planning.md，架构位置见 docs/01-architecture.md §6/§7。

核心设计要点：

1. **依赖有三种**：数据依赖（谁的数据喂给谁）、**先验依赖**（谁的结果改变谁的输入分布）、
   资源依赖（互斥资源）。先验依赖是最容易被忽略、也最能体现分解能力的地方。
2. **先验依赖有明确机制**（本模块修复自 docs/10-self-review.md F1）：
   产出方必须给出 ``PriorAdjustment``，消费方在开始执行前应用。
   ``PlanResult`` 结构性地禁止静默丢弃先验依赖。
3. **waves 是执行层**：同一 wave 内可并行，跨 wave 必须串行。拓扑正确性由校验器保证。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from app.schemas.trace import NodeError


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────
# 子任务
# ─────────────────────────────────────────────────────────────


class SubTaskKind(str, Enum):
    """子任务类型。与能力节点一一对应，便于 planner 生成与调度器派发。"""

    RETRIEVE_MEMORY = "retrieve_memory"
    INTERPRET_BEHAVIOR = "interpret_behavior"
    AGGREGATE_QUERY = "aggregate_query"
    """时间线聚合统计（如「本月 vs 上月叫声次数」）。
    注意：这是**聚合**而非相似度检索，纯向量 RAG 无法回答。"""

    RECORD_EVENT = "record_event"
    UPDATE_PROFILE = "update_profile"
    COMPANION_REPLY = "companion_reply"


class DependencyKind(str, Enum):
    """依赖类型。区分这三者是本设计的关键。"""

    DATA = "data"
    """数据依赖：前置任务的输出直接作为后置任务的输入。"""

    PRIOR = "prior"
    """**先验依赖**：前置任务的输出不进入后置任务的输入数据，
    但会**改变后置任务输入的概率分布**（如先验、阈值、权重）。

    例：聚合统计「本月叫声显著多于上月」会抬高
    ``interpret_behavior`` 中 ``door_attention`` 的场景先验，
    因此二者不能完全并行。

    机制：产出方在 ``SubTaskResult.prior_adjustments`` 中给出
    ``PriorAdjustment``；消费方在开始执行前读取并应用（见 docs/01 §6.2.1）。"""

    RESOURCE = "resource"
    """资源依赖：访问互斥资源（如写入同一宠物档案），必须串行以避免竞争。"""


class PriorTargetField(str, Enum):
    """可被先验调整的目标字段。**限制目标集合**，避免先验注入变成任意干预。

    只允许调整「概率/权重/阈值」，不允许直接写入结论。
    """

    SCENE_PRIOR = "scene_prior"
    """行为解释的场景先验（如 door_attention 的先验权重）"""

    CONTEXT_PRIOR = "context_prior"
    """情境分类的先验分布"""

    REJECTION_THRESHOLD = "rejection_threshold"
    """拒答阈值：检索质量差时应更倾向说「没有记录」"""

    RETRIEVAL_K = "retrieval_k"
    """检索数量"""

    FEATURE_WEIGHT = "feature_weight"
    """声学特征在似然计算中的权重"""


class PriorOperation(str, Enum):
    SCALE = "scale"
    """乘性调整：new = old * value"""

    DELTA = "delta"
    """加性调整：new = old + value"""

    SET = "set"
    """直接设定：new = value（仅用于阈值类）"""


class PriorAdjustment(BaseModel):
    """**先验依赖的实现机制**（docs/10-self-review.md F1）。

    一个子任务产出的、对其他子任务先验的调整量。
    这是「先验依赖」从名词变成可测试机制的关键。
    """

    target_task_id: str = Field(description="该调整作用于哪个子任务")
    target_field: PriorTargetField
    target_key: str | None = Field(
        default=None,
        description="更细的目标定位，如场景标签 'door_attention'。"
        "为 None 表示作用于该字段整体。",
    )
    operation: PriorOperation
    value: float
    rationale: str = Field(
        description="调整依据。**必填**——无法说明依据的先验调整不可接受（供审计与评测）"
    )
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _rationale_required(self) -> PriorAdjustment:
        if not self.rationale.strip():
            raise ValueError("先验调整必须说明依据，否则无法审计")
        return self

    @model_validator(mode="after")
    def _key_required_for_prior_field(self) -> PriorAdjustment:
        """场景/情境先验必须指定具体 key，否则调整对象不明确。"""
        needs_key = (
            PriorTargetField.SCENE_PRIOR,
            PriorTargetField.CONTEXT_PRIOR,
            PriorTargetField.FEATURE_WEIGHT,
        )
        if self.target_field in needs_key and not self.target_key:
            raise ValueError(
                f"target_field={self.target_field.value} 必须指定 target_key"
            )
        return self

    def apply_to(self, current: float) -> float:
        """应用调整。集中在一处，便于测试与复算。"""
        if self.operation is PriorOperation.SCALE:
            return current * self.value
        if self.operation is PriorOperation.DELTA:
            return current + self.value
        return self.value

    @property
    def ref(self) -> str:
        return f"{self.target_task_id}:{self.target_field.value}" + (
            f"/{self.target_key}" if self.target_key else ""
        )


class Dependency(BaseModel):
    """一条依赖声明。"""

    on_task_id: str
    kind: DependencyKind
    note: str = Field(description="说明依赖内容，供审计与评测比对")

    @property
    def is_prior(self) -> bool:
        return self.kind is DependencyKind.PRIOR


class SubTask(BaseModel):
    """一个可独立执行、可独立失败、可独立评测的子任务。"""

    task_id: str = Field(description="如 'T1'，在同一 Plan 内唯一")
    kind: SubTaskKind
    params: dict = Field(
        default_factory=dict,
        description="执行所需槽位，如 {'scene': 'door', 'pet_id': 'p1'}",
    )
    depends_on: list[Dependency] = Field(default_factory=list)

    optional: bool = Field(
        default=False,
        description="失败是否可容忍。True 时失败不阻塞整体计划，标记为 SKIPPED。",
    )
    rationale: str = Field(default="", description="为什么需要这个子任务（可审计）")

    @property
    def upstream_ids(self) -> set[str]:
        return {d.on_task_id for d in self.depends_on}

    @property
    def prior_upstream_ids(self) -> set[str]:
        """仅先验依赖的上游。这些任务必须先完成，但产出不进入本任务的输入数据。"""
        return {d.on_task_id for d in self.depends_on if d.is_prior}


# ─────────────────────────────────────────────────────────────
# 计划
# ─────────────────────────────────────────────────────────────


class Plan(BaseModel):
    """规划层的输出：分解结果 + 拓扑分层。"""

    plan_id: str | None = None
    user_id: str
    pet_id: str

    subtasks: list[SubTask]
    waves: list[list[str]] = Field(
        description=(
            "拓扑分层。同一 wave 内的 task_id 可并行执行；"
            "跨 wave 必须严格串行。由依赖分析产出。"
        )
    )

    decomposition_note: str = Field(
        description="分解理由的自然语言说明。用于审计与「过度分解率」评测。"
    )
    is_compound: bool = Field(
        default=True,
        description="False 表示这是单意图的快路径计划（仅一个 wave、一个子任务）",
    )

    created_at: datetime = Field(default_factory=_now)

    _task_index: dict[str, SubTask] = PrivateAttr(default_factory=dict)

    # ── 结构性约束：计划必须自洽 ──────────────────────────────

    @model_validator(mode="after")
    def _unique_task_ids(self) -> Plan:
        ids = [t.task_id for t in self.subtasks]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"task_id 重复：{dupes}")
        return self

    @model_validator(mode="after")
    def _waves_cover_all_exactly_once(self) -> Plan:
        flat = [tid for wave in self.waves for tid in wave]
        if len(flat) != len(set(flat)):
            raise ValueError("同一个 task_id 不得出现在多个 wave 中")
        declared = {t.task_id for t in self.subtasks}
        if set(flat) != declared:
            missing = declared - set(flat)
            extra = set(flat) - declared
            raise ValueError(
                f"waves 与 subtasks 不一致：缺失 {missing or '无'}，多余 {extra or '无'}"
            )
        return self

    @model_validator(mode="after")
    def _topological_order_respected(self) -> Plan:
        """依赖必须排在更早的 wave 中，否则计划不可执行。"""
        position = {
            tid: idx for idx, wave in enumerate(self.waves) for tid in wave
        }
        for task in self.subtasks:
            for dep in task.depends_on:
                if dep.on_task_id not in position:
                    raise ValueError(
                        f"{task.task_id} 依赖不存在的任务 {dep.on_task_id}"
                    )
                if position[dep.on_task_id] >= position[task.task_id]:
                    raise ValueError(
                        f"拓扑序违规：{task.task_id} 依赖 {dep.on_task_id}，"
                        f"但后者排在同一或更晚的 wave"
                    )
        return self

    @model_validator(mode="after")
    def _build_index(self) -> Plan:
        self._task_index = {t.task_id: t for t in self.subtasks}
        return self

    # ── 查询接口（docs/10-self-review.md F4：文档曾引用不存在的方法）──

    def task(self, task_id: str) -> SubTask | None:
        """按 id 取子任务。不存在返回 None。

        优先走私有索引；索引未命中时回退线性扫描并补建。
        这层防御是因为 ``Plan`` 是 DTO，不对调用方做出不可变承诺——
        若外部直接修改 ``subtasks``，私有索引会失效。
        （注意：此类修改同时会让 ``waves`` 与 ``subtasks`` 不一致，
        计划本身已无效，但查询接口不应因此静默返回错误结果。）
        """
        hit = self._task_index.get(task_id)
        if hit is not None:
            return hit
        for t in self.subtasks:
            if t.task_id == task_id:
                self._task_index[task_id] = t
                return t
        return None

    def require_task(self, task_id: str) -> SubTask:
        task = self.task(task_id)
        if task is None:
            raise KeyError(f"计划中不存在子任务 {task_id}")
        return task

    def wave_tasks(self, wave_idx: int) -> list[SubTask]:
        """取某一 wave 的全部子任务（该 wave 内可并行执行）。"""
        if not 0 <= wave_idx < len(self.waves):
            raise IndexError(f"wave 下标越界：{wave_idx}（共 {len(self.waves)} 个）")
        return [self.require_task(tid) for tid in self.waves[wave_idx]]

    def first_wave(self) -> list[SubTask]:
        return self.wave_tasks(0)

    def next_wave_index(self, wave_idx: int) -> int | None:
        """下一 wave 下标；已是最后一 wave 则返回 None（执行结束）。"""
        nxt = wave_idx + 1
        return nxt if nxt < len(self.waves) else None

    # ── 派生信息 ─────────────────────────────────────────────

    @property
    def max_parallelism(self) -> int:
        """最大并行度 = 最宽 wave 的宽度。用于「并行收益」评测。"""
        return max((len(w) for w in self.waves), default=0)

    @property
    def serial_depth(self) -> int:
        """串行深度 = wave 数量。用于估算串行执行延迟。"""
        return len(self.waves)

    def prior_dependencies(self) -> list[tuple[str, str, Dependency]]:
        """全部先验依赖，返回 (产出方, 消费方, 依赖声明)。

        评测「依赖正确率」时重点比对这一类。
        """
        return [
            (d.on_task_id, t.task_id, d)
            for t in self.subtasks
            for d in t.depends_on
            if d.is_prior
        ]

    def declared_prior_refs(self) -> list[str]:
        """先验依赖的规范字符串形式，如 'T3->T2'。"""
        return [f"{producer}->{consumer}" for producer, consumer, _ in self.prior_dependencies()]


# ─────────────────────────────────────────────────────────────
# 执行结果
# ─────────────────────────────────────────────────────────────


class SubTaskStatus(str, Enum):
    SUCCESS = "success"
    DEGRADED = "degraded"
    """完成但能力降级（如声学特征提取失败，退化为文本推断）。"""

    FAILED = "failed"
    SKIPPED = "skipped"
    """因前置依赖失败或 optional 而跳过。"""


class SubTaskResult(BaseModel):
    task_id: str
    kind: SubTaskKind
    status: SubTaskStatus

    output: dict | None = None
    error: NodeError | None = None
    degraded_note: str | None = Field(
        default=None,
        description="降级说明。**正确性相关的降级必须可见**（docs/01 §8）。",
    )
    latency_ms: int | None = None

    prior_adjustments: list[PriorAdjustment] = Field(
        default_factory=list,
        description=(
            "本任务对其他子任务先验的调整（docs/01 §6.2.1）。"
            "仅当本任务是某个 PRIOR 依赖的产出方时才有值。"
        ),
    )

    @model_validator(mode="after")
    def _degraded_requires_note(self) -> SubTaskResult:
        if self.status is SubTaskStatus.DEGRADED and not self.degraded_note:
            raise ValueError("DEGRADED 状态必须说明降级内容，否则用户无法察觉")
        return self

    def adjustments_for(self, task_id: str) -> list[PriorAdjustment]:
        """取作用于指定子任务的先验调整。"""
        return [a for a in self.prior_adjustments if a.target_task_id == task_id]

    @property
    def is_usable(self) -> bool:
        return self.status in (SubTaskStatus.SUCCESS, SubTaskStatus.DEGRADED)


class PriorSkip(BaseModel):
    """先验依赖被跳过时的说明。**不允许静默丢弃。**"""

    dependency: str = Field(description="如 'T3->T2'")
    reason: str


class PlanResult(BaseModel):
    """`result_merger` 节点的输出。"""

    plan_id: str | None = None
    results: list[SubTaskResult] = Field(default_factory=list)

    merge_notes: list[str] = Field(
        default_factory=list,
        description="合并期的冲突/冗余检测结论，如「聚合结论已用于调整解释先验」",
    )

    declared_prior_dependencies: list[str] = Field(
        default_factory=list,
        description="计划中声明的先验依赖，如 ['T3->T2']。用于对账。",
    )
    applied_prior_dependencies: list[str] = Field(
        default_factory=list,
        description="实际生效的先验依赖。用于验证依赖分析是否真的被执行，而非停留在纸面。",
    )
    skipped_prior_dependencies: list[PriorSkip] = Field(
        default_factory=list,
        description="未生效的先验依赖及原因。**必须给出原因**，否则依赖分析失去意义。",
    )

    @model_validator(mode="after")
    def _prior_deps_fully_accounted(self) -> PlanResult:
        """先验依赖必须被完整处置：要么应用，要么说明为何跳过。

        这条不变量解决的是 docs/10-self-review.md §2 指出的问题——
        「先验依赖」如果只是文档里的名词，静默丢弃时无人察觉。
        """
        declared = set(self.declared_prior_dependencies)
        accounted = set(self.applied_prior_dependencies) | {
            s.dependency for s in self.skipped_prior_dependencies
        }
        missing = declared - accounted
        if missing:
            raise ValueError(
                f"先验依赖未被处置：{sorted(missing)}。"
                "既未应用也未说明跳过原因，会让依赖分析失去意义。"
            )
        extra = accounted - declared
        if extra:
            raise ValueError(f"处置了未声明的先验依赖：{sorted(extra)}")
        overlap = set(self.applied_prior_dependencies) & {
            s.dependency for s in self.skipped_prior_dependencies
        }
        if overlap:
            raise ValueError(f"同一先验依赖既标记应用又标记跳过：{sorted(overlap)}")
        return self

    # ── 派生信息 ─────────────────────────────────────────────

    @property
    def failed(self) -> list[SubTaskResult]:
        return [r for r in self.results if r.status is SubTaskStatus.FAILED]

    @property
    def skipped(self) -> list[SubTaskResult]:
        return [r for r in self.results if r.status is SubTaskStatus.SKIPPED]

    @property
    def degraded(self) -> list[SubTaskResult]:
        return [r for r in self.results if r.status is SubTaskStatus.DEGRADED]

    @property
    def is_partial_success(self) -> bool:
        """部分成功：有可用项，也有失败/跳过项。

        处理原则：返回已完成部分，并**明确说明哪部分失败**（docs/09 §9.4）。
        """
        succeeded = [r for r in self.results if r.is_usable]
        incomplete = self.failed + self.skipped
        return bool(succeeded) and bool(incomplete)

    def result_for(self, task_id: str) -> SubTaskResult | None:
        for r in self.results:
            if r.task_id == task_id:
                return r
        return None

    def render_partial_notice(self) -> str:
        """生成「部分完成」的用户可见说明。"""
        if not self.is_partial_success:
            return ""
        parts = []
        if self.failed:
            parts.append("未能完成：" + "、".join(r.task_id for r in self.failed))
        if self.skipped:
            parts.append("已跳过：" + "、".join(r.task_id for r in self.skipped))
        if self.degraded:
            notes = "；".join(
                r.degraded_note for r in self.degraded if r.degraded_note
            )
            parts.append(f"降级完成：{notes}")
        return " | ".join(parts)
