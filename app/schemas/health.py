"""健康模块契约：监测与分诊，**不是诊断**。

边界依据《动物诊疗机构管理办法》第二条 / 第五条 / 第十九条，
详见 docs/07-health.md。

设计要点：
- 级别枚举的名称本身就承载语义（没有 `NORMAL` / `HEALTHY`，只有 `NO_DEVIATION_DETECTED`），
  从类型层面阻止「输出健康」这一危险行为。
- 覆盖率不足时结构上禁止输出结论（见 `HealthAssessment` 的校验器）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

#: 覆盖率低于此值时只允许输出「数据不足，无法评估」。
MIN_COVERAGE_FOR_ASSESSMENT = 0.3

#: 任何情况下都禁止出现在健康输出中的表述。
#: 「没检测到异常」与「没有异常」是两件事，把前者说成后者是本模块最危险的失败模式。
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "健康",
    "正常",
    "没问题",
    "没问题了",
    "放心",
    "不用担心",
    "没事",
    "不用去医院",
    "无需就医",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UrgencyLevel(str, Enum):
    """分诊级别。

    刻意不提供 ``NORMAL`` / ``HEALTHY``：系统只能声明「未发现偏离」，
    不能声明「健康」。
    """

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    """数据覆盖率不足，无法评估。绝不可降级表述为「未发现异常」。"""

    NO_DEVIATION_DETECTED = "NO_DEVIATION_DETECTED"
    """已评估但未发现偏离。**注意：这不等于健康。**"""

    OBSERVE = "L1"
    """轻微或单次偏离，建议继续记录观察。"""

    VET_VISIT_RECOMMENDED = "L2"
    """持续或多信号偏离，建议预约兽医。"""

    EMERGENCY = "L3"
    """红旗命中。必须抢占式展示，不可被弱化。"""


class HealthRecordSource(str, Enum):
    USER_INPUT = "user_input"
    ACOUSTIC_PIPELINE = "acoustic_pipeline"
    """复用行为解释器的声学特征，作为叫声相关的健康信号。"""

    VISION = "vision"
    WEIGHT_LOG = "weight_log"


class HealthRecord(BaseModel):
    """一条健康信号记录。

    属于健康敏感数据：加密存储、用户可导出与删除（见 docs/07-health.md §H4）。
    """

    record_id: str | None = None
    user_id: str
    pet_id: str
    session_id: str | None = Field(
        default=None,
        description=(
            "产生这条健康信号的会话。\n\n"
            "**结构化上报时有值，日报聚合时为 None** —— 日报跨会话，\n"
            "给它填一个具体会话会是错误的归属。\n\n"
            "用途是追溯「这个信号是在哪次交流里提到的」，与红旗求值无关。"
        ),
    )

    signal: str = Field(
        description="必须存在于 red_flags.yaml 的 signal_vocabulary 中，如 litter_box.urine_output"
    )
    value: float | bool | str | None = None
    unit: str | None = None

    recorded_at: datetime
    source: HealthRecordSource

    sensitive: bool = Field(default=True, description="健康数据默认标记为敏感")
    consent_version: str | None = Field(
        default=None, description="记录采集时用户同意的隐私政策版本"
    )
    retention_days: int | None = Field(
        default=None, description="保留期；None 表示遵循账号级策略"
    )
    created_at: datetime = Field(default_factory=_now)


class SignalDeviation(BaseModel):
    """Track B 的产出：相对该个体基线的偏离。

    本期 Track B 仅留接口 + 合成数据演示（见 docs/07-health.md §12）。
    """

    signal: str
    direction: str = Field(description="increase / decrease")
    magnitude: float = Field(description="稳健 z 分数（基于中位数与 MAD）")
    duration_days: int
    baseline_window: str = Field(description="基线窗口，如 '2026-08-22/2026-09-11'")
    baseline_median: float | None = None
    observed_value: float | None = None
    note: str


class RedFlagHit(BaseModel):
    """Track A 的产出：命中的红旗规则。不可由 LLM 生成。"""

    rule_id: str
    urgency: UrgencyLevel
    title: str
    message: str
    action: str
    matched_conditions: list[str] = Field(default_factory=list)
    aggravating_conditions: list[str] = Field(default_factory=list)
    rule_version: str
    sources: list[str] = Field(default_factory=list)


class HealthAssessment(BaseModel):
    """健康评估结果。"""

    assessment_id: str | None = None
    user_id: str
    pet_id: str
    as_of: datetime = Field(default_factory=_now)

    level: UrgencyLevel
    coverage: float = Field(ge=0.0, le=1.0)
    coverage_note: str

    signals_assessed: list[str] = Field(default_factory=list)
    signals_missing: list[str] = Field(default_factory=list)

    findings: list[SignalDeviation] = Field(default_factory=list)
    red_flags_triggered: list[RedFlagHit] = Field(default_factory=list)

    recommendation: str
    disclaimer: str = Field(
        default=("本结果为基于行为记录的偏离提示，不是诊断，不能替代兽医检查。"),
        min_length=1,
    )
    must_not_be_read_as: str = Field(
        default="未发现异常不代表健康",
        min_length=1,
    )

    rule_version: str | None = None

    # ── 结构性约束：把诚实性原则编码进类型 ──────────────────────

    @field_validator("recommendation", "coverage_note")
    @classmethod
    def _no_forbidden_phrases(cls, v: str) -> str:
        for phrase in FORBIDDEN_PHRASES:
            if phrase in v:
                raise ValueError(
                    f"健康输出中禁止出现「{phrase}」。"
                    "健康模块不得声明健康或排除疾病（见 docs/07-health.md §2.2）。"
                )
        return v

    @model_validator(mode="after")
    def _enforce_coverage_gate(self) -> HealthAssessment:
        """覆盖率不足时，**不得声称「未发现偏离」**。

        ## 两个方向必须分开看（这里很容易写宽，而写宽的后果是危险的）

        | 情形 | 正确输出 | 理由 |
        | --- | --- | --- |
        | 覆盖率低 + **无命中** | `INSUFFICIENT_DATA` | 「没测」不能变成「没事」 |
        | 覆盖率低 + **有命中** | **保留命中级别** | 已存在的危险信号不因「别的没测」而失效 |

        初版把所有非 `INSUFFICIENT_DATA` 的级别都拦了 ——
        于是一个已命中的急诊红旗会被压成「数据不足」。

        `UrgencyLevel.INSUFFICIENT_DATA` 的 docstring 写的是
        「绝不可**降级表述**为「未发现异常」」。
        **把 EMERGENCY 降级成 INSUFFICIENT_DATA 也是一次降级，
        而且是危险的那个方向** —— 假阴性的代价是猫可能死亡（非对称代价）。

        这与 B11/B12（检测过宽）是同一类错误：
        为了防住一个方向，把反方向也拦住了。
        """
        if self.coverage < MIN_COVERAGE_FOR_ASSESSMENT:
            if (
                not self.red_flags_triggered
                and self.level != UrgencyLevel.INSUFFICIENT_DATA
            ):
                raise ValueError(
                    f"覆盖率 {self.coverage:.2f} < {MIN_COVERAGE_FOR_ASSESSMENT} "
                    "且无红旗命中时，级别必须为 INSUFFICIENT_DATA。"
                    "禁止把「没有数据」伪装成「没有异常」。"
                )
            if self.findings:
                raise ValueError("覆盖率不足时不得输出任何 findings")
        return self

    @model_validator(mode="after")
    def _enforce_emergency_consistency(self) -> HealthAssessment:
        if self.level == UrgencyLevel.EMERGENCY and not self.red_flags_triggered:
            raise ValueError("EMERGENCY 级别必须由红旗规则触发，不可由模型判定")
        return self

    @model_validator(mode="after")
    def _no_healthy_when_never_say_healthy(self) -> HealthAssessment:
        """命中的红旗若声明 never_say_healthy，则输出中不得出现安慰性表述。"""
        if self.red_flags_triggered and any(
            phrase in self.recommendation for phrase in FORBIDDEN_PHRASES
        ):
            raise ValueError("红旗命中时禁止出现安慰性表述")
        return self


class HealthDataPolicy(BaseModel):
    """健康数据的隐私策略。

    面试价值：健康数据的泄露后果远重于聊天记录，因此单独定义策略，
    而不是沿用通用数据策略。
    """

    encrypted_at_rest: bool = True
    user_can_export: bool = True
    user_can_delete: bool = True
    default_retention_days: int = 730
    on_account_deletion: str = "cascade_hard_delete"
    requires_explicit_consent: bool = True
    note: str = (
        "健康记录属于敏感数据。采集需显式同意；用户可导出与删除；"
        "删除为硬删除（非软标记），且需级联至由该记录派生的评估结果。"
    )
