"""守卫契约。

设计全文见 docs/01-architecture.md §3.7、docs/07-health.md §8。

守卫节点是**诚实性边界的技术强制点**——不能只靠 prompt。
「prompt 是建议，节点是强制」。
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ViolationType(str, Enum):
    IDENTITY_MISMATCH = "identity_mismatch"
    """身份一致性：出现了该猫档案中没有的外貌特征（可能是别的猫的特征串入）"""

    UNTRACEABLE_CLAIM = "untraceable_claim"
    """证据可追溯：事实性断言无法回溯到检索结果、测量值或用户输入"""

    OVERCERTAINTY = "overcertainty"
    """过度确定性：把推测表述为确定结论（违反三层信息分离）"""

    NUMERIC_TAMPERING = "numeric_tampering"
    """数值篡改：LLM 修改了评分层计算的概率数字"""

    OUT_OF_SCOPE = "out_of_scope"
    """越界：兽医诊断、用药建议、预后判断等"""

    HEALTH_BOUNDARY = "health_boundary"
    """健康边界：出现「健康」「正常」等排除性表述，或覆盖率不足时给出结论"""

    ROLEPLAY_LEAKAGE = "roleplay_leakage"
    """拟人化泄漏：把拟人化设定层的表达包装成事实"""

    ROLEPLAY_OVERREACH = "roleplay_overreach"
    """拟人化越界：用拟人化表达了**需求或生理状态**。

    与 ``ROLEPLAY_LEAKAGE`` 的区别：泄漏是「把设定说成了事实」，
    越界是「说了不该由拟人化表达的内容」——即使它没被声称成事实。

    判据不是「像不像猫说的话」，而是
    **「用户会不会据此改变对猫的判断或照顾行为」**（DESIGN.md 边界 4）：

    | 内容 | 影响照顾决策 | 允许 |
    | --- | --- | --- |
    | 「本喵心情不错」 | 否 | ✅ |
    | 「本喵想你啦」 | 否 | ✅ |
    | 「本喵饿了」 | 是（可能过量喂食） | ❌ |
    | 「本喵肚子疼」 | 是（可能延误就医） | ❌ |

    ⚠️ **但这只在娱乐层成立。**
    生理词出现在**健康层**（`InfoLayer.HEALTH`）是允许的——
    诊断之外、带严肃语气、且可追溯的硥状描述正是健康层该说的话。
    同一个词在不同层合法或违规，这个规则由 `app/schemas/story.py` 的
    契约强制，不靠 prompt。
    """


class Severity(str, Enum):
    """违规严重度。决定守卫的处理方式。"""

    CRITICAL = "critical"
    """必须重写；重写后仍违规 → 降级为保守回答"""

    MAJOR = "major"
    """重写一次"""

    MINOR = "minor"
    """记录但放行"""


class Violation(BaseModel):
    type: ViolationType
    severity: Severity
    detail: str
    span: str | None = Field(
        default=None, description="触发违规的原文片段，便于定位与展示"
    )
    expected: str | None = Field(
        default=None, description="正确做法，供重写时参考"
    )


class GuardResult(BaseModel):
    """守卫结果。"""

    passed: bool
    violations: list[Violation] = Field(default_factory=list)

    rewritten: str | None = Field(
        default=None, description="重写后的文本（若有 CRITICAL / MAJOR 违规）"
    )
    rewrite_attempts: int = 0

    degrade_to_conservative: bool = Field(
        default=False,
        description=(
            "重写后仍违规 → 降级为保守回答：只陈述已知事实 + 建议观察，不给结论。"
            "这是最后的安全网，必须用户可见。"
        ),
    )
    degraded_notice: str | None = Field(
        default=None,
        description="降级说明。正确性相关的降级必须用户可见（docs/01 §6）。",
    )

    @model_validator(mode="after")
    def _consistency(self) -> GuardResult:
        if self.passed and self.violations:
            critical = [
                v for v in self.violations
                if v.severity in (Severity.CRITICAL, Severity.MAJOR)
            ]
            if critical:
                raise ValueError("存在 CRITICAL/MAJOR 违规时不得 passed=True")
        if self.degrade_to_conservative and not self.degraded_notice:
            raise ValueError("降级为保守回答时必须给出用户可见的说明")
        if self.degrade_to_conservative and self.passed:
            raise ValueError("降级状态不应标记为 passed")
        return self

    @property
    def worst_severity(self) -> Severity | None:
        order = [Severity.MINOR, Severity.MAJOR, Severity.CRITICAL]
        present = [v.severity for v in self.violations]
        for s in reversed(order):
            if s in present:
                return s
        return None

    def summary(self) -> str:
        if self.passed and not self.violations:
            return "通过"
        parts = [f"{v.type.value}({v.severity.value})" for v in self.violations]
        return "、".join(parts)
