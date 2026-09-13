"""健康模块：监测与分诊，**不是诊断**。

定位见 `docs/DESIGN.md` §5.4 与 `docs/07-health.md`。

## 三不原则（由契约强制，不靠 prompt）

| 原则 | 落点 |
| --- | --- |
| **不诊断** | 不输出疾病名称作为结论；`FORBIDDEN_PHRASES` 拦截排除性表述 |
| **不排除** | 永不输出「健康」「没问题」「正常」；`UrgencyLevel` 没有 `NORMAL` |
| **不阻塞** | 红旗命中时 L3 抢占其他输出（展示分层 A2） |

## 双轨

| | Track A 红旗规则 | Track B 基线漂移 |
| --- | --- | --- |
| 方法 | 确定性规则（`data/health/red_flags.yaml`） | 稳健统计（MAD / CUSUM） |
| 可否学习 | **绝对不可** | 必须学习 |

**为什么红旗不能学习**：学习型系统会把罕见的高代价事件优化掉。
这个理由也决定了**为什么 L3 不适合接 RAG** —— 检索的失败是静默的，
而这里假阴性的代价是猫可能死亡。详见 `docs/DESIGN.md` §5.4.1。

## 本期范围

Track A（红旗规则求值 + 记录落地）已完成。
Track B 需要 21 天个体基线，留接口。
"""

from app.health.redflags import (
    RedFlagRule,
    RedFlagTable,
    RedFlagTableError,
    eval_condition,
    evaluate,
)
from app.health.store import (
    DuplicateHealthRecord,
    HealthRecordStore,
    HealthStoreError,
    InMemoryHealthStore,
)
from app.health.writer import SUBJECT_TO_SIGNAL, HealthAdmission, HealthWriter

__all__ = [
    "RedFlagRule",
    "RedFlagTable",
    "RedFlagTableError",
    "eval_condition",
    "evaluate",
    "HealthRecordStore",
    "InMemoryHealthStore",
    "HealthStoreError",
    "DuplicateHealthRecord",
    "HealthWriter",
    "HealthAdmission",
    "SUBJECT_TO_SIGNAL",
]
