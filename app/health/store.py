"""健康数据存储。

## 为什么与 `MemoryStore` 分开

不是洁癖，是**数据策略不同**（见 `HealthDataPolicy`）：

| | 通用记忆 | 健康记录 |
| --- | --- | --- |
| 加密 | 常规 | **必须静态加密** |
| 保留期 | 不定 | **730 天** |
| 删除 | 单条软删 | **级联硬删**（含派生评估） |
| 同意 | 隐含 | **需显式同意**（`consent_version`） |

共用一个 store 会让这些差异被日常代码慢慢抹平 —— 而它们是合规要求，不是偏好。

## 「另走 health_records 表」这句话在代码里的落点

`app/memory/flywheel.py` 会拒绝 `EventType.HEALTH` 进通用记忆，理由是
「健康信号另走 health_records 表，避免检索时混入」—— 本模块就是那张表。

在此之前这个理由只是**一句话**：拒掉了但没人接。现在有接收方了。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from app.schemas.health import HealthAssessment, HealthRecord


class HealthStoreError(Exception):
    pass


class DuplicateHealthRecord(HealthStoreError):
    pass


@runtime_checkable
class HealthRecordStore(Protocol):
    """健康记录与评估的存储接口。

    所有方法都必须按 ``user_id`` + ``pet_id`` 过滤 ——
    多租户隔离在这张表上比在通用记忆上更要紧（健康数据更敏感）。
    """

    def insert_record(self, record: HealthRecord) -> HealthRecord: ...

    def list_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HealthRecord]: ...

    def save_assessment(self, assessment: HealthAssessment) -> HealthAssessment: ...

    def list_assessments(
        self, *, user_id: str, pet_id: str, limit: int | None = None
    ) -> list[HealthAssessment]: ...

    def delete_pet_data(self, *, user_id: str, pet_id: str) -> int:
        """**硬删除**该宠物的全部健康数据，返回删除条数。

        `HealthDataPolicy.on_account_deletion == "cascade_hard_delete"`：
        连派生的评估结果一起删。软标记不满足要求 ——
        一条标了 `deleted=True` 的健康记录仍然是泄露风险。
        """
        ...


class InMemoryHealthStore:
    """内存实现。用于测试与离线开发。

    **不满足生产要求**（无加密、无持久化），但它把接口固定下来，
    使调用方不必等到真实数据库就位就能被测试覆盖。
    """

    def __init__(self) -> None:
        self._records: dict[str, HealthRecord] = {}
        self._assessments: dict[str, HealthAssessment] = {}
        self._seq = 0

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq:05d}"

    def insert_record(self, record: HealthRecord) -> HealthRecord:
        stored = record
        if not stored.record_id:
            stored = record.model_copy(update={"record_id": self._next_id("hr")})
        assert stored.record_id is not None  # noqa: S101
        if stored.record_id in self._records:
            raise DuplicateHealthRecord(f"健康记录已存在：{stored.record_id}")
        self._records[stored.record_id] = stored
        return stored

    def list_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HealthRecord]:
        out = [
            r
            for r in self._records.values()
            if r.user_id == user_id and r.pet_id == pet_id
        ]
        if since is not None:
            out = [r for r in out if r.recorded_at >= since]
        out.sort(key=lambda r: r.recorded_at)
        return out[:limit] if limit is not None else out

    def save_assessment(self, assessment: HealthAssessment) -> HealthAssessment:
        stored = assessment
        if not stored.assessment_id:
            stored = assessment.model_copy(
                update={"assessment_id": self._next_id("ha")}
            )
        assert stored.assessment_id is not None  # noqa: S101
        self._assessments[stored.assessment_id] = stored
        return stored

    def list_assessments(
        self, *, user_id: str, pet_id: str, limit: int | None = None
    ) -> list[HealthAssessment]:
        out = [
            a
            for a in self._assessments.values()
            if a.user_id == user_id and a.pet_id == pet_id
        ]
        out.sort(key=lambda a: a.as_of)
        return out[-limit:] if limit is not None else out

    def delete_pet_data(self, *, user_id: str, pet_id: str) -> int:
        """级联硬删：记录 + 评估一起清掉。"""
        record_ids = [
            k
            for k, v in self._records.items()
            if v.user_id == user_id and v.pet_id == pet_id
        ]
        assessment_ids = [
            k
            for k, v in self._assessments.items()
            if v.user_id == user_id and v.pet_id == pet_id
        ]
        for k in record_ids:
            del self._records[k]
        for k in assessment_ids:
            del self._assessments[k]
        return len(record_ids) + len(assessment_ids)
