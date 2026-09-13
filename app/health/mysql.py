"""健康数据的 MySQL 实现。

## 为什么与 `MySQLStore` 分开、但共用连接池

分开是**数据策略**决定的（见 `app/health/store.py` 的模块 docstring）：
健康数据要静态加密、730 天保留、级联硬删、显式同意 —— 与通用记忆不同。
把两者塞进一个类会让这些差异在日常代码里被慢慢抹平，而它们是合规要求。

共用连接池是因为那只是**资源**，不是策略。两个池只会让连接数翻倍。

## `value` 的三列展开

领域模型里 `HealthRecord.value` 是 `float | bool | str | None`。
这里存成 `value_kind` + 三个具体列，而不是序列化成 JSON：

**红旗求值要按数值比较**（`temperature > 39.5`）。把数字塞进 JSON
就再也用不上索引、比较运算和 CHECK 约束 —— 而那一层正是「数据是否可信」的保证。

读取时 `value_kind` 决定从哪一列取，并由数据库 CHECK 保证
「kind 与哪一列非空」始终一致（见 ddl.sql）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from app.schemas.health import (
    HealthAssessment,
    HealthRecord,
    HealthRecordSource,
    RedFlagHit,
    SignalDeviation,
    UrgencyLevel,
)
from app.store.mysql import MySQLPool, _json_dump, _json_load, _row_field, _to_db

__all__ = ["MySQLHealthRecordStore"]


def _from_db(dt: datetime | None) -> datetime | None:
    """naive UTC → aware UTC（与 MySQLStore 同一语义）。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _split_value(value: float | bool | str | None) -> tuple[str, Any, Any, Any]:
    """把联合类型的 value 拆成 (kind, number, bool, text)。

    ⚠️ **`bool` 必须在 `int` 之前判断** —— Python 里 `True` 是 `int` 的实例，
    顺序反了会把布尔值存进 numeric 列，而 `1.0` 在红旗求值里
    与「用户点了是」是完全不同的语义。
    """
    if value is None:
        return ("absent", None, None, None)
    if isinstance(value, bool):
        return ("bool", None, int(value), None)
    if isinstance(value, (int, float)):
        return ("number", float(value), None, None)
    return ("text", None, None, str(value))


def _join_value(
    kind: str, number: Any, boolean: Any, text: Any
) -> float | bool | str | None:
    """三列 → 联合类型。"""
    if kind == "number":
        return float(number) if number is not None else None
    if kind == "bool":
        return bool(boolean) if boolean is not None else None
    if kind == "text":
        return str(text) if text is not None else None
    return None


class MySQLHealthRecordStore:
    """`HealthRecordStore` 的 MySQL 实现。"""

    def __init__(self, *, pool: MySQLPool) -> None:
        self._pool = pool

    def describe(self) -> dict[str, str]:
        return {"kind": "mysql", "dsn": self._pool.descriptor, "table": "health_records"}

    # ── 记录 ──────────────────────────────────────────────

    def insert_record(self, record: HealthRecord) -> HealthRecord:
        record_id = record.record_id or str(uuid4())
        stored = record.model_copy(update={"record_id": record_id})
        kind, number, boolean, text = _split_value(stored.value)

        sql = """
            INSERT INTO health_records (
                record_id, user_id, pet_id, session_id, `signal`,
                value_kind, value_number, value_bool, value_text,
                unit, recorded_at, source, `sensitive`, consent_version,
                retention_days, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self._pool.acquire() as conn, conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        record_id,
                        stored.user_id,
                        stored.pet_id,
                        stored.session_id,
                        stored.signal,
                        kind,
                        number,
                        boolean,
                        text,
                        stored.unit,
                        _to_db(stored.recorded_at),
                        stored.source.value
                        if hasattr(stored.source, "value")
                        else str(stored.source),
                        int(bool(stored.sensitive)),
                        stored.consent_version,
                        stored.retention_days,
                        _to_db(stored.created_at),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            from app.health.store import DuplicateHealthRecord

            args = getattr(exc, "args", ())
            if args and args[0] == 1062:
                raise DuplicateHealthRecord(f"健康记录已存在：{record_id}") from exc
            raise
        return stored

    def list_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[HealthRecord]:
        sql = "SELECT * FROM health_records WHERE user_id = %s AND pet_id = %s"
        params: list[Any] = [user_id, pet_id]
        if since is not None:
            sql += " AND recorded_at >= %s"
            params.append(_to_db(since))
        sql += " ORDER BY recorded_at"
        # LIMIT 放最后：先按时间正序取前 N 条（最早 N 条），
        # 而不是「最近的 N 条」—— 与内存实现保持一致。
        if limit is not None:
            sql += " LIMIT %s"
            params.append(int(limit))

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_record(r) for r in rows]

    # ── 评估 ──────────────────────────────────────────────

    def save_assessment(self, assessment: HealthAssessment) -> HealthAssessment:
        assessment_id = assessment.assessment_id or str(uuid4())
        stored = assessment.model_copy(update={"assessment_id": assessment_id})

        sql = """
            INSERT INTO health_assessments (
                assessment_id, user_id, pet_id, as_of, level, coverage,
                coverage_note, signals_assessed, signals_missing, findings,
                red_flags_triggered, recommendation, disclaimer,
                must_not_be_read_as, rule_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                level = VALUES(level), coverage = VALUES(coverage),
                coverage_note = VALUES(coverage_note),
                signals_assessed = VALUES(signals_assessed),
                signals_missing = VALUES(signals_missing),
                findings = VALUES(findings),
                red_flags_triggered = VALUES(red_flags_triggered),
                recommendation = VALUES(recommendation),
                rule_version = VALUES(rule_version)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    assessment_id,
                    stored.user_id,
                    stored.pet_id,
                    _to_db(stored.as_of),
                    stored.level.value if hasattr(stored.level, "value") else stored.level,
                    float(stored.coverage),
                    stored.coverage_note,
                    _json_dump(list(stored.signals_assessed)),
                    _json_dump(list(stored.signals_missing)),
                    _dump_models(stored.findings),
                    _dump_models(stored.red_flags_triggered),
                    stored.recommendation,
                    stored.disclaimer,
                    stored.must_not_be_read_as,
                    stored.rule_version,
                ),
            )
        return stored

    def list_assessments(
        self, *, user_id: str, pet_id: str, limit: int | None = None
    ) -> list[HealthAssessment]:
        sql = (
            "SELECT * FROM health_assessments "
            "WHERE user_id = %s AND pet_id = %s ORDER BY as_of"
        )
        params: list[Any] = [user_id, pet_id]
        if limit is not None:
            # 评估取**最近** N 条（与记录相反）—— 与内存实现一致。
            # 直接用 ORDER BY as_of DESC LIMIT 会返回倒序，所以要反转。
            sql = (
                "SELECT * FROM health_assessments "
                "WHERE user_id = %s AND pet_id = %s ORDER BY as_of DESC LIMIT %s"
            )
            params.append(int(limit))

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())

        if limit is not None:
            rows.reverse()
        return [_row_to_assessment(r) for r in rows]

    # ── 删除 ──────────────────────────────────────────────

    def delete_pet_data(self, *, user_id: str, pet_id: str) -> int:
        """**级联硬删**：记录 + 派生评估一起清掉。

        软标记不满足要求 —— 一条标了 `deleted=True` 的健康记录
        仍然是泄露风险（`HealthDataPolicy.on_account_deletion`）。
        """
        removed = 0
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                "DELETE FROM health_records WHERE user_id = %s AND pet_id = %s",
                (user_id, pet_id),
            )
            removed += cur.rowcount or 0
            cur.execute(
                "DELETE FROM health_assessments WHERE user_id = %s AND pet_id = %s",
                (user_id, pet_id),
            )
            removed += cur.rowcount or 0
        return removed


# =============================================================================
# 行映射
# =============================================================================


def _dump_models(items: Sequence[Any]) -> str:
    return json.dumps(
        [i.model_dump(mode="json") if hasattr(i, "model_dump") else i for i in items],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _row_to_record(row: Any) -> HealthRecord:
    return HealthRecord(
        record_id=_row_field(row, "record_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        signal=_row_field(row, "signal"),
        value=_join_value(
            _row_field(row, "value_kind"),
            _row_field(row, "value_number"),
            _row_field(row, "value_bool"),
            _row_field(row, "value_text"),
        ),
        unit=_row_field(row, "unit"),
        recorded_at=_from_db(_row_field(row, "recorded_at")),
        source=HealthRecordSource(_row_field(row, "source")),
        sensitive=bool(_row_field(row, "sensitive")),
        consent_version=_row_field(row, "consent_version"),
        retention_days=_row_field(row, "retention_days"),
        created_at=_from_db(_row_field(row, "created_at")),
    )


def _row_to_assessment(row: Any) -> HealthAssessment:
    level = _row_field(row, "level")
    return HealthAssessment(
        assessment_id=_row_field(row, "assessment_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        as_of=_from_db(_row_field(row, "as_of")),
        level=UrgencyLevel(level),
        coverage=float(_row_field(row, "coverage")),
        coverage_note=_row_field(row, "coverage_note"),
        signals_assessed=list(_json_load(_row_field(row, "signals_assessed"), [])),
        signals_missing=list(_json_load(_row_field(row, "signals_missing"), [])),
        findings=[
            SignalDeviation.model_validate(f)
            for f in _json_load(_row_field(row, "findings"), [])
        ],
        red_flags_triggered=[
            RedFlagHit.model_validate(f)
            for f in _json_load(_row_field(row, "red_flags_triggered"), [])
        ],
        recommendation=_row_field(row, "recommendation"),
        disclaimer=_row_field(row, "disclaimer"),
        must_not_be_read_as=_row_field(row, "must_not_be_read_as"),
        rule_version=_row_field(row, "rule_version"),
    )
