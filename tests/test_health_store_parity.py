"""健康存储的契约一致性测试。

与 `test_store_parity.py` 同一思路，但针对 `HealthRecordStore`。
单独成文是因为**数据策略不同**：健康数据要静态加密、730 天保留、
级联硬删、显式同意（见 `app/health/store.py` 的模块 docstring）——
把它塞进通用存储的测试里，这些差异会被通用断言淹没。

## 本文件重点覆盖的三处易漂移点

1. **`value` 的三列展开**
   `float | bool | str | None` 在 MySQL 里拆成 `value_kind` + 三列。
   最容易错的是 `bool`：Python 里 `True` 是 `int` 的实例，
   判断顺序反了会把布尔存进 numeric 列，读回来变成 `1.0` ——
   而 `1.0` 在红旗求值里与「用户点了是」是完全不同的语义。

2. **`limit` 的方向**
   `list_records(limit)` 取**最早** N 条，`list_assessments(limit)` 取**最近** N 条。
   方向相反是刻意的（与 `list_messages` / `list_recent_messages` 同一设计），
   而写反了条数还对得上。

3. **级联硬删**
   软标记不满足要求 —— 一条标了 `deleted=True` 的健康记录仍是泄露风险。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pytest

from app.health.store import InMemoryHealthStore
from app.schemas.health import (
    HealthAssessment,
    HealthRecord,
    HealthRecordSource,
    UrgencyLevel,
)

NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _mysql_ready() -> bool:
    return all(
        (os.environ.get(k) or "").strip()
        for k in ("MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")
    )


@pytest.fixture(params=["memory", "mysql"] if _mysql_ready() else ["memory"])
def health_store(request: pytest.FixtureRequest) -> Iterator[Any]:
    if request.param == "memory":
        yield InMemoryHealthStore()
        return

    from app.health.mysql import MySQLHealthRecordStore
    from app.store.factory import build_store_from_env

    bundle = build_store_from_env(dim=1024)
    pool = bundle.store._pool  # noqa: SLF001 - 测试需要清理
    with pool.acquire() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE health_records")
        cur.execute("TRUNCATE TABLE health_assessments")

    try:
        yield MySQLHealthRecordStore(pool=pool)
    finally:
        bundle.close()


def make_record(**kw: Any) -> HealthRecord:
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "session_id": "sess-1",
        "signal": "appetite",
        "value": 2.0,
        "unit": "score",
        "recorded_at": NOW,
        "source": HealthRecordSource.USER_INPUT,
        "sensitive": True,
        "consent_version": "v1",
        "created_at": NOW,
    }
    defaults.update(kw)
    return HealthRecord(**defaults)


def make_assessment(**kw: Any) -> HealthAssessment:
    """构造一个**契约合法**的评估。

    ⚠️ 契约里有一条硬约束（决策 D12）：
    **覆盖率 < 0.3 且无红旗命中时，级别必须是 `INSUFFICIENT_DATA`**。
    初版夹具固定 coverage=0.25 却按需改级别，于是构造直接被合同拒绝 ——
    那是契约在正常工作（禁止把「没数据」伪装成「没异常」），
    写错的是夹具。所以这里的默认 coverage 取 0.8，
    需要 `INSUFFICIENT_DATA` 的用例自己把 coverage 降下去。
    """
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "as_of": NOW,
        "level": UrgencyLevel.NO_DEVIATION_DETECTED,
        "coverage": 0.8,
        "coverage_note": "已采集 12/17 项信号",
        "signals_assessed": ["appetite", "water_intake"],
        "signals_missing": ["weight_kg"],
        "findings": [],
        "red_flags_triggered": [],
        "recommendation": "继续记录，保持观察。",
        "disclaimer": "本结果为基于行为记录的偏离提示，不是诊断。",
        "must_not_be_read_as": "未发现异常不代表健康",
        "rule_version": "2026.09.1",
    }
    defaults.update(kw)
    return HealthAssessment(**defaults)


def _insufficient(**kw: Any) -> HealthAssessment:
    """低覆盖度 → 只能是 INSUFFICIENT_DATA（契约强制）。"""
    kw.setdefault("coverage", 0.25)
    kw.setdefault("level", UrgencyLevel.INSUFFICIENT_DATA)
    kw.setdefault("coverage_note", "未采集的信号 17 项")
    return make_assessment(**kw)


# =============================================================================
# 记录
# =============================================================================


class TestRecords:
    def test_numeric_value_roundtrip(self, health_store: Any):
        stored = health_store.insert_record(make_record(value=38.7, unit="celsius"))
        assert stored.record_id

        got = health_store.list_records(user_id="user-1", pet_id="pet-1")
        assert len(got) == 1
        assert got[0].value == pytest.approx(38.7)
        assert isinstance(got[0].value, float)
        assert got[0].unit == "celsius"

    def test_bool_value_roundtrip_stays_bool(self, health_store: Any):
        """**最容易错的一处。** `True` 是 `int` 的实例 —— 判断顺序反了会变成 1.0。

        布尔与数字在红旗求值里语义完全不同（「有没有这个症状」vs「症状数值多少」），
        而 `1.0` 恰好也是合法数值，所以这种错**不会报错**。
        """
        health_store.insert_record(make_record(signal="vomiting_present", value=True))
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]

        assert got.value is True, f"布尔值被读成了 {got.value!r}（类型 {type(got.value)}）"

    def test_false_bool_is_not_none(self, health_store: Any):
        """`False` 不能变成 `None`。

        `None` 在红旗求值里是「不知道」，而 `False` 是「没有」——
        把「明确没有」读成「不知道」会让覆盖率虚低、结论停在无法评估。
        """
        health_store.insert_record(make_record(signal="seizure_present", value=False))
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.value is False, f"False 被读成了 {got.value!r}"

    def test_text_value_roundtrip(self, health_store: Any):
        health_store.insert_record(make_record(signal="stool_note", value="偏软"))
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.value == "偏软"

    def test_none_value_roundtrip(self, health_store: Any):
        """`None` 必须原样往返 —— 它代表「不知道」，不是「0」。

        对话里提取的健康记录一律 `value=None`（决策 D39），
        因为解析中文数词不可靠、而错了会误触红旗。
        丢了这条语义，系统会把「主人提过但没给数字」当成「数值是 0」。
        """
        health_store.insert_record(make_record(signal="water_intake", value=None))
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.value is None

    def test_zero_is_not_none(self, health_store: Any):
        """`0` 与 `None` 必须可区分 —— 一个不存在的数被读成 0 会误触红旗。"""
        health_store.insert_record(make_record(signal="vomiting_count", value=0.0))
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.value == 0.0
        assert got.value is not None

    def test_tenant_isolation(self, health_store: Any):
        health_store.insert_record(make_record(signal="appetite"))
        health_store.insert_record(make_record(signal="appetite", user_id="user-2"))
        health_store.insert_record(make_record(signal="appetite", pet_id="pet-2"))

        got = health_store.list_records(user_id="user-1", pet_id="pet-1")
        assert len(got) == 1

    def test_since_filter(self, health_store: Any):
        health_store.insert_record(
            make_record(signal="old", recorded_at=NOW - timedelta(days=10))
        )
        health_store.insert_record(make_record(signal="new", recorded_at=NOW))

        got = health_store.list_records(
            user_id="user-1", pet_id="pet-1", since=NOW - timedelta(days=1)
        )
        assert [r.signal for r in got] == ["new"]

    def test_limit_takes_earliest(self, health_store: Any):
        """`list_records(limit)` 取**最早** N 条（与评估相反，是刻意的）。"""
        for i in range(5):
            health_store.insert_record(
                make_record(signal=f"s{i}", recorded_at=NOW + timedelta(hours=i))
            )

        got = health_store.list_records(user_id="user-1", pet_id="pet-1", limit=2)
        assert [r.signal for r in got] == ["s0", "s1"], (
            "记录按时间正序取前 N 条（最早），不是最近的 N 条"
        )

    def test_metadata_fields_roundtrip(self, health_store: Any):
        health_store.insert_record(
            make_record(
                signal="weight_kg",
                value=4.2,
                sensitive=True,
                consent_version="v2",
                retention_days=730,
            )
        )
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.sensitive is True
        assert got.consent_version == "v2"
        assert got.retention_days == 730
        assert got.source is HealthRecordSource.USER_INPUT

    def test_datetime_roundtrip_keeps_utc(self, health_store: Any):
        saved = health_store.insert_record(make_record())
        got = health_store.list_records(user_id="user-1", pet_id="pet-1")[0]
        assert got.recorded_at.tzinfo is not None
        delta = abs((got.recorded_at - saved.recorded_at).total_seconds())
        assert delta < 1, f"时间偏移 {delta} 秒"


# =============================================================================
# 评估
# =============================================================================


class TestAssessments:
    def test_roundtrip_with_findings(self, health_store: Any):
        from app.schemas.health import RedFlagHit, SignalDeviation

        assessment = make_assessment(
            level=UrgencyLevel.EMERGENCY,
            coverage=0.6,
            findings=[
                SignalDeviation(
                    signal="appetite",
                    direction="down",
                    magnitude=2.5,
                    duration_days=3,
                    baseline_window="21d",
                    baseline_median=2.8,
                    observed_value=0.5,
                    note="连续三天食欲下降",
                )
            ],
            red_flags_triggered=[
                RedFlagHit(
                    rule_id="urinary_obstruction",
                    urgency=UrgencyLevel.EMERGENCY,
                    title="排尿困难",
                    message="公猫排尿困难可能是急症",
                    action="立即就医",
                    matched_conditions=["urination_count < 1"],
                    aggravating_conditions=["male"],
                    rule_version="2026.09.1",
                    sources=["AAHA 2024"],
                )
            ],
        )
        health_store.save_assessment(assessment)

        got = health_store.list_assessments(user_id="user-1", pet_id="pet-1")
        assert len(got) == 1
        assert got[0].level is UrgencyLevel.EMERGENCY
        assert len(got[0].findings) == 1
        assert got[0].findings[0].signal == "appetite"
        assert got[0].findings[0].magnitude == pytest.approx(2.5)
        assert len(got[0].red_flags_triggered) == 1
        assert got[0].red_flags_triggered[0].rule_id == "urinary_obstruction"
        assert got[0].red_flags_triggered[0].sources == ["AAHA 2024"]
        assert got[0].must_not_be_read_as

    def test_all_levels_roundtrip(self, health_store: Any):
        """五个等级都必须能存能读。

        曾经前端把等级名写成自造的 `monitor`/`urgent`，
        而真实值是 `L1`/`L2`/`L3` —— 五个分支全部落空，
        页面永远显示「没有拿到评估结果」。等级名是**唯一真相，不可意译**。

        这里撞上了三条契约硬约束，**它们的错误方向都是对的**：
        - 覆盖率 < 0.3 且无红旗 → 必须是 `INSUFFICIENT_DATA`（D12）
        - `EMERGENCY` 必须由红旗规则触发，不可由模型判定
        - 覆盖率不足时不得输出任何 findings

        所以夹具必须按等级给出**合法组合**，而不是固定一套字段改等级 ——
        后者会让测试在「契约拒绝非法输入」时报错，而那是契约在正常工作。
        """
        for i, level in enumerate(UrgencyLevel):
            health_store.save_assessment(self._valid_for(level, i))

        got = health_store.list_assessments(user_id="user-1", pet_id="pet-1")
        assert {a.level for a in got} == set(UrgencyLevel)

    @staticmethod
    def _valid_for(level: UrgencyLevel, i: int) -> HealthAssessment:
        """按等级给出契约合法的评估。"""
        base = {"assessment_id": f"a{i}", "as_of": NOW + timedelta(minutes=i), "level": level}
        if level is UrgencyLevel.INSUFFICIENT_DATA:
            return _insufficient(**base)
        if level is UrgencyLevel.EMERGENCY:
            # EMERGENCY 必须由红旗触发
            from app.schemas.health import RedFlagHit

            return make_assessment(
                coverage=0.5,
                red_flags_triggered=[
                    RedFlagHit(
                        rule_id="urinary_obstruction",
                        urgency=UrgencyLevel.EMERGENCY,
                        title="排尿困难",
                        message="公猫排尿困难可能是急症",
                        action="立即就医",
                        matched_conditions=["urination_count < 1"],
                        aggravating_conditions=["male"],
                        rule_version="2026.09.1",
                        sources=["AAHA 2024"],
                    )
                ],
                **base,
            )
        return make_assessment(**base)

    def test_limit_takes_latest(self, health_store: Any):
        """`list_assessments(limit)` 取**最近** N 条 —— 与记录相反。

        方向写反了条数还对得上，而拿到的是最旧的评估：
        用户看到的是几周前「无法评估」，而最新那条其实是 L3。
        """
        for i in range(5):
            health_store.save_assessment(
                make_assessment(
                    assessment_id=f"a{i}",
                    as_of=NOW + timedelta(hours=i),
                    coverage=0.3 + float(i) / 10,
                )
            )

        got = health_store.list_assessments(user_id="user-1", pet_id="pet-1", limit=2)
        assert len(got) == 2
        assert got[-1].coverage == pytest.approx(0.7), (
            "最后一条应是最新的评估（coverage 0.7 = i=4）"
        )
        assert got[0].coverage == pytest.approx(0.6)

    def test_tenant_isolation(self, health_store: Any):
        health_store.save_assessment(make_assessment(assessment_id="mine"))
        health_store.save_assessment(make_assessment(assessment_id="theirs", user_id="user-2"))

        got = health_store.list_assessments(user_id="user-1", pet_id="pet-1")
        assert {a.assessment_id for a in got} == {"mine"}


# =============================================================================
# 级联硬删
# =============================================================================


class TestCascadeDelete:
    def test_delete_removes_records_and_assessments(self, health_store: Any):
        """**硬删**，不是软标记。

        `HealthDataPolicy.on_account_deletion == "cascade_hard_delete"`：
        连派生的评估一起删。一条标了 `deleted=True` 的健康记录
        仍然是泄露风险。
        """
        health_store.insert_record(make_record())
        health_store.save_assessment(make_assessment())

        removed = health_store.delete_pet_data(user_id="user-1", pet_id="pet-1")
        assert removed >= 2, f"应删掉记录 + 评估，实际 {removed}"

        assert health_store.list_records(user_id="user-1", pet_id="pet-1") == []
        assert health_store.list_assessments(user_id="user-1", pet_id="pet-1") == []

    def test_delete_does_not_touch_other_tenants(self, health_store: Any):
        health_store.insert_record(make_record())
        health_store.insert_record(make_record(user_id="user-2"))

        health_store.delete_pet_data(user_id="user-1", pet_id="pet-1")

        assert len(health_store.list_records(user_id="user-2", pet_id="pet-1")) == 1, (
            "删除自己的健康数据不得影响别的租户"
        )
