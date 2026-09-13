"""健康数据的写入与评估。

## 本模块补的是一个**已经登记但没人接**的缺口

`app/memory/flywheel.py` 会拒绝 `EventType.HEALTH` 进通用记忆，理由是
「健康信号另走 health_records 表」—— 但在此之前**没有任何东西把它写进那张表**。

于是 `app/digest` 提取出的健康信号：记忆层正确拒绝、日报正确显示、
**却没有任何地方落地**。这个缺口在 `DailySummary.notes` 里可见，
现在有接收方了。

## 一条必须说清楚的边界

**对话里提到的健康信号，适合「记录与提示观察」，不适合「触发急诊红旗」。**

理由：红旗规则要的是**结构化信号**（`litter_box.visit_frequency` 是次数、
`respiratory.rate` 是次数/分），而对话文本给的是「它今天吐了两次」。

所以从对话来的记录**一律 `value=None`**：

| 做法 | 后果 |
| --- | --- |
| 从「吐了**两次**」提取 2 | **必须解析中文数词** —— B11/B12 已证明这条路不可靠，而且错了会误触红旗 |
| 一律 `value=None` | 信号「存在但未量化」→ 进 `signals_missing` → **评估给出 `INSUFFICIENT_DATA`** |

后者是诚实的：系统知道「主人提到了呕吐」，也知道「我不知道几次」，
于是它说「数据不足」，而不是猜一个数字或假装没事。

红旗由**结构化通道**（`record_signal`）触发 —— 猫砂盆计数、呼吸频率这类
本来就有数值的信号。两条通道分开，各司其职。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date as date_type
from datetime import datetime, timezone

from app.health.redflags import RedFlagTable, evaluate
from app.health.store import HealthRecordStore
from app.schemas.digest import DigestCandidate
from app.schemas.health import (
    MIN_COVERAGE_FOR_ASSESSMENT,
    HealthAssessment,
    HealthRecord,
    HealthRecordSource,
    UrgencyLevel,
)
from app.schemas.memory import EventType

#: 对话里提取出的 subject → 结构化信号名。
#:
#: **保守映射**：只在概念明确对应时才映射。
#: 映射错了会让一个信号看起来「已采集」（实际值为 None），
#: 从而污染 `signals_missing` 的判断 —— 宁可不映射。
SUBJECT_TO_SIGNAL: dict[str, str] = {
    "vomit": "gi.vomiting_frequency",
    "vomiting": "gi.vomiting_frequency",
    "diarrhea": "gi.diarrhea",
    "weight": "weight.value",
    "respiratory": "respiratory.rate",
    "breathing": "respiratory.pattern",
    "seizure": "neuro.seizure",
    "toxin": "toxin.suspected_ingestion",
    "bleeding": "bleeding.uncontrolled",
    "mucosa": "mucosa.color",
    "appetite": "meal.hours_since_last_intake",
    "demeanor": "general.demeanor",
    "mobility": "neuro.mobility",
}

#: 未映射 subject 的前缀。它**不在**规则表词汇表里，
#: 因此永远不会参与红旗求值 —— 这是刻意的。
UNSTRUCTURED_PREFIX = "unstructured."


@dataclass(frozen=True)
class HealthAdmission:
    """一次健康数据写入的结果。"""

    records: list[HealthRecord] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    """``(内容, 未写入原因)``。**不让任何东西静默消失。**"""

    @property
    def written_count(self) -> int:
        return len(self.records)


class HealthWriter:
    """健康记录写入器。

    与 `MemoryWriter` 并列，但**不共用**：健康数据的同意、加密、
    保留期与硬删除要求不同（见 `HealthDataPolicy`）。
    """

    def __init__(
        self,
        *,
        store: HealthRecordStore,
        redflags: RedFlagTable,
        consent_version: str,
    ) -> None:
        if not consent_version:
            # 健康数据需要显式同意（HealthDataPolicy.requires_explicit_consent）。
            # 没有同意版本号就不该写入 —— 这不是形式要求，是合规要求。
            raise ValueError(
                "写入健康数据必须提供 consent_version"
                "（HealthDataPolicy.requires_explicit_consent = True）"
            )
        self.store = store
        self.redflags = redflags
        self.consent_version = consent_version

    # ── 通道一：对话提取（记录，不触发红旗） ──

    def admit_digest_candidates(
        self,
        candidates: list[DigestCandidate],
        *,
        user_id: str,
        pet_id: str,
        day: date_type,
        at: datetime | None = None,
    ) -> HealthAdmission:
        """把每日总结里的健康候选写入 health_records。

        只有 ``event_type == HEALTH`` 的候选会被处理；其余返回为 rejected
        并说明原因（**调用方可以看到自己传错了什么**）。
        """
        recorded_at = at or datetime.combine(
            day, datetime.min.time(), tzinfo=timezone.utc
        )
        records: list[HealthRecord] = []
        rejected: list[tuple[str, str]] = []

        for cand in candidates:
            if cand.event_type is not EventType.HEALTH:
                rejected.append(
                    (
                        cand.content,
                        f"非健康事件（{cand.event_type.value}），不写健康记录",
                    )
                )
                continue

            signal = SUBJECT_TO_SIGNAL.get(cand.subject.lower())
            if signal is None:
                signal = f"{UNSTRUCTURED_PREFIX}{cand.subject.lower()[:40]}"

            records.append(
                HealthRecord(
                    user_id=user_id,
                    pet_id=pet_id,
                    signal=signal,
                    # **一律 None**：对话文本给不出可靠的量化值（见模块 docstring）。
                    # 解析中文数词已被 B11/B12 证明不可靠，且错了会误触红旗。
                    value=None,
                    unit=None,
                    recorded_at=recorded_at,
                    source=HealthRecordSource.USER_INPUT,
                    consent_version=self.consent_version,
                )
            )

        stored = [self.store.insert_record(r) for r in records]
        return HealthAdmission(records=stored, rejected=rejected)

    # ── 通道二：结构化信号（可触发红旗） ──

    def record_signal(
        self,
        *,
        user_id: str,
        pet_id: str,
        signal: str,
        value: float | bool | str | None,
        source: HealthRecordSource,
        at: datetime,
        unit: str | None = None,
        session_id: str | None = None,
    ) -> HealthRecord:
        """记录一个**结构化**信号值。红旗由这条通道触发。

        与 `admit_digest_candidates` 的区别只有一个字：**值**。
        这里有值，所以能比较、能触发；那里没有值，所以只能记录。

        ``session_id`` 仅用于追溯。日报聚合跨会话，所以那条通道传 ``None`` ——
        给它填一个具体会话会是错的归属。
        """
        return self.store.insert_record(
            HealthRecord(
                user_id=user_id,
                pet_id=pet_id,
                session_id=session_id,
                signal=signal,
                value=value,
                unit=unit,
                recorded_at=at,
                source=source,
                consent_version=self.consent_version,
            )
        )

    # ── 评估 ──

    def assess(
        self,
        *,
        user_id: str,
        pet_id: str,
        signals: dict[str, float | bool | str | None],
    ) -> HealthAssessment:
        """对一组信号做红旗求值与分诊。

        **覆盖率低于门限时给出 `INSUFFICIENT_DATA`，绝不可降级表述为「未发现异常」。**
        """
        hits, undecidable = evaluate(self.redflags, signals)

        vocabulary = self.redflags.signal_names()
        # 只把「词汇表内的信号」计入覆盖率 —— 未结构化的记录参与了记录，
        # 但不参与评估，把它们算进覆盖率会虚报「我检查过了」
        structured = {
            k: v for k, v in signals.items() if k in vocabulary and v is not None
        }
        missing = sorted(vocabulary - set(structured))

        coverage = len(structured) / len(vocabulary) if vocabulary else 0.0

        if hits:
            level = max((h.urgency for h in hits), key=lambda u: u.value)
        elif coverage < MIN_COVERAGE_FOR_ASSESSMENT or undecidable:
            level = UrgencyLevel.INSUFFICIENT_DATA
        else:
            level = UrgencyLevel.NO_DEVIATION_DETECTED

        note_parts: list[str] = []
        if undecidable:
            note_parts.append(
                f"{len(undecidable)} 条规则因缺少信号而无法评估：{'、'.join(undecidable)}"
            )
        if missing:
            note_parts.append(f"未采集的信号 {len(missing)} 项")
        if not note_parts:
            note_parts.append("全部已启用规则均已评估")

        assessment = HealthAssessment(
            user_id=user_id,
            pet_id=pet_id,
            level=level,
            coverage=round(coverage, 4),
            coverage_note="；".join(note_parts),
            signals_assessed=sorted(structured),
            signals_missing=missing,
            red_flags_triggered=hits,
            recommendation=_recommendation(level, hits),
            rule_version=self.redflags.version,
        )
        return self.store.save_assessment(assessment)


def _recommendation(level: UrgencyLevel, hits: list) -> str:
    """建议文案。

    **不出现「健康」「正常」这类排除性表述** —— `FORBIDDEN_PHRASES` 会拦，
    但这里就不该生成它们（第一道防线在前端，不是靠事后过滤）。
    """
    if level is UrgencyLevel.EMERGENCY and hits:
        return hits[0].action
    if level is UrgencyLevel.VET_VISIT_RECOMMENDED:
        return "建议尽快联系兽医。"
    if level is UrgencyLevel.OBSERVE:
        return "继续记录并观察变化。"
    if level is UrgencyLevel.INSUFFICIENT_DATA:
        return "目前采集到的数据不足以评估，请继续记录。"
    return "本次未发现偏离。若你观察到异常，请以你的判断为准并联系兽医。"
