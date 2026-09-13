"""记忆飞轮、检索与存储的测试。

覆盖 docs/DESIGN.md §3.5 的核心主张，以及 docs/ARCHITECTURE.md §2.6 的三个隔离强制点。

这些测试**不需要大模型、不需要网络**（``HashEmbedder`` + ``MockLLM`` 可离线跑）——
这是 ``DESIGN.md`` §6.5「可复现」要求的实现。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from app.llm import HashEmbedder, cosine
from app.memory import (
    MemoryWriter,
    build_context_block,
    expire_pending,
    is_eligible_for_promotion,
    judge_value,
    make_dedup_key,
    pending_memories,
    recency_score,
    retrieve,
    route_by_confidence,
)
from app.schemas import (
    EventType,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    PetProfile,
    Polarity,
    Species,
    VisualProfile,
)
from app.store import (
    DuplicateMemory,
    InMemoryStore,
    InvariantViolation,
    NotFound,
)

NOW = datetime.now(timezone.utc)
EMB = HashEmbedder()


# ─────────────────────────────────────────────────────────────
# 夹具
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def pet(store: InMemoryStore) -> PetProfile:
    p = PetProfile(
        pet_id="pet-1",
        user_id="user-1",
        name="团团",
        species=Species.CAT,
        visual=VisualProfile(fur_color="橘白", face_shape="圆脸"),
        must_keep_features=["橘白短毛", "圆脸", "胸口白毛"],
    )
    store.save_pet(p)
    return p


@pytest.fixture
def other_pet(store: InMemoryStore) -> PetProfile:
    p = PetProfile(
        pet_id="pet-2",
        user_id="user-1",
        name="阿橘",
        species=Species.CAT,
        visual=VisualProfile(fur_color="橘白"),
    )
    store.save_pet(p)
    return p


def make_event(**kw: Any) -> MemoryEvent:
    """构造测试用记忆。

    自动处理不变量：``SYSTEM_INFERENCE`` 不得为 ``ACTIVE``，
    否则契约层会在构造时就报错（那正是它该做的）。

    用 ``model_validate`` 而非 ``model_copy(update=...)`` —— 后者
    **不会重跑校验器**，会让非法组合（如 SYSTEM_INFERENCE + ACTIVE）
    静默通过，从而让「不变量必须被拒绝」的测试失去意义。
    """
    source = kw.get("source", MemorySource.USER_OBSERVATION)
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "event_type": EventType.PREFERENCE,
        "subject": "vacuum",
        "content": "它很怕吸尘器，开起来就躲",
        "polarity": Polarity.NEGATIVE,
        "source": MemorySource.USER_OBSERVATION,
        "confidence": 0.9,
        "status": (
            MemoryStatus.PENDING_CONFIRMATION
            if source is MemorySource.SYSTEM_INFERENCE
            else MemoryStatus.ACTIVE
        ),
    }
    return MemoryEvent.model_validate({**defaults, **kw})


# ═══════════════════════════════════════════════════════════════
# 存储：租户隔离的三个强制点
# ═══════════════════════════════════════════════════════════════


class TestTenantIsolation:
    """ARCHITECTURE.md §2.6 —— 隔离必须强制，不靠调用方自觉。"""

    def test_wrong_user_gets_not_found(self, store, pet):
        """T1/T2：归属不符与不存在返回同一个错误，不泄露资源存在性。"""
        with pytest.raises(NotFound):
            store.get_pet(user_id="someone-else", pet_id="pet-1")

    def test_missing_pet_is_same_error(self, store):
        with pytest.raises(NotFound):
            store.get_pet(user_id="user-1", pet_id="does-not-exist")

    def test_search_never_crosses_pets(self, store, pet, other_pet):
        """T3：**隔离是硬过滤**，别的宠物的记忆根本不进入候选。

        若把它做成「打个低分」，它会泄漏到 top-K —— 用户会看到
        「团团最近比较警惕」（实际是阿橘的记忆）。
        """
        store.insert_memory(
            make_event(pet_id="pet-1", content="团团喜欢趴窗台晒太阳"),
            vector=EMB.embed("团团喜欢趴窗台晒太阳"),
        )
        store.insert_memory(
            make_event(pet_id="pet-2", content="阿橘非常警惕陌生人靠近"),
            vector=EMB.embed("阿橘非常警惕陌生人靠近"),
        )

        results = retrieve(
            store=store,
            embedder=EMB,
            user_id="user-1",
            pet_id="pet-1",
            query="它最近怎么样",
            k=10,
        )
        assert results
        assert {r.event.pet_id for r in results} == {"pet-1"}

    def test_get_memory_scoped_by_pet(self, store, pet, other_pet):
        ev = store.insert_memory(make_event(pet_id="pet-2"))
        with pytest.raises(NotFound):
            store.get_memory(user_id="user-1", pet_id="pet-1", memory_id=ev.memory_id)


class TestStoreInvariants:
    """ARCHITECTURE.md §2.2 —— 存储层挡的是「绕过契约的写入路径」。"""

    def test_dedup_key_is_unique(self, store, pet):
        ev = make_event(dedup_key="k1")
        store.insert_memory(ev)
        with pytest.raises(DuplicateMemory):
            store.insert_memory(make_event(dedup_key="k1"))

    def test_system_inference_cannot_be_stored_active(self, store, pet):
        """**深度防御**：契约层已经拦过，存储层再拦一次。

        用 ``model_construct`` 绕过 Pydantic 校验来模拟「绕过契约的写入路径」。
        """
        rogue = MemoryEvent.model_construct(
            user_id="user-1",
            pet_id="pet-1",
            event_type=EventType.BEHAVIOR,
            subject="door",
            content="它想让你开门",
            polarity=Polarity.NEUTRAL,
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.ACTIVE,
            confidence=0.6,
            support_count=1,
        )
        with pytest.raises(InvariantViolation, match="I1"):
            store.insert_memory(rogue)

    def test_search_excludes_non_active(self, store, pet):
        """模拟 partial index ``WHERE status='active'``。"""
        active = store.insert_memory(
            make_event(content="喜欢逗猫棒", dedup_key="a"),
            vector=EMB.embed("喜欢逗猫棒"),
        )
        pending = make_event(
            content="可能想出去玩",
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.PENDING_CONFIRMATION,
            dedup_key="b",
        )
        store.insert_memory(pending, vector=EMB.embed("可能想出去玩"))

        hits = store.search_memories(
            user_id="user-1",
            pet_id="pet-1",
            query_vector=EMB.embed("喜欢逗猫棒"),
            limit=10,
        )
        ids = {h.event.memory_id for h in hits}
        assert active.memory_id in ids
        assert all(h.event.status is MemoryStatus.ACTIVE for h in hits), (
            "非 active 记忆不得进入召回"
        )


# ═══════════════════════════════════════════════════════════════
# 飞轮：准入
# ═══════════════════════════════════════════════════════════════


class TestValueJudgment:
    """DESIGN.md §3.5 —— 判据是「未来会不会被再次检索」。"""

    @pytest.mark.parametrize(
        "content,reason_keyword",
        [
            ("啊呀", "过短"),
            ("好的", "无信息量"),
            ("嗯", "无信息量"),
            ("谢谢", "无信息量"),
            ("它为什么一直叫？", "疑问句"),
        ],
    )
    def test_rejects_low_value(self, content, reason_keyword):
        ok, reason = judge_value(make_event(content=content))
        assert ok is False
        assert reason_keyword in reason

    def test_accepts_durable_attribute(self):
        ok, _ = judge_value(make_event(content="它很怕吸尘器，开起来就躲"))
        assert ok is True

    def test_chinese_numeral_in_ordinary_word_is_not_a_quantity(self):
        """回归：「一开就跑」里的「一」不是数量，不得触发「需确认」。

        早期实现逐个字符检查汉字数字，导致**大量正常句子被误判为含数字**，
        全部落入 PENDING —— 记忆永远不会生效。
        """
        from app.memory.flywheel import _looks_like_quantity

        assert not _looks_like_quantity("它一开始很怕吸尘器")
        assert not _looks_like_quantity("它一直这样，十分黏人")
        assert _looks_like_quantity("它每天早上 7 点叫我起床")
        assert _looks_like_quantity("它一天要吃三次")
        assert _looks_like_quantity("它大概有 4.5 公斤")

    def test_health_goes_elsewhere(self):
        """健康信号走 health_records 表，不进通用记忆，避免检索时混入。"""
        ok, reason = judge_value(
            make_event(event_type=EventType.HEALTH, content="今天没吃东西")
        )
        assert ok is False
        assert "health_records" in reason


class TestConfidenceRouting:
    """DESIGN.md §3.5 R1 —— 防自我强化。"""

    def test_system_inference_always_pending(self):
        """**不变量 I1 的源头**：系统推断永远到不了 ACTIVE。"""
        status, reason = route_by_confidence(
            make_event(source=MemorySource.SYSTEM_INFERENCE, confidence=0.6)
        )
        assert status is MemoryStatus.PENDING_CONFIRMATION
        assert "不得作为事实" in reason

    def test_user_observation_is_active(self):
        status, _ = route_by_confidence(
            make_event(source=MemorySource.USER_OBSERVATION)
        )
        assert status is MemoryStatus.ACTIVE

    def test_numeric_content_needs_confirmation(self):
        """ASR 转写错误会被当作事实存下且事后无法察觉，故含数量/时间需确认。"""
        status, reason = route_by_confidence(
            make_event(content="它每天早上 7 点叫我起床")
        )
        assert status is MemoryStatus.PENDING_CONFIRMATION
        assert "数量" in reason


class TestDedupKey:
    def test_normalizes_case_and_whitespace(self):
        a = make_dedup_key(make_event(content="  它  怕  吸尘器 "))
        b = make_dedup_key(make_event(content="它 怕 吸尘器"))
        assert a == b

    def test_includes_type_and_subject(self):
        a = make_dedup_key(make_event(content="x", subject="vacuum"))
        b = make_dedup_key(make_event(content="x", subject="brush"))
        assert a != b


# ═══════════════════════════════════════════════════════════════
# 飞轮：决策与落地
# ═══════════════════════════════════════════════════════════════


class TestFlywheelWrite:
    def test_plain_write(self, store, pet):
        w = MemoryWriter(store=store, embedder=EMB)
        result = w.apply([w.decide(make_event(content="它很怕吸尘器，开起来就躲"))])
        assert len(result.written_memory_ids) == 1
        assert not result.skipped

    def test_reject_is_not_silent(self, store, pet):
        """拒绝也要有明确去向与原因，不能静默丢弃。"""
        w = MemoryWriter(store=store, embedder=EMB)
        d = w.decide(make_event(content="好的"))
        assert d.action.value == "reject"
        assert d.reason
        result = w.apply([d])
        assert result.skipped and result.skipped[0][0] == "reject"

    def test_reinforce_not_duplicate(self, store, pet):
        """**去重是强化，不是重复插入**（DESIGN.md §3.5）。"""
        w = MemoryWriter(store=store, embedder=EMB)
        content = "它很怕吸尘器，开起来就躲"

        w.apply([w.decide(make_event(content=content))])
        assert len(store.list_memories(user_id="user-1", pet_id="pet-1")) == 1

        for _ in range(2):
            d = w.decide(make_event(content=content))
            assert d.action.value == "reinforce"
            w.apply([d])

        memories = store.list_memories(user_id="user-1", pet_id="pet-1")
        assert len(memories) == 1, "说三次应产生 1 条记忆，而不是 3 条"
        assert memories[0].support_count == 3

    def test_conflict_creates_supersede_chain(self, store, pet):
        """冲突消解用**取代链**而非覆盖 —— 历史必须保留。

        「上个月喜欢」与「这周不喜欢」是演变，不是矛盾；
        但这两条时间范围重叠，构成真实冲突。
        """
        w = MemoryWriter(store=store, embedder=EMB)
        old = make_event(
            subject="cat_wand",
            content="它喜欢逗猫棒",
            polarity=Polarity.POSITIVE,
        )
        w.apply([w.decide(old)])

        new = make_event(
            subject="cat_wand",
            content="最近它不喜欢逗猫棒了",
            polarity=Polarity.NEGATIVE,
            valid_from=NOW - timedelta(days=7),
        )
        d = w.decide(new)
        assert d.action.value == "supersede"
        w.apply([d])

        all_mem = store.list_memories(
            user_id="user-1", pet_id="pet-1", include_non_active=True
        )
        active = [m for m in all_mem if m.status is MemoryStatus.ACTIVE]
        superseded = [m for m in all_mem if m.status is MemoryStatus.SUPERSEDED]

        assert len(active) == 1 and active[0].polarity is Polarity.NEGATIVE
        assert len(superseded) == 1, "旧记忆必须保留（支持「以前喜欢什么」类问题）"
        assert superseded[0].polarity is Polarity.POSITIVE

    def test_disjoint_time_ranges_do_not_conflict(self, store, pet):
        """**时间范围重叠是冲突的必要条件**，缺了它会误判演变。"""
        w = MemoryWriter(store=store, embedder=EMB)
        w.apply(
            [
                w.decide(
                    make_event(
                        subject="cat_wand",
                        content="上个月它喜欢逗猫棒",
                        polarity=Polarity.POSITIVE,
                        valid_from=NOW - timedelta(days=40),
                        valid_to=NOW - timedelta(days=10),
                    )
                )
            ]
        )
        d = w.decide(
            make_event(
                subject="cat_wand",
                content="这周它不怎么玩逗猫棒",
                polarity=Polarity.NEGATIVE,
                valid_from=NOW - timedelta(days=7),
            )
        )
        assert d.action.value == "write", "时间不重叠的两条应并存，不是冲突"

    def test_lower_trust_does_not_overwrite(self, store, pet):
        """新信息来自更低可信度来源 → **不覆盖**，进 PENDING。"""
        w = MemoryWriter(store=store, embedder=EMB)
        w.apply(
            [
                w.decide(
                    make_event(
                        subject="cat_wand",
                        content="它喜欢逗猫棒",
                        polarity=Polarity.POSITIVE,
                        source=MemorySource.USER_CORRECTION,
                    )
                )
            ]
        )
        d = w.decide(
            make_event(
                subject="cat_wand",
                content="它不喜欢逗猫棒",
                polarity=Polarity.NEGATIVE,
                source=MemorySource.SYSTEM_INFERENCE,
                confidence=0.6,
            )
        )
        assert d.action.value == "pending", d.reason

    def test_no_active_system_inference_after_full_pipeline(self, store, pet):
        """**端到端断言**：跑完整管线后，ACTIVE 的 system_inference 恒为 0。

        这是 DESIGN.md 里那条「可测断言」的实现。
        """
        w = MemoryWriter(store=store, embedder=EMB)
        candidates = [
            make_event(
                content=f"它可能想出去玩（推断 {i}）",
                source=MemorySource.SYSTEM_INFERENCE,
                confidence=0.6,
                subject=f"door-{i}",
            )
            for i in range(5)
        ]
        w.apply([w.decide(c) for c in candidates])

        violators = [
            m
            for m in store.list_memories(
                user_id="user-1", pet_id="pet-1", include_non_active=True
            )
            if m.source is MemorySource.SYSTEM_INFERENCE
            and m.status is MemoryStatus.ACTIVE
        ]
        assert violators == [], f"不变量 I1 被破坏：{len(violators)} 条系统推断成了事实"


class TestPromotionAndExpiry:
    def test_promotion_needs_both_count_and_span(self):
        """只满足次数不满足跨度是不够的 —— 一天内说三次不代表稳定属性。"""
        same_day = make_event(
            event_type=EventType.PREFERENCE,
            support_count=3,
            created_at=NOW - timedelta(days=2),
            last_seen_at=NOW - timedelta(days=1),
        )
        assert not is_eligible_for_promotion(same_day)

        spanning = make_event(
            event_type=EventType.PREFERENCE,
            support_count=3,
            created_at=NOW - timedelta(days=30),
            last_seen_at=NOW - timedelta(days=1),
        )
        assert is_eligible_for_promotion(spanning)

    def test_behavior_not_promoted(self):
        """事件不是属性 —— 「昨天害怕吸尘器」不该晋升为身份特征。"""
        ev = make_event(
            event_type=EventType.BEHAVIOR,
            support_count=10,
            created_at=NOW - timedelta(days=60),
            last_seen_at=NOW,
        )
        assert not is_eligible_for_promotion(ev)

    def test_expire_pending(self, store, pet):
        old = make_event(
            content="它可能想出门",
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.PENDING_CONFIRMATION,
            created_at=NOW - timedelta(days=30),
        )
        # 必须捕获返回值：memory_id 是 insert 时才分配的，原对象上仍为 None
        stored = store.insert_memory(old)
        expired = expire_pending(store, user_id="user-1", pet_id="pet-1", ttl_days=14)
        assert len(expired) == 1
        assert (
            store.get_memory(
                user_id="user-1", pet_id="pet-1", memory_id=stored.memory_id
            ).status
            is MemoryStatus.REJECTED
        )


# ═══════════════════════════════════════════════════════════════
# 检索
# ═══════════════════════════════════════════════════════════════


class TestRecency:
    """DESIGN.md §3.5 —— 用统一衰减率会让系统忘掉猫「一直喜欢」的东西。"""

    def test_preference_does_not_decay(self):
        old = make_event(
            event_type=EventType.PREFERENCE,
            occurred_at=NOW - timedelta(days=365),
        )
        assert recency_score(old, now=NOW) == 1.0

    def test_context_decays(self):
        recent = make_event(
            event_type=EventType.CONTEXT, occurred_at=NOW - timedelta(days=1)
        )
        old = make_event(
            event_type=EventType.CONTEXT, occurred_at=NOW - timedelta(days=21)
        )
        assert recency_score(recent, now=NOW) > recency_score(old, now=NOW)

    def test_behavior_decays_slower_than_context(self):
        t = NOW - timedelta(days=10)
        b = make_event(event_type=EventType.BEHAVIOR, occurred_at=t)
        c = make_event(event_type=EventType.CONTEXT, occurred_at=t)
        assert recency_score(b, now=NOW) > recency_score(c, now=NOW)


class TestRetrieval:
    def test_empty_when_no_memories(self, store, pet):
        assert (
            retrieve(
                store=store,
                embedder=EMB,
                user_id="user-1",
                pet_id="pet-1",
                query="它怕什么",
            )
            == []
        )

    def test_offline_embedder_cannot_do_semantic_ranking(self):
        """明确记录能力边界：弱向量器不承担语义排序，那需要真实 embedding。

        这条测试的存在意义是**防止有人误以为离线链路可以验证语义检索质量**。
        """
        q = EMB.embed("它怕什么东西")
        vac = EMB.embed("它很怕吸尘器的声音")
        assert cosine(q, vac) < 0.5, "哈希向量器不应在此表现出语义能力"

    def test_relevant_first(self, store, pet):
        """字面重叠的查询应把对应记忆排在前面。

        ⚠️ **这是 ``HashEmbedder`` 的能力边界**：它是字符 n-gram 哈希，
        **不是语义模型**。所以只能测「字面重叠 → 排序正确」，
        不能测「语义相关 → 排序正确」。后者需要真实 embedding（需 API，不进离线测试）。

        这个边界是刻意的：宁可明确它能测什么，也不要用一个弱向量器
        去过一个它不可能通过的语义断言。
        """
        w = MemoryWriter(store=store, embedder=EMB)
        for content in ["它很怕吸尘器的声音", "它喜欢逗猫棒", "它每天中午睡觉"]:
            w.apply([w.decide(make_event(content=content, subject=content))])

        hits = retrieve(
            store=store,
            embedder=EMB,
            user_id="user-1",
            pet_id="pet-1",
            query="吸尘器",
            k=3,
        )
        assert hits
        assert "吸尘器" in hits[0].event.content

    def test_mmr_increases_diversity(self, store, pet):
        """MMR 应避免 top-N 全是同一件事的变体。"""
        w = MemoryWriter(store=store, embedder=EMB)
        for content in [
            "它很怕吸尘器",
            "它很怕吹风机",
            "它很怕理发器",
            "它喜欢在窗台晒太阳",
        ]:
            w.apply([w.decide(make_event(content=content, subject=content))])

        hits = retrieve(
            store=store,
            embedder=EMB,
            user_id="user-1",
            pet_id="pet-1",
            query="它怕什么",
            k=3,
        )
        assert len(hits) == 3

    def test_score_breakdown_has_components(self, store, pet):
        w = MemoryWriter(store=store, embedder=EMB)
        w.apply([w.decide(make_event(content="它很怕吸尘器", subject="vacuum"))])
        hits = retrieve(
            store=store,
            embedder=EMB,
            user_id="user-1",
            pet_id="pet-1",
            query="吸尘器",
            k=1,
        )
        b = hits[0].breakdown
        assert b is not None
        assert b.total == pytest.approx(hits[0].score)


class TestContextInjection:
    def test_declares_absence_of_records(self, store, pet):
        """**幻觉的主要来源**是模型用常识补全，所以必须显式声明「检索不到就是没有」。"""
        block = build_context_block(pet=pet, items=[], pending=[])
        rendered = block.render()
        assert "没有记录" in rendered

    def test_separates_confirmed_and_unconfirmed(self, store, pet):
        w = MemoryWriter(store=store, embedder=EMB)
        w.apply([w.decide(make_event(content="它很怕吸尘器", subject="vacuum"))])
        items = retrieve(
            store=store, embedder=EMB, user_id="user-1", pet_id="pet-1", query="吸尘器"
        )

        guessed = store.insert_memory(
            make_event(
                content="它可能想出门",
                subject="door",
                source=MemorySource.SYSTEM_INFERENCE,
                status=MemoryStatus.PENDING_CONFIRMATION,
            )
        )
        block = build_context_block(
            pet=pet,
            items=items,
            pending=pending_memories(store, user_id="user-1", pet_id="pet-1"),
        )
        rendered = block.render()
        assert "unconfirmed" in rendered
        assert guessed.content in rendered
        assert "系统推断，未经用户确认" in rendered

    def test_identity_block_carries_hard_constraints(self, pet):
        block = build_context_block(pet=pet, items=[], pending=[])
        assert "橘白短毛" in block.pet_identity
        assert "圆脸" in block.pet_identity


# ═══════════════════════════════════════════════════════════════
# 基础设施
# ═══════════════════════════════════════════════════════════════


class TestOfflineCapability:
    """DESIGN.md §6.5 —— 无 API Key / 无网络时也必须能跑通。"""

    def test_hash_embedder_is_deterministic(self):
        a = EMB.embed("它很怕吸尘器")
        b = EMB.embed("它很怕吸尘器")
        assert a == b
        assert len(a) == EMB.dim

    def test_identical_text_similarity_is_one(self):
        v = EMB.embed("它很怕吸尘器")
        assert cosine(v, v) == pytest.approx(1.0)

    def test_different_text_has_lower_similarity(self):
        a = EMB.embed("它很怕吸尘器")
        b = EMB.embed("完全不同的另一句话内容")
        assert cosine(a, b) < 0.92

    def test_empty_text_does_not_crash(self):
        v = EMB.embed("")
        assert len(v) == EMB.dim
