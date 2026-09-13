"""契约一致性测试：**同一组断言跑在两套存储实现上**。

## 为什么这组测试决定了 MySQL 实现是否可信

`MySQLStore` 是一整条新的代码路径。只给它写「专属测试」是不够的 ——
那只能证明「它做了我想到要测的事」，无法证明**它和内存实现行为一致**。
而系统的其余部分（编排、飞轮、日报、健康）全都是对着内存实现写出来的，
它们默认了内存实现的那些细节：排序方向、截断方向、报错类型、默认过滤。

所以这里把「行为」抽出来，让两套实现各自通过：

    @pytest.fixture(params=["memory", "mysql"])

任何一个断言只在一边通过，就说明两者有语义漂移 —— 而漂移的那一边
在生产里是 MySQL，在测试里是内存，**永远不会有人发现**。

## 最容易漂移的四处（本文件重点覆盖）

| 位置 | 漂移后果 |
|---|---|
| `list_recent_messages` 的排序与截断方向 | 注入的是最旧的 N 条而不是最近的，条数还对得上 |
| `list_memories` 的默认状态过滤 | 把 superseded 的记忆当现行事实用 |
| `list_assessments(limit)` 的方向 | 拿到最旧的评估而不是最新的 |
| `DuplicateMemory` 的抛出 | 飞轮的去重强化整条路径失效 |

## 运行方式

内存后端永远跑。MySQL 后端需要配置：

    MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE

未配置时**跳过而不是失败** —— 否则本地开发会无法运行整个测试套件。
但 CI 里配了就必须跑（见 .github/workflows/ci.yml）。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import pytest

from app.schemas import (
    AcousticFeatures,
    BehaviorAction,
    ContextLabel,
    EventType,
    EvidenceMode,
    IntentCandidate,
    MeowRecord,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    PendingInterpretation,
    PetProfile,
    Polarity,
    SessionMessage,
    Species,
    VisualProfile,
)
from app.store import DuplicateMemory, InMemoryStore, NotFound
from app.store.memory import InMemoryStore as _MemStore

NOW = datetime.now(timezone.utc).replace(microsecond=0)


# =============================================================================
# 夹具
# =============================================================================


def _mysql_env() -> dict[str, str] | None:
    host = (os.environ.get("MYSQL_HOST") or "").strip()
    if not host:
        return None
    required = {
        "MYSQL_USER": os.environ.get("MYSQL_USER", ""),
        "MYSQL_PASSWORD": os.environ.get("MYSQL_PASSWORD", ""),
        "MYSQL_DATABASE": os.environ.get("MYSQL_DATABASE", ""),
    }
    if not all(required.values()):
        return None
    return {"MYSQL_HOST": host, **required}


_MYSQL_READY = _mysql_env() is not None


@pytest.fixture(params=["memory", "mysql"] if _MYSQL_READY else ["memory"])
def backend(request: pytest.FixtureRequest) -> Iterator[tuple[Any, str]]:
    """产出 `(store, backend_name)`。

    每个用例前清库 —— 测试之间的隔离必须是真的，
    残留数据会让「查不到」与「查到了别人的」变得无法区分。
    """
    name = request.param

    if name == "memory":
        yield InMemoryStore(), "memory"
        return

    # ── MySQL ──
    from app.store.factory import build_store_from_env
    from app.store.mysql import MySQLPool

    bundle = build_store_from_env(dim=1024)
    store = bundle.store
    assert store is not None

    pool: MySQLPool = store._pool  # noqa: SLF001 - 测试需要直接清理
    tables = (
        "memories",
        "meow_records",
        "pending_interpretations",
        "session_messages",
        "health_records",
        "health_assessments",
        "pets",
    )
    with pool.acquire() as conn, conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        for t in tables:
            cur.execute(f"TRUNCATE TABLE {t}")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1")

    try:
        yield store, "mysql"
    finally:
        bundle.close()


@pytest.fixture
def store(backend: tuple[Any, str]) -> Any:
    return backend[0]


@pytest.fixture
def backend_name(backend: tuple[Any, str]) -> str:
    return backend[1]


# =============================================================================
# 构造器
# =============================================================================


def make_pet(**kw: Any) -> PetProfile:
    defaults: dict[str, Any] = {
        "pet_id": "pet-1",
        "user_id": "user-1",
        "name": "团团",
        "species": Species.CAT,
        "breed": "英短",
        "visual": VisualProfile(fur_color="橘白", fur_length="短毛", eye_color="黄"),
        "must_keep_features": ["橘白短毛", "圆脸"],
        "observed_but_unstable": ["胸口白毛"],
        "identity_prompt": "This is the same real cat.",
        # 显式带上时间：默认值由模型自己生成，会让「时间往返」这类断言
        # 变成在测「构造到断言之间的耗时」
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(kw)
    return PetProfile(**defaults)


def make_event(**kw: Any) -> MemoryEvent:
    source = kw.get("source", MemorySource.USER_OBSERVATION)
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "layer": "episode",
        "event_type": EventType.PREFERENCE,
        "subject": "vacuum",
        "content": "它很怕吸尘器",
        "polarity": Polarity.NEGATIVE,
        "source": source,
        "confidence": 0.9,
        "status": (
            MemoryStatus.PENDING_CONFIRMATION
            if source is MemorySource.SYSTEM_INFERENCE
            else MemoryStatus.ACTIVE
        ),
        "valid_from": NOW,
        "created_at": NOW,
        "updated_at": NOW,
    }
    defaults.update(kw)
    # 用 model_validate 而非 model_copy：后者不重跑校验器，
    # 会让非法组合静默通过
    return MemoryEvent.model_validate(defaults)


def make_features() -> AcousticFeatures:
    return AcousticFeatures(
        duration=0.8,
        f0_mean=520.0,
        f0_range=180.0,
        f0_slope=12.0,
        call_rate=3.0,
        ici_mean=1.2,
        rms_mean=0.11,
        roughness=0.3,
        unavailable=[],
    )


def make_meow(**kw: Any) -> MeowRecord:
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "session_id": "sess-1",
        "context": ContextLabel.FOOD_WAITING,
        "features": make_features(),
        "actions": [BehaviorAction.NEAR_FOOD_BOWL],
        "resolution": "给了罐头就停了",
        "recorded_at": NOW,
        "source": MemorySource.MEOW_LABEL,
        "status": MemoryStatus.ACTIVE,
    }
    defaults.update(kw)
    return MeowRecord(**defaults)


def make_pending(**kw: Any) -> PendingInterpretation:
    defaults: dict[str, Any] = {
        "interpretation_id": "interp-1",
        "user_id": "user-1",
        "pet_id": "pet-1",
        "session_id": "sess-1",
        "features": make_features(),
        "evidence_mode": EvidenceMode.MEASURED_ONLY,
        "candidates": [
            IntentCandidate(context=ContextLabel.FOOD_WAITING, display="等吃的", matched_count=0)
        ],
        "created_at": NOW,
    }
    defaults.update(kw)
    return PendingInterpretation(**defaults)


def make_message(**kw: Any) -> SessionMessage:
    defaults: dict[str, Any] = {
        "user_id": "user-1",
        "pet_id": "pet-1",
        "session_id": "sess-1",
        "role": "user",
        "content": "团团今天吃了罐头",
        "at": NOW,
    }
    defaults.update(kw)
    return SessionMessage(**defaults)


def vec(seed: float) -> list[float]:
    """确定性向量。不做归一化 —— 余弦相似度自己会处理。"""
    return [seed, seed * 0.5, 1.0 - seed] + [0.0] * 1021


# =============================================================================
# 档案
# =============================================================================


class TestPets:
    def test_save_and_get_roundtrip(self, store: Any):
        pet = make_pet()
        store.save_pet(pet)
        got = store.get_pet(user_id="user-1", pet_id="pet-1")

        assert got.pet_id == "pet-1"
        assert got.name == "团团"
        assert got.breed == "英短"
        assert got.visual.fur_color == "橘白"
        # 身份锚点与不稳定特征必须**分开**返回 —— 合并会让调用方
        # 分不清哪个能用于校验
        assert got.must_keep_features == ["橘白短毛", "圆脸"]
        assert got.observed_but_unstable == ["胸口白毛"]
        assert got.identity_prompt

    def test_get_pet_wrong_user_is_not_found(self, store: Any):
        store.save_pet(make_pet())
        with pytest.raises(NotFound):
            store.get_pet(user_id="user-2", pet_id="pet-1")

    def test_get_missing_pet_is_not_found(self, store: Any):
        with pytest.raises(NotFound):
            store.get_pet(user_id="user-1", pet_id="nope")

    def test_list_pets_is_tenant_scoped(self, store: Any):
        store.save_pet(make_pet(pet_id="pet-1", user_id="user-1", name="团团"))
        store.save_pet(make_pet(pet_id="pet-2", user_id="user-2", name="别人家的"))

        mine = store.list_pets(user_id="user-1")
        assert [p.pet_id for p in mine] == ["pet-1"]

    def test_save_pet_is_upsert(self, store: Any):
        store.save_pet(make_pet(name="团团"))
        store.save_pet(make_pet(name="团团改名了"))

        pets = store.list_pets(user_id="user-1")
        assert len(pets) == 1, "同一 pet_id 重复保存应更新，而不是插入第二条"
        assert pets[0].name == "团团改名了"

    def test_datetime_roundtrip_keeps_utc(self, store: Any):
        """时区必须原样回来。

        领域模型是 aware UTC，MySQL DATETIME 不带时区 ——
        不在读写两端统一转换的话，时间会**静默偏移**。

        对比基准是**保存时那个对象的 `created_at`**，不是模块级的 NOW：
        `PetProfile` 自己会生成默认时间，拿一个外部常量去比，
        实际上测的是「构造与断言之间的耗时」。
        """
        saved = store.save_pet(make_pet())
        got = store.get_pet(user_id="user-1", pet_id="pet-1")

        assert got.created_at.tzinfo is not None, "读回来必须是 aware 的"
        delta = abs((got.created_at - saved.created_at).total_seconds())
        assert delta < 1, (
            f"时间偏移了 {delta} 秒（存 {saved.created_at} → 读 {got.created_at}），"
            f"说明读写两端的时区转换不一致"
        )


# =============================================================================
# 记忆
# =============================================================================


class TestMemories:
    def test_insert_assigns_id(self, store: Any):
        stored = store.insert_memory(make_event(memory_id=None))
        assert stored.memory_id, "内存实现与 MySQL 都必须补一个 id"

    def test_get_memory_roundtrip(self, store: Any):
        stored = store.insert_memory(make_event(content="它怕吸尘器"))
        got = store.get_memory(
            user_id="user-1", pet_id="pet-1", memory_id=stored.memory_id
        )
        assert got.content == "它怕吸尘器"
        assert got.subject == "vacuum"
        assert got.polarity is Polarity.NEGATIVE
        assert got.event_type is EventType.PREFERENCE

    def test_get_memory_wrong_tenant_is_not_found(self, store: Any):
        """**隔离的核心断言。**"""
        stored = store.insert_memory(make_event())
        with pytest.raises(NotFound):
            store.get_memory(user_id="user-2", pet_id="pet-1", memory_id=stored.memory_id)
        with pytest.raises(NotFound):
            store.get_memory(user_id="user-1", pet_id="pet-2", memory_id=stored.memory_id)

    def test_duplicate_dedup_key_raises(self, store: Any):
        """去重键冲突必须抛 `DuplicateMemory`，让飞轮转成强化。

        不抛的话会插入第二条 —— 于是同一件事在检索结果里占两个位置，
        而「强化」这条机制永远不会被触发。
        """
        store.insert_memory(make_event(dedup_key="k1"))
        with pytest.raises(DuplicateMemory):
            store.insert_memory(make_event(dedup_key="k1"))

    def test_null_dedup_key_allows_duplicates(self, store: Any):
        """没有 dedup_key 时不应去重 —— 那是两条独立观察。"""
        store.insert_memory(make_event(dedup_key=None))
        store.insert_memory(make_event(dedup_key=None))
        assert len(store.list_memories(user_id="user-1", pet_id="pet-1")) == 2

    def test_find_by_dedup_key(self, store: Any):
        stored = store.insert_memory(make_event(dedup_key="k1"))
        found = store.find_by_dedup_key(user_id="user-1", pet_id="pet-1", dedup_key="k1")
        assert found is not None
        assert found.memory_id == stored.memory_id

        assert store.find_by_dedup_key(user_id="user-2", pet_id="pet-1", dedup_key="k1") is None

    def test_update_memory(self, store: Any):
        stored = store.insert_memory(make_event(support_count=1))
        updated = stored.model_copy(update={"support_count": 5, "content": "改过了"})
        store.update_memory(updated)

        got = store.get_memory(
            user_id="user-1", pet_id="pet-1", memory_id=stored.memory_id
        )
        assert got.support_count == 5
        assert got.content == "改过了"

    def test_update_missing_raises(self, store: Any):
        ghost = make_event(memory_id="ghost")
        with pytest.raises(NotFound):
            store.update_memory(ghost)

    def test_list_memories_excludes_non_retrievable_by_default(self, store: Any):
        """默认只返回 active / pending_confirmation。

        superseded 的记忆若被当成现行事实，会出现「它以前喜欢 X」
        与「它现在喜欢 X」同时成立的自相矛盾输出。
        """
        store.insert_memory(make_event(memory_id="m-active"))
        store.insert_memory(
            make_event(memory_id="m-superseded", status=MemoryStatus.SUPERSEDED)
        )

        default = store.list_memories(user_id="user-1", pet_id="pet-1")
        assert {m.memory_id for m in default} == {"m-active"}

        allof = store.list_memories(
            user_id="user-1", pet_id="pet-1", include_non_active=True
        )
        assert {m.memory_id for m in allof} == {"m-active", "m-superseded"}

    def test_list_memories_is_tenant_scoped(self, store: Any):
        store.insert_memory(make_event(memory_id="mine", user_id="user-1", pet_id="pet-1"))
        store.insert_memory(make_event(memory_id="theirs", user_id="user-2", pet_id="pet-1"))
        store.insert_memory(make_event(memory_id="other-pet", user_id="user-1", pet_id="pet-2"))

        got = store.list_memories(user_id="user-1", pet_id="pet-1")
        assert {m.memory_id for m in got} == {"mine"}

    def test_list_memories_session_filter(self, store: Any):
        store.insert_memory(make_event(memory_id="s1", session_id="sess-1"))
        store.insert_memory(make_event(memory_id="s2", session_id="sess-2"))

        got = store.list_memories(user_id="user-1", pet_id="pet-1", session_id="sess-1")
        assert {m.memory_id for m in got} == {"s1"}

    def test_find_conflicts_opposite_polarity(self, store: Any):
        """同主体 + 反极性 + 时间重叠 → 冲突。"""
        store.insert_memory(
            make_event(
                memory_id="old",
                subject="vacuum",
                polarity=Polarity.NEGATIVE,
                content="它怕吸尘器",
            )
        )
        candidate = make_event(
            memory_id="new",
            subject="vacuum",
            polarity=Polarity.POSITIVE,
            content="它不怕吸尘器了",
        )
        conflicts = store.find_conflicts(user_id="user-1", pet_id="pet-1", candidate=candidate)
        assert {c.memory_id for c in conflicts} == {"old"}

    def test_find_conflicts_is_tenant_scoped(self, store: Any):
        store.insert_memory(
            make_event(
                memory_id="theirs",
                user_id="user-2",
                subject="vacuum",
                polarity=Polarity.NEGATIVE,
            )
        )
        candidate = make_event(
            memory_id="new", subject="vacuum", polarity=Polarity.POSITIVE
        )
        assert store.find_conflicts(user_id="user-1", pet_id="pet-1", candidate=candidate) == []

    def test_invariant_active_system_inference_rejected(self, store: Any):
        """不变量 I1（防自我强化）在**两套实现**里都必须挡。

        内存实现靠 `_assert_invariants`，MySQL 靠 DB CHECK + 契约层。
        两边都挡不住的话，系统推断会变成「事实」，
        而它的幻觉会随对话轮次放大（docs/04 §3.5 R1）。
        """
        with pytest.raises(Exception):  # noqa: B017 - 契约层与存储层都可能抛
            store.insert_memory(
                make_event(
                    memory_id="bad",
                    source=MemorySource.SYSTEM_INFERENCE,
                    status=MemoryStatus.ACTIVE,
                )
            )


# =============================================================================
# 向量召回
# =============================================================================


class TestVectorSearch:
    def test_search_returns_indexed_memories(self, store: Any):
        a = store.insert_memory(make_event(memory_id="m1"), vector=vec(1.0))
        b = store.insert_memory(make_event(memory_id="m2"), vector=vec(0.1))
        assert {a.memory_id, b.memory_id} == {"m1", "m2"}

        items = store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(1.0), limit=10
        )
        assert len(items) == 2
        # 更接近的排在前面
        assert items[0].event.memory_id == "m1"

    def test_search_is_tenant_scoped(self, store: Any):
        """**隔离的正确性底线**：别的宠物的记忆根本不进入候选。"""
        store.insert_memory(
            make_event(memory_id="theirs", user_id="user-2", pet_id="pet-1"),
            vector=vec(1.0),
        )
        store.insert_memory(
            make_event(memory_id="other-pet", user_id="user-1", pet_id="pet-2"),
            vector=vec(1.0),
        )
        store.insert_memory(make_event(memory_id="mine", user_id="user-1", pet_id="pet-1"),
                            vector=vec(1.0))

        items = store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(1.0), limit=10
        )
        assert {i.event.memory_id for i in items} == {"mine"}

    def test_search_respects_limit(self, store: Any):
        for i in range(5):
            store.insert_memory(make_event(memory_id=f"m{i}"), vector=vec(0.1 * i))
        items = store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(0.5), limit=2
        )
        assert len(items) == 2

    def test_search_only_returns_retrievable(self, store: Any):
        store.insert_memory(
            make_event(memory_id="superseded", status=MemoryStatus.SUPERSEDED),
            vector=vec(1.0),
        )
        items = store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(1.0), limit=10
        )
        assert items == []

    def test_search_without_vectors_returns_empty(self, store: Any):
        store.insert_memory(make_event(memory_id="m1"))  # 无向量
        items = store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(1.0), limit=10
        )
        assert items == []


# =============================================================================
# 叫声记录
# =============================================================================


class TestMeowRecords:
    def test_insert_and_list_roundtrip(self, store: Any):
        stored = store.insert_meow_record(make_meow())
        assert stored.record_id

        got = store.list_meow_records(user_id="user-1", pet_id="pet-1")
        assert len(got) == 1
        assert got[0].context is ContextLabel.FOOD_WAITING
        assert got[0].actions == [BehaviorAction.NEAR_FOOD_BOWL]
        assert got[0].resolution == "给了罐头就停了"
        assert got[0].features.f0_mean == pytest.approx(520.0)

    def test_list_excludes_unconfirmed_by_default(self, store: Any):
        store.insert_meow_record(make_meow(record_id="confirmed"))
        store.insert_meow_record(
            make_meow(record_id="pending", status=MemoryStatus.PENDING_CONFIRMATION)
        )

        default = store.list_meow_records(user_id="user-1", pet_id="pet-1")
        assert {r.record_id for r in default} == {"confirmed"}

        allof = store.list_meow_records(
            user_id="user-1", pet_id="pet-1", only_confirmed=False
        )
        assert {r.record_id for r in allof} == {"confirmed", "pending"}

    def test_list_is_tenant_scoped(self, store: Any):
        store.insert_meow_record(make_meow(record_id="mine"))
        store.insert_meow_record(make_meow(record_id="theirs", user_id="user-2"))

        got = store.list_meow_records(user_id="user-1", pet_id="pet-1")
        assert {r.record_id for r in got} == {"mine"}

    def test_list_ordered_by_recorded_at(self, store: Any):
        store.insert_meow_record(make_meow(record_id="later", recorded_at=NOW))
        store.insert_meow_record(
            make_meow(record_id="earlier", recorded_at=NOW - timedelta(hours=2))
        )
        got = store.list_meow_records(user_id="user-1", pet_id="pet-1")
        # 平局时按时间稳定排序 —— 案例推理的「最像的那次」不能随存储顺序变
        assert [r.record_id for r in got] == ["earlier", "later"]

    def test_features_roundtrip_preserves_unavailable(self, store: Any):
        """`unavailable` 必须原样往返。

        丢了它，推理侧就会把「测不出」当成「测出来是 0」——
        那是静默编造（DESIGN §3.6）。
        """
        feats = make_features()
        feats = feats.model_copy(update={"unavailable": ["call_rate", "ici_mean"]})
        store.insert_meow_record(make_meow(features=feats))

        got = store.list_meow_records(user_id="user-1", pet_id="pet-1")[0]
        assert set(got.features.unavailable) == {"call_rate", "ici_mean"}


# =============================================================================
# 待标注解释
# =============================================================================


class TestPendingInterpretations:
    def test_save_and_get(self, store: Any):
        store.save_pending_interpretation(make_pending())
        got = store.get_pending_interpretation(
            user_id="user-1", pet_id="pet-1", interpretation_id="interp-1"
        )
        assert got is not None
        assert got.evidence_mode is EvidenceMode.MEASURED_ONLY
        assert got.features.f0_mean == pytest.approx(520.0)
        assert len(got.candidates) == 1
        assert got.candidates[0].context is ContextLabel.FOOD_WAITING

    def test_wrong_tenant_returns_none(self, store: Any):
        """归属不符返回 None 而不是抛错 —— 不泄露资源存在性。

        这与 `get_memory` 抛 `NotFound` 是**刻意的差异**：
        这个方法的调用方（标注端点）需要区分「找不到」与「出错」，
        而两者对外都映射成 404。
        """
        store.save_pending_interpretation(make_pending())
        assert (
            store.get_pending_interpretation(
                user_id="user-2", pet_id="pet-1", interpretation_id="interp-1"
            )
            is None
        )
        assert (
            store.get_pending_interpretation(
                user_id="user-1", pet_id="pet-2", interpretation_id="interp-1"
            )
            is None
        )

    def test_missing_returns_none(self, store: Any):
        assert (
            store.get_pending_interpretation(
                user_id="user-1", pet_id="pet-1", interpretation_id="nope"
            )
            is None
        )


# =============================================================================
# 会话消息
# =============================================================================


class TestSessionMessages:
    def test_insert_and_list_window(self, store: Any):
        store.insert_message(make_message(content="一", at=NOW - timedelta(days=2)))
        store.insert_message(make_message(content="二", at=NOW))

        windowed = store.list_messages(
            user_id="user-1", pet_id="pet-1", since=NOW - timedelta(days=1)
        )
        assert [m.content for m in windowed] == ["二"]

    def test_list_ordered_ascending(self, store: Any):
        store.insert_message(make_message(content="晚", at=NOW))
        store.insert_message(make_message(content="早", at=NOW - timedelta(hours=1)))

        got = store.list_messages(user_id="user-1", pet_id="pet-1")
        assert [m.content for m in got] == ["早", "晚"]

    def test_list_is_tenant_scoped(self, store: Any):
        store.insert_message(make_message(content="我的"))
        store.insert_message(make_message(content="别人的", user_id="user-2"))

        got = store.list_messages(user_id="user-1", pet_id="pet-1")
        assert [m.content for m in got] == ["我的"]

    def test_recent_messages_returns_newest_in_ascending_order(self, store: Any):
        """**最容易漂移的一处。**

        「最近 N 条」+「按时间正序返回」这两个要求方向相反：
        - 直接正序取 N 条 → 拿到**最旧**的 N 条（条数还对，静默错）
        - 倒序取 N 条后不反转 → 顺序反了，注入 prompt 时时间线倒流

        两套实现都必须先倒序取、再反转为正序。
        """
        for i in range(6):
            store.insert_message(
                make_message(content=f"msg{i}", at=NOW + timedelta(minutes=i))
            )

        recent = store.list_recent_messages(user_id="user-1", pet_id="pet-1", limit=3)
        assert [m.content for m in recent] == ["msg3", "msg4", "msg5"], (
            "必须是「最近 3 条」且按时间正序 —— 不是最旧的 3 条，也不是倒序的 3 条"
        )

    def test_recent_messages_session_filter(self, store: Any):
        store.insert_message(make_message(content="s1-早", session_id="s1", at=NOW))
        store.insert_message(
            make_message(content="s1-晚", session_id="s1", at=NOW + timedelta(minutes=1))
        )
        store.insert_message(
            make_message(content="s2", session_id="s2", at=NOW + timedelta(minutes=2))
        )

        got = store.list_recent_messages(
            user_id="user-1", pet_id="pet-1", session_id="s1", limit=10
        )
        assert [m.content for m in got] == ["s1-早", "s1-晚"]

    def test_recent_messages_zero_limit(self, store: Any):
        store.insert_message(make_message())
        assert store.list_recent_messages(user_id="user-1", pet_id="pet-1", limit=0) == []

    def test_recent_messages_is_tenant_scoped(self, store: Any):
        store.insert_message(make_message(content="我的"))
        store.insert_message(make_message(content="别人的", user_id="user-2"))
        store.insert_message(make_message(content="别的猫", pet_id="pet-2"))

        got = store.list_recent_messages(user_id="user-1", pet_id="pet-1", limit=10)
        assert [m.content for m in got] == ["我的"]


# =============================================================================
# 跨表：级联删除
# =============================================================================


class TestCascadeDelete:
    def test_delete_pet_data_removes_everything(self, store: Any):
        """级联硬删：软标记不满足要求（一条标了 deleted 的记录仍是泄露风险）。

        只对实现了 `delete_pet_data` 的存储断言 ——
        `MemoryStore` 协议本身没这个方法（它在 HealthRecordStore 上），
        这里是 MySQL 实现的额外能力。
        """
        deleter = getattr(store, "delete_pet_data", None)
        if not callable(deleter):
            pytest.skip("该后端没有 delete_pet_data")

        store.save_pet(make_pet())
        store.insert_memory(make_event(memory_id="m1"), vector=vec(1.0))
        store.insert_meow_record(make_meow(record_id="r1"))
        store.save_pending_interpretation(make_pending())
        store.insert_message(make_message())

        removed = deleter(user_id="user-1", pet_id="pet-1")
        assert removed >= 4, f"应删掉至少 4 条，实际 {removed}"

        assert store.list_memories(user_id="user-1", pet_id="pet-1") == []
        assert store.list_meow_records(user_id="user-1", pet_id="pet-1", only_confirmed=False) == []
        assert store.list_messages(user_id="user-1", pet_id="pet-1") == []
        assert store.search_memories(
            user_id="user-1", pet_id="pet-1", query_vector=vec(1.0), limit=10
        ) == [], "向量索引里的条目也必须清掉，否则已删数据仍会影响检索"

    def test_delete_does_not_touch_other_tenants(self, store: Any):
        deleter = getattr(store, "delete_pet_data", None)
        if not callable(deleter):
            pytest.skip("该后端没有 delete_pet_data")

        store.insert_memory(make_event(memory_id="mine", user_id="user-1", pet_id="pet-1"))
        store.insert_memory(make_event(memory_id="theirs", user_id="user-2", pet_id="pet-1"))

        deleter(user_id="user-1", pet_id="pet-1")

        assert store.list_memories(user_id="user-2", pet_id="pet-1"), (
            "删除自己的数据不得影响别的租户"
        )


# =============================================================================
# 后端自身
# =============================================================================


def test_mysql_backend_actually_selected(store: Any, backend_name: str):
    """确认参数化真的在跑 MySQL，而不是两边都悄悄跑了内存。

    这是最隐蔽的一种「假绿」：夹具写错时 params 里两个值都返回内存实现，
    于是所有断言都通过，而 MySQL 一行都没执行。
    """
    if backend_name == "mysql":
        assert not isinstance(store, _MemStore), (
            "backend_name 说是 mysql，但拿到的是内存实现 —— 夹具写错了"
        )
        assert type(store).__name__ == "MySQLStore"
    else:
        assert isinstance(store, _MemStore)
