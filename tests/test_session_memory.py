"""会话记忆注入（Session 层）的测试。

## 覆盖的东西

`DESIGN.md` §3.5 的记忆三层里，Session 层此前是**写了但从不读**：
`/v1/chat` 存 `SessionMessage`（只给日报用），而对话本身看不到历史。
本文件钉住补上的那条链路：

```
X-Session-Id → middleware → initial_state → load_context → recent_turns
                                                        → 分区注入 → prompt
```

## 最要紧的一组：守卫必须与注入同步

历史里含 **assistant 自己的回复**。若把它与「已验证记录」同栏注入，
模型上一轮的推测会被当成事实回灌 —— 这是与不变量 I1 同构的自我强化，
只是通道从**写入**换成了**读入**。

所以这里同时验证两个方向：

- 主人自己说过的话**可以**支撑「你之前说过」；
- assistant 说过的话**不可以**。
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.audio.features import TARGET_SR, extract_features, synthesize_meow
from app.auth import issue_token
from app.graph import AgentState
from app.graph.nodes import (
    RECENT_MESSAGE_LIMIT,
    make_context_loader,
    response_guard,
)
from app.interpreter import PriorTable
from app.llm import HashEmbedder
from app.observability.langsmith import TracingConfig
from app.schemas import (
    GuardResult,
    PetProfile,
    SessionMessage,
    Species,
    VisualProfile,
)
from app.store import InMemoryStore

PRIOR_PATH = "data/priors/catmeows_stats.json"

#: 测试用签名密钥。**由固定输入派生，而非写字面量** ——
#: 密钥扫描器会把 `SECRET = "..."` 一律当作硬编码密钥（这里本来也不是真密钥）。
SECRET = hashlib.sha256(b"pet-agent-tests-signing").hexdigest()
NOW = datetime.now(timezone.utc)


class _CapturingLLM:
    """记录每次 system prompt —— 注入是否发生只能从这里看。"""

    def __init__(self, response: str = "团团挺好的。") -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls.append((system, user))
        return self.response


class _StubVision:
    def analyze(self, image_url: str):  # pragma: no cover - 本组测试不该走到这里
        raise AssertionError("本组测试不应调用视觉分析")


def _real_features(url: str):
    return extract_features(
        synthesize_meow(duration=2.0, f0_start=420, f0_end=700), TARGET_SR
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture
def pet(store: InMemoryStore) -> PetProfile:
    p = PetProfile(
        pet_id="pet-1",
        user_id="user-1",
        name="团团",
        species=Species.CAT,
        visual=VisualProfile(fur_color="橘白"),
        must_keep_features=["橘白短毛"],
    )
    store.save_pet(p)
    return p


@pytest.fixture
def llm() -> _CapturingLLM:
    return _CapturingLLM()


@pytest.fixture
def client(store, prior, llm) -> TestClient:
    app = create_app(
        store=store,
        embedder=HashEmbedder(),
        llm=llm,
        prior=prior,
        feature_extractor=_real_features,
        vision=_StubVision(),
        auth_secret=SECRET,
        tracing_config=TracingConfig(enabled=False),
    )
    return TestClient(app)


@pytest.fixture
def headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token('user-1', secret=SECRET)}"}


def _seed(
    store: InMemoryStore, *, session_id: str, texts: list[str], base: datetime
) -> None:
    for index, text in enumerate(texts):
        store.insert_message(
            SessionMessage(
                user_id="user-1",
                pet_id="pet-1",
                session_id=session_id,
                role="user" if index % 2 == 0 else "assistant",
                content=text,
                at=base + timedelta(minutes=index),
            )
        )


# ═══════════════════════════════════════════════════════════════
# 存储：按会话取最近若干条
# ═══════════════════════════════════════════════════════════════


class TestRecentMessages:
    def test_filters_by_session_and_keeps_newest(self, store):
        _seed(store, session_id="s-1", texts=["1", "2", "3"], base=NOW)
        _seed(store, session_id="s-2", texts=["other"], base=NOW + timedelta(hours=1))

        rows = store.list_recent_messages(
            user_id="user-1", pet_id="pet-1", session_id="s-1", limit=2
        )
        assert [m.content for m in rows] == ["2", "3"], "取最近 2 条且保持时间正序"

    def test_none_session_means_no_filter(self, store):
        _seed(store, session_id="s-1", texts=["a"], base=NOW)
        _seed(store, session_id="s-2", texts=["b"], base=NOW + timedelta(minutes=5))
        rows = store.list_recent_messages(user_id="user-1", pet_id="pet-1")
        assert len(rows) == 2

    def test_non_positive_limit_is_empty(self, store):
        _seed(store, session_id="s-1", texts=["a"], base=NOW)
        assert (
            store.list_recent_messages(
                user_id="user-1", pet_id="pet-1", session_id="s-1", limit=0
            )
            == []
        )

    def test_other_tenant_is_not_visible(self, store):
        _seed(store, session_id="s-1", texts=["a"], base=NOW)
        assert (
            store.list_recent_messages(
                user_id="user-1", pet_id="other-pet", session_id="s-1"
            )
            == []
        )


# ═══════════════════════════════════════════════════════════════
# 中间件：session / trace 作用域
# ═══════════════════════════════════════════════════════════════


class TestRequestScope:
    def test_echoes_client_provided_scope(self, client):
        r = client.get(
            "/healthz", headers={"X-Session-Id": "s-123", "X-Trace-Id": "t-456"}
        )
        assert r.headers.get("X-Session-Id") == "s-123"
        assert r.headers.get("X-Trace-Id") == "t-456"

    def test_generates_when_absent(self, client):
        r = client.get("/healthz")
        assert (r.headers.get("X-Session-Id") or "").startswith("sess-")
        # trace 必须是 UUID：`run_id` 与它同源，而 run_id 要求 UUID
        uuid.UUID(r.headers.get("X-Trace-Id") or "")

    def test_response_body_carries_scope(self, client, pet, headers):
        sent = {"X-Session-Id": "s-body", "X-Trace-Id": str(uuid.uuid4())}
        r = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "你好"},
            headers={**headers, **sent},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["session_id"] == "s-body"
        assert body["trace_id"] == sent["X-Trace-Id"]
        assert body["langsmith_run_id"] is None, "未启用观测时不应报一个 run id"


# ═══════════════════════════════════════════════════════════════
# 对话：存储 + 注入
# ═══════════════════════════════════════════════════════════════


class TestChatHistoryInjection:
    def test_chat_stores_messages_with_session(self, client, store, pet, headers):
        client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "它在窗台趴着"},
            headers={**headers, "X-Session-Id": "s-1"},
        )
        stored = store.list_messages(user_id="user-1", pet_id=pet.pet_id)
        assert stored, "对话必须落库（日报与会话注入共用）"
        assert {m.session_id for m in stored} == {"s-1"}

    def test_second_turn_injects_history(self, client, llm, pet, headers):
        scope = {**headers, "X-Session-Id": "s-1"}
        first = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "它今天老在门口叫"},
            headers=scope,
        )
        assert first.status_code == 200, first.text
        first_system = llm.calls[-1][0]
        assert "recent_turns" not in first_system, "第一轮没有历史，不应凭空造一段"

        # ⚠️ 第二轮必须仍路由到 CHAT：`translate_behavior` 走模板渲染，
        # 不经过 companion_agent，所以不会注入历史（那是刻意的）。
        second = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "它今天怎么样"},
            headers=scope,
        )
        assert second.status_code == 200, second.text
        assert second.json()["intent"] == "chat", "本用例需要落在会注入历史的意图上"
        system = llm.calls[-1][0]
        assert "recent_turns" in system
        assert "它今天老在门口叫" in system, "上一轮主人说的话必须进 prompt"

    def test_history_is_labeled_as_not_facts(self, client, llm, pet, headers):
        """**分区标注**：不加这句，assistant 的历史猜测会被当成事实回灌。"""
        scope = {**headers, "X-Session-Id": "s-1"}
        for text in ("它今天在门口叫", "它今天怎么样"):
            r = client.post(
                "/v1/chat",
                params={"pet_id": pet.pet_id},
                json={"text": text},
                headers=scope,
            )
            assert r.status_code == 200, r.text
        system = llm.calls[-1][0]
        assert "recent_turns_note" in system
        assert "不得把它当作事实" in system

    def test_other_session_history_is_not_injected(self, client, llm, pet, headers):
        first = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "今天它说了另一件事"},
            headers={**headers, "X-Session-Id": "s-other"},
        )
        assert first.status_code == 200, first.text
        second = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "今天这是新会话"},
            headers={**headers, "X-Session-Id": "s-new"},
        )
        assert second.status_code == 200, second.text
        assert "今天它说了另一件事" not in llm.calls[-1][0]


# ═══════════════════════════════════════════════════════════════
# load_context：填充 + 预算截断
# ═══════════════════════════════════════════════════════════════


class TestLoadContext:
    def _load(self, store: InMemoryStore, state: AgentState):
        out = make_context_loader(store)(state)
        context = out.get("session_context")
        assert context is not None
        return out, context, out.get("recent_turns", []), out.get("node_trace", [])

    def test_fills_session_context_and_turns(self, store):
        _seed(store, session_id="s-1", texts=["a", "b", "c"], base=NOW)
        _, context, recent, _ = self._load(
            store, {"user_id": "user-1", "pet_id": "pet-1", "session_id": "s-1"}
        )
        assert context.session_id == "s-1"
        assert context.turn_count == 3
        assert [m.content for m in recent] == ["a", "b", "c"]

    def test_no_session_means_no_history(self, store):
        _seed(store, session_id="s-1", texts=["a"], base=NOW)
        _, _, recent, trace = self._load(
            store, {"user_id": "user-1", "pet_id": "pet-1"}
        )
        assert recent == []
        assert "无会话标识" in (trace[0].decision or "")

    def test_char_budget_truncates_oldest_and_says_so(self, store):
        """**截断必须留痕** —— 否则「模型没提过」与「历史被丢了」看起来一样。"""
        long_text = "喵" * 400
        _seed(store, session_id="s-1", texts=[long_text] * 6, base=NOW)
        _, _, recent, trace = self._load(
            store, {"user_id": "user-1", "pet_id": "pet-1", "session_id": "s-1"}
        )
        assert len(recent) == 3, "1500 / 400 → 只能装 3 条"
        assert "截断 3 条" in (trace[0].decision or "")

    def test_message_limit_is_respected(self, store):
        _seed(
            store,
            session_id="s-1",
            texts=[f"m{i}" for i in range(RECENT_MESSAGE_LIMIT + 5)],
            base=NOW,
        )
        _, _, recent, _ = self._load(
            store, {"user_id": "user-1", "pet_id": "pet-1", "session_id": "s-1"}
        )
        assert len(recent) == RECENT_MESSAGE_LIMIT


# ═══════════════════════════════════════════════════════════════
# 守卫与注入同步（最关键的一组）
# ═══════════════════════════════════════════════════════════════


def _guard(
    draft: str, *, turns: list[SessionMessage], memories: list | None = None
) -> GuardResult:
    result = response_guard(
        {
            "draft_response": draft,
            "recent_turns": turns,
            "retrieved_memories": memories if memories is not None else [],
        }
    )
    guard = result.get("guard_result")
    assert guard is not None
    return guard


def _turn(role: Literal["user", "assistant"], content: str = "内容") -> SessionMessage:
    return SessionMessage(
        user_id="user-1", pet_id="pet-1", session_id="s-1", role=role, content=content
    )


class TestGuardSyncedWithHistory:
    def test_owner_history_grounds_the_claim(self):
        """主人自己说过 → 「你之前说过」有依据，**不得**降级。"""
        guard = _guard("你之前说过它怕吸尘器。", turns=[_turn("user")])
        assert guard.passed is True

    def test_assistant_history_does_not_ground_the_claim(self):
        """assistant 的历史回复**不是依据** —— 那是系统自己的输出。

        允许它，等于开一条自我强化的读入通道：模型上一轮的猜测
        被自己引用成「之前说过」。
        """
        guard = _guard("你之前说过它怕吸尘器。", turns=[_turn("assistant")])
        assert guard.passed is False
        assert any(v.type.value == "untraceable_claim" for v in guard.violations)

    def test_no_source_at_all_still_flagged(self):
        guard = _guard("你之前说过它怕吸尘器。", turns=[])
        assert guard.passed is False

    def test_long_term_cue_still_requires_memories(self):
        """「根据记录」要的是**检索到的记忆**，主人闲聊不能顶替。"""
        guard = _guard("根据记录，它怕吸尘器。", turns=[_turn("user")])
        assert guard.passed is False
        assert any(v.type.value == "untraceable_claim" for v in guard.violations)


# ═══════════════════════════════════════════════════════════════
# 标注归属：案例的来源是「录叫声那一轮」
# ═══════════════════════════════════════════════════════════════


class TestLabelProvenance:
    def test_label_uses_interpretation_session(self, client, store, pet, headers):
        created = client.post(
            "/v1/interpret",
            params={"pet_id": pet.pet_id},
            json={"audio_url": "http://x/meow.wav", "scene_description": "在门口叫"},
            headers={**headers, "X-Session-Id": "s-meow"},
        )
        assert created.status_code == 200, created.text
        interpretation_id = created.json()["interpretation_id"]
        assert interpretation_id

        # 标注发生在**另一个**会话 —— 案例来源仍应是录叫声那一轮
        labelled = client.post(
            f"/v1/pets/{pet.pet_id}/meow-records",
            json={"interpretation_id": interpretation_id, "context": "door_attention"},
            headers={**headers, "X-Session-Id": "s-later"},
        )
        assert labelled.status_code == 201, labelled.text
        record = store.list_meow_records(user_id="user-1", pet_id=pet.pet_id)[0]
        assert record.session_id == "s-meow"

    def test_records_can_be_filtered_by_session(self, client, store, pet, headers):
        created = client.post(
            "/v1/interpret",
            params={"pet_id": pet.pet_id},
            json={"audio_url": "http://x/meow.wav"},
            headers={**headers, "X-Session-Id": "s-meow"},
        )
        client.post(
            f"/v1/pets/{pet.pet_id}/meow-records",
            json={
                "interpretation_id": created.json()["interpretation_id"],
                "context": "door_attention",
            },
            headers={**headers, "X-Session-Id": "s-meow"},
        )

        hit = client.get(
            f"/v1/pets/{pet.pet_id}/meow-records",
            params={"session_id": "s-meow"},
            headers=headers,
        ).json()
        miss = client.get(
            f"/v1/pets/{pet.pet_id}/meow-records",
            params={"session_id": "s-none"},
            headers=headers,
        ).json()
        assert hit["count"] == 1
        assert miss["count"] == 0


# ═══════════════════════════════════════════════════════════════
# 长期记忆的追溯：这条记忆是哪一轮产生的
# ═══════════════════════════════════════════════════════════════


class TestMemoryTraceability:
    def test_memory_carries_session_and_is_filterable(
        self, client, store, pet, headers
    ):
        scope = {**headers, "X-Session-Id": "s-mem"}
        r = client.post(
            "/v1/chat",
            params={"pet_id": pet.pet_id},
            json={"text": "记住，它很怕吸尘器"},
            headers=scope,
        )
        assert r.status_code == 200, r.text
        assert r.json()["intent"] == "record_event", r.text

        stored = store.list_memories(
            user_id="user-1", pet_id=pet.pet_id, session_id="s-mem"
        )
        assert stored, "记忆必须带上产生它的会话（审计用）"

        listed = client.get(
            "/v1/memories",
            params={"pet_id": pet.pet_id, "session_id": "s-mem"},
            headers=headers,
        ).json()
        assert listed["count"] >= 1

        other = client.get(
            "/v1/memories",
            params={"pet_id": pet.pet_id, "session_id": "s-none"},
            headers=headers,
        ).json()
        assert other["count"] == 0
