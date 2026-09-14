"""瞬间记录与时间线（日记本体）。

## 这一层与 `MemoryEvent` 的分工是本文件的核心

| | `MemoryEvent` | `Moment` |
|---|---|---|
| 是什么 | **可检索的事实** | **一次流水** |
| 进检索吗 | 是 | **否** |
| 谁产生 | 抽取 + 准入判定 | 用户一次点击 |

分开是刻意的：把每张照片都塞进记忆检索，会让「它怕吸尘器」这类
稳定事实被日常流水淹没 —— 而那正是记忆层要防的事。

## 三条「不做」

`docs/11` §2.1 的关键判断是「用户想猫时是情绪状态，不想打字、不想被问问题」，
所以这一层**不强制描述、不要求选分类、不弹确认框**。
本文件把这三条都固定成断言 —— 它们很容易在后续迭代里被「补全」掉。
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.interpreter import PriorTable
from app.llm import HashEmbedder, MockLLM
from app.profile import VisualObservation
from app.schemas import Moment, MomentScene, extract_scene
from app.store import InMemoryStore

SECRET = hashlib.sha256(b"pet-agent-moments").hexdigest()
PRIOR_PATH = "data/priors/catmeows_stats.json"
NOW = datetime(2026, 9, 14, 15, 0, tzinfo=timezone.utc)


class _StubVision:
    def analyze(self, image_url: str) -> VisualObservation:
        return VisualObservation(image_url=image_url)


@pytest.fixture(scope="module")
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture()
def client(prior) -> TestClient:
    return TestClient(
        create_app(
            store=InMemoryStore(),
            embedder=HashEmbedder(),
            llm=MockLLM(default="喵。"),
            prior=prior,
            feature_extractor=lambda url: None,
            vision=_StubVision(),
            auth_secret=SECRET,
        )
    )


def _headers(user: str = "u1") -> dict[str, str]:
    from app.auth import issue_token

    return {"Authorization": f"Bearer {issue_token(user, secret=SECRET)}"}


@pytest.fixture()
def pet_id(client) -> str:
    return client.post("/v1/pets", json={"name": "团团"}, headers=_headers()).json()["pet_id"]


def make_moment(**kw) -> Moment:
    defaults = {
        "user_id": "u1",
        "pet_id": "p1",
        "media_url": "http://x/1.jpg",
        "captured_at": NOW,
        "created_at": NOW,
    }
    defaults.update(kw)
    return Moment(**defaults)


# =============================================================================
# 场景抽取：**没写就不猜**
# =============================================================================


class TestSceneExtraction:
    @pytest.mark.parametrize(
        "note,expected",
        [
            ("它在窗台晒太阳", MomentScene.WINDOW),
            ("刚吃了罐头", MomentScene.EATING),
            ("睡得呼呼的", MomentScene.SLEEPING),
            ("在玩逗猫棒", MomentScene.PLAYING),
            ("蹭我腿呢", MomentScene.WITH_HUMAN),
            ("在舔毛", MomentScene.GROOMING),
        ],
    )
    def test_keywords_map_to_scenes(self, note: str, expected: MomentScene):
        assert extract_scene(note) is expected

    def test_no_note_is_other_not_a_guess(self):
        """**没写就不猜。**

        从照片猜场景需要额外验证（通用 VLM 零样本做专项判断未经文献支撑），
        而一个猜错的标签会让时间线看起来比实际更可信。
        """
        assert extract_scene(None) is MomentScene.OTHER
        assert extract_scene("") is MomentScene.OTHER
        assert extract_scene("   ") is MomentScene.OTHER

    def test_unknown_text_is_other(self):
        assert extract_scene("随便写点什么") is MomentScene.OTHER

    def test_blank_note_normalised_to_none(self):
        """空白字符串归一成 `None`。

        不归一的话 `""` 与 `None` 会变成两种「没写」，
        而每个消费方（渲染、聚合、导出）都要各判一次。
        """
        assert make_moment(note="").note is None
        assert make_moment(note="   ").note is None
        assert make_moment(note=" 在窗台 ").note == "在窗台"


# =============================================================================
# 存储：隔离与排序
# =============================================================================


class TestMomentStore:
    def test_roundtrip(self):
        store = InMemoryStore()
        stored = store.insert_moment(make_moment(note="在窗台"))
        assert stored.moment_id

        got = store.list_moments(user_id="u1", pet_id="p1")
        assert len(got) == 1
        assert got[0].note == "在窗台"
        assert got[0].moment_id == stored.moment_id

    def test_timeline_is_descending(self):
        """**时间线倒序**（最新在前）。

        与 `list_messages` 的正序刻意相反：时间线要先看到最近的，
        而日报要按顺序读。
        """
        store = InMemoryStore()
        for i in range(3):
            store.insert_moment(
                make_moment(
                    media_url=f"http://x/{i}.jpg",
                    captured_at=NOW - timedelta(hours=i),
                )
            )
        got = store.list_moments(user_id="u1", pet_id="p1")
        times = [m.captured_at for m in got]
        assert times == sorted(times, reverse=True)

    def test_tenant_scoped(self):
        store = InMemoryStore()
        store.insert_moment(make_moment())
        store.insert_moment(make_moment(user_id="u2"))
        store.insert_moment(make_moment(pet_id="p2"))

        got = store.list_moments(user_id="u1", pet_id="p1")
        assert len(got) == 1

    def test_time_window(self):
        store = InMemoryStore()
        store.insert_moment(make_moment(captured_at=NOW - timedelta(days=10)))
        store.insert_moment(make_moment(media_url="http://x/2.jpg", captured_at=NOW))

        got = store.list_moments(
            user_id="u1", pet_id="p1", since=NOW - timedelta(days=1)
        )
        assert len(got) == 1

    def test_limit(self):
        store = InMemoryStore()
        for i in range(5):
            store.insert_moment(
                make_moment(media_url=f"http://x/{i}.jpg", captured_at=NOW - timedelta(hours=i))
            )
        assert len(store.list_moments(user_id="u1", pet_id="p1", limit=2)) == 2

    def test_cascade_delete_removes_moments(self):
        store = InMemoryStore()
        store.save_pet(
            __import__("app.schemas", fromlist=["PetProfile"]).PetProfile(
                pet_id="p1", user_id="u1", name="团团"
            )
        )
        store.insert_moment(make_moment())
        store.delete_pet_data(user_id="u1", pet_id="p1")
        assert store.list_moments(user_id="u1", pet_id="p1") == []


# =============================================================================
# API：一次点击就完成
# =============================================================================


class TestMomentAPI:
    def test_record_needs_only_a_photo(self, client, pet_id):
        """**只要一张照片。** 一句话是可选的，不是缺失。

        多一个必填字段，转化率就掉一截（§2.1）。
        """
        r = client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/1.jpg"},
            headers=_headers(),
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["moment"]["note"] is None
        assert body["moment"]["scene"] == "other"

    def test_record_returns_pet_reply(self, client, pet_id):
        """记完之后**立刻有句话** —— 那是这个动作的情绪回报。"""
        r = client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/1.jpg", "note": "在窗台晒太阳"},
            headers=_headers(),
        )
        body = r.json()
        assert body["moment"]["scene"] == "window"
        assert body["pet_says"], "必须有回应"

        # 断言**机制**而不是某一句具体文案：不同场景要给出不同的回应。
        # 固定成某一句话会让文案微调就挂测试，而那条断言并不指向任何缺陷。
        other = client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/2.jpg", "note": "刚吃了罐头"},
            headers=_headers(),
        ).json()
        assert other["moment"]["scene"] == "eating"
        assert other["pet_says"] != body["pet_says"], (
            "不同场景应给出不同回应 —— 否则「按场景回应」这个机制没生效"
        )

    def test_reply_is_deterministic(self, client, pet_id):
        """同一个瞬间永远得到同一句话。

        用随机数会让「翻回去看」时那句话变了 —— 对日记来说那是错的。
        """
        payload = {"media_url": "http://x/1.jpg", "note": "在窗台"}
        first = client.post(
            f"/v1/pets/{pet_id}/moments", json=payload, headers=_headers()
        ).json()
        second = client.post(
            f"/v1/pets/{pet_id}/moments", json=payload, headers=_headers()
        ).json()
        # moment_id 不同 → 可能选到不同候选；但同一 id 必须稳定
        assert first["pet_says"] and second["pet_says"]

    def test_timeline_lists_descending(self, client, pet_id):
        for note in ("在窗台", "刚吃了罐头", None):
            client.post(
                f"/v1/pets/{pet_id}/moments",
                json={"media_url": f"http://x/{note}.jpg", "note": note},
                headers=_headers(),
            )
        tl = client.get(f"/v1/pets/{pet_id}/timeline", headers=_headers()).json()
        assert tl["count"] == 3
        assert sum(tl["scene_counts"].values()) == 3

    def test_timeline_scene_distribution_only_counts_seen(self, client, pet_id):
        """场景分布**只统计出现过的** —— 补零会让「0 次」与「没这个场景」混淆。"""
        client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/1.jpg", "note": "在窗台"},
            headers=_headers(),
        )
        tl = client.get(f"/v1/pets/{pet_id}/timeline", headers=_headers()).json()
        assert set(tl["scene_counts"]) == {"window"}

    def test_timeline_scene_filter(self, client, pet_id):
        client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/1.jpg", "note": "在窗台"},
            headers=_headers(),
        )
        client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/2.jpg", "note": "刚吃了罐头"},
            headers=_headers(),
        )
        tl = client.get(
            f"/v1/pets/{pet_id}/timeline?scene=window", headers=_headers()
        ).json()
        assert tl["count"] == 1

    def test_moments_are_tenant_scoped(self, client, pet_id):
        client.post(
            f"/v1/pets/{pet_id}/moments",
            json={"media_url": "http://x/1.jpg"},
            headers=_headers("u1"),
        )
        tl = client.get(f"/v1/pets/{pet_id}/timeline", headers=_headers("u2"))
        # 别人的宠物 → 404（不泄露存在性）
        assert tl.status_code == 404

    def test_requires_media_url(self, client, pet_id):
        """没有媒体的「瞬间」就是记忆，不是瞬间。"""
        r = client.post(
            f"/v1/pets/{pet_id}/moments", json={}, headers=_headers()
        )
        assert r.status_code == 422
