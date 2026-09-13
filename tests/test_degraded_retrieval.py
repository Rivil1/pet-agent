"""向量索引不可用时的降级行为。

## 这组测试防的是一种「静默的错误答案」

索引不可用时返回 `[]` 在**类型上完全合法**，在**行为上完全错误**：

| 情况 | 用户看到 | 应当看到 |
|---|---|---|
| 确实没有记忆 | 「我还不知道它这件事」 | 同左 —— 正确 |
| 索引挂了 | 「没有相关记录」（**假的确定**） | 「检索暂时不可用」 |

第二种情况下，系统对一个它压根没查过的问题作了答 ——
而用户无法分辨，因为两种回答长得一模一样。

所以这里断言三件事：
1. 检索**不抛异常**（纯 MySQL 部署是合法配置，不该 500）
2. 降级**被记录**（trace 里有 degraded 标记）
3. 响应顶层的 `degraded` **反映节点级降级**（不能只反映守卫）

第 3 条是真实缺陷的回归：初版 `degraded` 只看 guard，
于是 trace 标了降级而顶层说「一切正常」。
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.auth import issue_token
from app.interpreter import PriorTable
from app.llm import HashEmbedder, MockLLM
from app.memory.retrieval import RetrievalResult, retrieve_with_status
from app.profile import VisualObservation
from app.store import InMemoryStore
from app.store.vectors import UnavailableVectorIndex, VectorIndexUnavailable

SECRET = hashlib.sha256(b"pet-agent-degraded-retrieval").hexdigest()
PRIOR_PATH = "data/priors/catmeows_stats.json"


class _StubVision:
    def analyze(self, image_url: str) -> VisualObservation:
        return VisualObservation(image_url=image_url)


class _BrokenSearchStore:
    """把 `search_memories` 换成「索引不可用」的存储包装。

    ## 为什么用包装而不是往 InMemoryStore 里塞一个坏索引

    `InMemoryStore` 自己做暴力余弦检索，**没有可替换的索引对象** ——
    它是内存实现，不是「内存存储 + 内存索引」的组合。
    硬塞一个 `_index` 属性进去只是在测试里造假象：真实系统里
    触发这条路径的是「MySQL + Milvus 挂了」，而不是那个假属性。

    包装只覆盖一个方法，其余全部委托 —— 于是被测的是
    **`retrieve_with_status` 对 `VectorIndexUnavailable` 的处理**，
    而不是某种测试专用的内部状态。
    """

    def __init__(self, inner: Any, reason: str) -> None:
        self._inner = inner
        self._reason = reason

    def search_memories(self, **_kw: Any) -> Any:
        raise VectorIndexUnavailable(self._reason)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


# =============================================================================
# 检索层
# =============================================================================


class TestRetrieveWithStatus:
    def test_available_index_returns_items_not_degraded(self, prior):
        """对照组：索引正常但没记忆 → **不是**降级。

        没有这一条，下面「降级」的断言可能是永真
        （比如实现无脑返回 degraded=True 也能通过）。
        """
        store = InMemoryStore()
        result = retrieve_with_status(
            store=store,
            embedder=HashEmbedder(),
            user_id="user-1",
            pet_id="pet-1",
            query="它怕吸尘器",
        )
        assert isinstance(result, RetrievalResult)
        assert result.items == []
        assert not result.degraded, "索引正常但没有记忆 → 不是降级"

    def test_unavailable_index_degrades_instead_of_raising(self):
        """**核心断言**：不可用时降级，不抛。

        纯 MySQL 部署（没配 Milvus）是合法配置。
        在检索路径上抛异常会让每一次对话都 500 ——
        这与「只有向量检索这一个能力缺失」是不相称的。
        """
        store = _BrokenSearchStore(InMemoryStore(), "Milvus 不可达（测试）")

        result = retrieve_with_status(
            store=store,
            embedder=HashEmbedder(),
            user_id="user-1",
            pet_id="pet-1",
            query="它怕吸尘器",
        )
        assert result.items == []
        assert result.degraded, "索引不可用必须标记为降级"
        assert "Milvus" in (result.degraded_reason or ""), (
            "降级原因要带出原话，否则运维不知道到底哪里坏了"
        )

    def test_retrieve_still_returns_plain_list(self):
        """旧入口保持兼容：既有调用方与测试不受影响。"""
        from app.memory.retrieval import retrieve

        items = retrieve(
            store=InMemoryStore(),
            embedder=HashEmbedder(),
            user_id="user-1",
            pet_id="pet-1",
            query="x",
        )
        assert isinstance(items, list)

    def test_unavailable_index_raises_on_direct_search(self):
        """**直接**调 `search_memories` 仍应抛 —— 降级是上层策略。

        在存储层把异常吞成 `[]`，就等于把「查不到」与「没有」
        永久地混为一谈，上层再也没有机会区分它们。
        """
        index = UnavailableVectorIndex("测试用不可用索引")
        with pytest.raises(VectorIndexUnavailable):
            index.search(user_id="u", pet_id="p", vector=[0.0] * 1024)


# =============================================================================
# 端到端：响应顶层的 degraded 必须反映节点降级
# =============================================================================


def _app(prior: PriorTable, store: Any) -> TestClient:
    return TestClient(
        create_app(
            store=store,
            embedder=HashEmbedder(),
            llm=MockLLM(default="喵。"),
            prior=prior,
            feature_extractor=lambda url: None,
            vision=_StubVision(),
            auth_secret=SECRET,
        )
    )


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token('u1', secret=SECRET)}"}


class TestResponseDegradedFlag:
    def test_baseline_is_not_degraded(self, prior):
        """对照组必须干净 —— 否则下面的断言证明不了什么。"""
        client = _app(prior, InMemoryStore())
        headers = _headers()
        pet_id = client.post("/v1/pets", json={"name": "团团"}, headers=headers).json()["pet_id"]

        body = client.post(
            f"/v1/chat?pet_id={pet_id}", json={"text": "你好"}, headers=headers
        ).json()
        assert body["degraded"] is False

    def test_chat_succeeds_and_reports_degraded(self, prior):
        """**真实缺陷的回归。**

        初版：trace 里 `memory_retriever` 标了 `degraded=true`，
        而响应顶层 `degraded` 为 `false`（它只看 guard）。
        前端据此显示「一切正常」，而用户看到的是一句没有依据的
        「没有相关记录」。
        """
        client = _app(prior, _BrokenSearchStore(InMemoryStore(), "Milvus 不可用（测试替身）"))
        headers = _headers()
        pet_id = client.post("/v1/pets", json={"name": "团团"}, headers=headers).json()["pet_id"]

        resp = client.post(
            f"/v1/chat?pet_id={pet_id}", json={"text": "你好"}, headers=headers
        )
        assert resp.status_code == 200, f"不应 500：{resp.text[:300]}"

        body = resp.json()
        retriever = [t for t in body["trace"] if t["node"] == "memory_retriever"]
        assert retriever, (
            f"缺 memory_retriever 节点（trace: {[t['node'] for t in body['trace']]}）。\n"
            f"意图是 {body['intent']} —— 猜疑意图会走 clarify_ask 分支，不经过检索。\n"
            f"这不是实现错了，而是测试用例选错了输入：本测试要的是「检索路径上的降级」。"
        )
        assert retriever[0]["degraded"] is True, "节点应标降级"

        assert body["degraded"] is True, (
            "顶层 degraded 必须反映节点级降级 —— "
            "否则前端显示一切正常，而实际是检索没查成"
        )
        assert body["degraded_notice"], "降级必须带可读的原因"
        assert "memory_retriever" in body["degraded_notice"]
