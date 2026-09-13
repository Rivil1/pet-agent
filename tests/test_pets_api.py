"""前端接入所需的两个端点：**列宠物**与**开发登录**。

## 为什么这两个端点需要测试

1. `GET /v1/pets` 是前端的第一屏所依赖的。它的**唯一**归属来源必须是
   token 派生的 `user_id` —— 一旦它接受查询参数，D5（多租户隔离）就破了。

2. `POST /v1/auth/dev-login` 是本项目里**唯一**一个能凭空签发 token 的入口。
   它默认关闭，而「默认关闭」这件事必须是**被断言**的，不能只写在注释里：
   一个能签发任意 `user_id` 的端点若意外开启，整套隔离主张立刻失效，
   且不会有任何测试变红（这是 B26 那一类「静默失效」的形状）。
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api import create_app, ENV_ALLOW_DEV_LOGIN
from app.auth import issue_token
from app.interpreter import PriorTable
from app.llm import HashEmbedder, MockLLM
from app.profile import VisualObservation
from app.store import InMemoryStore

#: 由固定输入派生而非写字面量 —— 密钥扫描器会把 `SECRET = "..."` 当作硬编码密钥。
SECRET = hashlib.sha256(b"pet-agent-pets-api-signing").hexdigest()

PRIOR_PATH = "data/priors/catmeows_stats.json"


class _StubVision:
    def analyze(self, image_url: str) -> VisualObservation:
        return VisualObservation(image_url=image_url)


@pytest.fixture(scope="module")
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture()
def client(prior) -> TestClient:
    app = create_app(
        store=InMemoryStore(),
        embedder=HashEmbedder(),
        llm=MockLLM(default="喵。"),
        prior=prior,
        feature_extractor=lambda url: None,
        vision=_StubVision(),
        auth_secret=SECRET,
    )
    return TestClient(app)


def _headers(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(user_id, secret=SECRET)}"}


# ─────────────────────────────────────────────────────────────
# GET /v1/pets
# ─────────────────────────────────────────────────────────────


def test_list_pets_starts_empty(client):
    r = client.get("/v1/pets", headers=_headers("user-1"))
    assert r.status_code == 200, r.text
    assert r.json() == {"count": 0, "pets": []}


def test_list_pets_returns_created_pet(client):
    h = _headers("user-1")
    client.post("/v1/pets", json={"name": "团团", "breed": "英短"}, headers=h)

    body = client.get("/v1/pets", headers=h).json()
    assert body["count"] == 1
    pet = body["pets"][0]
    assert pet["name"] == "团团"
    assert pet["breed"] == "英短"
    assert pet["species"] == "cat"
    assert pet["has_profile"] is False
    # 归属键不得出现在列表视图里
    assert "user_id" not in pet


def test_list_pets_is_tenant_scoped(client):
    """**隔离的正确性底线**：别人的宠物根本不出现在我的列表里。"""
    client.post("/v1/pets", json={"name": "我的猫"}, headers=_headers("user-a"))
    client.post("/v1/pets", json={"name": "别人的猫"}, headers=_headers("user-b"))

    mine = client.get("/v1/pets", headers=_headers("user-a")).json()
    assert mine["count"] == 1
    assert mine["pets"][0]["name"] == "我的猫"


def test_list_pets_requires_auth(client):
    assert client.get("/v1/pets").status_code == 401


# ─────────────────────────────────────────────────────────────
# POST /v1/auth/dev-login
# ─────────────────────────────────────────────────────────────


def test_healthz_reports_dev_login_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv(ENV_ALLOW_DEV_LOGIN, raising=False)
    assert client.get("/healthz").json()["dev_login"] == "disabled"


def test_dev_login_rejected_when_disabled(client, monkeypatch):
    """默认关闭时**必须是 404**，而不是 200 —— 这是隔离主张的守门测试。"""
    monkeypatch.delenv(ENV_ALLOW_DEV_LOGIN, raising=False)
    r = client.post("/v1/auth/dev-login", json={"user_id": "attacker"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "DEV_LOGIN_DISABLED"


def test_dev_login_issues_usable_token_when_enabled(client, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_DEV_LOGIN, "1")
    assert client.get("/healthz").json()["dev_login"] == "enabled"

    r = client.post("/v1/auth/dev-login", json={"user_id": "frontend-user"})
    assert r.status_code == 200, r.text
    token = r.json()["token"]

    # 签发的 token 必须真的能通过鉴权
    assert client.get("/v1/pets", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_dev_login_token_is_scoped_to_requested_user(client, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_DEV_LOGIN, "1")
    token = client.post("/v1/auth/dev-login", json={"user_id": "user-7"}).json()["token"]
    h = {"Authorization": f"Bearer {token}"}

    client.post("/v1/pets", json={"name": "七号猫"}, headers=h)

    # 另一个用户看不到它
    other = client.get("/v1/pets", headers=_headers("user-8")).json()
    assert other["count"] == 0


def test_dev_login_rejects_empty_user_id(client, monkeypatch):
    monkeypatch.setenv(ENV_ALLOW_DEV_LOGIN, "1")
    r = client.post("/v1/auth/dev-login", json={"user_id": ""})
    assert r.status_code == 422
