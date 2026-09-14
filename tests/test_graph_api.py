"""编排与 API 测试。

**重点覆盖降级路径。**

这个文件的存在有一个具体原因：pyright 在 `app/graph/nodes.py` 里发现了
`Severity.DEGRADED`（应为 `ErrorSeverity.DEGRADED`）——
也就是说 **全部降级路径一旦真被触发就会 AttributeError 崩溃**，
而当时没有一个测试走过那些路径。

「错误处理代码从来没被执行过」是很常见的盲区。所以这里显式地为每条降级路径
都写了用例：LLM 失败、特征提取失败、照片全失败、守卫拦截、路由不确定。

同时覆盖 `docs/ARCHITECTURE.md` §2.6 的隔离强制点与 §4.2 的鉴权。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from app.api import create_app
from app.audio.features import TARGET_SR, extract_features, synthesize_meow
from app.auth import InvalidToken, issue_token, verify_token
from app.graph import (
    StateError,
    build_graph,
    initial_state,
    parse_time_range,
    tenant_of,
)
from app.interpreter import PriorTable
from app.llm import HashEmbedder, MockLLM
from app.profile import VisualObservation
from app.schemas import (
    AcousticFeatures,
    ContextLabel,
    PetProfile,
    RawInput,
    Species,
    VisualProfile,
)
from app.store import InMemoryStore
from fastapi.testclient import TestClient

PRIOR_PATH = "data/priors/catmeows_stats.json"
SECRET = "test-secret-please-rotate-in-production"
NOW = datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────
# 夹具与替身
# ─────────────────────────────────────────────────────────────


class _FailingLLM:
    """总是抛异常的 LLM —— 用于验证降级路径。"""

    def complete(self, *, system: str, user: str, temperature: float = 0.0) -> str:
        raise RuntimeError("上游模型不可用")


class _FailingAudio:
    def __call__(self, url: str) -> AcousticFeatures:
        raise ValueError("无法下载或解码音频")


class _StubVision:
    """按预设返回观察结果，`bad` 集合中的 URL 视为失败。"""

    def __init__(
        self, observations: dict[str, VisualObservation], bad: set[str] | None = None
    ):
        self._observations = observations
        self._bad = bad or set()

    def analyze(self, image_url: str) -> VisualObservation:
        if image_url in self._bad:
            raise ValueError("图像分析失败")
        return self._observations[image_url]


def _real_features(url: str) -> AcousticFeatures:
    """真实走一遍声学特征提取（合成音频，离线）。"""
    return extract_features(
        synthesize_meow(duration=4.0, f0_start=420, f0_end=780), TARGET_SR
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def prior() -> PriorTable:
    return PriorTable.load(PRIOR_PATH)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def pet(store: InMemoryStore) -> PetProfile:
    p = PetProfile(
        pet_id="pet-1",
        user_id="user-1",
        name="团团",
        species=Species.CAT,
        visual=VisualProfile(fur_color="橘白"),
        must_keep_features=["橘白短毛", "圆脸"],
    )
    store.save_pet(p)
    return p


@pytest.fixture
def client(store, embedder, prior) -> TestClient:
    app = create_app(
        store=store,
        embedder=embedder,
        llm=MockLLM(default="团团看起来挺好的，我记着它喜欢趴窗台。"),
        prior=prior,
        feature_extractor=_real_features,
        vision=_StubVision({}),
        auth_secret=SECRET,
    )
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token('user-1', secret=SECRET)}"}


# ═══════════════════════════════════════════════════════════════
# 鉴权（ARCHITECTURE.md §4.2）
# ═══════════════════════════════════════════════════════════════


class TestToken:
    def test_roundtrip(self):
        token = issue_token("user-1", secret=SECRET)
        assert verify_token(token, secret=SECRET) == "user-1"

    def test_expiry(self):
        token = issue_token("user-1", secret=SECRET, ttl_seconds=10, now=1000)
        with pytest.raises(InvalidToken, match="过期"):
            verify_token(token, secret=SECRET, now=2000)

    def test_wrong_secret_rejected(self):
        token = issue_token("user-1", secret=SECRET)
        with pytest.raises(InvalidToken, match="签名"):
            verify_token(token, secret="another-secret")

    def test_tampered_user_id_rejected(self):
        """**这是隔离的地基**：篡改 user_id 必须导致签名失败。"""
        import base64

        other = issue_token("victim", secret=SECRET)
        payload, _, signature = other.rpartition(".")
        forged_payload = (
            base64.urlsafe_b64encode(b"attacker.9999999999").decode().rstrip("=")
        )
        with pytest.raises(InvalidToken):
            verify_token(f"{forged_payload}.{signature}", secret=SECRET)

    @pytest.mark.parametrize("bad", ["", "no-dot", "a.b.c", ".", "abc."])
    def test_malformed_rejected(self, bad):
        with pytest.raises(InvalidToken):
            verify_token(bad, secret=SECRET)

    def test_empty_user_id_rejected(self):
        with pytest.raises(ValueError, match="不得为空"):
            issue_token("  ", secret=SECRET)


class TestApiAuth:
    def test_missing_header_is_401(self, client):
        assert client.get("/v1/pets/pet-1/profile").status_code == 401

    def test_bad_scheme_is_401(self, client):
        r = client.get("/v1/pets/pet-1/profile", headers={"Authorization": "Basic xyz"})
        assert r.status_code == 401

    def test_invalid_token_is_401(self, client):
        r = client.get(
            "/v1/pets/pet-1/profile", headers={"Authorization": "Bearer garbage"}
        )
        assert r.status_code == 401

    def test_healthz_needs_no_auth(self, client):
        assert client.get("/healthz").status_code == 200


class TestApiTenantIsolation:
    def test_other_users_pet_is_404_not_403(self, client, pet):
        """归属不符返回 404 —— 区分 403 会泄露资源存在性。"""
        other = {"Authorization": f"Bearer {issue_token('user-2', secret=SECRET)}"}
        r = client.get("/v1/pets/pet-1/profile", headers=other)
        assert r.status_code == 404

    def test_missing_pet_is_same_404(self, client, auth_headers):
        r = client.get("/v1/pets/nope/profile", headers=auth_headers)
        assert r.status_code == 404

    def test_chat_on_other_users_pet_rejected(self, client, pet):
        other = {"Authorization": f"Bearer {issue_token('user-2', secret=SECRET)}"}
        r = client.post(
            "/v1/chat", params={"pet_id": "pet-1"}, json={"text": "你好"}, headers=other
        )
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════
# 状态访问器
# ═══════════════════════════════════════════════════════════════


class TestStateAccessors:
    def test_tenant_of_ok(self):
        s = initial_state(user_id="u", pet_id="p", raw_input=RawInput(text="x"))
        assert tenant_of(s) == ("u", "p")

    def test_tenant_of_missing_raises_state_error(self):
        """绕过 ``initial_state`` 构造的状态应给**说明白了的错误**，而非 KeyError。"""
        with pytest.raises(StateError, match="租户标识"):
            tenant_of({})

    def test_tenant_of_empty_string_rejected(self):
        """空租户标识比缺失更危险 —— 它会静默匹配不到任何数据。"""
        s = initial_state(user_id="u", pet_id="p", raw_input=RawInput(text="x"))
        s["user_id"] = ""
        with pytest.raises(StateError):
            tenant_of(s)


class TestTimeNormalization:
    """docs/10-self-review.md U2/F6 —— 冲突判定依赖它。"""

    def test_relative_week(self):
        r = parse_time_range("这几天它一直叫", now=NOW)
        assert r is not None and r.start is not None and r.end is not None
        assert (NOW - r.end).total_seconds() < 86400

    def test_last_month_is_disjoint_from_this_week(self):
        """「上个月」与「这周」必须解析成**不重叠**的区间，否则冲突判定会把演变当矛盾。"""
        last = parse_time_range("上个月它喜欢逗猫棒", now=NOW)
        this = parse_time_range("这周它不怎么玩", now=NOW)
        assert last is not None and this is not None
        assert last.end is not None and this.start is not None
        assert last.end < this.start

    def test_iso_date(self):
        r = parse_time_range("2026-09-01 那天", now=NOW)
        assert r is not None and r.start is not None
        assert r.start.date().isoformat() == "2026-09-01"

    @pytest.mark.parametrize("text", ["它很可爱", "", None])
    def test_unrecognized_returns_none(self, text):
        """**不猜**：识别不了就是 None（无界 = 当前有效）。"""
        assert parse_time_range(text, now=NOW) is None


# ═══════════════════════════════════════════════════════════════
# 编排：正常路径
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def graph(store, embedder, prior):
    return build_graph(
        store=store,
        embedder=embedder,
        llm=MockLLM(default="团团挺想你的，它今天在窗台晒太阳。"),
        prior=prior,
        feature_extractor=_real_features,
        vision=_StubVision({}),
        # 案例推理的样本库：从同一个 store 读（多租户过滤在 store 内）
        records_lookup=lambda state: store.list_meow_records(
            user_id=tenant_of(state)[0], pet_id=tenant_of(state)[1]
        ),
    )


def _invoke(graph, *, text: str | None = None, **raw_kw):
    raw = RawInput(text=text, **raw_kw) if text else RawInput(**raw_kw)
    state = initial_state(user_id="user-1", pet_id="pet-1", raw_input=raw)
    if text:
        state["transcribed_text"] = text
    return graph.invoke(state)


class TestHappyPaths:
    def test_all_routes_map_to_real_nodes(self, graph):
        """回归 B14：``ROUTE_TARGETS`` 的每个目标都必须是真节点。

        接线层最容易漏：加了边却忘了加节点，会让**整张图无法编译**。
        这条断言的成本很低，却能拦住整个编排不可用的情况。
        """
        expected = {
            "load_context",
            "understand_input",
            "memory_retriever",
            "companion_agent",
            "behavior_interpreter",
            "render_interpretation",
            "memory_extractor",
            "memory_writer",
            "record_acknowledge",
            "profile_analyzer",
            "clarify_ask",
            "response_guard",
        }
        assert expected <= set(graph.nodes)

    def test_chat(self, graph, pet):
        r = _invoke(graph, text="团团今天怎么样")
        assert r["intent"].value == "chat"
        assert "窗台" in r["final_response"]
        assert r["guard_result"] is not None and r["guard_result"].passed

    def test_memory_query(self, graph, pet):
        r = _invoke(graph, text="还记得它怕什么吗")
        assert r["intent"].value == "memory_query"

    def test_record_writes_and_reports(self, graph, pet, store):
        """**记录意图的响应必须报告写入结果**（DESIGN.md §5.3）。"""
        r = _invoke(graph, text="记住它很怕吸尘器的声音")
        assert r["intent"].value == "record_event"
        assert r["written_memory_ids"]
        assert "记住啦" in r["final_response"]
        assert len(store.list_memories(user_id="user-1", pet_id="pet-1")) == 1

    def test_interpret_with_real_features(self, graph, pet):
        r = _invoke(graph, audio_url="http://x/meow.wav", audio_kind="cat_meow")
        assert r["intent"].value == "translate_behavior"
        interp = r["interpretation"]
        assert interp is not None
        # 测试环境的先验是**占位数据**，因此走 measured_only：
        # 有测量、无可用推断模型 → 禁止输出后验概率（D31 / fail-closed）。
        # 这并非降级意外，而是刻意设计：占位先验永远不产生后验。
        assert interp.evidence_mode.value == "measured_only"
        assert all(c.posterior is None for c in interp.candidates)
        assert interp.evidence, "证据链不得为空"
        # 断言**证据确实到达了用户**，而不是某个固定标签。
        # 渲染文案是猫的口吻（「我这么想是因为…」），标签会变；
        # 而「证据有没有出现在回复里」才是这个测试要保的东西 ——
        # 它一旦断了，用户就只能看到一个没有依据的猜测。
        assert "我这么想是因为" in r["final_response"], r["final_response"][:200]
        claimed = interp.evidence[0].statement[:12]
        assert claimed in r["final_response"], (
            f"证据原文未出现在回复中：{claimed!r}"
        )
        # 归因值（+0.45 这类）**不该**出现在给用户看的文本里
        assert "0.4" not in r["final_response"].replace("0.45", ""), (
            "log_odds 贡献值是分析中间量，不该渲染给用户"
        )

    def test_case_based_when_records_suffice(self, graph, pet, store):
        """记录够多时应当走到 case_based，并给出**计数**而非概率。"""
        from app.schemas import BehaviorAction, MeowRecord

        # 与图节点实际用到的特征同源，保证相似度高于阈值
        feats = _real_features("http://x/seed.wav")
        for i in range(3):
            store.insert_meow_record(
                MeowRecord(
                    user_id="user-1",
                    pet_id=pet.pet_id,
                    context=ContextLabel.DOOR_ATTENTION,
                    features=feats,
                    actions=[BehaviorAction.SCRATCH_DOOR],
                    resolution="开门它就出去了",
                    recorded_at=datetime(2026, 3, 1 + i, tzinfo=timezone.utc),
                )
            )
        r = _invoke(graph, audio_url="http://x/meow.wav", audio_kind="cat_meow")
        interp = r["interpretation"]
        assert interp is not None
        assert interp.evidence_mode.value == "case_based"
        assert interp.case_total >= 3
        assert interp.similar_cases
        # 计数模式下**不得**出现后验概率（契约也会校验）
        assert all(c.posterior is None for c in interp.candidates)
        assert any(c.matched_count >= 3 for c in interp.candidates)
        # 结果（什么让它停了）必须出现在案例里 —— 它是主人经验的核心
        assert any(c.resolution for c in interp.similar_cases)

    def test_ambiguous_goes_to_clarify_not_chat(self, graph, pet):
        """**AMBIGUOUS 是独立分支**，不回退到闲聊（DESIGN.md §5.1）。"""
        r = _invoke(graph, text="嗯")
        assert r["intent"].value == "ambiguous"
        assert any(t.node == "clarify_ask" for t in r["node_trace"])
        # 澄清文案是**猫的口吻**（身份不分轨），断言行为而不是某句固定措辞 ——
        # 措辞会随文案调整而变，而「有没有走澄清分支」才是这个测试要保的东西。
        assert "想问" in r["final_response"], (
            f"澄清回复应给出可选方向，实际：{r['final_response']}"
        )
        assert "它" not in r["final_response"].replace("问它", ""), (
            "说话的是它本人，不应用第三人称指代自己"
        )

    def test_profile_is_loaded_before_downstream_nodes(self, graph, pet):
        """回归：档案必须在入口就被加载。

        早期实现没有这一步，导致 ``companion_agent`` 永远看到 ``pet=None``，
        所有回复都退化成同一句「先上传照片」—— 而且**不报错**。
        """
        r = _invoke(graph, text="团团今天怎么样")
        assert r["pet_profile"] is not None
        nodes = [t.node for t in r["node_trace"]]
        assert nodes[0] == "load_context", f"档案应在最早一步加载，实际顺序：{nodes}"

    def test_trace_present_for_every_route(self, graph, pet):
        r = _invoke(graph, text="团团今天怎么样")
        nodes = [t.node for t in r["node_trace"]]
        assert "understand_input" in nodes
        assert "response_guard" in nodes, "所有面向用户的文本都必须过守卫（W3）"


# ═══════════════════════════════════════════════════════════════
# 编排：降级路径（B13 的教训）
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def degraded_graph(store, embedder, prior):
    return build_graph(
        store=store,
        embedder=embedder,
        llm=_FailingLLM(),
        prior=prior,
        feature_extractor=_FailingAudio(),
        vision=_StubVision({}, bad={"http://x/bad.jpg"}),
    )


class TestDegradationPaths:
    """**这些路径曾经从未被执行过**，导致其中的枚举用错都没人发现。"""

    def test_llm_failure_degrades_not_crashes(self, degraded_graph, pet):
        r = _invoke(degraded_graph, text="团团今天怎么样")
        assert r["final_response"], "降级后仍必须给出可读回复"
        errors = r.get("errors", [])
        assert errors, "降级必须留痕"
        assert errors[0].node == "companion_agent"
        assert errors[0].severity.value == "degraded"
        assert errors[0].is_user_visible

    def test_audio_failure_degrades_without_crash(self, degraded_graph, pet):
        r = _invoke(
            degraded_graph, audio_url="http://x/meow.wav", audio_kind="cat_meow"
        )
        errors = r.get("errors", [])
        assert errors and errors[0].node == "behavior_interpreter"
        # 降级必须可见：不能悄悄返回一个看似正常的解释
        assert errors[0].severity.value == "degraded"

    def test_missing_audio_yields_text_only(self, graph, pet):
        """无音频 → ``text_only``，**禁止输出数值置信度**。"""
        r = _invoke(graph, text="它为什么一直叫")
        interp = r["interpretation"]
        assert interp is not None
        assert interp.evidence_mode.value == "text_only"
        assert all(c.posterior is None for c in interp.candidates)
        assert "不给出数值置信度" in interp.limitations

    def test_guard_blocks_untraceable_claim(self, store, embedder, prior, pet):
        """检索为空时不得出现「有过记录」类断言 —— 幻觉最直接的来源。"""
        g = build_graph(
            store=store,
            embedder=embedder,
            llm=MockLLM(default="你之前说过它怕吸尘器，我记得它的习惯。"),
            prior=prior,
            feature_extractor=_real_features,
            vision=_StubVision({}),
        )
        r = _invoke(g, text="团团今天怎么样")
        guard = r["guard_result"]
        assert guard is not None and not guard.passed
        assert any(v.type.value == "untraceable_claim" for v in guard.violations)
        assert guard.degrade_to_conservative and guard.degraded_notice

    def test_guard_blocks_forbidden_phrase(self, store, embedder, prior, pet):
        g = build_graph(
            store=store,
            embedder=embedder,
            llm=MockLLM(default="别担心，它很健康，没问题。"),
            prior=prior,
            feature_extractor=_real_features,
            vision=_StubVision({}),
        )
        r = _invoke(g, text="团团今天怎么样")
        guard = r["guard_result"]
        assert guard is not None and not guard.passed
        assert any(v.type.value == "health_boundary" for v in guard.violations)

    def test_echo_only_exempts_user_wording(self, graph, pet):
        """回显用户原话不应因用户用词而降级。

        否则「记住，它没问题」会被改成「我这边暂时没有相关记录」，用户无法理解。
        """
        r = _invoke(graph, text="记住，它一向没问题")
        assert "记住啦" in r["final_response"], (
            "guard_mode=echo_only 时应跳过禁用词检查；"
            f"实际响应：{r['final_response']!r}"
        )

    def test_profile_all_photos_fail_is_fatal(self, store, embedder, prior, pet):
        g = build_graph(
            store=store,
            embedder=embedder,
            llm=MockLLM(),
            prior=prior,
            feature_extractor=_real_features,
            vision=_StubVision({}, bad={"http://x/1.jpg", "http://x/2.jpg"}),
        )
        r = _invoke(
            g, text="这是它的照片", image_urls=["http://x/1.jpg", "http://x/2.jpg"]
        )
        errors = r.get("errors", [])
        assert errors and errors[0].severity.value == "fatal"
        assert "失败" in r["final_response"]


# ═══════════════════════════════════════════════════════════════
# 档案：多图取交集
# ═══════════════════════════════════════════════════════════════


def _obs(url: str, **kw) -> VisualObservation:
    return VisualObservation(image_url=url, **kw)


class TestProfileIntersection:
    """`DESIGN.md` §2.3 —— 多图取交集区分稳定特征与偶然特征。"""

    def test_stable_vs_unstable(self):
        from app.profile import intersect_observations

        obs = [
            _obs(
                "a",
                fur_color="橘白",
                face_shape="圆脸",
                distinctive_features=("左耳缺口",),
            ),
            _obs(
                "b",
                fur_color="橘白",
                face_shape="圆脸",
                distinctive_features=("左耳缺口",),
            ),
            _obs(
                "c",
                fur_color="橘白",
                face_shape="圆脸",
                distinctive_features=("尾巴有环",),
            ),
        ]
        c = intersect_observations(obs, min_support=2)
        assert "橘白" in c.stable and "圆脸" in c.stable
        assert "左耳缺口" in c.stable
        assert "尾巴有环" in c.unstable, "只出现一次的特征不得进 must_keep_features"

    def test_failed_photos_do_not_count_as_support(self):
        """失败的照片**既不算支持也不算反对** —— 否则会系统性拉低所有特征的支持度。"""
        from app.profile import intersect_observations

        obs = [
            _obs("a", fur_color="橘白"),
            _obs("b", fur_color="橘白"),
            _obs("c", ok=False, error="decode failed"),
        ]
        c = intersect_observations(obs, min_support=2)
        assert "橘白" in c.stable, "两张成功照片的一致特征应为稳定"

    def test_all_failed_raises(self):
        from app.profile import build_draft

        with pytest.raises(ValueError, match="全部"):
            build_draft([_obs("a", ok=False), _obs("b", ok=False)])

    def test_coverage_note_reported(self):
        from app.profile import build_draft

        draft = build_draft(
            [
                _obs("a", fur_color="橘白"),
                _obs("b", fur_color="橘白"),
                _obs("c", ok=False),
            ]
        )
        assert draft.analyzed_photo_count == 2
        assert draft.failed_photo_urls == ["c"]
        assert draft.coverage_note and "2/3" in draft.coverage_note

    def test_majority_vote_is_deterministic(self):
        """并列时取字典序最小者 —— 同一批照片必须得到同一份档案。"""
        from app.profile import build_draft

        obs = [_obs("a", fur_color="橘白"), _obs("b", fur_color="白色")]
        first = build_draft(obs).visual.fur_color
        assert all(build_draft(obs).visual.fur_color == first for _ in range(5))

    def test_identify_from_photos_isolates_failures(self):
        from app.profile import identify_from_photos

        analyzer = _StubVision(
            {
                "a": _obs("a", fur_color="橘白", face_shape="圆脸"),
                "c": _obs("c", fur_color="橘白", face_shape="圆脸"),
            },
            bad={"b"},
        )
        draft = identify_from_photos(["a", "b", "c"], analyzer)
        assert draft.analyzed_photo_count == 2
        assert "橘白" in draft.must_keep_features


# ═══════════════════════════════════════════════════════════════
# API 端到端
# ═══════════════════════════════════════════════════════════════


class TestApiEndpoints:
    def test_create_and_read_pet(self, client, auth_headers):
        created = client.post("/v1/pets", json={"name": "团团"}, headers=auth_headers)
        assert created.status_code == 201
        pet_id = created.json()["pet_id"]

        got = client.get(f"/v1/pets/{pet_id}/profile", headers=auth_headers)
        assert got.status_code == 200
        assert got.json()["name"] == "团团"

    def test_chat_endpoint(self, client, auth_headers, pet):
        r = client.post(
            "/v1/chat",
            params={"pet_id": "pet-1"},
            json={"text": "团团今天怎么样"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        body = r.json()
        assert body["final_response"]
        assert body["intent"] == "chat"
        assert body["trace"], "应返回 node_trace 以便观察执行路径"

    def test_audio_without_kind_is_400(self, client, auth_headers, pet):
        """有音频但没声明类型必须报错而非猜 —— 两条链路完全不同。"""
        r = client.post(
            "/v1/chat",
            params={"pet_id": "pet-1"},
            json={"audio_url": "http://x/a.wav"},
            headers=auth_headers,
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_FAILED"

    def test_interpret_endpoint_returns_evidence(self, client, auth_headers, pet):
        r = client.post(
            "/v1/interpret",
            params={"pet_id": "pet-1"},
            json={"audio_url": "http://x/meow.wav", "scene_description": "它对着门叫"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        interp = r.json()["interpretation"]
        assert interp is not None
        assert interp["evidence"]
        assert any(e["kind"] == "prior" for e in interp["evidence"]), (
            "场景应产生先验证据"
        )

    def test_record_then_list_memories(self, client, auth_headers, pet):
        r = client.post(
            "/v1/chat",
            params={"pet_id": "pet-1"},
            json={"text": "记住它很怕吸尘器的声音"},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["written_memory_ids"]

        listed = client.get(
            "/v1/memories", params={"pet_id": "pet-1"}, headers=auth_headers
        )
        assert listed.status_code == 200
        assert listed.json()["count"] == 1

    def test_profile_draft_requires_confirmation(self, auth_headers, pet, store):
        """``confirm=False`` 时只返回草案、**不落库**（DESIGN.md §3.3）。"""
        vision = _StubVision(
            {
                "a": _obs("a", fur_color="橘白", face_shape="圆脸"),
                "b": _obs("b", fur_color="橘白", face_shape="圆脸"),
            }
        )
        client2 = TestClient(
            create_app(
                store=store,
                embedder=HashEmbedder(),
                llm=MockLLM(),
                prior=PriorTable.load(PRIOR_PATH),
                feature_extractor=_real_features,
                vision=vision,
                auth_secret=SECRET,
            )
        )
        r = client2.post(
            "/v1/pets/pet-1/profile",
            json={"image_urls": ["a", "b"], "confirm": False},
            headers=auth_headers,
        )
        assert r.status_code == 200
        assert r.json()["confirmed"] is False
        assert "profile" not in r.json()
        assert store.get_pet(user_id="user-1", pet_id="pet-1").must_keep_features == [
            "橘白短毛",
            "圆脸",
        ], "未确认时不得改动档案"

        r2 = client2.post(
            "/v1/pets/pet-1/profile",
            json={"image_urls": ["a", "b"], "confirm": True},
            headers=auth_headers,
        )
        assert r2.status_code == 200
        assert "profile" in r2.json()
