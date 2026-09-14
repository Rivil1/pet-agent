"""HTTP API。

对应 docs/ARCHITECTURE.md §4（API 设计）。

两个实现要点：

1. **依赖全部注入**（``create_app`` 的参数）。测试可传入 mock，
   离线跑通全部端点 —— `DESIGN.md` §6.5 的可复现要求。
2. **``user_id`` 只从 token 派生**，绝不从请求体读（`ARCHITECTURE.md` §4.2 A1）。
   这是多租户隔离的地基：``user_id`` 一旦可伪造，隔离主张就不成立。
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import InvalidToken, bearer_token, verify_token
from app.auth.token import DEFAULT_TTL_SECONDS, issue_token
from app.digest import summarize_day
from app.graph import build_graph, initial_state, tenant_of
from app.habits import detect_habits
from app.habits.answer import render_habit_report
from app.health import HealthWriter, InMemoryHealthStore, RedFlagTable
from app.interpreter import PriorTable
from app.llm import UNTRUSTED_MODES, Embedder, LLMClient, describe_providers
from app.memory import MemoryWriter
from app.observability.langsmith import (
    TracingConfig,
    build_tracer,
    resolve_run_id,
    run_config,
)
from app.profile import VisionAnalyzer
from app.schemas import (
    AudioKind,
    BehaviorAction,
    ContextLabel,
    HealthRecordSource,
    MeowRecord,
    PendingInterpretation,
    PetProfile,
    RawInput,
    SessionMessage,
    Species,
)
from app.store.base import MemoryStore, NotFound
from app.story import compose_story, render_story

#: 红旗规则表路径。规则是**数据**，可由执业兽医维护而不必改代码。
_RED_FLAGS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "health" / "red_flags.yaml"
)

#: 开发登录开关。**默认关闭** —— 详见 `dev_login` 端点的 docstring。
ENV_ALLOW_DEV_LOGIN = "PET_AGENT_ALLOW_DEV_LOGIN"


def _env_flag(name: str) -> bool:
    """读一个布尔环境变量。只有显式真值才算开。"""
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _pet_brief(pet: PetProfile) -> dict[str, Any]:
    """宠物的列表视图。**不含 ``user_id``** —— 客户端不需要，也不该看到归属键。"""
    visual = pet.visual
    return {
        "pet_id": pet.pet_id,
        "name": pet.name,
        "species": pet.species.value
        if hasattr(pet.species, "value")
        else str(pet.species),
        "breed": pet.breed,
        "has_profile": bool(pet.must_keep_features) or bool(visual.fur_color),
        "must_keep_features": list(pet.must_keep_features),
        "fur_color": visual.fur_color,
        "eye_color": visual.eye_color,
        "created_at": pet.created_at.isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# 请求 / 响应模型
# ─────────────────────────────────────────────────────────────


class CreatePetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    breed: str | None = None


class DevLoginRequest(BaseModel):
    """开发/演示用的登录请求。

    ⚠️ **只在 `PET_AGENT_ALLOW_DEV_LOGIN=1` 时可用**（默认关闭）。
    理由：它接受任意 ``user_id`` 并签发合法 token —— 默认开启就等于
    让 D5（多租户隔离）形同虚设：任何人都能伪造任意用户身份。
    这是前端本地联调的便利开关，不是生产鉴权方案。
    """

    user_id: str = Field(min_length=1, max_length=64)


class ProfileRequest(BaseModel):
    image_urls: list[str] = Field(min_length=1, max_length=10)
    confirm: bool = Field(
        default=False,
        description="用户是否已确认草案。false 时只返回草案，不落库（DESIGN.md §2.3）。",
    )


class ChatRequest(BaseModel):
    text: str = ""
    audio_url: str | None = None
    audio_kind: AudioKind | None = None
    image_urls: list[str] = Field(default_factory=list)
    scene_description: str | None = None


class InterpretRequest(BaseModel):
    audio_url: str
    scene_description: str | None = None


class LabelMeowRequest(BaseModel):
    """主人对一次叫声的标注。

    **刻意不接受 `features`** —— 声学特征必须由服务端产生（`MEASURED` 承诺）。
    若允许客户端提交，那个承诺立刻失效：客户端可以发任意数字，
    而案例推理的相似度会建在它们上面。
    """

    interpretation_id: str
    context: ContextLabel
    actions: list[BehaviorAction] = Field(default_factory=list)
    resolution: str | None = Field(
        default=None,
        description="**结果**：后来什么让它停了，如「开门它就出去了」。",
    )


class StoryResponse(BaseModel):
    """「宠物的一天」。"""

    date: str
    title: str
    story_text: str
    beats: list[dict[str, Any]] = Field(default_factory=list)
    excluded_count: int = 0
    health_notice: str | None = None
    disclaimer: str = ""
    digest_notes: list[str] = Field(default_factory=list)
    rejection_rate: float = 0.0


class HealthSignalRequest(BaseModel):
    """一个**结构化**健康信号。"""

    signal: str = Field(description="信号名，必须来自 red_flags.yaml 的词汇表")
    value: float | bool | str
    at: datetime | None = None


class HealthResponse(BaseModel):
    level: str
    coverage: float
    coverage_note: str
    recommendation: str
    disclaimer: str
    must_not_be_read_as: str
    rule_version: str | None = None
    red_flags: list[dict[str, Any]] = Field(default_factory=list)
    signals_missing_count: int = 0
    record_count: int = 0


class TraceItem(BaseModel):
    node: str
    latency_ms: int | None = None
    decision: str | None = None
    degraded: bool = False


class TurnResponse(BaseModel):
    """一次编排的响应。

    ``node_trace`` 是刻意暴露的：它让「系统为什么这么答」可被观察，
    而不只是一个黑箱结果。
    """

    final_response: str
    intent: str
    route_confidence: float
    session_id: str = Field(
        default="",
        description=(
            "本次请求所属会话。**同一会话的多次请求共享它** —— "
            "会话记忆注入按它取回最近若干轮。\n\n"
            "请求未带 `X-Session-Id` 时由服务端生成，并通过同名响应头回传。"
        ),
    )
    trace_id: str = Field(
        default="",
        description=(
            "本次请求的链路标识。**与 LangSmith 的 run id 同源** ——"
            "trace_id 是 UUID 时直接用作 run id，否则由它确定性派生。\n\n"
            "于是「本地看到的 trace」与「LangSmith 里那条」是同一个标识，不需要对照表。"
        ),
    )
    langsmith_run_id: str | None = Field(
        default=None,
        description="LangSmith 的 run id（未启用观测时为 None）。",
    )
    degraded: bool = False
    degraded_notice: str | None = None
    trace: list[TraceItem] = Field(default_factory=list)
    retrieved_count: int = 0
    written_memory_ids: list[str] = Field(default_factory=list)
    interaction_mode: str = Field(
        default="companion",
        description=(
            "本次互动走的模式（`companion` / `analysis`）。\n\n"
            "**它只决定说多少证据，不决定谁在说话** —— 两轨的身份都是那只猫本人。\n"
            "带它是为了让「为什么这次开始讲证据了」可回答。"
        ),
    )
    provider_mode: str = Field(
        default="unknown",
        description=(
            "本次响应由真模型还是占位实现产生（`live` / `mock`）。\n\n"
            "**每个响应都带**：没有它，mock 输出与真实输出在调用方看来一样。"
        ),
    )
    mock_notice: str | None = Field(
        default=None,
        description="mock 模式下的显式提示。**不为 None 时不要把这些输出当成真实推理。**",
    )
    suggested_actions: list[str] = Field(
        default_factory=list,
        description=(
            "多模态模型观察到的动作，**供标注表单预填**。\n\n"
            "它只是候选 —— 主人提交时仍然自己选定。"
            "录完自动填上「抓门」，主人只需确认或改一下。"
        ),
    )
    interpretation_id: str | None = Field(
        default=None,
        description=(
            "本次解释的存档 ID。**主人用它来标注**（情境/动作/结果）。\n\n"
            "标注时不重传音频也不提交特征 —— 服务端按这个 ID 取回它自己存的。"
        ),
    )
    interpretation: dict[str, Any] | None = None


# ─────────────────────────────────────────────────────────────
# 应用工厂
# ─────────────────────────────────────────────────────────────


def create_app(
    *,
    store: MemoryStore,
    embedder: Embedder,
    llm: LLMClient,
    prior: PriorTable,
    feature_extractor: Callable[[str], Any],
    vision: VisionAnalyzer,
    auth_secret: str,
    health_store: InMemoryHealthStore | None = None,
    health_consent_version: str = "v1",
    media_extractor: Any | None = None,
    tracing_config: TracingConfig | None = None,
) -> FastAPI:
    app = FastAPI(title="pet-agent", version="0.1.0")

    # ── 观测（LangSmith）──
    # 取舍见 app/observability/langsmith.py：**失败不致命，但不静默**。
    tracing = tracing_config or TracingConfig.from_env()
    tracer, tracing_error = build_tracer(tracing)
    observability_info: dict[str, str] = tracing.describe()
    if tracing_error:
        observability_info["langsmith_error"] = tracing_error

    @app.middleware("http")
    async def _request_scope(
        request: Request, call_next: Callable[[Request], Awaitable[Any]]
    ) -> Any:
        """解析（或生成）会话与链路标识，并在**所有响应**回传。

        为什么用中间件而不是给 12 个端点各加一个参数：
        漏掉任何一个，「全链路」就断了，而那种断裂是静默的。
        中间件让它无法遗漏。
        """
        session_id = (request.headers.get("X-Session-Id") or "").strip()
        trace_id = (request.headers.get("X-Trace-Id") or "").strip()
        request.state.session_id = session_id or f"sess-{uuid4().hex[:12]}"
        request.state.trace_id = trace_id or str(uuid4())
        response = await call_next(request)
        response.headers["X-Session-Id"] = request.state.session_id
        response.headers["X-Trace-Id"] = request.state.trace_id
        return response

    def scope_of(request: Request) -> tuple[str, str]:
        """取 ``(session_id, trace_id)`` —— 由中间件保证一定存在。"""
        return request.state.session_id, request.state.trace_id

    # 健康记录另走一张表（数据策略不同：加密 / 保留期 / 级联硬删 / 显式同意）。
    # 未注入时用内存实现，让整条链路在测试里可跑。
    # 推断当前跑的是真模型还是占位（**不是传参** —— 调用方会忘，见 B22）
    provider_info = describe_providers(llm, embedder, vision)

    def dev_login_enabled() -> bool:
        """开发登录是否开启。**每次请求时读**，不在装配时冻结。

        两个理由：
        1. 它是**部署开关** —— 改了环境变量后重启即可生效，不该被装配顺序左右；
        2. 冻结成局部变量会让「装配时机」变成一个隐蔽的依赖：
           测试与运行时行为不一致，而那种不一致是静默的。

        默认关闭的理由见 `dev_login` 端点。
        """
        return _env_flag(ENV_ALLOW_DEV_LOGIN)

    def _store_info() -> dict[str, str]:
        """存储后端描述。

        ⚠️ **在请求时调用，不在装配时冻结。**
        `app.state.store_bundle` 是 `create_app` **返回之后**才挂上去的
        （见 `app/bootstrap.py`）—— 在这里提前求值会永远拿到兜底分支，
        于是 `/healthz` 永远报 “unknown”，而那是静默的：
        探针看起来正常，只是信息是错的。
        """
        bundle = getattr(app.state, "store_bundle", None)
        if bundle is not None and hasattr(bundle, "describe"):
            try:
                return dict(bundle.describe())
            except Exception as exc:  # noqa: BLE001 - 探针不能因为描述失败而挂
                return {"store": "error", "reason": str(exc)[:200]}
        # 注入式装配（测试）走这里。**不编造细节**，只报类型。
        return {"store": type(store).__name__, "durable": "unknown"}

    memory_writer = MemoryWriter(store=store, embedder=embedder)
    _health_store = health_store or InMemoryHealthStore()
    health_writer = HealthWriter(
        store=_health_store,
        redflags=RedFlagTable.load(_RED_FLAGS_PATH),
        consent_version=health_consent_version,
    )
    graph = build_graph(
        store=store,
        embedder=embedder,
        llm=llm,
        prior=prior,
        feature_extractor=feature_extractor,
        vision=vision,
        # 案例推理的样本库。**不接这个参数的话，行为解释永远只能走到
        # measured_only** —— 因为案例推理需要主人的标注记录。
        records_lookup=lambda state: store.list_meow_records(
            user_id=tenant_of(state)[0], pet_id=tenant_of(state)[1]
        ),
        # 多模态事实提取。**不传时行为解释仍然可用** ——
        # 只是没有 OBSERVED 证据（画面/音频层的观察）。
        media_extractor=media_extractor,
    )

    # ── 鉴权依赖 ─────────────────────────────────────────
    def current_user(authorization: str | None = Header(default=None)) -> str:
        try:
            return verify_token(bearer_token(authorization), secret=auth_secret)
        except InvalidToken as exc:
            # 不区分失败原因（ARCHITECTURE.md §4.2）——区分会泄露信息给攻击者
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "UNAUTHORIZED", "message": str(exc)},
            ) from exc

    def owned_pet(user_id: str, pet_id: str) -> PetProfile:
        """取宠物并**校验归属**。不符返回 404 而非 403（不泄露资源存在性）。"""
        try:
            return store.get_pet(user_id=user_id, pet_id=pet_id)
        except NotFound as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "NOT_FOUND", "message": f"pet {pet_id} 不存在"},
            ) from exc

    # ── 异常处理 ─────────────────────────────────────────
    @app.exception_handler(ValueError)
    async def value_error_handler(_request, exc: ValueError):
        """契约层的校验失败 → 400（而非 500）。"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": {"code": "VALIDATION_FAILED", "message": str(exc)}},
        )

    # ── 端点 ─────────────────────────────────────────────

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """存活探针。**不需要鉴权** —— 否则无法用于负载均衡健康检查。

        同时暴露 **provider 模式**：跑的是真模型还是占位实现。

        为什么放在探针里：真实模型需要 `DASHSCOPE_API_KEY`，
        而**没配密钥时系统会退到 mock 而不报错**。
        不把这个状态暴露出来，一次演示会静默地变成假演示 ——
        输出看起来正常，只是内容是写死的。
        """
        return {
            "status": "ok",
            "providers": provider_info,
            "observability": observability_info,
            # 存储后端与向量索引状态。
            #
            # **必须暴露**：内存后端与 MySQL 后端在功能上无法从行为区分，
            # 但一个重启就丢数据、另一个不会。看不到这一项时，
            # 「数据没了」会被当成 bug 排查很久，而它其实是配置。
            "storage": _store_info(),
            # 前端据此决定要不要显示「开发登录」入口。
            # 暴露它是安全的：它只说明开关状态，不泄露密钥。
            "dev_login": "enabled" if dev_login_enabled() else "disabled",
        }

    @app.get("/v1/pets")
    def list_pets(user_id: str = Depends(current_user)) -> dict[str, Any]:
        """列出当前用户的全部宠物。

        **按 token 派生的 ``user_id`` 过滤**，不接受任何客户端传入的归属参数 ——
        这是 ``ARCHITECTURE.md`` §4.2 A1 的落点：归属只从凭证来。
        """
        pets = store.list_pets(user_id=user_id)
        return {
            "count": len(pets),
            "pets": [_pet_brief(p) for p in pets],
        }

    @app.post("/v1/pets", status_code=status.HTTP_201_CREATED)
    def create_pet(
        body: CreatePetRequest, user_id: str = Depends(current_user)
    ) -> dict[str, Any]:
        import uuid

        pet = PetProfile(
            pet_id=str(uuid.uuid4()),
            user_id=user_id,
            name=body.name,
            species=Species.CAT,
            breed=body.breed,
        )
        store.save_pet(pet)
        return {"pet_id": pet.pet_id, "name": pet.name}

    @app.post("/v1/auth/dev-login")
    def dev_login(body: DevLoginRequest) -> dict[str, Any]:
        """开发/演示用签发 token。**默认关闭。**

        开启方式::

            PET_AGENT_ALLOW_DEV_LOGIN=1

        为什么默认关闭：它签发**任意 user_id** 的合法 token。
        默认开就等于把 D5（多租户隔离）取消 —— 隔离的前提是
        ``user_id`` 不可伪造，而这里它完全可伪造。

        生产环境请用 ``python -m app.bootstrap --issue-token <user>`` 签发，
        或接入真实的登录体系。
        """
        if not dev_login_enabled():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "DEV_LOGIN_DISABLED",
                    "message": (
                        "开发登录未开启。设置 PET_AGENT_ALLOW_DEV_LOGIN=1 后重启，"
                        "或使用 `python -m app.bootstrap --issue-token <user>` 签发 token。"
                    ),
                },
            )
        token = issue_token(
            body.user_id, secret=auth_secret, ttl_seconds=DEFAULT_TTL_SECONDS
        )
        return {
            "token": token,
            "user_id": body.user_id,
            "expires_in": DEFAULT_TTL_SECONDS,
        }

    @app.post("/v1/pets/{pet_id}/profile")
    def build_profile(
        pet_id: str,
        body: ProfileRequest,
        user_id: str = Depends(current_user),
    ) -> dict[str, Any]:
        """上传照片建档案。

        ``confirm=False``（默认）时**只返回草案、不落库** ——
        因为 ``DESIGN.md`` §3.3 要求档案必须经用户确认。
        """
        pet = owned_pet(user_id, pet_id)
        state = initial_state(
            user_id=user_id,
            pet_id=pet_id,
            raw_input=RawInput(image_urls=body.image_urls, text="这是它的照片"),
        )
        result = graph.invoke(state)

        draft = result.get("profile_draft")
        if draft is None:
            errors = [e.message for e in result.get("errors", [])]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "PROFILE_ANALYSIS_FAILED",
                    "message": "照片分析失败，无法建立档案",
                    "details": {"errors": errors},
                },
            )

        payload = {
            "draft": draft.model_dump(mode="json"),
            "message": result.get("final_response", ""),
            "confirmed": body.confirm,
        }

        if body.confirm:
            updated = pet.model_copy(
                update={
                    "visual": draft.visual,
                    "must_keep_features": draft.must_keep_features,
                    "observed_but_unstable": draft.observed_but_unstable,
                    "breed": draft.breed or pet.breed,
                }
            )
            store.save_pet(updated)
            payload["profile"] = updated.model_dump(mode="json")

        return payload

    @app.get("/v1/pets/{pet_id}/profile")
    def read_profile(
        pet_id: str, user_id: str = Depends(current_user)
    ) -> dict[str, Any]:
        return owned_pet(user_id, pet_id).model_dump(mode="json")

    @app.post("/v1/chat")
    def chat(
        pet_id: str,
        body: ChatRequest,
        request: Request,
        user_id: str = Depends(current_user),
    ) -> TurnResponse:
        owned_pet(user_id, pet_id)
        session_id, trace_id = scope_of(request)
        raw = _to_raw_input(body)
        response = _run(
            graph,
            user_id=user_id,
            pet_id=pet_id,
            raw=raw,
            session_id=session_id,
            trace_id=trace_id,
            store=store,
            provider_info=provider_info,
            tracer=tracer,
        )

        # ── 把这一轮存成会话消息 ──
        # 同一批消息服务两个用途：① 日报（`app/digest`）的输入；
        # ② 下一轮的**会话记忆注入**（按 session_id 取最近 N 条）。
        # **记忆事件不能当日报输入** —— 那是总结的产物，拿它当输入就循环了。
        if body.text:
            store.insert_message(
                SessionMessage(
                    user_id=user_id,
                    pet_id=pet_id,
                    session_id=session_id,
                    role="user",
                    content=body.text,
                )
            )
        if response.final_response:
            store.insert_message(
                SessionMessage(
                    user_id=user_id,
                    pet_id=pet_id,
                    session_id=session_id,
                    role="assistant",
                    content=response.final_response,
                )
            )
        return response

    @app.post("/v1/interpret")
    def interpret_meow(
        pet_id: str,
        body: InterpretRequest,
        request: Request,
        user_id: str = Depends(current_user),
    ) -> TurnResponse:
        owned_pet(user_id, pet_id)
        session_id, trace_id = scope_of(request)
        raw = RawInput(
            audio_url=body.audio_url,
            audio_kind=AudioKind.CAT_MEOW,
            scene_description=body.scene_description,
        )
        return _run(
            graph,
            user_id=user_id,
            pet_id=pet_id,
            raw=raw,
            session_id=session_id,
            trace_id=trace_id,
            store=store,
            provider_info=provider_info,
            tracer=tracer,
        )

    # ── 主人标注（案例推理的燃料） ──────────────────────

    @app.post(
        "/v1/pets/{pet_id}/meow-records",
        status_code=status.HTTP_201_CREATED,
    )
    def label_meow(
        pet_id: str,
        body: LabelMeowRequest,
        request: Request,
        user_id: str = Depends(current_user),
    ) -> dict[str, Any]:
        """把一次解释标注为一条叫声记录。

        **这是案例推理的输入路径。** 没有它，`MeowRecord` 永远为空，
        `case_based` 模式永远走不到 —— 而那是「学习主人的经验」的全部落点。

        特征从服务端存档里按 `interpretation_id` 取回，
        **不接受客户端提交** —— 否则 `MEASURED` 的承诺失效。
        """
        owned_pet(user_id, pet_id)
        session_id, _ = scope_of(request)
        pending = store.get_pending_interpretation(
            user_id=user_id, pet_id=pet_id, interpretation_id=body.interpretation_id
        )
        if pending is None:
            # 归属不符与不存在返回同一个响应 —— 不泄露资源存在性
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="找不到这次解释（或它不属于该宠物）",
            )

        record = store.insert_meow_record(
            MeowRecord(
                user_id=user_id,
                pet_id=pet_id,
                # 案例的真实来源是**录叫声那一轮**，而不是标注发生的那一轮 ——
                # 标注可能是很久之后补的。所以优先用解释自己的 session。
                session_id=pending.session_id or session_id,
                context=body.context,
                features=pending.features,
                actions=body.actions,
                resolution=body.resolution,
                recorded_at=pending.created_at,
            )
        )
        confirmed = store.list_meow_records(user_id=user_id, pet_id=pet_id)
        return {
            "record_id": record.record_id,
            "context": record.context.value,
            "actions": [a.value for a in record.actions],
            "resolution": record.resolution,
            "confirmed_records": len(confirmed),
        }

    @app.get("/v1/pets/{pet_id}/meow-records")
    def list_meow_records(
        pet_id: str,
        user_id: str = Depends(current_user),
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """案例推理的样本库。数量直接决定解释能走到哪个模式。

        ``?session_id=`` 用于**全链路追溯**（这批案例来自哪一轮），
        **不参与相似度计算**。
        """
        owned_pet(user_id, pet_id)
        records = store.list_meow_records(
            user_id=user_id, pet_id=pet_id, session_id=session_id
        )
        return {
            "count": len(records),
            "records": [
                {
                    "record_id": r.record_id,
                    "context": r.context.value,
                    "actions": [a.value for a in r.actions],
                    "resolution": r.resolution,
                    "recorded_at": r.recorded_at.isoformat(),
                }
                for r in records
            ],
        }

    # ── 日报（「宠物的一天」） ─────────────────────────

    @app.get("/v1/pets/{pet_id}/story")
    def daily_story(
        pet_id: str,
        day: str,
        user_id: str = Depends(current_user),
    ) -> StoryResponse:
        """生成某一天的「宠物的一天」。

        内部串起 `app.digest`（提取 + quote 校验）与 `app.story`（娱乐层），
        并把健康信号**改道**到 `health_records` —— 这正是 §5.4.2 的闭环。
        """
        owned_pet(user_id, pet_id)
        target = _parse_day(day)
        start = datetime.combine(target, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        messages = store.list_messages(
            user_id=user_id, pet_id=pet_id, since=start, until=end
        )
        digest_messages = [m.as_digest_message() for m in messages]
        existing = store.list_memories(user_id=user_id, pet_id=pet_id)

        digest = summarize_day(
            day=target,
            messages=digest_messages,
            writer=memory_writer,
            llm=llm,
            user_id=user_id,
            pet_id=pet_id,
            existing_memories=existing,
            health_sink=health_writer,
        )
        story = compose_story(digest.summary, messages=digest_messages)

        from app.story import render_beat_for_panel

        return StoryResponse(
            date=target.isoformat(),
            title=story.title,
            # 只有**真的跑过提取**时才展示说明。
            #
            # 初版无条件传 notes，于是「当天没对话」（skipped_reason 非空）
            # 也会显示一条说明 —— 而那两件事的含义完全不同：
            #   没消息        → 正常状态，一句都不用说
            #   有消息但失败  → 可能是系统的问题，必须说
            story_text=render_story(
                story,
                digest_notes=(None if digest.skipped_reason else digest.summary.notes),
            ),
            beats=[render_beat_for_panel(b) for b in story.beats],
            excluded_count=len(story.excluded),
            health_notice=story.health_notice,
            disclaimer=story.disclaimer,
            digest_notes=digest.summary.notes,
            rejection_rate=round(digest.summary.rejection_rate, 4),
        )

    # ── 健康 ──────────────────────────────────────────

    @app.get("/v1/pets/{pet_id}/health")
    def health(pet_id: str, user_id: str = Depends(current_user)) -> HealthResponse:
        """健康监测与分诊结果。**不是诊断。**

        信号源目前是结构化记录（`record_signal` 通道）。
        对话提取的健康记录**值为 None**，因此进 `signals_missing`
        并导致 `INSUFFICIENT_DATA` —— 那是诚实的。
        """
        owned_pet(user_id, pet_id)
        records = _health_store.list_records(user_id=user_id, pet_id=pet_id)

        # 只把**有值**的结构化信号交给红旗求值。
        # 值为 None 的一律不参与 —— 「不知道」不能变成「未命中」。
        signals: dict[str, Any] = {}
        for rec in records:
            if rec.value is not None:
                signals[rec.signal] = rec.value

        assessment = health_writer.assess(
            user_id=user_id, pet_id=pet_id, signals=signals
        )
        return HealthResponse(
            level=assessment.level.value,
            coverage=assessment.coverage,
            coverage_note=assessment.coverage_note,
            recommendation=assessment.recommendation,
            disclaimer=assessment.disclaimer,
            must_not_be_read_as=assessment.must_not_be_read_as,
            rule_version=assessment.rule_version,
            red_flags=[
                {
                    "rule_id": h.rule_id,
                    "urgency": h.urgency.value,
                    "title": h.title,
                    "message": h.message,
                    "action": h.action,
                    "sources": h.sources,
                }
                for h in assessment.red_flags_triggered
            ],
            signals_missing_count=len(assessment.signals_missing),
            record_count=len(records),
        )

    @app.post("/v1/pets/{pet_id}/health/signals", status_code=status.HTTP_201_CREATED)
    def record_health_signal(
        pet_id: str,
        body: HealthSignalRequest,
        request: Request,
        user_id: str = Depends(current_user),
    ) -> dict[str, Any]:
        """记录一个**结构化**健康信号。红旗由这条通道触发。

        与对话提取的区别只有一个字：**值**。这里有值，所以能比较、能触发；
        那里没有值，所以只能记录。
        """
        owned_pet(user_id, pet_id)
        session_id, _ = scope_of(request)
        record = health_writer.record_signal(
            user_id=user_id,
            pet_id=pet_id,
            signal=body.signal,
            value=body.value,
            source=HealthRecordSource.USER_INPUT,
            at=body.at or datetime.now(timezone.utc),
            session_id=session_id,
        )
        return {
            "record_id": record.record_id,
            "signal": record.signal,
            "value": record.value,
        }

    @app.get("/v1/memories")
    def list_memories(
        pet_id: str,
        user_id: str = Depends(current_user),
        include_non_active: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        owned_pet(user_id, pet_id)
        memories = store.list_memories(
            user_id=user_id,
            pet_id=pet_id,
            include_non_active=include_non_active,
            session_id=session_id,
        )
        return {
            "count": len(memories),
            "memories": [
                {
                    "memory_id": m.memory_id,
                    "content": m.content,
                    "status": m.status.value,
                    "source": m.source.value,
                    "confidence": m.confidence,
                    "support_count": m.support_count,
                }
                for m in memories
            ],
        }

    @app.get("/v1/pets/{pet_id}/habits")
    def list_habits(
        pet_id: str,
        user_id: str = Depends(current_user),
        text: bool = False,
    ) -> dict[str, Any]:
        """这只猫的习惯。**全部数字由代码聚合得出，不经过任何模型。**

        `text=true` 时额外返回一段渲染好的说明（同样不经模型）。

        与 `GET /v1/memories` 的分工：

        | 端点 | 回答 |
        |---|---|
        | `/v1/memories` | 原始记忆条目（**逐条**）|
        | `/v1/habits` | 从这些条目**聚合**出的模式（次数/天数/时段/规律性）|

        两者都需要：前者可核对，后者可回答「它有什么习惯」——
        而那是聚合问题，不是相似度问题（`docs/06-roadmap.md` §5.1）。
        """
        owned_pet(user_id, pet_id)

        # **租户过滤在 store 层完成**：`list_habits` 不做租户检查。
        # 把过滤下推到查询而不是在这里再筛一遍 —— 后者会让人
        # 以为「不过滤也安全」。
        memories = store.list_memories(
            user_id=user_id, pet_id=pet_id, include_non_active=True
        )
        report = detect_habits(memories)

        payload: dict[str, Any] = {
            "considered_events": report.considered_events,
            "skipped_events": report.skipped_events,
            "skip_reasons": report.skip_reasons,
            "limitations": list(report.limitations),
            "established_count": len(report.established()),
            "habits": [
                {
                    "subject": h.subject,
                    "content": h.content,
                    "event_type": h.event_type.value,
                    "strength": h.strength.value,
                    "trend": h.trend.value,
                    # 计数：每一个都能由存储的事件重算
                    "observations": h.observations,
                    "distinct_days": h.distinct_days,
                    "span_days": h.span_days,
                    "first_seen": h.first_seen.isoformat(),
                    "last_seen": h.last_seen.isoformat(),
                    # 时间分布
                    "time_histogram": {
                        k.value: v for k, v in h.time_histogram.items()
                    },
                    "dominant_time": h.dominant_time.value if h.dominant_time else None,
                    "time_concentration": h.time_concentration,
                    "regularity": h.regularity,
                    # 可核对：哪些记忆支撑这条习惯
                    "evidence_ids": list(h.evidence_ids),
                    # 为什么不能声称更多（**空的才是可疑的**）
                    "limitations": list(h.limitations),
                }
                for h in report.habits
            ],
        }
        if text:
            payload["text"] = render_habit_report(report)
        return payload

    return app


# ─────────────────────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────────────────────


def _trust_notice(provider_info: dict[str, str] | None) -> str | None:
    """当前输出是否可以当作真实推理。

    **`unknown` 与 `mock` 同等处理** —— 无法确认是真模型时，
    应当告诉调用方别当真，而不是默默让它以为一切正常。
    """
    mode = (provider_info or {}).get("mode", "unknown")
    # 用 `UNTRUSTED_MODES` 而不是手写两个字面量 ——
    # 它一变这里必须跟着变，否则「不可信」集合会有两个真相来源。
    if mode not in UNTRUSTED_MODES:
        return None
    if mode == "mock":
        return (
            "⚠️ 当前为 **mock 模式**（未配置任何供应商密钥）："
            "以上回复与判断来自占位实现，不是真实模型推理。"
        )
    return (
        "⚠️ 无法确认 provider 是否为真实模型（检测到混用的实现）："
        "请用 `/healthz` 核对 `providers` 字段，不要把这些输出当作真实推理。"
    )


def _parse_day(value: str) -> date:
    """解析 `YYYY-MM-DD`。**非法输入直接 400，不猜日期。**"""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"day 必须为 YYYY-MM-DD，得到 {value!r}",
        ) from exc


def _to_raw_input(body: ChatRequest) -> RawInput:
    """把请求体转成契约对象。

    有音频但没声明 ``audio_kind`` 时**显式报错**而非猜 ——
    用户语音与猫咪叫声走完全不同的链路（`DESIGN.md` §3.6）。
    """
    if body.audio_url and body.audio_kind is None:
        raise ValueError(
            "提供 audio_url 时必须指定 audio_kind："
            "user_voice 走 ASR 转写，cat_meow 走声学特征提取"
        )
    return RawInput(
        text=body.text or None,
        audio_url=body.audio_url,
        audio_kind=body.audio_kind,
        image_urls=body.image_urls,
        scene_description=body.scene_description,
    )


def _run(
    graph,
    *,
    user_id: str,
    pet_id: str,
    raw: RawInput,
    session_id: str,
    trace_id: str,
    store: MemoryStore | None = None,
    provider_info: dict[str, str] | None = None,
    tracer: Any | None = None,
) -> TurnResponse:
    """执行一次编排并组装响应。

    ``session_id`` / ``trace_id`` 由中间件解析（缺省服务端生成），
    这里把它们传进状态与 LangGraph 的 ``config`` —— 于是
    「会话链路」与「LangSmith run」用的是同一组标识
    （见 `app/observability/langsmith.py`）。

    ``store`` 传入时，会把带声学特征的解释**存档**并返回 `interpretation_id` ——
    那是主人标注的入口（`POST /v1/pets/{pid}/meow-records`）。
    """
    state = initial_state(
        user_id=user_id,
        pet_id=pet_id,
        raw_input=raw,
        session_id=session_id,
        trace_id=trace_id,
    )
    if raw.text:
        state["transcribed_text"] = raw.text

    # `run_id` 与 `trace_id` 同源；未启用 tracer 时 config 仍带 metadata
    config = run_config(
        trace_id=trace_id,
        session_id=session_id,
        user_id=user_id,
        pet_id=pet_id,
        tracer=tracer,
    )
    result = graph.invoke(state, config=config)
    langsmith_run_id = str(resolve_run_id(trace_id)) if tracer is not None else None

    guard = result.get("guard_result")
    interp = result.get("interpretation")
    features = result.get("acoustic_features")
    node_traces = list(result.get("node_trace", []))

    # ── 降级的**聚合** ──
    #
    # 初版只看 guard 的 `degrade_to_conservative`，于是记忆检索降级
    # （例如 Milvus 不可用）时，**节点 trace 里标了 degraded=true，
    # 而响应顶层的 degraded 却是 false** —— 前端据此判断“一切正常”，
    # 而用户看到的是一句没有依据的“没有相关记录”。
    #
    # 降级是“这次结果没有正常产生”这一类事实的整体属性，
    # 所以看全部来源，而不是只看守卫。
    degraded_nodes = [t for t in node_traces if t.degraded]
    guard_degraded = bool(guard and guard.degrade_to_conservative)
    degraded = guard_degraded or bool(degraded_nodes)

    notice = guard.degraded_notice if guard else None
    if notice is None and degraded_nodes:
        notice = "；".join(f"{t.node}: {t.decision}" for t in degraded_nodes)

    # ── 存档解释，等主人标注 ──
    # 只有**真有声学特征**时才存档：没有特征就没有可标注的东西，
    # 而存档一个空特征会让标注路径写入一条无意义的案例。
    interpretation_id = _archive_interpretation(
        store=store,
        user_id=user_id,
        pet_id=pet_id,
        interp=interp,
        features=features,
        session_id=session_id,
    )

    return TurnResponse(
        final_response=result.get("final_response") or "",
        intent=(result.get("intent").value if result.get("intent") else "unknown"),
        route_confidence=result.get("route_confidence", 0.0),
        session_id=session_id,
        trace_id=trace_id,
        langsmith_run_id=langsmith_run_id,
        degraded=degraded,
        degraded_notice=notice,
        trace=[
            TraceItem(
                node=t.node,
                latency_ms=t.latency_ms,
                decision=t.decision,
                degraded=t.degraded,
            )
            for t in node_traces
        ],
        retrieved_count=len(result.get("retrieved_memories", [])),
        written_memory_ids=result.get("written_memory_ids", []),
        interaction_mode=result.get("interaction_mode", "companion"),
        provider_mode=(provider_info or {}).get("mode", "unknown"),
        mock_notice=_trust_notice(provider_info),
        suggested_actions=[
            a.value for a in result.get("pending_action_suggestions", [])
        ],
        interpretation_id=interpretation_id,
        interpretation=(interp.model_dump(mode="json") if interp is not None else None),
    )


def _archive_interpretation(
    *, store, user_id: str, pet_id: str, interp, features, session_id: str | None = None
) -> str | None:
    """把一次解释连同它**服务端产生的特征**存档。

    特征存在服务端而不是让客户端回传 —— 这是 `MEASURED` 承诺的唯一兑现方式。
    客户端若能提交特征，它就能发任意数字，而案例推理的相似度会建在它们上面。
    """
    if store is None or interp is None or features is None:
        return None

    import uuid

    interpretation_id = f"itp-{uuid.uuid4().hex[:12]}"
    store.save_pending_interpretation(
        PendingInterpretation(
            interpretation_id=interpretation_id,
            user_id=user_id,
            pet_id=pet_id,
            session_id=session_id,
            features=features,
            evidence_mode=interp.evidence_mode,
            candidates=list(interp.candidates),
        )
    )
    return interpretation_id
