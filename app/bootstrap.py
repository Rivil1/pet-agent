"""运行时装配：**环境变量 → 可启动的 FastAPI app**。

## 为什么需要这个模块

`create_app` 是**依赖注入**的（利于离线测试），但生产需要一个地方把真实依赖装进去。
此前没有任何模块做这件事，于是 Dockerfile 里的

```
uvicorn app.api.main:create_app --factory
```

**必然失败** —— `create_app` 的 store / embedder / llm / prior / feature_extractor /
vision / auth_secret 全是必填关键字参数，`--factory` 会用零参调用它。

后果不是「Docker 起不来」这么简单，而是：

> **多厂商配置层（决策 D46）从未真正生效。** `build_providers` 只被测试调用过。

所以这个模块是多厂商改造的**收口点**，不是附属品。

## 装配顺序

```
环境变量
  └─ build_providers()        # 供应商 / 每能力模型 / 端点（见 app/llm/providers.py）
       ├─ llm       ─┐
       ├─ embedder   ├─► create_app(...)
       ├─ vision     │      ├─ InMemoryStore
       └─ multimodal ┘      ├─ PriorTable.load(data/priors/...)
                            ├─ fetch_audio_features   （MEASURED 证据的 URL 适配器）
                            └─ auth_secret            （缺则**拒绝启动**）
```

## 密钥与鉴权：fail-closed

| 缺失项 | 行为 | 理由 |
|---|---|---|
| 供应商密钥 | 退到 mock，但 `/healthz` 与每个响应**显式标注** | D45：假演示不得静默发生 |
| `PET_AGENT_AUTH_SECRET` | **直接拒绝启动** | 没有它 `user_id` 可伪造，多租户隔离主张不成立（D5） |
"""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# 加载 .env 文件（如果存在）
load_dotenv()

from app.api import create_app
from app.audio.features import TARGET_SR, extract_features
from app.interpreter import PriorTable
from app.llm import Providers, build_providers
from app.store.factory import StoreBundle, build_store_from_env

#: 项目根（`app/` 的上一级）。用于定位 `data/`。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 默认先验表。可由 `PET_AGENT_PRIORS_PATH` 覆盖。
DEFAULT_PRIORS_PATH = PROJECT_ROOT / "data" / "priors" / "catmeows_stats.json"

#: 鉴权密钥的环境变量名，按优先级排列。
#:
#: 第一个是本项目规范名；后两个是兼容别名 —— 架构文档 §8.3 里叫 `AUTH_DEV_TOKEN`，
#: 但它描述的是「一个 token」，而这里需要的是**签名密钥**（签发任意用户 token），
#: 概念不同，所以规范名独立。
_AUTH_ENV_NAMES = ("PET_AGENT_AUTH_SECRET", "AUTH_SECRET", "AUTH_DEV_TOKEN")
ENV_PRIORS_PATH = "PET_AGENT_PRIORS_PATH"

#: 下载音频的超时（秒）。声学特征是 `MEASURED` 证据，超时后节点会降级但保留观察证据。
ENV_MEDIA_TIMEOUT_S = "PET_AGENT_MEDIA_TIMEOUT_S"


class BootstrapError(RuntimeError):
    """装配失败。**这是配置错误，不是运行时故障** —— 应当在启动时立刻暴露。"""


def _env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def resolve_auth_secret() -> str:
    """读鉴权密钥。**缺失直接报错**，不退到默认值。

    默认值会让「忘了配」表现成「配了一个所有人都知道的密钥」——
    那比启动失败危险得多。
    """
    for name in _AUTH_ENV_NAMES:
        secret = _env(name)
        if secret:
            return secret
    allowed = " / ".join(_AUTH_ENV_NAMES)
    raise BootstrapError(
        f"缺少鉴权密钥。请设置 {allowed} 之一。\n"
        "它用于签发/校验 user token —— 没有它 user_id 可被伪造，租户隔离不成立。"
    )


def resolve_priors_path() -> Path:
    path = Path(_env(ENV_PRIORS_PATH) or DEFAULT_PRIORS_PATH)
    if not path.is_file():
        raise BootstrapError(f"先验文件不存在：{path}（用 {ENV_PRIORS_PATH} 覆盖）")
    return path


def _media_timeout_s() -> float:
    """下载超时。**非法值报错而不是静默用默认值** —— 与管理层的处理一致。"""
    raw = _env(ENV_MEDIA_TIMEOUT_S)
    if raw is None:
        return 30.0
    try:
        return float(raw)
    except ValueError as exc:
        raise BootstrapError(f"{ENV_MEDIA_TIMEOUT_S}={raw!r} 不是数字") from exc


def fetch_audio_features(audio_url: str) -> Any:
    """`feature_extractor` 的生产适配器：URL → 声学特征。

    契约层给的是 URL，而 `librosa` 只吃本地路径或文件对象 —— 这一步负责桥接。
    产出的是 **`MEASURED`** 证据（代码可重算），与多模态观察的 `OBSERVED` 不同。

    **延迟导入 httpx / librosa**：它们较慢，且离线测试不需要。
    """
    import httpx
    import numpy as np

    # 动态导入：librosa 的静态解析依赖分析器选中的解释器，
    # 而它是运行时依赖 —— 用 import_module 避免把「分析器找不到」
    # 误报成「代码写错」。
    librosa = importlib.import_module("librosa")

    timeout = _media_timeout_s()
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(audio_url)
        response.raise_for_status()
        payload = response.content

    y, sr = librosa.load(io.BytesIO(payload), sr=TARGET_SR, mono=True)
    return extract_features(np.asarray(y, dtype=np.float32), sr)


def create_app_from_env(
    *,
    providers: Providers | None = None,
    auth_secret: str | None = None,
    feature_extractor: Any | None = None,
    prior_path: str | Path | None = None,
    store: Any | None = None,
    health_store: Any | None = None,
) -> Any:
    """把环境变量装配成一个可启动的 app。

    参数只为**测试注入**而存在 —— 生产路径是零参调用
    （`uvicorn app.bootstrap:create_app_from_env --factory`）。

    Raises:
        BootstrapError: 缺少鉴权密钥，或先验文件不存在。
        ValueError: 存储配置不完整（例如指定了 mysql 但缺 MYSQL_PASSWORD）。
    """
    built = providers if providers is not None else build_providers()
    secret = auth_secret or resolve_auth_secret()
    priors = PriorTable.load(prior_path or resolve_priors_path())

    # 存储后端：显式注入 > 环境变量。
    #
    # 向量维度从 provider 配置取，**不重新读环境变量** ——
    # 两处各读一次的话，改了 `PET_AGENT_EMBED_MODEL`（维度不同）却只重启了一半，
    # 建 collection 时才会因为维度不符报错，而那个报错离根因很远。
    bundle: StoreBundle | None = None
    if store is not None:
        resolved_store = store
        resolved_health = health_store
    else:
        bundle = build_store_from_env(dim=built.config.embed_dim if built.config else 1024)
        resolved_store = bundle.store
        resolved_health = health_store if health_store is not None else bundle.health_store

    # 装配结果挂到 app.state 上，供 /healthz 与运维查看
    app = create_app(
        store=resolved_store,
        embedder=built.embedder,
        llm=built.llm,
        prior=priors,
        feature_extractor=feature_extractor or fetch_audio_features,
        vision=built.vision,
        auth_secret=secret,
        # `built.multimodal` 为 None 时节点走「未配置」降级，
        # 而不是「配置了但失败」—— 两者对用户的提示不同。
        media_extractor=built.multimodal,
        health_store=resolved_health,
    )
    if bundle is not None:
        app.state.store_bundle = bundle
    return app


def _main(argv: list[str] | None = None) -> int:
    """开发辅助 CLI。

    没有它，一个刚启动的 app 是**用不了**的：所有端点都要 Bearer token，
    而没有地方签发。这里提供最小可用的签发入口。
    """
    import argparse

    from app.auth.token import issue_token

    parser = argparse.ArgumentParser(description="pet-agent 运行时装配辅助")
    parser.add_argument(
        "--issue-token", metavar="USER_ID", help="用环境密钥签发一个 user token"
    )
    parser.add_argument(
        "--describe", action="store_true", help="打印当前 provider 装配结果"
    )
    args = parser.parse_args(argv)

    if args.issue_token:
        print(issue_token(args.issue_token, secret=resolve_auth_secret()))
        return 0
    if args.describe:
        providers = build_providers()
        for key, value in providers.describe().items():
            print(f"{key}: {value}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
