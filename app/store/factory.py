"""存储装配：**环境变量 → 可用的存储组合**。

## 选择逻辑（fail-closed，且**永远说清用的是哪一个**）

| 配置 | 结果 |
|---|---|
| `MYSQL_HOST` 已设 | MySQL 主库；`MILVUS_URI` 也设了就用 Milvus，否则内存向量索引 |
| `MYSQL_HOST` 未设 | 全部走内存（测试与离线开发） |

**没有「配置了但连不上就悄悄退回内存」这条路径。** 理由与 D45 同源：
静默降级会让一次真实演示变成假演示 —— 数据看起来存下来了，
重启之后全没了。连不上就报错，让问题在启动时暴露。

## 为什么向量索引可以是内存的

向量索引是**纯加速器**：MySQL 里存了向量本身，索引丢了可以回填
（`MySQLStore.backfill_vectors()`）。所以在没有 Milvus 时用 `InMemoryVectorIndex`
是合理的降级 —— 检索结果仍然正确，只是进程重启后需要回填。

这与「静默降级」的区别在于：**降级状态是可见的**
（`describe()` 会出现在 `/healthz` 里），而且不丢数据。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.store.base import MemoryStore
from app.store.memory import InMemoryStore
from app.store.mysql import MySQLPool, MySQLStore
from app.store.vectors import (
    InMemoryVectorIndex,
    VectorIndex,
    VectorIndexUnavailable,
)

__all__ = ["StoreBundle", "build_store_from_env", "apply_schema"]

DDL_PATH = Path(__file__).resolve().parent / "ddl.sql"

#: 存储后端选择：`auto`（默认，按环境变量推断）/ `memory` / `mysql`
ENV_STORE = "PET_AGENT_STORE"


def _env(name: str) -> str | None:
    value = (os.environ.get(name) or "").strip()
    return value or None


def _int_env(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} 不是整数") from exc


@dataclass
class StoreBundle:
    """一组装配好的存储。

    `close()` 是必需的：连接池不关会在测试里累积，
    最终以 `Too many connections` 的形式暴露 —— 而那个报错
    离真正的原因（忘了 close）非常远。
    """

    store: MemoryStore
    health_store: Any
    backend: str
    detail: dict[str, str] = field(default_factory=dict)
    vector_index: VectorIndex | None = None
    _closables: list[Any] = field(default_factory=list, repr=False)

    def describe(self) -> dict[str, str]:
        out = {"store": self.backend}
        out.update(self.detail)
        if self.vector_index is not None:
            out.update({f"vector.{k}": v for k, v in self.vector_index.describe().items()})
        return out

    def close(self) -> None:
        for obj in self._closables:
            closer = getattr(obj, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - 关闭失败不应影响其他清理
                    pass


def _build_vector_index(*, dim: int) -> tuple[VectorIndex, list[Any]]:
    """建向量索引。Milvus 未配置时退回内存实现。"""
    uri = _env("MILVUS_URI")
    if uri is None:
        host = _env("MILVUS_HOST")
        if host:
            port = _int_env("MILVUS_PORT", 19530)
            uri = f"http://{host}:{port}"

    if uri is None:
        return InMemoryVectorIndex(dim=dim), []

    from app.store.milvus import MilvusVectorIndex

    index = MilvusVectorIndex(
        uri=uri,
        token=_env("MILVUS_TOKEN"),
        collection=_env("MILVUS_COLLECTION") or "pet_memory",
        dim=dim,
    )
    return index, []


def apply_schema(pool: MySQLPool) -> int:
    """执行 DDL（全部 `CREATE TABLE IF NOT EXISTS`，**幂等**）。

    不引 migration 框架：这个阶段表结构还在动，而 migration 的价值
    在于「多个环境的历史版本不一致」—— 现在只有一个环境。

    真正的 migration 需求出现时（线上有数据不能重建），
    再引 Alembic 也不迟；那时 DDL 已经是稳定的基线。

    Returns:
        执行的语句数。
    """
    sql_text = DDL_PATH.read_text(encoding="utf-8")
    statements = _split_sql(sql_text)

    executed = 0
    with pool.acquire() as conn, conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
            executed += 1
    return executed


def _split_sql(text: str) -> list[str]:
    """按分号切语句，**跳过注释行**。

    为什么不用简单的 `text.split(";")`：DDL 里的注释含分号
    （「删掉；不保留」这类中文标点不算，但英文分号会出现），
    切错会把半句 SQL 发出去，而 MySQL 报的是语法错误 ——
    与真正的原因（切分逻辑）差很远。
    """
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("--"):
            continue
        lines.append(raw)

    cleaned = "\n".join(lines)
    return [s.strip() for s in cleaned.split(";") if s.strip()]


def build_store_from_env(*, dim: int = 1024) -> StoreBundle:
    """按环境变量装配存储。

    Args:
        dim: 向量维度。必须与 embedding 模型一致 —— 不一致会在建 collection /
             索引写入时报错，而那个报错离「配置写错了」有一段距离，
             所以这里显式传进来而不是各自读环境变量。

    Raises:
        ValueError: 显式指定了 mysql 但缺少必要配置。
    """
    backend = (_env(ENV_STORE) or "auto").lower()
    if backend not in {"auto", "memory", "mysql"}:
        raise ValueError(
            f"{ENV_STORE}={backend!r} 不是已知后端（允许：auto / memory / mysql）"
        )

    host = _env("MYSQL_HOST")

    if backend == "memory" or (backend == "auto" and not host):
        from app.health.store import InMemoryHealthStore

        return StoreBundle(
            store=InMemoryStore(),
            health_store=InMemoryHealthStore(),
            backend="memory",
            detail={
                # 显式说明为什么是内存 —— 否则「数据重启就没了」
                # 会在很久以后才被发现，而那时已经丢过一次演示数据了
                "reason": "未配置 MYSQL_HOST"
                if backend == "auto"
                else f"{ENV_STORE}=memory",
                "durable": "false",
            },
        )

    if not host:
        raise ValueError(
            f"{ENV_STORE}=mysql 但未设置 MYSQL_HOST。\n"
            f"必需：MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE"
        )

    missing = [
        name
        for name in ("MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE")
        if not _env(name)
    ]
    if missing:
        raise ValueError(f"MySQL 配置不完整，缺少：{', '.join(missing)}")

    pool = MySQLPool(
        host=host,
        port=_int_env("MYSQL_PORT", 3306),
        user=_env("MYSQL_USER") or "",
        password=_env("MYSQL_PASSWORD") or "",
        database=_env("MYSQL_DATABASE") or "",
        size=_int_env("MYSQL_POOL_SIZE", 8),
    )

    statements = apply_schema(pool)

    index, closables = _build_vector_index(dim=dim)
    store = MySQLStore(pool=pool, vector_index=index, dim=dim)

    from app.health.mysql import MySQLHealthRecordStore

    detail: dict[str, str] = {
        "durable": "true",
        "schema_statements": str(statements),
        "pool_size": str(pool._size),  # noqa: SLF001 - 只用于展示
    }

    # 索引自检：Milvus 连不上时**不阻止启动**（它只是加速器），
    # 但要把状态说清楚 —— 这是「失败可见」而不是「失败被隐藏」。
    try:
        index.count()
        detail["vector_ready"] = "true"
    except VectorIndexUnavailable as exc:
        detail["vector_ready"] = "false"
        detail["vector_error"] = str(exc)[:200]

    return StoreBundle(
        store=store,
        health_store=MySQLHealthRecordStore(pool=pool),
        backend="mysql",
        detail=detail,
        vector_index=index,
        _closables=[pool, *closables],
    )
