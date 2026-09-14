"""MySQL 存储实现（主库，决策 D5 的物理落点）。

## 为什么 SQL 语句是模块级常量而不是内联在方法里

因为**「新加了一个查询但忘了加租户过滤」是本项目最危险的一类缺陷**。
它不会报错，只会让 A 用户的猫读到 B 用户的记忆 —— 而这正是 D5
（多宠物 + 多用户全链路隔离）要防的事。

内联的 SQL 只能靠人读代码来审。提成常量之后，`tests/test_store_sql.py`
可以逐条断言「每条按租户读取的语句都同时含 `user_id` 与 `pet_id`」，
于是**遗漏会变成测试失败**，而不是一个只在生产被利用的越权。

同一个理由适用于 `_TENANT_SCOPED_SQL` 这个字典本身：它是一份可枚举的清单。

## 时区：所有 DATETIME 都是 UTC

领域模型全是 aware UTC，而 MySQL 的 `DATETIME` **不带时区**。
不在这一层统一转换，读写会各按各的时区解释 —— 而那是静默的：
时间会偏移，没有任何报错。所以 `_to_db` / `_from_db` 是强制的收口点，
任何新的写库路径都必须过它们。

## 向量为什么存在 MySQL 里

检索走 Milvus，但向量本身也持久化在 MySQL（`memories.vector`）。
因为重建索引需要**重新调用 embedding API**（要花钱），
而持久化它的代价只是每行 4KB。于是 Milvus 变成纯索引：
丢了跑一次 `backfill_vectors()` 即可，不必重新付费。
"""

from __future__ import annotations

import json
import logging
import struct
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.schemas import (
    AcousticFeatures,
    BehaviorAction,
    ContextLabel,
    EvidenceMode,
    IntentCandidate,
    MemoryEvent,
    MemoryItem,
    MemorySource,
    MemoryStatus,
    MeowRecord,
    Moment,
    MomentScene,
    PendingInterpretation,
    PetProfile,
    RetrievalSource,
    SessionMessage,
    Species,
    Trait,
    VisualProfile,
)
from app.store.base import DuplicateMemory, MemoryStore, NotFound
from app.store.vectors import (
    VectorEntry,
    VectorIndex,
    VectorIndexUnavailable,
)

logger = logging.getLogger(__name__)

__all__ = ["MySQLPool", "MySQLStore"]


# =============================================================================
# 连接池
# =============================================================================


class MySQLPool:
    """极简连接池。

    ## 为什么是队列而不是 thread-local

    thread-local 更简单，但它没有上限：uvicorn 的线程池有多大就有多少连接，
    而且**无法显式关闭**（每个线程各自持有）。测试里反复建 app 会泄漏连接，
    直到 MySQL 报 `Too many connections`。

    队列池把「池里有多少连接」变成一个可控数字，且 `close()` 能真正关干净。

    ## 为什么每次都 ping

    MySQL 的 `wait_timeout`（默认 8 小时）会悄悄掐掉空闲连接。
    池子里握着一个已断的连接，下次使用时报的是
    `(2006, 'MySQL server has gone away')` —— 而调用方看到的是「查询失败」，
    不是「连接过期」，排查方向会完全跑偏。`ping(reconnect=True)` 把这个
    状态在当前这一层修掉。
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 3306,
        user: str,
        password: str,
        database: str,
        charset: str = "utf8mb4",
        size: int = 8,
        acquire_timeout_s: float = 10.0,
        connect_timeout_s: float = 10.0,
    ) -> None:
        import pymysql  # 延迟导入：未装 MySQL 驱动时不应影响其他后端

        self._pymysql = pymysql
        # 这里不再做 `int(port)` 之类的防御性转换：
        # 参数的类型标注已经声明为 int，而 `build_store_from_env` 也用
        # `_int_env()` 校验过了 —— 重复转换只是把「调用方传错了」
        # 藏进一个看起来更宽容的构造函数里。
        self._connect_kwargs: dict[str, Any] = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "charset": charset,
            # DictCursor 是**必需**的：行映射全部按列名取值。
            # 用默认的元组游标会让 _row_field 抛错，
            # 而不是返回一个“看起来能用但字段全错位”的对象。
            "cursorclass": pymysql.cursors.DictCursor,
            "autocommit": False,
            "connect_timeout": connect_timeout_s,
            "read_timeout": 30,
            "write_timeout": 30,
        }
        self._size = max(1, size)
        self._acquire_timeout_s = acquire_timeout_s
        self._pool: list[Any] = []
        self._idle = threading.Semaphore(0)
        self._lock = threading.Lock()
        self._created = 0
        self._closed = False

    @property
    def descriptor(self) -> str:
        """给日志/健康检查用的描述。**不含密码。**"""
        k = self._connect_kwargs
        return f"mysql://{k['user']}@{k['host']}:{k['port']}/{k['database']}"

    def _new_connection(self) -> Any:
        """建一个新连接。

        ⚠️ **调用方必须已经占好名额**（`_created` 已经加过）。

        刻意不让本方法自己加锁计数：初版是那样写的，而
        `_checkout` 在持有 `self._lock` 的情况下调它 ——
        `threading.Lock` **不可重入**，于是第一次取连接就死锁。
        而它的表现是「进程挂住不报错」，非常难定向。
        把「记账」与「建连接」分开之后，锁的持有范围就只有几行。
        """
        return self._pymysql.connect(**self._connect_kwargs)

    def _reserve_slot(self) -> bool:
        """尝试占用一个连接名额。返回是否成功。**不建连接**，不持锁向外调用。"""
        with self._lock:
            if self._created >= self._size:
                return False
            self._created += 1
            return True

    def _release_slot(self) -> None:
        with self._lock:
            self._created -= 1

    def _connect_or_release(self) -> Any:
        """建连接；失败则退还名额（否则名额泄漏，池会慢慢枯竭）。"""
        try:
            return self._new_connection()
        except Exception:
            self._release_slot()
            raise

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """借出一个连接，作用域结束自动归还。

        **成功提交、异常回滚** —— 调用方不需要自己写 commit/rollback，
        写一次总会有人忘，而「忘了 rollback」会把这个连接带着未结束的事务
        还回池里，污染下一个使用者。
        """
        if self._closed:
            raise RuntimeError("连接池已关闭")

        conn = self._checkout()
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception as rb_exc:  # noqa: BLE001
                # **必须留痕。** 回滚失败意味着这个连接可能带着未结束的事务
                # 被还回池里 —— 下一个使用者会看到脏数据，而那时完全看不出
                # 根因在哪。不在这里说，那件事就永远不会被知道。
                #
                # 但仍然 `raise` 原始异常：用 rollback 的失败覆盖它，
                # 会把「查询为什么错」换成「回滚为什么错」，后者是次要信息。
                logger.warning("回滚失败（将保留原始异常）：%s", rb_exc)
            raise
        finally:
            self._checkin(conn)

    def _checkout(self) -> Any:
        """取一个可用连接。三级策略：空闲 → 新建 → 等待。

        **每一级都不在持锁时做 I/O** —— 建连接与 ping 都可能阻塞，
        握着锁做它们会把池变成串行瓶颈（而症状是「偶尔很慢」，更难查）。
        """
        # 1) 有空闲连接
        if self._idle.acquire(timeout=0):
            with self._lock:
                conn = self._pool.pop() if self._pool else None
            if conn is not None:
                return self._revive(conn)
            # 信号量与队列不一致（理论上不会发生）：退还 permit 后走下一级
            self._idle.release()

        # 2) 未达上限：新建
        if self._reserve_slot():
            return self._connect_or_release()

        # 3) 已达上限：等归还
        if not self._idle.acquire(timeout=self._acquire_timeout_s):
            raise TimeoutError(
                f"等待数据库连接超时（{self._acquire_timeout_s}s，池大小 {self._size}）。\n"
                f"常见原因：有代码借了连接却没归还，或慢查询占满了池。"
            )
        with self._lock:
            conn = self._pool.pop() if self._pool else None
        if conn is None:
            self._idle.release()
            raise RuntimeError("连接池状态异常：信号量与队列不一致")
        return self._revive(conn)

    def _revive(self, conn: Any) -> Any:
        """确保借出的连接是活的。失败则重建（名额不变，因为旧的废了）。

        ⚠️ **不用 `ping(reconnect=True)`** —— PyMySQL 2.x 弃用了那个参数
        （会有 DeprecationWarning，而且它把「重连」藏在一个返回值里）。
        自己先 ping 一次、失败再重建，行为更显式，也不依赖已弃用的路径。
        """
        try:
            conn.ping()  # 不传 reconnect：连接死了就抛，由下面处理
            return conn
        except Exception:  # noqa: BLE001
            try:
                conn.close()
            except Exception as close_exc:  # noqa: BLE001
                # 关一个已经死掉的连接失败，不影响重建 —— 但记下来，
                # 因为「关不掉」持续出现通常意味着文件描述符在泄漏。
                logger.debug("关闭失效连接时出错：%s", close_exc)
            self._release_slot()
            if self._reserve_slot():
                return self._connect_or_release()
            raise RuntimeError("连接失效后无法重建：池名额已满") from None

    def _checkin(self, conn: Any) -> None:
        with self._lock:
            if self._closed:
                try:
                    conn.close()
                finally:
                    self._created -= 1
                return
            self._pool.append(conn)
        self._idle.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conns = list(self._pool)
            self._pool.clear()
            self._created -= len(conns)
        for conn in conns:
            try:
                conn.close()
            except Exception as close_exc:  # noqa: BLE001
                # 显式关闭失败意味着这连接可能还开着 —— 那是一个真实的
                # 资源泄漏，不是「清理时的小插曲」。所以要说出来。
                logger.warning("关闭连接失败（可能泄漏）：%s", close_exc)


# =============================================================================
# 类型转换（读写两端的唯一收口点）
# =============================================================================


def _to_db(dt: datetime | None) -> datetime | None:
    """aware → naive UTC。MySQL DATETIME 不带时区，存进去的必须是 UTC。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # 已经是 naive：按 UTC 解释。**不猜本地时区** —— 猜错会静默偏移。
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _from_db(dt: datetime | None) -> datetime | None:
    """naive UTC → aware UTC。"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _pack_vector(vector: Sequence[float] | None) -> bytes | None:
    """float32 打包。用 binary 而不是 JSON：4KB vs 约 20KB，且无需解析。"""
    if vector is None:
        return None
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack_vector(blob: Any) -> list[float] | None:
    if not blob:
        return None
    raw = bytes(blob)
    if len(raw) % 4 != 0:
        raise ValueError(f"向量字节数不是 4 的倍数：{len(raw)}")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_load(raw: Any, default: Any) -> Any:
    """读 JSON 列。**损坏时抛错而不是回退默认值。**

    回退默认值会把「数据坏了」表现成「这个字段是空的」——
    例如 `supersedes` 读成 `[]` 会让一条本应被取代的记忆重新生效。
    """
    if raw is None:
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # **记一笔再抛，而不是吞掉。**
        #
        # 本函数的契约就是「JSON 坏了要抛」：把 DecodeError 吞成默认值
        # 会让「数据损坏」表现成「字段为空」，而后者在业务上是合法的
        # （比如 `supersedes` 真的可以为空）——两者一旦不可区分，
        # 一条本应被取代的记忆会重新生效。
        #
        # 而「直接抛、什么都不说」也是不够的：数据库里有一列解析不了时，
        # 调用方看到的是 `JSONDecodeError`，看不出**是哪一列、什么内容**。
        logger.error("JSON 列损坏：%r", raw[:120] if isinstance(raw, str) else raw)
        raise exc


def _model_dump_list(items: Sequence[Any]) -> str:
    """把 Pydantic 模型列表转成可入库的 JSON。"""
    out = []
    for item in items:
        if hasattr(item, "model_dump"):
            out.append(item.model_dump(mode="json"))
        else:
            out.append(item)
    return _json_dump(out)


# =============================================================================
# 租户范围内的 SQL 清单
#
# ⚠️ 每一条都**必须**同时含 user_id 与 pet_id。
#     tests/test_store_sql.py 会逐条断言这一点 —— 新加查询忘了过滤会红。
# =============================================================================

_TENANT_SCOPED_SQL: dict[str, str] = {
    "get_pet": """
        SELECT * FROM pets WHERE pet_id = %s AND user_id = %s
    """,
    "get_memory": """
        SELECT * FROM memories
        WHERE memory_id = %s AND user_id = %s AND pet_id = %s
    """,
    "find_by_dedup_key": """
        SELECT * FROM memories
        WHERE dedup_key = %s AND user_id = %s AND pet_id = %s
    """,
    "find_conflicts": """
        SELECT * FROM memories
        WHERE user_id = %s AND pet_id = %s
          AND subject = %s AND status = 'active' AND polarity <> %s
    """,
    "list_memories": """
        SELECT * FROM memories
        WHERE user_id = %s AND pet_id = %s
    """,
    "list_meow_records": """
        SELECT * FROM meow_records
        WHERE user_id = %s AND pet_id = %s
    """,
    "get_pending_interpretation": """
        SELECT * FROM pending_interpretations
        WHERE interpretation_id = %s AND user_id = %s AND pet_id = %s
    """,
    "list_messages": """
        SELECT * FROM session_messages
        WHERE user_id = %s AND pet_id = %s
    """,
    "list_recent_messages": """
        SELECT * FROM session_messages
        WHERE user_id = %s AND pet_id = %s
    """,
    "hydrate_memories": """
        SELECT * FROM memories
        WHERE memory_id IN ({placeholders}) AND user_id = %s AND pet_id = %s
    """,
    "delete_pet_memories": """
        DELETE FROM memories WHERE user_id = %s AND pet_id = %s
    """,
    "delete_pet_meow_records": """
        DELETE FROM meow_records WHERE user_id = %s AND pet_id = %s
    """,
    "delete_pet_messages": """
        DELETE FROM session_messages WHERE user_id = %s AND pet_id = %s
    """,
    "delete_pet_pending": """
        DELETE FROM pending_interpretations WHERE user_id = %s AND pet_id = %s
    """,
    "delete_pet": """
        DELETE FROM pets WHERE pet_id = %s AND user_id = %s
    """,
    "list_moments": """
        SELECT * FROM moments
        WHERE user_id = %s AND pet_id = %s
    """,
    "delete_pet_moments": """
        DELETE FROM moments WHERE user_id = %s AND pet_id = %s
    """,
    "select_vectors": """
        SELECT memory_id, user_id, pet_id, vector FROM memories
        WHERE vector IS NOT NULL AND user_id = %s AND pet_id = %s
    """,
}


class MySQLStore(MemoryStore):
    """`MemoryStore` 的 MySQL 实现。

    向量检索委托给 `VectorIndex`（Milvus 或内存）。
    **MySQL 始终是真相**：即使索引给出 id，回表时仍会再按租户过滤一次。
    """

    def __init__(
        self,
        *,
        pool: MySQLPool,
        vector_index: VectorIndex,
        dim: int = 1024,
    ) -> None:
        self._pool = pool
        self._index = vector_index
        self._dim = dim
        # 索引写失败的最近一次原因。**不静默** —— 由 describe() 暴露。
        self._index_error: str | None = None
        self._index_lag = 0
        self._meta_lock = threading.Lock()

    # ── 索引健全性 ────────────────────────────────────────

    @property
    def index_error(self) -> str | None:
        return self._index_error

    @property
    def index_lag(self) -> int:
        """自上次成功写入索引以来，有多少条记忆没能进索引。"""
        return self._index_lag

    def describe(self) -> dict[str, str]:
        """给 `/healthz` 用。"""
        out = {"kind": "mysql", "dsn": self._pool.descriptor}
        out.update({f"vector.{k}": v for k, v in self._index.describe().items()})
        if self._index_lag:
            out["vector.lag"] = str(self._index_lag)
        if self._index_error:
            out["vector.error"] = self._index_error[:200]
        return out

    def _index_write(self, entries: Sequence[VectorEntry]) -> None:
        """写索引。**失败不抛出，但记账。**

        取舍：MySQL 是真相，且向量也持久化了 —— 所以索引写失败
        **不构成数据丢失**，可以靠 `backfill_vectors()` 补。
        为此让用户的写入失败是不划算的（他会以为没记下来）。

        但「不抛出」不等于「不说」：`index_lag` 与 `index_error`
        会出现在 `/healthz` 里。静默吞掉才是不可接受的。
        """
        if not entries:
            return
        try:
            self._index.upsert(entries)
            with self._meta_lock:
                self._index_error = None
                self._index_lag = max(0, self._index_lag - len(entries))
        except VectorIndexUnavailable as exc:
            with self._meta_lock:
                self._index_error = str(exc)
                self._index_lag += len(entries)
            logger.warning("向量索引写入失败，已记账待回填：%s", exc)

    def _index_delete(self, memory_ids: Sequence[str]) -> None:
        try:
            self._index.delete(list(memory_ids))
        except VectorIndexUnavailable as exc:
            with self._meta_lock:
                self._index_error = str(exc)
            logger.warning("向量索引删除失败：%s", exc)

    def backfill_vectors(self) -> int:
        """从 MySQL 把向量回填进索引。**幂等**（索引的 upsert 是覆盖语义）。

        用途：Milvus 重建 / 索引落后 / 首次接入 Milvus。
        返回回填条数。**逐租户分页**，避免一次把全库向量读进内存。
        """
        total = 0
        with self._pool.acquire() as conn, conn.cursor() as cur:
            # 先取租户清单，再逐租户回填 —— 这样查询始终带租户过滤，
            # 与 _TENANT_SCOPED_SQL 的约定一致（也便于按租户限流）
            cur.execute(
                "SELECT DISTINCT user_id, pet_id FROM memories WHERE vector IS NOT NULL"
            )
            tenants = cur.fetchall()

        for row in tenants:
            user_id = row["user_id"] if isinstance(row, dict) else row[0]
            pet_id = row["pet_id"] if isinstance(row, dict) else row[1]
            total += self._backfill_tenant(user_id=user_id, pet_id=pet_id)
        return total

    def _backfill_tenant(self, *, user_id: str, pet_id: str) -> int:
        written = 0
        last_id = ""
        batch = 500
        while True:
            with self._pool.acquire() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT memory_id, vector FROM memories "
                    "WHERE vector IS NOT NULL AND user_id = %s AND pet_id = %s "
                    "AND memory_id > %s ORDER BY memory_id LIMIT %s",
                    (user_id, pet_id, last_id, batch),
                )
                rows = list(cur.fetchall())
            if not rows:
                return written

            entries: list[VectorEntry] = []
            for row in rows:
                mid = row["memory_id"] if isinstance(row, dict) else row[0]
                blob = row["vector"] if isinstance(row, dict) else row[1]
                vec = _unpack_vector(blob)
                if vec is None:
                    continue
                entries.append(
                    VectorEntry(
                        memory_id=mid, user_id=user_id, pet_id=pet_id, vector=vec
                    )
                )
                last_id = mid

            if entries:
                # 回填是显式运维动作，失败必须抛出 —— 与后台记账不同，
                # 这里调用方正在等结果，静默返回 0 会让他以为回填成功了。
                self._index.upsert(entries)
                written += len(entries)

    # ── 档案 ──────────────────────────────────────────────

    def save_pet(self, pet: PetProfile) -> PetProfile:
        sql = """
            INSERT INTO pets (
                pet_id, user_id, name, species, breed, visual,
                must_keep_features, observed_but_unstable, traits,
                appearance_embedding_id, identity_prompt, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                name = VALUES(name), species = VALUES(species), breed = VALUES(breed),
                visual = VALUES(visual),
                must_keep_features = VALUES(must_keep_features),
                observed_but_unstable = VALUES(observed_but_unstable),
                traits = VALUES(traits),
                appearance_embedding_id = VALUES(appearance_embedding_id),
                identity_prompt = VALUES(identity_prompt),
                updated_at = VALUES(updated_at)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    pet.pet_id,
                    pet.user_id,
                    pet.name,
                    _enum_value(pet.species),
                    pet.breed,
                    _json_dump(pet.visual.model_dump(mode="json")),
                    _json_dump(list(pet.must_keep_features)),
                    _json_dump(list(pet.observed_but_unstable)),
                    _model_dump_list(pet.traits),
                    pet.appearance_embedding_id,
                    pet.identity_prompt,
                    _to_db(pet.created_at),
                    _to_db(pet.updated_at),
                ),
            )
        return pet

    def get_pet(self, *, user_id: str, pet_id: str) -> PetProfile:
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(_TENANT_SCOPED_SQL["get_pet"], (pet_id, user_id))
            row = cur.fetchone()
        # 归属不符与不存在返回同一个错误 —— 不泄露资源存在性（§4.2 A2）
        if row is None:
            raise NotFound(f"pet {pet_id} 不存在")
        return _row_to_pet(row)

    def list_pets(self, *, user_id: str) -> list[PetProfile]:
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM pets WHERE user_id = %s ORDER BY created_at",
                (user_id,),
            )
            rows = list(cur.fetchall())
        return [_row_to_pet(r) for r in rows]

    def delete_pet_data(self, *, user_id: str, pet_id: str) -> int:
        """**级联硬删**该宠物的全部数据，返回删除条数。

        与健康数据同理（`HealthDataPolicy`）：软标记不满足删除要求。
        向量也要一起清 —— 只删 MySQL 会让索引里留下孤儿条目，
        而孤儿条目仍会被检索命中，于是「已删除的数据」还能影响结果。

        ⚠️ **包括宠物档案行本身。**
        初版漏了 `pets`，于是「删掉这只猫」之后它还留在列表里 ——
        而用户看到的「删除成功」与实际不符。
        评测的收尾清理最先撞上这个：`pets` 表里全是残留。
        """
        removed = 0
        with self._pool.acquire() as conn, conn.cursor() as cur:
            for key in (
                "delete_pet_memories",
                "delete_pet_meow_records",
                "delete_pet_messages",
                "delete_pet_pending",
                "delete_pet_moments",
            ):
                cur.execute(_TENANT_SCOPED_SQL[key], (user_id, pet_id))
                removed += cur.rowcount or 0
            # 档案行最后删（其余表逻辑上都挂在它下面）
            cur.execute(
                "DELETE FROM pets WHERE pet_id = %s AND user_id = %s",
                (pet_id, user_id),
            )
            removed += cur.rowcount or 0

        try:
            index = self._index
            deleter = getattr(index, "delete_by_tenant", None)
            if callable(deleter):
                deleter(user_id=user_id, pet_id=pet_id)
        except VectorIndexUnavailable as exc:
            with self._meta_lock:
                self._index_error = str(exc)
            logger.warning("删除宠物时向量清理失败：%s", exc)

        return removed

    # ── 瞬间（日记本体） ─────────────────────────────

    def insert_moment(self, moment: Moment) -> Moment:
        moment_id = moment.moment_id or str(uuid4())
        stored = moment.model_copy(update={"moment_id": moment_id})
        sql = """
            INSERT INTO moments (
                moment_id, user_id, pet_id, session_id, media_url,
                note, scene, captured_at, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    moment_id,
                    stored.user_id,
                    stored.pet_id,
                    stored.session_id,
                    stored.media_url,
                    stored.note,
                    _enum_value(stored.scene),
                    _to_db(stored.captured_at),
                    _to_db(stored.created_at),
                ),
            )
        return stored

    def list_moments(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> list[Moment]:
        """按时间**倒序**取（最新在前）。

        与 `list_recent_messages` 的「先倒序取再反转」不同 ——
        时间线**要的就是倒序**，不需要反转回来。
        """
        sql = _TENANT_SCOPED_SQL["list_moments"]
        params: list[Any] = [user_id, pet_id]
        if since is not None:
            sql += " AND captured_at >= %s"
            params.append(_to_db(since))
        if until is not None:
            sql += " AND captured_at < %s"
            params.append(_to_db(until))
        sql += " ORDER BY captured_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_moment(r) for r in rows]

    # ── 记忆 ──────────────────────────────────────────────

    def insert_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent:
        memory_id = event.memory_id or str(uuid4())
        if event.memory_id is None:
            event = event.model_copy(update={"memory_id": memory_id})

        sql = """
            INSERT INTO memories (
                memory_id, user_id, pet_id, session_id, layer, event_type,
                subject, content, polarity, valid_from, valid_to, occurred_at,
                source, confidence, status, support_count, last_seen_at,
                superseded_by, supersedes, dedup_key, vector, created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self._pool.acquire() as conn, conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        memory_id,
                        event.user_id,
                        event.pet_id,
                        event.session_id,
                        _enum_value(event.layer),
                        _enum_value(event.event_type),
                        event.subject,
                        event.content,
                        _enum_value(event.polarity),
                        _to_db(event.valid_from),
                        _to_db(event.valid_to),
                        _to_db(event.occurred_at),
                        _enum_value(event.source),
                        event.confidence,
                        _enum_value(event.status),
                        event.support_count,
                        _to_db(event.last_seen_at),
                        event.superseded_by,
                        _json_dump(list(event.supersedes)),
                        event.dedup_key,
                        _pack_vector(vector),
                        _to_db(event.created_at),
                        _to_db(event.updated_at),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            if _is_duplicate_key(exc):
                raise DuplicateMemory(f"dedup_key={event.dedup_key} 已存在") from exc
            raise

        if vector is not None:
            self._index_write(
                [
                    VectorEntry(
                        memory_id=memory_id,
                        user_id=event.user_id,
                        pet_id=event.pet_id,
                        vector=vector,
                    )
                ]
            )
        return event

    def update_memory(
        self, event: MemoryEvent, *, vector: list[float] | None = None
    ) -> MemoryEvent:
        if event.memory_id is None:
            raise NotFound("memory_id 为空，无法更新")

        # 先确认存在且属于该租户 —— MySQL 的 UPDATE 影响 0 行无法区分
        # 「不存在」与「值没变」，而这两种情况的调用方语义完全不同。
        self.get_memory(
            user_id=event.user_id, pet_id=event.pet_id, memory_id=event.memory_id
        )

        sql = """
            UPDATE memories SET
                session_id = %s, layer = %s, event_type = %s, subject = %s,
                content = %s, polarity = %s, valid_from = %s, valid_to = %s,
                occurred_at = %s, source = %s, confidence = %s, status = %s,
                support_count = %s, last_seen_at = %s, superseded_by = %s,
                supersedes = %s, dedup_key = %s, updated_at = %s
            WHERE memory_id = %s AND user_id = %s AND pet_id = %s
        """
        params: list[Any] = [
            event.session_id,
            _enum_value(event.layer),
            _enum_value(event.event_type),
            event.subject,
            event.content,
            _enum_value(event.polarity),
            _to_db(event.valid_from),
            _to_db(event.valid_to),
            _to_db(event.occurred_at),
            _enum_value(event.source),
            event.confidence,
            _enum_value(event.status),
            event.support_count,
            _to_db(event.last_seen_at),
            event.superseded_by,
            _json_dump(list(event.supersedes)),
            event.dedup_key,
            _to_db(event.updated_at),
            event.memory_id,
            event.user_id,
            event.pet_id,
        ]
        if vector is not None:
            sql = sql.replace(
                "dedup_key = %s, updated_at = %s",
                "dedup_key = %s, vector = %s, updated_at = %s",
            )
            params.insert(-4, _pack_vector(vector))

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))

        if vector is not None:
            self._index_write(
                [
                    VectorEntry(
                        memory_id=event.memory_id,
                        user_id=event.user_id,
                        pet_id=event.pet_id,
                        vector=vector,
                    )
                ]
            )
        return event

    def get_memory(self, *, user_id: str, pet_id: str, memory_id: str) -> MemoryEvent:
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(_TENANT_SCOPED_SQL["get_memory"], (memory_id, user_id, pet_id))
            row = cur.fetchone()
        if row is None:
            raise NotFound(f"memory {memory_id} 不存在")
        return _row_to_memory(row)

    def find_by_dedup_key(
        self, *, user_id: str, pet_id: str, dedup_key: str
    ) -> MemoryEvent | None:
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                _TENANT_SCOPED_SQL["find_by_dedup_key"],
                (dedup_key, user_id, pet_id),
            )
            row = cur.fetchone()
        return _row_to_memory(row) if row else None

    def find_conflicts(
        self, *, user_id: str, pet_id: str, candidate: MemoryEvent
    ) -> list[MemoryEvent]:
        """冲突判定委托给契约层的 `conflicts_with`。

        SQL 只做粗筛（同主体 + 反极性 + active），
        精确判定（时间范围重叠）仍在契约层 —— 保证各存储实现行为一致。
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                _TENANT_SCOPED_SQL["find_conflicts"],
                (user_id, pet_id, candidate.subject, _enum_value(candidate.polarity)),
            )
            rows = list(cur.fetchall())

        out: list[MemoryEvent] = []
        for row in rows:
            ev = _row_to_memory(row)
            if ev.memory_id == candidate.memory_id:
                continue
            if candidate.conflicts_with(ev):
                out.append(ev)
        return out

    def list_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        include_non_active: bool = False,
        session_id: str | None = None,
    ) -> list[MemoryEvent]:
        sql = _TENANT_SCOPED_SQL["list_memories"]
        params: list[Any] = [user_id, pet_id]
        if not include_non_active:
            # `is_retrievable_by_default` 的 SQL 等价物。
            # 刻意写在这里而不是让调用方传状态列表：默认值必须与内存实现一致。
            sql += " AND status IN ('active', 'pending_confirmation')"
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(session_id)
        sql += " ORDER BY created_at"

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_memory(r) for r in rows]

    # ── 叫声记录 ──────────────────────────────────────────

    def insert_meow_record(self, record: MeowRecord) -> MeowRecord:
        record_id = record.record_id or str(uuid4())
        stored = record.model_copy(update={"record_id": record_id})
        sql = """
            INSERT INTO meow_records (
                record_id, user_id, pet_id, session_id, context, features,
                actions, resolution, recorded_at, source, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    record_id,
                    stored.user_id,
                    stored.pet_id,
                    stored.session_id,
                    _enum_value(stored.context),
                    _json_dump(stored.features.model_dump(mode="json")),
                    _json_dump([_enum_value(a) for a in stored.actions]),
                    stored.resolution,
                    _to_db(stored.recorded_at),
                    _enum_value(stored.source),
                    _enum_value(stored.status),
                ),
            )
        return stored

    def list_meow_records(
        self,
        *,
        user_id: str,
        pet_id: str,
        only_confirmed: bool = True,
        session_id: str | None = None,
    ) -> list[MeowRecord]:
        sql = _TENANT_SCOPED_SQL["list_meow_records"]
        params: list[Any] = [user_id, pet_id]
        if only_confirmed:
            # 未确认的样本不构成证据 —— 与内存实现一致
            sql += " AND status = 'active'"
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(session_id)
        sql += " ORDER BY recorded_at"

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_meow(r) for r in rows]

    # ── 待标注解释 ────────────────────────────────────────

    def save_pending_interpretation(
        self, pending: PendingInterpretation
    ) -> PendingInterpretation:
        sql = """
            INSERT INTO pending_interpretations (
                interpretation_id, user_id, pet_id, session_id, features,
                evidence_mode, candidates, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                features = VALUES(features),
                evidence_mode = VALUES(evidence_mode),
                candidates = VALUES(candidates)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    pending.interpretation_id,
                    pending.user_id,
                    pending.pet_id,
                    pending.session_id,
                    _json_dump(pending.features.model_dump(mode="json")),
                    _enum_value(pending.evidence_mode),
                    _model_dump_list(pending.candidates),
                    _to_db(pending.created_at),
                ),
            )
        return pending

    def get_pending_interpretation(
        self, *, user_id: str, pet_id: str, interpretation_id: str
    ) -> PendingInterpretation | None:
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                _TENANT_SCOPED_SQL["get_pending_interpretation"],
                (interpretation_id, user_id, pet_id),
            )
            row = cur.fetchone()
        # 归属不符返回 None 而非抛错 —— 与 NotFound 同理，不泄露存在性
        return _row_to_pending(row) if row else None

    # ── 会话消息 ──────────────────────────────────────────

    def insert_message(self, message: SessionMessage) -> SessionMessage:
        stored = message.with_id()
        sql = """
            INSERT INTO session_messages (
                message_id, user_id, pet_id, session_id, role, content, at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    stored.message_id,
                    stored.user_id,
                    stored.pet_id,
                    stored.session_id,
                    stored.role,
                    stored.content,
                    _to_db(stored.at),
                ),
            )
        return stored

    def list_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[SessionMessage]:
        sql = _TENANT_SCOPED_SQL["list_messages"]
        params: list[Any] = [user_id, pet_id]
        if since is not None:
            sql += " AND at >= %s"
            params.append(_to_db(since))
        if until is not None:
            sql += " AND at < %s"
            params.append(_to_db(until))
        sql += " ORDER BY at"

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_message(r) for r in rows]

    def list_recent_messages(
        self,
        *,
        user_id: str,
        pet_id: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[SessionMessage]:
        """取最近 `limit` 条，**按时间正序返回**。

        先按时间倒序取 N 条再反转 —— 直接正序取会拿到最旧的 N 条，
        而那是完全不同的一件事（而且是静默的：条数对得上）。
        """
        if limit <= 0:
            return []
        sql = _TENANT_SCOPED_SQL["list_recent_messages"]
        params: list[Any] = [user_id, pet_id]
        if session_id is not None:
            sql += " AND session_id = %s"
            params.append(session_id)
        sql += " ORDER BY at DESC LIMIT %s"
        params.append(limit)

        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = list(cur.fetchall())
        return [_row_to_message(r) for r in reversed(rows)]

    # ── 向量召回 ──────────────────────────────────────────

    def search_memories(
        self,
        *,
        user_id: str,
        pet_id: str,
        query_vector: list[float],
        limit: int = 20,
    ) -> list[MemoryItem]:
        """向量召回。

        Raises:
            VectorIndexUnavailable: 索引不可用。**刻意不返回空列表** ——
                空结果与「确实没有相关记忆」在调用方看来一样，
                而那会让系统自信地说出「没有相关记录」（见 vectors.py）。
        """
        hits = self._index.search(
            user_id=user_id, pet_id=pet_id, vector=query_vector, limit=limit
        )
        if not hits:
            return []

        ids = [h.memory_id for h in hits]
        placeholders = ", ".join(["%s"] * len(ids))

        # 回表时**再过滤一次租户**。索引已经过滤过了，这里是第二道防线：
        # 索引配错（比如 collection 复用、filter 表达式写漏）时，
        # 这一层仍然挡得住，而代价只是一次主键查询。
        sql = _TENANT_SCOPED_SQL["hydrate_memories"].format(placeholders=placeholders)
        with self._pool.acquire() as conn, conn.cursor() as cur:
            cur.execute(sql, (*ids, user_id, pet_id))
            rows = list(cur.fetchall())

        by_id = {_row_field(r, "memory_id"): r for r in rows}

        out: list[MemoryItem] = []
        for hit in hits:
            row = by_id.get(hit.memory_id)
            if row is None:
                # 索引有、MySQL 没有 → 孤儿条目（删除时索引清理失败）。
                # 跳过是对的，但**要留痕**：持续出现说明删除路径有问题。
                logger.warning(
                    "向量索引存在孤儿条目 memory_id=%s（MySQL 中不存在或不属于该租户）",
                    hit.memory_id,
                )
                continue
            event = _row_to_memory(row)
            if not event.is_retrievable_by_default:
                continue
            out.append(
                MemoryItem(
                    event=event,
                    score=max(0.0, hit.score),
                    retrieval_source=RetrievalSource.VECTOR,
                    matched_on="embedding",
                )
            )
        return out


# =============================================================================
# 行映射
# =============================================================================


def _enum_value(value: Any) -> Any:
    """枚举 → 字符串。非枚举原样返回。"""
    return value.value if hasattr(value, "value") else value


def _row_field(row: Any, name: str) -> Any:
    """取列值。DictCursor 与普通 Cursor 都兼容。"""
    if isinstance(row, dict):
        return row.get(name)
    raise TypeError(
        "需要 DictCursor 才能按列名取值。请用 cursor(pymysql.cursors.DictCursor) 建游标。"
    )


def _is_duplicate_key(exc: Exception) -> bool:
    """判断是否是唯一键冲突。

    PyMySQL 的 `IntegrityError` 没有结构化错误码，只能看 args[0]（errno）。
    1062 = ER_DUP_ENTRY。
    """
    args = getattr(exc, "args", ())
    return bool(args) and args[0] == 1062


def _as_number(value: Any, *, field: str, row_hint: Any) -> float:
    """把数据库列值转成数字。**转换失败时记一笔再抛。**

    为什么不吞成 0.0：这一列的语义是「置信度」。一个读不出来的置信度
    如果静默变成 0.0，那条记忆会从「确定」变成「完全不可信」，
    而下游（排序、晋升）看到的是一个合法的数字 —— 没人会发现。

    为什么也不"直接抛、什么都不说"：数据库里出现非数字时，
    调用方看到的是 `ValueError`，看不出**是哪一列、哪一行**。
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        logger.error("列 %s 无法转成数字（row=%s）：%r", field, row_hint, value)
        raise exc


def _as_int(value: Any, *, field: str, row_hint: Any, default: int | None = None) -> int:
    """把数据库列值转成整数。`default` 只对 **NULL/空值** 生效。

    区分「NULL」（允许回退默认值）与「非数字」（必须抛）——
    把后者也回退，一条损坏的行会变成一个看起来正常的数字。
    """
    if value is None or value == "":
        if default is None:
            logger.error("列 %s 为空且无默认值（row=%s）", field, row_hint)
            raise ValueError(f"列 {field} 为空且无默认值")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        logger.error("列 %s 无法转成整数（row=%s）：%r", field, row_hint, value)
        raise exc


def _row_to_pet(row: Any) -> PetProfile:
    traits = [
        Trait.model_validate(t) if isinstance(t, dict) else t
        for t in _json_load(_row_field(row, "traits"), [])
    ]
    return PetProfile(
        pet_id=_row_field(row, "pet_id"),
        user_id=_row_field(row, "user_id"),
        name=_row_field(row, "name"),
        species=Species(_row_field(row, "species")),
        breed=_row_field(row, "breed"),
        visual=VisualProfile.model_validate(_json_load(_row_field(row, "visual"), {})),
        must_keep_features=list(_json_load(_row_field(row, "must_keep_features"), [])),
        observed_but_unstable=list(
            _json_load(_row_field(row, "observed_but_unstable"), [])
        ),
        traits=traits,
        appearance_embedding_id=_row_field(row, "appearance_embedding_id"),
        identity_prompt=_row_field(row, "identity_prompt"),
        created_at=_from_db(_row_field(row, "created_at")),
        updated_at=_from_db(_row_field(row, "updated_at")),
    )


def _row_to_memory(row: Any) -> MemoryEvent:
    return MemoryEvent(
        memory_id=_row_field(row, "memory_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        layer=_row_field(row, "layer"),
        event_type=_row_field(row, "event_type"),
        subject=_row_field(row, "subject"),
        content=_row_field(row, "content"),
        polarity=_row_field(row, "polarity"),
        valid_from=_from_db(_row_field(row, "valid_from")),
        valid_to=_from_db(_row_field(row, "valid_to")),
        occurred_at=_from_db(_row_field(row, "occurred_at")),
        source=MemorySource(_row_field(row, "source")),
        confidence=_as_number(
            _row_field(row, "confidence"),
            field="confidence",
            row_hint=_row_field(row, "memory_id"),
        ),
        status=MemoryStatus(_row_field(row, "status")),
        support_count=_as_int(
            _row_field(row, "support_count"),
            field="support_count",
            row_hint=_row_field(row, "memory_id"),
            default=1,
        ),
        last_seen_at=_from_db(_row_field(row, "last_seen_at")),
        superseded_by=_row_field(row, "superseded_by"),
        supersedes=list(_json_load(_row_field(row, "supersedes"), [])),
        dedup_key=_row_field(row, "dedup_key"),
        created_at=_from_db(_row_field(row, "created_at")),
        updated_at=_from_db(_row_field(row, "updated_at")),
    )


def _row_to_moment(row: Any) -> Moment:
    return Moment(
        moment_id=_row_field(row, "moment_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        media_url=_row_field(row, "media_url"),
        note=_row_field(row, "note"),
        scene=MomentScene(_row_field(row, "scene")),
        captured_at=_from_db(_row_field(row, "captured_at")),
        created_at=_from_db(_row_field(row, "created_at")),
    )


def _row_to_meow(row: Any) -> MeowRecord:
    return MeowRecord(
        record_id=_row_field(row, "record_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        context=ContextLabel(_row_field(row, "context")),
        features=AcousticFeatures.model_validate(
            _json_load(_row_field(row, "features"), {})
        ),
        actions=[BehaviorAction(a) for a in _json_load(_row_field(row, "actions"), [])],
        resolution=_row_field(row, "resolution"),
        recorded_at=_from_db(_row_field(row, "recorded_at")),
        source=MemorySource(_row_field(row, "source")),
        status=MemoryStatus(_row_field(row, "status")),
    )


def _row_to_pending(row: Any) -> PendingInterpretation:
    return PendingInterpretation(
        interpretation_id=_row_field(row, "interpretation_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        features=AcousticFeatures.model_validate(
            _json_load(_row_field(row, "features"), {})
        ),
        evidence_mode=EvidenceMode(_row_field(row, "evidence_mode")),
        candidates=[
            IntentCandidate.model_validate(c)
            for c in _json_load(_row_field(row, "candidates"), [])
        ],
        created_at=_from_db(_row_field(row, "created_at")),
    )


def _row_to_message(row: Any) -> SessionMessage:
    return SessionMessage(
        message_id=_row_field(row, "message_id"),
        user_id=_row_field(row, "user_id"),
        pet_id=_row_field(row, "pet_id"),
        session_id=_row_field(row, "session_id"),
        role=_row_field(row, "role"),
        content=_row_field(row, "content"),
        at=_from_db(_row_field(row, "at")),
    )
