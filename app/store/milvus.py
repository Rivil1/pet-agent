"""Milvus 向量索引。

对应决策：**MySQL 为准 + Milvus 作可重建索引**。
这里只存 `(memory_id, user_id, pet_id, vector)` —— 正文一律在 MySQL。
于是 Milvus 丢了不是数据丢失，跑一次回填即可恢复（不需要重新调 embedding API，
因为向量本身也持久化在 MySQL 里）。

## ⚠️ 这个文件里最要紧的一件事：表达式转义

Milvus 的标量过滤是一个**字符串表达式**：

    user_id == "cat-mom" and pet_id == "0189e64d-..."

而 `user_id` 是**攻击者可控的**：开发登录允许任意用户自取名，
真实场景下昵称也可能进到标识里。如果直接把它拼进表达式，
一个叫 `" or user_id != "` 的用户就能让过滤条件恒真 —— 读出所有人的记忆。

所以这里的做法是**白名单校验而不是转义**：

    只允许 `[A-Za-z0-9_.:-]`，其余一律拒绝

理由是转义要枚举所有边界情况（引号、反斜杠、换行、编码），
而白名单只需要判断一次。而且这个字符集完全覆盖 UUID 与常见标识符 ——
被拒绝的输入不代表「合法用户被误伤」，代表**从来没有这种合法值**。

（这与项目其他地方 fail-closed 的取向一致：不确定时宁可拒绝。）
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from app.store.vectors import (
    ScoredMemory,
    VectorEntry,
    VectorIndexError,
    VectorIndexUnavailable,
)

__all__ = ["MilvusVectorIndex"]

#: 租户标识允许的字符集。**这是注入防线，不是格式偏好。**
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def _literal(value: str, *, field: str) -> str:
    """把一个租户标识变成 Milvus 字符串字面量。

    Raises:
        VectorIndexError: 含白名单外的字符。**刻意不尝试转义**（见模块 docstring）。
    """
    if not _SAFE_ID.match(value):
        raise VectorIndexError(
            f"{field} 含非法字符，拒绝构造过滤表达式：{value[:40]!r}。\n"
            f"允许的字符集：[A-Za-z0-9_.:-]，长度 1–128。\n"
            f"这是表达式注入的防线 —— 不校验的话，一个恶意标识就能让过滤条件恒真。"
        )
    return f'"{value}"'


class MilvusVectorIndex:
    """Milvus 后端。连接与建表都是**延迟**的。

    ## 为什么延迟

    构造索引时 Milvus 可能还没起来（compose 里它比后端慢得多）。
    在 `__init__` 里连会让「后端启动」依赖「Milvus 已就绪」，
    而 Milvus 只是**加速器**：它不在时系统应该降级并声明，
    而不是起不来。

    所以连接在首次使用时建立，失败则抛 `VectorIndexUnavailable`，
    由调用方决定怎么呈现。
    """

    def __init__(
        self,
        *,
        uri: str,
        token: str | None = None,
        collection: str = "pet_memory",
        dim: int = 1024,
        timeout_s: float = 10.0,
        metric_type: str = "COSINE",
    ) -> None:
        self._uri = uri
        self._token = token or None
        self._collection = collection
        self._dim = dim
        self._timeout_s = timeout_s
        self._metric_type = metric_type
        self._client: Any = None
        self._ready = False

    # ── 连接与建表 ────────────────────────────────────────

    def _ensure(self) -> Any:
        """返回可用的 client。首次调用时连接并确保集合存在。"""
        if self._client is not None and self._ready:
            return self._client

        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的可读提示
            raise VectorIndexUnavailable(
                "未安装 pymilvus。安装：pip install pymilvus"
            ) from exc

        try:
            kwargs: dict[str, Any] = {"uri": self._uri}
            if self._token:
                kwargs["token"] = self._token
            client = MilvusClient(**kwargs)

            if not client.has_collection(self._collection, timeout=self._timeout_s):
                self._create_collection(client)

            self._client = client
            self._ready = True
            return client
        except VectorIndexUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - 任何连接失败都归为「不可用」
            raise VectorIndexUnavailable(
                f"Milvus 不可用（{self._uri} / collection={self._collection}）：{exc}"
            ) from exc

    def _create_collection(self, client: Any) -> None:
        """建集合。标量字段建 INVERTED 索引 —— 每次检索都按它们过滤。"""
        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("memory_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("user_id", DataType.VARCHAR, max_length=128)
        schema.add_field("pet_id", DataType.VARCHAR, max_length=64)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self._metric_type,
        )
        # 隔离过滤走的是标量字段。不建索引的话每次检索都是全量扫描，
        # 而「隔离过滤」是最不该成为瓶颈的那一步。
        index_params.add_index(field_name="user_id", index_type="INVERTED")
        index_params.add_index(field_name="pet_id", index_type="INVERTED")

        client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
            timeout=self._timeout_s,
        )

    # ── VectorIndex 协议 ─────────────────────────────────

    def upsert(self, entries: Sequence[VectorEntry]) -> int:
        if not entries:
            return 0
        client = self._ensure()

        rows: list[dict[str, Any]] = []
        for e in entries:
            if len(e.vector) != self._dim:
                raise VectorIndexError(
                    f"向量维度不符：期望 {self._dim}，得到 {len(e.vector)}"
                    f"（memory_id={e.memory_id}）"
                )
            rows.append(
                {
                    "memory_id": e.memory_id,
                    "vector": list(e.vector),
                    "user_id": e.user_id,
                    "pet_id": e.pet_id,
                }
            )

        try:
            result = client.upsert(
                collection_name=self._collection, data=rows, timeout=self._timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexUnavailable(f"Milvus 写入失败：{exc}") from exc

        # 不同版本返回的计数键名不一致，取不到就按提交条数报
        if isinstance(result, dict):
            for key in ("upsert_count", "insert_count", "upsert_cnt"):
                if key in result:
                    return int(result[key])
        return len(rows)

    def search(
        self,
        *,
        user_id: str,
        pet_id: str,
        vector: list[float],
        limit: int = 20,
    ) -> list[ScoredMemory]:
        if limit <= 0:
            return []
        if len(vector) != self._dim:
            raise VectorIndexError(f"查询向量维度不符：期望 {self._dim}，得到 {len(vector)}")

        client = self._ensure()

        # 隔离过滤在**查询侧**，由 Milvus 执行 —— 不是取回来再筛。
        # 见 app/store/vectors.py 的模块 docstring：后筛会破坏 limit 语义，
        # 而且越权数据已经进了进程内存。
        expr = f"user_id == {_literal(user_id, field='user_id')} and pet_id == {_literal(pet_id, field='pet_id')}"

        try:
            raw = client.search(
                collection_name=self._collection,
                data=[list(vector)],
                filter=expr,
                limit=limit,
                output_fields=["memory_id"],
                search_params={"metric_type": self._metric_type},
                timeout=self._timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexUnavailable(f"Milvus 检索失败：{exc}") from exc

        return _parse_hits(raw)

    def delete(self, memory_ids: Sequence[str]) -> int:
        ids = [m for m in memory_ids if m]
        if not ids:
            return 0
        client = self._ensure()
        try:
            result = client.delete(
                collection_name=self._collection, ids=ids, timeout=self._timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexUnavailable(f"Milvus 删除失败：{exc}") from exc
        if isinstance(result, dict):
            for key in ("delete_count", "delete_cnt"):
                if key in result:
                    return int(result[key])
        return len(ids)

    def delete_by_tenant(self, *, user_id: str, pet_id: str) -> int:
        """删掉某只猫的全部向量。**级联硬删用**（账号/宠物删除）。

        用表达式而不是「先查 id 再删」：后者在删除过程中若有人写入，
        会漏掉新增的那条。
        """
        client = self._ensure()
        expr = f"user_id == {_literal(user_id, field='user_id')} and pet_id == {_literal(pet_id, field='pet_id')}"
        try:
            result = client.delete(
                collection_name=self._collection, filter=expr, timeout=self._timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexUnavailable(f"Milvus 按租户删除失败：{exc}") from exc
        if isinstance(result, dict):
            for key in ("delete_count", "delete_cnt"):
                if key in result:
                    return int(result[key])
        return 0

    def count(self) -> int:
        client = self._ensure()
        try:
            stats = client.get_collection_stats(self._collection, timeout=self._timeout_s)
        except Exception as exc:  # noqa: BLE001
            raise VectorIndexUnavailable(f"Milvus 统计失败：{exc}") from exc
        # 不同版本键名不同：row_count / num_entities
        for key in ("row_count", "num_entities"):
            if isinstance(stats, dict) and key in stats:
                return int(stats[key])
        return 0

    def describe(self) -> dict[str, str]:
        base = {
            "kind": "milvus",
            "collection": self._collection,
            "dim": str(self._dim),
            "metric": self._metric_type,
        }
        # describe 在 /healthz 里被调用，**不能因为 Milvus 挂了就抛** ——
        # 探针本身必须始终可回答，否则「后端活着但索引连不上」这个状态
        # 就变成一个无法观测的黑盒。
        try:
            base["count"] = str(self.count())
            base["connected"] = "true"
        except VectorIndexUnavailable as exc:
            base["connected"] = "false"
            base["reason"] = str(exc)[:200]
        return base


def _parse_hits(raw: Any) -> list[ScoredMemory]:
    """把 Milvus 的返回解析成 `ScoredMemory`。

    不同版本的返回形状不一样（`List[List[dict]]` 里可能是 `id` + `distance`，
    也可能把字段放在 `entity` 下），所以这里对两种都兼容 ——
    写死一种的话，升级 pymilvus 会让检索**静默返回空**。
    """
    if not raw:
        return []

    hits = raw[0] if isinstance(raw, list) and raw and isinstance(raw[0], list) else raw
    out: list[ScoredMemory] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        entity = hit.get("entity") if isinstance(hit.get("entity"), dict) else {}
        mid = hit.get("memory_id") or hit.get("id") or entity.get("memory_id")
        if not mid:
            continue
        score = hit.get("distance", hit.get("score", 0.0))
        try:
            out.append(ScoredMemory(str(mid), float(score)))
        except (TypeError, ValueError):
            continue
    return out
