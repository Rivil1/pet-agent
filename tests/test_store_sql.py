"""存储层的静态检查：**不连数据库也能跑**。

## 为什么这组测试值得单独存在

「新加了一个查询但忘了加租户过滤」是本项目最危险的一类缺陷：

- 它**不报错**。查询照常返回结果，只是多了别人的数据。
- 它**只在生产被利用**。单测若只覆盖「自己的数据」，永远发现不了。
- 它**是 D5（多租户隔离）的全部内容**。隔离不是某个功能，是每个查询的属性。

所以这里不测行为，测**结构**：

1. 每条按租户读取的语句都必须含 `user_id = %s`
2. 宠物范围内的表还必须含 `pet_id = %s`
3. 语句引用的表必须在 DDL 里真实存在（拼错表名在运行时才炸）
4. DDL 里每张业务表都必须有 `user_id` 列

第 3、4 条同样重要：前者防「SQL 写错」，后者防「新加一张表忘了加租户列」——
而后者会让该表上的所有查询天然无法过滤租户。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.store.mysql import _TENANT_SCOPED_SQL

DDL_PATH = Path(__file__).resolve().parent.parent / "app" / "store" / "ddl.sql"

#: 宠物就是主键本身，不需要额外的 pet_id 过滤。
#: 其余表都是「多行属于同一只宠物」，必须双重过滤。
_PET_IS_PRIMARY_KEY = {"get_pet"}

#: 含健康数据的表不在 `MySQLStore` 里，单独在 app/health/mysql.py。
#: 它们由 test_health_sql_is_tenant_scoped 覆盖。


def _ddl_tables() -> set[str]:
    sql = DDL_PATH.read_text(encoding="utf-8")
    return {m.group(1).lower() for m in re.finditer(r"CREATE TABLE IF NOT EXISTS (\w+)", sql)}


def _ddl_columns(table: str) -> list[str]:
    """粗解析某张表的列名。

    不引 SQL parser：DDL 是我们自己写的，格式稳定。

    ⚠️ **必须靠「第二个词是类型」来判定列，而不是「行首是标识符」。**
    初版只看行首，于是 CHECK 约束的续行
    （`OR (value_kind = ...)` / `AND value_number IS NULL ...`）
    被当成了名为 `or` / `and` 的列 —— 而那两个恰好是 MySQL 保留字，
    于是保留字检查会报出两个**根本不存在**的列。
    假阳性会让真问题淹在噪声里。
    """
    sql = DDL_PATH.read_text(encoding="utf-8")
    m = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*ENGINE",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return []

    cols: list[str] = []
    for line in m.group(1).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if re.match(r"^(PRIMARY KEY|UNIQUE KEY|KEY|CONSTRAINT|INDEX|FOREIGN)\b", stripped, re.I):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        # 类型可能带长度/精度，如 CHAR(36) / DECIMAL(10,2) / ENUM('a','b')。
        # ⚠️ 不能用 strip("(),") —— 它只去**两端**，
        # `CHAR(36)` 会变成 `CHAR(36`（中间的括号留着），于是类型匹配全失败。
        # 而失败的方向很坑：带括号的类型（CHAR/VARCHAR/DECIMAL）全部消失，
        # 只剩 JSON / TEXT 这类无括号的 —— 于是「列名解析」看着还有结果，
        # 只是少了一大半，而后面的断言就变成了假绿。
        type_token = parts[1].split("(")[0]
        name = parts[0].strip(",").strip("`")
        if not re.match(r"^\w+$", name):
            continue
        if not _SQL_TYPE.match(type_token):
            continue
        cols.append(name.lower())
    return cols


#: 列定义里的类型关键字。只有第二个词命中才算列。
_SQL_TYPE = re.compile(
    r"^(CHAR|VARCHAR|TEXT|TINYTEXT|MEDIUMTEXT|LONGTEXT|BINARY|VARBINARY|BLOB"
    r"|TINYBLOB|MEDIUMBLOB|LONGBLOB|INT|INTEGER|TINYINT|SMALLINT|MEDIUMINT|BIGINT"
    r"|DECIMAL|NUMERIC|FLOAT|DOUBLE|REAL|BIT|BOOL|BOOLEAN|DATE|DATETIME|TIMESTAMP"
    r"|TIME|YEAR|JSON|ENUM|SET|GEOMETRY)$",
    re.IGNORECASE,
)


def _tables_referenced(sql: str) -> set[str]:
    """从语句里抽出表名。只看 FROM / JOIN / UPDATE / INTO 之后那个词。"""
    found = set()
    for m in re.finditer(
        r"\b(?:FROM|JOIN|UPDATE|INTO)\s+(\w+)", sql, re.IGNORECASE
    ):
        found.add(m.group(1).lower())
    return found


# ─────────────────────────────────────────────────────────────
# 1 & 2：每条语句的租户过滤
# ─────────────────────────────────────────────────────────────


def test_every_scoped_query_filters_user_id():
    """**核心断言**：漏掉 `user_id` 就是跨用户越权。"""
    missing = [
        name
        for name, sql in _TENANT_SCOPED_SQL.items()
        if "user_id = %s" not in sql
    ]
    assert not missing, (
        f"以下语句缺少 user_id 过滤，会导致跨用户越权：{missing}\n"
        f"这是 D5（多租户隔离）的破口，不是风格问题。"
    )


def test_pet_scoped_queries_filter_pet_id():
    missing = [
        name
        for name, sql in _TENANT_SCOPED_SQL.items()
        if name not in _PET_IS_PRIMARY_KEY and "pet_id = %s" not in sql
    ]
    assert not missing, (
        f"以下语句缺少 pet_id 过滤，会导致同一用户的不同宠物之间串味：{missing}"
    )


def test_scoped_sql_registry_is_not_empty():
    """空字典会让上面两条断言永远通过 —— 这是最容易出现的「假绿」。"""
    assert len(_TENANT_SCOPED_SQL) >= 10, (
        f"租户语句清单只有 {len(_TENANT_SCOPED_SQL)} 条，疑似被清空或改名 —— "
        f"那样上面两条断言会变成永真。"
    )


def test_get_pet_exemption_is_justified():
    """`get_pet` 是唯一豁免 pet_id 的语句，且理由是「pet_id 就是主键」。

    这条断言防止豁免清单被悄悄扩大：新增豁免必须同时改这里，
    而改这里需要写下理由。
    """
    assert _PET_IS_PRIMARY_KEY == {"get_pet"}
    sql = _TENANT_SCOPED_SQL["get_pet"]
    assert "WHERE pet_id = %s" in sql, "get_pet 必须按主键查，而不是全表扫"


# ─────────────────────────────────────────────────────────────
# 3：语句引用的表必须存在
# ─────────────────────────────────────────────────────────────


def test_referenced_tables_exist_in_ddl():
    tables = _ddl_tables()
    assert tables, "没有从 ddl.sql 解析到任何表 —— 正则或格式变了"

    unknown: dict[str, set[str]] = {}
    for name, sql in _TENANT_SCOPED_SQL.items():
        refs = _tables_referenced(sql) - tables
        if refs:
            unknown[name] = refs

    assert not unknown, (
        f"以下语句引用了 DDL 里不存在的表（拼错表名在运行时才会炸）：{unknown}\n"
        f"DDL 中的表：{sorted(tables)}"
    )


# ─────────────────────────────────────────────────────────────
# 4：每张业务表都必须有租户列
# ─────────────────────────────────────────────────────────────

#: 这些表不直接承载租户数据，或本身就是租户键。
_NON_TENANT_TABLES: set[str] = set()

#: 必须有 pet_id 的表（健康评估等派生表也要 —— 删除时要能按宠物级联）。
_REQUIRE_PET_ID = {
    "memories",
    "meow_records",
    "pending_interpretations",
    "session_messages",
    "health_records",
    "health_assessments",
}


@pytest.mark.parametrize(
    "table",
    [
        "pets",
        "memories",
        "meow_records",
        "pending_interpretations",
        "session_messages",
        "health_records",
        "health_assessments",
    ],
)
def test_every_table_has_user_id(table: str):
    cols = _ddl_columns(table)
    assert cols, f"没解析到 {table} 的列 —— DDL 格式可能变了"
    assert "user_id" in cols, (
        f"表 {table} 缺少 user_id 列。\n"
        f"缺了它，这张表上的查询**无法**过滤租户 —— 隔离在物理层就不成立。"
    )


@pytest.mark.parametrize("table", sorted(_REQUIRE_PET_ID))
def test_pet_scoped_tables_have_pet_id(table: str):
    cols = _ddl_columns(table)
    assert "pet_id" in cols, f"表 {table} 缺少 pet_id 列（同一用户的多只宠物会串味）"


@pytest.mark.parametrize("table", sorted(_REQUIRE_PET_ID))
def test_pet_scoped_tables_index_starts_with_tenant(table: str):
    """隔离过滤必须走索引。

    索引不以 (user_id, pet_id) 打头时，过滤会退化成全表扫描 ——
    而「隔离」是最不该成为性能瓶颈的那一步：一旦它慢，
    就会有人去「优化」掉它。
    """
    sql = DDL_PATH.read_text(encoding="utf-8")
    m = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\n\)\s*ENGINE",
        sql,
        re.DOTALL | re.IGNORECASE,
    )
    assert m, f"找不到 {table} 的定义"
    body = m.group(1)

    index_cols = [
        tuple(c.strip() for c in idx.group(1).split(","))
        for idx in re.finditer(r"KEY\s+\w+\s*\(([^)]+)\)", body)
    ]
    assert index_cols, f"表 {table} 没有任何 KEY 定义"

    leading_with_tenant = [
        cols for cols in index_cols if cols[:2] == ("user_id", "pet_id")
    ]
    assert leading_with_tenant, (
        f"表 {table} 没有以 (user_id, pet_id) 打头的索引。\n"
        f"现有索引：{index_cols}\n"
        f"隔离过滤必须走索引，否则它会被当成性能问题而被绕过。"
    )


# ─────────────────────────────────────────────────────────────
# 5：DDL 里的关键不变量约束
# ─────────────────────────────────────────────────────────────


def test_ddl_has_dedup_unique_key():
    """`idx_mem_dedup`：重复写入必须能转成强化，而不是插入第二条。"""
    sql = DDL_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"UNIQUE KEY\s+\w+\s*\(\s*pet_id\s*,\s*dedup_key\s*\)", sql, re.I
    ), "memories 缺少 (pet_id, dedup_key) 唯一索引 —— 去重强化会失效"


def test_ddl_blocks_active_system_inference():
    """不变量 I1（防自我强化）必须在**数据库层**也挡一道。

    应用层挡的是「正常路径写错」，这里挡的是绕过契约的写入路径
    （脚本、迁移、未来的新代码）。
    """
    sql = DDL_PATH.read_text(encoding="utf-8")
    assert re.search(r"ck_mem_no_active_inference", sql), "缺少 I1 的 DB 层约束"
    assert re.search(
        r"NOT\s*\(\s*source\s*=\s*'system_inference'\s*AND\s*status\s*=\s*'active'\s*\)",
        sql,
        re.I,
    ), "ck_mem_no_active_inference 的表达式不对"


def test_ddl_health_value_shape_is_constrained():
    """健康值的三列展开必须有 CHECK 保证一致。

    不一致时读出来是 None，而 None 在红旗求值里意味着「不知道」——
    一条明明有值的记录会被静默当成缺失。
    """
    sql = DDL_PATH.read_text(encoding="utf-8")
    assert re.search(r"ck_health_value_shape", sql), "缺少健康值形状约束"
