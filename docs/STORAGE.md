# 存储层

## 目录

- [拓扑与取舍](#拓扑与取舍)
- [为什么这样分](#为什么这样分)
- [表结构](#表结构)
- [本地连线上库](#本地连线上库)
- [降级行为](#降级行为)
- [运维](#运维)
- [已知约束](#已知约束)

---

## 拓扑与取舍

```
                    ┌──────────────────┐
   写入 ──────────► │  MySQL（真相）    │  正文 + 向量（float32 打包）
                    └────────┬─────────┘
                             │ 回填（幂等）
                             ▼
                    ┌──────────────────┐
   检索 ◄────────── │  Milvus（索引）   │  只存 memory_id + 租户字段 + 向量
                    └──────────────────┘
```

**决策**：MySQL 为准，Milvus 是可重建索引。

**理由**：重建索引需要**重新调用 embedding API**（要花钱）。
所以向量本身也持久化在 MySQL（`memories.vector`，1024 维 = 4KB/行）——
代价很小，而收益是 Milvus 变成纯索引：丢了跑一次
`MySQLStore.backfill_vectors()` 即可，不必重新付费。

```
PET_AGENT_STORE=auto|memory|mysql    # 缺省 auto：有 MYSQL_HOST 就走 mysql

MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE
MYSQL_POOL_SIZE=8

MILVUS_HOST / MILVUS_PORT            # 或直接给 MILVUS_URI
MILVUS_TOKEN / MILVUS_COLLECTION
```

`/healthz` 的 `storage` 字段会说明当前用的是哪一个：

```json
{
  "store": "mysql",
  "durable": "true",
  "schema_statements": "7",
  "pool_size": "8",
  "vector_ready": "false",
  "vector_error": "Milvus 不可用（…）",
  "vector.kind": "milvus",
  "vector.connected": "false"
}
```

**为什么必须暴露这一项**：内存后端与 MySQL 后端在**功能上无法从行为区分**，
但一个重启就丢数据、另一个不会。看不到它时，「数据没了」会被当成 bug
排查很久，而它其实是配置。

---

## 为什么这样分

### 租户隔离落在**物理层**

每张业务表都有 `user_id` + `pet_id`，且**索引以它们打头**：

```sql
KEY idx_mem_recall (user_id, pet_id, status)
```

索引顺序是刻意的：隔离过滤必须走索引。一旦它慢，
就会有人来「优化」掉它 —— 而隔离不是可以优化的东西。

### 隔离过滤无法被忘掉

所有按租户读取的语句集中在 `_TENANT_SCOPED_SQL` 这一个字典里，
而 `tests/test_store_sql.py` 逐条断言：

- 每条含 `user_id = %s`
- 宠物范围内的表还含 `pet_id = %s`
- 引用的表在 DDL 里真实存在
- 每张业务表都有 `user_id` 列，且索引以 `(user_id, pet_id)` 打头

**「新加一个查询忘了加过滤」会变成测试失败，而不是一个只在生产被利用的越权。**

这条断言被验证过是有效的：故意去掉 `get_memory` 的 `user_id` 过滤，
测试立刻报 `缺少 user_id 过滤，会导致跨用户越权：['get_memory']`。

### 向量检索的租户过滤在**查询侧**

`VectorIndex.search()` 强制要求 `user_id` + `pet_id`（keyword-only、无默认值）。
在 Milvus 侧用表达式过滤，而不是「取回来再用应用代码筛」：

- 后筛会破坏 `limit` 语义（取 top-20 筛掉别人的，可能只剩 3 条）
- 越权数据已经进了进程内存 —— 筛掉只是不返回，那一刻隔离已经破了

回表时**再过滤一次**（第二道防线）。代价是一次主键查询；
收益是索引配错时仍然挡得住。

---

## 表结构

`app/store/ddl.sql`。7 张表，全部 `CREATE TABLE IF NOT EXISTS`。

| 表 | 内容 | 特殊约束 |
| --- | --- | --- |
| `pets` | 档案 + 身份锚点 | `must_keep_features` 与 `observed_but_unstable` **分列** |
| `memories` | 记忆事件 | `UNIQUE (pet_id, dedup_key)`；`CHECK` 挡 `system_inference + active` |
| `meow_records` | 主人标注的叫声 | 特征只存服务端 |
| `pending_interpretations` | 待标注解释 | 同上 |
| `session_messages` | 对话原文 | 与 `memories` 分开（日报输入不能是总结的产物） |
| `health_records` | 健康信号 | 值拆成 `kind` + 三列 + `CHECK` 保证一致 |
| `health_assessments` | 健康评估 | 随记录级联硬删 |

### 三个容易踩的点

**1. 时区**
领域模型是 aware UTC，MySQL `DATETIME` 不带时区。
`_to_db` / `_from_db` 是强制的收口点 —— 不做转换**不会报错**，只会静默偏移。

**2. 保留字**
`signal`、`sensitive` 都是 MySQL 保留字，必须加反引号。
不加时报的是语法错误，而错误信息指向的是**下一行**。
（`sensitive` 从 8.0.13 起保留 —— 更旧的版本不会报错，
所以「本地能跑」不代表「线上能跑」。）

**3. 布尔 vs 数字**
Python 里 `True` 是 `int` 的实例。`_split_value` 必须**先判 `bool` 再判数字**，
否则布尔会被存进 numeric 列、读回来变成 `1.0` ——
而 `1.0` 在红旗求值里与「用户点了是」是完全不同的语义。

---

## 本地连线上库

数据库只绑服务器回环，**不对公网开放**。本地通过 SSH 隧道访问：

```bash
scripts/db-tunnel.sh            # 前台（Ctrl-C 断开）
scripts/db-tunnel.sh --daemon   # 后台
scripts/db-tunnel.sh --status   # 查看连通性
scripts/db-tunnel.sh --stop     # 停止
```

然后本地 `.env`：

```bash
MYSQL_HOST=127.0.0.1
MYSQL_PORT=13306        # 刻意不用 3306：本机已有 MySQL 时会连到错误的库
MYSQL_USER=pet_user
MYSQL_PASSWORD=<服务器 .env 里的值>
MYSQL_DATABASE=pet_agent

MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
```

> **为什么不用「直接开 3306」**：那会把「user_id 不可伪造」这条隔离前提
> 降级成「密码别泄露」。隧道让数据库在网络上**不可达**，
> 只有持有私钥的人能连进去。

> **隧道脚本里的两个坑**（都已修）：
>
> 1. 不禁用 GSSAPI 时，ssh 会先试 `gssapi-with-mic`，
>    在没有 Kerberos 凭据的机器上会挂很久 —— 表现为
>    「ssh 进程活着、端口却没绑上」，日志里只有一句
>    `No Kerberos credentials available`，很容易误判成网络问题。
> 2. 检查连通性不能用 `/dev/tcp`（dash 不支持），
>    否则「检查工具不可用」会被读成「端口不通」。

### 跑集成测试

配好上面的环境变量后，一致性测试会自动包含 MySQL 后端：

```bash
MYSQL_HOST=127.0.0.1 MYSQL_PORT=13306 ... python -m pytest tests/ -q
```

未配置时**跳过而不是失败**（本地开发不该被阻塞）；
CI 里配了就必须跑。

---

## 降级行为

| 情况 | 行为 |
| --- | --- |
| 未配 `MYSQL_HOST` | 走内存，`durable: false` |
| 配了 mysql 但缺密码 | **启动失败**（不静默退回内存） |
| MySQL 连不上 | **启动失败**（同上） |
| Milvus 连不上 | **启动成功**，检索降级 |

**为什么 Milvus 挂了不阻止启动**：它只是加速器 ——
向量在 MySQL 里有，索引可以回填，**不丢数据**。
而「静默降级」与「可见降级」的区别在于：降级状态出现在
`/healthz` 与每次响应的 trace 里。

### 降级时的响应

```json
{
  "degraded": true,
  "degraded_notice": "memory_retriever: 检索降级：Milvus 不可用（…）",
  "trace": [
    {"node": "memory_retriever", "degraded": true, "decision": "检索降级：…"}
  ]
}
```

这条链路修过一个真实缺陷：初版响应顶层的 `degraded` **只看守卫**，
于是 `memory_retriever` 在 trace 里标了降级、顶层却说「一切正常」，
前端据此显示正常 —— 而用户看到的是一句没有依据的「没有相关记录」。
现在 `degraded` 聚合全部来源。

---

## 运维

```bash
# 启动 / 停止数据服务（compose profile）
docker compose -f docker-compose.prod.yml --profile data up -d
docker compose -f docker-compose.prod.yml --profile data down

# 只启 MySQL（Milvus 用远程实例时）
docker compose -f docker-compose.prod.yml --profile data up -d mysql
```

> **小内存机器（1.8GB）的调优已在 compose 里**：
> MySQL 关掉 `performance_schema`（它默认占 200MB+）、
> `innodb_buffer_pool_size=128M`、`max_connections=50`；Milvus 限 700M。
> 不调的话，「MySQL 占一半内存」看起来像数据量问题，实则是默认值问题 ——
> 实测调优前 MySQL 占 508MB，而数据量是几十行。

```bash
# 连进 MySQL
docker compose -f docker-compose.prod.yml exec mysql \
  mysql -upet_user -p pet_agent

# 看后端用的哪个存储
curl -s http://127.0.0.1/healthz | python3 -m json.tool | grep -A12 storage

# 回填向量索引（Milvus 重建后 / 索引落后时）
docker compose -f docker-compose.prod.yml exec pet-agent python -c "
from app.store.factory import build_store_from_env
b = build_store_from_env()
print('回填', b.store.backfill_vectors(), '条')
b.close()
"
```

### 表结构变更

目前是**启动时自动建表**（`apply_schema()`，全部 `IF NOT EXISTS`，幂等）。
不引 migration 框架的理由：这个阶段表结构还在动，而 migration 的价值
在于「多个环境的历史版本不一致」—— 现在只有一个环境。

**线上有数据不能重建时**再引 Alembic，那时 DDL 已是稳定基线。

所以：**加列可以直接改 `ddl.sql` 并重启**（`IF NOT EXISTS` 不会改已有表），
但**改已有列不会生效** —— 需要手工 `ALTER TABLE`。

---

## 已知约束

| 约束 | 影响 | 缓解 |
| --- | --- | --- |
| 向量存 MySQL 占空间 | 1024 维 = 4KB/行 | 可接受；真要省可以只存索引 |
| 无 migration 框架 | 改列需手工 ALTER | 见上 |
| 连接池是手写的 | 无自动重试、无指标 | 有 ping 检测与重建；池大小可调 |
| Milvus 计数要跨段查 | `get_collection_stats` 不可信 | 用 `count(*)`，见下 |

### Milvus 的 `row_count` 会说谎

`get_collection_stats()` 只统计**已 flush 的持久化段**，
而写入先进 growing segment。实测：刚写入 5 条时它报 `0`，
而 `query` 能查到全部 5 条。

所以 `MilvusVectorIndex.count()` 用 `count(*)` 聚合查询。

**为什么这个数字很重要**：它出现在 `/healthz` 里。
报 0 的后果不是「少一个数字」—— 运维看到 `vector.count: 0`
会去跑一次白做的回填，或者据此判断「索引没生效」
而去查一个根本不存在的问题。

### 手写连接池的取舍

`MySQLPool` 是队列池（非 thread-local），理由：

- thread-local 没有上限 —— uvicorn 线程池多大就有多少连接，
  而且**无法显式关闭**（每线程各自持有）。测试里反复建 app 会泄漏连接，
  直到 MySQL 报 `Too many connections`。
- 队列池把连接数变成可控数字，`close()` 能真正关干净。

**它踩过一个死锁**：初版 `_checkout()` 在持有 `self._lock` 时调用
`_new_connection()`，而后者也要拿同一把锁 —— `threading.Lock` 不可重入，
于是第一次取连接就挂住。表现是「进程挂住不报错」，非常难定向。
现在「记账」与「建连接」是分开的，且**持锁时不做任何 I/O**。
