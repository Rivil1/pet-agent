# 系统架构 · pet-agent

> **本文与 [DESIGN.md](DESIGN.md) 的分工**
>
> | 文档 | 视角 | 内容 |
> | --- | --- | --- |
> | `DESIGN.md` §2 | **逻辑架构** | 分层、编排图、节点契约、`AgentState`、降级策略 |
> | **本文** | **物理 / 实现架构** | 部署拓扑、存储 schema、API 契约、运行时、可观测性、安全 |
>
> **本文不重复逻辑架构。** 节点职责、编排流转、机制设计一律引用 `DESIGN.md`。
>
> 若两者冲突：**逻辑以 `DESIGN.md` 为准，物理以实现为准。**

| 项 | 值 |
| --- | --- |
| 阶段 | P0（4–6 周可演示闭环，见 `DESIGN.md` §7.1） |
| 已实现 | 契约层 8 模块（2,283 行）+ 不变量测试 92 项 |
| 本文状态 | 设计完成，**未实现** |

---

# 1. 架构总览

## 1.1 三个视图

```
逻辑视图        DESIGN.md §2（节点 + 状态流转）
                    │
实现视图        本文 §2–§6（存储 / API / 运行时 / 可观测性）
                    │
部署视图        本文 §8（容器拓扑）
```

## 1.2 部署拓扑（P0）

**单体进程起步**，但模块边界单向。

```
                        ┌──────────────────────┐
   客户端 ──HTTPS──────►│  api (FastAPI)       │
                        │  ├ 契约校验          │
                        │  ├ LangGraph 编排    │
                        │  └ 同步节点          │
                        └───────┬──────────────┘
                                │ 同一事务
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ PostgreSQL │  │  job_queue │  │ 对象存储    │
        │ + pgvector │  │  (同库表)  │  │ (MinIO/FS) │
        └────────────┘  └──────┬─────┘  └────────────┘
                               │ FOR UPDATE SKIP LOCKED
                               ▼
                        ┌──────────────────────┐
                        │  worker (同镜像)      │
                        │  ├ memory_writer     │
                        │  ├ tts_output        │
                        │  ├ visual_assessor   │
                        │  └ 吐槽生成           │
                        └──────────────────────┘

可选：Redis（Session 层缓存）
```

**为什么单体 + 单库**：P0 只有四步闭环。引入微服务或独立向量库会带来
**一致性与运维成本**，而收益为零。模块边界靠代码约束，不靠网络边界。

## 1.3 模块依赖方向（单向，不可逆）

```
app/api ──► app/graph ──► app/nodes ──► app/schemas
                │              │              ▲
                │              ▼              │
                └──────► app/store ───────────┘
                               │
                               ▼
                        app/infra（LLM / ASR / TTS / 声学）
```

| 规则 | 说明 |
| --- | --- |
| D1 | `schemas` **不依赖任何内部模块**（纯契约，可被任意层引用） |
| D2 | `nodes` **不直接访问数据库**，只通过 `store` 抽象注入 |
| D3 | `nodes` **不依赖 `graph`**（节点不知道自己被谁编排） |
| D4 | `api` **不含业务判断**，只做鉴权、校验、序列化 |
| D5 | `store` 方法签名**强制要求 `user_id` + `pet_id`**（见 §2.6） |

> D1 已由现状保证：`app/schemas/` 只依赖 `pydantic` 与标准库，92 项测试可独立运行。

---

# 2. 存储架构

## 2.1 选型与理由

| 数据 | 选型 | 理由 | 被否决的备选 |
| --- | --- | --- | --- |
| 结构化数据 | **PostgreSQL 16** | 事务、JSONB、数组、全文检索俱全 | — |
| 向量 | **pgvector**（同库） | 「先结构化过滤再向量检索」需同库 JOIN；**一致性成本 < 性能收益** | Milvus：跨库一致性差（向量删了关系记录没删） |
| 任务队列 | **PostgreSQL 表 + `SKIP LOCKED`** | 与业务数据同事务 → outbox 无需分布式事务；不引入新组件 | Redis+RQ：多一个组件，且事务跨界 |
| 会话缓存 | Redis（**可选**） | 天然 TTL | 进程内 dict：多实例失效 |
| 对象存储 | 开发：本地 FS / 生产：MinIO | S3 兼容，可平滑迁移 | 直接存 DB：大对象伤 DB |

> `DESIGN.md` §3.5 的结论是 pgvector 优先，本文给出实现层理由：
> **outbox 模式要求业务写入与入队同事务**，同库直接省掉分布式事务。
> 这是本项目选单体 + 单库的核心动因。

## 2.2 关系库 schema

### `pets` · 宠物档案（身份锚定）

```sql
CREATE TABLE pets (
  id                 UUID PRIMARY KEY,
  user_id            UUID NOT NULL,
  name               TEXT NOT NULL,
  species            TEXT NOT NULL DEFAULT 'cat',
  breed              TEXT,
  visual             JSONB NOT NULL DEFAULT '{}',   -- VisualProfile
  must_keep_features TEXT[] NOT NULL DEFAULT '{}',  -- 身份硬约束
  unstable_features  TEXT[] NOT NULL DEFAULT '{}',
  identity_prompt    TEXT,
  appearance_vec     vector(512),   -- 外观一致性校验（非个体鉴定）
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_pets_user ON pets (user_id);
```

### `pet_traits` · 性格/喜好/习惯（含层级归属）

```sql
CREATE TABLE pet_traits (
  id          UUID PRIMARY KEY,
  pet_id      UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  layer       TEXT NOT NULL CHECK (layer IN ('observed','inferred','roleplay')),
  text        TEXT NOT NULL,
  source      TEXT NOT NULL,
  confidence  REAL CHECK (
                (layer = 'inferred' AND confidence IS NOT NULL) OR
                (layer <> 'inferred' AND confidence IS NULL)
              ),  -- 与契约层 Trait 校验器一致
  observed_at TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_traits_pet_layer ON pet_traits (pet_id, layer);
```

### `memory_events` · 记忆（核心表）

```sql
CREATE TABLE memory_events (
  id            UUID PRIMARY KEY,
  user_id       UUID NOT NULL,
  pet_id        UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  layer         TEXT NOT NULL CHECK (layer IN ('profile','episode','session')),
  event_type    TEXT NOT NULL,
  subject       TEXT NOT NULL,          -- 冲突检测条件 ①
  content       TEXT NOT NULL,
  polarity      TEXT NOT NULL,          -- 冲突检测条件 ②
  valid_from    TIMESTAMPTZ,            -- 冲突检测条件 ③
  valid_to      TIMESTAMPTZ,
  occurred_at   TIMESTAMPTZ,
  source        TEXT NOT NULL,
  confidence    REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  status        TEXT NOT NULL,
  support_count INT  NOT NULL DEFAULT 1 CHECK (support_count >= 1),
  last_seen_at  TIMESTAMPTZ,
  dedup_key     TEXT,
  embedding     vector(1024),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

  CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to),

  -- ⭐ 不变量 I1 在数据库层再强制一次（深度防御）
  CONSTRAINT no_active_system_inference
    CHECK (NOT (source = 'system_inference' AND status = 'active'))
);
```

> **为什么要把 I1 在 DB 层再写一遍**：
> 应用层校验**可被绕过**（运维脚本、手写 SQL、未来的新代码路径、数据导入）。
> 应用层挡的是**错误**，DB 约束挡的是**不可能**。
> 对一个「模型输出不得成为事实」的安全底线，值得双重强制。

**但这引入一个实现问题**：DB 约束抛的是 `IntegrityError`（底层异常），
契约层抛的是友好校验错误。**若不在 `store` 层翻译，用户会看到 500。**

```python
CONSTRAINT_TO_DOMAIN_ERROR = {
    "no_active_system_inference": (ValidationError, "系统推断不得作为事实存储"),
    "idx_mem_dedup":              (ConflictError,   "记忆已存在，应转为强化"),
    "coverage_gate":              (ValidationError, "覆盖率不足，必须输出「无法评估」"),
    "emergency_needs_redflag":    (ValidationError, "EMERGENCY 必须由红旗规则触发"),
}

async def translate_integrity_error(exc: IntegrityError) -> Exception:
    """把 DB 层的 IntegrityError 翻译为契约层的领域错误。

    规则：DB 约束是最后一道防线，但**用户看到的必须是领域错误，不是 500**。
    无法匹配的约束名 → 原样抛出（说明有未登记的约束）。
    """
```

> 初稿遗漏了这一层。**「深度防御」的代价就是要有翻译层**，否则
> 多一道防线 = 多一种 500。

### 索引

```sql
-- 默认检索：只查 active，按 pet 过滤（对应 DESIGN.md §3.5 检索管线 ①）
CREATE INDEX idx_mem_active ON memory_events (pet_id, event_type, occurred_at DESC)
  WHERE status = 'active';

-- 向量：partial index，只索引 active（避免把 PENDING/SUPERSEDED 拉进召回）
CREATE INDEX idx_mem_vec ON memory_events
  USING hnsw (embedding vector_cosine_ops)
  WHERE status = 'active';

-- 去重：唯一索引 → 强化而非重复插入，在 DB 层强制
CREATE UNIQUE INDEX idx_mem_dedup ON memory_events (pet_id, dedup_key)
  WHERE dedup_key IS NOT NULL;

-- 取代链
CREATE INDEX idx_mem_superseded ON memory_events (superseded_by)
  WHERE superseded_by IS NOT NULL;
```

### `memory_supersedes` · 取代链

```sql
CREATE TABLE memory_supersedes (
  from_id UUID NOT NULL REFERENCES memory_events(id) ON DELETE CASCADE,
  to_id   UUID NOT NULL REFERENCES memory_events(id) ON DELETE CASCADE,
  PRIMARY KEY (from_id, to_id)
);
```

### `moments` · 瞬间记录（P1）

```sql
CREATE TABLE moments (
  id          UUID PRIMARY KEY,
  user_id     UUID NOT NULL,
  pet_id      UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  media_url   TEXT NOT NULL,
  media_kind  TEXT NOT NULL CHECK (media_kind IN ('image','video')),
  scene_tag   TEXT,
  captured_at TIMESTAMPTZ NOT NULL,
  caption     TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_moments_pet_time ON moments (pet_id, captured_at DESC);
```

### `meow_samples` · 叫声样本（行为解释器的训练数据）

```sql
CREATE TABLE meow_samples (
  id             UUID PRIMARY KEY,
  user_id        UUID NOT NULL,
  pet_id         UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  audio_url      TEXT NOT NULL,
  features       JSONB NOT NULL,        -- AcousticFeatures
  context_label  TEXT,                  -- 用户确认的 context（非 intent）
  prior_version  TEXT NOT NULL,         -- 复现要求（DESIGN.md §3.6）
  confirmed_at   TIMESTAMPTZ,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_meow_pet ON meow_samples (pet_id, confirmed_at DESC);
```

### `health_records` / `health_assessments`

```sql
CREATE TABLE health_records (
  id            UUID PRIMARY KEY,
  user_id       UUID NOT NULL,
  pet_id        UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  signal        TEXT NOT NULL,
  value_num     DOUBLE PRECISION,
  value_text    TEXT,
  unit          TEXT,
  recorded_at   TIMESTAMPTZ NOT NULL,
  source        TEXT NOT NULL,
  sensitive     BOOLEAN NOT NULL DEFAULT true,
  consent_version TEXT,
  retention_days  INT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE health_assessments (
  id             UUID PRIMARY KEY,
  user_id        UUID NOT NULL,
  pet_id         UUID NOT NULL REFERENCES pets(id) ON DELETE CASCADE,
  level          TEXT NOT NULL,
  coverage       REAL NOT NULL CHECK (coverage BETWEEN 0 AND 1),
  findings       JSONB NOT NULL DEFAULT '[]',
  red_flags      JSONB NOT NULL DEFAULT '[]',
  rule_version   TEXT NOT NULL,
  as_of          TIMESTAMPTZ NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- 不变量 I11：覆盖率不足时级别必须是 INSUFFICIENT_DATA
  CONSTRAINT coverage_gate CHECK (
    coverage >= 0.3 OR level = 'INSUFFICIENT_DATA'
  ),
  -- 不变量 I12：EMERGENCY 必须由红旗触发
  CONSTRAINT emergency_needs_redflag CHECK (
    level <> 'EMERGENCY' OR jsonb_array_length(red_flags) > 0
  )
);
```

### `job_queue` · 任务队列 + outbox

```sql
CREATE TABLE job_queue (
  id           BIGSERIAL PRIMARY KEY,
  kind         TEXT NOT NULL,          -- memory_write | tts | visual_assess | banter
  payload      JSONB NOT NULL,
  -- required | best_effort  —— 见 §3.1 的副作用分级
  durability   TEXT NOT NULL DEFAULT 'best_effort'
                 CHECK (durability IN ('required','best_effort')),
  status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending','running','done','failed')),
  attempts     INT NOT NULL DEFAULT 0,
  max_attempts INT NOT NULL DEFAULT 5,
  run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
  locked_by    TEXT,
  locked_at    TIMESTAMPTZ,
  last_error   TEXT,
  idem_key     TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_job_ready ON job_queue (run_after)
  WHERE status = 'pending';
CREATE UNIQUE INDEX idx_job_idem ON job_queue (idem_key)
  WHERE idem_key IS NOT NULL;
```

### `audit_log` · 记忆写入审计

```sql
CREATE TABLE audit_log (
  id         BIGSERIAL PRIMARY KEY,
  user_id    UUID NOT NULL,
  pet_id     UUID NOT NULL,
  action     TEXT NOT NULL,      -- memory_write / supersede / reject
  target_id  UUID,
  detail     JSONB NOT NULL,
  trace_id   TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

对应 `DESIGN.md` §3.5 的可复现要求 ③「每次解释落库」与记忆写入可审计。

## 2.3 向量检索设计

### 维度

| 用途 | 维度 | 模型 |
| --- | --- | --- |
| 记忆语义检索 | 1024 | 文本 embedding |
| 外观一致性校验 | 512 | 图像 embedding |

> ⚠️ **上表的维度数字待与 DashScope 文档核对。** 维度必须与实际模型一致，
> 否则建表后**无法插入**。schema 里的 `vector(N)` 需与此保持一致。

> ⚠️ 外观向量**不是个体识别**（`DESIGN.md` §1.5 边界 1）。
> 它只作冲突告警，`must_keep_features` 文本才是身份硬约束。

### 带过滤 ANN 的已知问题

pgvector 的 HNSW 在带 `WHERE` 过滤时存在**召回不足 K 条**的问题
（过滤后有效候选少于 K）。

| 缓解手段 | 说明 |
| --- | --- |
| **partial index** | `WHERE status='active'`，把过滤条件下推到索引（§2.2 已用） |
| `hnsw.ef_search` 调大 | 提高召回，代价是延迟 |
| iterative scan | pgvector 0.8+ 提供，**选型时需确认版本** |
| 超量召回后二次截断 | 应用层兜底：召回 3K 再过滤取 K |

**决策**：P0 用 partial index + `ef_search=100`；**上线前需实测召回率**（评测项 E11）。

## 2.4 对象存储布局

```
s3://pet-agent/{user_id}/{pet_id}/
    moments/{yyyy}/{mm}/{moment_id}.jpg|mp4
    meow/{yyyy}/{mm}/{sample_id}.wav
    tts/{yyyy}/{mm}/{hash}.mp3
```

| 规则 | 说明 |
| --- | --- |
| P1 | 路径含 `user_id`，便于按用户删除 |
| P2 | 上传后**剥离 EXIF**（含地理位置） |
| P3 | 媒体原文件不加密；**健康记录字段加密**（§7.2） |
| P4 | 删除走「标记 → 异步清理」，保留期见 §2.5 |

## 2.5 数据生命周期与删除

| 数据 | 保留 | 删除语义 |
| --- | --- | --- |
| `memory_events` | 长期 | 用户可删（**硬删除**） |
| `health_records` | `retention_days`（默认 730 天） | 用户可导可删；**硬删除 + 级联至派生评估** |
| `moments` 媒体 | 长期 | 用户可删 |
| `meow_samples` | 长期 | 用户可删 |
| `audit_log` | 长期 | **不随用户删除**（审计用途）；仅存 ID 与动作，不存内容 |
| `job_queue` | `done` 后 7 天 | 自动清理 |
| Session（Redis） | TTL 24h | 自动过期 |

> 对应 `DESIGN.md` §7.3 U16（健康数据加密/删除策略需落地）——本节给出方案，实现待做。

## 2.6 租户隔离的三个强制点

`DESIGN.md` 反复强调「多租户隔离必须强制，不靠调用方自觉」。**具体在哪强制**：

| # | 强制点 | 机制 |
| --- | --- | --- |
| **T1** | 鉴权层 | `user_id` 从 **token 派生**，**绝不从请求体/查询参数读** |
| **T2** | 资源归属校验 | 路径含 `pet_id` → 先校验 `pets.user_id == token.user_id`，否则 404（**不是 403**，避免泄露存在性） |
| **T3** | 仓储层签名 | 所有查询方法强制 `user_id` + `pet_id` 关键字参数，无默认值 |

**升级选项**：PostgreSQL **Row Level Security**
（`SET LOCAL app.user_id` + policy），使隔离在数据库层不可绕过。
P0 不必，但架构上留位。

---

# 3. 异步与副作用（关键设计）

## 3.1 同步 / 异步边界

`DESIGN.md` §2.2 已定义「两种写入顺序」。**本文给出实现层判据**：

| 副作用 | 同步/异步 | 分级 | 理由 |
| --- | --- | --- | --- |
| **RECORD_EVENT 的写入** | **同步** | `required` | 响应要报告写入结果（W4）；失败必须如实告知 |
| **健康记录写入** | **同步** | `required` | 用户主动录入，丢失不可接受 |
| 其他 `memory_writer` | **异步** | `required` | 记忆缺失是正确性问题，但无需阻塞响应 |
| 视觉评估 | 异步 | `required` | 可重试；失败降级为「未评估」 |
| TTS | 异步 | `best_effort` | 失败静默（`DESIGN.md` §2.5） |
| 人设吐槽 | 异步 | `best_effort` | 娱乐层，可丢（`DESIGN.md` §3.2 T4） |

> **判据**：用户响应是否依赖其结果 → 同步；缺失是否影响正确性 → `required`。

## 3.2 副作用的可靠投递

### P0 决策：**同步写，不用队列**

> 初稿在这里拿着两个方案不放手（「先同步，出现性能问题再切，但切换成本不低」）——
> 那不是一个决策，是**决策逃避**。本节补上明确结论。

| | **P0（采用）** | P1（演进） |
|---|---|---|
| `memory_writer` | **同步**（请求内） | 异步 + outbox |

**P0 选同步的四条理由**：

| # | 理由 |
| --- | --- |
| 1 | `memory_writer` 主要是 LLM 抽取（约 1 s），而用户本已在等 ≈3.6 s，**边际成本可接受** |
| 2 | 同步能**立即发现失败** → 直接满足 W4「写入失败必须如实报告」 |
| 3 | 省掉 worker、队列、卡死回收、幂等键**整套分布式副作用机制** |
| 4 | **切换成本低**：把调用点从 `await writer.apply()` 换成 `await jobs.enqueue()`，接口不变 |

> **不要为想象中的规模引入真实复杂度。** P0 只有四步闭环，
> 队列的唯一收益（不阻塞响应）在 P0 并不需要。

### P1 · 演进为事务性 outbox（不用 BackgroundTasks）

**问题**：`BackgroundTasks` 在进程崩溃时**丢任务**。
若响应已返回而记忆没写，用户以为记住了 → 静默失败（`DESIGN.md` 代价矩阵里的 0.9 项）。

**方案**：业务写入与入队**同一事务**。

```python
async def handle_turn(...):
    async with db.transaction():
        # ① 业务写入（若有）
        written = await store.memory_writer.apply(decisions)
        # ② 入队副作用 —— 同事务，原子
        for job in build_jobs(written, ...):
            await store.jobs.enqueue(job)   # INSERT INTO job_queue
    return response
    # 事务提交后，worker 才能看到任务。进程崩溃 → 任务仍在库里 → 可恢复
```

**worker 取任务**：

```sql
WITH picked AS (
  SELECT id FROM job_queue
  WHERE status = 'pending' AND run_after <= now()
  ORDER BY id
  FOR UPDATE SKIP LOCKED          -- 多 worker 安全，无需额外锁
  LIMIT 1
)
UPDATE job_queue j
SET status = 'running', locked_by = $1, locked_at = now(),
    attempts = attempts + 1
FROM picked WHERE j.id = picked.id
RETURNING j.*;
```

| 机制 | 说明 |
| --- | --- |
| 重试 | 指数退避，`attempts >= max_attempts` → `failed` |
| 卡死回收 | `status='running' AND locked_at < now() - interval '5 min'` → 重置 `pending` |
| `required` 最终失败 | **告警**（§6.4） |
| `best_effort` 最终失败 | 静默丢弃 |

> 为什么不用 Redis/Celery：与业务数据同事务是 outbox 的本质要求。
> **单库换来的是「任务绝不丢失」这个保证**，代价是吞吐上限。P0 完全够用。

## 3.3 幂等

| 场景 | 机制 |
| --- | --- |
| 客户端重复提交 | `Idempotency-Key` 头 → `job_queue.idem_key` 唯一索引 |
| worker 重试 | 任务本身幂等（`memory_writer` 用 `dedup_key` 唯一索引兜底） |
| 记忆重复写入 | `idx_mem_dedup` 唯一索引 → 冲突即转「强化」 |

---

# 4. API 设计

## 4.1 端点（P0 四个，其余标注阶段）

| 方法 | 路径 | 说明 | 阶段 | 同步 |
| --- | --- | --- | --- | --- |
| `POST` | `/v1/pets` | 建宠物，返回 `pet_id` | P0 | 同步 |
| `POST` | `/v1/pets/{pet_id}/profile` | 上传 3–5 张照片建档案 | **P0** | 同步 |
| `GET` | `/v1/pets/{pet_id}/profile` | 读档案 | P0 | 同步 |
| `POST` | `/v1/chat` | 对话（含记忆检索与拒答） | **P0** | 同步 |
| `POST` | `/v1/interpret` | 行为解释（上传叫声） | **P0** | 同步 |
| `GET` | `/v1/memories` | 查看记忆列表 | P0 | 同步 |
| `DELETE` | `/v1/memories/{id}` | 删除记忆 | P1 | 同步 |
| `POST` | `/v1/moments` | 记录瞬间（照片/视频） | P1 | 同步（吐槽异步） |
| `POST` | `/v1/health/records` | 录入健康信号 | P1 | 同步 |
| `GET` | `/v1/health/assessment` | 取健康评估 | P1 | 同步 |
| `GET` | `/v1/jobs/{id}` | 查异步任务状态 | P1 | 同步 |

**P0 闭环对应**：① `/v1/pets` + `/profile`，② `/v1/chat`，③ `/v1/interpret`，④ 测试套件。

### 请求示例 · 行为解释

```http
POST /v1/interpret
Authorization: Bearer <token>
Content-Type: application/json

{
  "pet_id": "…",
  "audio_url": "…/meow.wav",
  "audio_kind": "cat_meow",
  "scene_description": "它对着门叫"
}
```

```jsonc
// 200
{
  "evidence_mode": "acoustic_plus_history",
  "candidates": [
    { "context": "door_attention", "posterior": 0.68,
      "display": "很可能是在关注门外动静", "log_odds": 0.75 }
  ],
  "evidence": [
    { "kind": "measured", "statement": "叫声时长 0.82s，长于它日常索食均值 0.41s",
      "source": "acoustic:duration", "value": 0.82, "reference": 0.41,
      "log_odds_contribution": 1.34 }
  ],
  "individualization": 0.12,
  "suggested_observation": "观察它是否伴随抓门、来回踱步",
  "limitations": "仅凭叫声无法确定真实需求；若伴随持续焦躁，建议就医观察。"
}
```

> 响应体**直接复用 `BehaviorInterpretation` 契约**——API 层不定义新结构。

## 4.2 鉴权与租户注入

```
Authorization: Bearer <token>
        │
        ▼
   验签 → user_id（服务端派生）
        │
        ▼
   路径/体中的 pet_id → 校验归属（pets.user_id == user_id）
        │
        ▼
   注入 AgentState.user_id / pet_id
```

| 规则 | 说明 |
| --- | --- |
| A1 | `user_id` **只能**从 token 派生。请求体中的 `user_id` 一律忽略 |
| A2 | 归属不符返回 **404**（非 403），避免泄露资源存在性 |
| A3 | 所有 store 调用必须携带二者（§2.6 T3） |

### P0 最小可行鉴权（**解决 U6**）

> ⚠️ **初稿把 U6 标为「未解决」，那是偷懒——最小方案成本很低。**

| # | 组件 | 实现 |
| --- | --- | --- |
| 1 | `users` 表 | `(id UUID PK, email TEXT UNIQUE, created_at)` |
| 2 | token 格式 | `base64url(user_id.exp).hmac_sha256(secret)` |
| 3 | 密钥 | `AUTH_SECRET` 环境变量 |
| 4 | 签发 | `make token USER=<uuid>`（开发用） |
| 5 | 校验 | FastAPI dependency：验签 + 验 `exp` → 注入 `user_id` |

**成本**：约 60 行代码 + 1 张表。
**效果**：`user_id` **不可伪造** → §2.6 的 T1/T2/T3 三个强制点全部成立，
**多用户隔离端到端可强制**。

**为何不用 JWT / OAuth**：本项目不需要跨服务、不需要第三方登录、不需要刷新令牌。
HMAC 签名 token 覆盖了「不可伪造」这个**唯一**需求。

**升级路径**：接入真实用户体系时替换校验依赖即可，仓储层与节点层无感知。

> 结论：U6 从「未解决」改为 **P0 解决，方案已定**。W1 随之关闭。

## 4.3 错误模型

```jsonc
{
  "error": {
    "code": "INSUFFICIENT_COVERAGE",
    "message": "数据不足，无法评估",
    "details": { "coverage": 0.14, "required": 0.3 },
    "trace_id": "…"
  }
}
```

| code | HTTP | 语义 |
| --- | --- | --- |
| `VALIDATION_FAILED` | 400 | 契约校验失败 |
| `UNAUTHORIZED` | 401 | 无/无效 token |
| `NOT_FOUND` | 404 | 资源不存在**或不属于当前用户** |
| `BUDGET_EXCEEDED` | 429 | 超出请求预算（§5.4） |
| `UPSTREAM_ERROR` | 502 | LLM / ASR / TTS 不可用 |
| `INSUFFICIENT_DATA` | 200 | **不是错误**：业务上的「无法评估」 |
| `DEGRADED` | 200 | 完成但降级，`details.degraded_note` 必有 |

> `INSUFFICIENT_DATA` 与 `DEGRADED` **必须是 200**——
> 它们是**正确的业务结果**，不是错误。用 4xx 表达会丢失「这是被设计的行为」这一信息。

## 4.4 流式

`/v1/chat` 支持 `Accept: text/event-stream`，按节点粒度推送：

```
event: node       data: {"node":"understand_input","latency_ms":412}
event: token      data: {"text":"团团"}
event: guard      data: {"passed":true}
event: done       data: {"tts_pending":true}
```

好处：**降级可见性**能实时传达（如 `guard` 事件带 `degraded_notice`）。

---

# 5. 运行时

## 5.1 延迟预算（P0 `/v1/chat`）

> ⚠️ **下表的数字是估算，不是实测。** LLM 调用耗时的估计误差可达 2–3 倍。
> **没有依据的数字比没有数字更危险**——它会让人误以为有把握。
> 实现后须用 `agent_node_latency_seconds`（§6.2）实测替换，
> 并据此反推 p95 是否在可接受范围（`DESIGN.md` §7.3 U5 的预算上限需由它定）。

| 阶段 | p50 | p95 | 备注 |
| --- | --- | --- | --- |
| 鉴权 + 契约校验 | 5 ms | 15 ms | 纯本地 |
| `understand_input` | 400 ms | 900 ms | LLM |
| `memory_retriever` | 60 ms | 150 ms | pgvector + 关系库 |
| `companion_agent` | 1200 ms | 2500 ms | LLM |
| `response_guard`（规则层） | 5 ms | 15 ms | 纯本地，**必跑** |
| `response_guard`（LLM 层） | 0 ms | 900 ms | **仅条件触发**，见下 |
| 序列化 | 5 ms | 15 ms | — |
| **合计** | **≈1.7 s** | **≈3.6 s** | |

### 守卫分两层（关键优化）

```
response_guard
  ├─ 规则层（必跑，<10ms）
  │    禁词、数值篡改、结构完整性、拟人化词表、展示层级
  └─ LLM 层（条件触发）
       仅当：草稿含事实性断言 / 规则层有疑问 / ANALYSIS 模式
```

> 若守卫每次都调 LLM，p95 会多约 1 s，且成本翻倍。
> **规则能覆盖的就不调模型**——这也与「守卫是强制点而非建议」一致：
> 规则层是硬门，LLM 层是补漏。

## 5.2 并发模型

| 组件 | 模型 |
| --- | --- |
| `api` | asyncio（FastAPI），I/O 密集 |
| 阻塞调用（librosa 特征提取） | `run_in_executor`（CPU 密集，勿阻塞事件循环） |
| `worker` | N 个协程轮询 `job_queue`，`SKIP LOCKED` 保证不重复取 |
| DB 连接池 | asyncpg，`min=2 max=10` |

## 5.3 超时、重试、降级矩阵

| 依赖 | 超时 | 重试 | 失败降级 |
| --- | --- | --- | --- |
| LLM（生成） | 20 s | 1 次（指数退避） | 兜底模板 + `DEGRADED` |
| LLM（路由） | 8 s | 1 次 | 转 `AMBIGUOUS`（澄清） |
| VLM（档案抽取） | 30 s | 1 次 | 单张跳过；全失败 → 报错 |
| ASR | 15 s | 1 次 | 提示用户打字 |
| TTS | 15 s | 2 次 | 静默 |
| 声学特征 | 10 s | 0 | 降为 `text_only`，**必须标注** |
| pgvector | 3 s | 1 次 | 跳过向量召回，只用关系库（静默） |

> **重试原则**：只重试**幂等**且**可能瞬时失败**的调用。
> 生成类调用重试会产生不同结果，因此**重试上限为 1**，避免成本失控。

## 5.4 请求预算（强制点）

`DESIGN.md` §7.3 U5。本文给出实现位置：

```python
@dataclass(frozen=True)
class RequestBudget:
    max_tokens: int = 12_000
    max_latency_ms: int = 8_000
    max_upstream_calls: int = 6
    max_subtasks: int = 4          # 规划层用
```

| 强制点 | 时机 |
| --- | --- |
| token 累计 | `infra/llm` 客户端包装器，每次调用后累加 |
| 延迟 | 编排层每个节点前后检查 `monotonic()` |
| 上游调用数 | 每次外部调用前 `budget.consume()` |

超限 → `BudgetExceeded` → **降级而非报错**：

| 场景 | 降级 |
| --- | --- |
| token 超限 | 跳过 LLM 守卫层，只用规则层 |
| 延迟超限 | 跳过 TTS、记忆写入（异步本就不阻塞） |
| 调用数超限 | 规划层裁掉 `optional=True` 的子任务 |

---

# 6. 可观测性

## 6.1 日志

结构化 JSON，每行含：

```jsonc
{ "ts":"…", "level":"info", "trace_id":"…", "user_id":"…", "pet_id":"…",
  "node":"behavior_interpreter", "latency_ms":1840, "tokens":{"in":812,"out":340},
  "decision":"evidence_mode=acoustic_plus_history top=door_attention p=0.68",
  "degraded":false }
```

| 规则 | 说明 |
| --- | --- |
| L1 | **不记录**原始音频、照片 URL、健康记录值（隐私） |
| L2 | `decision` 字段记录节点关键决策，便于事后归因 |
| L3 | 日志与 `node_trace` 用同一 `trace_id` 关联 |

## 6.2 指标

| 指标 | 类型 | 用途 |
| --- | --- | --- |
| `agent_node_latency_seconds{node}` | Histogram | 定位瓶颈（§5.1 预算对账） |
| `llm_tokens_total{node,model}` | Counter | 成本 |
| `llm_errors_total{node,kind}` | Counter | 上游健康度 |
| `job_queue_depth{kind,status}` | Gauge | 积压 |
| `job_queue_failed_total{kind}` | Counter | `required` 任务最终失败 |
| `guard_violation_total{type,severity}` | Counter | 守卫命中分布 |
| **`invariant_violation_total{name}`** | Counter | **见 §6.4** |
| `retrieval_empty_total` | Counter | 拒答频率（幻觉风险的反向指标） |

## 6.3 trace

一次请求一个 `trace_id`，一个节点一个 span。
`AgentState.node_trace` 与 `errors` 已是契约（`app/schemas/trace.py`）。

### 会话与调用链：`session_id` / `trace_id`

| 标识 | 粒度 | 作用 |
| --- | --- | --- |
| `session_id` | **跨请求**（一次会话多次请求） | 会话记忆注入（最近 N 轮）；落库实体的追溯键 |
| `trace_id` | 单请求 | 与 LangSmith `run_id` **同源**，可跳到对应调用树 |

两者都由中间件从 `X-Session-Id` / `X-Trace-Id` 请求头解析，缺省由服务端生成，
并通过同名响应头回传 → **12 个端点不可能漏传**。

> `session_id` **不是隔离键**：隔离仍靠 `user_id` + `pet_id`（§2.6）。
> 它只回答「这是哪一轮」，不参与权限判断。

接入 LangSmith 后每次请求是一棵可回放的调用树；关联方式是**同一个标识**
（`trace_id` 是 UUID 时直接作 `run_id`，否则 `uuid5` 确定性派生），
而不是维护一张对照表。观测失败 **fail-soft**：不影响对话，但启用状态与错误暴露在 `/healthz`。

## 6.4 把不变量运维化（关键）

`DESIGN.md` §4 的 I1–I18 目前**只是测试断言**。它们还应该是**运行时告警指标**：

```python
INVARIANTS = {
    "I1_no_active_system_inference": "SELECT count(*) FROM memory_events "
        "WHERE source='system_inference' AND status='active'",
    "I11_coverage_gate": "SELECT count(*) FROM health_assessments "
        "WHERE coverage < 0.3 AND level <> 'INSUFFICIENT_DATA'",
}

async def check_invariants(db):
    for name, sql in INVARIANTS.items():
        n = await db.scalar(sql)
        invariant_violation_total.labels(name=name).set(n)
```

| 规则 | 说明 |
| --- | --- |
| V1 | 每个不变量一条只读 SQL，周期执行（如 5 分钟） |
| V2 | **任何非零值立即告警**——不变量被破坏说明有人绕过了应用层 |
| V3 | 这些查询**不修改数据**，只观测 |

> **测试断言挡的是提交时，运行指标挡的是运行时。**
> 两者都需要：测试无法发现线上数据被脚本写坏。

---

# 7. 安全与隐私

## 7.1 威胁与对策

| # | 威胁 | 对策 |
| --- | --- | --- |
| S1 | 伪造 `user_id` 读他人数据 | `user_id` 仅从 token 派生（§4.2 A1）；升级 RLS |
| S2 | 通过 `pet_id` 探测资源存在性 | 归属不符返回 404（A2） |
| S3 | LLM Prompt 注入（用户输入含指令） | 输入与系统提示**结构分离**；`response_guard` 检查输出越界 |
| S4 | 上传恶意媒体 | 类型/大小校验、**剥离 EXIF**、病毒扫描（生产） |
| S5 | 成本攻击（刷接口） | 速率限制 + 请求预算（§5.4） |
| S6 | 健康数据泄露 | 字段级加密（§7.2） |
| S7 | 越权删除 | 硬删除前校验归属 + **二次确认** |

## 7.2 健康数据加密（对应 U16）

```sql
-- 敏感字段用应用层加密后存 bytea，密钥来自 KMS/环境变量
ALTER TABLE health_records
  ADD COLUMN value_enc BYTEA;    -- value_num/value_text 加密版本
```

| 规则 | 说明 |
| --- | --- |
| E1 | 加密/解密在 `store` 层，节点与 API 无感知 |
| E2 | 密钥**不落库**，来自环境变量或 KMS |
| E3 | 密钥轮换：存 `key_version`，支持双读 |
| E4 | 用户删除 = 硬删除 + 级联删除派生评估 |

## 7.3 速率限制

| 维度 | 限制（P0 建议） |
| --- | --- |
| 每用户 `/v1/chat` | 20 次/分钟 |
| 每用户 `/v1/interpret` | 10 次/分钟 |
| 每用户媒体上传 | 100 MB/日 |
| 全局 LLM 调用 | 由请求预算 + 队列深度保护 |

实现：Redis 令牌桶；无 Redis 时用 DB 计数（P0 可接受）。

---

# 8. 部署与本地开发

## 8.1 `docker-compose.yml`

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: petagent
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes: [ "pgdata:/var/lib/postgresql/data" ]
    ports: [ "5432:5432" ]

  minio:                      # 开发期可省，用本地 FS
    image: minio/minio
    command: server /data --console-address ":9001"
    ports: [ "9000:9000", "9001:9001" ]

  api:
    build: .
    command: uvicorn app.api.main:app --host 0.0.0.0 --port 8000
    env_file: [ .env ]
    depends_on: [ db ]
    ports: [ "8000:8000" ]

  worker:
    build: .
    command: python -m app.worker
    env_file: [ .env ]
    depends_on: [ db ]

volumes: { pgdata: {} }
```

**`api` 与 `worker` 同一镜像**，仅 `command` 不同——保证依赖与契约版本一致。

## 8.2 开发命令（对应 `DESIGN.md` §6.5）

```bash
make up        # docker compose up -d
make migrate   # 建表 + 索引
make demo      # 灌样例数据 + 起 Web UI
make eval      # 跑评测，输出报告
make test      # 契约不变量测试（92 项，1 秒）
make check     # 文档一致性检查
```

## 8.3 配置

| 变量 | 用途 | 必填 |
| --- | --- | --- |
| `DATABASE_URL` | Postgres 连接（**存储层未实现**，当前为内存 store） | ⬜ |
| `PET_AGENT_PROVIDER` | 供应商：`dashscope` / `ark`（缺省按 key 名推断） | ⬜ |
| `DASHSCOPE_API_KEY` / `ARK_API_KEY` | 供应商主密钥 | 二者至少一个 |
| `PET_AGENT_API_KEY` | 裸密钥（不携带供应商身份 → 用**有文档的默认供应商**） | ⬜ |
| `PET_AGENT_<能力>_{MODEL,BASE_URL,PATH,API_KEY,TIMEOUT_S}` | **逐能力**覆盖（LLM / VISION / EMBED / MULTIMODAL） | ⬜ |
| `PET_AGENT_AUTH_SECRET` | HMAC 签名密钥（兼容 `AUTH_SECRET` / `AUTH_DEV_TOKEN`）。**缺则拒绝启动** | ✅ |
| `PET_AGENT_PRIORS_PATH` | 先验表路径（缺省 `data/priors/catmeows_stats.json`） | ⬜ |
| `PET_AGENT_MEDIA_TIMEOUT_S` | 下载待分析音频的超时（秒） | ⬜ |
| `PET_AGENT_TRACING` | 观测开关。缺省「有 key 就启用」；设为 `0` 可**压过** key（CI / 测试） | ⬜ |
| `PET_AGENT_LANGSMITH_API_KEY` | LangSmith 密钥（兼容 `LANGSMITH_API_KEY` / `LANGCHAIN_API_KEY`） | ⬜ |
| `PET_AGENT_LANGSMITH_PROJECT` | 项目名（缺省 `pet-agent`） | ⬜ |
| `PET_AGENT_LANGSMITH_ENDPOINT` | 自托管 / 区域端点 | ⬜ |
| `S3_ENDPOINT` / `S3_BUCKET` | 对象存储 | ⬜（可退化本地 FS） |
| `HEALTH_ENC_KEY` | 健康数据加密密钥 | P1 |
| `MOCK_PROVIDER` | `1` 时全部外部调用走桩（离线可跑） | ⬜ |

> 供应商与逐能力覆盖见 `DESIGN.md` §3.14（决策 D46）。
> 会话标识与观测见 `DESIGN.md` §3.15（决策 D47）。
> 多厂商后 `/healthz` 会报出实际使用的 `*_model` / `*_endpoint` / `mixed`（**不含密钥**）。
>
> `MOCK_PROVIDER=1` 是 `DESIGN.md` §6.5「mock provider，离线可跑全部评测」的实现开关。

---

# 9. 技术选型表

| 层 | 选型 | 理由 | 备选与放弃原因 |
| --- | --- | --- | --- |
| 语言 | **Python 3.10**（当前环境 3.10.12） | 生态（librosa / LangGraph） | 初稿误写 3.11；改为与实际环境一致。**不盲目升级**——无收益 |
| Web | FastAPI | async、Pydantic 原生契合契约层 | Flask：无 async |
| 编排 | LangGraph | 条件路由 + 状态归约 + `Send` fan-out | 手写状态机：可做但 trace 与归约要自己实现 |
| 模型 | DashScope（Qwen） | 与 `ai-file-agent` 同 Key；OpenAI 兼容模式便于换 | OpenAI：国内访问需代理 |
| 关系库 | PostgreSQL 16 | 见 §2.1 | MySQL：JSONB / 数组 / RLS 能力弱 |
| 向量 | pgvector | 同库 JOIN + 一致性 | Milvus：跨库一致性成本 |
| 队列 | PG 表 + SKIP LOCKED | outbox 同事务 | Redis+RQ：事务跨界 |
| 声学 | librosa + pyin | `f0_slope` 有文献支撑 | 自训练：无数据 |

---

# 10. 已知薄弱点（诚实清单）

| # | 薄弱点 | 影响 | 处置 |
| --- | --- | --- | --- |
| ~~W1~~ | ~~P0 鉴权是单用户 dev token~~ | **已关闭** | §4.2 给出 P0 最小可行鉴权（HMAC token + `users` 表，约 60 行），`DESIGN.md` U6 随之解决 |
| W2 | 队列 + outbox 对 P0 可能偏重 | P0 只有四步闭环，同步写也许够 | 可先同步，出现性能问题再切；但**切换成本不低**，故先设计 |
| W3 | 带过滤 ANN 的召回问题**只标注未实测** | 可能实际召回不足 K | 上线前实测（E11） |
| W4 | 无 schema 迁移策略 | 变更无路径 | 建议 Alembic，**本文未展开** |
| W5 | 无备份/恢复方案 | 数据丢失风险 | 生产必需，**本文未展开** |
| W6 | 测试分层未定义 | 只有契约测试，无集成/E2E 策略 | 需补：节点单测（mock store）+ 编排集成测试 |
| W7 | 成本只有 token 预算，**无单价估算** | 无法判断经济可行性 | 需按 DashScope 单价折算单次成本 |
| W8 | 与 `ai-file-agent` 的复用关系未写 | 重复建设的可能 | 可复用：DashScope 调用封装、Milvus 经验、MySQL 模式 |
| W9 | 缓存一致性边界未定义 | Session 在 Redis，但 LangGraph 状态在哪？ | 需明确：`AgentState` 只活于单次请求，跨请求靠 Session + DB |
| W10 | 单点 | 单 api 实例 | 无状态设计可水平扩，但 worker 与 Session 需确认 |

---

# 11. 与 DESIGN.md 的对应

| `DESIGN.md` | 本文实现位置 |
| --- | --- |
| §1.5 边界 1（外观校验非个体鉴定） | §2.3 向量维度与用途 |
| §2.1 分层 | §1.2 部署拓扑 + §1.3 依赖方向 |
| §2.2 唯一写入点 | §3.1 同步/异步边界 + §3.2 outbox |
| §2.5 失败降级 | §5.3 超时/重试/降级矩阵 |
| §3.5 记忆三层与检索管线 | §2.2 schema + §2.3 索引 |
| §4 安全不变量 I1–I18 | §2.2 DB 约束 + §6.4 运行时指标 |
| §6.3 基线与消融 | §8.3 `MOCK_PROVIDER` |
| §6.5 可复现要求 | §8.2 make 命令 |
| §7.3 U5 请求预算 | §5.4 |
| §7.3 U6 鉴权 | §4.2（**P0 解决**：HMAC token + `users` 表） |
| §7.3 U16 健康数据加密 | §7.2 |
