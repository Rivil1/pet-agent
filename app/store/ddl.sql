-- =============================================================================
-- pet-agent MySQL 表结构
--
-- ## 三条贯穿全表的规则
--
-- 1. **每张业务表都有 `user_id` + `pet_id`，且索引以它们打头。**
--    这不是「顺便加的过滤条件」—— 它是决策 D5（多租户隔离）在物理层的落点。
--    索引顺序也刻意如此：隔离过滤必须走索引，否则「忘了加 where」
--    会退化成全表扫描而不是报错。
--
-- 2. **所有时间列都是 `DATETIME(6)` 且存 UTC。**
--    MySQL 的 DATETIME 不带时区，而领域模型全是 aware UTC。
--    不在这一层说清楚，读写就会各按各的时区解释，而那是**静默的**：
--    时间会偏移，没有任何报错。
--
-- 3. **`vector` 列存在 MySQL 里，尽管检索走 Milvus。**
--    向量是「可重建的派生数据」，但重建需要**重新调用 embedding API**（要花钱）。
--    持久化它，Milvus 就是纯粹的索引，丢了幂等回填即可 —— 不必重新付费。
--
-- ## 关于 `COLLATE utf8mb4_unicode_ci`
--
-- 必须显式声明。默认排序规则下，`subject` 这类列的等值比较可能不区分大小写
-- 甚至不区分重音 —— 而「同一个 subject」是冲突判定的前提，
-- 让排序规则去决定它是否相等是危险的。
-- =============================================================================


-- ─────────────────────────────────────────────────────────────
-- 宠物档案
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pets (
    pet_id                  CHAR(36)     NOT NULL COMMENT 'UUID',
    user_id                 VARCHAR(128) NOT NULL COMMENT '归属，来自 token，不可客户端指定',
    name                    VARCHAR(64)  NOT NULL,
    species                 VARCHAR(16)  NOT NULL DEFAULT 'cat',
    breed                   VARCHAR(64)  NULL,

    -- 视觉档案整块存 JSON：它是「一次分析的整体产物」，
    -- 拆成列会让「哪些字段参与了多图取交集」这个语义丢失。
    visual                  JSON         NOT NULL,
    -- 身份锚点与不稳定特征**分列存放**，因为它们的可信度不同（见 DESIGN §2.3）。
    -- 合并成一个列表会让调用方分不清哪个能用于校验。
    must_keep_features      JSON         NOT NULL,
    observed_but_unstable   JSON         NOT NULL,
    traits                  JSON         NOT NULL,

    appearance_embedding_id VARCHAR(128) NULL,
    identity_prompt         TEXT         NOT NULL,
    created_at              DATETIME(6)  NOT NULL COMMENT 'UTC',
    updated_at              DATETIME(6)  NOT NULL COMMENT 'UTC',

    PRIMARY KEY (pet_id),
    KEY idx_pets_tenant (user_id, created_at)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '宠物档案（身份锚点）';


-- ─────────────────────────────────────────────────────────────
-- 记忆事件
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memories (
    memory_id     CHAR(36)       NOT NULL COMMENT 'UUID',
    user_id       VARCHAR(128)   NOT NULL,
    pet_id        CHAR(36)       NOT NULL,
    session_id    VARCHAR(64)    NULL COMMENT '追溯用，不是隔离键',
    layer         VARCHAR(16)    NOT NULL COMMENT 'profile|episode|session',
    event_type    VARCHAR(16)    NOT NULL,
    subject       VARCHAR(255)   NOT NULL COMMENT '冲突判定的主键之一',
    content       TEXT           NOT NULL,
    polarity      VARCHAR(16)    NOT NULL,
    valid_from    DATETIME(6)    NULL,
    valid_to      DATETIME(6)    NULL,
    occurred_at   DATETIME(6)    NULL,
    source        VARCHAR(32)    NOT NULL,
    confidence    FLOAT          NOT NULL,
    status        VARCHAR(24)    NOT NULL,
    support_count INT            NOT NULL DEFAULT 1 COMMENT '强化次数，数据飞轮的核心计数',
    last_seen_at  DATETIME(6)    NULL,
    superseded_by CHAR(36)       NULL,
    supersedes    JSON           NOT NULL COMMENT '被我取代的 memory_id 列表',

    -- 去重键：非空时 (pet_id, dedup_key) 唯一。
    -- MySQL 的唯一索引允许多个 NULL，正好表达「只有带键的行才去重」。
    dedup_key     VARCHAR(191)   NULL,

    -- float32 打包的向量（1024 维 = 4096 字节）。供 Milvus 索引重建。
    vector        VARBINARY(8192) NULL,

    created_at    DATETIME(6)    NOT NULL,
    updated_at    DATETIME(6)    NOT NULL,

    PRIMARY KEY (memory_id),

    -- 架构文档 §2.2 的 idx_mem_dedup：重复写入应转为**强化**，不是插入第二条
    UNIQUE KEY uq_mem_dedup (pet_id, dedup_key),

    -- 隔离 + 召回：partial index 的等价物（WHERE status='active' 由查询侧给出）
    KEY idx_mem_recall (user_id, pet_id, status),
    KEY idx_mem_session (user_id, pet_id, session_id),
    -- 冲突检测按 subject 找候选
    KEY idx_mem_subject (user_id, pet_id, subject, status),

    -- 不变量 I1（防自我强化）在**数据库层**再挡一次。
    -- 应用层挡的是「正常路径写错」，这里挡的是「绕过契约的写入路径」
    -- （脚本、迁移、未来的新代码）。见 docs/04-memory.md §3.5 R1。
    CONSTRAINT ck_mem_no_active_inference
        CHECK (NOT (source = 'system_inference' AND status = 'active')),

    -- 时间范围自洽
    CONSTRAINT ck_mem_valid_range
        CHECK (valid_from IS NULL OR valid_to IS NULL OR valid_from <= valid_to)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '记忆事件（三层）';


-- ─────────────────────────────────────────────────────────────
-- 叫声记录（案例推理的样本库）
--
-- 与 memories 分开：它带声学特征，且**只能由主人的标注产生**。
-- 混进通用记忆会让「哪些是主人说的、哪些是系统推断的」这个区分消失。
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS meow_records (
    record_id   CHAR(36)     NOT NULL,
    user_id     VARCHAR(128) NOT NULL,
    pet_id      CHAR(36)     NOT NULL,
    session_id  VARCHAR(64)  NULL,
    context     VARCHAR(32)  NOT NULL COMMENT '主人标注的情境（固定词表）',
    features    JSON         NOT NULL COMMENT '声学特征，服务端产生，不接受客户端提交',
    actions     JSON         NOT NULL COMMENT '主人可观察到的动作（固定词表）',
    resolution  TEXT         NULL COMMENT '后来什么让它停了 —— k-NN 最有价值的一列',
    recorded_at DATETIME(6)  NOT NULL,
    source      VARCHAR(32)  NOT NULL,
    status      VARCHAR(24)  NOT NULL,

    PRIMARY KEY (record_id),
    KEY idx_meow_tenant (user_id, pet_id, status, recorded_at)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '主人标注的叫声记录（案例推理样本）';


-- ─────────────────────────────────────────────────────────────
-- 待标注解释
--
-- **特征只存在服务端**：客户端标注时只发情境/动作/结果，
-- 服务端按 id 取回自己存的特征。否则 MEASURED 的承诺失效。
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pending_interpretations (
    interpretation_id CHAR(36)     NOT NULL,
    user_id           VARCHAR(128) NOT NULL,
    pet_id            CHAR(36)     NOT NULL,
    session_id        VARCHAR(64)  NULL,
    features          JSON         NOT NULL,
    evidence_mode     VARCHAR(32)  NOT NULL,
    candidates        JSON         NOT NULL,
    created_at        DATETIME(6)  NOT NULL,

    PRIMARY KEY (interpretation_id),
    KEY idx_pending_tenant (user_id, pet_id, created_at)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '待主人标注的解释存档';


-- ─────────────────────────────────────────────────────────────
-- 会话消息（日报输入 + 会话记忆注入）
--
-- 与 memories 是**两个不同的东西**：这里是原始对话，
-- 那里是总结的产物。拿 memories 当日报输入会形成循环（D32）。
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_messages (
    message_id CHAR(36)     NOT NULL,
    user_id    VARCHAR(128) NOT NULL,
    pet_id     CHAR(36)     NOT NULL,
    session_id VARCHAR(64)  NULL,
    role       VARCHAR(16)  NOT NULL COMMENT 'user|assistant',
    content    MEDIUMTEXT   NOT NULL,
    at         DATETIME(6)  NOT NULL,

    PRIMARY KEY (message_id),
    KEY idx_msg_window (user_id, pet_id, at) COMMENT '日报按时间窗口取',
    KEY idx_msg_session (user_id, pet_id, session_id, at) COMMENT '会话记忆取最近 N 条'
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '会话消息原文';


-- ─────────────────────────────────────────────────────────────
-- 健康记录
--
-- 数据策略与记忆不同（静态加密 / 730 天保留 / 级联硬删 / 显式同意），
-- 所以是独立的表，不是 memories 的一个 event_type（D38）。
--
-- `value` 在领域模型里是 `float | bool | str | None` 的联合。
-- 拆成 kind + 三列而不是序列化成 JSON：
--   **红旗求值要按数值比较**，把数字藏进 JSON 里就再也用不上索引与比较运算。
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_records (
    record_id       CHAR(36)     NOT NULL,
    user_id         VARCHAR(128) NOT NULL,
    pet_id          CHAR(36)     NOT NULL,
    session_id      VARCHAR(64)  NULL,

    -- ⚠️ `signal` 是 MySQL 的保留字（SIGNAL 语句），必须加反引号。
    -- 不加的话报的是语法错误，而错误信息指向的是下一行，
    -- 很容易往错误的方向查。
    `signal`        VARCHAR(64)  NOT NULL,

    value_kind      VARCHAR(16)  NOT NULL COMMENT 'number|bool|text|absent',
    value_number    DOUBLE       NULL,
    value_bool      TINYINT(1)   NULL,
    value_text      VARCHAR(255) NULL,

    unit            VARCHAR(32)  NULL,
    recorded_at     DATETIME(6)  NOT NULL,
    source          VARCHAR(32)  NOT NULL,
    -- ⚠️ `sensitive` 从 MySQL 8.0.13 起也是保留字（INTERSECT/EXCEPT 语法引入）。
    `sensitive`     TINYINT(1)   NOT NULL DEFAULT 1,
    consent_version VARCHAR(16)  NULL,
    retention_days  INT          NULL,
    created_at      DATETIME(6)  NOT NULL,

    PRIMARY KEY (record_id),
    KEY idx_health_tenant (user_id, pet_id, recorded_at),
    KEY idx_health_signal (user_id, pet_id, `signal`, recorded_at),

    -- kind 与「哪个值列非空」必须一致 —— 否则读出来的 value
    -- 会变成 None，而 None 在红旗求值里意味着「不知道」，
    -- 于是一条明明有值的记录会被静默当成缺失。
    CONSTRAINT ck_health_value_shape CHECK (
        (value_kind = 'number' AND value_number IS NOT NULL)
        OR (value_kind = 'bool' AND value_bool IS NOT NULL)
        OR (value_kind = 'text' AND value_text IS NOT NULL)
        OR (value_kind = 'absent'
            AND value_number IS NULL AND value_bool IS NULL AND value_text IS NULL)
    )
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '健康信号记录';


-- ─────────────────────────────────────────────────────────────
-- 健康评估（派生结果，随记录级联硬删）
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_assessments (
    assessment_id      CHAR(36)     NOT NULL,
    user_id            VARCHAR(128) NOT NULL,
    pet_id             CHAR(36)     NOT NULL,
    as_of              DATETIME(6)  NOT NULL,
    level              VARCHAR(32)  NOT NULL COMMENT 'L1|L2|L3|NO_DEVIATION_DETECTED|INSUFFICIENT_DATA',
    coverage           FLOAT        NOT NULL,
    coverage_note      TEXT         NOT NULL,
    signals_assessed   JSON         NOT NULL,
    signals_missing    JSON         NOT NULL,
    findings           JSON         NOT NULL,
    red_flags_triggered JSON        NOT NULL,
    recommendation     TEXT         NOT NULL,
    disclaimer         TEXT         NOT NULL,
    must_not_be_read_as TEXT        NOT NULL,
    rule_version       VARCHAR(32)  NULL,

    PRIMARY KEY (assessment_id),
    KEY idx_assess_tenant (user_id, pet_id, as_of)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '健康评估结果（派生，级联硬删）';


-- ─────────────────────────────────────────────────────────────
-- 瞬间（日记本体，docs/11 §2.1）
--
-- 与 memories **刻意分开**：
--   memories = 可检索的事实（走向量召回）
--   moments  = 一次流水（按时间线走）
--
-- 把每张照片都塞进记忆检索，会让「它怕吸尘器」这类稳定事实
-- 被日常流水淹没 —— 而那正是记忆层要防的事。
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS moments (
    moment_id   CHAR(36)     NOT NULL,
    user_id     VARCHAR(128) NOT NULL,
    pet_id      CHAR(36)     NOT NULL,
    session_id  VARCHAR(64)  NULL,
    media_url   TEXT         NOT NULL COMMENT '照片 / 短视频地址',
    note        VARCHAR(500) NULL COMMENT '用户可选的一句话（可为空，不是缺失）',
    scene       VARCHAR(24)  NOT NULL COMMENT '系统抽取的场景标签；没写就不猜 = other',
    captured_at DATETIME(6)  NOT NULL COMMENT '记录时刻 —— 瞬间的意义就在于「刚刚发生」',
    created_at  DATETIME(6)  NOT NULL,

    PRIMARY KEY (moment_id),
    -- 时间线按时间倒序取，索引顺序与之匹配
    KEY idx_moment_timeline (user_id, pet_id, captured_at),
    -- 按场景聚合（周回顾要按场景分组）
    KEY idx_moment_scene (user_id, pet_id, scene, captured_at)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_unicode_ci
  COMMENT = '瞬间记录（日记）';
