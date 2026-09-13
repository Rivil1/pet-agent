# 01 · 架构与节点契约

## 1. 分层

```
┌─────────────────────────────────────────────────────────────┐
│  API 层 (FastAPI)                                            │
│  /pets  /pets/{id}/profile  /chat  /interpret  /memories     │
│  职责：鉴权、租户注入、输入规范化、异步任务、SSE 流式         │
└───────────────────────────┬─────────────────────────────────┘
                            │  AgentState（强类型）
┌───────────────────────────▼─────────────────────────────────┐
│  编排层 (LangGraph)                                          │
│  ① 路由：快路径（单意图固定链路）/ 慢路径（planner 分解）     │
│  ② 调度：fan-out / fan-in / wave 分层                         │
│  ③ 守卫：放行 / 重写 / 降级                                   │
│  ④ 副作用：记忆写回                                           │
│  职责：状态流转、条件分支、并行调度、失败降级、全链路 trace    │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  能力层（纯函数式节点，无状态、可单测）                       │
│  understand_input │ planner │ execute_subtask                │
│  profile_analyzer │ memory_retriever │ companion_agent       │
│  behavior_interpreter │ result_merger │ response_guard       │
│  memory_writer    │ tts_output                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  基础设施层                                                  │
│  LLM/VLM(DashScope) │ ASR │ TTS │ 声学特征(librosa)          │
│  关系库 │ 向量库 │ 对象存储                                   │
└─────────────────────────────────────────────────────────────┘
```

**分层原则**

1. **节点无状态**：所有状态显式放在 `AgentState` 里传递，节点不持有跨请求状态。
   好处：可单测、可重放、可并行。
2. **能力层不直接访问数据库**：通过 `store` 抽象注入，便于替换与 mock。
3. **编排层不做业务判断**：只做路由、调度与降级，业务逻辑全在节点内。

## 2. 编排图

### 2.1 顶层：快路径 / 慢路径 / 澄清

```
                              ┌──────────────┐
   request ──────────────────►│  asr_input   │  (仅当 audio_kind=user_voice)
                              └──────┬───────┘
                                     ▼
                              ┌──────────────┐
                              │understand_   │  intent + confidence
                              │   input      │  + slots + sub_intents
                              └──────┬───────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │ 快路径                      │ 慢路径                      │ 澄清
        │ 单意图 & conf ≥ θ           │ COMPOUND 或 conf < θ_ask    │
        ▼                            ▼                            ▼
  ┌─────────────────┐        ┌───────────────┐            ┌────────────┐
  │  固定链路        │        │   planner     │            │ clarify_ask│
  │ （见 §2.2）      │        │               │            └──────┬─────┘
  └────────┬────────┘        └───────┬───────┘                   ▼
           │                         ▼                          END
           │                 ┌───────────────┐
           │                 │   fan-out     │  Send API，按 wave 派发
           │                 └───────┬───────┘
           │                         ▼
           │            ┌─────────────────────────┐
           │            │  execute_subtask × N    │  同 wave 内并行
           │            │  （跨 wave 串行）        │
           │            └───────────┬─────────────┘
           │                        ▼
           │            ┌─────────────────────────┐
           │            │     result_merger       │  合并 + 冲突检测
           │            └───────────┬─────────────┘
           │                        │
           └────────────┬───────────┘
                        ▼
                ┌───────────────┐
                │ response_guard│  身份一致 / 证据可追溯 / 过度确定性
                └───────┬───────┘
                        ▼
                ┌───────────────┐
                │  tts_output   │
                └───────┬───────┘
                        ▼
                ┌───────────────┐
                │ memory_writer │  数据飞轮：抽取候选记忆
                └───────┬───────┘
                        ▼
                       END
```

### 2.2 快路径内部链路

**写入顺序有两种，理由不同**（这是 docs/10-self-review.md A2 的修复）：

```
【类型一】响应需报告写入结果  →  写入早于响应
RECORD_EVENT ──► record_extractor ──► memory_writer ──► response_guard ──► tts_output ──► END
                                        ↑
                             响应要说「已记录：…」，
                             所以必须先知道写入的真实结果

【类型二】写入是副作用  →  写入晚于响应
CHAT / MEMORY_QUERY ──► memory_retriever ──► companion_agent ──┐
TRANSLATE_BEHAVIOR ──► behavior_interpreter ───────────────────┤
PROFILE_UPDATE ──► profile_analyzer ──► 用户确认 ──────────────┤
                                                               ▼
                            response_guard → tts_output → memory_writer → END
```

**唯一写入点原则**（消除原设计「快路径直接写 + 公共尾部也写」的重复写入矛盾）：

| # | 规则 |
| --- | --- |
| W1 | 任何节点与子任务**都不得直接写长期记忆**，只能产出 `candidate_memories` |
| W2 | `memory_writer` 是**唯一写入点**，也是唯一决策点（write / pending / reinforce / supersede / reject） |
| W3 | 所有面向用户的文本（**含「已记录：…」确认语**）都必须经过 `response_guard` |
| W4 | 写入失败时，响应必须如实报告未写成（正确性相关降级必须可见，见 §8） |

> **`RECORD_EVENT` 的「执行 + 明确反馈」**（§5.3）仍然成立，但它落在**写入顺序**与
> **措辞层**，不再是「绕过守卫直接写」。原设计中「`RECORD_EVENT` 不经过 `guard`」的说法已废止。

### 2.3 慢路径示例（复合请求）

```
用户：「团团最近总在门口叫，帮我记一下。它是不是想出去？
       顺便看看这个月是不是比上个月叫得多」

understand_input → intent=COMPOUND, sub_intents=[RECORD_EVENT, TRANSLATE_BEHAVIOR, MEMORY_QUERY]
        ↓
planner:
  T1  record_event("最近总在门口叫")            无依赖
  T3  aggregate_query(叫声次数, 本月 vs 上月)    无依赖
  T2  interpret_behavior(scene=门口)            依赖 T3（**先验依赖**）

  waves = [[T1, T3], [T2]]        最大并行度 2，串行深度 2
        ↓
wave 0（并行）:  T1 ✓            T3 ✓（发现本月显著多于上月）
        ↓
wave 1:          T2（door_attention 先验上调后执行）
        ↓
result_merger:   合并三份结果；记录 applied_prior_dependencies = ["T3->T2"]
        ↓
response_guard → tts_output → memory_writer → END
```

> **为什么 T2 不能和 T3 并行**：这不是数据依赖（T2 不消费 T3 的数据），
> 而是**先验依赖**——T3 的结论改变了 T2 输入的概率分布。详见 §6.2。

## 3. 节点契约

每个节点声明：**输入 → 输出 → 失败行为**。这是本项目的核心接口文档。

### 3.1 `asr_input`

| 项 | 内容 |
| --- | --- |
| 输入 | `raw_input.audio_url`（仅当 `audio_kind = user_voice`） |
| 输出 | `transcribed_text` |
| 依赖 | DashScope Paraformer |
| 失败 | 降级为 `transcribed_text = None` + `errors`，路由提示用户「没听清，可以打字吗」 |
| 备注 | Paraformer 文件识别要求**公网可访问 URL**，不接受本地文件/Base64。若走本地文件，改用流式 `Recognition`（WebSocket）。**此节点只处理用户语音，不处理猫叫。** |

### 3.2 `understand_input`

| 项 | 内容 |
| --- | --- |
| 输入 | `transcribed_text` 或 `raw_input.text`，`session_context` |
| 输出 | `intent`、`route_confidence`、`route_slots`（含 `sub_intents`） |
| 依赖 | LLM（结构化输出，enum 约束）+ 意图策略表 |
| 失败 | 按 `INTENT_POLICIES` 的**逐意图阈值**判定；低于 `min_confidence` → `AMBIGUOUS`，**不猜测** |
| 备注 | 必须读会话上下文做指代消解与省略补全（「它」「那它为什么这样」） |

### 3.3 `planner`

| 项 | 内容 |
| --- | --- |
| 输入 | `sub_intents`、`route_slots`、`pet_profile` |
| 输出 | `Plan`：`subtasks` + `waves` + `decomposition_note` |
| 依赖 | LLM（生成子任务）+ **代码**（依赖分析与拓扑排序） |
| 失败 | 计划不合法（拓扑违规/覆盖不全，由 `Plan` 校验器拦下）→ 重规划一次；仍失败 → 降级为快路径取第一个子意图 |
| 备注 | **依赖分析必须区分数据依赖与先验依赖**（§6.2）。这是本节点最核心的逻辑，不能交给 LLM 判断 |

### 3.4 `execute_subtask`

| 项 | 内容 |
| --- | --- |
| 输入 | `SubTask`（由 `Send` 派发）+ 上游子任务结果 |
| 输出 | `SubTaskResult`（经 `operator.add` 归约合并） |
| 依赖 | 按 `kind` 分派到对应能力节点 |
| 失败 | `optional=True` → 标记 `SKIPPED` 不阻塞；否则 `FAILED`，其后继任务也标记 `SKIPPED` |
| 备注 | 状态为 `DEGRADED` 时**必须**给出 `degraded_note`（契约层强制） |

### 3.5 `result_merger`

| 项 | 内容 |
| --- | --- |
| 输入 | `subtask_results[]` |
| 输出 | `plan_result` + 合并后的 draft |
| 依赖 | 代码 |
| 失败 | 合并冲突无法消解 → 保守合并（并列呈现，不强行归纳） |
| 备注 | 需记录 `applied_prior_dependencies`，用于验证先验依赖**真的被执行了**（否则依赖分析只是纸面功夫） |

### 3.6 `pet_profile_analyzer`

| 项 | 内容 |
| --- | --- |
| 输入 | `raw_input.image_urls[]`（3–5 张）、用户文字描述 |
| 输出 | `profile_draft`：`VisualProfile` + `must_keep_features[]` |
| 依赖 | Qwen-VL |
| 失败 | 单张失败跳过并记录；全部失败 → 明确报错，**不编造特征** |
| 备注 | 多图结果需**取交集**得到稳定特征，只出现一次的特征进 `observed_but_unstable`，不进 `must_keep_features` |

### 3.7 `memory_retriever`

| 项 | 内容 |
| --- | --- |
| 输入 | `pet_id`、查询文本、`intent` |
| 输出 | `retrieved_memories[]`（带 `score` 与 `source`） |
| 依赖 | 向量库 + 关系库，**强制按 `pet_id` 过滤** |
| 失败 | 空结果 → 返回空数组，由下游明确说「我没有相关记录」，**不得由 LLM 补全** |
| 备注 | 三层检索：档案层（主键直读）→ 事件层（混合检索）→ 会话层（直读） |

### 3.8 `companion_agent`

| 项 | 内容 |
| --- | --- |
| 输入 | `pet_profile`、`retrieved_memories`、`transcribed_text`、`session_context.emotion` |
| 输出 | `draft_response` |
| 依赖 | LLM |
| 失败 | 超时/异常 → 兜底模板回复 + 标记 `degraded` |
| 备注 | Prompt 必须显式注入「四层信息分离」规则；**AI 推测的每句必须能指回具体条目**；记忆以结构化 JSON 注入而非自由文本拼接 |

### 3.9 `behavior_interpreter`

| 项 | 内容 |
| --- | --- |
| 输入 | `raw_input.audio_url`（叫声）、`scene_description`、`pet_id` 的历史样本 |
| 输出 | `BehaviorInterpretation`：特征 + `IntentCandidate[]`（**后验概率**）+ `EvidenceItem[]` |
| 依赖 | librosa（特征）+ 向量检索（个体历史）+ 静态先验文件（数据集） |
| 失败 | 特征提取失败 → 退化为纯文本推断，契约层**强制** `evidence_mode = text_only` 且**禁止**输出数值置信度 |
| 备注 | **LLM 只做措辞化，不产生概率数字**。详见 [03-behavior-interpreter.md](03-behavior-interpreter.md) |

### 3.10 `response_guard`

| 项 | 内容 |
| --- | --- |
| 输入 | `draft_response`、`interpretation`、`pet_profile`、`retrieved_memories` |
| 输出 | `GuardResult`：`{ passed, violations[], rewritten }` |
| 检查项 | ① 身份一致性 ② 证据可追溯 ③ 过度确定性 ④ 越界（健康诊断等）⑤ 数值篡改 ⑥ 拟人化泄漏 |
| 失败 | 违规 → 重写一次；仍违规 → **降级为保守回答**（只陈述已知事实 + 建议观察） |
| 备注 | 这是「诚实性声明」的技术强制点。**prompt 是建议，节点是强制** |

### 3.11 `tts_output`

| 项 | 内容 |
| --- | --- |
| 输入 | `final_response`、角色音色配置 |
| 输出 | `tts_audio_url` |
| 依赖 | DashScope TTS |
| 失败 | 静默降级：只返回文本，`tts_audio_url = None` |
| 备注 | 非关键路径，失败不得影响主响应 |

### 3.12 `memory_writer`

| 项 | 内容 |
| --- | --- |
| 输入 | 本轮对话、`pet_profile`、已有记忆 |
| 输出 | `candidate_memories[]` → 过滤后写入，返回 `written_memory_ids[]` |
| 依赖 | LLM 抽取 + **代码**（去重、冲突判定、置信度路由） |
| 失败 | 抽取失败 → 本轮不写记忆（**宁可不写，不可写错**） |
| 备注 | 详见 [04-memory.md](04-memory.md)（数据飞轮） |

## 4. AgentState

```python
class AgentState(TypedDict, total=False):
    # 身份与租户（贯穿全链路，任何存储操作必须带这两个字段）
    user_id: str
    pet_id: str
    session_id: str
    trace_id: str

    # 原始输入
    raw_input: RawInput
    transcribed_text: str | None

    # 路由
    intent: InputIntent
    route_confidence: float
    route_slots: RouteSlots
    """含 sub_intents（COMPOUND 时）与 time_range（冲突检测必需字段）"""

    # 规划层（慢路径）
    plan: Plan | None
    current_wave: int
    """当前 wave 下标。planner 产出 waves，调度器据此逐层派发（§7）。"""

    subtask_results: Annotated[list[SubTaskResult], operator.add]
    plan_result: PlanResult | None

    # 检索上下文
    pet_profile: PetProfile | None
    retrieved_memories: list[MemoryItem]
    session_context: SessionContext

    # 能力节点产物
    profile_draft: PetProfileDraft | None
    acoustic_features: AcousticFeatures | None
    interpretation: BehaviorInterpretation | None
    draft_response: str | None

    # 守卫与输出
    guard_result: GuardResult | None
    final_response: str | None
    tts_audio_url: str | None

    # 副作用
    candidate_memories: list[MemoryEvent]
    written_memory_ids: list[str]

    # 可观测性
    errors: Annotated[list[NodeError], operator.add]
    node_trace: Annotated[list[NodeTrace], operator.add]
```

`subtask_results` / `errors` / `node_trace` 使用 `Annotated[..., operator.add]` 归约，
使 planner fan-out 出的多个并行分支能正确合并写入（LangGraph reducer 语义）。

## 5. 意图路由与策略

### 5.1 路由表

| intent | 触发条件 | 后续节点序列 |
| --- | --- | --- |
| `CHAT` | 日常闲聊、情绪表达 | retriever → companion → guard → tts → writer |
| `MEMORY_QUERY` | 询问「你还记得…」 | retriever → companion → guard → tts → writer |
| `TRANSLATE_BEHAVIOR` | 含猫叫音频，或描述叫声/行为 | interpreter → guard → tts → writer |
| `PROFILE_UPDATE` | 上传照片、修正档案 | analyzer → **用户确认** → writer |
| `RECORD_EVENT` | 明确告知要记一件事 | writer（**执行 + 明确反馈**） |
| `COMPOUND` | 复合请求（多子意图） | planner → fan-out → merger → guard → tts → writer |
| `AMBIGUOUS` | 置信度低于该意图的 `min_confidence` | clarify_ask |

**关键设计**：`AMBIGUOUS` 是**独立分支**而非 fallback 到 `CHAT`。
猜测用户意图会产生错误记忆，进而污染长期状态。宁可多问一句。

### 5.2 意图边界按「下游行为差异」划分

这是意图体系的设计依据（详见 [09-intent-and-planning.md](09-intent-and-planning.md) §3）：

> 两个意图如果在下游的**动作、约束、失败处理**上完全一致，就应该合并。
> 只有下游行为不同，才值得单独成一类。

`CHAT` 与 `MEMORY_QUERY` 走同一条链路，但分开——因为它们的**动作许可不同**：

| 维度 | `CHAT` | `MEMORY_QUERY` |
| --- | --- | --- |
| 性质 | 对话 | **审计**（用户要求系统复述它已知的） |
| 记忆写入 `memory_write` | 允许（推断进 `PENDING_CONFIRMATION`） | **禁止** |
| 召回数量 `retrieval_k` | 5 | 8 |
| 拒答严格度 `rejection_strict` | 常规 | 更严 |
| 响应报告写入 | 否 | 否 |

**核心差异是 `memory_write` 的动作许可，不是「有无拒答约束」。**
`retrieval_note` 是 `ContextInjectionBlock` 的默认字段，**所有意图都带**——
原文档声称「后者有强制拒答约束」不成立（见 [10-self-review.md](10-self-review.md) A1）。

审计不得产生新记录，这是审计的基本原则：**审计动作不能修改被审计对象**。
因此这是**动作许可差异**（行为差异），不是参数差异——边界判据仍然成立。

> 「禁止写回」是三条意图边界里**最弱但成立**的一条。若未来要简化意图体系，
> 它是第一个候选合并对象（可用 slot `readonly: bool` 表达）。

### 5.3 代价敏感阈值：不存在一条全局阈值

分类错误不是等价的：

| 误判方向 | 后果 | 用户何时发现 | 代价 |
| --- | --- | --- | --- |
| `CHAT` → `RECORD_EVENT` | 闲聊被写成记忆 | 很久以后 | **0.9（污染长期状态）** |
| `RECORD_EVENT` → `CHAT` | 以为记住了，实际没记 | 可能永不发现 | **0.9（静默失败）** |
| `CHAT` → `AMBIGUOUS` | 该问未问，强行猜测 | 不察觉 | 0.6 |
| `CHAT` → `TRANSLATE_BEHAVIOR` | 无证据推测（守卫可拦） | 立刻 | 0.5 |
| 任意 → `AMBIGUOUS` | 多问一句 | 立刻 | 0.15 |

完整矩阵在 `app/schemas/input.py` 的 `MISROUTE_COSTS`，策略在 `INTENT_POLICIES`。

**`RECORD_EVENT` 的特殊处理**：两个方向代价都是 0.9 —— 精确率与召回率都不能牺牲时，
**靠阈值无解**。因此改为**改变交互范式**：

```
识别到可能是记录请求
  → 直接执行写入
  → 明确回复「已记录：…。记错了请告诉我」
  → 假阳性由用户一句话纠正（极低纠错成本）
  → 假阴性由系统反馈消除
```

> 设计原则：**当某类错误的两个方向都昂贵时，不要优化分类器，要改变交互设计**——
> 让用户能以极低成本纠正错误。

## 6. 规划层：任务分解

### 6.1 分解的判据：按**决策点**拆，不按数据流拆

> 一个节点值得独立，当且仅当它**产生了影响后续路径的决策**，
> 或者**有需要单独降级的失败模式**。

| 节点 | 决策点 | 独立失败模式 | 值得独立 |
| --- | --- | --- | --- |
| `understand_input` | ✅ 路由决策 | ✅ | ✅ |
| `planner` | ✅ 分解 + 拓扑决策 | ✅ | ✅ |
| `behavior_interpreter` | ✅ 概率计算 + 分档 | ✅ | ✅ |
| `response_guard` | ✅ 放行/重写/降级 | ✅ | ✅ |
| `memory_writer` | ✅ 写入/待确认/拒绝 | ✅ | ✅ |
| `memory_retriever` | ◐ 仅取数 | ✅ 空结果需处理 | ✅（弱） |
| `tts_output` | ❌ 纯转换 | ✅ 失败可静默 | ◐ |

**反例**：如果把「检索 + 生成」合成一个节点，就无法回答
「是检索错了还是生成错了」，也无法单独评测。

### 6.2 依赖分析：两种依赖

**不是看数据流，而是看：一个子任务的输出是否改变另一个子任务的输入分布。**

| 依赖类型 | 含义 | 例 |
| --- | --- | --- |
| `DATA` 数据依赖 | 前置输出直接作为后置输入 | 检索 → 生成 |
| **`PRIOR` 先验依赖** | 前置输出**不进入**后置的输入数据，但**改变其输入的概率分布** | 聚合统计 → 行为解释的场景先验 |
| `RESOURCE` 资源依赖 | 访问互斥资源，须串行避免竞争 | 同时写同一宠物档案 |

**先验依赖是最容易被忽略、也最能体现分解能力的地方。** 举例：

```
T3 aggregate_query("本月 vs 上月 叫声次数") → 发现本月显著多于上月
T2 interpret_behavior(scene=门口)
```

朴素数据流分析会认为 T1/T2/T3 互不依赖、全部可并行——**漏掉真实依赖**。
实际上 T3 的结论应**上调 T2 中 `door_attention` 的场景先验**
（高频重复的叫声更可能是持续性诉求，而非偶发）。

**因此本节点必须把「依赖分析」交给代码**，而不是让 LLM 自由判断——
LLM 倾向于只看数据流，且同一输入两次调用可能给出不同拓扑。

### 6.2.1 先验依赖的实现机制

> 这一小节修复 docs/10-self-review.md F1：原先「先验依赖」只有声明、没有接口。

**产出方**：当某任务是某个 `PRIOR` 依赖的上游时，它必须在
`SubTaskResult.prior_adjustments` 中给出结构化的 `PriorAdjustment`：

```python
PriorAdjustment(
    target_task_id="T2",                            # 作用于哪个子任务
    target_field=PriorTargetField.SCENE_PRIOR,      # 只允许调整概率/权重/阈值
    target_key="door_attention",                    # 具体场景标签
    operation=PriorOperation.SCALE,                 # scale / delta / set
    value=1.4,
    rationale="本月叫声显著多于上月，持续性诉求可能性上升",  # 必填，供审计
    confidence=0.7,
)
```

**消费方**：`execute_subtask` 在本任务开始前，从上游结果中取出
所有 `target_task_id == 本任务` 的调整并应用到本地先验表，过程记入 `node_trace`。

**三条结构性约束**（均由契约层强制）：

| # | 约束 | 理由 |
| --- | --- | --- |
| P1 | `target_field` 只能是**概率 / 权重 / 阈值**类字段 | 防止先验注入退化为「直接写入结论」 |
| P2 | `rationale` 必填 | 无法说明依据的先验调整不可审计 |
| P3 | `PlanResult` 要求先验依赖**要么应用、要么说明跳过原因** | 禁止静默丢弃，否则依赖分析只是纸面功夫 |

P3 的对账机制：

```python
PlanResult(
    declared_prior_dependencies=["T3->T2"],
    applied_prior_dependencies=["T3->T2"],      # 或
    skipped_prior_dependencies=[PriorSkip(dependency="T3->T2", reason="T3 失败")],
)
# 缺失任一项 → 构造时抛错
```

**可测试性**：评测可以断言「T3 产出的 adjustment 确实改变了 T2 的后验」——
这才让「先验依赖」从名词变成了机制（对应 docs/09 §10 的「依赖正确率」指标）。

### 6.3 粒度控制

| 拆太粗 | 拆太细 |
| --- | --- |
| 无法定位错误 | 编排开销、状态膨胀 |
| 无法独立降级 | 每步引入 LLM 不确定性 |
| 无法单独评测 | 延迟累加 |

**粒度原则**：停在**决策点**，不要停在中转步骤。

### 6.4 防止过度分解：快慢路径分流

简单请求不该走规划层：

```
understand_input
  ├─ 单意图 & conf ≥ θ        ──► 快路径（固定链路）       ~1s
  ├─ COMPOUND 或 conf < θ_ask  ──► 慢路径（planner 分解）  ~4s
  └─ conf < min_confidence     ──► clarify_ask
```

「过度分解率」是评测项（[DESIGN.md](DESIGN.md) §6.2 E18）：简单请求被错误地走了 planner 的比例。

## 7. 并行与调度

### 7.1 机制

使用 LangGraph 的 `Send` API 做动态 fan-out（子任务数量在运行期才确定，静态图无法表达）：

```python
from langgraph.types import Send

def fan_out(state: AgentState) -> list[Send]:
    """派发当前 wave 的全部子任务，同一 wave 内并行执行。"""
    plan, wave_idx = state["plan"], state["current_wave"]
    return [
        Send("execute_subtask", {"subtask": st, "wave": wave_idx})
        for st in plan.wave_tasks(wave_idx)
    ]
```

fan-in 靠 `subtask_results` 的 `operator.add` reducer 自动归并。
每层 wave 执行完毕后，调度边用 `plan.next_wave_index(wave_idx)` 决定是进入下一层还是
交给 `result_merger`。

**先验依赖的应用点**在 `execute_subtask` 内部：从 `state["subtask_results"]`
中取出上游结果，收集 `target_task_id == subtask.task_id` 的 `PriorAdjustment`
并应用到本地先验表（§6.2.1）。

### 7.2 wave 分层语义

| 属性 | 含义 |
| --- | --- |
| `waves: list[list[str]]` | 同 wave 内可并行；跨 wave 严格串行 |
| `max_parallelism` | 最宽 wave 的宽度，即并行收益上限 |
| `serial_depth` | wave 数量，用于估算串行延迟 |

`Plan` 的校验器**结构性地保证**：
① waves 覆盖所有子任务恰好一次；② 依赖必定排在更早的 wave。
非法计划在构造时即抛错，不会进入执行。

### 7.3 并行收益的度量

评测需对比**串行执行 vs wave 并行**的 p95 延迟（[06-roadmap.md](06-roadmap.md) §2.3）。

注意先验依赖会降低并行度：上例中若忽略 `T3→T2`，可全并行（延迟更低），
但解释质量下降。**这是正确性与延迟的显式权衡，必须被记录而非悄悄优化掉。**

## 8. 失败与降级策略

| 失败点 | 降级行为 | 用户可见性 |
| --- | --- | --- |
| ASR 失败 | 请用户打字 | 显式提示 |
| 路由不确定 | 澄清提问 | 显式提问 |
| 计划非法（拓扑/覆盖） | 重规划一次；仍失败 → 取第一个子意图走快路径 | 显式说明 |
| 子任务部分失败 | **部分成功**：返回已完成部分 + 说明缺了哪部分 | 显式说明 |
| 子任务 `optional=True` 失败 | 标记 `SKIPPED`，不阻塞 | 显式说明 |
| 视觉抽取部分失败 | 用成功图片建档案，标注覆盖度 | 显式提示「基于 N 张照片」 |
| 视觉抽取全失败 | 中止，要求重传 | 显式报错 |
| 记忆检索为空 | 明确说「没有相关记录」 | 显式说明 |
| 声学特征提取失败 | 退化为文本推断 | **必须标注证据模式** |
| LLM 超时 | 兜底模板 | 标注降级 |
| **写入失败（RECORD_EVENT）** | **响应如实报告未写成**，不得说「已记录」 | 显式说明 |
| **先验依赖未生效** | 按未调整的先验继续，并在 `skipped_prior_dependencies` 说明原因 | 静默（内部审计可见） |
| 守卫重写后仍违规 | 保守回答 | 标注不确定性 |
| TTS 失败 | 无语音 | 静默 |
| 记忆抽取失败 | 本轮不写 | 静默 |
| 向量库不可用 | 跳过向量召回，只用关系库 | 静默降级 |

**降级原则**：涉及**正确性**的降级必须用户可见（证据模式、覆盖度、不确定性、缺失的部分）；
涉及**体验**的降级可静默（TTS、记忆写入）。

这条原则在契约层被强制：`SubTaskResult` 状态为 `DEGRADED` 时**必须**给出 `degraded_note`，
`GuardResult` 降级为保守回答时**必须**给出用户可见说明。

## 9. 为什么这样拆节点（面试要点）

| 问题 | 回答 |
| --- | --- |
| 为什么不用一个大 prompt？ | ① 无法单测 ② 无法分别降级 ③ 无法观测哪一步出错 ④ 上下文长度不可控 |
| 拆节点的判据是什么？ | **按决策点拆，不按数据流拆**。见 §6.1 |
| 为什么 `understand_input` 单独成节点？ | 路由错误的代价是**错误记忆写入**，污染长期状态，必须可独立评测与设阈值 |
| 为什么意图阈值不是一个全局值？ | 不同误判方向的代价差异极大（0.15 vs 0.9），必须按意图对配置 |
| 为什么 `RECORD_EVENT` 不走阈值判定？ | 两个方向代价都高，阈值无解 → 改用「执行 + 明确反馈」改变交互范式 |
| 为什么需要 planner 而不让 LLM 直接生成执行计划？ | 依赖分析需要区分**先验依赖**，LLM 只看数据流会漏；且拓扑必须可复现 |
| 为什么要区分数据依赖与先验依赖？ | 先验依赖不体现在数据流上，但会显著影响输出质量。忽略它会得到「更快但更差」的结果 |
| 为什么守卫在编排层而不是 prompt 里？ | prompt 是建议，节点是强制。诚实性边界必须可验证、可拒绝 |
| 为什么 `behavior_interpreter` 不调 LLM 算概率？ | LLM 产生的数字不可复现、不可审计。「证据」必须可回溯到计算 |
| 为什么 `memory_writer` 放在响应之后？ | 记忆抽取是副作用，不应阻塞主响应；生产环境应异步化 |
| 并行会不会降低正确性？ | 会——先验依赖会强制串行。这个权衡被显式记录在 `Plan.waves` 里，而非被悄悄优化掉 |
