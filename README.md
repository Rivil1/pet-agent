# pet-agent

基于真实宠物身份的**经验记录与调取** Agent。

> 面向真实宠物的长期个性化陪伴系统：以**一只真实存在的猫**为身份锚点，
> 把主人对这只猫的经验记下来、在用得上时调出来，并提供拟人化陪伴与行为解读。
>
> **知识对象是「主人的经验」，不是「猫的心思」**（D29）——「猫的意图」不可验证
> （人类自己也只有 26–40%，三分类随机水平 33%）；「主人的经验」可验证，
> **且使评测第一次成为可能**。

---

## 1. 文档索引

> ### ⭐ 先读 [docs/DESIGN.md](docs/DESIGN.md)
>
> 它是本项目的**单一事实来源**：自洽、完整、可直接据此实现。
> 下表 `00`–`15` 是**推理存档**——记录了「为什么这样决定」的完整论证过程，
> 适合面试前复习。**若存档与 `DESIGN.md` 冲突，以 `DESIGN.md` 为准。**
>
> 同步债的唯一视图是 [15-doc-audit.md §7](docs/15-doc-audit.md#7-同步债登记表)；
> 各文档不再各自维护同步清单——那正是遗漏的成因。

| 文档 | 内容 | 状态 |
| --- | --- | --- |
| [docs/DESIGN.md](docs/DESIGN.md) | ⭐ **最终版设计文档（逻辑架构 / 单一事实来源）** | ✅ |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **系统架构（物理/实现）：存储 schema、API、运行时、可观测性、安全** | ✅ |
| [docs/00-product.md](docs/00-product.md) | 产品定位、能力边界、诚实性声明、非目标 | ✅ |
| [docs/01-architecture.md](docs/01-architecture.md) | 分层架构、LangGraph 编排、节点契约、失败降级 | ✅ |
| ~~02-data-model / 05-eval~~ | 已并入 [docs/DESIGN.md](docs/DESIGN.md)（§2.3+§3.5 / §6），不再单独成文 | — |
| [docs/03-behavior-interpreter.md](docs/03-behavior-interpreter.md) | 意图解释的证据链与后验概率推导（核心） | ✅ |
| [docs/04-memory.md](docs/04-memory.md) | 记忆三层、数据飞轮、冲突消解、防自我强化 | ✅ |
| [docs/06-roadmap.md](docs/06-roadmap.md) | 含金量提升路线、评测基线、消融实验、范围收敛 | ✅ |
| [docs/07-health.md](docs/07-health.md) | 健康异常监测与分诊（非诊断） | ✅ |
| [docs/08-capability-map.md](docs/08-capability-map.md) | RAG / Agent 技术能力体现地图、面试问答映射、演示脚本 | ✅ |
| [docs/09-intent-and-planning.md](docs/09-intent-and-planning.md) | 意图识别与任务分解：深度标志、代价矩阵、规划层设计 | ✅ |
| [docs/10-self-review.md](docs/10-self-review.md) | 需求与设计自审：矛盾、缺失机制、未验证假设、修改清单 | ✅ |
| [docs/11-companion-design.md](docs/11-companion-design.md) | 陪伴产品化：场景定义、分轨机制、主动触达、优先级重排 | ✅ |
| [docs/12-anthropomorphism.md](docs/12-anthropomorphism.md) | 拟人化边界的批判性评估：可做表达层，不可做事实层 | ✅ |
| [docs/13-multimodal-assessment.md](docs/13-multimodal-assessment.md) | 多模态观测：FGS / SRR / BCS 三个有文献依据的可测量项 | ✅ |
| [docs/14-entertainment.md](docs/14-entertainment.md) | 娱乐化边界与形态：三条禁令、展示分层、年度报告 | ✅ |
| [docs/BUGS.md](docs/BUGS.md) | **失败案例集（面试材料）**：26 个实现级缺陷，含实测数字与可迁移教训 | ✅ |
| [docs/15-doc-audit.md](docs/15-doc-audit.md) | 设计文档审查：内容矛盾、**同步债登记表**、防复发机制 | ✅ |
| [docs/16-model-selection.md](docs/16-model-selection.md) | **模型选型（2026 Q1）**：六个组件的决策与依据强度，含反直觉结论与自托管阈值 | ✅ |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | 术语表：消除命名冲突（记忆三层 / 信息分层 / 展示分层） | ✅ |

> **同步债的唯一视图是 [15-doc-audit.md §7](docs/15-doc-audit.md#7-同步债登记表)。**
> 各文档不再各自维护同步清单——那正是遗漏的成因。

## 2. 已定技术决策（Decision Log）

| # | 决策 | 理由 | 日期 |
| --- | --- | --- | --- |
| D1 | Python + LangGraph 为主干，FastAPI 暴露接口 | 图编排表达力最强，最适合讲 Agent 设计 | - |
| D2 | 多模态走 DashScope（Qwen-VL + Paraformer） | 与既有 `ai-file-agent` 同一套 Key | - |
| D3 | 身份一致性 = **文本特征锁定 + 向量外观校验** | 通用 embedding 做不了个体再识别，口径必须诚实 | - |
| D4 | 翻译证据 = **声学特征 + 历史个体样本 + 公开数据集先验** | 证据必须可复现，不能由 LLM 生成 | - |
| D5 | 多宠物 + 多用户全链路隔离 | 记忆串味是最致命的正确性缺陷 | - |
| D6 | 猫叫**不做 ASR 转写**，只做声学特征提取 | 猫叫不是语音；Paraformer 转写会产出乱码 | - |
| D7 | 置信度由检索/评分层计算，**LLM 不产生数字** | 这是「证据可追溯」的唯一保证 | - |
| D8 | 本期含评测集 / ASR / TTS / 数据飞轮 | 全量范围 | - |
| D9 | 健康模块定位为**监测与分诊**，不做诊断 | 《动物诊疗机构管理办法》第二条/第五条/第十九条 | - |
| D10 | 红旗规则存为 YAML 数据，须经兽医审核方可启用 | 阈值型规则来源不一致，不确定的阈值宁可不用 | - |
| D11 | Track A 红旗偏向过度触发（优化 recall） | 假阴性代价 >> 假阳性代价 | - |
| D12 | 覆盖率 < 0.3 时输出「无法评估」，禁止输出「未发现异常」 | 防止把「没数据」伪装成「没问题」 | - |
| D13 | 健康模块本期只做 Track A + 数据模型 | Track B 需 21 天基线，demo 场景无法积累 | - |
| D14 | 红旗规则仅启用多来源兽医手册强一致者 | 阈值型规则来源不一致，不确定的数值宁可不给 | - |
| D15 | 健康数据加密存储 + 用户可导出/删除（硬删除+级联） | 健康数据泄露后果远重于聊天记录 | - |
| D16 | 引入规划层（`planner` + Send fan-out + wave 调度），新增 `COMPOUND` 意图 | 任务分解是面试高频考点；与「时间线聚合问答」共用同一规划层 | - |
| D17 | 依赖分析区分 **数据依赖 / 先验依赖 / 资源依赖**，交由**代码**执行 | LLM 只看数据流会漏掉先验依赖，且拓扑不可复现 | - |
| D18 | 意图阈值改为**按意图对的代价矩阵**；`RECORD_EVENT` 改用「执行 + 明确反馈」 | 误判代价不对称（0.15–0.9）；该类两方向代价都高，阈值无解 → 改交互范式 | - |
| D19 | **优先级重排：P0 = 4–6 周可演示闭环（建档案 / 记忆+拒答 / 行为解释证据链 / 不变量测试）** | 旧 P0 五项中四项服务 S1，而 S1 替代品竞争最强（手机相册）、痛点最弱；且主动触达缺推送渠道、实际做不出来 | - |
| D20 | 主动触达 **P0 → P2**（阻塞：缺推送渠道） | 需求与实现之间存在断裂；无 App / 无服务号认证则能力落不了地 | - |
| D21 | FGS **移出路线图，降为可行性实验** | 文献验证的是专用 CNN，通用 VLM 零样本做 FGS 无人验证过；不应占用主线预算 | - |
| D22 | 规划层 / 多用户 **不实现** | 不影响产品的就不是需求；预期触发率 <10% | - |
| D23 | 照片/视频的娱乐反馈采用**人设吐槽**，**否决颜值评分** | 评分有梯度→诱导摆拍与打扰猫（禁令 3）；且会把「被毛凌乱」这类真实健康指标消解成玩笑 | - |
| D24 | 人设吐槽记入 P1，本期不实现 | 依赖档案（人设）先建立；且吐槽为异步增强，不阻塞 P0 记录链路 | - |
| D25 | 模型选型：`qwen-plus` / **`qwen3-vl-flash`**（原 `qwen-vl-max`）/ **`text-embedding-v4`**（原 v3）/ `paraformer-v2` / `cosyvoice-v3.5-flash` | 视觉输入价降至 1/10；v4 与 v3 **同价**而 MTEB Retrieval +3.89 | [16](docs/16-model-selection.md) |
| D26 | **猫叫理解不接外部 API**，维持自研声学特征 + 分层贝叶斯 | 文献的 95.94% **未验证跨个体泛化**；接入会把可解释后验换成黑箱 | [16 §4](docs/16-model-selection.md) |
| D27 | 行为解释器的论证改为**「人类自己也接近随机」** | 人类 26–40% vs 随机 33% 有直接证据；原论证（我们能从声学推断情境）在文献上站不住 | [16 §9](docs/16-model-selection.md) |
| D28 | **自托管只为隐私，不为省钱** | 算过：自托管 VLM ¥0.0017/张 vs API ¥0.0008/张，**反而贵一倍**。唯一例外是 ASR（SenseVoice CPU 17× 实时） | [16 §8](docs/16-model-selection.md) |
| **D29** | **定位重述：知识对象是「主人的经验」，不是「猫的意图」**；Agent 只作辅助（档案员 + 检索者 + 归纳者） | 「猫的意图」不可验证（人类 26–40% vs 随机 33%）；「主人的经验」可验证，**且使评测第一次成为可能** | [DESIGN §1.1](docs/DESIGN.md) |
| **D30** | 信息分层 **3 层 → 4 层**（拆出「主人记录」） | 系统测量可复现，主人记录含判断，**可信度来源不同不能同级** | [DESIGN 边界 3](docs/DESIGN.md) |
| **D31** | 推断模型由**参数化贝叶斯**改为**案例推理（k-NN）**；阈值语义由「总样本数」改为「**单个情境的相似案例数**」 | 参数化高斯在 n=3 时估不出方差；计数是可机械验证的事实 | [16 §4](docs/16-model-selection.md) |
| **D32** | 新增**每日总结**（对话 → 事实与记忆），核心机制是 **quote 机械校验** | 单轮提取看不到聚合模式与跨轮矛盾；而 LLM 会做「合理但无依据」的归纳 | [DESIGN §3.7](docs/DESIGN.md) |
| **D33** | 总结产出一律以 `EPISODE` 写入，**不直接写 `Profile`** | 一条当天说的话不足以成为稳定事实；晋升交给飞轮的重复次数 + 时间跨度 | [DESIGN §3.7](docs/DESIGN.md) |
| **D34** | 日报定位为**娱乐产物**（「宠物的一天」），但**必须从已验证候选派生**，不读原始对话 | 否则 quote 校验被绕过、健康信号被娱乐化、叙事覆盖事实（A4） | [DESIGN §3.8](docs/DESIGN.md) |
| **D35** | **健康信号可用宠物口吻叙述，但强制归 L3**：口吻是表达形式，娱乐化是调性，两者可分开 | 用户会因此改变照顾行为 → 必须在健康层且语气严肃；同一个生理词在不同层合法或违规 | [DESIGN §3.8](docs/DESIGN.md) |
| **D36** | 漫画/写真**本期只留结构**（`StoryBeat.photo_ref` + `render_beat_for_panel`），不生成图像 | 成本（可能比 TTS 还贵）/ 一致性 / **风险最高：漫画格无法标注不确定性** | [DESIGN §3.8](docs/DESIGN.md) |
| **D37** | 健康模块**不接运行时 RAG**；RAG 仅可用于①**离线辅助起草红旗规则（人签字）**②**常识问答（L1/L2）** | 检索失败是静默的，而假阴性代价是猫可能死亡；且临床文献会加剧越界 | [DESIGN §5.4.1](docs/DESIGN.md) |
| **D38** | 接入 `health_records` 出口（`app/health/`），**健康信号闭环** | 此前「记忆层拒了、日报显示了、没地方存」——已登记但无接收方 | [DESIGN §5.4.2](docs/DESIGN.md) |
| **D39** | 对话提取的健康记录**一律 `value=None`**，不解析中文数词 | 解析不可靠（B11/B12）且**错了会误触红旗**；`None` 诚实地进 `signals_missing` → `INSUFFICIENT_DATA` | [DESIGN §5.4.2](docs/DESIGN.md) |
| **D40** | 行为解释统一入口 `interpret_meow`，**模式选择 fail-closed** | 之前图节点直接用占位先验算后验 —— 那个 `0.41` 是编造的 | [DESIGN §3.9](docs/DESIGN.md) |
| **D41** | **用多模态模型提取事实**（先用模型跑通，真实调用中优化）；但模型产出的是 `OBSERVED` **不是** `MEASURED` | 自定义管线是在没有真实数据时设计的；而「模型做观察 / 代码做测量」的分界是「能不能重算」 | [DESIGN §3.10](docs/DESIGN.md) |
| **D42** | 模型**不得**推断因果、判断神态、伪装成测量 | 神态是红旗信号（漏报代价是猫可能死亡）；因果必须主人提供 | [DESIGN §3.10](docs/DESIGN.md) |
| **D43** | **MVP 闭环合上**：`/v1/interpret` 返回 `interpretation_id`，主人凭它标注；特征**只存服务端** | 此前 `insert_meow_record` 零调用 —— 「学习主人的经验」**没有输入路径** | [DESIGN §3.11](docs/DESIGN.md) |
| **D44** | 多模态观察接入解释：产出 `OBSERVED` 证据 + **预填标注动作**；**不参与模式选择、不改变候选排序** | 它是描述不是测量；影响判断会让未校验的模型驱动决策 | [DESIGN §3.12](docs/DESIGN.md) |
| **D45** | provider 模式**三态**推断（`mock`/`live`/`unknown`），暴露在 `/healthz` 与每个响应 | 没配密钥会静默退到 mock；而初版两分法把 mock 报成 `live`（B26） | [DESIGN §3.13](docs/DESIGN.md) |
| **D46** | **多厂商：每个能力各持一份 `CapabilityConfig`**（LLM / 视觉 / 向量 / 多模态可分别指向不同供应商）；接入火山方舟；`/healthz` 暴露模型与端点；新增运行时装配 `app/bootstrap.py` | 四类能力在各家覆盖不同（DeepSeek 无向量），绑在一条配置上就无法「A 家 LLM + B 家向量」；且此前**没有任何入口调用 `build_providers`** —— 改造等于没生效 | [DESIGN §3.14](docs/DESIGN.md) |
| **D47** | **会话记忆注入 + 全链路可追溯**：`X-Session-Id` 覆盖所有端点与落库实体；Session 层首次**被读**（最近 10 轮 + 字符预算）；接入 **LangSmith**，且 **`trace_id` 就是 `run_id`** | 此前 Session 层「写了但从不读」，多轮指代无从谈起；而历史含 assistant 自己的回复，与 `known_facts` 混栏会开一条**读入式自我强化**通道（与 I1 同构） | [DESIGN §3.15](docs/DESIGN.md) |

## 3. 环境

本项目在 **WSL2 (Ubuntu 22.04)** 下开发，代码位于 Windows 文件系统以便 IDE 直接打开。

| 项 | 位置 |
| --- | --- |
| 项目代码 | `E:\workspace\pet-agent` ↔ `/mnt/e/workspace/pet-agent` |
| 虚拟环境 | `/root/.venvs/pet-agent`（放 WSL 原生盘，避免 9p 文件系统拖慢安装） |
| 包源 | 清华镜像（venv 内 `pip.conf`） |

```bash
# 激活环境
source /root/.venvs/pet-agent/bin/activate

# 或直接调用
/root/.venvs/pet-agent/bin/python -m pytest
```

### 测试

契约层不变量测试，**不需要大模型、不需要网络**，1 秒内跑完：

```bash
/root/.venvs/pet-agent/bin/python -m pytest tests/ -q
```

这些测试验证的不是「功能是否正确」，而是**「设计约束是否被结构性强制」**——
例如防自我强化、计划拓扑合法性、先验依赖不可静默丢弃、无证据时禁止输出数值置信度。

### 启动

```bash
# 1. 必填：鉴权签名密钥（缺失会**拒绝启动** —— 否则 user_id 可伪造）
export PET_AGENT_AUTH_SECRET=dev-secret

# 2. 起服务（无供应商密钥时自动走 mock，/healthz 会明确标注）
python -m uvicorn app.bootstrap:create_app_from_env --factory --host 0.0.0.0 --port 8000

# 3. 签发一个开发用 token；核对实际装配到的厂商 / 模型 / 端点
python -m app.bootstrap --issue-token user-1
python -m app.bootstrap --describe
```

> `MOCK_PROVIDER=1` 会让**视觉显式失败**（`UnavailableVision`）—— 那是刻意的（D45）。
> 想离线跑通档案链路就**不要设它**：无密钥时会用确定性的 `MockVision`。

一次跑通全部端点（**真 uvicorn 进程 + 真 HTTP + 真实音频下载**）：

```bash
bash scripts/run_smoke.sh
```

> **已知环境问题**：WSL 的 Windows 互通（`binfmt_misc` 的 `WSLInterop`）在 WSL 启动早期未注册，
> 导致 `.exe` 调用间歇性报 `Exec format error`。需要通过 `cmd.exe` 调用 Windows 侧工具时，
> 先 `cat /proc/sys/fs/binfmt_misc/WSLInterop` 确认存在。

## 4. 目录结构

```text
pet-agent/
├── docs/                   设计文档（本阶段主交付物）
├── app/
│   ├── schemas/            领域契约（Pydantic）—— 前后端与各节点共享的接口定义
│   ├── graph/              LangGraph 编排
│   │   ├── state.py        AgentState 定义
│   │   ├── nodes/          各能力节点
│   │   └── build.py        图组装
│   ├── audio/              声学特征提取（librosa）
│   ├── llm/                模型客户端封装
│   ├── store/              存储抽象（关系库 + 向量库）
│   ├── api/                FastAPI 路由
│   └── eval/               评测集与跑批
├── web/                    简易 Web UI
└── data/
    ├── priors/             公开数据集先验（CatMeows 上下文分布）
    └── fixtures/           评测与开发用样例
```

## 5. 当前进度

- [x] 需求对齐（产品边界、技术栈、范围）
- [x] 关键事实核实（CatMeows 数据集、DashScope 接口约束、兽医红旗来源、诊疗合规）
- [x] 健康模块设计（定位为监测与分诊，非诊断）
- [x] **最终版设计文档 `docs/DESIGN.md`（自洽，已消除 D1–D13 全部矛盾）**
- [x] 设计文档（19/19：DESIGN.md 逻辑 + ARCHITECTURE.md 物理 + **BUGS.md 失败案例集** + **16-model-selection.md 选型** + 存档 + 术语表）
- [x] **失败案例集**：17 个实现级缺陷（**9 个完全静默 + 3 个「会崩但从未被触发」**），全部附回归测试
- [x] **P0 全部四步完成**：
  - ① 建档案（多图取交集 + 覆盖度 + 「不编造」prompt）
  - ② 对话 + 记忆（飞轮 / 强化 / 冲突消解 / 防自我强化 / 混合检索）
  - ③ 传猫叫 → 证据链（声学特征 + 分层贝叶斯 + 对数几率归因）
  - ④ 端到端 API（6 个端点 + HMAC 鉴权 + 租户隔离）
- [x] **每日总结**（对话 → 事实与记忆）：`app/digest/`，**quote 机械校验反编造**
- [x] **「宠物的一天」**（娱乐层）：`app/story/`，三条禁令**机械强制**（非 prompt）
- [x] **健康模块**：`app/health/`，红旗**三值逻辑**求值 + `health_records` 闭环
- [x] **案例推理（k-NN）**：`app/interpreter/case_based.py`，学习**主人自己的标注历史**；冷启动走测量模式
- [x] **多模态事实提取**：`app/extract/`，三道防线防编造；产出 `OBSERVED` 而非 `MEASURED`
- [x] **MVP 闭环**：录叫声 → 解释 → **主人标注** → 下次调出历史案例（11 个端点）
- [x] **多模态观察接入**：`OBSERVED` 证据 + 标注预填；声学失败时观察不丢
- [x] **provider 模式可见**：`/healthz` + 每个响应带 `provider_mode` / `mock_notice`
- [x] **多厂商模型客户端（D46）**：DashScope / 火山方舟；**每个能力一份配置**（LLM / 视觉 / 向量 / 多模态可分别指向不同供应商），逐能力环境变量覆盖
  - `app/llm/providers.py` 配置层 + `tests/test_providers.py`（35 项离线）
  - `/healthz` 暴露 `*_model` / `*_endpoint` / `mixed`（**不含密钥**）—— 只看类名看不出在调哪一家
  - 全逻辑用 `httpx.MockTransport` 离线测试覆盖
  - ⚠️ **尚未对真实 API 验证** —— 跑 `python scripts/live_check.py --provider {dashscope,ark}` 确认
- [x] **运行时装配**：`app/bootstrap.py`（环境变量 → 可启动 app）；修好此前**必然 TypeError** 的 Dockerfile 入口（`app.api.main:create_app` 全是必填 kw-only 参数）
- [x] **会话记忆注入 + 全链路可追溯（D47）**：`X-Session-Id` / `X-Trace-Id` 中间件覆盖全部端点与响应；Session 层首次**被读**（最近 10 轮 + 字符预算，截断留痕）
  - 注入**分区**：「已验证记录」与「最近对话（非事实）」分开，防止 assistant 历史被当成事实回灌
  - `response_guard` 同步：长期记忆类断言仍需检索命中；对话历史类可由**主人自己说过的话**满足，**assistant 的历史不算依据**
  - 落库实体均带 `session_id`：`SessionMessage` / `PendingInterpretation` / `MeowRecord` / `HealthRecord` / `MemoryEvent`；list 端点支持 `?session_id=` 过滤
- [x] **LangSmith 观测**：`app/observability/langsmith.py`，**`trace_id` 即 run id**（UUID 直用 / 非 UUID 用 `uuid5` 确定性派生），并写入 `metadata` 可在 LangSmith 按它检索
  - fail-soft：观测失败不影响对话；启用状态与错误暴露在 `/healthz`；**测试环境强制关闭**（`conftest.py`）
- [x] 契约不变量测试（**631 项**全通过，无需大模型与网络）
- [x] **静态类型检查：生产代码 0 error**（`pyrightconfig.json`）
- [ ] P1：记录/时间线、对话分轨、SRR 呼吸频率
- [ ] 同步债仍待处理（唯一视图：docs/15-doc-audit.md §7；但 `DESIGN.md` 已先行消除）
- [x] 文档一致性自动检查（`scripts/doc_check.sh`，7 类检查）
- [x] 领域契约代码（8 个模块，`import app.schemas` 通过）
- [x] **`tests/` 契约不变量测试（92 项全通过，无需大模型/网络，1 秒内跑完）**
- [x] **自审 F1–F4（P0）已修复**（见 docs/10-self-review.md §5.1）
- [ ] 自审 F5–F11 仍待处理（F5 置信度校准优先级最高）
- [ ] 产品化改造（见 docs/11-companion-design.md §9，需同步 00/01/04/06 + 新增契约）
- [ ] LangGraph 骨架
- [ ] 评测集
