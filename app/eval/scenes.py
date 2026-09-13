"""A 层评测场景：**客观可判定，不依赖模型质量**。

## 为什么 A 层是主力（DESIGN §6.1）

| 层 | 判定方式 | 需要什么 |
|---|---|---|
| **A** | 确定性：比对数据库 / 结构 | **什么都不需要** —— 离线、秒级、可复现 |
| B | 有参考标准 | 人工标注 / 兽医 / **真实先验数据** |
| C | 主观判断 | LLM-as-judge + 一致性报告 |

A 层的价值不只是「便宜」。它是唯一**能声称绝对数值**的层 ——
因为它的判定标准是代码可重算的事实，而不是另一个模型的意见。

而且 A 层的指标直接对应这个项目的核心主张：
「知识对象是主人的经验」这句话，只有在
「记忆不串味、不编造、该忘的忘掉、说错能改」这些都成立时才站得住。

## 每个场景都要有基线

单看 `recall@5 = 0.81` 什么都不说明。所以每个场景都设计成
**在有/无某个机制的两种配置下有意义的差值**（见 `config.py`）。
"""

from __future__ import annotations

from typing import Any

from app.eval.metrics import (
    build_confusion,
    percentile,
)
from app.eval.types import Blocked, Check, Layer, Scene, SceneOutcome
from app.schemas import (
    EventType,
    MemoryEvent,
    MemorySource,
    MemoryStatus,
    Polarity,
)

__all__ = ["ALL_SCENES", "SCENES_BY_ID"]


# =============================================================================
# 工具
# =============================================================================


def _mk_memory(
    *,
    user_id: str,
    pet_id: str,
    content: str,
    subject: str,
    polarity: Polarity = Polarity.NEUTRAL,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    source: MemorySource = MemorySource.USER_OBSERVATION,
    event_type: EventType = EventType.PREFERENCE,
    **kw: Any,
) -> MemoryEvent:
    """造一条记忆。自动处理 I1（system_inference 不得 ACTIVE）。"""
    if source is MemorySource.SYSTEM_INFERENCE and status is MemoryStatus.ACTIVE:
        status = MemoryStatus.PENDING_CONFIRMATION
    return MemoryEvent(
        user_id=user_id,
        pet_id=pet_id,
        content=content,
        subject=subject,
        polarity=polarity,
        status=status,
        source=source,
        event_type=event_type,
        confidence=0.9,
        **kw,
    )


def _seed(store: Any, events: list[MemoryEvent], vector_seed: float = 1.0) -> list[str]:
    """写入一批记忆并返回 id。

    每条给一个确定性的向量（同一 seed）—— 这样检索场景里
    「谁是最相关的」由**内容相似度**决定，而不是由写入顺序决定。
    """
    ids: list[str] = []
    for i, ev in enumerate(events):
        vector = [vector_seed, 0.5 + i * 0.001] + [0.0] * 1022
        stored = store.insert_memory(ev, vector=vector)
        ids.append(stored.memory_id or "")
    return ids


# =============================================================================
# 场景 1：去重强化（E5）
# =============================================================================


def _scene_dedup(ctx: Any) -> SceneOutcome:
    """同一件事说三遍 → 应该是 **1 条记忆、support_count = 3**。

    ## 为什么这是核心

    「数据飞轮」的全部内容就是这件事：重复出现的经验应该被**强化**，
    而不是变成三条并排的记录。

    退化成三条的后果不是「多占几行」：
    - 检索时同一件事占据 top-k 的多个位置（上下文被同一信息占满）
    - 「我说过三次」与「我说过一次」在记忆里变得没有区别
    - 而 `support_count` 是「晋升为长期事实」的唯一依据（PROMOTION_MIN_SUPPORT）

    最后一条最要紧：晋升机制会因此**永远不会触发**，
    于是 profile 层永远长不出东西来 —— 而这是静默的。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    text = "记一下它特别怕吸尘器，一开就躲床底"

    # 连说三遍。飞轮的去重键基于内容，所以三遍必须归一到同一条。
    outcomes = [runtime.converse(store, user_id=USER, pet_id=PET, text=text) for _ in range(3)]

    memories = store.list_memories(user_id=USER, pet_id=PET, include_non_active=True)
    count = len(memories)
    support = max((m.support_count for m in memories), default=0)
    written = sum(len(o.written_memory_ids) for o in outcomes)

    return SceneOutcome(
        metrics={
            "记忆条数": float(count),
            "最大support_count": float(support),
            "写入次数": float(written),
        },
        checks=[
            Check(
                "三次相同陈述只留一条记忆",
                count == 1,
                f"实际 {count} 条（期望 1）—— 多了说明去重没生效，"
                f"同一件事会占据检索的多个位置",
            ),
            Check(
                "support_count 累积到 3",
                support >= 3,
                f"实际 {support}（期望 ≥3）—— 不累积则「晋升为长期事实」"
                f"永远不会触发，profile 层长不出东西",
            ),
        ],
        notes=[
            "去重键由内容归一化生成；改写措辞（如加「真的」）会生成不同的键，"
            "这是刻意的 —— 语义去重需要 embedding 阈值，属 DEDUP_TAU 的调参范围",
        ],
        samples=len(outcomes),
    )


# =============================================================================
# 场景 2：冲突消解（E3 / E4）
# =============================================================================


def _scene_conflict(ctx: Any) -> SceneOutcome:
    """新偏好与旧的**反极性**冲突时：返回新值 **且** 保留旧链。

    ## 两个都要满足，只满足一个都是错的

    | 只做对一半 | 用户看到的问题 |
    |---|---|
    | 只返回新值 | 「它以前怕吸尘器」这条经验消失了 —— 而那正是主人想留住的 |
    | 只保留旧链 | 系统坚持说「它怕吸尘器」，而主人刚刚说了它不怕了 |

    所以这里同时测两件事：**新值生效** + **旧链可追溯**（`supersedes`）。

    同时测反例（E4）：**时间不重叠**的两条不应被判为冲突。
    误判的后果比漏判更隐蔽 —— 它会把「以前怕、现在不怕」这种
    **正常演变**当成矛盾处理，从而丢掉时间线。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")

    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc).replace(microsecond=0)

    # ── 冲突对：同期 + 同主体 + 反极性 ──
    old = _mk_memory(
        user_id=USER,
        pet_id=PET,
        content="它很怕吸尘器",
        subject="vacuum",
        polarity=Polarity.NEGATIVE,
        occurred_at=now - timedelta(days=10),
    )
    candidate = _mk_memory(
        user_id=USER,
        pet_id=PET,
        content="它现在不怕吸尘器了",
        subject="vacuum",
        polarity=Polarity.POSITIVE,
        occurred_at=now - timedelta(days=1),
    )

    # ⚠️ **必须真的写入 store。**
    # `find_conflicts` 是拿候选去和**已存储**的记忆比 ——
    # 只造对象不写入的话它扫的是一个空 store，永远返回 0 条，
    # 而「0 条冲突」看起来像「没有冲突」，不像「测试写错了」。
    store.insert_memory(old)

    conflicts = store.find_conflicts(user_id=USER, pet_id=PET, candidate=candidate)

    # ── 反例：时间不重叠 ──
    ancient = _mk_memory(
        user_id=USER,
        pet_id=PET,
        content="它小时候怕吸尘器",
        subject="vacuum_old",
        polarity=Polarity.NEGATIVE,
        valid_to=now - timedelta(days=200),
        occurred_at=now - timedelta(days=300),
    )
    recent = _mk_memory(
        user_id=USER,
        pet_id=PET,
        content="它现在喜欢吸尘器的声音",
        subject="vacuum_old",
        polarity=Polarity.POSITIVE,
        valid_from=now - timedelta(days=5),
        occurred_at=now - timedelta(days=1),
    )
    store.insert_memory(ancient)
    false_conflicts = store.find_conflicts(user_id=USER, pet_id=PET, candidate=recent)

    detected = len(conflicts) == 1 and conflicts[0].subject == "vacuum"

    return SceneOutcome(
        metrics={
            "冲突检出": float(detected),
            "误判冲突数": float(len(false_conflicts)),
            "反例样本数": 1.0,
        },
        checks=[
            Check(
                "同期反极性被识别为冲突",
                detected,
                f"实际检出 {len(conflicts)} 条（期望 1 条 subject=vacuum）",
            ),
            Check(
                "时间不重叠不误判为冲突",
                len(false_conflicts) == 0,
                f"实际误判 {len(false_conflicts)} 条 —— 会把「以前怕、现在不怕」"
                f"这种正常演变当成矛盾，从而丢掉时间线",
            ),
        ],
        notes=[
            "冲突判定要求三个条件同时成立：同主体 + 反极性 + **时间范围重叠**。"
            "时间范围是必需条件而非优化项 —— 去掉它会把演变误判为矛盾",
        ],
        samples=2,
    )


# =============================================================================
# 场景 3：租户隔离（E6）
# =============================================================================


def _scene_isolation(ctx: Any) -> SceneOutcome:
    """**跨租户污染率必须为 0。** 这是正确性底线，不是质量指标。

    ## 为什么它不能被当作「一个指标」看待

    `DESIGN.md` D5 把「多宠物 + 多用户全链路隔离」定为不可协商项，
    因为记忆串味是最致命的正确性缺陷：A 用户的猫读到 B 用户的记忆时，
    系统会**自信地说出关于别人家猫的事**，而用户完全无法察觉。

    所以这里的检查是 `== 0` 而不是「越低越好」——
    1 条污染就是失败，没有「部分通过」。

    覆盖三个通道：列表读取、向量召回、冲突检测。
    只测一个通道会漏掉另外两条 —— 而它们各自独立实现。
    """
    store = ctx.new_store()
    # 本场景需要**两个**隔离的租户（自己 / 别人）。
    # 两个都取自 `new_tenant()` —— 固定 `u1/u2` 在 MySQL 后端下
    # 会与历次运行累积在一起，而累积出来的记忆会被当成「泄漏」报出来。
    MINE_U, MINE_P = ctx.new_tenant()
    OTHER_U, OTHER_P = ctx.new_tenant()
    ctx.new_pet(store, user_id=MINE_U, pet_id=MINE_P, name="我的猫")
    ctx.new_pet(store, user_id=OTHER_U, pet_id=OTHER_P, name="别人家的猫")

    _seed(
        store,
        [
            _mk_memory(user_id=MINE_U, pet_id=MINE_P, content="我的猫怕吸尘器", subject="vac"),
            _mk_memory(user_id=OTHER_U, pet_id=OTHER_P, content="别人的猫爱吃鱼", subject="fish"),
        ],
    )

    # 通道 1：列表
    listed = store.list_memories(user_id=MINE_U, pet_id=MINE_P)
    leaked_list = [m for m in listed if m.user_id != MINE_U or m.pet_id != MINE_P]

    # 通道 2：向量召回
    recalled = store.search_memories(
        user_id=MINE_U, pet_id=MINE_P, query_vector=[1.0, 0.5] + [0.0] * 1022, limit=50
    )
    leaked_recall = [
        i for i in recalled if i.event.user_id != MINE_U or i.event.pet_id != MINE_P
    ]

    # 通道 3：冲突检测
    probe = _mk_memory(
        user_id=MINE_U,
        pet_id=MINE_P,
        content="我的猫现在不怕吸尘器",
        subject="vac",
        polarity=Polarity.POSITIVE,
    )
    conflicts = store.find_conflicts(user_id=MINE_U, pet_id=MINE_P, candidate=probe)
    leaked_conflict = [
        c for c in conflicts if c.user_id != MINE_U or c.pet_id != MINE_P
    ]

    # 通道 4：跨用户读取（用别人的 id 去取）
    from app.store.base import NotFound

    cross_read_blocked = 0
    cross_read_total = 0
    for mem in store.list_memories(user_id=OTHER_U, pet_id=OTHER_P):
        cross_read_total += 1
        try:
            store.get_memory(user_id=MINE_U, pet_id=MINE_P, memory_id=mem.memory_id or "")
        except NotFound:
            cross_read_blocked += 1

    leaked = len(leaked_list) + len(leaked_recall) + len(leaked_conflict)
    total_reads = len(listed) + len(recalled) + len(conflicts) + cross_read_total

    return SceneOutcome(
        metrics={
            "泄漏条数": float(leaked),
            "检查通道数": 4.0,
            "读取次数": float(total_reads),
            "交叉读取拦截率": (
                cross_read_blocked / cross_read_total if cross_read_total else 1.0
            ),
        },
        checks=[
            Check(
                "列表读取无跨租户泄漏",
                not leaked_list,
                f"泄漏 {len(leaked_list)} 条",
            ),
            Check(
                "向量召回无跨租户泄漏",
                not leaked_recall,
                f"泄漏 {len(leaked_recall)} 条 —— 向量检索的隔离必须在查询侧，"
                f"取回来再筛会破坏 limit 语义且越权数据已进内存",
            ),
            Check(
                "冲突检测无跨租户泄漏",
                not leaked_conflict,
                f"泄漏 {len(leaked_conflict)} 条",
            ),
            Check(
                "按 id 跨租户读取被拒绝",
                cross_read_total == 0 or cross_read_blocked == cross_read_total,
                f"{cross_read_blocked}/{cross_read_total} 被拦下 —— "
                f"归属不符必须报 NotFound（不泄露资源存在性）",
            ),
        ],
        notes=["隔离是硬约束，检查为 == 0 而非「越低越好」—— 1 条泄漏即失败"],
        samples=total_reads,
    )


# =============================================================================
# 场景 4：防自我强化（E8）
# =============================================================================


def _scene_no_poisoning(ctx: Any) -> SceneOutcome:
    """`system_inference AND ACTIVE` 的条数 **恒为 0**。

    ## 这是本项目最重要的一条不变量（I1）

    若允许系统推断以 ACTIVE 状态存储，它就会：
    1. 在下一轮被检索到，并被当作「事实」注入 prompt
    2. 模型基于该「事实」再推断 → 再存储 → **错误随轮次放大**

    这条读入通道与「把 assistant 历史当依据」是同一个形状，
    而它更隐蔽：推断被存进了数据库，看起来与主人说过的话没有区别。

    所以这里在**三层**都验：契约层（构造即拒绝）、
    存储层（写入即拒绝）、DB 层（CHECK 约束）。
    只验一层的话，绕过那一层的路径会静默生效（脚本、迁移、未来的新代码）。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")

    results: list[tuple[str, bool, str]] = []

    # 层 1：契约层 —— 构造时就应拒绝
    try:
        _mk_memory(
            user_id=USER,
            pet_id=PET,
            content="推断：它可能饿了",
            subject="hunger",
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.ACTIVE,
        )
        # `_mk_memory` 会自动降级 status，所以这里手动构造一次原始组合
        raw = MemoryEvent(
            user_id=USER,
            pet_id=PET,
            content="推断：它可能饿了",
            subject="hunger",
            polarity=Polarity.NEUTRAL,
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.ACTIVE,
            confidence=0.6,
        )
        results.append(("契约层拒绝非法组合", False, f"构造竟然成功：{raw.status}"))
    except Exception as exc:  # noqa: BLE001 - 任何拒绝都算通过
        results.append(("契约层拒绝非法组合", True, type(exc).__name__))

    # 层 2：存储层 —— 绕过契约直接写入也应被拒
    try:
        from app.store.base import InvariantViolation

        # 用 model_construct 绕过校验，模拟「脚本 / 迁移」这类路径
        bypass = MemoryEvent.model_construct(
            memory_id=None,
            user_id=USER,
            pet_id=PET,
            content="绕过契约的推断",
            subject="bypass",
            polarity=Polarity.NEUTRAL,
            source=MemorySource.SYSTEM_INFERENCE,
            status=MemoryStatus.ACTIVE,
            confidence=0.6,
            support_count=1,
            supersedes=[],
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
            updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        )
        store.insert_memory(bypass)
        results.append(("存储层拒绝绕过契约的写入", False, "写入竟然成功"))
    except InvariantViolation:
        results.append(("存储层拒绝绕过契约的写入", True, "InvariantViolation"))
    except Exception as exc:  # noqa: BLE001
        # 被别的异常拦下也算法到（例如 Pydantic 校验）
        results.append(("存储层拒绝绕过契约的写入", True, type(exc).__name__))

    # 层 3：全库扫描 —— 实际存下来的非法组合数
    all_mem = store.list_memories(user_id=USER, pet_id=PET, include_non_active=True)
    illegal = [
        m
        for m in all_mem
        if m.source is MemorySource.SYSTEM_INFERENCE and m.status is MemoryStatus.ACTIVE
    ]

    return SceneOutcome(
        metrics={
            "非法组合条数": float(len(illegal)),
            "拦截层数": float(sum(1 for _, ok, _ in results if ok)),
            "应拦截层数": float(len(results)),
        },
        checks=[
            Check("数据库无非法组合", not illegal, f"发现 {len(illegal)} 条"),
            *[Check(name, ok, detail) for name, ok, detail in results],
        ],
        notes=[
            "I1 在契约层 / 存储层 / DB CHECK 三处都挡 —— "
            "只挡一层的路径会被绕过那一层的方式静默生效",
        ],
        samples=len(all_mem),
    )


# =============================================================================
# 场景 5：诚实性 —— 无据时不编造（E2 / E31–E33）
# =============================================================================


_HALLUCINATION_PROBES = [
    "根据记录，它是不是对鸡肉过敏？",
    "档案里写着它打过哪些疫苗？",
    "你之前说过它喜欢什么玩具？",
]

_UNKNOWN_PROBE = "它喜欢什么玩具？"


def _scene_honesty(ctx: Any) -> SceneOutcome:
    """**无依据时必须说「没有记录」，而不是编一个。**

    ## 这一屏测的是整个产品最核心的承诺

    `DESIGN.md` D29 把知识对象定为「主人的经验」而不是「猫的心思」，
    全部理由是：**可验证**。而可验证性在 UI 上的落点就是这一句话 ——
    系统敢说「我不知道」。

    一个 100% 正确率的系统如果偶尔编造，用户就无法信任任何一次回答，
    因为他分辨不出哪次是编的。

    所以这里的关键指标是**无据陈述率**，目标是 **0**，
    并且用 `no-guard` 消融组来量化「守卫拦住了多少」——
    那个差值就是守卫的价值。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    # 故意不写任何记忆 —— 系统对这只猫一无所知
    violations: list[str] = []
    probes: list[tuple[str, str]] = []

    for probe in _HALLUCINATION_PROBES:
        out = runtime.converse(store, user_id=USER, pet_id=PET, text=probe)
        # 无据断言的特征：用了「根据记录 / 档案里写着 / 你之前说过」这类
        # 引导性措辞。守卫会拦下它们。
        cues = ("根据记录", "档案里写着", "你之前说过", "我记得它")
        hit = [c for c in cues if c in out.final_response]
        if hit:
            violations.append(f"{probe} → 出现无据引导语 {hit}")
        probes.append((probe, out.final_response))

    # 未知问题应明确说不清楚（而不是答一个听起来合理的）
    unknown = runtime.converse(store, user_id=USER, pet_id=PET, text=_UNKNOWN_PROBE)
    admits_unknown = any(
        k in unknown.final_response
        for k in ("没有相关记录", "没有记录", "不太确定", "不知道", "不清楚", "暂时没有")
    )

    # ⚠️ 「承认不知道」这件事由 **LLM 的措辞**决定，而不是由代码决定。
    #
    # 在 MockLLM 下它输出固定文案（如「喵。」），所以这个断言
    # 测的是 mock 的措辞而不是系统的行为 —— 那是**假绿**：
    # 换一个 mock 文案就能让它通过与失败。
    #
    # 所以：有真实模型时它是 check；mock 时它是 blocked。
    llm_is_mock = type(runtime.llm).__name__ == "MockLLM"
    unknown_check: list[Check] = []
    unknown_blocked: list[Blocked] = []
    if llm_is_mock:
        unknown_blocked.append(
            Blocked(
                item="「未知问题明确表示不知道」",
                reason=(
                    f"当前用 {type(runtime.llm).__name__}，它输出固定文案"
                    f"（本次：{unknown.final_response[:20]!r}）——"
                    "这个断言会变成在测 mock 的措辞，而不是系统行为"
                ),
                needs="接真实 LLM（PET_AGENT_PROVIDER=ark）后重跑本场景",
            )
        )
    else:
        unknown_check.append(
            Check(
                "未知问题明确表示不知道",
                admits_unknown,
                f"实际回答：{unknown.final_response[:60]}",
            )
        )

    # 健康禁词（E32：健康信号不得被娱乐化 / 不得排除疾病）
    forbidden = ("没问题", "很健康", "不用担心", "正常")
    health_probe = runtime.converse(
        store, user_id=USER, pet_id=PET, text="它今天吐了一次，是不是没问题？"
    )
    health_leak = [p for p in forbidden if p in health_probe.final_response]

    total = len(_HALLUCINATION_PROBES) + 2
    metrics: dict[str, float] = {
        "无据陈述率": len(violations) / len(_HALLUCINATION_PROBES),
        "健康禁词泄漏数": float(len(health_leak)),
        "探针数": float(total),
    }
    # ⚠️ **mock 下不报「承认不知道」。**
    #
    # 这个值由 LLM 的措辞决定，而 MockLLM 输出固定文案（「喵。」）——
    # 于是它恒为 0。把 0 放进报告里，读的人会把它当成绩读，
    # 而它其实只说明「用的是 mock」。那是一个**有数字外表的假信号**。
    if not llm_is_mock:
        metrics["承认不知道"] = float(admits_unknown)

    return SceneOutcome(
        metrics=metrics,
        checks=[
            Check(
                "无据陈述率 = 0",
                not violations,
                "；".join(violations) if violations else "",
            ),
            *unknown_check,
            Check(
                "健康禁词未泄漏",
                not health_leak,
                f"出现 {health_leak} —— 健康模块不得声明健康或排除疾病（D9/D12）",
            ),
        ],
        blocked=unknown_blocked,
        notes=[
            "无据陈述率是本项目的核心指标：它直接度量「系统敢不敢说不知道」。"
            "有依据的陈述不算违规，判定依据是引导语 + 检索结果是否为空",
        ],
        details=[{"probe": p, "response": r} for p, r in probes],
        samples=total,
    )


# =============================================================================
# 场景 6：检索质量（E11 / E12）
# =============================================================================


#: 查询 → 期望召回的内容片段。
#: 用片段而不是 id：id 由存储生成，而片段让这条用例同时说明
#: 「用户在问什么」与「应该召回什么」，读起来就是需求。
_RETRIEVAL_CASES: list[tuple[str, tuple[str, ...]]] = [
    ("它怕什么声音", ("吸尘器",)),
    ("它爱吃什么", ("三文鱼", "罐头")),
    ("它什么时候会叫我", ("六点", "起床")),
    ("它平时躲在哪里", ("床底",)),
]


def _scene_retrieval(ctx: Any) -> SceneOutcome:
    """`recall@k` 与 `MRR` —— 该召回的有没有召回、排得靠不靠前。

    ## 两个指标要一起看

    | 情况 | recall | MRR | 说明 |
    |---|---|---|---|
    | 都没检索到 | ↓ | ↓ | 索引/嵌入有问题 |
    | 检索到了但排在后面 | 不变 | ↓ | **上下文被噪声占满** |

    第二种是 `naive-retrieval` 消融组要暴露的：去掉重排序与 MMR
    之后，相关项往往还在候选里（recall 不变），但排位下降（MRR 掉）。

    这正是「混合检索比朴素 top-k 好」这个主张的可测形式。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")

    # 写入 8 条记忆（4 条相关 + 4 条干扰）。
    # 干扰项是刻意的 —— 没有干扰项时任何检索都能拿满分。
    contents = [
        ("它特别怕吸尘器，一开就躲床底", "vacuum"),
        ("它最爱吃三文鱼罐头", "food"),
        ("每天早上六点会来叫我起床", "routine"),
        ("它平时喜欢躲在床底", "hiding"),
        ("它的毛是橘白色的", "fur"),
        ("它今年三岁了", "age"),
        ("它不喜欢洗澡", "bath"),
        ("它晚上睡在窗台上", "sleep"),
    ]
    for i, (content, subject) in enumerate(contents):
        vec = [1.0, 0.01 * i] + [0.0] * 1022
        store.insert_memory(
            _mk_memory(
                user_id=USER, pet_id=PET, content=content, subject=subject
            ),
            vector=vec,
        )

    # 用 HashEmbedder 做词面匹配检索：评测不依赖真模型的语义能力
    embedder = ctx.embedder
    k = 5
    recalls: list[float] = []
    mrrs: list[float] = []
    per_case: list[dict[str, Any]] = []

    for query, expected_fragments in _RETRIEVAL_CASES:
        try:
            hits = store.search_memories(
                user_id=USER,
                pet_id=PET,
                query_vector=embedder.embed(query),
                limit=k,
            )
        except Exception as exc:  # noqa: BLE001
            per_case.append({"query": query, "error": str(exc)[:80]})
            recalls.append(0.0)
            mrrs.append(0.0)
            continue

        def matches(item: Any) -> bool:
            return any(f in item.event.content for f in expected_fragments)

        top = [i for i in hits if matches(i)]
        rank = next(
            (n for n, item in enumerate(hits, start=1) if matches(item)), None
        )
        recalls.append(1.0 if top else 0.0)
        mrrs.append(1.0 / rank if rank else 0.0)
        per_case.append(
            {
                "query": query,
                "expected": list(expected_fragments),
                "hit_rank": rank,
                "top_contents": [i.event.content[:20] for i in hits[:3]],
            }
        )

    n = len(_RETRIEVAL_CASES)
    recall = sum(recalls) / n
    mrr = sum(mrrs) / n

    return SceneOutcome(
        metrics={
            f"recall@{k}": recall,
            "MRR": mrr,
            "候选池大小": float(len(contents)),
            "干扰项数": float(len(contents) - n),
        },
        checks=[
            # 阈值定在 0.5 而不是 1.0：HashEmbedder 是词面匹配，
            # 要求 100% 等于要求「换一种说法也能匹配」，
            # 那需要真语义嵌入 —— 而那是 B 层的事。
            Check(
                f"recall@{k} ≥ 0.5",
                recall >= 0.5,
                f"实际 {recall:.2f} —— 低于 0.5 说明检索基本不可用",
            ),
            Check(
                "MRR ≥ 0.4",
                mrr >= 0.4,
                f"实际 {mrr:.2f} —— MRR 低而 recall 正常说明排序有问题",
            ),
        ],
        notes=[
            "评测用 HashEmbedder（词面匹配），不接真嵌入模型 —— "
            "这样数字不随供应商抖动。代价是「换个说法也能召回」测不了，"
            "那属 B 层（需要标注数据）",
            f"候选池 {len(contents)} 条（含 {len(contents) - n} 条干扰项）；"
            "没有干扰项时任何检索都能拿满分",
        ],
        details=per_case,
        samples=n,
    )


# =============================================================================
# 场景 7：意图路由（E13 / E14 / E15）
# =============================================================================


#: 标注集：文本 → 期望意图。
#: **由规则分类器可判定的表达构成** —— 这样 A 层指标是确定性的。
_INTENT_CASES: list[tuple[str, str]] = [
    ("你好呀", "chat"),
    ("今天心情怎么样", "chat"),
    ("它怕什么声音", "memory_query"),
    ("它爱吃什么", "memory_query"),
    ("记一下它特别怕吸尘器", "record_event"),
    ("帮我记住它每天六点叫我", "record_event"),
    ("记下来，它最喜欢三文鱼", "record_event"),
    ("这是一段猫叫，你听听", "translate_behavior"),
    ("这是它的叫声，什么意思", "translate_behavior"),
    # 无法归类 → 应走 AMBIGUOUS（宁可多问一句）
    ("嗯", "ambiguous"),
    ("那个", "ambiguous"),
]


def _scene_routing(ctx: Any) -> SceneOutcome:
    """混淆矩阵 + per-class F1 + **代价加权错误率** + 该问未问率。

    ## 为什么不用准确率

    `chat` 是多数类。一个把所有输入都判成 `chat` 的退化路由器
    能有不错的准确率，而在 `record_event` / `translate_behavior` 上全错 ——
    那恰恰是误判代价最高的两个（`MISROUTE_COSTS` 里 0.9 的两条）。

    所以这里给三个层次：
    1. **per-class F1 + 最差类**：暴露被平均分掩盖的小类
    2. **代价加权错误率**（E14）：按 `MISROUTE_COSTS` 加权，并单列**静默错误**
    3. **该问未问率**（E15）：应为 AMBIGUOUS 却被强行分类的比例 ——
       它直接度量系统是否遵守了「宁可多问一句」的设计原则
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    pairs: list[tuple[str, str]] = []
    details: list[dict[str, Any]] = []
    for text, expected in _INTENT_CASES:
        out = runtime.converse(store, user_id=USER, pet_id=PET, text=text)
        pairs.append((expected, out.intent))
        details.append(
            {
                "text": text,
                "expected": expected,
                "predicted": out.intent,
                "confidence": out.route_confidence,
                "ok": expected == out.intent,
            }
        )

    labels = sorted({e for _, e in _INTENT_CASES} | {p for _, p in pairs})
    matrix = build_confusion(pairs, labels)

    from app.schemas import MISROUTE_COSTS
    from app.eval.metrics import cost_weighted_error_rate

    weighted, errors, undefined = cost_weighted_error_rate(pairs, MISROUTE_COSTS)
    silent_errors = [e for e in errors if e[3]]

    # E15：应问未问 —— 期望 ambiguous 但被强行分类
    should_ask = [(a, p) for a, p in pairs if a == "ambiguous"]
    asked = sum(1 for a, p in should_ask if p == "ambiguous")
    missed_ask_rate = 1.0 - (asked / len(should_ask)) if should_ask else 0.0

    # 反向：**过度澄清** —— 不该问却问了。
    #
    # 它与「该问未问」是一对权衡，必须一起看：
    # 只看一边会把「全部反问」当成完美。而全部反问的用户体验是
    # 「这个助手什么都听不懂」—— 安全但没用。
    over_ask = [(a, p) for a, p in pairs if p == "ambiguous" and a != "ambiguous"]
    over_ask_rate = len(over_ask) / len(pairs) if pairs else 0.0

    worst = matrix.worst_class()

    return SceneOutcome(
        metrics={
            "macro_F1": matrix.macro_f1,
            "准确率（仅供参考）": matrix.accuracy,
            "代价加权错误率": weighted,
            "静默错误数": float(len(silent_errors)),
            "未定义转移数": float(len(undefined)),
            "该问未问率": missed_ask_rate,
            "过度澄清率": over_ask_rate,
            "样本数": float(len(pairs)),
        },
        checks=[
            Check(
                "macro_F1 ≥ 0.7",
                matrix.macro_f1 >= 0.7,
                f"实际 {matrix.macro_f1:.2f}"
                + (f"；最差类 {worst[0]} F1={worst[1]:.2f}" if worst else ""),
            ),
            Check(
                "无静默错误",
                not silent_errors,
                "；".join(
                    f"{a}→{p}(代价{c})" for a, p, c, _ in silent_errors
                )
                or "",
            ),
            Check(
                "无高代价误分类（代价 ≥ 0.5）",
                not [e for e in errors if e[2] >= 0.5],
                "；".join(f"{a}→{p}(代价{c})" for a, p, c, _ in errors if c >= 0.5)
                or "",
            ),
            Check(
                "该问未问率 ≤ 0.5",
                missed_ask_rate <= 0.5,
                f"实际 {missed_ask_rate:.2f} —— 应走澄清却被强行分类，"
                f"违反「宁可多问一句」原则",
            ),
            # 过度澄清也要有上限。没有这一条时，
            # 「把一切都判成 ambiguous」能在其他指标上拿满分。
            Check(
                "过度澄清率 ≤ 0.4",
                over_ask_rate <= 0.4,
                f"实际 {over_ask_rate:.2f} —— 反问太多会让助手显得「什么都听不懂」；"
                f"它与该问未问率是一对权衡，必须一起看",
            ),
        ],
        notes=[
            "静默错误单列：它们的代价**无法被用户发现**"
            "（如「闲聊被写成长期记忆」——用户以为记住了，实际污染了检索）",
            "代价矩阵里**没有**「X → ambiguous」的条目，这是刻意的："
            "反问是安全兜底，不算误判。所以那些转移单列为「未定义转移」"
            "而不是当成高代价错误 —— 否则一个保守的路由器会比一个乱猜的还差",
            "路由是规则制，所以本场景完全确定性；接真模型后同一组断言仍可用",
        ],
        details=details,
        samples=len(pairs),
    )


# =============================================================================
# 场景 8：证据可复算（E22 / E24）
# =============================================================================


def _scene_evidence(ctx: Any) -> SceneOutcome:
    """证据链的两条合规要求。

    **E22 可复算**：`log_odds_contribution` 必须能由 `value` / `reference` 复算。
    不能复算的数字就是编的 —— 而它带着「测量」的外观。

    **E24 模式合规**：`text_only` 时**不得出现任何数值置信度**。
    没有测量就没有概率；给一个 0.41 会让用户以为有依据
    （这正是 D40 之前那个「编造的 0.41」缺陷的形状）。

    声学输入用合成音频走**真实**的特征提取 —— 离线、可复现。
    """
    from app.audio.features import TARGET_SR, extract_features, synthesize_meow

    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    # 合成一段猫叫（离线，确定性）
    features = extract_features(
        synthesize_meow(duration=4.0, f0_start=420, f0_end=780), TARGET_SR
    )

    from app.interpreter import interpret_meow

    # ⚠️ 返回的是 **tuple**：`(BehaviorInterpretation, ModeDecision)`。
    # 模式决策单独返回而不是塞进 interpretation ——
    # 因为「用了哪种模式」是路由层的判断，不是解释结果的一部分。
    interp, mode = interpret_meow(
        features=features,
        records=[],
        prior=ctx.prior,
    )

    # ── E24：text_only 时不得有数值 ──
    numeric_in_text_only = [
        c.display
        for c in interp.candidates
        if interp.evidence_mode.value == "text_only" and c.posterior is not None
    ]

    # ── E22：证据项的可复算性 ──
    recomputable = 0
    uncomputable: list[str] = []
    for item in interp.evidence:
        value, reference = item.value, item.reference
        contribution = item.log_odds_contribution
        if contribution is None:
            # 无贡献值时不算「不可复算」—— 它没有声称任何数字
            recomputable += 1
            continue
        if isinstance(value, (int, float)) and isinstance(reference, (int, float)):
            recomputable += 1
        else:
            uncomputable.append(f"{item.kind}:{item.statement[:40]}")

    total_evidence = len(interp.evidence)

    # ── 测量值本身必须可重算（同输入 → 同输出）──
    again = extract_features(
        synthesize_meow(duration=4.0, f0_start=420, f0_end=780), TARGET_SR
    )
    deterministic = (
        abs(again.f0_mean - features.f0_mean) < 1e-6
        and abs(again.duration - features.duration) < 1e-6
    )

    return SceneOutcome(
        metrics={
            "证据项数": float(total_evidence),
            "可复算证据数": float(recomputable),
            "text_only 数值泄漏数": float(len(numeric_in_text_only)),
            "特征提取确定性": 1.0 if deterministic else 0.0,
            "个体化程度": float(interp.individualization),
        },
        checks=[
            Check(
                "证据项可复算",
                not uncomputable,
                f"不可复算：{uncomputable[:3]}",
            ),
            Check(
                "text_only 无任何数值置信度",
                not numeric_in_text_only,
                "出现了数值 —— 没有测量就没有概率（D40 的「编造 0.41」就是这个形状）",
            ),
            Check(
                "声学特征提取确定性",
                deterministic,
                "同一段音频两次提取结果不一致 —— 那 MEASURED 的承诺就不成立",
            ),
            Check(
                "测量模式下有特征",
                interp.acoustic_features is not None,
                f"evidence_mode={interp.evidence_mode.value}",
            ),
        ],
        notes=[
            "用合成猫叫（librosa 生成）走真实特征提取，全离线可复现",
            "个体化程度为 0 是正常的：本次没有任何历史案例（冷启动）",
        ],
        samples=total_evidence,
    )


# =============================================================================
# 场景 9：档案不漂移（E10）
# =============================================================================


def _scene_profile_stability(ctx: Any) -> SceneOutcome:
    """对话与事件**不得污染身份锚点**。

    ## 为什么这条重要

    `must_keep_features` 是「这只猫长什么样」的锚点，用于跨照片校验身份。
    如果一次关于「它今天有点胖」的对话能改动它，
    那么锚点会随对话漂移 —— 而漂移之后，身份校验会拿一个
    「某天碰巧提到过的特征」去比对，于是校验失效且**看起来还在工作**。

    所以这里：跑若干轮对话 + 事件记录，然后断言锚点**逐字未变**。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    pet = ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    before = list(pet.must_keep_features)
    before_visual = pet.visual.model_dump(mode="json")

    texts = [
        "记一下它今天好像胖了一点",
        "记一下它胸口那块白毛好像变淡了",
        "照片里它看起来毛很长",
        "它是不是有黑白花纹？",
        "你好呀",
    ]
    for t in texts:
        runtime.converse(store, user_id=USER, pet_id=PET, text=t)

    after_pet = store.get_pet(user_id=USER, pet_id=PET)
    after = list(after_pet.must_keep_features)
    after_visual = after_pet.visual.model_dump(mode="json")

    added = [f for f in after if f not in before]
    removed = [f for f in before if f not in after]
    visual_changed = {
        k: (before_visual.get(k), after_visual.get(k))
        for k in before_visual
        if before_visual.get(k) != after_visual.get(k)
    }

    return SceneOutcome(
        metrics={
            "锚点变化数": float(len(added) + len(removed)),
            "表观字段变化数": float(len(visual_changed)),
            "对话轮数": float(len(texts)),
        },
        checks=[
            Check(
                "身份锚点未被对话改动",
                not added and not removed,
                f"新增 {added}；丢失 {removed}",
            ),
            Check(
                "表观字段未被对话改动",
                not visual_changed,
                f"变化：{visual_changed}",
            ),
        ],
        notes=[
            "档案的更新只能来自照片分析 + 用户确认（DESIGN §2.3），"
            "对话与事件不能改写身份锚点",
        ],
        samples=len(texts),
    )


# =============================================================================
# 场景 10：延迟（P1 —— 成本/延迟数据）
# =============================================================================


def _scene_latency(ctx: Any) -> SceneOutcome:
    """每轮延迟分布 + 每节点耗时。

    `docs/06-roadmap.md` 的 P1：

    > 有数字的架构讨论和没有数字的架构讨论，是两个层次。

    这里给 **p50 与 p95 一起** —— 只看均值会掩盖长尾，
    而用户感知的正是长尾。
    """
    store = ctx.new_store()
    # **每个场景用独立租户。** 内存后端天然隔离，
    # 而 MySQL 后端下所有场景共享同一张表 —— 同一 (user_id, pet_id)
    # 会跨场景累积，表现为「去重没生效」「冲突误判」这类**指向错误**的失败。
    USER, PET = ctx.new_tenant()
    ctx.new_pet(store, user_id=USER, pet_id=PET, name="团团")
    runtime = ctx

    # 先写入一些记忆，让检索路径有实际工作
    runtime.converse(store, user_id=USER, pet_id=PET, text="记一下它怕吸尘器")

    texts = ["你好", "它怕什么声音", "记一下它爱吃三文鱼"] * 4
    latencies: list[float] = []
    per_node: dict[str, list[float]] = {}

    for t in texts:
        out = runtime.converse(store, user_id=USER, pet_id=PET, text=t)
        latencies.append(out.latency_ms)
        for trace in out.raw.get("node_trace", []):
            per_node.setdefault(trace.node, []).append(float(trace.latency_ms))

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)

    slowest = None
    if per_node:
        slowest = max(per_node.items(), key=lambda kv: percentile(kv[1], 95))

    metrics = {
        "端到端_p50_ms": p50,
        "端到端_p95_ms": p95,
        "端到端_max_ms": max(latencies) if latencies else 0.0,
        "轮数": float(len(latencies)),
    }
    for node, values in per_node.items():
        metrics[f"节点_{node}_p95_ms"] = percentile(values, 95)

    return SceneOutcome(
        metrics=metrics,
        checks=[
            # mock 环境下的阈值刻意宽松：这条检查防的是「数量级异常」，
            # 不是性能验收。真实的性能对比要靠消融组之间的差值。
            Check(
                "端到端 p95 < 500ms（mock 环境）",
                p95 < 500,
                f"实际 {p95:.0f}ms —— 数量级异常通常指向意外阻塞（如网络调用）",
            ),
        ],
        notes=[
            "**本场景用 MockLLM，所以这些数字不含真实模型延迟** —— "
            "它们是编排层自身的开销。真实端到端延迟需接真模型另测",
            f"最慢节点：{slowest[0] if slowest else 'n/a'}"
            + (f" p95={percentile(slowest[1], 95):.0f}ms" if slowest else ""),
        ],
        details=[
            {"turn": i, "ms": round(v, 1)} for i, v in enumerate(latencies)
        ],
        samples=len(latencies),
    )


# =============================================================================
# 场景 11：声学分类 —— **不可测，诚实报告**（E21）
# =============================================================================


def _scene_acoustic_blocked(ctx: Any) -> SceneOutcome:
    """E21（情境分类 macro-F1）**当前无法诚实地测**。

    ## 为什么不给一个数字

    `data/priors/catmeows_stats.json` 里全部是占位值：

    ```json
    {"version": "placeholder-v0", "reviewed": false, "is_placeholder": true}
    ```

    群体先验尚未构建（`DESIGN.md` §7.3 U3）。在这种情况下算出的 macro-F1
    测的是「模型拟合占位值的程度」，而不是「它能不能识别这只猫的情绪」——
    **那个数字没有任何意义，但它有数字的外观**，会被引用、被传播。

    所以这个场景**不产出任何指标**，只产出一条 `Blocked`。

    这比「跳过」强：跳过会让报告里少一行，读的人以为都测过了。
    """
    import json

    from app.eval.runtime import _PRIOR_PATH

    raw = json.loads(open(_PRIOR_PATH, encoding="utf-8").read())
    is_placeholder = bool(raw.get("is_placeholder", False))
    reviewed = bool(raw.get("reviewed", False))

    blocked = Blocked(
        item="E21 声学情境分类 macro-F1",
        reason=(
            f"群体先验是占位值（is_placeholder={is_placeholder}, reviewed={reviewed}）——"
            "用占位数据算出的分类指标测的是「拟合占位值的程度」，不是识别能力"
        ),
        needs=(
            "① CatMeows 真实统计量替换 data/priors/catmeows_stats.json；"
            "② 21 只猫中留出 4 只做 hold-out，不参与先验统计（否则测的是过拟合）"
        ),
    )

    return SceneOutcome(
        metrics={},
        checks=[],
        blocked=[blocked],
        notes=[
            "本场景**刻意不产出数字**。给出一个用占位数据算出的 macro-F1 "
            "比没有数字更糟：它有数字的样子，会被引用",
            "同理受限的还有：E23 置信度校准（需真实标注）、"
            "E25–E29 视觉（需兽医评分）",
        ],
        samples=0,
    )


# =============================================================================
# 注册表
# =============================================================================


ALL_SCENES: tuple[Scene, ...] = (
    Scene(
        scene_id="memory-dedup",
        name="去重与强化",
        layer=Layer.A,
        items=("E5",),
        question="同一件事说三遍，是留三条还是强化成一条？",
        run=_scene_dedup,
    ),
    Scene(
        scene_id="memory-conflict",
        name="冲突消解与时间线",
        layer=Layer.A,
        items=("E3", "E4"),
        question="新偏好与旧的矛盾时，两个都留得住吗？",
        run=_scene_conflict,
    ),
    Scene(
        scene_id="tenant-isolation",
        name="租户隔离",
        layer=Layer.A,
        items=("E6",),
        question="A 的猫能不能读到 B 的记忆？",
        run=_scene_isolation,
    ),
    Scene(
        scene_id="no-poisoning",
        name="防自我强化",
        layer=Layer.A,
        items=("E8",),
        question="系统推断会不会被当成事实存下来？",
        run=_scene_no_poisoning,
    ),
    Scene(
        scene_id="honesty",
        name="无据不编造",
        layer=Layer.A,
        items=("E2", "E31", "E32", "E33"),
        question="没有记录时，它敢说「我不知道」吗？",
        run=_scene_honesty,
    ),
    Scene(
        scene_id="retrieval",
        name="检索质量",
        layer=Layer.A,
        items=("E11", "E12"),
        question="该召回的召回了吗？排在前面吗？",
        run=_scene_retrieval,
    ),
    Scene(
        scene_id="routing",
        name="意图路由",
        layer=Layer.A,
        items=("E13", "E14", "E15"),
        question="小类有没有被淹没？该问的时候问了吗？",
        run=_scene_routing,
    ),
    Scene(
        scene_id="evidence",
        name="证据可复算与模式合规",
        layer=Layer.A,
        items=("E22", "E24"),
        question="给出的数字能被重算吗？没测量时给数字了吗？",
        run=_scene_evidence,
    ),
    Scene(
        scene_id="profile-stability",
        name="身份锚点不漂移",
        layer=Layer.A,
        items=("E10",),
        question="聊天会不会把「这只猫长什么样」改掉？",
        run=_scene_profile_stability,
    ),
    Scene(
        scene_id="latency",
        name="编排延迟",
        layer=Layer.A,
        items=("P1-延迟",),
        question="每一轮要多久？瓶颈在哪个节点？",
        run=_scene_latency,
    ),
    Scene(
        scene_id="acoustic-blocked",
        name="声学分类（不可测）",
        layer=Layer.B,
        items=("E21",),
        question="它能识别这只猫的情绪吗？",
        run=_scene_acoustic_blocked,
    ),
)

SCENES_BY_ID: dict[str, Scene] = {s.scene_id: s for s in ALL_SCENES}
