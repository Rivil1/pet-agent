#!/usr/bin/env python
"""真实 API 冒烟检查。

**为什么需要它**：`app/llm/openai_compat.py`、`app/llm/providers.py` 与
`app/profile/openai_compat.py` 的所有逻辑都用 ``httpx.MockTransport`` 测过了，
但那只验证了「我的代码自洽」，**不能验证「我对 API 的理解正确」**。
两者解决不同问题，且后者只能人工确认一次。

用法::

    export DASHSCOPE_API_KEY=sk-...            # 或 ARK_API_KEY
    python scripts/live_check.py                       # LLM + Embedder
    python scripts/live_check.py --provider ark        # 显式指定供应商
    python scripts/live_check.py --image-url https://... # 加上视觉

**供应商无关**：端点、模型名、路径全部来自 `ProviderConfig`，
与 `build_providers` 用的是同一套配置解析 —— 所以这个脚本验证的
正是生产会走的那条路径。

退出码 0 表示全部通过。

> 这个脚本存在的理由就是 `docs/BUGS.md` B13–B17：
> **写了一堆 API 代码但从没调过**是最容易被忽略的失败模式。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import (  # noqa: E402
    ENV_API_KEY,
    CapabilityConfig,
    MissingCredential,
    OpenAICompatEmbedder,
    OpenAICompatLLM,
    ProviderConfig,
    ProviderKind,
    UpstreamError,
    build_config,
    detect_provider,
)
from app.llm.base import cosine  # noqa: E402
from app.llm.providers import DEFAULT_PROVIDER  # noqa: E402
from app.profile.openai_compat import OpenAICompatVision  # noqa: E402

GREEN = "\033[32m✓\033[0m"
RED = "\033[31m✗\033[0m"
YELLOW = "\033[33m!\033[0m"

#: 已核实的单价（元/百万 token，华北2）。**仅 DashScope 参考** ——
#: 方舟等其它供应商的价目不同，成本数字只在 `kind=dashscope` 时可比。
PRICE_IN_PER_M = 0.8
PRICE_OUT_PER_M = 4.0


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, title: str, detail: str = "") -> None:
        print(f"  {GREEN} {title}" + (f"  {detail}" if detail else ""))

    def bad(self, title: str, detail: str = "") -> None:
        self.failures.append(title)
        print(f"  {RED} {title}" + (f"  {detail}" if detail else ""))

    def warn(self, title: str, detail: str = "") -> None:
        print(f"  {YELLOW} {title}" + (f"  {detail}" if detail else ""))


def _timed(fn, *a, **kw):
    start = time.monotonic()
    out = fn(*a, **kw)
    return out, (time.monotonic() - start) * 1000


def check_llm(cfg: CapabilityConfig, rep: Report) -> None:
    # 模型名**从配置读**，不写死：
    # 初版把 "text-embedding-v3" / "qwen-vl-max" 硬编码在标签里，
    # 而默认值早已换成 v4 / qwen3-vl-flash —— 于是脚本**报的是 A、测的是 B**。
    print(f"\n[1/4] 文本生成 ({cfg.model} @ {cfg.endpoint()})")
    llm = OpenAICompatLLM(config=cfg)
    try:
        text, ms = _timed(
            llm.complete,
            system="你是一个简洁的助手。",
            user="用一句话说明猫为什么会在门口叫。",
        )
    except (UpstreamError, MissingCredential) as exc:
        rep.bad("调用失败", str(exc))
        return

    if not text.strip():
        rep.bad("返回空内容")
        return
    rep.ok("调用成功", f"{ms:.0f}ms，返回 {len(text)} 字")
    print(f"      模型原文：{text[:80]}{'…' if len(text) > 80 else ''}")

    cost = (
        llm.usage.prompt_tokens * PRICE_IN_PER_M
        + llm.usage.completion_tokens * PRICE_OUT_PER_M
    ) / 1_000_000
    rep.ok(
        "token 统计",
        f"in={llm.usage.prompt_tokens} out={llm.usage.completion_tokens} "
        f"≈{cost:.5f} 元",
    )
    if llm.usage.prompt_tokens == 0:
        rep.warn("未取到 token 用量", "上游可能没有返回 usage 字段")


def check_embedder(cfg: CapabilityConfig, rep: Report) -> None:
    """**这里做语义检查，而不只是「返回了 1024 个数」。**

    真实 embedding 与哈希向量的唯一区别就是语义：相似句子的向量必须更近。
    只检查维度等于 1024 无法区分两者。
    """
    print(f"\n[2/4] 文本向量化 ({cfg.model} @ {cfg.endpoint()})")
    emb = OpenAICompatEmbedder(config=cfg)

    near_a = "它很怕吸尘器的声音，一开就跑"
    near_b = "听到吸尘器就躲起来"
    far = "它喜欢在窗台晒太阳"

    try:
        vecs, ms = _timed(emb.embed_many, [near_a, near_b, far])
    except (UpstreamError, MissingCredential) as exc:
        rep.bad("调用失败", str(exc))
        return

    if len(vecs) != 3:
        rep.bad("返回向量条数不符", f"期望 3，得到 {len(vecs)}")
        return
    rep.ok("调用成功", f"{ms:.0f}ms")

    dims = {len(v) for v in vecs}
    if dims != {cfg.dim}:
        rep.bad("维度不符", f"期望 {cfg.dim}，得到 {dims}")
    else:
        rep.ok("维度正确", f"{cfg.dim}（与 vector({cfg.dim}) 一致）")

    sim_near = cosine(vecs[0], vecs[1])
    sim_far = cosine(vecs[0], vecs[2])
    rep.ok("相似度", f"语义相近 {sim_near:.3f} / 语义无关 {sim_far:.3f}")

    if sim_near <= sim_far:
        rep.bad(
            "**语义区分失败**",
            "相近句子的相似度没有更高 —— 检索会退化成随机排序",
        )
    else:
        margin = sim_near - sim_far
        rep.ok("语义区分正确", f"margin={margin:+.3f}")
        if margin < 0.05:
            rep.warn("margin 偏小", "检索区分度可能不足，需在评测集上验证 recall@k")

    prompt_tokens = emb.usage.prompt_tokens
    cost = prompt_tokens * 0.5 / 1_000_000
    rep.ok("token 统计", f"in={prompt_tokens} ≈{cost:.6f} 元")


def check_vision(cfg: CapabilityConfig, rep: Report, image_url: str) -> None:
    print(f"\n[3/4] 视觉分析 ({cfg.model} @ {cfg.endpoint()})")
    vision = OpenAICompatVision(config=cfg)
    try:
        obs, ms = _timed(vision.analyze, image_url)
    except (UpstreamError, MissingCredential) as exc:
        rep.bad("调用失败", str(exc))
        return

    rep.ok("调用成功", f"{ms:.0f}ms")
    terms = obs.terms()
    if not terms:
        rep.warn(
            "未抽到任何特征",
            "模型判断这张照片可用信息不足（usable=false）。"
            "这是**正确行为**（不编造），请换一张正面清晰的照片再试",
        )
    else:
        rep.ok("抽取到特征", "、".join(terms))

    rep.ok("JSON 解析", "严格 JSON 结构校验通过")


def check_multimodal(cfg: CapabilityConfig, rep: Report, media_url: str) -> None:
    """多模态事实提取。

    **这一项尤其需要实测**，因为它的接口形状有几处是按文档写的、没验证过：

    - 音频/视频是否都用 `video_url` 内容块传（而不是 `audio_url`）
    - omni 类模型名是否可用
    - 输出的 JSON 是否与我们的解析器兼容

    而且它连着**三道防线**（prompt + 词表 + 契约），
    真实模型很可能返回词表外的值 —— 那正是要看的结果（会进 `ignored_terms`）。
    """
    from app.extract import MultimodalExtractor
    from app.schemas import MediaKind

    # ``model`` 留空即取 ``cfg.model`` —— 这份 cfg 就是 MULTIMODAL 能力自己的配置。
    extractor = MultimodalExtractor(config=cfg)
    print(f"\n[4/4] 多模态事实提取 ({cfg.model} @ {cfg.endpoint()})")
    try:
        obs, ms = _timed(extractor.extract, media_url, media_kind=MediaKind.VIDEO)
    except Exception as exc:  # noqa: BLE001 — 冒烟脚本要报告而不是崩
        rep.bad("调用失败", f"{type(exc).__name__}: {exc}")
        return

    if not obs.ok:
        rep.bad("返回不可用", obs.error or "(无原因)")
        return

    rep.ok("调用成功", f"{ms:.0f}ms，source_model={obs.source_model}")
    rep.ok(
        "解析结果",
        f"动作={[a.value for a in obs.actions]} "
        f"物体={obs.scene_objects} 迹象={len(obs.described_signs)}",
    )
    if obs.ignored_terms:
        # **不是错误，但要看** —— 高频出现词表外的值说明 prompt 或词表要改
        rep.warn(
            f"有 {len(obs.ignored_terms)} 个词表外的值被丢弃",
            f"{obs.ignored_terms}（说明 prompt 或词表需调整）",
        )
    if not obs.is_usable:
        rep.warn("无实质内容", "模型判断这段媒体看不出东西（可能是正确行为）")


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 API 冒烟检查")
    parser.add_argument(
        "--provider",
        choices=[k.value for k in ProviderKind],
        default=None,
        help="供应商。缺省时按环境变量推断（显式 key 名 > 裸 key 的默认供应商）",
    )
    parser.add_argument("--image-url", help="用于视觉检查的公开图片 URL")
    parser.add_argument("--media-url", help="用于多模态提取检查的视频/音频 URL")
    parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖所有能力的默认端点（按能力用 PET_AGENT_<能力>_BASE_URL 分别覆盖）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="覆盖所有能力的超时（秒）；缺省用各能力自己的 / PET_AGENT_<能力>_TIMEOUT_S",
    )
    args = parser.parse_args()

    key = (os.environ.get(ENV_API_KEY) or "").strip()
    print("=" * 62)
    print("pet-agent 真实 API 冒烟检查")
    print("=" * 62)

    # 与 `build_providers` 同一套推断：显式 --provider 优先，
    # 否则按环境变量；只有裸 `PET_AGENT_API_KEY` 时用有文档的默认供应商。
    kind = ProviderKind(args.provider) if args.provider else detect_provider()
    if kind is ProviderKind.MOCK and key:
        kind = ProviderKind(DEFAULT_PROVIDER)
    if kind is ProviderKind.MOCK:
        print(f"\n{RED} 未设置任何真实供应商密钥")
        print("   请先配置密钥后再运行：")
        print("     export DASHSCOPE_API_KEY=sk-...   # 或 ARK_API_KEY")
        print(
            f"     export {ENV_API_KEY}=sk-...         "
            f"# 裸 key（默认供应商 {DEFAULT_PROVIDER}）"
        )
        print("\n   注意：本脚本**不会**退回 mock —— 静默退回会让一次")
        print("   真实验证变成假的。")
        return 2

    try:
        cfg = build_config(kind=kind, api_key=key or None)
    except ValueError as exc:
        print(f"\n{RED} {exc}")
        return 2

    cfg = _override(cfg, base_url=args.base_url, timeout_s=args.timeout)

    print(f"\n供应商: {cfg.kind}（混用={cfg.is_mixed}）")
    for name, cap in (
        ("LLM   ", cfg.llm),
        ("视觉  ", cfg.vision),
        ("向量  ", cfg.embed),
        ("多模态", cfg.multimodal),
    ):
        if cap is None:
            print(f"  {name}: （未配置）")
            continue
        print(f"  {name}: {cap.model} @ {cap.endpoint()}")
    resolved_key = cfg.llm.api_key or key
    print(f"密钥: {resolved_key[:6]}…{resolved_key[-4:]}（长度 {len(resolved_key)}）")

    rep = Report()
    check_llm(cfg.llm, rep)
    check_embedder(cfg.embed, rep)
    if args.image_url:
        check_vision(cfg.vision, rep, args.image_url)
    else:
        print(f"\n[3/4] 视觉分析  {YELLOW} 跳过（未提供 --image-url）")
        print("      视觉需要**公网可访问**的图片 URL —— 与 ASR 同属一个约束")

    if args.media_url:
        if cfg.multimodal is None:
            print(
                f"\n[4/4] 多模态事实提取  {YELLOW} "
                "跳过（该供应商未配置 MULTIMODAL 能力）"
            )
        else:
            check_multimodal(cfg.multimodal, rep, args.media_url)
    else:
        print(f"\n[4/4] 多模态事实提取  {YELLOW} 跳过（未提供 --media-url）")

    print("\n" + "=" * 62)
    if rep.failures:
        print(f"{RED} {len(rep.failures)} 项失败：")
        for f in rep.failures:
            print(f"    · {f}")
        return 1
    print(f"{GREEN} 全部通过")
    print("\n下一步：把结果记进 docs/BUGS.md（首次真实调用往往能暴露新问题）")
    return 0


def _override(
    cfg: ProviderConfig, *, base_url: str | None, timeout_s: float | None
) -> ProviderConfig:
    """应用 ``--base-url`` / ``--timeout``。

    `CapabilityConfig` 是 frozen dataclass，且 base_url 是**每个能力各自的字段**
    （方舟与 DashScope 的端点不同），所以只能 replace 而不能赋值。

    ``None`` 表示「不改」—— 尤其 timeout：``PET_AGENT_<能力>_TIMEOUT_S``
    是按能力设的，缺省值不应当把用户的按能力设置冲掉。
    """

    def fix(cap: CapabilityConfig) -> CapabilityConfig:
        changes: dict[str, object] = {}
        if base_url:
            changes["base_url"] = base_url
        if timeout_s is not None:
            changes["timeout_s"] = timeout_s
        return replace(cap, **changes) if changes else cap

    return replace(
        cfg,
        llm=fix(cfg.llm),
        vision=fix(cfg.vision),
        embed=fix(cfg.embed),
        multimodal=fix(cfg.multimodal) if cfg.multimodal else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
