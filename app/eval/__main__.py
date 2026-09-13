"""评测 CLI。

    python -m app.eval                    # 基线
    python -m app.eval --ablation         # 基线与全部消融组
    python -m app.eval --list             # 列出场景与配置
    python -m app.eval --only honesty     # 只跑一个场景
    python -m app.eval --json             # 只输出 JSON（给 CI）
    python -m app.eval --strict           # 有断言失败时以非零码退出

`--strict` 是给 CI 用的：让它能在 PR 上跑，断言失败即构建失败。
默认**不** strict —— 手动跑评测时，看到失败比被中断更有用。
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from app.eval.config import PRESETS
from app.eval.report import render_json, render_markdown, write_report
from app.eval.runner import compare, run_all_configs, run_suite
from app.eval.scenes import ALL_SCENES


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.eval",
        description="pet-agent 评测：产出带基线的数字",
    )
    p.add_argument(
        "--config",
        default="baseline",
        help=f"配置名。可选：{', '.join(PRESETS)}",
    )
    p.add_argument(
        "--ablation",
        action="store_true",
        help="跑基线与全部消融组并输出对比",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="同 --ablation（别名，语义更明确）",
    )
    p.add_argument(
        "--only",
        default=None,
        help="只跑这些场景（逗号分隔的 scene_id）",
    )
    p.add_argument("--list", action="store_true", help="列出场景与配置后退出")
    p.add_argument("--json", action="store_true", help="输出 JSON 而非 Markdown")
    p.add_argument(
        "--out",
        default="reports",
        help="报告输出目录（默认 reports/）",
    )
    p.add_argument("--no-write", action="store_true", help="不写文件，只打到 stdout")
    p.add_argument(
        "--strict",
        action="store_true",
        help="有断言失败 / 场景崩溃时以退出码 1 结束（CI 用）",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="不打印进度")
    return p


def _cmd_list() -> int:
    print("场景：")
    for s in ALL_SCENES:
        items = "/".join(s.items)
        print(f"  {s.scene_id:22} [{s.layer.value}] {items:16} {s.name}")
        print(f"  {'':22} {s.question}")
    print()
    print("配置：")
    for name, cfg in PRESETS.items():
        print(f"  {name:18} {cfg.describe()}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        return _cmd_list()

    if args.config not in PRESETS:
        print(
            f"未知配置 {args.config!r}。可选：{', '.join(PRESETS)}",
            file=sys.stderr,
        )
        return 2

    only = [s.strip() for s in args.only.split(",")] if args.only else None
    verbose = not args.quiet and not args.json

    if args.ablation or args.all_configs:
        suites = run_all_configs(verbose=verbose)
        order = ["baseline", *[n for n in PRESETS if n != "baseline"]]
        ordered = [suites[n] for n in order if n in suites]
        # 用**声明好的配对**，而不是「基线与每个实验组」。
        # 有些对比只在特定两组之间才有意义 —— 例如守卫的价值不在
        # `baseline → no-guard`（中性 LLM 下两边都是 0），
        # 而在 `faulty-llm-guarded → faulty-llm-unguarded`。
        from app.eval.config import COMPARISON_PAIRS

        comparisons = [
            compare(suites[a], suites[b])
            for a, b in COMPARISON_PAIRS
            if a in suites and b in suites
        ]
    else:
        suite = run_suite(PRESETS[args.config], only=only, verbose=verbose)
        ordered = [suite]
        comparisons = []

    # 单配置时也允许 --only 生效（`run_suite` 已处理）

    payload = (
        render_json(ordered)
        if args.json
        else render_markdown(ordered, comparisons)
    )

    if args.json:
        print(payload)
    elif args.no_write:
        # 只看不写时把报告打到 stdout
        print(payload)

    if not args.no_write:
        md_path, json_path = write_report(ordered, comparisons, out_dir=args.out)
        print(f"\n📄 {md_path}")
        print(f"🧾 {json_path}")

    if args.strict:
        failed = sum(len(s.failed) for s in ordered)
        errored = sum(len(s.errored) for s in ordered)
        if failed or errored:
            print(
                f"\n❌ {failed} 个场景有断言失败，{errored} 个崩溃",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
