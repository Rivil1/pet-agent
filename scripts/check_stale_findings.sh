#!/usr/bin/env bash
# 核验那两条反复出现的「陈旧发现」是否真实存在。
#
# 背景：自动化检查反复报告
#   test_memory_store.py L97/L109 与 test_interpreter.py L304/L349/L410
# 而每次核验都显示那些位置是别的内容。本脚本用**多种独立手段**判定，
# 供任何人随时复现（对应 docs/BUGS.md §5.1 的 V1–V6 规则）。
#
# 用法：bash scripts/check_stale_findings.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=/root/.venvs/pet-agent/bin/python

echo "════ V1 文件身份（sha256，防止核验错文件）════"
for f in tests/test_memory_store.py tests/test_interpreter.py; do
  printf '  %s  sha=%s\n' "$f" "$(sha256sum "$f" | cut -c1-12)"
done

echo
echo "════ V2 报告位置的真实内容 ════"
for spec in "tests/test_memory_store.py 97" "tests/test_memory_store.py 109" \
            "tests/test_interpreter.py 304" "tests/test_interpreter.py 349" \
            "tests/test_interpreter.py 410"; do
  set -- $spec
  printf '  %s L%s: %s\n' "$1" "$2" "$(sed -n "${2}p" "$1")"
done

echo
echo "════ V4 换探测手段：全仓库搜缺陷形态（不只这两个文件）════"
echo -n "  空 dict() 调用: "; grep -rn "dict()" tests/ app/ 2>/dev/null | wc -l
echo -n "  '**' 解包进构造函数: "; grep -rnE "(MemoryEvent|MeowRecord|PetProfile)\(\*\*" tests/ app/ 2>/dev/null | wc -l

echo
echo "════ V4b AST 级判定（不看文本，看语法树）════"
"$PY" - <<'PY'
import ast, pathlib
empty_dict, missing = [], []
for p in list(pathlib.Path("tests").rglob("*.py")) + list(pathlib.Path("app").rglob("*.py")):
    tree = ast.parse(p.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
           and n.func.id == "dict" and not n.args and not n.keywords:
            empty_dict.append(f"{p}:{n.lineno}")
    defined = {x.name for x in tree.body if isinstance(x, (ast.FunctionDef, ast.ClassDef))}
    used = {x.id for x in ast.walk(tree) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
    for name in ("posterior_of", "logit_of"):
        if name in used and name not in defined:
            missing.append(f"{p}: 引用但未定义 {name}")
print("  真正的空 dict() 调用:", empty_dict or "0 处")
print("  引用但未定义的 helper:", missing or "0 处")
PY

echo
echo "════ V5 清字节码缓存后重跑（缓存里同时有 3.9/3.10 两套）════"
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo -n "  全项目 pyright: "; "$PY" -m pyright 2>&1 | tail -1
echo -n "  指名两文件:     "; "$PY" -m pyright tests/test_memory_store.py tests/test_interpreter.py 2>&1 | tail -1
echo -n "  测试:           "; "$PY" -m pytest tests/test_memory_store.py tests/test_interpreter.py -q 2>&1 | tail -1

echo
echo "════ 结论 ════"
echo "  若上面全部为零/无错，则报告里的那五条是**陈旧快照**。"
echo "  详见 docs/BUGS.md §5.1（V1–V6）。"
