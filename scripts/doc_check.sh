#!/usr/bin/env bash
# 设计文档一致性检查。
#
# 用途：把「文档审查」从人工翻阅变成可复现的自动检查。
# 背景：docs/15-doc-audit.md 发现的问题几乎都是「文档间不同步」，
#      而不同步之所以无人察觉，是因为检查靠人眼。
#
# 用法：  bash scripts/doc_check.sh
# 退出码：0 = 全部通过；1 = 发现问题

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

# 优先用项目 venv 的 python（合约检查需要 import app.schemas）
if [ -x /root/.venvs/pet-agent/bin/python ]; then
  PY=/root/.venvs/pet-agent/bin/python
else
  PY=python3
fi

ISSUES=0
WARNINGS=0

section() { printf '\n\033[1m%s\033[0m\n' "$1"; }
fail() {
  printf '  \033[31m✗ %s\033[0m\n' "$1"
  ISSUES=$((ISSUES + 1))
}
warn() {
  printf '  \033[33m! %s\033[0m\n' "$1"
  WARNINGS=$((WARNINGS + 1))
}
ok() { printf '  \033[32m✓ %s\033[0m\n' "$1"; }

# ─────────────────────────────────────────────────────────────
section "1. README 索引 vs 实际文件"
# ─────────────────────────────────────────────────────────────
for f in $(grep -o 'docs/[0-9][0-9]-[a-z-]*\.md' README.md | sed 's|docs/||' | sort -u); do
  if [ -f "docs/$f" ]; then
    ok "$f"
  else
    # 索引里标了 ⬜ 的是计划中未写的，可接受；被正文链接的才算悬空
    if grep -q "($f)" docs/*.md 2>/dev/null; then
      fail "$f 未写但被正文链接（悬空链接）"
    else
      warn "$f 未写（仅索引列示，可接受）"
    fi
  fi
done

# 反向：有文件但索引没列
for f in docs/*.md; do
  b=$(basename "$f")
  grep -q "$b" README.md || fail "$b 存在但未列入 README 索引"
done

# ─────────────────────────────────────────────────────────────
section "2. 内部链接目标存在性"
# ─────────────────────────────────────────────────────────────
grep -oh '](\(\.\./\)*docs/[^)]*\.md\|]([0-9][0-9]-[a-z-]*\.md' docs/*.md |
  sed 's|](||; s|\.\./||; s|docs/||' | sort -u | while read -r f; do
  [ -f "docs/$f" ] && echo "OK $f" || echo "MISS $f"
done | grep '^MISS' | while read -r _ f; do fail "链接目标缺失：$f"; done
ok "（若无 ✗ 则全部存在）"

# ─────────────────────────────────────────────────────────────
section "3. 术语一致性（以 DESIGN.md 为权威）"
# ─────────────────────────────────────────────────────────────
# 权威文档必须用规范名；旧存档已标记为 reasoning archive，允许保留旧名。
AUTH=docs/DESIGN.md
for canonical in "记忆三层" "信息分层" "展示分层"; do
  if grep -q "$canonical" "$AUTH"; then
    ok "$AUTH 使用规范名「$canonical」"
  else
    warn "$AUTH 未出现规范名「$canonical」"
  fi
done
# 只检查正文，排除附录（附录的「已废弃术语」列表里出现旧名是正确的）
AUTH_BODY=$(sed '/^# 附录/,$d' docs/DESIGN.md)
for deprecated in "三层信息分离" "三层信息架构" "三层架构" "猫语翻译"; do
  c=$(printf '%s' "$AUTH_BODY" | grep -c "$deprecated" 2>/dev/null || true)
  if [ "${c:-0}" -gt 0 ]; then
    fail "$AUTH 正文仍使用已废弃术语「$deprecated」（应仅在附录的废弃列表中提及）"
  fi
done
ok "（若无 ✗ 则权威文档正文无废弃术语）"

# ─────────────────────────────────────────────────────────────
section "4. 数字一致性"
# ─────────────────────────────────────────────────────────────
# 意图数量：
#   枚举值数（含兜底态 AMBIGUOUS）应等于「有策略数 + 兜底态数」
n_enum=$(
  $PY - <<'PY' 2>/dev/null || echo "?"
import re, pathlib
src = pathlib.Path("app/schemas/input.py").read_text(encoding="utf-8")
body = src.split("class InputIntent(str, Enum):")[1].split("\nclass ", 1)[0]
print(len(re.findall(r"^\s{4}[A-Z_]+ = \"", body, re.M)))
PY
)
n_policies=$(
  $PY - <<'PY' 2>/dev/null || echo "?"
import sys, pathlib
sys.path.insert(0, ".")
from app.schemas import INTENT_POLICIES
print(len(INTENT_POLICIES))
PY
)
declared=$(grep -o "当前 [0-9]* 个意图" docs/09-intent-and-planning.md | grep -o '[0-9]*' | head -1)
printf '  意图枚举值 %s 个，有路由策略 %s 个，docs/09 声明 %s 个\n' \
  "${n_enum:-?}" "${n_policies:-?}" "${declared:-?}"
# docs/09 统计的是「有策略的意图」，应等于 n_policies；
# 若它等于 n_enum 则说明把兜底态 AMBIGUOUS 也算进去了，概念混淆。
if [ "${declared:-x}" = "${n_policies:-y}" ]; then
  ok "一致（声明值 = 有策略意图数 $n_policies；枚举共 $n_enum 个，含兜底态 AMBIGUOUS）"
elif [ "${declared:-x}" = "${n_enum:-y}" ]; then
  fail "docs/09 声明值等于枚举值 $n_enum，隐含把兜底态 AMBIGUOUS 当作普通意图"
else
  fail "不一致：声明 $declared，有策略 $n_policies，枚举 $n_enum"
fi

# 文档进度声明
claimed=$(grep -o "设计文档（[0-9]*/[0-9]*" README.md | grep -o '[0-9]*/[0-9]*' | head -1)
nfiles=$(ls docs/*.md | wc -l)
nindex=$(grep -c '^| \[docs/' README.md)
printf '  文档进度：README 声明 %s，实际文件 %s，索引条目 %s\n' "${claimed:-?}" "$nfiles" "$nindex"
[ "${claimed%%/*}" = "$nfiles" ] && [ "${claimed##*/}" = "$nindex" ] &&
  ok "一致" || fail "声明 $claimed 与实际（$nfiles 文件 / $nindex 索引）不符"

# ─────────────────────────────────────────────────────────────
section "5. 产品化改造是否已落入权威文档"
# ─────────────────────────────────────────────────────────────
# 不再检查旧存档（00/01/04）——它们已被 DESIGN.md 取代。
# 当前要防的是：权威文档漏掉了已决定的产品机制。
AUTH=docs/DESIGN.md
for concept in InteractionMode InfoLayer Moment CompanionResponse PriorAdjustment; do
  c=$(grep -c "$concept" "$AUTH" 2>/dev/null || true)
  if [ "${c:-0}" -eq 0 ]; then
    fail "$AUTH 缺少 $concept"
  else
    ok "$concept（$c 次）"
  fi
done

for kw in "想猫时" "看不懂它时"; do
  if grep -q "$kw" "$AUTH"; then
    ok "场景定义包含「$kw」"
  else
    fail "$AUTH 缺场景「$kw」"
  fi
done

# 旧存档是否已在 README 标记为 reasoning archive
if grep -q "推理存档\|reasoning archive" README.md; then
  ok "旧存档已在 README 标记为推理存档"
else
  warn "旧存档未在 README 标记为存档，读者可能误当权威"
fi

# ─────────────────────────────────────────────────────────────
section "6. 已修复问题的表述是否仍为现在时（易误导）"
# ─────────────────────────────────────────────────────────────
if [ -f docs/10-self-review.md ]; then
  if grep -q "修复记录" docs/10-self-review.md; then
    # 检查已修条目是否在正文标注
    for tag in A1 A2; do
      if sed -n "/^### $tag/,/^### /p" docs/10-self-review.md | grep -q "已修"; then
        ok "$tag 正文已标注状态"
      else
        warn "$tag 已在 §5.1 标记修复，但正文仍以现在时描述问题"
      fi
    done
  fi
fi

# ─────────────────────────────────────────────────────────────
section "7. 被引用的数据产物是否存在"
# ─────────────────────────────────────────────────────────────
# 若权威文档已诚实声明该产物未构建，则降为警告（诚实标注是允许的状态）。
for p in $(grep -oh 'data/[a-z/]*\.json\|data/[a-z/]*/[a-z_]*\.\(json\|yaml\)' docs/*.md | sort -u); do
  if [ -e "$p" ]; then
    ok "$p"
  elif grep -q "$p" docs/DESIGN.md 2>/dev/null &&
    grep -qE "未构建|不存在|待构建" docs/DESIGN.md; then
    warn "$p 未构建（DESIGN.md 已诚实声明，见 §7.3 未决项）"
  else
    fail "$p 被文档引用但不存在且未声明"
  fi
done

# ─────────────────────────────────────────────────────────────
printf '\n\033[1m结论：%d 个错误，%d 个警告\033[0m\n' "$ISSUES" "$WARNINGS"
[ "$ISSUES" -eq 0 ] || exit 1
