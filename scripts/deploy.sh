#!/usr/bin/env bash
# =============================================================================
# pet-agent 部署脚本（本地 → 服务器）
#
# ## 为什么不是「在服务器上 git pull」
#
# 服务器在境内，**连不上 github.com**（实测 `git fetch` 返回
# `Empty reply from server`）。所以代码必须由本地**推**过去。
#
# 镜像也在服务器本地构建（而不是推镜像过去）：服务器已配好
# npm / pypi / debian / docker registry 的国内镜像源，构建能跑通，
# 而且省掉传 700MB+ 镜像的时间。
#
# 用法：
#   scripts/deploy.sh              # 部署到默认服务器
#   SERVER=1.2.3.4 scripts/deploy.sh
#   scripts/deploy.sh --no-build   # 只同步代码，不重建
# =============================================================================

set -euo pipefail

SERVER="${SERVER:-47.102.186.248}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/aaa.pem}"
REMOTE_DIR="${REMOTE_DIR:-/opt/pet-agent}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=15)

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; DIM=$'\033[2m'; NC=$'\033[0m'
info()  { echo "${GREEN}[INFO]${NC} $*"; }
warn()  { echo "${YELLOW}[WARN]${NC} $*"; }
error() { echo "${RED}[ERR ]${NC} $*" >&2; }

BUILD=1
[[ "${1:-}" == "--no-build" ]] && BUILD=0

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# ── 前置检查 ─────────────────────────────────────────────────
[[ -f "$SSH_KEY" ]] || { error "找不到 SSH 私钥：$SSH_KEY"; exit 1; }
# 私钥权限过宽会被 ssh 拒绝，而提示信息很容易被忽略
KEY_MODE=$(stat -c '%a' "$SSH_KEY" 2>/dev/null || echo '')
if [[ -n "$KEY_MODE" && "$KEY_MODE" != "600" && "$KEY_MODE" != "400" ]]; then
  warn "私钥权限为 $KEY_MODE，ssh 会拒绝加载。正在拷到临时文件并收紧权限…"
  cp "$SSH_KEY" /tmp/.pet-deploy-key
  chmod 600 /tmp/.pet-deploy-key
  SSH_KEY=/tmp/.pet-deploy-key
fi

SSH=(ssh -i "$SSH_KEY" "${SSH_OPTS[@]}" "$SSH_USER@$SERVER")
SCP=(scp -i "$SSH_KEY" "${SSH_OPTS[@]}")

# ── 1. 本地预检 ──────────────────────────────────────────────
info "预检：前端类型与构建"
if [[ -d web/node_modules ]]; then
  (cd web && npm run --silent typecheck)
  (cd web && npm run --silent build)
else
  warn "web/node_modules 不存在，跳过前端构建预检"
fi

# ── 2. 打包 ─────────────────────────────────────────────────
info "打包源码"
TARBALL="$(mktemp -t pet-src-XXXXXX.tar.gz)"
tar \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='dist' \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='.ruff_cache' \
  --exclude='*.log' \
  --exclude='.env' \
  -czf "$TARBALL" .
info "包大小：$(du -h "$TARBALL" | cut -f1)"

# ── 3. 上传 ─────────────────────────────────────────────────
info "上传到 $SSH_USER@$SERVER"
"${SCP[@]}" "$TARBALL" "$SSH_USER@$SERVER:/root/pet-src.tar.gz"
rm -f "$TARBALL"

# ── 4. 服务器上替换 + 重建 ───────────────────────────────────
info "同步代码"
"${SSH[@]}" "
  set -e
  cd '$REMOTE_DIR'

  # .env 只存在于服务器，且被 .gitignore 排除 —— 任何同步路径都拿不到它，
  # 所以必须先备份，否则一次部署就会把生产密钥抹掉。
  cp .env /root/pet-agent.env.backup

  find . -mindepth 1 -maxdepth 1 ! -name '.env' -exec rm -rf {} +
  tar -xzf /root/pet-src.tar.gz -C .
  cp /root/pet-agent.env.backup .env
  rm -f /root/pet-src.tar.gz
  echo '  ✓ 代码已同步，.env 已保留'
"

if [[ "$BUILD" == "1" ]]; then
  info "服务器本地构建（这一步较慢，国内镜像源限速）"
  "${SSH[@]}" "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml build"
fi

info "重启服务"
"${SSH[@]}" "
  cd '$REMOTE_DIR'
  docker compose -f docker-compose.prod.yml up -d --remove-orphans
  docker image prune -f >/dev/null 2>&1 || true
"

# ── 5. 健康检查 ─────────────────────────────────────────────
info "健康检查（Nginx → 后端 完整路径）"
if "${SSH[@]}" "
  for i in \$(seq 1 45); do
    if curl -sf http://127.0.0.1/healthz >/dev/null; then exit 0; fi
    sleep 2
  done
  exit 1
"; then
  echo
  info "✅ 部署成功"
  "${SSH[@]}" "curl -s http://127.0.0.1/healthz" | head -c 400
  echo
  echo
  echo "${DIM}  前端   http://$SERVER/${NC}"
  echo "${DIM}  API    http://$SERVER/healthz${NC}"
  echo "${DIM}  文档   http://$SERVER/docs${NC}"
else
  error "❌ 健康检查超时"
  "${SSH[@]}" "cd '$REMOTE_DIR' && docker compose -f docker-compose.prod.yml ps && docker compose -f docker-compose.prod.yml logs --tail=60"
  exit 1
fi
