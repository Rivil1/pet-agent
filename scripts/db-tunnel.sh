#!/usr/bin/env bash
# =============================================================================
# 数据库隧道：本地 → 服务器 MySQL / Milvus
#
# ## 为什么是隧道而不是直接暴露端口
#
# 把 3306 开在公网上，等于把「user_id 不可伪造」这条隔离前提
# 降级成「密码别泄露」—— 而本项目整个 D5（多租户隔离）主张就建在前者上。
# 隧道让数据库在网络上**不可达**，只有持有 SSH 私钥的人能连进去。
#
# 用法：
#   scripts/db-tunnel.sh              # 前台运行，Ctrl-C 断开
#   scripts/db-tunnel.sh --daemon     # 后台运行
#   scripts/db-tunnel.sh --stop       # 停掉后台隧道
#   scripts/db-tunnel.sh --status     # 看隧道状态
#
# 连上后本地这样配：
#   MYSQL_HOST=127.0.0.1
#   MYSQL_PORT=13306
#   MILVUS_HOST=127.0.0.1
#   MILVUS_PORT=19530
# =============================================================================

set -euo pipefail

SERVER="${SERVER:-47.102.186.248}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/aaa.pem}"

# 本地端口刻意不用 3306：避免和本机已有的 MySQL 撞车，
# 那种撞车会连到**错误的库**上，而且看起来一切正常。
LOCAL_MYSQL_PORT="${LOCAL_MYSQL_PORT:-13306}"
LOCAL_MILVUS_PORT="${LOCAL_MILVUS_PORT:-19530}"
LOCAL_MILVUS_WEB_PORT="${LOCAL_MILVUS_WEB_PORT:-19091}"

PIDFILE="${TMPDIR:-/tmp}/pet-agent-tunnel.pid"
LOGFILE="${TMPDIR:-/tmp}/pet-agent-tunnel.log"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; DIM=$'\033[2m'; NC=$'\033[0m'

[[ -f "$SSH_KEY" ]] || { echo "${RED}[ERR]${NC} 找不到私钥：$SSH_KEY" >&2; exit 1; }

# 私钥权限过宽会被 ssh 拒绝
KEY_MODE=$(stat -c '%a' "$SSH_KEY" 2>/dev/null || echo '')
if [[ -n "$KEY_MODE" && "$KEY_MODE" != "600" && "$KEY_MODE" != "400" ]]; then
  cp "$SSH_KEY" /tmp/.pet-tunnel-key && chmod 600 /tmp/.pet-tunnel-key
  SSH_KEY=/tmp/.pet-tunnel-key
fi

is_running() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null || echo '')
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

start_tunnel() {
  local foreground="$1"
  if is_running; then
    echo "${YELLOW}隧道已在运行${NC}（PID $(cat "$PIDFILE")）。先 --stop 再重启。"
    return 0
  fi

  local args=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=no
    -o ExitOnForwardFailure=yes      # 端口占用时立刻失败，而不是假装连上了
    -o ServerAliveInterval=30        # 保活：否则空闲会被 NAT 掐掉
    -o ServerAliveCountMax=3
    # ⚠️ 必须禁用 GSSAPI。
    #
    # 不禁用时，ssh 会先尝试 gssapi-with-mic，在没有 Kerberos 凭据的机器上
    # 这一步会挂很久（实测 15 秒仍然没走到公钥认证）——
    # 表现为「ssh 进程活着、端口却没绑上」，而日志里
    # 只看到一句 “No Kerberos credentials available”，很容易误判为网络问题。
    # 这台机器只用公钥，所以直接指定认证方式，跳过整个协商。
    -o GSSAPIAuthentication=no
    -o PreferredAuthentications=publickey
    -o ConnectTimeout=15
    -N
    -L "${LOCAL_MYSQL_PORT}:127.0.0.1:3306"
    -L "${LOCAL_MILVUS_PORT}:127.0.0.1:19530"
    -L "${LOCAL_MILVUS_WEB_PORT}:127.0.0.1:9091"
    "$SSH_USER@$SERVER"
  )

  if [[ "$foreground" == "1" ]]; then
    echo "${GREEN}隧道已建立${NC}（Ctrl-C 断开）"
    print_endpoints
    exec ssh "${args[@]}"
  fi

  # 后台：用 setsid 脱离当前会话，否则脚本退出后隧道一起没了
  setsid ssh "${args[@]}" >"$LOGFILE" 2>&1 &
  echo $! > "$PIDFILE"
  # 需要等一下：ssh 要完成认证才会绑端口。等太短会误报失败。
  sleep 5

  if is_running; then
    if port_open "$LOCAL_MYSQL_PORT"; then
      echo "${GREEN}隧道已在后台建立${NC}（PID $(cat "$PIDFILE")）"
      print_endpoints
    else
      echo "${YELLOW}ssh 进程在，但本地端口未就绪${NC}——可能仍在认证。"
      echo "${DIM}稍后用 --status 复查。若持续未就绪，看 $LOGFILE${NC}"
      print_endpoints
    fi
  else
    echo "${RED}隧道启动失败${NC}" >&2
    cat "$LOGFILE" >&2 || true
    rm -f "$PIDFILE"
    exit 1
  fi
}

stop_tunnel() {
  if ! is_running; then
    echo "${DIM}没有在运行的隧道${NC}"
    rm -f "$PIDFILE"
    return 0
  fi
  local pid; pid=$(cat "$PIDFILE")
  kill "$pid" 2>/dev/null || true
  sleep 1
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "${GREEN}隧道已停止${NC}"
}

status_tunnel() {
  if is_running; then
    echo "${GREEN}运行中${NC}（PID $(cat "$PIDFILE")）"
    print_endpoints
    echo
    echo "连通性："
    check_port "$LOCAL_MYSQL_PORT" "MySQL"
    check_port "$LOCAL_MILVUS_PORT" "Milvus gRPC"
  else
    echo "${YELLOW}未运行${NC}"
    return 1
  fi
}

check_port() {
  local port="$1" name="$2"
  if port_open "$port"; then
    echo "  ${GREEN}✓${NC} $name  127.0.0.1:$port"
  else
    echo "  ${RED}✗${NC} $name  127.0.0.1:$port 不通"
  fi
}

#: 用 python 而不是 /dev/tcp：后者不是所有 shell 都支持（dash 就没有），
#: 而“检查工具不可用”会被误读成“端口不通”。
port_open() {
  python3 -c "
import socket, sys
s = socket.socket(); s.settimeout(3)
try:
    s.connect(('127.0.0.1', int(sys.argv[1])))
except Exception:
    sys.exit(1)
finally:
    s.close()
" "$1" 2>/dev/null
}

print_endpoints() {
  echo
  echo "  ${DIM}MySQL      127.0.0.1:$LOCAL_MYSQL_PORT${NC}"
  echo "  ${DIM}Milvus     127.0.0.1:$LOCAL_MILVUS_PORT${NC}"
  echo "  ${DIM}Milvus UI  127.0.0.1:$LOCAL_MILVUS_WEB_PORT${NC}"
  echo
  echo "  ${DIM}本地 .env：MYSQL_HOST=127.0.0.1  MYSQL_PORT=$LOCAL_MYSQL_PORT${NC}"
}

case "${1:-}" in
  --stop)   stop_tunnel ;;
  --status) status_tunnel ;;
  --daemon) start_tunnel 0 ;;
  "")       start_tunnel 1 ;;
  *)        echo "用法：$0 [--daemon|--stop|--status]" >&2; exit 1 ;;
esac
