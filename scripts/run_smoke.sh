#!/usr/bin/env bash
# pet-agent 真实运行冒烟：真 uvicorn 进程 + 真 HTTP 打全部业务端点。
#
# 为什么要有一个「真进程」的冒烟（而不是只用 TestClient）：
# TestClient 不经过真实 ASGI 服务器、不经过真实网络下载音频，
# 因此会发现不了启动期错误（中间件注册、先验加载、uvicorn factory 调用约定）。
#
# 用法：bash scripts/run_smoke.sh
# 退出码：0 = 全部通过。

set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PY=/root/.venvs/pet-agent/bin/python
PORT=8199
MEDIA_PORT=8291
SECRET="${PET_AGENT_AUTH_SECRET:-dev-secret}"
UV_LOG=$(mktemp)
MEDIA_LOG=$(mktemp)

cleanup() {
  [ -n "${UV_PID:-}" ] && kill "$UV_PID" 2>/dev/null
  [ -n "${MEDIA_PID:-}" ] && kill "$MEDIA_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT

echo "== 1/3 生成真实音频（声学特征需要可下载的 wav）=="
"$PY" - <<'PY' || exit 2
import soundfile as sf
from app.audio.features import TARGET_SR, synthesize_meow
sf.write("/tmp/meow.wav", synthesize_meow(duration=4.0, f0_start=420, f0_end=780), TARGET_SR)
print("  /tmp/meow.wav ok")
PY

echo "== 2/3 起静态音频服务 + uvicorn（真进程）=="
(cd /tmp && "$PY" -m http.server "$MEDIA_PORT" >"$MEDIA_LOG" 2>&1) &
MEDIA_PID=$!

env -u DASHSCOPE_API_KEY -u ARK_API_KEY -u PET_AGENT_API_KEY -u MOCK_PROVIDER \
  -u LANGSMITH_API_KEY -u LANGCHAIN_API_KEY \
  PET_AGENT_AUTH_SECRET="$SECRET" \
  PET_AGENT_TRACING=0 \
  "$PY" -m uvicorn app.bootstrap:create_app_from_env \
  --factory --host 127.0.0.1 --port "$PORT" >"$UV_LOG" 2>&1 &
UV_PID=$!

for _ in $(seq 1 60); do
  sleep 0.5
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
done

if ! curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  echo "!! 服务未起来。uvicorn 日志："
  tail -30 "$UV_LOG"
  exit 1
fi
echo "  uvicorn 已就绪 :$PORT"

echo "== 3/3 打全部业务端点 =="
PYTHONPATH=. PET_AGENT_AUTH_SECRET="$SECRET" "$PY" scripts/smoke_all_endpoints.py
RC=$?

echo
echo "== uvicorn 日志尾（确认无隐藏异常）=="
tail -25 "$UV_LOG"
exit "$RC"
