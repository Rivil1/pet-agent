"""pet-agent 真实 HTTP 冒烟：打全部业务端点。

跑法（由 run_smoke.sh 调用）：
    PYTHONPATH=. python /tmp/pet_smoke.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import httpx

from app.auth.token import issue_token

BASE = "http://127.0.0.1:8199"
#: 必须与服务端一致 —— 从同一个环境变量读，而不是在脚本里写一份副本
#: （写副本时「脚本与服务端密钥不一致」会表现成 401，而不是配置错误）。
SECRET = os.environ.get("PET_AGENT_AUTH_SECRET", "")
SESSION = "s-smoke"
TRACE = "11111111-2222-3333-4444-555555555555"
MEOW_URL = "http://127.0.0.1:8291/meow.wav"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"{mark}  {name}" + (f"   — {detail}" if detail else ""))


def _is_uuid(value: str) -> bool:
    import uuid

    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def main() -> int:
    if not SECRET:
        print("FATAL: 未设置 PET_AGENT_AUTH_SECRET（需与服务端一致）")
        return 2
    today = datetime.now(timezone.utc).date().isoformat()

    with httpx.Client(base_url=BASE, timeout=60.0) as c:
        token = issue_token("u-smoke", secret=SECRET)
        h = {
            "Authorization": f"Bearer {token}",
            "X-Session-Id": SESSION,
            "X-Trace-Id": TRACE,
        }

        # ── 探针与作用域 ──
        r = c.get("/healthz", headers=h)
        j = r.json()
        check(
            "GET /healthz",
            r.status_code == 200 and j.get("status") == "ok",
            f"provider={j['providers']['mode']} langsmith={j['observability']['langsmith_enabled']}",
        )
        check(
            "响应头回显客户端 X-Session-Id",
            r.headers.get("X-Session-Id") == SESSION,
            r.headers.get("X-Session-Id", ""),
        )
        check(
            "响应头回显客户端 X-Trace-Id",
            r.headers.get("X-Trace-Id") == TRACE,
            r.headers.get("X-Trace-Id", ""),
        )

        # 不带 scope 的请求必须由服务端生成（而不是漏掉）
        bare = c.get("/healthz")
        gen_session = bare.headers.get("X-Session-Id", "")
        gen_trace = bare.headers.get("X-Trace-Id", "")
        check(
            "未带 scope 时服务端生成（sess- 前缀 + UUID trace）",
            gen_session.startswith("sess-") and _is_uuid(gen_trace),
            f"{gen_session} / {gen_trace}",
        )

        # ── 建宠物 + 档案 ──
        r = c.post("/v1/pets", json={"name": "团团", "breed": "中华田园猫"}, headers=h)
        check("POST /v1/pets", r.status_code == 201, r.text[:120])
        if r.status_code != 201:
            return _summary()
        pet = r.json()["pet_id"]

        r = c.post(
            f"/v1/pets/{pet}/profile",
            json={"image_urls": ["http://x/a.jpg", "http://x/b.jpg"], "confirm": True},
            headers=h,
        )
        check("POST profile（confirm=true 落库）", r.status_code == 200, r.text[:160])

        r = c.get(f"/v1/pets/{pet}/profile", headers=h)
        check(
            "GET profile",
            r.status_code == 200,
            f"must_keep={r.json().get('must_keep_features')}",
        )

        # ── 对话 + 会话记忆注入 ──
        r = c.post(
            f"/v1/chat?pet_id={pet}",
            json={"text": "它今天老在门口叫"},
            headers=h,
        )
        j1 = r.json()
        check(
            "POST /v1/chat（turn 1）",
            r.status_code == 200,
            f"intent={j1.get('intent')} session={j1.get('session_id')}",
        )
        check("body 回传 trace_id", j1.get("trace_id") == TRACE, j1.get("trace_id", ""))

        r = c.post(
            f"/v1/chat?pet_id={pet}",
            json={"text": "它今天怎么样"},
            headers=h,
        )
        j2 = r.json()
        hist = [
            t["decision"] for t in j2.get("trace", []) if t["node"] == "load_context"
        ]
        inj = [
            t["decision"] for t in j2.get("trace", []) if t["node"] == "companion_agent"
        ]
        check(
            "会话历史注入（turn 2）",
            any("历史" in d and "0 条" not in d for d in hist),
            f"{hist} / {inj}",
        )

        # ── 行为解释（真实音频 + 真实特征提取）──
        r = c.post(
            f"/v1/interpret?pet_id={pet}",
            json={"audio_url": MEOW_URL, "scene_description": "在门口叫"},
            headers=h,
        )
        j3 = r.json()
        itp = j3.get("interpretation_id")
        mode = (j3.get("interpretation") or {}).get("evidence_mode")
        check("POST /v1/interpret", r.status_code == 200 and bool(itp), f"mode={mode}")

        # ── 标注（案例推理的燃料）──
        r = c.post(
            f"/v1/pets/{pet}/meow-records",
            json={
                "interpretation_id": itp,
                "context": "door_attention",
                "actions": ["scratch_door"],
                "resolution": "开门它就出去了",
            },
            headers=h,
        )
        check("POST meow-records（主人标注）", r.status_code == 201, r.text[:160])

        r = c.get(
            f"/v1/pets/{pet}/meow-records",
            params={"session_id": SESSION},
            headers=h,
        )
        check(
            "GET meow-records?session_id=",
            r.status_code == 200 and r.json().get("count") == 1,
            f"count={r.json().get('count')}",
        )

        # ── 健康：结构化信号 → 红旗 ──
        r = c.post(
            f"/v1/pets/{pet}/health/signals",
            json={"signal": "litter_box.urine_output", "value": "none"},
            headers=h,
        )
        check("POST health/signals", r.status_code == 201, r.text[:160])

        r = c.get(f"/v1/pets/{pet}/health", headers=h)
        jh = r.json()
        flags = [f["rule_id"] for f in jh.get("red_flags", [])]
        check(
            "GET health（尿闭红旗 → L3 急诊级）",
            r.status_code == 200
            and jh.get("level") == "L3"
            and "urinary_obstruction" in flags,
            f"level={jh.get('level')} flags={flags}",
        )

        # ── 日报 / 记忆 ──
        r = c.get(f"/v1/pets/{pet}/story", params={"day": today}, headers=h)
        check(
            "GET story?day=",
            r.status_code == 200,
            f"title={(r.json().get('title') or '')[:30]}",
        )

        r = c.get("/v1/memories", params={"pet_id": pet}, headers=h)
        check(
            "GET /v1/memories", r.status_code == 200, f"count={r.json().get('count')}"
        )

        # ── 隔离与鉴权 ──
        r = c.get("/v1/memories", params={"pet_id": pet})
        check("无 token → 401", r.status_code == 401, str(r.status_code))

        other = {"Authorization": f"Bearer {issue_token('u-other', secret=SECRET)}"}
        r = c.get(f"/v1/pets/{pet}/profile", headers=other)
        check("他人 token 访问本人宠物 → 404", r.status_code == 404, str(r.status_code))

    return _summary()


def _summary() -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    print()
    print(f"== {passed}/{len(results)} passed ==")
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
