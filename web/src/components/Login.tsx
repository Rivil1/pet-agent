import { useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { HealthzResponse } from '../api/types'
import { Button, Field, Input, Notice } from './ui'
import { IconWave } from './icons'

/**
 * 登录屏。
 *
 * ## 两条路径，且**第一条是正路**
 *
 * 1. 粘贴 token —— 由 `python -m app.bootstrap --issue-token <user>` 签发。
 *    这是生产路径，也永远是兜底路径。
 * 2. 开发登录（`PET_AGENT_ALLOW_DEV_LOGIN=1` 时后端才提供）。
 *
 * 第 2 条**只在 `/healthz` 报告 `enabled` 时才显示**。
 * 无条件显示一个默认会 404 的按钮，会让「配置没生效」表现成「功能坏了」。
 */
export function Login({ onDone }: { onDone: (token: string) => void }) {
  const [health, setHealth] = useState<HealthzResponse | null>(null)
  const [healthErr, setHealthErr] = useState<string | null>(null)
  const [token, setTokenInput] = useState('')
  const [userId, setUserId] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    api
      .healthz()
      .then((h) => alive && setHealth(h))
      .catch((e: unknown) => {
        if (!alive) return
        setHealthErr(e instanceof ApiError ? e.message : '无法连接后端')
      })
    return () => {
      alive = false
    }
  }, [])

  const devLoginEnabled = health?.dev_login === 'enabled'
  const providerMode = health?.providers?.mode

  async function handleDevLogin() {
    const id = userId.trim()
    if (!id) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.devLogin(id)
      onDone(res.token)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '登录失败')
    } finally {
      setBusy(false)
    }
  }

  function handleTokenSubmit() {
    const t = token.trim()
    if (!t) return
    onDone(t)
  }

  return (
    <div className="paper-grain paper-vignette relative flex min-h-full flex-col justify-center px-5 py-10">
      <div className="relative z-10 mx-auto w-full max-w-[420px]">
        {/* 封面区 */}
        <div className="animate-rise-in text-center">
          <div className="mx-auto mb-5 flex h-14 w-14 rotate-[-3deg] items-center justify-center rounded-[3px] border-[1.5px] border-persimmon/60 text-persimmon">
            <IconWave width={28} height={28} />
          </div>
          <h1 className="font-display text-[32px] font-semibold leading-none tracking-[-0.02em] text-ink">
            猫事
          </h1>
          <p className="mt-1 font-display text-[13px] italic tracking-wide text-ink-faint">
            A Field Notebook for One Cat
          </p>
          <div className="mx-auto mt-4 h-px w-24 origin-center animate-draw-line bg-rule" />
          <p className="mx-auto mt-4 max-w-[34ch] text-[13px] leading-relaxed text-ink-soft">
            把你和这只猫之间的经验记下来，
            <br />
            在用得上时调出来。
          </p>
        </div>

        {/* 状态 */}
        {healthErr && (
          <div className="mt-7 animate-rise-in">
            <Notice tone="danger" title="后端不可达">
              {healthErr}
            </Notice>
          </div>
        )}

        {health && (
          <div className="mt-6 animate-rise-in text-center">
            <span
              className={`stamp ${
                providerMode === 'live'
                  ? 'border-moss text-moss'
                  : 'border-ochre text-ochre'
              }`}
            >
              {providerMode === 'live' ? '真实模型已接入' : `模型模式：${providerMode ?? '未知'}`}
            </span>
            {providerMode && providerMode !== 'live' && (
              <p className="mt-3 text-[12px] leading-relaxed text-ink-faint">
                当前不是真实模型输出 —— 演示可以，但别当成真实推理。
              </p>
            )}
          </div>
        )}

        {/* 登录卡片 */}
        <div className="card tape mt-9 animate-rise-in p-5 [animation-delay:120ms]">
          {devLoginEnabled ? (
            <>
              <p className="mb-1 font-display text-[15px] font-semibold text-ink">开始记录</p>
              <p className="mb-4 text-[12px] leading-relaxed text-ink-faint">
                开发登录已开启。随便取一个名字，数据只属于这个名字。
              </p>
              <Field label="你的标识">
                <Input
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleDevLogin()}
                  placeholder="例如 cat-mom"
                  autoComplete="username"
                />
              </Field>
              <Button
                className="mt-3 w-full"
                loading={busy}
                disabled={!userId.trim()}
                onClick={handleDevLogin}
              >
                进入
              </Button>
              {error && <p className="mt-3 text-[12px] text-brick">{error}</p>}

              <div className="my-5 flex items-center gap-3">
                <div className="hr-dashed flex-1" />
                <span className="text-[11px] uppercase tracking-[0.14em] text-ink-faint">
                  或
                </span>
                <div className="hr-dashed flex-1" />
              </div>
            </>
          ) : (
            <p className="mb-4 text-[12px] leading-relaxed text-ink-faint">
              粘贴一个访问令牌。用命令行签发：
              <br />
              <code className="mt-1 inline-block rounded-[2px] bg-paper-sunk px-1.5 py-0.5 font-mono text-[11px] text-ink-soft">
                python -m app.bootstrap --issue-token 你的名字
              </code>
            </p>
          )}

          <Field label="访问令牌">
            <Input
              value={token}
              onChange={(e) => setTokenInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleTokenSubmit()}
              placeholder="粘贴 token"
              autoComplete="off"
              spellCheck={false}
            />
          </Field>
          <Button
            variant="ghost"
            className="mt-3 w-full"
            disabled={!token.trim()}
            onClick={handleTokenSubmit}
          >
            使用令牌进入
          </Button>
        </div>

        <p className="mt-6 animate-rise-in text-center text-[11px] leading-relaxed text-ink-faint [animation-delay:200ms]">
          令牌存在本机浏览器，不会上传到别处。
        </p>
      </div>
    </div>
  )
}
