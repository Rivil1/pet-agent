import { useState } from 'react'
import { api, ApiError } from '../api/client'
import type { BehaviorAction, ContextLabel, MeowRecord } from '../api/types'
import { Button, Input, Notice } from './ui'
import { IconCheck, IconClose } from './icons'
import { ACTION_LABELS, CONTEXT_LABELS, cx, stagger } from '../lib/utils'

const CONTEXTS: ContextLabel[] = [
  'food_waiting',
  'door_attention',
  'affection_brushing',
  'isolation_distress',
  'greeting',
  'other',
]

/** 常用动作排在前面 —— 它们覆盖绝大多数场景。 */
const PRIMARY_ACTIONS: BehaviorAction[] = [
  'near_food_bowl',
  'look_at_door',
  'scratch_door',
  'rub_leg',
  'approach_human',
  'pacing',
]

const MORE_ACTIONS: BehaviorAction[] = [
  'tail_up',
  'purring',
  'belly_up',
  'hiding',
  'grooming',
  'arched_back',
  'avoid_contact',
  'other',
]

/**
 * 主人标注表单。
 *
 * ## 两个刻意的设计约束
 *
 * 1. **只问「你能观察到什么」，不问「它想干什么」。**
 *    主人能可靠观察「它在抓门」，无法可靠观察「它想出去」——
 *    人自己判断猫的意图准确率只有 26–40%（三分类随机是 33%）。
 *    用可验证的目标做监督，才谈得上评测（D29）。
 *
 * 2. **不提交任何声学特征。** 特征由服务端按 `interpretation_id`
 *    从自己的存档里取回。若允许客户端提交，`MEASURED` 的承诺立刻失效 ——
 *    客户端可以发任意数字，而案例推理的相似度会建在它们上面。
 *
 * 模型观察到的动作会**预填**（`suggestedActions`），但只是候选：
 * 主人仍然自己选定。这是 D44 —— 观察是描述，不是测量，不改变候选排序。
 */
export function LabelSheet({
  petId,
  interpretationId,
  suggestedActions,
  onClose,
  onLabelled,
}: {
  petId: string
  interpretationId: string
  suggestedActions: string[]
  onClose: () => void
  onLabelled: (rec: MeowRecord) => void
}) {
  const [context, setContext] = useState<ContextLabel | null>(null)
  const [actions, setActions] = useState<BehaviorAction[]>(
    suggestedActions.filter((a): a is BehaviorAction => a in ACTION_LABELS),
  )
  const [showMore, setShowMore] = useState(false)
  const [resolution, setResolution] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  function toggleAction(a: BehaviorAction) {
    setActions((prev) => (prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]))
  }

  async function submit() {
    if (!context) return
    setBusy(true)
    setError(null)
    try {
      const rec = await api.labelMeow(petId, {
        interpretation_id: interpretationId,
        context,
        actions,
        ...(resolution.trim() ? { resolution: resolution.trim() } : {}),
      })
      onLabelled(rec)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '标注失败')
    } finally {
      setBusy(false)
    }
  }

  const allActions = showMore ? [...PRIMARY_ACTIONS, ...MORE_ACTIONS] : PRIMARY_ACTIONS

  return (
    <div className="fixed inset-0 z-40 flex items-end justify-center sm:items-center">
      {/* 遮罩 */}
      <button
        className="absolute inset-0 animate-rise-in bg-ink/25 backdrop-blur-[2px]"
        onClick={onClose}
        aria-label="关闭"
      />

      {/* 抽屉 */}
      <div className="relative z-10 max-h-[86vh] w-full max-w-[560px] animate-rise-in overflow-y-auto rounded-t-[6px] border border-rule bg-paper pb-safe shadow-lift sm:rounded-[4px]">
        <div className="sticky top-0 z-10 flex items-center justify-between border-b border-rule bg-paper/95 px-4 py-3 backdrop-blur-sm">
          <div>
            <p className="font-display text-[16px] font-semibold text-ink">当时是什么情况？</p>
            <p className="mt-0.5 text-[11.5px] text-ink-faint">
              只填你亲眼看到的。你的标注会成为下次判断的依据。
            </p>
          </div>
          <button
            onClick={onClose}
            className="shrink-0 p-1.5 text-ink-faint transition hover:text-brick"
            aria-label="关闭"
          >
            <IconClose width={18} height={18} />
          </button>
        </div>

        <div className="space-y-5 px-4 py-4">
          {error && <Notice tone="danger">{error}</Notice>}

          {/* 情境 */}
          <div>
            <p className="mb-2 text-[12px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
              情境 <span className="text-persimmon">*</span>
            </p>
            <div className="grid grid-cols-2 gap-2">
              {CONTEXTS.map((c, i) => (
                <button
                  key={c}
                  onClick={() => setContext(c)}
                  style={{ animationDelay: stagger(i, 30) }}
                  className={cx(
                    'animate-rise-in rounded-[3px] border px-3 py-2.5 text-left text-[13px] font-medium transition',
                    context === c
                      ? 'border-persimmon/60 bg-persimmon/[0.09] text-persimmon-deep'
                      : 'border-rule bg-paper-raised text-ink-soft hover:border-ink-faint/45',
                  )}
                >
                  {CONTEXT_LABELS[c]}
                </button>
              ))}
            </div>
          </div>

          {/* 动作 */}
          <div>
            <p className="mb-2 text-[12px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
              它当时在做什么
              {suggestedActions.length > 0 && (
                <span className="ml-2 font-normal normal-case tracking-normal text-ochre">
                  已按画面观察预填
                </span>
              )}
            </p>
            <div className="flex flex-wrap gap-2">
              {allActions.map((a) => (
                <button
                  key={a}
                  onClick={() => toggleAction(a)}
                  className={cx(
                    'rounded-pill border px-3 py-1.5 text-[12.5px] transition',
                    actions.includes(a)
                      ? 'border-persimmon/55 bg-persimmon/[0.09] text-persimmon-deep'
                      : 'border-rule bg-paper-raised text-ink-soft hover:border-ink-faint/45',
                  )}
                >
                  {actions.includes(a) && <IconCheck width={12} height={12} className="mr-1 inline" />}
                  {ACTION_LABELS[a]}
                </button>
              ))}
            </div>
            {!showMore && (
              <button
                onClick={() => setShowMore(true)}
                className="mt-2.5 text-[12px] font-semibold text-ink-faint transition hover:text-persimmon"
              >
                更多动作 →
              </button>
            )}
          </div>

          {/* 结果 */}
          <div>
            <p className="mb-2 text-[12px] font-semibold uppercase tracking-[0.1em] text-ink-faint">
              后来什么让它停了
            </p>
            <Input
              value={resolution}
              onChange={(e) => setResolution(e.target.value)}
              placeholder="可选，例如：开了门它就出去了"
            />
            <p className="mt-1.5 text-[11px] leading-relaxed text-ink-faint">
              这一项最有价值 —— 它让「哪种做法有效」变成可检索的先例。
            </p>
          </div>

          <div className="flex gap-2 pb-2">
            <Button
              className="flex-1"
              loading={busy}
              disabled={!context}
              onClick={submit}
            >
              记下来
            </Button>
            <Button variant="quiet" onClick={onClose}>
              取消
            </Button>
          </div>

          <p className="text-[11px] leading-relaxed text-ink-faint">
            提交时**不发送声学特征** —— 服务端按这次解释的 ID 取回它自己存的。
            否则「可测量」就变成「客户端说了算」。
          </p>
        </div>
      </div>
    </div>
  )
}
