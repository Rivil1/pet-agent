import { useState } from 'react'
import type { TurnResponse } from '../api/types'
import { IconChevron } from './icons'
import { cx } from '../lib/utils'

/**
 * 编排轨迹。
 *
 * ## 为什么把节点级 trace 暴露给用户
 *
 * 因为「降级」在这套系统里是一个**正常且重要的状态**：
 * 声学提取失败、多模态未配置、检索为空 —— 每种都改变了结论的强度，
 * 但最终的句子可能看起来一样。
 *
 * 隐藏它，用户就分不清「测了但没命中」与「根本没测成」。
 * 而这个区别正是整个项目「诚实性」的落点（决策 D40 / D45）。
 */
export function TraceStrip({ turn }: { turn: TurnResponse }) {
  const [open, setOpen] = useState(false)
  const degradedCount = turn.trace.filter((t) => t.degraded).length

  return (
    <div className="rounded-[3px] border border-rule bg-paper-sunk/35">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left"
      >
        <span className="text-[11px] uppercase tracking-[0.1em] text-ink-faint">过程</span>
        <span className="font-mono text-[10.5px] text-ink-faint">
          {turn.trace.length} 节点 · {turn.trace.reduce((s, t) => s + t.latency_ms, 0)}ms
        </span>
        {degradedCount > 0 && (
          <span className="rounded-[2px] bg-ochre/15 px-1.5 py-px font-mono text-[10px] text-ochre">
            {degradedCount} 降级
          </span>
        )}
        {turn.provider_mode !== 'live' && (
          <span className="rounded-[2px] bg-brick/12 px-1.5 py-px font-mono text-[10px] text-brick">
            {turn.provider_mode}
          </span>
        )}
        <IconChevron
          width={14}
          height={14}
          className={cx(
            'ml-auto shrink-0 text-ink-faint transition-transform',
            open && 'rotate-90',
          )}
        />
      </button>

      {open && (
        <div className="animate-rise-in border-t border-dashed border-rule px-3 py-2.5">
          <ol className="space-y-1.5">
            {turn.trace.map((t, i) => (
              <li key={i} className="flex items-baseline gap-2.5">
                <span
                  className={cx(
                    'mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full',
                    t.degraded ? 'bg-ochre' : 'bg-moss/65',
                  )}
                  aria-hidden
                />
                <span className="w-[104px] shrink-0 truncate font-mono text-[10.5px] text-ink-faint">
                  {t.node}
                </span>
                <span className="min-w-0 flex-1 text-[11.5px] leading-relaxed text-ink-soft">
                  {t.decision}
                </span>
                <span className="shrink-0 font-mono text-[10px] tabular-nums text-ink-faint">
                  {t.latency_ms}ms
                </span>
              </li>
            ))}
          </ol>

          <dl className="mt-3 space-y-1 border-t border-dashed border-rule pt-2.5">
            <Row label="意图" value={`${turn.intent} (${turn.route_confidence.toFixed(2)})`} />
            <Row label="召回" value={`${turn.retrieved_count} 条`} />
            {turn.written_memory_ids.length > 0 && (
              <Row label="新写入记忆" value={`${turn.written_memory_ids.length} 条`} />
            )}
            <Row label="trace" value={turn.trace_id.slice(0, 18)} />
          </dl>
        </div>
      )}
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="text-[11px] text-ink-faint">{label}</dt>
      <dd className="font-mono text-[10.5px] text-ink-soft">{value}</dd>
    </div>
  )
}
