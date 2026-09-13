import { useState } from 'react'
import type { AcousticFeatures, BehaviorInterpretation, MeowRecord } from '../api/types'
import { EvidenceBadge, ModeBadge, OwnerMark, CautionStamp } from './EvidenceBadge'
import { IconChevron, IconWave } from './icons'
import { ACTION_LABELS, CONTEXT_LABELS, cx } from '../lib/utils'

/** 展示哪些测量值、怎么显示。**每一项都是代码可重算的量**。 */
const MEASURED_ROWS: Array<{
  key: keyof AcousticFeatures
  label: string
  unit: string
  decimals: number
}> = [
  { key: 'f0_mean', label: '基频均值', unit: 'Hz', decimals: 0 },
  { key: 'f0_slope', label: '基频走向', unit: '', decimals: 3 },
  { key: 'f0_range', label: '基频跨度', unit: 'Hz', decimals: 0 },
  { key: 'duration', label: '时长', unit: 's', decimals: 2 },
  { key: 'roughness', label: '粗糙度', unit: '', decimals: 3 },
]

/**
 * 行为解释的证据卡。
 *
 * ## 这一屏在做什么
 *
 * 它把「我们凭什么这么说」摊开给用户看。项目的诚实性主张是
 * 「置信度由检索/评分层计算，LLM 不产生数字」（D7），
 * 而用户判断该信多少的唯一依据就是这张卡上的分层：
 *
 * - **测量值**（实线 / 苔绿）：代码从音频算出来的，可重算 → 敢给小数点
 * - **观察**（虚线 / 赭黄）：模型对画面的描述，未校验 → 明确标注它不是测量
 * - **先例**（检索）：这只猫自己的历史标注 → 用户能自己核查对错
 * - **仅文字**模式：不给任何数值 → 这是刻意的，不是缺数据
 *
 * 一个细节：`text_only` 时**不渲染候选概率**，因为那会让「没测量」
 * 看起来像「有测量但概率低」。这两件事对用户的意义完全不同。
 */
export function EvidenceCard({
  interp,
  label,
  onLabel,
}: {
  interp: BehaviorInterpretation
  label?: MeowRecord | undefined
  onLabel?: (() => void) | undefined
}) {
  const [openEvidence, setOpenEvidence] = useState(false)
  const [openFeatures, setOpenFeatures] = useState(false)

  const feats = interp.acoustic_features
  const unavailable = new Set(feats?.unavailable ?? [])
  const hasMeasurement = interp.evidence_mode !== 'text_only'
  const canLabel = Boolean(onLabel)

  return (
    <div className="card overflow-hidden animate-rise-in">
      {/* 头部：模式 + 个体化程度 */}
      <div className="flex flex-wrap items-center gap-2 border-b border-rule bg-paper-sunk/45 px-3.5 py-2.5">
        <IconWave width={15} height={15} className="text-ink-faint" />
        <ModeBadge mode={interp.evidence_mode} />
        {interp.case_total > 0 && (
          <span className="font-mono text-[10.5px] text-ink-faint">
            历史样本 {interp.case_total}
          </span>
        )}
        {interp.sample_count > 0 && (
          <span className="font-mono text-[10.5px] text-ink-faint">
            本次参与 {interp.sample_count}
          </span>
        )}
      </div>

      {/* 候选 */}
      <div className="px-3.5 py-3">
        {interp.candidates.length === 0 ? (
          <p className="text-[13px] leading-relaxed text-ink-faint">
            没有得出可陈述的解释。
          </p>
        ) : interp.evidence_mode === 'text_only' ? (
          // 仅文字：**不给数值** —— 没有测量就不该有概率
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[14px] font-semibold text-ink">
              {CONTEXT_LABELS[interp.candidates[0].context] ?? interp.candidates[0].display}
            </span>
            <CautionStamp>无测量依据</CautionStamp>
          </div>
        ) : (
          <ul className="space-y-2.5">
            {interp.candidates.slice(0, 3).map((c, i) => (
              <li key={c.context}>
                <div className="mb-1 flex items-baseline justify-between gap-3">
                  <span
                    className={cx(
                      'text-[13.5px]',
                      i === 0 ? 'font-semibold text-ink' : 'text-ink-soft',
                    )}
                  >
                    {CONTEXT_LABELS[c.context] ?? c.display}
                  </span>
                  {c.posterior !== null ? (
                    <span className="shrink-0 font-mono text-[12px] tabular-nums text-ink-soft">
                      {(c.posterior * 100).toFixed(0)}%
                    </span>
                  ) : (
                    <span className="shrink-0 text-[11px] italic text-ink-faint">
                      不足以给数
                    </span>
                  )}
                </div>
                {c.posterior !== null && (
                  <div className="h-[3px] w-full overflow-hidden rounded-full bg-paper-sunk">
                    <div
                      className={cx(
                        'h-full rounded-full transition-[width] duration-700',
                        i === 0 ? 'bg-persimmon/75' : 'bg-ink-faint/40',
                      )}
                      style={{ width: `${Math.max(2, c.posterior * 100)}%` }}
                    />
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}

        {/* 个体化程度 */}
        {hasMeasurement && interp.individualization > 0 && (
          <p className="mt-3 text-[11.5px] leading-relaxed text-ink-faint">
            个体化程度 {Math.round(interp.individualization * 100)}% ——
            这只猫自己的样本把群体先验往它的习惯上拉了这么多。
          </p>
        )}

        {/* 限制声明：**永远显示**，不折叠 */}
        {interp.limitations && (
          <div className="mt-3 rounded-[2px] border border-l-[3px] border-rule border-l-ochre/60 bg-ochre/[0.05] px-3 py-2.5">
            <p className="text-[11.5px] leading-relaxed text-ink-soft">{interp.limitations}</p>
          </div>
        )}
      </div>

      {/* 测量值（可展开） */}
      {feats && (
        <div className="border-t border-rule">
          <button
            onClick={() => setOpenFeatures((v) => !v)}
            className="flex w-full items-center justify-between px-3.5 py-2.5 text-left transition hover:bg-paper-sunk/40"
          >
            <span className="flex items-center gap-2">
              <EvidenceBadge kind="measured" />
              <span className="text-[12px] text-ink-soft">本次测到的数字</span>
            </span>
            <IconChevron
              width={15}
              height={15}
              className={cx('text-ink-faint transition-transform', openFeatures && 'rotate-90')}
            />
          </button>

          {openFeatures && (
            <div className="animate-rise-in border-t border-dashed border-rule bg-paper-sunk/25 px-3.5 py-3">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                {MEASURED_ROWS.map(({ key, label, unit, decimals }) => {
                  const missing = unavailable.has(key)
                  const value = feats[key]
                  const numeric = typeof value === 'number'
                  return (
                    <div key={key} className="flex items-baseline justify-between gap-2">
                      <dt className="text-[11.5px] text-ink-faint">{label}</dt>
                      <dd
                        className={cx(
                          'font-mono text-[12px] tabular-nums',
                          missing || !numeric ? 'text-ink-faint line-through' : 'text-ink',
                        )}
                      >
                        {missing || !numeric ? '测不出' : `${value.toFixed(decimals)}${unit}`}
                      </dd>
                    </div>
                  )
                })}
              </dl>
              {unavailable.size > 0 && (
                <p className="mt-2.5 text-[11px] leading-relaxed text-ink-faint">
                  划掉的是本次**测不出**的特征。它们被跳过，而不是当成 0 ——
                  把测不出的填成 0 是静默编造。
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {/* 证据链（可展开） */}
      {interp.evidence.length > 0 && (
        <div className="border-t border-rule">
          <button
            onClick={() => setOpenEvidence((v) => !v)}
            className="flex w-full items-center justify-between px-3.5 py-2.5 text-left transition hover:bg-paper-sunk/40"
          >
            <span className="text-[12px] text-ink-soft">
              证据链 <span className="font-mono text-ink-faint">{interp.evidence.length}</span>
            </span>
            <IconChevron
              width={15}
              height={15}
              className={cx('text-ink-faint transition-transform', openEvidence && 'rotate-90')}
            />
          </button>

          {openEvidence && (
            <ul className="animate-rise-in space-y-2.5 border-t border-dashed border-rule bg-paper-sunk/25 px-3.5 py-3">
              {interp.evidence.map((ev, i) => (
                <li key={i} className="flex gap-2.5">
                  <EvidenceBadge kind={ev.kind} className="mt-px shrink-0" />
                  <div className="min-w-0 flex-1">
                    <p className="text-[12.5px] leading-relaxed text-ink-soft">{ev.statement}</p>
                    <p className="mt-0.5 font-mono text-[10.5px] text-ink-faint">{ev.source}</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* 先例 */}
      {interp.similar_cases.length > 0 && (
        <div className="border-t border-rule px-3.5 py-3">
          <p className="mb-2 text-[12px] text-ink-soft">
            这只猫以前这样叫时
            <span className="ml-1.5 font-mono text-[10.5px] text-ink-faint">
              {interp.similar_cases.length} 条
            </span>
          </p>
          <ul className="space-y-2">
            {interp.similar_cases.slice(0, 3).map((c) => (
              <li
                key={c.record_id}
                className="rounded-[2px] border border-rule bg-paper px-2.5 py-2"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[12.5px] font-semibold text-ink">
                    {CONTEXT_LABELS[c.context] ?? c.context}
                  </span>
                  <span className="shrink-0 font-mono text-[10.5px] text-ink-faint">
                    相似 {Math.round(c.similarity * 100)}%
                  </span>
                </div>
                {c.actions.length > 0 && (
                  <p className="mt-1 text-[11.5px] leading-relaxed text-ink-soft">
                    {c.actions.map((a) => ACTION_LABELS[a] ?? a).join(' · ')}
                  </p>
                )}
                {c.resolution && (
                  <p className="mt-0.5 text-[11.5px] leading-relaxed text-ink-faint">
                    后来：{c.resolution}
                  </p>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 建议观察项 */}
      {interp.suggested_observation && interp.evidence_mode === 'text_only' && (
        <div className="border-t border-rule px-3.5 py-3">
          <p className="mb-1 text-[11px] font-bold uppercase tracking-[0.1em] text-ink-faint">
            可以补充
          </p>
          <p className="text-[12.5px] leading-relaxed text-ink-soft">
            {interp.suggested_observation}
          </p>
        </div>
      )}

      {/* 标注区：闭环入口 */}
      <div className="border-t border-rule bg-paper-sunk/40 px-3.5 py-3">
        {label ? (
          <div className="flex flex-wrap items-center gap-2">
            <OwnerMark>你已标注</OwnerMark>
            <span className="text-[12.5px] text-ink-soft">
              {CONTEXT_LABELS[label.context] ?? label.context}
              {label.actions.length > 0 &&
                ` · ${label.actions.map((a) => ACTION_LABELS[a] ?? a).join(' · ')}`}
            </span>
          </div>
        ) : canLabel ? (
          <>
            <button
              onClick={onLabel}
              className="w-full rounded-[3px] border border-persimmon/45 bg-persimmon/[0.06] px-3 py-2.5 text-[12.5px] font-semibold text-persimmon-deep transition hover:bg-persimmon/[0.11]"
            >
              这只猫当时是什么情况？→ 告诉它
            </button>
            <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
              你的标注是它学习这只猫的唯一途径。只接受你能**观察到**的（动作、情境），
              不问你「它想干什么」—— 那个没法验证。
            </p>
          </>
        ) : (
          <p className="text-[11.5px] leading-relaxed text-ink-faint">
            本次没有测到声学特征，所以没有可标注的东西 ——
            标注需要真实测量做锚点。
          </p>
        )}
      </div>
    </div>
  )
}
