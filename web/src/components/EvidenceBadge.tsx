import type { EvidenceKind, EvidenceMode } from '../api/types'
import { EVIDENCE_KIND_LABELS, EVIDENCE_MODE_LABELS, cx } from '../lib/utils'

/**
 * 证据分层徽章。
 *
 * ## 为什么它不是一个普通标签
 *
 * 整个系统的诚实性建立在「不同来源的证据不可混同」上：
 *
 * | 层 | 含义 | 能不能重算 |
 * |---|---|---|
 * | `measured` | 代码从音频算出的数字 | ✅ 能 |
 * | `observed` | 模型对画面的**描述** | ❌ 不能，且未校验 |
 * | `retrieved` | 从这只猫的历史里检索到的先例 | ✅ 可核查 |
 * | `prior` | 公开数据集的群体统计 | ⚠️ 不是这只猫 |
 *
 * 把它们渲染成同一种灰底小标签，等于在 UI 上把这个分层抹平 ——
 * 而用户正是靠这个分层判断「我该信多少」。所以每种证据有不同的
 * **边框形态**（实线/虚线/点线/下划线），不看图例也能感到区别。
 */
export function EvidenceBadge({ kind, className }: { kind: EvidenceKind; className?: string }) {
  return (
    <span className={cx(`ev ev-${kind}`, className)} title={EVIDENCE_KIND_HINTS[kind]}>
      {EVIDENCE_KIND_LABELS[kind] ?? kind}
    </span>
  )
}

const EVIDENCE_KIND_HINTS: Record<EvidenceKind, string> = {
  measured: '代码从音频直接量出的数字，可重算',
  observed: '模型对画面的描述 —— 是描述，不是测量，未经校验',
  retrieved: '从这只猫自己的历史标注里检索到的先例',
  prior: '公开数据集的群体统计，不是这只猫',
}

/**
 * 证据模式徽章。
 *
 * `text_only` 与 `measured_only` 必须看起来就不一样 ——
 * 前者**不允许出现任何数值置信度**（没有测量就没有数字），
 * 如果 UI 上它们长得一样，用户会以为「仅文字」也有概率支撑。
 */
export function ModeBadge({ mode }: { mode: EvidenceMode }) {
  const meta = EVIDENCE_MODE_LABELS[mode] ?? { label: mode, hint: '' }
  const isNumeric = mode !== 'text_only'

  return (
    <span
      className={cx(
        'inline-flex items-center gap-1.5 rounded-[2px] border px-2 py-[3px] text-[11px] font-semibold',
        isNumeric
          ? 'border-moss/45 bg-moss/[0.08] text-moss'
          : 'border-ink-faint/40 bg-paper-sunk text-ink-faint',
      )}
      title={meta.hint}
    >
      <span
        className={cx('h-1.5 w-1.5 rounded-full', isNumeric ? 'bg-moss' : 'border border-ink-faint')}
        aria-hidden
      />
      {meta.label}
    </span>
  )
}

/**
 * 「不可当作结论」的印章。用于 `text_only` 与 mock 模式。
 *
 * 刻意做成印章的样子并微微倾斜：它是一个**覆盖在内容上的判断**，
 * 不是内容本身的一部分。
 */
export function CautionStamp({ children }: { children: React.ReactNode }) {
  return <span className="stamp border-brick text-brick">{children}</span>
}

/** 「可核查」印章。 */
export function VerifiedStamp({ children }: { children: React.ReactNode }) {
  return <span className="stamp border-moss text-moss">{children}</span>
}

/** 主人标注标记 —— 手写感，与系统产出的证据区分开。 */
export function OwnerMark({ children }: { children: React.ReactNode }) {
  return <span className="ev ev-owner">{children}</span>
}
