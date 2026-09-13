import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from 'react'
import { cx } from '../lib/utils'

// ─────────────────────────────────────────────────────────────
// 按钮
// ─────────────────────────────────────────────────────────────

type ButtonVariant = 'primary' | 'ghost' | 'quiet'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
  loading?: boolean
}

export function Button({
  variant = 'primary',
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cx(
        variant === 'primary' && 'btn-primary',
        variant === 'ghost' && 'btn-ghost',
        variant === 'quiet' && 'btn-quiet',
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading && <Spinner />}
      {children}
    </button>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        'inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-[1.5px] border-current border-r-transparent',
        className,
      )}
      aria-hidden
    />
  )
}

// ─────────────────────────────────────────────────────────────
// 卡片
// ─────────────────────────────────────────────────────────────

interface CardProps {
  children: ReactNode
  className?: string
  /** 左侧装订线 —— 用于「一条记录」这类可归属的内容。 */
  binding?: boolean
  ruled?: boolean
  tape?: boolean
}

export function Card({ children, className, binding, ruled, tape }: CardProps) {
  return (
    <div
      className={cx(
        'card relative p-4',
        binding && 'binding pl-5',
        ruled && 'ruled',
        tape && 'tape mt-3',
        className,
      )}
    >
      {children}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// 表单
// ─────────────────────────────────────────────────────────────

interface FieldProps {
  label?: string
  hint?: string
  error?: string | null
  children: ReactNode
  className?: string
}

export function Field({ label, hint, error, children, className }: FieldProps) {
  return (
    <label className={cx('block', className)}>
      {label && (
        <span className="mb-1.5 block text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-faint">
          {label}
        </span>
      )}
      {children}
      {error ? (
        <span className="mt-1.5 block text-[12px] text-brick">{error}</span>
      ) : hint ? (
        <span className="mt-1.5 block text-[12px] leading-relaxed text-ink-faint">{hint}</span>
      ) : null}
    </label>
  )
}

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cx('field', className)} {...rest} />
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cx('field resize-none', className)} {...rest} />
}

// ─────────────────────────────────────────────────────────────
// 空态 / 分区标题
// ─────────────────────────────────────────────────────────────

export function Empty({
  title,
  hint,
  action,
}: {
  title: string
  hint?: string
  action?: ReactNode
}) {
  return (
    <div className="animate-rise-in px-6 py-14 text-center">
      <div className="mx-auto mb-4 h-10 w-10 rotate-[-4deg] rounded-[2px] border-[1.5px] border-dashed border-rule" />
      <p className="font-display text-[17px] font-semibold text-ink">{title}</p>
      {hint && (
        <p className="mx-auto mt-2 max-w-[30ch] text-[13px] leading-relaxed text-ink-faint">{hint}</p>
      )}
      {action && <div className="mt-5 flex justify-center">{action}</div>}
    </div>
  )
}

export function SectionTitle({
  children,
  right,
  className,
}: {
  children: ReactNode
  right?: ReactNode
  className?: string
}) {
  return (
    <div className={cx('mb-2.5 flex items-baseline justify-between gap-3', className)}>
      <h2 className="font-display text-[13px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {children}
      </h2>
      {right}
    </div>
  )
}

/** 手写笔触的一级标题 —— 用在页面顶部。 */
export function PageTitle({ children, sub }: { children: ReactNode; sub?: string }) {
  return (
    <header className="mb-5">
      <h1 className="font-display text-[26px] font-semibold leading-tight tracking-[-0.01em] text-ink">
        {children}
      </h1>
      {sub && <p className="mt-1.5 text-[13px] leading-relaxed text-ink-faint">{sub}</p>}
      <div className="mt-3 h-px origin-left animate-draw-line bg-rule" />
    </header>
  )
}

// ─────────────────────────────────────────────────────────────
// 提示条
// ─────────────────────────────────────────────────────────────

export function Notice({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'danger' | 'ok'
  title?: string
  children: ReactNode
}) {
  const toneClass = {
    info: 'border-rule bg-paper-sunk text-ink-soft',
    warn: 'border-ochre/45 bg-ochre/[0.07] text-ink-soft',
    danger: 'border-brick/50 bg-brick/[0.07] text-ink-soft',
    ok: 'border-moss/45 bg-moss/[0.07] text-ink-soft',
  }[tone]

  return (
    <div className={cx('rounded-[3px] border border-l-[3px] px-3.5 py-3', toneClass)}>
      {title && (
        <p className="mb-1 text-[12px] font-bold uppercase tracking-[0.1em] text-ink">{title}</p>
      )}
      <div className="text-[13px] leading-relaxed">{children}</div>
    </div>
  )
}
