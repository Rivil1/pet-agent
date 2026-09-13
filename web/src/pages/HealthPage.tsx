import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { HealthResponse } from '../api/types'
import { Button, Card, Field, Input, Notice, PageTitle, SectionTitle, Spinner } from '../components/ui'
import { EvidenceBadge, CautionStamp } from '../components/EvidenceBadge'
import { IconRefresh } from '../components/icons'
import { HEALTH_LEVELS, cx, stagger } from '../lib/utils'

/** 可记录的结构化信号。**只收能填数字的** —— 自由文本不可靠（决策 D39）。 */
const SIGNALS: Array<{ key: string; label: string; unit: string; placeholder: string }> = [
  { key: 'appetite', label: '食欲', unit: '（0–3）', placeholder: '0 不吃 · 3 正常' },
  { key: 'water_intake', label: '饮水量', unit: 'ml/天', placeholder: '例如 200' },
  { key: 'urination_count', label: '排尿次数', unit: '次/天', placeholder: '例如 3' },
  { key: 'vomiting_count', label: '呕吐次数', unit: '次/天', placeholder: '例如 0' },
  { key: 'activity_level', label: '活跃度', unit: '（0–3）', placeholder: '0 嗜睡 · 3 正常' },
  { key: 'weight_kg', label: '体重', unit: 'kg', placeholder: '例如 4.2' },
]

/**
 * 健康页。
 *
 * ## 这一屏的每一个设计选择都来自「假阴性代价是猫可能死亡」
 *
 * 1. **覆盖率不足时显示「无法评估」，而不是「未发现异常」。**
 *    把「没数据」显示成「没问题」是这套系统里最危险的一种 UI 缺陷
 *    （决策 D12）。所以 `INSUFFICIENT_DATA` 必须是独立的、显眼的第三态 ——
 *    既不是「好」也不是「坏」。
 * 2. **免责声明与「不得读作」常驻**，不折叠。这是合规要求，不是文案。
 * 3. **只收结构化数字。** 对话里提到的健康信息一律 `value=None` 进
 *    `signals_missing` —— 解析中文数词不可靠，而错了会误触红旗（D39）。
 */
export function HealthPage({ petId, petName }: { petId: string; petName: string }) {
  const [data, setData] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [openForm, setOpenForm] = useState(false)

  const [signal, setSignal] = useState(SIGNALS[0].key)
  const [value, setValue] = useState('')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [saved, setSaved] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await api.health(petId))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '读取失败')
    } finally {
      setLoading(false)
    }
  }, [petId])

  useEffect(() => {
    void load()
  }, [load])

  async function submit() {
    const num = Number(value)
    if (!value.trim() || Number.isNaN(num)) return
    setBusy(true)
    setError(null)
    try {
      await api.recordSignal(petId, {
        signal,
        value: num,
        ...(note.trim() ? { note: note.trim() } : {}),
        consent_version: 'v1',
      })
      setValue('')
      setNote('')
      setSaved(true)
      setOpenForm(false)
      await load()
      window.setTimeout(() => setSaved(false), 2400)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '记录失败')
    } finally {
      setBusy(false)
    }
  }

  const level = data
    ? (HEALTH_LEVELS[data.level] ?? {
        // 未知分级：**原样展示并标注**，而不是当成「没结果」。
        // 后端加了一个新等级而前端忘了同步时，
        // 「没拿到结果」会把一个显眼的契约渗漏伪装成一次网络小抖动。
        label: data.level,
        tone: 'muted' as const,
        hint: '前端还不认识这个分级 —— 请按原值理解，或更新前端',
      })
    : null
  const toneClass = level
    ? {
        danger: 'border-brick/55 bg-brick/[0.07]',
        warn: 'border-ochre/55 bg-ochre/[0.07]',
        ok: 'border-moss/55 bg-moss/[0.07]',
        muted: 'border-ink-faint/45 bg-paper-sunk',
      }[level.tone]
    : ''

  const coveragePct = data ? Math.round(data.coverage * 100) : 0
  const insufficient = data?.level === 'INSUFFICIENT_DATA'

  return (
    <div>
      <PageTitle sub="监测与分诊，不是诊断。它不会告诉你「没事」。">
        {petName} 的健康
      </PageTitle>

      {error && (
        <div className="mb-4">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2.5 py-14 text-ink-faint">
          <Spinner />
          <span className="text-[13px]">评估中…</span>
        </div>
      ) : data && level ? (        <>
          {/* 分级 —— 三值逻辑的第三态必须显眼 */}
          <Card className={cx('mb-5 animate-rise-in border-l-[4px]', toneClass)}>
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="font-display text-[21px] font-semibold text-ink">
                {level.label}
              </span>
              {insufficient && <CautionStamp>不等于没问题</CautionStamp>}
            </div>
            <p className="mt-1 text-[12px] text-ink-faint">{level.hint}</p>

            {data.recommendation && (
              <p className="mt-3 border-t border-dashed border-rule pt-3 text-[13.5px] leading-relaxed text-ink">
                {data.recommendation}
              </p>
            )}

            {/* 覆盖率 */}
            <div className="mt-3.5 border-t border-dashed border-rule pt-3">
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <span className="text-[12px] text-ink-faint">数据覆盖度</span>
                <span className="font-mono text-[12.5px] tabular-nums text-ink">
                  {coveragePct}%
                </span>
              </div>
              <div className="h-[3px] overflow-hidden rounded-full bg-paper-sunk">
                <div
                  className={cx(
                    'h-full transition-[width] duration-700',
                    insufficient ? 'bg-ink-faint/55' : 'bg-moss/70',
                  )}
                  style={{ width: `${Math.max(2, coveragePct)}%` }}
                />
              </div>
              {data.coverage_note && (
                <p className="mt-2 text-[11.5px] leading-relaxed text-ink-faint">
                  {data.coverage_note}
                </p>
              )}
            </div>
          </Card>

          {/* 命中红旗 */}
          {data.red_flags.length > 0 && (
            <>
              <SectionTitle>命中的规则</SectionTitle>
              <ul className="mb-5 space-y-2">
                {data.red_flags.map((f, i) => (
                  <li
                    key={i}
                    style={{ animationDelay: stagger(i, 50) }}
                    className="card animate-rise-in border-l-[3px] border-l-brick/60 p-3.5"
                  >
                    <p className="text-[13px] leading-relaxed text-ink">
                      {String(f.message ?? f.signal ?? JSON.stringify(f))}
                    </p>
                    {f.signal && (
                      <p className="mt-1 font-mono text-[10.5px] text-ink-faint">
                        {String(f.signal)}
                        {f.id ? ` · ${String(f.id)}` : ''}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}

          {/* 缺口 */}
          {data.signals_missing_count > 0 && (
            <div className="mb-5">
              <Notice tone="warn" title={`还有 ${data.signals_missing_count} 项没有数据`}>
                缺数据不会被当作「正常」。这些项记为「不知道」，
                所以整体结论停在「{level.label}」。
              </Notice>
            </div>
          )}

          <SectionTitle>记录信号</SectionTitle>
          {saved && (
            <div className="mb-3 animate-rise-in">
              <Notice tone="ok">已记录。评估已重新计算。</Notice>
            </div>
          )}

          <Button variant={openForm ? 'quiet' : 'primary'} onClick={() => setOpenForm((v) => !v)}>
            {openForm ? '取消' : '记录一项'}
          </Button>

          {openForm && (
            <Card tape className="mt-4 animate-rise-in p-4">
              <div className="space-y-3">
                <Field label="项目">
                  <select
                    value={signal}
                    onChange={(e) => setSignal(e.target.value)}
                    className="field"
                  >
                    {SIGNALS.map((s) => (
                      <option key={s.key} value={s.key}>
                        {s.label} {s.unit}
                      </option>
                    ))}
                  </select>
                </Field>

                <Field
                  label="数值"
                  hint="只收数字。对话里提到的健康信息不会被当成数值 —— 那不可靠，而错了会误触红旗。"
                >
                  <Input
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && submit()}
                    placeholder={SIGNALS.find((s) => s.key === signal)?.placeholder}
                    inputMode="decimal"
                  />
                </Field>

                <Field label="备注（可选）">
                  <Input
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="例如 早上量的"
                  />
                </Field>

                <Button
                  loading={busy}
                  disabled={!value.trim() || Number.isNaN(Number(value))}
                  onClick={submit}
                >
                  记录
                </Button>
              </div>
            </Card>
          )}

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <Button variant="quiet" onClick={load}>
              <IconRefresh width={15} height={15} />
              重新评估
            </Button>
            {data.rule_version && (
              <span className="font-mono text-[10.5px] text-ink-faint">
                规则版本 {data.rule_version}
              </span>
            )}
            <span className="font-mono text-[10.5px] text-ink-faint">
              {data.record_count} 条记录
            </span>
          </div>

          {/* 合规声明：常驻，不折叠 */}
          <div className="mt-5 space-y-2.5">
            <div className="rounded-[3px] border border-l-[3px] border-rule border-l-ink-faint/45 bg-paper-sunk/45 px-3.5 py-3">
              <div className="mb-1.5 flex items-center gap-2">
                <EvidenceBadge kind="prior" />
                <span className="text-[11px] font-bold uppercase tracking-[0.1em] text-ink-soft">
                  不得读作
                </span>
              </div>
              <p className="text-[12px] leading-relaxed text-ink-soft">
                {data.must_not_be_read_as}
              </p>
            </div>

            <p className="rounded-[3px] border border-l-[3px] border-rule border-l-ochre/55 bg-ochre/[0.05] px-3.5 py-3 text-[11.5px] leading-relaxed text-ink-soft">
              {data.disclaimer}
            </p>
          </div>
        </>
      ) : (
        <Notice tone="warn">没有拿到评估结果。</Notice>
      )}
    </div>
  )
}
