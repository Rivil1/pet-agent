import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { StoryResponse } from '../api/types'
import { Button, Card, Empty, Notice, PageTitle, SectionTitle, Spinner } from '../components/ui'
import { EvidenceBadge, CautionStamp } from '../components/EvidenceBadge'
import { IconChevron, IconRefresh } from '../components/icons'
import { cx, formatDayShort, shiftDay, stagger, todayISO } from '../lib/utils'

const TIME_LABELS: Record<string, string> = {
  morning: '早上',
  noon: '中午',
  afternoon: '下午',
  evening: '傍晚',
  night: '晚上',
  unknown: '某时',
}

/**
 * 「宠物的一天」。
 *
 * ## 这一屏的边界是刻意的
 *
 * 它**不读原始对话**（决策 D34）。日报从已经过 quote 机械校验的候选派生 ——
 * 否则「一句话的合理归纳」会绕过校验，把没发生的事写进故事里。
 *
 * 所以页面上有两个必须常驻的东西：
 * 1. **免责声明** —— 它是娱乐产物，不是观察结论
 * 2. **排除计数** —— 有多少条候选因校验失败被丢掉
 *
 * 第 2 条尤其重要：不显示它，用户会以为「故事完整反映了一天」，
 * 而实际上被丢掉的正是无法核实的那部分。
 */
export function StoryPage({ petId, petName }: { petId: string; petName: string }) {
  const [day, setDay] = useState(todayISO())
  const [story, setStory] = useState<StoryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showNotes, setShowNotes] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setStory(await api.story(petId, day))
    } catch (e) {
      setStory(null)
      setError(e instanceof ApiError ? e.message : '生成失败')
    } finally {
      setLoading(false)
    }
  }, [petId, day])

  useEffect(() => {
    void load()
  }, [load])

  const isToday = day === todayISO()

  return (
    <div>
      <PageTitle sub="从当天已验证的记录里，拼出它的一天。">{petName} 的一天</PageTitle>

      {/* 日期切换 */}
      <div className="mb-5 flex items-center justify-between gap-2 rounded-[3px] border border-rule bg-paper-raised px-2 py-1.5">
        <button
          onClick={() => setDay((d) => shiftDay(d, -1))}
          className="rounded-[2px] px-2 py-1.5 text-ink-faint transition hover:text-ink"
          aria-label="前一天"
        >
          <IconChevron width={17} height={17} className="rotate-180" />
        </button>

        <button onClick={() => setDay(todayISO())} className="flex flex-col items-center px-2">
          <span className="font-display text-[16px] font-semibold text-ink">
            {formatDayShort(day)}
          </span>
          <span className="text-[10.5px] text-ink-faint">
            {isToday ? '今天' : day === shiftDay(todayISO(), -1) ? '昨天' : day}
          </span>
        </button>

        <button
          onClick={() => setDay((d) => shiftDay(d, 1))}
          disabled={isToday}
          className="rounded-[2px] px-2 py-1.5 text-ink-faint transition hover:text-ink disabled:opacity-25"
          aria-label="后一天"
        >
          <IconChevron width={17} height={17} />
        </button>
      </div>

      {error && (
        <div className="mb-4">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {loading ? (
        <div className="flex items-center gap-2.5 py-14 text-ink-faint">
          <Spinner />
          <span className="text-[13px]">翻当天的记录…</span>
        </div>
      ) : !story || story.beats.length === 0 ? (
        <Empty
          title={isToday ? '今天还没有可写的记录' : `${formatDayShort(day)} 没有记录`}
          hint="先去对话里说说这一天发生了什么。日报只从经过校验的记录派生 —— 没记录就不会编。"
          action={
            <Button variant="ghost" onClick={load}>
              <IconRefresh width={15} height={15} />
              重新生成
            </Button>
          }
        />
      ) : (
        <>
          {/* 健康提示（若有） */}
          {story.health_notice && (
            <div className="mb-4">
              <Notice tone="warn" title="健康信号">
                {story.health_notice}
              </Notice>
            </div>
          )}

          {/* 标题 */}
          <div className="mb-5 animate-rise-in">
            <h2 className="font-display text-[22px] font-semibold leading-snug tracking-[-0.01em] text-ink">
              {story.title}
            </h2>
            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              <CautionStamp>娱乐产物</CautionStamp>
              <span className="font-mono text-[10.5px] text-ink-faint">
                {story.beats.length} 幕
              </span>
              {story.excluded_count > 0 && (
                <span className="font-mono text-[10.5px] text-ochre">
                  排除 {story.excluded_count} 条
                </span>
              )}
            </div>
          </div>

          {/* 幕 */}
          <ol className="space-y-3.5">
            {story.beats.map((beat, i) => (
              <li key={i} style={{ animationDelay: stagger(i, 70) }} className="animate-rise-in">
                <Card binding className="p-4">
                  <div className="mb-2 flex items-center gap-2">
                    <span className="rounded-[2px] bg-paper-sunk px-1.5 py-px font-mono text-[10px] uppercase tracking-[0.08em] text-ink-soft">
                      {TIME_LABELS[beat.time_of_day] ?? beat.time_of_day}
                    </span>
                    <span className="font-mono text-[10px] text-ink-faint">
                      {String(i + 1).padStart(2, '0')}
                    </span>
                  </div>
                  <p className="text-[14.5px] leading-relaxed text-ink">{beat.caption}</p>
                  {beat.photo_ref && (
                    <p className="mt-2 font-mono text-[10.5px] text-ink-faint">
                      配图：{beat.photo_ref}
                    </p>
                  )}
                </Card>
              </li>
            ))}
          </ol>

          {/* 完整文本 */}
          <SectionTitle className="mt-6">全文</SectionTitle>
          <Card ruled className="ruled p-4">
            <p className="whitespace-pre-wrap text-[13.5px] leading-[1.75rem] text-ink-soft">
              {story.story_text}
            </p>
          </Card>

          {/* 校验说明 */}
          <div className="mt-5 space-y-2.5">
            <div className="flex items-center gap-2">
              <EvidenceBadge kind="retrieved" />
              <span className="text-[11.5px] text-ink-faint">
                每一幕都对应一条经过原文校验的记录
              </span>
            </div>

            <button
              onClick={() => setShowNotes((v) => !v)}
              className="flex w-full items-center gap-2 text-left text-[12px] text-ink-faint transition hover:text-ink-soft"
            >
              <IconChevron
                width={14}
                height={14}
                className={cx('transition-transform', showNotes && 'rotate-90')}
              />
              生成说明与排除情况
            </button>

            {showNotes && (
              <div className="animate-rise-in rounded-[3px] border border-rule bg-paper-sunk/40 px-3.5 py-3">
                <dl className="space-y-1.5">
                  <div className="flex items-baseline justify-between gap-3">
                    <dt className="text-[11.5px] text-ink-faint">排除条数</dt>
                    <dd className="font-mono text-[11.5px] text-ink">{story.excluded_count}</dd>
                  </div>
                  <div className="flex items-baseline justify-between gap-3">
                    <dt className="text-[11.5px] text-ink-faint">校验拒绝率</dt>
                    <dd className="font-mono text-[11.5px] text-ink">
                      {(story.rejection_rate * 100).toFixed(0)}%
                    </dd>
                  </div>
                </dl>

                {story.digest_notes.length > 0 && (
                  <ul className="mt-2.5 space-y-1 border-t border-dashed border-rule pt-2.5">
                    {story.digest_notes.map((n, i) => (
                      <li key={i} className="text-[11.5px] leading-relaxed text-ink-soft">
                        · {n}
                      </li>
                    ))}
                  </ul>
                )}

                <p className="mt-3 border-t border-dashed border-rule pt-2.5 text-[11px] leading-relaxed text-ink-faint">
                  「校验拒绝率」是被丢掉的候选占比 —— 那些是模型归纳出来、
                  但在原文里找不到依据的句子。这个数字越高，说明这一天的记录越零碎。
                </p>
              </div>
            )}
          </div>

          {/* 免责声明：常驻，不折叠 */}
          <p className="mt-5 rounded-[3px] border border-l-[3px] border-rule border-l-ochre/55 bg-ochre/[0.05] px-3.5 py-3 text-[11.5px] leading-relaxed text-ink-soft">
            {story.disclaimer}
          </p>
        </>
      )}
    </div>
  )
}
