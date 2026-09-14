import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { Moment, RecordMomentResponse, TimelineResponse } from '../api/types'
import { Button, Card, Empty, Notice, Spinner } from '../components/ui'
import { MomentComposer } from '../components/MomentComposer'
import { IconCamera, IconClose, IconPlus } from '../components/icons'
import { cx, stagger, todayISO } from '../lib/utils'

/**
 * 日记：时间线。
 *
 * ## 为什么它是首页
 *
 * `docs/11` §8.1 把「记录动作 + 照片时间线」列为 P0 第一条，理由是
 * 「**无此则产品不成立**」。而它此前**完全没实现** ——
 * 前端是「对话 / 档案 / 一天 / 健康 / 猫」五个工具页，
 * 而想猫的人打开它看到的第一个东西不该是一个查询界面。
 *
 * ## 与「一天」（日报）的分工
 *
 * | | 日记（这一页） | 一天（日报） |
 * |---|---|---|
 * | 来源 | 用户**自己**记的瞬间 | 系统从对话**派生**的叙事 |
 * | 谁说的话 | 用户 + 猫的回应 | 系统写的 |
 * | 可信度 | 用户亲眼所见 | 需过 quote 校验 |
 *
 * 两者都要，但不能混 —— 所以是两个页面，不是一个。
 */
export function TimelinePage({ petId, petName }: { petId: string; petName: string }) {
  const [data, setData] = useState<TimelineResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [composing, setComposing] = useState(false)
  const [reply, setReply] = useState<RecordMomentResponse | null>(null)
  const [sceneFilter, setSceneFilter] = useState<string | null>(null)

  const load = useCallback(
    async (scene?: string | null) => {
      setLoading(true)
      setError(null)
      try {
        const res = await api.timeline(petId, {
          days: 90,
          limit: 200,
          ...(scene ? { scene } : {}),
        })
        setData(res)
      } catch (e) {
        setError(e instanceof ApiError ? e.message : '读取时间线失败')
      } finally {
        setLoading(false)
      }
    },
    [petId],
  )

  useEffect(() => {
    void load(sceneFilter)
  }, [load, sceneFilter])

  function handleRecorded(result: RecordMomentResponse) {
    // 猫的回应立刻显示 —— 那是这个动作的情绪回报
    setReply(result)
    setComposing(false)
    void load(sceneFilter)
  }

  const moments = data?.moments ?? []
  const grouped = groupByDay(moments)

  return (
    <div>
      {/* 头部：它 + 一句邀请 */}
      <header className="mb-5">
        <h1 className="font-display text-[26px] font-semibold leading-tight tracking-[-0.01em] text-ink">
          {petName} 的日记
        </h1>
        <p className="mt-1.5 text-[13px] leading-relaxed text-ink-faint">
          {moments.length > 0
            ? `已经记下 ${moments.length} 个瞬间。`
            : '看到什么就记一笔 —— 一句话都不用写。'}
        </p>
        <div className="mt-3 h-px origin-left animate-draw-line bg-rule" />
      </header>

      {/* 猫的回应 */}
      {reply && (
        <div className="mb-4 animate-rise-in">
          <Card binding className="border-persimmon/40 bg-persimmon/[0.045] py-3.5">
            <div className="flex items-start gap-3">
              {reply.moment.media_url && (
                <img
                  src={reply.moment.media_url}
                  alt=""
                  className="h-14 w-14 shrink-0 rounded-[2px] border border-rule object-cover"
                />
              )}
              <div className="min-w-0 flex-1">
                <p className="text-[14.5px] leading-relaxed text-ink">{reply.pet_says}</p>
                <p className="mt-1 text-[11px] text-ink-faint">
                  记下了 ·{' '}
                  <span className="rounded-[2px] bg-paper-sunk px-1.5 py-px">
                    {reply.moment.scene_display}
                  </span>
                </p>
              </div>
              <button
                onClick={() => setReply(null)}
                className="shrink-0 text-ink-faint transition hover:text-brick"
                aria-label="关掉"
              >
                <IconClose width={15} height={15} />
              </button>
            </div>
          </Card>
        </div>
      )}

      {/* 记录入口 */}
      {composing ? (
        <div className="mb-6 animate-rise-in">
          <MomentComposer petId={petId} petName={petName} onRecorded={handleRecorded} />
          <button
            onClick={() => setComposing(false)}
            className="mt-2 text-[12px] text-ink-faint transition hover:text-ink"
          >
            先不记了
          </button>
        </div>
      ) : (
        <Button className="mb-6 w-full py-3" onClick={() => setComposing(true)}>
          <IconPlus width={17} height={17} />
          记一个瞬间
        </Button>
      )}

      {error && (
        <div className="mb-4">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {/* 场景筛选 */}
      {data && Object.keys(data.scene_counts).length > 1 && (
        <div className="mb-4 flex flex-wrap items-center gap-1.5">
          <FilterChip
            active={sceneFilter === null}
            onClick={() => setSceneFilter(null)}
            label={`全部 ${moments.length || data.count}`}
          />
          {Object.entries(data.scene_counts).map(([scene, n]) => (
            <FilterChip
              key={scene}
              active={sceneFilter === scene}
              onClick={() => setSceneFilter(scene)}
              label={`${SCENE_LABELS[scene] ?? scene} ${n}`}
            />
          ))}
        </div>
      )}

      {/* 时间线 */}
      {loading ? (
        <div className="flex items-center gap-2.5 py-14 text-ink-faint">
          <Spinner />
          <span className="text-[13px]">翻开日记…</span>
        </div>
      ) : moments.length === 0 ? (
        <Empty
          title={sceneFilter ? '这一类还没有记录' : '日记还是空的'}
          hint={
            sceneFilter
              ? '换个分类看看，或者再记一笔。'
              : '每次看到它做了什么，随手记一笔就好 —— 照片就够了。'
          }
          action={
            !sceneFilter ? (
              <Button variant="ghost" onClick={() => setComposing(true)}>
                <IconCamera width={16} height={16} />
                记第一笔
              </Button>
            ) : undefined
          }
        />
      ) : (
        <div className="space-y-6">
          {grouped.map(([day, items], gi) => (
            <section key={day} style={{ animationDelay: stagger(gi, 60) }} className="animate-rise-in">
              <DayHeading day={day} count={items.length} />
              <ul className="space-y-2.5">
                {items.map((m) => (
                  <li key={m.moment_id}>
                    <MomentRow moment={m} />
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// 局部组件
// ─────────────────────────────────────────────────────────────

const SCENE_LABELS: Record<string, string> = {
  sleeping: '睡觉',
  playing: '玩耍',
  eating: '进食',
  window: '窗台',
  with_human: '和人待着',
  grooming: '舔毛',
  other: '其他',
}

/** 场景 → 一点颜色。**不用图标** —— 七个图标记不住，而标签一眼能读。 */
const SCENE_TONE: Record<string, string> = {
  sleeping: 'border-plum/45 text-plum',
  playing: 'border-persimmon/45 text-persimmon-deep',
  eating: 'border-ochre/50 text-ochre',
  window: 'border-moss/45 text-moss',
  with_human: 'border-persimmon/45 text-persimmon-deep',
  grooming: 'border-plum/45 text-plum',
  other: 'border-rule text-ink-faint',
}

function MomentRow({ moment }: { moment: Moment }) {
  const isImage = /\.(jpe?g|png|gif|webp)$/i.test(moment.media_url)
  return (
    <Card className="overflow-hidden p-0">
      {isImage && (
        <img
          src={moment.media_url}
          alt={moment.note ?? moment.scene_display}
          loading="lazy"
          className="max-h-80 w-full bg-paper-sunk object-cover"
        />
      )}
      <div className="flex items-start gap-2.5 px-3.5 py-3">
        <span
          className={cx(
            'mt-px shrink-0 rounded-[2px] border px-1.5 py-[3px] text-[10px] font-semibold',
            SCENE_TONE[moment.scene] ?? SCENE_TONE.other,
          )}
        >
          {moment.scene_display}
        </span>
        <p className="min-w-0 flex-1 text-[13.5px] leading-relaxed text-ink-soft">
          {moment.note || <span className="text-ink-faint">（没写字）</span>}
        </p>
        <span className="shrink-0 pt-0.5 font-mono text-[10px] text-ink-faint">
          {new Date(moment.captured_at).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </span>
      </div>
    </Card>
  )
}

function DayHeading({ day, count }: { day: string; count: number }) {
  const today = todayISO()
  const label =
    day === today ? '今天' : day === shift(today, -1) ? '昨天' : formatDayLabel(day)
  return (
    <div className="mb-2.5 flex items-baseline gap-2">
      <h2 className="font-display text-[13px] font-semibold uppercase tracking-[0.14em] text-ink-faint">
        {label}
      </h2>
      <span className="h-px flex-1 bg-rule" />
      <span className="font-mono text-[10px] text-ink-faint">{count} 笔</span>
    </div>
  )
}

function FilterChip({
  active,
  onClick,
  label,
}: {
  active: boolean
  onClick: () => void
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={cx(
        'rounded-pill border px-2.5 py-1 text-[11.5px] transition',
        active
          ? 'border-persimmon/55 bg-persimmon/[0.09] text-persimmon-deep'
          : 'border-rule bg-paper-raised text-ink-faint hover:text-ink-soft',
      )}
    >
      {label}
    </button>
  )
}

// ─────────────────────────────────────────────────────────────
// 工具
// ─────────────────────────────────────────────────────────────

function groupByDay(moments: Moment[]): Array<[string, Moment[]]> {
  const map = new Map<string, Moment[]>()
  for (const m of moments) {
    const day = m.captured_at.slice(0, 10)
    const list = map.get(day)
    if (list) list.push(m)
    else map.set(day, [m])
  }
  // 后端已倒序，Map 保持插入顺序 → 日期自然倒序
  return [...map.entries()]
}

function shift(iso: string, delta: number): string {
  const d = new Date(`${iso}T12:00:00`)
  d.setDate(d.getDate() + delta)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

function formatDayLabel(iso: string): string {
  const d = new Date(`${iso}T12:00:00`)
  return `${d.getMonth() + 1} 月 ${d.getDate()} 日`
}
