/** 小工具。刻意不引依赖 —— 需要的东西就这几行。 */

export function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(' ')
}

/** 2026-09-13 → 「9月13日」 */
export function formatDayShort(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return `${d.getMonth() + 1}月${d.getDate()}日`
}

/** 相对时间：刚刚 / 12 分钟前 / 3 小时前 / 9月13日 */
export function formatRelative(iso: string | null | undefined): string {
  if (!iso) return ''
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const diff = Date.now() - d.getTime()
  const min = Math.floor(diff / 60000)
  if (min < 1) return '刚刚'
  if (min < 60) return `${min} 分钟前`
  const hours = Math.floor(min / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days} 天前`
  return formatDayShort(iso)
}

/** 今天（本地时区）的 YYYY-MM-DD。故事接口按天取。 */
export function todayISO(): string {
  const d = new Date()
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

export function shiftDay(iso: string, deltaDays: number): string {
  const d = new Date(`${iso}T12:00:00`)
  d.setDate(d.getDate() + deltaDays)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

// ─────────────────────────────────────────────────────────────
// 词表：后端用固定词表（自由文本无法可靠匹配），这里给出中文显示
// ─────────────────────────────────────────────────────────────

export const CONTEXT_LABELS: Record<string, string> = {
  food_waiting: '等吃的',
  door_attention: '想出门 / 看门外',
  affection_brushing: '想要亲近 / 梳毛',
  isolation_distress: '独处不安',
  greeting: '打招呼',
  other: '其他 / 说不清',
}

export const ACTION_LABELS: Record<string, string> = {
  scratch_door: '抓门',
  pacing: '来回走',
  rub_leg: '蹭腿',
  tail_up: '竖尾巴',
  near_food_bowl: '在饭碗旁',
  look_at_door: '盯着门',
  hiding: '躲起来',
  purring: '呼噜',
  belly_up: '翻肚皮',
  approach_human: '主动靠近人',
  avoid_contact: '避开接触',
  grooming: '舔毛',
  arched_back: '弓背',
  other: '其他',
}

export const EVIDENCE_MODE_LABELS: Record<string, { label: string; hint: string }> = {
  acoustic_plus_history: {
    label: '声学 + 个体历史',
    hint: '有本次测量，也有这只猫的历史标注可比对',
  },
  case_based: {
    label: '先例比对',
    hint: '本次测不准，但历史里有相似情境可参照',
  },
  measured_only: {
    label: '仅本次测量',
    hint: '只有本次声学测量，还没有这只猫的历史样本',
  },
  text_only: {
    label: '仅文字描述',
    hint: '没测到声学特征，只有文字情境 —— 不给数值置信度',
  },
}

export const EVIDENCE_KIND_LABELS: Record<string, string> = {
  measured: '测量',
  observed: '观察',
  retrieved: '检索',
  prior: '先验',
}

/**
 * 健康分级的中文呈现。
 *
 * **键必须与后端 `UrgencyLevel` 逐字一致** —— 这是唯一真相，不是可以意译的名字。
 *
 * 语义上的两个关键点（都是产品决策，不是文案）：
 *
 * - `NO_DEVIATION_DETECTED` **不等于「健康」**。它只说「在当前采集到的信号里
 *   没有命中规则」。写「健康」会把一个覆盖度受限的结论说成事实。
 * - `INSUFFICIENT_DATA` 必须是**独立第三态**，不能归入「好」或「坏」。
 *   把它显示成「未发现异常」是这套系统里最危险的一种 UI 缺陷（决策 D12）。
 */
export const HEALTH_LEVELS: Record<
  string,
  { label: string; tone: 'danger' | 'warn' | 'ok' | 'muted'; hint: string }
> = {
  L3: { label: '立即就医', tone: 'danger', hint: '命中紧急红旗 —— 不要等' },
  L2: { label: '建议尽快就诊', tone: 'warn', hint: '有指标偏离，建议让兽医看' },
  L1: { label: '先观察', tone: 'warn', hint: '轻微偏离，继续记录看趋势' },
  NO_DEVIATION_DETECTED: {
    label: '未检测到偏离',
    tone: 'ok',
    hint: '在已采集的信号里未命中规则 —— 不等于「健康」',
  },
  INSUFFICIENT_DATA: {
    label: '无法评估',
    tone: 'muted',
    hint: '数据不足 —— 「不知道」不等于「没问题」',
  },
}

/** 生成一个稳定的伪随机延迟，用于交错入场动画。 */
export function stagger(index: number, stepMs = 55, max = 420): string {
  return `${Math.min(index * stepMs, max)}ms`
}
