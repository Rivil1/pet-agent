import { useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { RecordMomentResponse } from '../api/types'
import { Button, Input, Notice, Spinner } from './ui'
import { IconCamera, IconClose } from './icons'
import { cx } from '../lib/utils'

/**
 * 记录一个瞬间。
 *
 * ## 这一屏的全部设计目标是「一次点击就完成」
 *
 * `docs/11` §2.1 的判断是：
 *
 * > **用户想猫时是情绪状态**，不想打字、不想被问问题。
 * > 每多一个输入字段，转化率就掉一截。
 *
 * 所以：
 *
 * | 不做 | 为什么 |
 * |---|---|
 * | 不强制描述 | 想猫的时候不想打字 |
 * | 不要求选分类 | 分类由系统抽 |
 * | 不弹确认框 | 与档案相反 —— 档案是身份锚点（写错污染校验），瞬间是流水（错了再记一条） |
 *
 * 界面上因此只有一个必填项：**那张照片**。
 */
export function MomentComposer({
  petId,
  petName,
  onRecorded,
}: {
  petId: string
  petName: string
  onRecorded: (result: RecordMomentResponse) => void
}) {
  const [dataUrl, setDataUrl] = useState<string | null>(null)
  const [previewKind, setPreviewKind] = useState<'image' | 'video' | 'audio'>('image')
  const [note, setNote] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  function pickFile(file: File | undefined) {
    if (!file) return
    setError(null)
    // 8MB 上限与后端一致（后端还会再按魔数校验一次 —— 前端这一步只是提前告知）
    if (file.size > 8 * 1024 * 1024) {
      setError('文件超过 8MB。挑一张小一点的？')
      return
    }
    setPreviewKind(
      file.type.startsWith('video') ? 'video' : file.type.startsWith('audio') ? 'audio' : 'image',
    )
    const reader = new FileReader()
    reader.onload = () => setDataUrl(String(reader.result))
    reader.onerror = () => setError('读不出这个文件')
    reader.readAsDataURL(file)
  }

  async function submit() {
    if (!dataUrl) return
    setBusy(true)
    setError(null)
    try {
      const media = await api.uploadMedia(dataUrl)
      // 一句话是**可选**的 —— 空字符串不传，让后端按「没写」处理（归类为其他）
      const result = await api.recordMoment(petId, media.url, note.trim() || undefined)
      setDataUrl(null)
      setNote('')
      onRecorded(result)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '记录失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card tape p-4">
      <p className="mb-3 font-display text-[15px] font-semibold text-ink">
        记一个 {petName} 的瞬间
      </p>

      {error && (
        <div className="mb-3">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {/* 选照片 —— 唯一的必填动作 */}
      <input
        ref={fileRef}
        type="file"
        accept="image/*,video/*"
        className="hidden"
        onChange={(e) => pickFile(e.target.files?.[0])}
      />

      {dataUrl ? (
        <div className="relative mb-3 overflow-hidden rounded-[3px] border border-rule bg-paper-sunk">
          {previewKind === 'image' ? (
            <img src={dataUrl} alt="待记录" className="max-h-72 w-full object-contain" />
          ) : (
            <div className="flex items-center gap-2 px-3 py-6 text-[13px] text-ink-soft">
              <IconCamera width={18} height={18} />
              已选中一段{previewKind === 'video' ? '视频' : '音频'}
            </div>
          )}
          <button
            onClick={() => setDataUrl(null)}
            className="absolute right-2 top-2 rounded-full bg-ink/55 p-1.5 text-paper-raised transition hover:bg-ink/75"
            aria-label="移除"
          >
            <IconClose width={14} height={14} />
          </button>
        </div>
      ) : (
        <button
          onClick={() => fileRef.current?.click()}
          className={cx(
            'mb-3 flex w-full flex-col items-center gap-2 rounded-[3px]',
            'border-[1.5px] border-dashed border-rule bg-paper py-9',
            'text-ink-faint transition hover:border-persimmon/50 hover:text-persimmon',
          )}
        >
          <IconCamera width={26} height={26} />
          <span className="text-[13px] font-medium">选一张照片</span>
          <span className="text-[11px] opacity-70">照片就够了，不用写字</span>
        </button>
      )}

      {/* 可选的一句话 */}
      <Input
        value={note}
        onChange={(e) => setNote(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && dataUrl && submit()}
        placeholder="想说一句就说（可不填）"
        maxLength={200}
      />

      <div className="mt-3 flex items-center gap-2">
        <Button
          loading={busy}
          disabled={!dataUrl}
          onClick={submit}
          className="flex-1"
        >
          记下来
        </Button>
        {busy && <Spinner className="text-ink-faint" />}
      </div>

      <p className="mt-2.5 text-[11px] leading-relaxed text-ink-faint">
        场景（睡觉 / 玩耍 / 进食…）由我按你写的话归类。
        <span className="text-ink-faint/80">不写也没关系 —— 那就归到「其他」。</span>
      </p>
    </div>
  )
}
