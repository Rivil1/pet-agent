import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from '../api/client'
import type { PetProfile, ProfileDraft } from '../api/types'
import { Button, Card, Empty, Field, Input, Notice, PageTitle, SectionTitle, Spinner } from '../components/ui'
import { EvidenceBadge, VerifiedStamp, OwnerMark } from '../components/EvidenceBadge'
import { IconCamera, IconPlus, IconClose } from '../components/icons'
import { cx, stagger } from '../lib/utils'

/**
 * 档案页 —— 身份锚点 + 照片分析。
 *
 * ## 这一屏为什么必须区分「硬特征」与「不稳定特征」
 *
 * 后端把多图取交集得到的特征标为 `must_keep_features`，
 * 而只在部分照片里出现的标为 `observed_but_unstable`。
 * 这不是数据结构的细节，而是**身份一致性的保证**：
 *
 * - 硬特征 = 在每张照片里都成立 → 可以当作这只猫的锚点
 * - 不稳定特征 = 只在一张里出现（可能是光线、角度、姿势）→ 不能当锚点
 *
 * 把它们渲染成同一个列表，用户就会以为「胸口有白毛」和「好像有点胖」
 * 同等可靠，而这会导致后续的生成/校验用错锚点。
 *
 * 另外：`confirm=false` 时后端**不落库**。所以必须有一步显式的「确认」，
 * 否则用户以为存了，其实没有 —— 这种静默失败最难发现。
 */
export function ProfilePage({ petId, petName }: { petId: string; petName: string }) {
  const [profile, setProfile] = useState<PetProfile | null>(null)
  const [draft, setDraft] = useState<ProfileDraft | null>(null)
  const [imageUrls, setImageUrls] = useState<string[]>([])
  const [urlInput, setUrlInput] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setProfile(await api.getProfile(petId))
      setDraft(null)
      setImageUrls([])
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '读取档案失败')
    } finally {
      setLoading(false)
    }
  }, [petId])

  useEffect(() => {
    void load()
  }, [load])

  function addUrl() {
    const u = urlInput.trim()
    if (!u || imageUrls.includes(u)) return
    setImageUrls((prev) => [...prev, u])
    setUrlInput('')
  }

  async function analyze() {
    if (imageUrls.length === 0) return
    setBusy(true)
    setError(null)
    try {
      const res = await api.buildProfile(petId, imageUrls, false)
      setDraft(res as ProfileDraft)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '照片分析失败')
    } finally {
      setBusy(false)
    }
  }

  async function confirm() {
    if (imageUrls.length === 0) return
    setBusy(true)
    setError(null)
    try {
      await api.buildProfile(petId, imageUrls, true)
      setDraft(null)
      setImageUrls([])
      await load()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '确认失败')
    } finally {
      setBusy(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2.5 py-16 text-ink-faint">
        <Spinner />
        <span className="text-[13px]">读取档案…</span>
      </div>
    )
  }

  const visual = profile?.visual
  const hasProfile = Boolean(profile?.must_keep_features?.length || visual?.fur_color)

  return (
    <div>
      <PageTitle sub="这只猫长什么样，是后续所有判断的身份锚点。">{petName} 的档案</PageTitle>

      {error && (
        <div className="mb-4">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {/* 已有档案 */}
      {hasProfile ? (
        <>
          <SectionTitle>身份锚点</SectionTitle>
          <Card binding className="mb-5">
            <div className="mb-3 flex items-center gap-2">
              <EvidenceBadge kind="measured" />
              <span className="text-[11.5px] text-ink-faint">每张照片里都成立</span>
              <VerifiedStamp>可用于校验</VerifiedStamp>
            </div>

            {profile!.must_keep_features.length > 0 ? (
              <ul className="space-y-1.5">
                {profile!.must_keep_features.map((f, i) => (
                  <li
                    key={f}
                    style={{ animationDelay: stagger(i, 40) }}
                    className="animate-rise-in flex items-start gap-2 text-[14px] leading-relaxed text-ink"
                  >
                    <span className="mt-[9px] h-1 w-3 shrink-0 bg-persimmon/70" aria-hidden />
                    <span className="underline-ink">{f}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[13px] text-ink-faint">没有提取到稳定特征。</p>
            )}

            {/* 表观描述 */}
            <div className="mt-4 border-t border-dashed border-rule pt-3">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5">
                <Attr label="毛色" value={visual?.fur_color} />
                <Attr label="毛长" value={visual?.fur_length} />
                <Attr label="眼睛" value={visual?.eye_color} />
                <Attr label="体型" value={visual?.body_shape} />
                <Attr label="脸型" value={visual?.face_shape} />
              </dl>
            </div>
          </Card>

          {profile!.observed_but_unstable.length > 0 && (
            <>
              <SectionTitle>只在部分照片里出现</SectionTitle>
              <Card className="mb-5">
                <div className="mb-2.5 flex items-center gap-2">
                  <EvidenceBadge kind="observed" />
                  <span className="text-[11.5px] text-ink-faint">不能当锚点</span>
                </div>
                <ul className="flex flex-wrap gap-1.5">
                  {profile!.observed_but_unstable.map((f) => (
                    <li
                      key={f}
                      className="rounded-[2px] border border-dashed border-ochre/45 bg-ochre/[0.05] px-2 py-1 text-[12px] text-ink-soft"
                    >
                      {f}
                    </li>
                  ))}
                </ul>
                <p className="mt-2.5 text-[11.5px] leading-relaxed text-ink-faint">
                  可能只是光线、角度或姿势造成的。它们不进身份锚点 ——
                  否则每次校验都会拿一个碰巧的特征去比对。
                </p>
              </Card>
            </>
          )}
        </>
      ) : (
        <Empty
          title="还没有档案"
          hint="上传几张这只猫的照片，系统会取多张的交集，得出它的稳定特征。"
        />
      )}

      {/* 照片分析 */}
      <SectionTitle className="mt-6">用照片建立档案</SectionTitle>
      <Card tape className="p-4">
        <div className="space-y-2.5">
          <Field label="照片地址" hint="多张照片取交集 —— 只在部分照片出现的特征不会被当作锚点。">
            <div className="flex gap-2">
              <Input
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault()
                    addUrl()
                  }
                }}
                placeholder="https://…/cat.jpg"
                spellCheck={false}
              />
              <Button variant="ghost" className="shrink-0 px-3" onClick={addUrl}>
                <IconPlus width={16} height={16} />
              </Button>
            </div>
          </Field>

          {imageUrls.length > 0 && (
            <ul className="space-y-1.5">
              {imageUrls.map((u) => (
                <li
                  key={u}
                  className="flex items-center gap-2 rounded-[2px] border border-rule bg-paper px-2.5 py-1.5"
                >
                  <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-ink-soft">
                    {u}
                  </span>
                  <button
                    onClick={() => setImageUrls((prev) => prev.filter((x) => x !== u))}
                    className="shrink-0 text-ink-faint transition hover:text-brick"
                    aria-label="移除"
                  >
                    <IconClose width={14} height={14} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="flex flex-wrap gap-2 pt-1">
            <Button
              loading={busy}
              disabled={imageUrls.length === 0}
              onClick={analyze}
              className={cx(imageUrls.length === 0 && 'opacity-45')}
            >
              <IconCamera width={16} height={16} />
              先看草案
            </Button>
            {draft && (
              <Button variant="ghost" loading={busy} onClick={confirm}>
                确认并保存
              </Button>
            )}
          </div>

          <p className="text-[11px] leading-relaxed text-ink-faint">
            「先看草案」**不会保存任何东西** —— 档案必须经你确认才落库。
          </p>
        </div>
      </Card>

      {/* 草案 */}
      {draft && (
        <Card tape className="mt-5 animate-rise-in border-persimmon/40 p-4">
          <div className="mb-3 flex items-center justify-between gap-2">
            <span className="font-display text-[15px] font-semibold text-ink">草案（未保存）</span>
            <span className="stamp border-persimmon text-persimmon-deep">待确认</span>
          </div>

          {typeof draft.coverage === 'number' && (
            <div className="mb-3">
              <div className="mb-1.5 flex items-baseline justify-between">
                <span className="text-[12px] text-ink-faint">覆盖度</span>
                <span className="font-mono text-[12px] tabular-nums text-ink">
                  {Math.round((draft.coverage as number) * 100)}%
                </span>
              </div>
              <div className="h-[3px] overflow-hidden rounded-full bg-paper-sunk">
                <div
                  className="h-full bg-persimmon/70"
                  style={{ width: `${Math.min(100, (draft.coverage as number) * 100)}%` }}
                />
              </div>
              {draft.coverage_note && (
                <p className="mt-1.5 text-[11.5px] leading-relaxed text-ink-faint">
                  {draft.coverage_note}
                </p>
              )}
            </div>
          )}

          {Array.isArray(draft.must_keep_features) && draft.must_keep_features.length > 0 && (
            <div className="mb-3">
              <p className="mb-1.5 flex items-center gap-2 text-[12px] text-ink-faint">
                <EvidenceBadge kind="measured" />
                会写进身份锚点
              </p>
              <ul className="space-y-1">
                {(draft.must_keep_features as string[]).map((f) => (
                  <li key={f} className="flex items-start gap-2 text-[13.5px] text-ink">
                    <span className="mt-[8px] h-1 w-3 shrink-0 bg-persimmon/70" aria-hidden />
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {Array.isArray(draft.observed_but_unstable) && draft.observed_but_unstable.length > 0 && (
            <div>
              <p className="mb-1.5 flex items-center gap-2 text-[12px] text-ink-faint">
                <EvidenceBadge kind="observed" />
                不稳定，不会写进锚点
              </p>
              <ul className="flex flex-wrap gap-1.5">
                {(draft.observed_but_unstable as string[]).map((f) => (
                  <li
                    key={f}
                    className="rounded-[2px] border border-dashed border-ochre/45 px-2 py-0.5 text-[12px] text-ink-soft"
                  >
                    {f}
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="mt-3.5 flex items-center gap-2 text-[11.5px] text-ink-faint">
            <OwnerMark>由你确认</OwnerMark>
            上面这份还没保存。看一下对不对，再决定。
          </p>
        </Card>
      )}
    </div>
  )
}

function Attr({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="text-[12px] text-ink-faint">{label}</dt>
      <dd className={cx('text-[12.5px]', value ? 'text-ink' : 'text-ink-faint')}>
        {value || '—'}
      </dd>
    </div>
  )
}
