import { useState } from 'react'
import { api, ApiError } from '../api/client'
import type { PetBrief } from '../api/types'
import { Button, Card, Empty, Field, Input, Notice, PageTitle, SectionTitle } from '../components/ui'
import { IconCat, IconPlus } from '../components/icons'
import { cx, formatRelative, stagger } from '../lib/utils'

/**
 * 宠物管理。
 *
 * 这一屏是「归属」的可视化：每只猫是一个独立的世界，
 * 记忆、档案、健康数据都不跨猫流动。所以切换宠物是一次**上下文切换**，
 * 而不是一个筛选器 —— 文案与交互都按这个语义来。
 */
export function PetsPage({
  pets,
  activePetId,
  onSelect,
  onCreated,
  loading,
}: {
  pets: PetBrief[]
  activePetId: string | null
  onSelect: (pet: PetBrief) => void
  onCreated: (pet: PetBrief) => void
  loading: boolean
}) {
  const [creating, setCreating] = useState(false)
  const [name, setName] = useState('')
  const [breed, setBreed] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleCreate() {
    const n = name.trim()
    if (!n) return
    setBusy(true)
    setError(null)
    try {
      const created = await api.createPet(n, breed.trim() || undefined)
      // 后端只回 {pet_id, name}，补成列表项的形状
      const brief: PetBrief = {
        pet_id: created.pet_id,
        name: created.name,
        species: 'cat',
        breed: breed.trim() || null,
        has_profile: false,
        must_keep_features: [],
        fur_color: null,
        eye_color: null,
        created_at: new Date().toISOString(),
      }
      onCreated(brief)
      setName('')
      setBreed('')
      setCreating(false)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : '创建失败')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <PageTitle sub="每一只猫都是一个独立的世界，记录不会互相串味。">我的猫</PageTitle>

      {error && (
        <div className="mb-4">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {loading && pets.length === 0 ? (
        <div className="space-y-3">
          {[0, 1].map((i) => (
            <div key={i} className="card h-[86px] animate-pulse-soft bg-paper-sunk/60" />
          ))}
        </div>
      ) : pets.length === 0 && !creating ? (
        <Empty
          title="还没有记录任何一只猫"
          hint="先建一个档案，之后的对话、叫声解释、健康记录都会挂在它名下。"
          action={
            <Button onClick={() => setCreating(true)}>
              <IconPlus width={16} height={16} />
              建立第一只
            </Button>
          }
        />
      ) : (
        <>
          <SectionTitle
            right={
              <button
                onClick={() => setCreating((v) => !v)}
                className="flex items-center gap-1 text-[12px] font-semibold text-persimmon"
              >
                <IconPlus width={14} height={14} />
                新增
              </button>
            }
          >
            {pets.length} 只
          </SectionTitle>

          <div className="space-y-2.5">
            {pets.map((pet, i) => {
              const active = pet.pet_id === activePetId
              return (
                <button
                  key={pet.pet_id}
                  onClick={() => onSelect(pet)}
                  style={{ animationDelay: stagger(i) }}
                  className={cx(
                    'card binding animate-rise-in w-full p-4 pl-5 text-left transition',
                    active
                      ? 'border-persimmon/55 bg-persimmon/[0.045]'
                      : 'hover:border-ink-faint/45',
                  )}
                >
                  <div className="flex items-start gap-3">
                    <span
                      className={cx(
                        'mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-[3px] border',
                        active
                          ? 'border-persimmon/60 text-persimmon'
                          : 'border-rule text-ink-faint',
                      )}
                    >
                      <IconCat width={20} height={20} />
                    </span>

                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2">
                        <span className="font-display text-[17px] font-semibold text-ink">
                          {pet.name}
                        </span>
                        {active && (
                          <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-persimmon">
                            当前
                          </span>
                        )}
                      </div>

                      <p className="mt-0.5 truncate text-[12px] text-ink-faint">
                        {pet.breed || '未填品种'}
                        {pet.fur_color ? ` · ${pet.fur_color}` : ''}
                      </p>

                      <div className="mt-2 flex flex-wrap items-center gap-1.5">
                        {pet.has_profile ? (
                          <span className="ev ev-measured">档案已建</span>
                        ) : (
                          <span className="ev ev-inferred">档案待建</span>
                        )}
                        {pet.must_keep_features.slice(0, 2).map((f) => (
                          <span
                            key={f}
                            className="rounded-[2px] bg-paper-sunk px-1.5 py-[3px] text-[10px] text-ink-soft"
                          >
                            {f}
                          </span>
                        ))}
                        {pet.must_keep_features.length > 2 && (
                          <span className="text-[10px] text-ink-faint">
                            +{pet.must_keep_features.length - 2}
                          </span>
                        )}
                      </div>
                    </div>

                    <span className="shrink-0 pt-0.5 font-mono text-[10px] text-ink-faint">
                      {formatRelative(pet.created_at)}
                    </span>
                  </div>
                </button>
              )
            })}
          </div>
        </>
      )}

      {/* 新增表单 */}
      {creating && (
        <Card tape className="mt-5 animate-rise-in p-4">
          <p className="mb-3 font-display text-[15px] font-semibold text-ink">新建一只猫</p>
          <div className="space-y-3">
            <Field label="名字" hint="记录里会一直用它，取你平时叫的那个称呼。">
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                placeholder="例如 团团"
                autoFocus
              />
            </Field>
            <Field label="品种（可不填）">
              <Input
                value={breed}
                onChange={(e) => setBreed(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                placeholder="例如 英短"
              />
            </Field>
          </div>
          <div className="mt-4 flex gap-2">
            <Button loading={busy} disabled={!name.trim()} onClick={handleCreate}>
              建立
            </Button>
            <Button variant="quiet" onClick={() => setCreating(false)}>
              取消
            </Button>
          </div>
        </Card>
      )}
    </div>
  )
}
