import type { ReactNode } from 'react'
import type { PetBrief } from '../api/types'
import { cx } from '../lib/utils'
import { IconCat, IconChat, IconHealth, IconLogout, IconProfile, IconStory } from './icons'

export type TabKey = 'pets' | 'chat' | 'profile' | 'story' | 'health'

export const TABS: Array<{
  key: TabKey
  label: string
  Icon: (p: { width?: number; height?: number }) => ReactNode
}> = [
  { key: 'chat', label: '对话', Icon: IconChat },
  { key: 'profile', label: '档案', Icon: IconProfile },
  { key: 'story', label: '一天', Icon: IconStory },
  { key: 'health', label: '健康', Icon: IconHealth },
  { key: 'pets', label: '猫', Icon: IconCat },
]

/**
 * 应用外壳：顶栏 + 内容 + 底部标签栏。
 *
 * 顶栏常驻**当前宠物名**与**模型模式**两件事。
 * 前者是因为所有数据都隶属于某一只猫 —— 不显示它，
 * 用户会分不清自己正在记录哪一只（多宠物串味是这类产品最致命的缺陷）。
 * 后者是因为 mock 模式下的输出与真实输出长得一样，
 * 不常驻标注，一次演示会静默地变成假演示（决策 D45）。
 */
export function AppShell({
  activePet,
  petCount,
  providerMode,
  tab,
  onTab,
  onLogout,
  children,
}: {
  activePet: PetBrief | null
  petCount: number
  providerMode: string | null
  tab: TabKey
  onTab: (t: TabKey) => void
  onLogout: () => void
  children: ReactNode
}) {
  const isLive = providerMode === 'live'

  return (
    <div className="paper-grain paper-vignette relative flex h-full flex-col">
      {/* 顶栏 */}
      <header className="pt-safe relative z-20 border-b border-rule bg-paper/85 backdrop-blur-sm">
        <div className="mx-auto flex w-full max-w-[640px] items-center gap-3 px-4 pb-2.5 pt-1">
          <div className="flex min-w-0 flex-1 items-center gap-2.5">
            <span className="font-display text-[17px] font-semibold tracking-[-0.01em] text-ink">
              猫事
            </span>
            <span className="h-3.5 w-px bg-rule" />
            {activePet ? (
              <button
                onClick={() => onTab('pets')}
                className="flex min-w-0 items-center gap-1.5 text-left"
                title="切换宠物"
              >
                <span className="truncate text-[13px] font-semibold text-ink">
                  {activePet.name}
                </span>
                {petCount > 1 && (
                  <span className="shrink-0 rounded-[2px] bg-paper-sunk px-1 py-px font-mono text-[10px] text-ink-faint">
                    {petCount}
                  </span>
                )}
              </button>
            ) : (
              <span className="text-[13px] text-ink-faint">还没有猫</span>
            )}
          </div>

          {/* 模型模式：常驻，不可关闭 */}
          <span
            className={cx(
              'stamp shrink-0',
              isLive ? 'border-moss/70 text-moss' : 'border-ochre/70 text-ochre',
            )}
            title={
              isLive
                ? '输出由真实模型产生'
                : '输出不是真实模型推理的结果 —— 不要当作真实结论'
            }
          >
            {isLive ? 'LIVE' : (providerMode ?? '未知').toUpperCase()}
          </span>

          <button
            onClick={onLogout}
            className="shrink-0 p-1.5 text-ink-faint transition hover:text-brick"
            title="退出"
            aria-label="退出登录"
          >
            <IconLogout width={18} height={18} />
          </button>
        </div>
      </header>

      {/* 内容 */}
      <main className="relative z-10 flex-1 overflow-y-auto overscroll-contain">
        <div className="mx-auto w-full max-w-[640px] px-4 pb-6 pt-4">{children}</div>
      </main>

      {/* 底部标签栏 */}
      <nav className="pb-safe relative z-20 border-t border-rule bg-paper/92 backdrop-blur-sm">
        <div className="mx-auto flex w-full max-w-[640px] items-stretch">
          {TABS.map(({ key, label, Icon }) => {
            const active = tab === key
            return (
              <button
                key={key}
                onClick={() => onTab(key)}
                className={cx(
                  'relative flex flex-1 flex-col items-center gap-0.5 py-2 transition',
                  active ? 'text-persimmon' : 'text-ink-faint hover:text-ink-soft',
                )}
                aria-current={active ? 'page' : undefined}
              >
                {active && (
                  <span className="absolute -top-px left-1/2 h-[2px] w-7 -translate-x-1/2 bg-persimmon" />
                )}
                <Icon width={20} height={20} />
                <span className="text-[10.5px] font-semibold tracking-[0.02em]">{label}</span>
              </button>
            )
          })}
        </div>
      </nav>
    </div>
  )
}
