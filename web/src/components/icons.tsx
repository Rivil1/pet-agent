/**
 * 图标：手绘感的线性墨迹。
 *
 * 刻意不用图标库 —— 通用图标集会把「田野笔记」的调性拉回通用 SaaS。
 * 这些是 24×24 网格上的简笔，笔画粗细与 `stroke-linecap` 统一。
 */
import type { SVGProps } from 'react'

type IconProps = SVGProps<SVGSVGElement>

function Base({ children, ...rest }: IconProps & { children: React.ReactNode }) {
  return (
    <svg
      width="22"
      height="22"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      {children}
    </svg>
  )
}

/** 宠物：猫脸（耳 + 胡须） */
export const IconCat = (p: IconProps) => (
  <Base {...p}>
    <path d="M4.5 10.5 4 4.5l4 3" />
    <path d="M19.5 10.5 20 4.5l-4 3" />
    <path d="M4.5 11c0 5 3.3 8.5 7.5 8.5s7.5-3.5 7.5-8.5" />
    <path d="M9.3 11.6h.01M14.7 11.6h.01" strokeWidth="2.4" />
    <path d="M12 14.6v1.2M10.4 16.4h3.2" />
    <path d="M2.8 12.8l2.6-.5M21.2 12.8l-2.6-.5" />
  </Base>
)

/** 对话：说话的气泡，带一点手写尾巴 */
export const IconChat = (p: IconProps) => (
  <Base {...p}>
    <path d="M20 12.5c0 3.7-3.6 6.7-8 6.7-1 0-2-.15-2.9-.42L4.5 20.5l1-3.4A6.6 6.6 0 0 1 4 12.5C4 8.8 7.6 5.8 12 5.8s8 3 8 6.7Z" />
    <path d="M9 12h.01M12 12h.01M15 12h.01" strokeWidth="2.2" />
  </Base>
)

/** 档案：夹着照片的卡片 */
export const IconProfile = (p: IconProps) => (
  <Base {...p}>
    <rect x="3.5" y="5" width="17" height="14" rx="1.5" />
    <path d="M3.5 15.5l4.2-4 3.6 3.2 3-2.6 6.2 5.4" />
    <circle cx="9" cy="9.2" r="1.4" />
  </Base>
)

/** 故事：翻开的册子 */
export const IconStory = (p: IconProps) => (
  <Base {...p}>
    <path d="M12 7.5C10.4 6.3 8.4 5.8 5 5.9v12.2c3.4-.1 5.4.4 7 1.6" />
    <path d="M12 7.5c1.6-1.2 3.6-1.7 7-1.6v12.2c-3.4-.1-5.4.4-7 1.6" />
    <path d="M12 7.5v14.2" />
  </Base>
)

/** 健康：脉搏线 */
export const IconHealth = (p: IconProps) => (
  <Base {...p}>
    <path d="M3 12.5h3.4l1.8-4.5 2.6 9 2.3-6 1.5 3h6.4" />
  </Base>
)

/** 叫声：声波 */
export const IconWave = (p: IconProps) => (
  <Base {...p}>
    <path d="M4 12h1.6M8 8.5v7M12 5.5v13M16 9v6M20 11.2v1.6" />
  </Base>
)

export const IconPlus = (p: IconProps) => (
  <Base {...p}>
    <path d="M12 5.5v13M5.5 12h13" />
  </Base>
)

export const IconBack = (p: IconProps) => (
  <Base {...p}>
    <path d="M14.5 5.5 8 12l6.5 6.5" />
  </Base>
)

export const IconSend = (p: IconProps) => (
  <Base {...p}>
    <path d="M4.5 12 19.5 5l-6 14-2.4-5.6L4.5 12Z" />
  </Base>
)

export const IconRefresh = (p: IconProps) => (
  <Base {...p}>
    <path d="M19 12a7 7 0 1 1-2.4-5.3" />
    <path d="M19.5 4.5v4h-4" />
  </Base>
)

export const IconCamera = (p: IconProps) => (
  <Base {...p}>
    <path d="M4 8.5h2.6l1.3-2h8.2l1.3 2H20v10H4Z" />
    <circle cx="12" cy="13" r="3.1" />
  </Base>
)

export const IconCheck = (p: IconProps) => (
  <Base {...p}>
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </Base>
)

export const IconClose = (p: IconProps) => (
  <Base {...p}>
    <path d="M6 6l12 12M18 6L6 18" />
  </Base>
)

export const IconChevron = (p: IconProps) => (
  <Base {...p}>
    <path d="M9.5 5.5 16 12l-6.5 6.5" />
  </Base>
)

export const IconLogout = (p: IconProps) => (
  <Base {...p}>
    <path d="M14 5.5H6.5v13H14" />
    <path d="M12.5 12h7M17 8.8l3.2 3.2-3.2 3.2" />
  </Base>
)

/** 日记：一本摊开的本子 + 一条书签带 */
export const IconDiary = (p: IconProps) => (
  <Base {...p}>
    <path d="M6 3.5h11a1.5 1.5 0 0 1 1.5 1.5v14a1.5 1.5 0 0 1-1.5 1.5H6" />
    <path d="M6 3.5A1.5 1.5 0 0 0 4.5 5v14A1.5 1.5 0 0 0 6 20.5" />
    <path d="M7.5 8h8M7.5 11.5h8M7.5 15h5" />
    <path d="M15.5 3.5v5l1.5-1 1.5 1v-5" />
  </Base>
)
