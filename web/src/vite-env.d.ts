/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** 后端地址。同源部署时留空；跨域时填完整 origin。 */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
