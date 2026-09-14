/**
 * API 客户端。
 *
 * 三件事在这里收口，别处不该重复：
 *
 * 1. **归属只从 token 来**。这里没有 `user_id` 参数 —— 后端从凭证派生。
 *    前端多一个口子，后端就多一个被绕过的可能（`ARCHITECTURE.md` §4.2 A1）。
 * 2. **session_id 持久化**。会话记忆注入依赖它：每次请求换一个，
 *    「最近 N 轮」就永远是空的，多轮指代也就无从谈起（D47）。
 * 3. **错误归一**。后端的错误体有两种形状（`{detail: {code, message}}`
 *    与 FastAPI 的校验数组），调用方只面对一种。
 */
import type {
  DevLoginResponse,
  HealthResponse,
  HealthzResponse,
  MeowRecord,
  MeowRecordListResponse,
  MemoryListResponse,
  PetBrief,
  PetListResponse,
  PetProfile,
  ProfileDraft,
  MediaUploadResponse,
  RecordMomentResponse,
  StoryResponse,
  TimelineResponse,
  TurnResponse,
  AudioKind,
  BehaviorAction,
  ContextLabel,
} from './types'
/** 同源部署时留空；本地开发由 vite proxy 转发。 */
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''
const TOKEN_KEY = 'pet-agent.token'
const SESSION_KEY = 'pet-agent.session'
const PET_KEY = 'pet-agent.active-pet'
// ─────────────────────────────────────────────────────────────
// 凭据与会话
// ─────────────────────────────────────────────────────────────
export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}
export function setToken(token: string | null): void {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token)
    else localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* 隐私模式下 localStorage 可能不可写 —— 不该因此崩掉整个应用 */
  }
}
/**
 * 取（或生成）本设备的会话标识。
 *
 * **必须持久化**：会话记忆注入按它取最近 N 轮，
 * 每刷新一次页面就换一个 ID 的话，多轮对话在用户看来是「失忆」的。
 */
export function getSessionId(): string {
  try {
    const existing = localStorage.getItem(SESSION_KEY)
    if (existing) return existing
    const fresh = `web-${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`
    localStorage.setItem(SESSION_KEY, fresh)
    return fresh
  } catch {
    return `web-${Date.now().toString(36)}`
  }
}
/** 开一个新会话（用户主动「清空对话」时用）。 */
export function rotateSessionId(): string {
  try {
    localStorage.removeItem(SESSION_KEY)
  } catch {
    /* ignore */
  }
  return getSessionId()
}
export function getActivePetId(): string | null {
  try {
    return localStorage.getItem(PET_KEY)
  } catch {
    return null
  }
}
export function setActivePetId(petId: string | null): void {
  try {
    if (petId) localStorage.setItem(PET_KEY, petId)
    else localStorage.removeItem(PET_KEY)
  } catch {
    /* ignore */
  }
}
// ─────────────────────────────────────────────────────────────
// 错误
// ─────────────────────────────────────────────────────────────
export type ApiErrorKind = 'network' | 'auth' | 'validation' | 'notfound' | 'server' | 'unknown'
export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number
  readonly code: string | null
  constructor(kind: ApiErrorKind, message: string, status: number, code: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.code = code
  }
}
/** 把两种错误体归一成一句人话。 */
function humanize(body: unknown, status: number): { message: string; code: string | null } {
  if (typeof body === 'string' && body.trim()) return { message: body, code: null }
  if (body && typeof body === 'object') {
    const detail = (body as Record<string, unknown>).detail
    // 形状 A：{detail: {code, message}}
    if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
      const d = detail as Record<string, unknown>
      return {
        message: typeof d.message === 'string' ? d.message : `请求失败（HTTP ${status}）`,
        code: typeof d.code === 'string' ? d.code : null,
      }
    }
    // 形状 B：FastAPI 校验数组 [{loc, msg, type}]
    if (Array.isArray(detail)) {
      const parts = detail
        .map((item) => {
          const d = item as Record<string, unknown>
          const loc = Array.isArray(d.loc) ? d.loc.filter((x) => x !== 'body').join('.') : ''
          const msg = typeof d.msg === 'string' ? d.msg : '校验失败'
          return loc ? `${loc}: ${msg}` : msg
        })
        .filter(Boolean)
      return { message: parts.join('；') || '请求参数不合法', code: 'VALIDATION_FAILED' }
    }
    if (typeof detail === 'string') return { message: detail, code: null }
  }
  return { message: `请求失败（HTTP ${status}）`, code: null }
}
function classify(status: number): ApiErrorKind {
  if (status === 401) return 'auth'
  if (status === 404) return 'notfound'
  if (status === 422) return 'validation'
  if (status >= 500) return 'server'
  return 'unknown'
}
// ─────────────────────────────────────────────────────────────
// 请求
// ─────────────────────────────────────────────────────────────
interface RequestOptions {
  method?: string
  body?: unknown
  /** 是否需要 Bearer token。`/healthz` 与 dev-login 不需要。 */
  auth?: boolean
  signal?: AbortSignal
}
async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, auth = true, signal } = options
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'X-Session-Id': getSessionId(),
  }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (auth) {
    const token = getToken()
    if (token) headers.Authorization = `Bearer ${token}`
  }
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers,
      signal,
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError('network', '连不上服务器。请检查网络或后端是否在运行。', 0)
  }
  // 后端在**所有**响应回传 session/trace；跟随服务端的会话标识，
  // 避免「客户端以为的会话」与「服务端记录的会话」分叉。
  const echoed = response.headers.get('X-Session-Id')
  if (echoed) {
    try {
      localStorage.setItem(SESSION_KEY, echoed)
    } catch {
      /* ignore */
    }
  }
  if (response.status === 204) return undefined as T
  const text = await response.text()
  let parsed: unknown = null
  if (text) {
    try {
      parsed = JSON.parse(text)
    } catch {
      parsed = text
    }
  }
  if (!response.ok) {
    const { message, code } = humanize(parsed, response.status)
    throw new ApiError(classify(response.status), message, response.status, code)
  }
  return parsed as T
}
// ─────────────────────────────────────────────────────────────
// 端点
// ─────────────────────────────────────────────────────────────
export const api = {
  async healthz(): Promise<HealthzResponse> {
    return request<HealthzResponse>('/healthz', { auth: false })
  },
  async devLogin(userId: string): Promise<DevLoginResponse> {
    return request<DevLoginResponse>('/v1/auth/dev-login', {
      method: 'POST',
      auth: false,
      body: { user_id: userId },
    })
  },
  // ── 宠物 ──
  async listPets(): Promise<PetListResponse> {
    return request<PetListResponse>('/v1/pets')
  },
  async createPet(name: string, breed?: string): Promise<{ pet_id: string; name: string }> {
    return request('/v1/pets', {
      method: 'POST',
      body: { name, ...(breed ? { breed } : {}) },
    })
  },
  async getProfile(petId: string): Promise<PetProfile> {
    return request<PetProfile>(`/v1/pets/${encodeURIComponent(petId)}/profile`)
  },
  /**
   * 建/更新档案。
   *
   * `confirm=false` 时后端**只回草案、不落库** —— 档案必须经用户确认（DESIGN §2.3）。
   */
  async buildProfile(
    petId: string,
    imageUrls: string[],
    confirm = false,
  ): Promise<ProfileDraft | PetProfile> {
    return request(`/v1/pets/${encodeURIComponent(petId)}/profile`, {
      method: 'POST',
      body: { image_urls: imageUrls, confirm },
    })
  },
  // ── 对话 / 解释 ──
  async chat(
    petId: string,
    payload: {
      text?: string
      audio_url?: string
      audio_kind?: AudioKind
      image_urls?: string[]
      scene_description?: string
    },
    signal?: AbortSignal,
  ): Promise<TurnResponse> {
    return request<TurnResponse>(`/v1/chat?pet_id=${encodeURIComponent(petId)}`, {
      method: 'POST',
      body: payload,
      signal,
    })
  },
  /** 录叫声 → 解释。返回的 `interpretation_id` 是主人标注的入口。 */
  async interpret(
    petId: string,
    audioUrl: string,
    sceneDescription?: string,
    signal?: AbortSignal,
  ): Promise<TurnResponse> {
    return request<TurnResponse>(`/v1/interpret?pet_id=${encodeURIComponent(petId)}`, {
      method: 'POST',
      body: {
        audio_url: audioUrl,
        ...(sceneDescription ? { scene_description: sceneDescription } : {}),
      },
      signal,
    })
  },
  /** 主人标注。**这是案例推理的燃料** —— 没有它，系统学不到任何东西。 */
  async labelMeow(
    petId: string,
    payload: {
      interpretation_id: string
      context: ContextLabel
      actions?: BehaviorAction[]
      resolution?: string
    },
  ): Promise<MeowRecord> {
    return request<MeowRecord>(`/v1/pets/${encodeURIComponent(petId)}/meow-records`, {
      method: 'POST',
      body: payload,
    })
  },
  async listMeowRecords(
    petId: string,
    onlyConfirmed = true,
  ): Promise<MeowRecordListResponse> {
    return request<MeowRecordListResponse>(
      `/v1/pets/${encodeURIComponent(petId)}/meow-records?only_confirmed=${onlyConfirmed}`,
    )
  },
  // ── 记忆 / 故事 / 健康 ──
  async listMemories(petId: string, sessionId?: string): Promise<MemoryListResponse> {
    const q = new URLSearchParams({ pet_id: petId })
    if (sessionId) q.set('session_id', sessionId)
    return request<MemoryListResponse>(`/v1/memories?${q.toString()}`)
  },
  // ── 日记本体 ──
  /** 上传一张照片 / 一段视频，返回可引用的 URL。
   *
   * 用 base64 而不是 multipart：少一个依赖、少一种内容类型（见后端 docstring）。
   * 代价是体积约 +33%。 */
  async uploadMedia(dataUrl: string, filename?: string): Promise<MediaUploadResponse> {
    return request<MediaUploadResponse>('/v1/media', {
      method: 'POST',
      body: { data_base64: dataUrl, ...(filename ? { filename } : {}) },
    })
  },
  /** 记录一个瞬间。**只要一张照片**，一句话可选。
   *
   * 返回里带 `pet_says` —— 记完立刻有句话，那是这个动作的情绪回报。 */
  async recordMoment(
    petId: string,
    mediaUrl: string,
    note?: string,
  ): Promise<RecordMomentResponse> {
    return request<RecordMomentResponse>(`/v1/pets/${encodeURIComponent(petId)}/moments`, {
      method: 'POST',
      body: { media_url: mediaUrl, ...(note ? { note } : {}) },
    })
  },
  /** 时间线。**倒序**（最新在前）。 */
  async timeline(
    petId: string,
    opts: { days?: number; limit?: number; scene?: string } = {},
  ): Promise<TimelineResponse> {
    const q = new URLSearchParams()
    if (opts.days) q.set('days', String(opts.days))
    if (opts.limit) q.set('limit', String(opts.limit))
    if (opts.scene) q.set('scene', opts.scene)
    const suffix = q.toString() ? `?${q.toString()}` : ''
    return request<TimelineResponse>(
      `/v1/pets/${encodeURIComponent(petId)}/timeline${suffix}`,
    )
  },
  async story(petId: string, day: string): Promise<StoryResponse> {
    return request<StoryResponse>(
      `/v1/pets/${encodeURIComponent(petId)}/story?day=${encodeURIComponent(day)}`,
    )
  },
  async health(petId: string): Promise<HealthResponse> {
    return request<HealthResponse>(`/v1/pets/${encodeURIComponent(petId)}/health`)
  },
  async recordSignal(
    petId: string,
    payload: { signal: string; value?: number; note?: string; consent_version?: string },
  ): Promise<unknown> {
    return request(`/v1/pets/${encodeURIComponent(petId)}/health/signals`, {
      method: 'POST',
      body: payload,
    })
  },
}
export type { PetBrief }
