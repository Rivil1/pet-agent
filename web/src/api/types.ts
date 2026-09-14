/**
 * 后端契约的 TypeScript 镜像。
 *
 * 这些类型**手工对齐** `app/schemas/`（后端是 Pydantic，不产出 TS）。
 * 对齐点集中在这一个文件里，是为了让「后端改了字段」这件事只有一处要改 ——
 * 分散在各页面里会让漂移静默发生。
 */

// ─────────────────────────────────────────────────────────────
// 枚举：与 app/schemas 一一对应
// ─────────────────────────────────────────────────────────────

/** 证据模式。后端按可用数据 fail-closed 地选择，前端只展示不推断。 */
export type EvidenceMode =
  | 'acoustic_plus_history'
  | 'case_based'
  | 'measured_only'
  | 'text_only'

/** 证据种类。**这是整个产品的可信度分层**，UI 必须原样呈现。 */
export type EvidenceKind = 'measured' | 'retrieved' | 'prior' | 'observed'

/** 主人的情境标注。固定词表 —— 自由文本无法可靠匹配。 */
export type ContextLabel =
  | 'food_waiting'
  | 'door_attention'
  | 'affection_brushing'
  | 'isolation_distress'
  | 'greeting'
  | 'other'

/** 主人可**观察**到的动作。刻意不含意图 —— 意图是推断，动作是观察。 */
export type BehaviorAction =
  | 'scratch_door'
  | 'pacing'
  | 'rub_leg'
  | 'tail_up'
  | 'near_food_bowl'
  | 'look_at_door'
  | 'hiding'
  | 'purring'
  | 'belly_up'
  | 'approach_human'
  | 'avoid_contact'
  | 'grooming'
  | 'arched_back'
  | 'other'

export type AudioKind = 'cat_meow' | 'user_voice'

/**
 * 健康分级。
 *
 * ⚠️ 取值来自后端 `UrgencyLevel`（`app/health/redflags.py`）。
 * **是 `L1`/`L2`/`L3` 与两个大写常量，不是 `monitor`/`urgent` 这类自造名**。
 * 曾经按「听起来合理」的名字写过一版，结果五个分支全部落空 ——
 * 页面永远显示「没有拿到评估结果」，而没有任何测试变红。
 */
export type HealthLevel =
  | 'INSUFFICIENT_DATA'
  | 'NO_DEVIATION_DETECTED'
  | 'L1'
  | 'L2'
  | 'L3'

// ─────────────────────────────────────────────────────────────
// 宠物
// ─────────────────────────────────────────────────────────────

export interface PetBrief {
  pet_id: string
  name: string
  species: string
  breed: string | null
  has_profile: boolean
  must_keep_features: string[]
  fur_color: string | null
  eye_color: string | null
  created_at: string
}

export interface PetListResponse {
  count: number
  pets: PetBrief[]
}

export interface VisualProfile {
  fur_color: string | null
  fur_length: string | null
  eye_color: string | null
  body_shape: string | null
  face_shape: string | null
  distinctive_features: string[]
}

export interface PetProfile {
  pet_id: string
  user_id: string
  name: string
  species: string
  breed: string | null
  visual: VisualProfile
  /** 多图取交集得到的硬特征。**这些是身份锚点**，不得被后续描述覆盖。 */
  must_keep_features: string[]
  observed_but_unstable: string[]
  traits: string[]
  appearance_embedding_id: string | null
  identity_prompt: string
  created_at: string
  updated_at: string
}

/** 建档案的草案。`confirm=false` 时后端**不落库**，只回草案。 */
export interface ProfileDraft {
  pet_id?: string
  visual?: VisualProfile
  must_keep_features?: string[]
  observed_but_unstable?: string[]
  coverage?: number
  coverage_note?: string
  images_analyzed?: number
  images_failed?: number
  [key: string]: unknown
}

// ─────────────────────────────────────────────────────────────
// 对话 / 解释
// ─────────────────────────────────────────────────────────────

export interface TraceItem {
  node: string
  latency_ms: number
  decision: string
  degraded: boolean
}

export interface AcousticFeatures {
  duration: number
  f0_mean: number
  f0_range: number
  f0_slope: number
  call_rate: number
  ici_mean: number
  rms_mean: number
  roughness: number
  estimated_snr_db?: number | null
  quality?: string
  /** 本次**测不出**的特征名。推理侧跳过它们 —— 不是零。 */
  unavailable: string[]
}

export interface EvidenceItem {
  kind: EvidenceKind
  statement: string
  source: string
  value?: unknown
  reference?: unknown
  log_odds_contribution?: unknown
}

export interface IntentCandidate {
  context: ContextLabel
  posterior: number | null
  display: string
  log_odds: number | null
  matched_count: number
}

export interface CaseMatch {
  record_id: string
  similarity: number
  context: ContextLabel
  actions: BehaviorAction[]
  resolution: string | null
  recorded_at: string | null
}

export interface BehaviorInterpretation {
  evidence_mode: EvidenceMode
  acoustic_features: AcousticFeatures | null
  candidates: IntentCandidate[]
  evidence: EvidenceItem[]
  similar_cases: CaseMatch[]
  case_total: number
  individualization: number
  sample_count: number
  suggested_observation: string
  limitations: string
  prior_version: string | null
}

export interface TurnResponse {
  final_response: string
  intent: string
  route_confidence: number
  session_id: string
  trace_id: string
  langsmith_run_id: string | null
  degraded: boolean
  degraded_notice: string | null
  trace: TraceItem[]
  retrieved_count: number
  written_memory_ids: string[]
  provider_mode: string
  mock_notice: string | null
  suggested_actions: string[]
  /** 主人标注这个解释所需的 ID。**没有它就没法标注**（没有声学特征时为空）。 */
  interpretation_id: string | null
  interpretation: BehaviorInterpretation | null
}

// ─────────────────────────────────────────────────────────────
// 记忆 / 叫声记录
// ─────────────────────────────────────────────────────────────

export interface MemoryItem {
  event: {
    memory_id?: string
    content: string
    event_type?: string
    source?: string
    status?: string
    occurred_at?: string
    valid_from?: string | null
    valid_to?: string | null
    support_count?: number
    [key: string]: unknown
  }
  score: number
  retrieval_source: string
  matched_on: string
}

export interface MemoryListResponse {
  count: number
  memories: MemoryItem[]
}

export interface MeowRecord {
  record_id: string
  pet_id: string
  context: ContextLabel
  actions: BehaviorAction[]
  resolution: string | null
  status: string
  recorded_at: string
  similarity?: number
  [key: string]: unknown
}

export interface MeowRecordListResponse {
  count: number
  records: MeowRecord[]
}

// ─────────────────────────────────────────────────────────────
// 故事 / 健康
// ─────────────────────────────────────────────────────────────

export interface StoryBeat {
  caption: string
  photo_ref: string | null
  style: string
  time_of_day: string
}

export interface StoryResponse {
  date: string
  title: string
  story_text: string
  beats: StoryBeat[]
  excluded_count: number
  health_notice: string | null
  disclaimer: string
  digest_notes: string[]
  rejection_rate: number
}

export interface RedFlag {
  id?: string
  signal?: string
  severity?: string
  message?: string
  [key: string]: unknown
}

export interface HealthResponse {
  level: HealthLevel
  /** 数据覆盖度。**低于门限时禁止输出「未发现异常」**（决策 D12）。 */
  coverage: number
  coverage_note: string
  recommendation: string
  disclaimer: string
  must_not_be_read_as: string
  rule_version: string | null
  red_flags: RedFlag[]
  signals_missing_count: number
  record_count: number
}

export interface HealthSignalResponse {
  record_id: string
  [key: string]: unknown
}

// ─────────────────────────────────────────────────────────────
// 系统
// ─────────────────────────────────────────────────────────────

export interface HealthzResponse {
  status: string
  providers: Record<string, string>
  observability: Record<string, string>
  /** `enabled` 时前端才显示「开发登录」入口。 */
  dev_login: 'enabled' | 'disabled'
}

export interface DevLoginResponse {
  token: string
  user_id: string
  expires_in: number
}

// ─────────────────────────────────────────────────────────────
// 日记本体：瞬间与时间线（docs/11 §2.1–2.2）
// ─────────────────────────────────────────────────────────────

/** 场景标签。固定词表 —— 自由文本无法稳定聚合。 */
export type MomentScene =
  | 'sleeping'
  | 'playing'
  | 'eating'
  | 'window'
  | 'with_human'
  | 'grooming'
  | 'other'

export interface Moment {
  moment_id: string
  media_url: string
  /** 用户可选的一句话。**为空是正常的，不是缺失。** */
  note: string | null
  scene: MomentScene
  scene_display: string
  captured_at: string
}

export interface TimelineResponse {
  count: number
  days: number
  scene: string | null
  moments: Moment[]
  /** 场景分布。只统计出现过的 —— 补零会让「0 次」与「没这个场景」混淆。 */
  scene_counts: Record<string, number>
}

export interface RecordMomentResponse {
  moment: Moment
  /** 猫对这一刻的回应（模板生成，确定性） */
  pet_says: string
}

export interface MediaUploadResponse {
  url: string
  kind: 'image' | 'video' | 'audio'
  bytes: number
}
