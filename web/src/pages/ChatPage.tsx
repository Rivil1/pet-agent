import { useEffect, useRef, useState } from 'react'
import { api, ApiError, getSessionId, rotateSessionId } from '../api/client'
import type { MeowRecord, TurnResponse } from '../api/types'
import { Button, Card, Input, Notice, Spinner } from '../components/ui'
import { IconClose, IconRefresh, IconSend, IconWave } from '../components/icons'
import { cx } from '../lib/utils'
import { EvidenceCard } from '../components/EvidenceCard'
import { LabelSheet } from '../components/LabelSheet'
import { TraceStrip } from '../components/TraceStrip'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  turn?: TurnResponse
}

/**
 * 对话页 —— 陪伴 + 叫声解释 + 标注学习的合流处。
 *
 * ## 为什么标注入口长在对话里
 *
 * 「主人标注」是案例推理唯一的燃料，而它只在**刚看完解释**的那一刻才有意义 ——
 * 那时主人脑子里还有当时的情形。把标注做成独立页面，等于让用户走完解释后
 * 再去另一个地方凭记忆重建情境；这条路径实际不会被走，
 * 而这正是 D43 之前的状态：**「学习主人的经验」没有输入路径**。
 *
 * 所以标注按钮贴着解释，且**预填**模型观察到的东西 —— 主人只需确认或改一下。
 */
export function ChatPage({
  petId,
  petName,
  onMemoryChanged,
}: {
  petId: string
  petName: string
  onMemoryChanged: () => void
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [audioUrl, setAudioUrl] = useState('')
  const [showAudio, setShowAudio] = useState(false)
  const [scene, setScene] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /** 已标注的解释 → 标注记录。**按 interpretation_id 索引**，
   *  这样同一条解释无论渲染几次都只有一个标注状态。 */
  const [labels, setLabels] = useState<Record<string, MeowRecord>>({})
  const [labelTarget, setLabelTarget] = useState<TurnResponse | null>(null)

  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // 切换宠物 = 换上下文。对话流必须清空 ——
  // 否则会出现「上一只猫的话出现在这只猫的会话里」这种最致命的串味。
  useEffect(() => {
    setMessages([])
    setLabels({})
    setError(null)
    setLabelTarget(null)
    rotateSessionId()
  }, [petId])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, busy])

  async function send() {
    const text = input.trim()
    const audio = audioUrl.trim()
    if ((!text && !audio) || busy) return

    const label = audio ? scene.trim() || text || '（未描述情境）' : ''
    setMessages((prev) => [
      ...prev,
      {
        id: `u-${Date.now()}-${prev.length}`,
        role: 'user',
        text: audio ? `🎙 ${label}` : text,
      },
    ])
    setInput('')
    setError(null)
    setBusy(true)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const turn = audio
        ? await api.interpret(petId, audio, scene.trim() || undefined, controller.signal)
        : await api.chat(petId, { text }, controller.signal)

      setMessages((prev) => [
        ...prev,
        {
          id: `a-${Date.now()}-${prev.length}`,
          role: 'assistant',
          text: turn.final_response || '（没有回复）',
          turn,
        },
      ])

      if (audio) {
        setAudioUrl('')
        setScene('')
        setShowAudio(false)
      }
      if (turn.written_memory_ids.length > 0) onMemoryChanged()
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setError(e instanceof ApiError ? e.message : '请求失败')
    } finally {
      setBusy(false)
      abortRef.current = null
    }
  }

  function stop() {
    abortRef.current?.abort()
    setBusy(false)
  }

  function newSession() {
    setMessages([])
    setLabels({})
    rotateSessionId()
    setError(null)
  }

  const labelTargetId = labelTarget?.interpretation_id ?? null

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* 会话头 */}
      <div className="mb-3 flex items-center justify-between gap-2">
        <p className="min-w-0 truncate text-[12px] text-ink-faint">
          与 <span className="font-semibold text-ink-soft">{petName}</span> 的记录
          <span className="ml-2 font-mono text-[10px] opacity-55">
            {getSessionId().slice(0, 11)}
          </span>
        </p>
        <button
          onClick={newSession}
          disabled={messages.length === 0}
          className="flex shrink-0 items-center gap-1 text-[12px] font-semibold text-ink-faint transition hover:text-persimmon disabled:opacity-35"
        >
          <IconRefresh width={13} height={13} />
          新会话
        </button>
      </div>

      {error && (
        <div className="mb-3">
          <Notice tone="danger">{error}</Notice>
        </div>
      )}

      {/* 消息流 */}
      <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto pb-4">
        {messages.length === 0 && (
          <div className="animate-rise-in px-2 py-10 text-center">
            <div className="mx-auto mb-4 h-11 w-11 rotate-[-3deg] rounded-[3px] border-[1.5px] border-dashed border-rule" />
            <p className="font-display text-[16px] font-semibold text-ink">
              说说 {petName} 今天怎么样
            </p>
            <p className="mx-auto mt-2 max-w-[32ch] text-[12.5px] leading-relaxed text-ink-faint">
              也可以录一段叫声让它解释 —— 然后由你告诉它，猜得对不对。
            </p>
          </div>
        )}

        {messages.map((m) =>
          m.role === 'user' ? (
            <div key={m.id} className="flex animate-ink-in justify-end">
              <div className="max-w-[82%] rounded-[4px] rounded-br-[1px] border border-persimmon/35 bg-persimmon/[0.07] px-3.5 py-2.5">
                <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed text-ink">
                  {m.text}
                </p>
              </div>
            </div>
          ) : (
            <div key={m.id} className="animate-rise-in space-y-3">
              <Card binding className="rounded-bl-[1px] py-3.5">
                <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed text-ink">
                  {m.text}
                </p>
                {m.turn?.degraded && m.turn.degraded_notice && (
                  // 降级提示必须可见。
                  //
                  // 检索不可用时，模型会自信地说出「没有相关记录」——
                  // 而用户无法从这句话分辨「真的没记录」与「根本没查成」。
                  // 不显示这条，系统就在为自己没做过的事打包票。
                  <div className="mt-2.5 rounded-[2px] border border-l-[3px] border-rule border-l-ochre/60 bg-ochre/[0.05] px-2.5 py-2">
                    <p className="text-[11px] font-bold uppercase tracking-[0.1em] text-ochre">
                      部分能力未生效
                    </p>
                    <p className="mt-1 text-[11.5px] leading-relaxed text-ink-soft">
                      {m.turn.degraded_notice}
                    </p>
                  </div>
                )}
                {m.turn?.mock_notice && (
                  <p className="mt-2.5 text-[11.5px] leading-relaxed text-ochre">
                    {m.turn.mock_notice}
                  </p>
                )}
              </Card>

              {m.turn?.interpretation && (
                <EvidenceCard
                  interp={m.turn.interpretation}
                  label={m.turn.interpretation_id ? labels[m.turn.interpretation_id] : undefined}
                  onLabel={
                    m.turn.interpretation_id ? () => setLabelTarget(m.turn ?? null) : undefined
                  }
                />
              )}

              {m.turn && m.turn.trace.length > 0 && <TraceStrip turn={m.turn} />}
            </div>
          ),
        )}

        {busy && (
          <div className="flex animate-ink-in items-center gap-2.5 text-ink-faint">
            <Spinner />
            <span className="text-[12.5px]">
              {showAudio ? '正在提取声学特征…' : '正在整理…'}
            </span>
          </div>
        )}
      </div>

      {/* 输入区 */}
      <div className="shrink-0 border-t border-rule pt-3">
        {showAudio && (
          <div className="mb-2.5 animate-rise-in space-y-2 rounded-[3px] border border-rule bg-paper-raised p-3">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-[12px] font-semibold text-ink-soft">
                <IconWave width={15} height={15} />
                叫声解释
              </span>
              <button
                onClick={() => {
                  setShowAudio(false)
                  setAudioUrl('')
                  setScene('')
                }}
                className="text-ink-faint transition hover:text-brick"
                aria-label="收起"
              >
                <IconClose width={15} height={15} />
              </button>
            </div>
            <Input
              value={audioUrl}
              onChange={(e) => setAudioUrl(e.target.value)}
              placeholder="音频地址 https://…/meow.wav"
              spellCheck={false}
            />
            <Input
              value={scene}
              onChange={(e) => setScene(e.target.value)}
              placeholder="当时的情形（可选）：它在门口叫，一直看着门"
            />
            <p className="text-[11px] leading-relaxed text-ink-faint">
              声学特征由服务端从音频直接量出，不由客户端提交 ——
              这是「可测量」与「模型描述」的分界。
            </p>
          </div>
        )}

        <div className="flex items-end gap-2">
          <button
            onClick={() => setShowAudio((v) => !v)}
            className={cx(
              'shrink-0 rounded-[3px] border p-2.5 transition',
              showAudio
                ? 'border-persimmon/55 bg-persimmon/[0.08] text-persimmon'
                : 'border-rule bg-paper-raised text-ink-faint hover:text-ink-soft',
            )}
            title="录一段叫声让它解释"
            aria-label="叫声解释"
          >
            <IconWave width={18} height={18} />
          </button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
            rows={1}
            placeholder={`说说 ${petName}…`}
            className="field max-h-28 min-h-[42px] flex-1 resize-none py-2.5 leading-relaxed"
          />

          {busy ? (
            <Button variant="ghost" className="shrink-0 px-3" onClick={stop}>
              停
            </Button>
          ) : (
            <Button
              className="shrink-0 px-3.5"
              disabled={!input.trim() && !audioUrl.trim()}
              onClick={send}
              aria-label="发送"
            >
              <IconSend width={18} height={18} />
            </Button>
          )}
        </div>
      </div>

      {labelTarget && labelTargetId && (
        <LabelSheet
          petId={petId}
          interpretationId={labelTargetId}
          suggestedActions={labelTarget.suggested_actions}
          onClose={() => setLabelTarget(null)}
          onLabelled={(rec) => {
            setLabels((prev) => ({ ...prev, [labelTargetId]: rec }))
            setLabelTarget(null)
            onMemoryChanged()
          }}
        />
      )}
    </div>
  )
}
