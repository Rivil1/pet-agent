import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, getActivePetId, getToken, setActivePetId, setToken } from './api/client'
import type { HealthzResponse, PetBrief } from './api/types'
import { AppShell, type TabKey, TABS } from './components/AppShell'
import { Login } from './components/Login'
import { Spinner } from './components/ui'
import { PetsPage } from './pages/PetsPage'
import { ChatPage } from './pages/ChatPage'
import { ProfilePage } from './pages/ProfilePage'
import { StoryPage } from './pages/StoryPage'
import { HealthPage } from './pages/HealthPage'

/** 从 hash 读当前标签，使刷新后停在原处、链接可分享。 */
function readTab(): TabKey {
  const raw = window.location.hash.replace(/^#\/?/, '')
  const found = TABS.find((t) => t.key === raw)
  return found ? found.key : 'chat'
}

export function App() {
  const [token, setTokenState] = useState<string | null>(() => getToken())
  const [health, setHealth] = useState<HealthzResponse | null>(null)
  const [pets, setPets] = useState<PetBrief[]>([])
  const [activePetId, setActivePetIdState] = useState<string | null>(() => getActivePetId())
  const [tab, setTab] = useState<TabKey>(() => readTab())
  const [booting, setBooting] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  // ── 标签与 hash 同步 ──
  useEffect(() => {
    const onHash = () => setTab(readTab())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const goTab = useCallback((t: TabKey) => {
    window.location.hash = `#/${t}`
    setTab(t)
  }, [])

  // ── 探活（provider 模式常驻顶栏） ──
  useEffect(() => {
    let alive = true
    api
      .healthz()
      .then((h) => alive && setHealth(h))
      .catch(() => {
        /* 顶栏显示「未知」即可，不打断使用 */
      })
    return () => {
      alive = false
    }
  }, [token])

  // ── 拉宠物列表 ──
  const loadPets = useCallback(
    async (opts: { silent?: boolean } = {}) => {
      if (!getToken()) {
        setBooting(false)
        return
      }
      if (!opts.silent) setBooting(true)
      setLoadError(null)
      try {
        const res = await api.listPets()
        setPets(res.pets)

        // 选用：优先保留已选的；否则用第一只
        setActivePetIdState((current) => {
          const stillThere = current && res.pets.some((p) => p.pet_id === current)
          const next = stillThere ? current : (res.pets[0]?.pet_id ?? null)
          setActivePetId(next)
          return next
        })
      } catch (e) {
        if (e instanceof ApiError && e.kind === 'auth') {
          // token 过期或无效 —— 退回登录，而不是留在一个永远失败的界面上
          setToken(null)
          setTokenState(null)
        } else {
          setLoadError(e instanceof ApiError ? e.message : '加载失败')
        }
      } finally {
        setBooting(false)
      }
    },
    [],
  )

  useEffect(() => {
    void loadPets()
  }, [loadPets, token])

  function handleLogin(t: string) {
    setToken(t)
    setTokenState(t)
  }

  function handleLogout() {
    setToken(null)
    setTokenState(null)
    setActivePetId(null)
    setActivePetIdState(null)
    setPets([])
    window.location.hash = ''
  }

  function handleSelectPet(pet: PetBrief) {
    setActivePetIdState(pet.pet_id)
    setActivePetId(pet.pet_id)
    goTab('chat')
  }

  function handleCreated(pet: PetBrief) {
    setPets((prev) => [...prev, pet])
    setActivePetIdState(pet.pet_id)
    setActivePetId(pet.pet_id)
  }

  // ── 未登录 ──
  if (!token) {
    return <Login onDone={handleLogin} />
  }

  // ── 启动中 ──
  if (booting) {
    return (
      <div className="paper-grain paper-vignette flex h-full items-center justify-center">
        <div className="relative z-10 flex items-center gap-2.5 text-ink-faint">
          <Spinner />
          <span className="text-[13px]">正在翻开笔记…</span>
        </div>
      </div>
    )
  }

  const activePet = pets.find((p) => p.pet_id === activePetId) ?? null

  // 没有宠物时，除「猫」以外的标签都无意义 —— 直接指向建立宠物。
  // 这里按 `activePet === null` 直接分支，而不是先算一个 `needsPet` 布尔值：
  // 布尔值不会让 TS 收窄 `activePet`，于是一个「已经判过空」的地方
  // 仍然要写 `!` —— 而那正是真正漏判时最不会被发现的地方。
  function body() {
    if (activePet === null || tab === 'pets') {
      return (
        <PetsPage
          pets={pets}
          activePetId={activePetId}
          onSelect={handleSelectPet}
          onCreated={handleCreated}
          loading={false}
        />
      )
    }
    return (
      <PetScoped
        key={activePet.pet_id}
        tab={tab}
        pet={activePet}
        onMemoryChanged={() => void loadPets({ silent: true })}
      />
    )
  }

  return (
    <AppShell
      activePet={activePet}
      petCount={pets.length}
      providerMode={health?.providers?.mode ?? null}
      tab={tab}
      onTab={goTab}
      onLogout={handleLogout}
    >
      {loadError && (
        <div className="mb-4 rounded-[3px] border border-brick/45 bg-brick/[0.06] px-3.5 py-3 text-[12.5px] text-ink-soft">
          {loadError}
        </div>
      )}
      {body()}
    </AppShell>
  )
}

/**
 * 按标签渲染宠物相关页面。
 *
 * `key={pet_id}` 在上层 —— 切换宠物时整棵子树重建。
 * 这是刻意的：用 effect 手动清空各页面的局部状态（草稿、消息流、展开项）
 * 一定会漏掉某一处，而漏掉的那一处就是数据串味。
 */
function PetScoped({
  tab,
  pet,
  onMemoryChanged,
}: {
  tab: TabKey
  pet: PetBrief
  onMemoryChanged: () => void
}) {
  switch (tab) {
    case 'chat':
      return <ChatPage petId={pet.pet_id} petName={pet.name} onMemoryChanged={onMemoryChanged} />
    case 'profile':
      return <ProfilePage petId={pet.pet_id} petName={pet.name} />
    case 'story':
      return <StoryPage petId={pet.pet_id} petName={pet.name} />
    case 'health':
      return <HealthPage petId={pet.pet_id} petName={pet.name} />
    default:
      return null
  }
}
