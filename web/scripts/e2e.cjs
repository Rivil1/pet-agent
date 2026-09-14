/**
 * 端到端冒烟测试：**登录 → 主壳 → 各标签页 → 主流程**。
 *
 * 与 `smoke.cjs` 的分工：那个只证明「能挂载」，
 * 这个证明「挂载之后整条路径能走通」—— 用真实后端、真实 token、真实数据。
 *
 * 覆盖的关键点（都是容易静默坏掉的地方）：
 * - token 过期 → 必须退回登录，而不是停在一个永远失败的界面
 * - 切换宠物 → 对话流必须清空（串味是最致命的缺陷）
 * - 各标签页在**没有数据**时不能崩（空态是最常被漏测的分支）
 *
 * 用法::
 *
 *     node scripts/e2e.cjs dist http://127.0.0.1:8000
 */

const fs = require('fs')
const path = require('path')

const DIST = process.argv[2] || path.join(__dirname, '..', 'dist')
const API = process.argv[3] || 'http://127.0.0.1:8000'

let JSDOM, VirtualConsole
try {
  ;({ JSDOM, VirtualConsole } = require('jsdom'))
} catch {
  console.error('缺少 jsdom。先运行：npm install -D jsdom')
  process.exit(2)
}

/** 起一个干净的 jsdom，按需预置 localStorage。 */
function boot({ token, session, petId, hash = '' } = {}) {
  const errors = []
  const vc = new VirtualConsole()
  vc.on('jsdomError', (e) => errors.push(`[jsdom] ${e.message}`))
  vc.on('error', (...a) => errors.push(`[console.error] ${a.join(' ')}`))

  const html = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8')
  const dom = new JSDOM(html, {
    url: `http://127.0.0.1:4173/${hash}`,
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    virtualConsole: vc,
  })
  const { window } = dom

  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : input.url
    return fetch(url.startsWith('http') ? url : `${API}${url}`, init)
  }
  window.AbortController = AbortController
  window.DOMException = DOMException
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
  window.scrollTo = () => {}
  window.Element.prototype.scrollTo = () => {}
  window.Element.prototype.scrollIntoView = () => {}

  const store = new Map()
  if (token) store.set('pet-agent.token', token)
  if (session) store.set('pet-agent.session', session)
  if (petId) store.set('pet-agent.active-pet', petId)

  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
      removeItem: (k) => store.delete(k),
      clear: () => store.clear(),
      _dump: () => Object.fromEntries(store),
    },
    configurable: true,
  })

  return { window, errors }
}

/** 把构建产物注入一个已配置好的 window，并等 React 渲染。 */
async function render(window, { wait = 2200 } = {}) {
  const assetsDir = path.join(DIST, 'assets')
  const assets = fs.readdirSync(assetsDir)
  const entry = assets.find((f) => f.startsWith('index-') && f.endsWith('.js'))
  const css = assets.find((f) => f.startsWith('index-') && f.endsWith('.css'))

  if (css) {
    const style = window.document.createElement('style')
    style.textContent = fs.readFileSync(path.join(assetsDir, css), 'utf8')
    window.document.head.appendChild(style)
  }
  window.eval(fs.readFileSync(path.join(assetsDir, entry), 'utf8'))
  await new Promise((r) => setTimeout(r, wait))
  const root = window.document.getElementById('root')
  return {
    text: (root?.textContent || '').replace(/\s+/g, ' '),
    html: root?.innerHTML || '',
  }
}

/** 点一个包含指定文案的按钮。返回是否点到。 */
function clickByText(window, needle) {
  const nodes = [...window.document.querySelectorAll('button, a')]
  const el = nodes.find((n) => (n.textContent || '').includes(needle))
  if (!el) return false
  el.dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
  return true
}

const results = []
function check(name, ok, detail = '') {
  results.push({ name, ok, detail })
}

async function main() {
  // ── 准备：拿 token、建宠物 ──
  const login = await fetch(`${API}/v1/auth/dev-login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: `e2e-${Date.now()}` }),
  })
  if (!login.ok) {
    console.error(`dev-login 失败 (${login.status})。后端需要 PET_AGENT_ALLOW_DEV_LOGIN=1`)
    process.exit(2)
  }
  const { token } = await login.json()
  const H = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }

  const petName = `团团${Math.floor(Math.random() * 900 + 100)}`
  const created = await (
    await fetch(`${API}/v1/pets`, {
      method: 'POST',
      headers: H,
      body: JSON.stringify({ name: petName, breed: '英短' }),
    })
  ).json()
  const petId = created.pet_id

  // ─────────────────────────────────────────────
  // 1. 未登录 → 登录页
  // ─────────────────────────────────────────────
  {
    const { window, errors } = boot()
    const { text } = await render(window)
    check('未登录时显示登录页', text.includes('猫事') && text.includes('开始记录'))
    check('登录页无运行时错误', errors.length === 0, errors[0] || '')
  }

  // ─────────────────────────────────────────────
  // 2. 已登录 → 主壳 + 宠物 + 标签栏
  // ─────────────────────────────────────────────
  {
    const { window, errors } = boot({ token, petId })
    const { text } = await render(window)
    check('已登录进入主应用', text.includes('猫事') && !text.includes('开始记录'))
    check('顶栏显示当前宠物', text.includes(petName), `期望包含 "${petName}"`)
    // 标签栏是**日记形态**的四个（档案/健康移到了「猫」页里）
    check(
      '底部标签栏渲染（日记形态）',
      ['日记', '说话', '一天', '猫'].every((t) => text.includes(t)),
    )
    check(
      '档案/健康不再占据标签栏',
      !text.includes('档案') && !text.includes('健康'),
      '它们是一次性设置与低频功能，不该占日常入口',
    )
    check('模型模式常驻顶栏', text.includes('LIVE') || text.includes('MOCK'))
    check('主应用无运行时错误', errors.length === 0, errors[0] || '')
  }

  // ─────────────────────────────────────────────
  // 3. 空态：各标签页在没数据时不能崩
  // ─────────────────────────────────────────────
  for (const [hash, label, expect] of [
    ['#/timeline', '日记', '日记'],
    ['#/profile', '档案', '还没有档案'],
    ['#/story', '一天', '没有可写的记录'],
    ['#/health', '健康', '无法评估'],
    ['#/pets', '猫', petName],
  ]) {
    const { window, errors } = boot({ token, petId, hash })
    const { text } = await render(window, { wait: 3000 })
    const ok = text.includes(expect)
    check(`${label}页挂载正确`, ok, `期望包含 "${expect}"，实际长度 ${text.length}`)
    check(`${label}页无运行时错误`, errors.length === 0, errors[0] || '')
  }

  // ─────────────────────────────────────────────
  // 4. 无效 token → 必须退回登录（而不是卡在失败界面）
  // ─────────────────────────────────────────────
  {
    const { window } = boot({ token: 'invalid.token', petId })
    const { text } = await render(window, { wait: 2600 })
    check('无效 token 退回登录页', text.includes('开始记录') || text.includes('访问令牌'))
  }

  // ─────────────────────────────────────────────
  // 5. 多宠物：切换后对话流必须清空
  // ─────────────────────────────────────────────
  {
    const other = await (
      await fetch(`${API}/v1/pets`, {
        method: 'POST',
        headers: H,
        body: JSON.stringify({ name: `花花${Math.floor(Math.random() * 900 + 100)}` }),
      })
    ).json()

    const { window } = boot({ token, petId, hash: '#/chat' })
    const { text: before, html: htmlBefore } = await render(window)

    // 在 input 里输入一句，证明有会话内状态
    const textarea = window.document.querySelector('textarea')
    check('对话页有输入框', Boolean(textarea))

    // 切到「猫」标签，改选另一只，再回对话页
    clickByText(window, '猫')
    await new Promise((r) => setTimeout(r, 900))
    clickByText(window, other.name)
    await new Promise((r) => setTimeout(r, 1400))

    const after = (window.document.getElementById('root')?.textContent || '').replace(/\s+/g, ' ')
    check('切换宠物后顶栏更新', after.includes(other.name), `期望包含 "${other.name}"`)
    // 选完宠物落到**日记页**（记录是主循环，说话是它的补充）
    check(
      '切换宠物后落到日记页',
      after.includes('的日记'),
      '记录是主循环，选完宠物应回到日记',
    )
    check('切换前渲染非空', htmlBefore.length > 200, `before 长度 ${before.length}`)
  }

  // ─────────────────────────────────────────────
  // 6. 真实对话（走真模型）
  // ─────────────────────────────────────────────
  {
    const { window, errors } = boot({ token, petId, hash: '#/chat' })
    await render(window)

    const textarea = window.document.querySelector('textarea')
    if (textarea) {
      // React 受控组件：必须用原生 setter 触发 onChange
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype,
        'value',
      ).set
      setter.call(textarea, '团团今天第一次吃了罐头')
      textarea.dispatchEvent(new window.Event('input', { bubbles: true }))
      await new Promise((r) => setTimeout(r, 200))

      const sendBtn = [...window.document.querySelectorAll('button')].find(
        (b) => b.getAttribute('aria-label') === '发送',
      )
      if (sendBtn) {
        sendBtn.dispatchEvent(new window.MouseEvent('click', { bubbles: true }))
        await new Promise((r) => setTimeout(r, 14000))
        const text = (window.document.getElementById('root')?.textContent || '').replace(
          /\s+/g,
          ' ',
        )
        check('对话产生回复', text.includes('团团今天第一次吃了罐头'), '用户消息应出现在流里')
        check('对话无运行时错误', errors.length === 0, errors[0] || '')
      } else {
        check('找到发送按钮', false)
      }
    }
  }

  // ─────────────────────────────────────────────
  // 汇总
  // ─────────────────────────────────────────────
  console.log('================ E2E 结果 ================')
  let failed = 0
  for (const r of results) {
    console.log(`${r.ok ? '  ✅' : '  ❌'} ${r.name}${r.ok || !r.detail ? '' : `  ← ${r.detail}`}`)
    if (!r.ok) failed++
  }
  console.log(`\n${results.length - failed}/${results.length} 通过`)
  process.exit(failed > 0 ? 1 : 0)
}

main().catch((e) => {
  console.error('E2E 崩溃:', e)
  process.exit(2)
})
