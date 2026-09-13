/**
 * 前端挂载冒烟测试。
 *
 * 用 jsdom 加载**生产构建产物**，跑真实 fetch（打到本地后端），
 * 断言关键文案出现在 DOM 里。
 *
 * ## 为什么值得单独写一个
 *
 * `tsc` 通过 + `vite build` 成功**不能**证明运行时没问题：
 * 字段名写错、解构 undefined、effect 无限循环、组件抛错 ——
 * 这些全都能通过编译。只有真的挂载一次才看得见。
 *
 * 用法（需要先 `npm run build`，并在 8000 端口跑起后端）::
 *
 *     node scripts/smoke.cjs dist http://127.0.0.1:8000
 */

const fs = require('fs')
const path = require('path')

async function main() {
  // 延迟 require：jsdom 是 devDependency，缺失时给出可执行的提示而不是堆栈
  let JSDOM, VirtualConsole
  try {
    ;({ JSDOM, VirtualConsole } = require('jsdom'))
  } catch {
    console.error('缺少 jsdom。先运行：npm install -D jsdom')
    process.exit(2)
  }

  const DIST = process.argv[2] || path.join(__dirname, '..', 'dist')
  const API = process.argv[3] || 'http://127.0.0.1:8000'

  const errors = []
  const virtualConsole = new VirtualConsole()
  virtualConsole.on('jsdomError', (e) => errors.push(`[jsdom] ${e.message}`))
  virtualConsole.on('error', (...args) => errors.push(`[console.error] ${args.join(' ')}`))

  const html = fs.readFileSync(path.join(DIST, 'index.html'), 'utf8')
  const dom = new JSDOM(html, {
    url: 'http://127.0.0.1:4173/',
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    virtualConsole,
  })
  const { window } = dom

  // ── 补齐 jsdom 没有的浏览器 API ──
  // 同源请求转发到后端，等价于生产里的 Nginx 反代
  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : input.url
    const absolute = url.startsWith('http') ? url : `${API}${url}`
    return fetch(absolute, init)
  }
  window.AbortController = AbortController
  window.DOMException = DOMException
  window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} })
  window.scrollTo = () => {}
  window.Element.prototype.scrollTo = () => {}
  window.Element.prototype.scrollIntoView = () => {}

  const storage = new Map()
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem: (k) => (storage.has(k) ? storage.get(k) : null),
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: (k) => storage.delete(k),
      clear: () => storage.clear(),
    },
    configurable: true,
  })

  const assetsDir = path.join(DIST, 'assets')
  const assets = fs.readdirSync(assetsDir)
  const entry = assets.find((f) => f.startsWith('index-') && f.endsWith('.js'))
  const css = assets.find((f) => f.startsWith('index-') && f.endsWith('.css'))
  if (!entry) throw new Error(`在 ${assetsDir} 里找不到 index-*.js`)

  if (css) {
    const style = window.document.createElement('style')
    style.textContent = fs.readFileSync(path.join(assetsDir, css), 'utf8')
    window.document.head.appendChild(style)
  }

  // 构建产物是 ESM，在 window 上下文里执行以拿到 window/document
  window.eval(fs.readFileSync(path.join(assetsDir, entry), 'utf8'))

  await new Promise((r) => setTimeout(r, 2500))

  return { window, errors }
}

main()
  .then(({ window, errors }) => {
    const root = window.document.getElementById('root')
    const text = root ? root.textContent || '' : ''
    const inner = root ? root.innerHTML : ''

    const checks = [
      ['根节点已挂载', inner.length > 200],
      ['品牌名渲染', text.includes('猫事')],
      ['副标题渲染', text.includes('Field Notebook')],
      ['登录入口可用', text.includes('开始记录') || text.includes('访问令牌')],
      ['无 React 崩溃', !text.includes('Something went wrong')],
    ]

    let failed = 0
    console.log('--- 冒烟测试 ---')
    for (const [name, ok] of checks) {
      console.log(`${ok ? '  ✅' : '  ❌'} ${name}`)
      if (!ok) failed++
    }

    if (errors.length) {
      console.log('\n运行时错误:')
      for (const e of errors.slice(0, 10)) console.log('  ⚠️ ', e)
      failed++
    }

    console.log('\n--- 渲染文本 ---')
    console.log(text.replace(/\s+/g, ' ').slice(0, 300))

    process.exit(failed > 0 ? 1 : 0)
  })
  .catch((e) => {
    console.error('冒烟测试崩溃:', e)
    process.exit(2)
  })
