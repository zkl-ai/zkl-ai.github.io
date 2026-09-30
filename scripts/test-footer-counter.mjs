// 用最小 DOM stub 跑一遍 dist/index.html 里真实生成的页脚脚本，验证自排除逻辑。
import { readFileSync } from 'node:fs';

const html = readFileSync(process.argv[2] || 'dist/index.html', 'utf8');
const m = html.match(/<footer class="site-footer">[\s\S]*?<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error('footer script not found');
const baseCode = m[1];

async function run(label, { code = baseCode, url = 'http://zkl-ai.top/', ls = {}, ss = {}, storageThrows = false, expect }) {
  const state = {
    appended: [],
    fetchCalls: [],
    replaceState: [],
    local: { ...ls },
    session: { ...ss },
    counterEl: { style: {} },
    gcEl: { textContent: '--' },
  };

  const mkStorage = (bag) => ({
    getItem: (k) => (storageThrows ? (() => { throw new Error('blocked'); })() : (k in bag ? bag[k] : null)),
    setItem: (k, v) => { if (storageThrows) throw new Error('blocked'); bag[k] = String(v); },
    removeItem: (k) => { if (storageThrows) throw new Error('blocked'); delete bag[k]; },
  });

  const sandbox = {
    __state: state,
    location: new URL(url),
    URLSearchParams,
    JSON,
    console,
    history: { replaceState: (a, b, u) => state.replaceState.push(u) },
    localStorage: mkStorage(state.local),
    sessionStorage: mkStorage(state.session),
    fetch: (u) => {
      state.fetchCalls.push(u);
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ count: '1 234' }) });
    },
    document: {
      querySelector: (sel) => (sel === '.site-footer .counter' ? state.counterEl : null),
      getElementById: (id) => (id === 'gc-total' ? state.gcEl : null),
      createElement: () => {
        const el = { attrs: {} };
        el.setAttribute = (k, v) => { el.attrs[k] = v; };
        return el;
      },
      head: { appendChild: (el) => { state.appended.push(el); } },
    },
  };

  const names = ['location', 'URLSearchParams', 'history', 'localStorage', 'sessionStorage', 'fetch', 'document', 'JSON', 'console', '__state'];
  const fn = new Function(...names, code + '\nreturn __state;');

  let out;
  try {
    out = fn(...names.map((n) => sandbox[n]));
  } catch (e) {
    console.log(`✗ ${label} → 抛异常: ${e.message}`);
    return false;
  }

  // 让 fetch().then() 的回调先跑完（页脚数字是异步回填的）
  await new Promise((r) => setImmediate(r));

  const real = {
    scripts: out.appended.map((e) => e.src),
    attrs: out.appended.map((e) => e.attrs),
    fetches: out.fetchCalls,
    local: out.local,
    session: out.session,
    replaced: out.replaceState,
    hidden: out.counterEl.style.display === 'none',
    gc: out.gcEl.textContent,
  };  const ok = expect(real);
  console.log(`${ok ? '✓' : '✗'} ${label}`);
  console.log(
    `   脚本=${JSON.stringify(real.scripts)} localStorage=${JSON.stringify(real.local)}` +
      ` URL重写=${JSON.stringify(real.replaced)} 藏计数=${real.hidden} gc总数=${real.gc}`
  );
  return ok;
}

let all = true;
const B = 'https://busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js';
const G = 'https://gc.zgo.at/count.js';
const C = 'https://static.cloudflareinsights.com/beacon.min.js';
const gc = (v) => baseCode.replace('"busuanzi"', '"goatcounter"').replace('const gcCode = "";', `const gcCode = ${JSON.stringify(v)};`);

// 1. 默认（busuanzi，无标记）→ 加载不蒜子
all &= await run('1. busuanzi / 无标记 → 计数', {
  expect: (r) => r.scripts.length === 1 && r.scripts[0] === B && !r.hidden,
});

// 2. ?me=1 → 不加载、写入标记、URL 抹掉 me 参数、藏计数
all &= await run('2. ?me=1 → 自排除', {
  url: 'http://zkl-ai.top/blog/?me=1&tag=vLLM',
  expect: (r) =>
    r.scripts.length === 0 &&
    r.local.is_owner === '1' &&
    r.session.is_owner === '1' &&
    r.replaced[0] === '/blog/?tag=vLLM' &&
    r.hidden,
});

// 3. 已有标记 → 仍然不加载
all &= await run('3. 已有 is_owner 标记 → 不计数', {
  ls: { is_owner: '1' },
  expect: (r) => r.scripts.length === 0 && r.hidden,
});

// 4. 无痕：只有 sessionStorage 标记 → 不加载
all &= await run('4. 无痕会话内标记 → 不计数', {
  ss: { is_owner: '1' },
  expect: (r) => r.scripts.length === 0,
});

// 5. ?me=0 取消排除 → 恢复计数并清掉标记
all &= await run('5. ?me=0 → 恢复计数', {
  ls: { is_owner: '1' },
  ss: { is_owner: '1' },
  url: 'http://zkl-ai.top/?me=0',
  expect: (r) => r.scripts.length === 1 && !('is_owner' in r.local) && !('is_owner' in r.session),
});

// 6. ?me 不带值 → 视为开启排除
all &= await run('6. ?me（裸参数）→ 自排除', {
  url: 'http://zkl-ai.top/?me',
  expect: (r) => r.scripts.length === 0 && r.local.is_owner === '1' && r.replaced[0] === '/',
});

// 7. localStorage 被禁（Safari 隐私模式）→ 不崩、照常计数
all &= await run('7. storage 抛异常 → 不崩', {
  storageThrows: true,
  expect: (r) => r.scripts.length === 1,
});

// 8. GoatCounter 模式 → count.js + TOTAL.json + 回填数字
all &= await run('8. goatcounter → 计数 + 页脚总数', {
  code: gc('zkl-ai'),
  expect: (r) =>
    r.scripts[0] === G &&
    r.attrs[0]['data-goatcounter'] === 'https://zkl-ai.goatcounter.com/count' &&
    r.fetches[0] === 'https://zkl-ai.goatcounter.com/counter/TOTAL.json' &&
    r.gc === '1 234',
});

// 9. GoatCounter 但没填站点码 → 什么都不加载
all &= await run('9. goatcounter 未填码 → 安全跳过', {
  code: baseCode.replace('"busuanzi"', '"goatcounter"'),
  expect: (r) => r.scripts.length === 0 && r.fetches.length === 0,
});

// 10. Cloudflare 模式
all &= await run('10. cloudflare → beacon', {
  code: baseCode.replace('"busuanzi"', '"cloudflare"').replace('const cfToken = "";', 'const cfToken = "abc123";'),
  expect: (r) => r.scripts[0] === C && JSON.parse(r.attrs[0]['data-cf-beacon']).token === 'abc123',
});

// 11. none → 零请求
all &= await run('11. none → 零请求', {
  code: baseCode.replace('"busuanzi"', '"none"'),
  expect: (r) => r.scripts.length === 0,
});

console.log(all ? '\n全部通过' : '\n有用例失败');
process.exit(all ? 0 : 1);
