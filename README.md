# zkl-ai.top · 个人主页与博客

张坤龙（Zhangkunlong）的个人主页 + 博客，基于 [Astro](https://astro.build) 构建，部署在 GitHub Pages，绑定自定义域名 **zkl-ai.top**。

## 技术栈

- [Astro](https://astro.build) —— 静态站点框架（默认零 JS，快）
- [@astrojs/sitemap](https://docs.astro.build/zh-cn/guides/integrations-guide/sitemap/) —— 站点地图
- [@astrojs/rss](https://docs.astro.build/zh-cn/guides/rss/) —— RSS 订阅
- GitHub Actions + GitHub Pages —— 免费自动部署

## 目录结构

```
homepage/
├── public/
│   ├── CNAME              # 自定义域名（zkl-ai.top）
│   └── favicon.svg
├── src/
│   ├── components/        # Header / Footer 等组件
│   ├── content/
│   │   └── blog/          # 博客文章（Markdown）
│   ├── content.config.ts  # 内容集合 schema
│   ├── layouts/           # 页面布局
│   ├── pages/
│   │   ├── index.astro    # 主页（简历）
│   │   ├── blog/          # 博客列表 + 文章页
│   │   ├── rss.xml.js     # RSS
│   │   └── 404.astro
│   └── styles/global.css  # 全局样式
├── .github/workflows/deploy.yml
└── astro.config.mjs
```

## 本地开发

```bash
# 1. 安装依赖（首次）
npm install

# 2. 启动开发服务器
npm run dev
# 打开 http://localhost:4321

# 3. 构建生产版本
npm run build

# 4. 本地预览构建结果
npm run preview
```

## 部署到 GitHub Pages（免费）

### 1. 推送到 GitHub

```bash
cd homepage
git init
git add .
git commit -m "init: personal homepage"
git branch -M main
git remote add origin https://github.com/zkl-ai/<仓库名>.git
git push -u origin main
```

> 仓库名随意（例如 `zkl-ai.github.io` 或 `homepage`）。**推荐命名 `zkl-ai.github.io`**，
> 这样即便暂不绑定域名，也能直接通过 `https://zkl-ai.github.io` 访问。

### 2. 开启 GitHub Pages

在仓库 **Settings → Pages**：

1. **Source** 选择 **GitHub Actions**（`deploy.yml` 会自动构建并部署）
2. **Custom domain** 填写 `zkl-ai.top`，保存

推送到 `main` 分支后，Actions 会自动构建部署。

### 3. 绑定域名 zkl-ai.top

在域名服务商（如阿里云 / 腾讯云 / Cloudflare）的 **DNS 解析** 中添加记录：

**方式 A（推荐）—— 直接用裸域名 `zkl-ai.top`，添加 4 条 A 记录：**

| 类型 | 主机记录 | 记录值 |
|------|----------|--------|
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |

**方式 B —— 用 `www.zkl-ai.top`，添加 1 条 CNAME 记录：**

| 类型 | 主机记录 | 记录值 |
|------|----------|--------|
| CNAME | www | zkl-ai.github.io |

> 选 A 就用 `zkl-ai.top` 作为最终访问地址；选 B 则用 `www.zkl-ai.top`（并在 GitHub Pages
> 的 Custom domain 里填对应地址）。DNS 生效通常需要几分钟到几小时。

### 4. 验证

部署完成后，浏览器访问 **https://zkl-ai.top** 即可看到主页。

## 如何更新内容

### 修改简历（主页）

编辑 `src/pages/index.astro`，所有文字都在这个文件里。

### 写新博客

> 写之前先读 [BLOG_GUIDE.md](BLOG_GUIDE.md)（博客写作手册）：公式、图片、语气、发布前检查清单都在里面，能避开大部分已经踩过的坑。

在 `src/content/blog/` 下新建一个 `.md` 文件，顶部按下面格式写 frontmatter：

```md
---
title: "文章标题"
description: "一句话摘要，用于列表页和 SEO"
pubDate: 2026-09-28
tags: ["标签1", "标签2"]
category: inference-acceleration
draft: true
---

正文（Markdown）……
```

- `category`（可选）挂到对应专区，取值两个：`inference-acceleration`（多模态大模型推理加速）或 `agent-optimization`（多模态长程智能体优化）；不写则归入「全部文章」、不显示专区标签
- `draft: true` 表示草稿（不发布），写完删掉这一行或改成 `false` 即发布

推送到 GitHub 后自动发布。

### 修改站点配置

- 域名 / sitemap：`astro.config.mjs` 里的 `site`
- 导航链接：`src/components/Header.astro`
- 页脚信息：`src/components/Footer.astro`
- 配色 / 字体 / 间距：`src/styles/global.css`

## 待你确认的占位内容

以下内容是我根据你提供的信息填写的，上线前请核对：

1. **岗位名称**：「AI Agent 工程师（智能标注方向）」——你说第二个岗位名称没想好，这个先用着，想好了在
   `src/pages/index.astro` 里改（搜索「AI Agent 工程师」即可）。
2. **技能标签**：CUDA / 算子优化 / TensorRT 等是按方向推断的，请改成你真实的技能栈。
3. ~~竞赛第 3 条~~ ✅ 已确认并修正为「AFAC2023 金融智能挑战赛 · 季军」（基金趋势模拟预测赛题，813 支队伍）。
4. **个人头像**：目前没有头像图，如需可在 `public/` 放一张并引用。

## 访客统计

页脚计数由 `src/data/analytics.ts` 一处配置控制（改完重新部署即可），加载逻辑在 `src/components/Footer.astro`。

### ⚠️ 不蒜子的数字是虚高的

默认仍在用[不蒜子](https://busuanzi.ibruce.info)，它的 UV **只认一个第三方 cookie**（`busuanziId`，写在 `busuanzi.ibruce.info` 域下）：

- 同一浏览器**换设备 / 换浏览器 / 换浏览器配置文件**都算一个新访客；
- iPhone 和 iPad 上的 Safari（默认拦截第三方 cookie）、Chrome 无痕、Brave、Firefox 增强保护等下，**每刷新一页就 +1 个访客**；
- 它不认 IP、不做时间去重、不过滤爬虫。

2026-09-30 实测：无 cookie 连续请求同一站点，`site_uv` 从 437 → 438 → 439（同一 IP、同一秒各算一个）；带上同一个 `busuanziId` cookie 后 `site_uv` 不再增长。所以「23 位访客」≈「23 个还没被拦掉的 cookie」，不等于 23 个人。

### 排除自己的访问（推荐 `?me=1`）

在任意设备/浏览器上访问一次：

```
https://zkl-ai.top/?me=1
```

该浏览器就永久不再计数，页脚计数也会隐藏（`me` 参数会自动从地址栏抹掉，方便收藏干净链接）。取消：

```
https://zkl-ai.top/?me=0
```

- 标记存在 localStorage + sessionStorage，按「协议 + 域名」分开存：`http://zkl-ai.top` 和 `https://zkl-ai.top` **要各设一次**。
- 无痕窗口的标记只在本次会话有效，下次开无痕要再访问一次 `?me=1`。
- 老写法 `localStorage.setItem('is_owner','1')` 依然有效。

### 换成 GoatCounter（推荐，数字才接近「人」）

[GoatCounter](https://www.goatcounter.com) 对个人用途免费，**完全不用 cookie**：服务端按「IP + User-Agent」在 8 小时会话内去重，凡声明自己是爬虫的请求直接忽略。刷新页面、翻页都不会再涨访客数。

1. 到 https://www.goatcounter.com 注册，站点码取 `zkl-ai`（决定统计地址 `https://<站点码>.goatcounter.com`）。
2. 编辑 `src/data/analytics.ts`：

   ```ts
   provider: 'goatcounter',
   goatcounterCode: 'zkl-ai',
   ```

3. GoatCounter 后台打开 **Settings → Allow adding visitor counts on your website**，页脚才会显示总数（该接口最多缓存 4 小时）。
4. 建议再在 **Settings → Tracking → Ignore IPs** 里填上你常用的出口 IP：家里/公司的所有设备一次性全部排除，比逐台 `?me=1` 彻底。

已知偏差：广告拦截插件会拦掉 GoatCounter 约 1/3 的请求，所以它的数字偏小而不是偏大；爬虫不计数。

### 其他可选项

- `provider: 'cloudflare'` + 填 `cloudflareToken`：用 Cloudflare Web Analytics。需要先把 `zkl-ai.top` 的 DNS 迁到 Cloudflare 并开启代理；它没有公开计数接口，页脚不显示数字，只能去 Cloudflare 后台看。
- `provider: 'none'`：不加载任何统计脚本，页脚也不显示计数。

### 回归测试

```bash
npm run test:counter
```

先 build，再用一个最小 DOM stub 跑 `dist/index.html` 里真实生成的页脚脚本，覆盖 11 个用例（各 provider、`?me=1` / `?me=0`、无痕、storage 被禁等）。

## 已知问题：HTTPS 证书不匹配（待修）

2026-09-30 检查发现：`https://zkl-ai.top` 返回的证书是 GitHub 的 `CN=*.github.io`，与域名不匹配，浏览器会直接报「您的连接不是私密连接」（`ERR_CERT_COMMON_NAME_INVALID`）；而 `http://zkl-ai.top` 能正常打开（200）且**不会**跳转到 https。也就是说，现在直接在浏览器里输 `zkl-ai.top` 的真人访客大部分会被证书警告挡走。

DNS 本身没问题（阿里云 NS，4 条 A 记录指向 GitHub Pages，无 AAAA / CAA 记录），是 GitHub Pages 那边还没给这个自定义域名签下证书。

修法：

1. 仓库 **Settings → Pages → Custom domain** 重新填一次 `zkl-ai.top` 并 **Save**（触发 Let's Encrypt 重新签发）。
2. 等 15 分钟到几小时，用下面命令确认证书里的 `subject` 变成 `CN=zkl-ai.top`：

   ```bash
   curl -sv https://zkl-ai.top/ 2>&1 | grep 'subject:'
   ```

3. 证书就绪后勾上 **Enforce HTTPS**。

> 注意：https 修好后，`http://` 和 `https://` 是两个不同的源，`?me=1` 的自排除标记要在 https 下重新设一次。

## 常见问题（本地环境）

**`npm install` 报 EPERM（npm cache 目录有 root 权限文件）**

这是老版本 npm 的已知 bug，缓存目录里混入了 root 拥有的文件。任选其一：

```bash
# 方式 1：修正缓存目录所有权（推荐，一劳永逸）
sudo chown -R $(whoami) ~/.npm

# 方式 2：改用项目本地缓存（不改系统目录）
npm_config_cache=./.npm-cache npm install
```

**`astro build` 报 EPERM mkdir '~/Library/Preferences/astro'**

Astro 遥测想写配置目录被拦，禁用遥测即可：

```bash
ASTRO_TELEMETRY_DISABLED=1 npm run build
```

## License

代码可自由使用（MIT）。
