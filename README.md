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

在 `src/content/blog/` 下新建一个 `.md` 文件，顶部按下面格式写 frontmatter：

```md
---
title: "文章标题"
description: "一句话摘要，用于列表页和 SEO"
pubDate: 2026-09-28
tags: ["标签1", "标签2"]
---

正文（Markdown）……
```

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
3. **竞赛第 3 条**：`cse.sustech.edu.cn/graduate/1760.html` 这条的具体奖项名称我不确定，现在写的是
   「南方科技大学 · 研究生成果展示」，请补充准确标题。
4. **个人头像**：目前没有头像图，如需可在 `public/` 放一张并引用。

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
