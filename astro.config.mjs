import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
  // 你的正式域名，sitemap 与 RSS 会用到
  site: 'https://zkl-ai.top',
  integrations: [sitemap()],
  markdown: {
    // 支持 $...$（行内）与 $$...$$（独立成行）的 LaTeX 公式
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      theme: 'github-light',
    },
  },
});
