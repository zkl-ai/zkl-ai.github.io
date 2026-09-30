/**
 * 站点访问统计配置。
 *
 * 只改这个文件即可切换统计方案，Footer.astro 会自动跟着变。
 * 改完记得重新 build / 推送，GitHub Actions 会自动部署。
 */

export type AnalyticsProvider = 'busuanzi' | 'goatcounter' | 'cloudflare' | 'none';

export const analytics: {
  provider: AnalyticsProvider;
  goatcounterCode: string;
  cloudflareToken: string;
} = {
  /**
   * 用哪家统计：
   *
   * - 'goatcounter'【推荐】：无 cookie，服务端按「IP + User-Agent」在 8 小时会话内去重，
   *   自动忽略声明自己是爬虫的请求。不存在不蒜子那种「换个浏览器 / 刷新一次就 +1 个访客」
   *   的虚高问题。注意它会被广告拦截插件拦掉约 1/3 请求，所以数字偏小而不是偏大。
   * - 'busuanzi'：原来的不蒜子。UV 只认第三方 cookie，数字严重虚高（详见 README）。
   * - 'cloudflare'：Cloudflare Web Analytics。需要把域名 DNS 迁到 Cloudflare 并开启代理，
   *   且没有公开计数接口，页脚不显示数字，只能去 Cloudflare 后台看。
   * - 'none'：不加载任何统计脚本，页脚也不显示计数。
   */
  provider: 'busuanzi',

  /**
   * GoatCounter 站点码。注册 https://www.goatcounter.com 后，
   * 你的统计地址是 https://<站点码>.goatcounter.com，这里就填 <站点码>。
   * 例如统计地址是 https://zkl-ai.goatcounter.com 就填 'zkl-ai'。
   */
  goatcounterCode: '',

  /** Cloudflare Web Analytics 的 token（只有 provider 为 'cloudflare' 时才用到）。 */
  cloudflareToken: '',
};
