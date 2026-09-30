import { resume } from '../data/resume';

/**
 * 拼出 Person 结构化数据（schema.org）。
 *
 * 目的：让搜索引擎能把「张坤龙」这个名字和具体的人对上号。
 * 名字重名很多，光靠页面文本不够——结构化数据能把单位、学校、
 * 研究方向、论文、GitHub 这些一次性说清楚。
 *
 * 所有字段都从 resume.ts 派生，改了简历这里自动跟着变。
 */

const SITE = 'https://zkl-ai.top';
const GITHUB = 'https://github.com/zkl-ai';

export function personSchema() {
  const zh = resume.zh;
  const en = resume.en;

  // 学校去重（本科和硕士是同一所）
  const schools = [...new Set(zh.education.items.map((i) => i.school))];

  // 研究方向：技能分组名 + 具体标签
  const knowsAbout = [
    ...zh.skills.groups.map((g) => g.name),
    ...zh.skills.groups.flatMap((g) => g.tags),
  ];

  // 「美团 · 多模态大模型推理加速 & ...」→「美团」
  const employer = zh.experience.items[0]?.title.split(' · ')[0];

  // 「📍 上海」→「上海」
  const city = zh.hero.location.replace(/[^\u4e00-\u9fa5]/g, '').trim();

  return {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name: zh.hero.name,
    alternateName: en.hero.name,
    url: `${SITE}/`,
    description: zh.metaDescription,
    jobTitle: zh.hero.roles[0],
    email: `mailto:${zh.hero.email}`,
    address: {
      '@type': 'PostalAddress',
      addressLocality: city,
      addressCountry: 'CN',
    },
    worksFor: employer
      ? { '@type': 'Organization', name: employer }
      : undefined,
    alumniOf: schools.map((name) => ({
      '@type': 'CollegeOrUniversity',
      name,
    })),
    sameAs: [GITHUB],
    knowsAbout,
    subjectOf: zh.publications.items.map((p) => ({
      '@type': 'ScholarlyArticle',
      name: p.title,
      sameAs: p.href,
    })),
  };
}
