export type Lang = 'zh' | 'en';

export interface SkillGroup {
  name: string;
  /** 第一个 tag 会被高亮为强调色 */
  tags: string[];
}

export interface ResumeData {
  siteName: string;
  metaDescription: string;
  hero: {
    eyebrow: string;
    name: string;
    roles: [string, string];
    tagline: string;
    location: string;
    github: string;
    contact: string;
    blog: string;
  };
  about: {
    title: string;
    /** 段落内可含 <strong>、<a> 等内联 HTML */
    paragraphs: string[];
  };
  experience: {
    title: string;
    items: { badge: string; title: string; desc: string }[];
  };
  education: {
    title: string;
    items: { meta: string; school: string; major: string }[];
  };
  skills: {
    title: string;
    groups: SkillGroup[];
  };
  publications: {
    title: string;
    items: { badge: string; title: string; href: string; authors: string }[];
  };
  competitions: {
    title: string;
    items: { badge: string; title: string; href: string; desc: string }[];
  };
  blogCta: {
    title: string;
    desc: string;
    button: string;
  };
  nav: {
    about: string;
    experience: string;
    skills: string;
    publications: string;
    competitions: string;
    blog: string;
  };
  footer: {
    views: string;
    visitors: string;
    blog: string;
    rss: string;
  };
}

const zh: ResumeData = {
  siteName: '张坤龙',
  metaDescription:
    '张坤龙（Zhangkunlong）—— 多模态大模型推理加速 & AI Agent 工程师的个人简历与博客。',
  hero: {
    eyebrow: 'Zhangkunlong · zkl-ai',
    name: '张坤龙',
    roles: ['多模态大模型推理加速工程师', 'AI Agent 工程师'],
    tagline: '专注 AI 系统工程落地的工程师：让大模型推理更快更省，把 AI Agent 真正跑进生产。',
    location: '📍 上海',
    github: 'GitHub',
    contact: '联系我',
    blog: '阅读博客',
  },
  about: {
    title: '关于我',
    paragraphs: [
      '我是张坤龙，一名专注 <strong>AI 系统工程落地</strong> 的工程师，现居上海，目前在美团从事 <strong>多模态大模型推理加速</strong> 与 <strong>AI Agent</strong> 相关工作。',
      '我毕业于<a href="https://www.sustech.edu.cn" target="_blank" rel="noopener">南方科技大学</a>（本科：计算机科学与技术，硕士：电子科学与技术）。在 vLLM 社区提交过若干 PR 贡献，发表过面向边缘设备的硬件感知 DNN 压缩方向论文（获 Best Poster 最佳海报奖），并在天池、Kaggle 等竞赛中多次获奖。',
    ],
  },
  experience: {
    title: '工作经历',
    items: [
      {
        badge: '2026.07 - 至今',
        title: '美团 · 多模态大模型推理加速 & 多模态长程 AI Agent 降本增效',
        desc: '负责多模态大模型（Qwen-VL 系列）的推理加速，以及多模态长程 AI Agent 的降本增效与工程落地。',
      },
      {
        badge: '2025.04 - 2025.09',
        title: '美团 · 多模态大模型推理加速（实习）',
        desc: '参与多模态大模型推理加速相关工作。',
      },
    ],
  },
  education: {
    title: '教育背景',
    items: [
      { meta: '2023.09 - 2026.06 · 硕士', school: '南方科技大学', major: '电子科学与技术' },
      { meta: '2019.09 - 2023.06 · 本科', school: '南方科技大学', major: '计算机科学与技术' },
    ],
  },
  skills: {
    title: '技能',
    groups: [
      {
        name: '大模型推理加速',
        tags: ['vLLM', 'SGLang', '模型量化（GPTQ / SpineQuant 等）', 'Vision Token Compression', 'SpecDecode（投机解码）'],
      },
      {
        name: 'Agent 优化方向',
        tags: ['RSI（自进化 Agent）', '记忆进化（Experience Memory）', 'Skill 进化（Skill Bank）', 'Context / Harness 优化', '长程任务编排'],
      },
      {
        name: '多模态 & 生成式模型',
        tags: ['Qwen-VL', '视觉语言模型', 'Diffusion'],
      },
      {
        name: '编程语言 & 工具',
        tags: ['Python', 'PyTorch', 'TensorRT', 'Triton Inference Server', 'Hugging Face', 'Git', 'Linux'],
      },
    ],
  },
  publications: {
    title: '论文',
    items: [
      {
        badge: 'Best Poster · 最佳海报奖',
        title: 'Hardware-aware DNN Compression for Homogeneous Edge Devices',
        href: 'https://doi.org/10.1109/DOCS67533.2025.11200827',
        authors:
          'Zhang K, Li G, Lu N, et al. — 2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS), IEEE, 2025, pp. 387–392.',
      },
    ],
  },
  competitions: {
    title: '竞赛',
    items: [
      {
        badge: '天池 · 季军',
        title: 'AFAC2023 金融智能挑战赛',
        href: 'https://cse.sustech.edu.cn/graduate/1760.html',
        desc: '基金趋势模拟预测赛题（金融场景理解方向），813 支队伍中荣获季军 · 队伍「稳盈预测」',
      },
      {
        badge: 'Kaggle · 银牌',
        title: 'Optiver - Trading at the Close',
        href: 'https://www.kaggle.com/competitions/optiver-trading-at-the-close',
        desc: '排名 205 / 4436',
      },
      {
        badge: 'Kaggle · 铜牌',
        title: 'LMSYS - Chatbot Arena Human Preference Predictions',
        href: 'https://www.kaggle.com/competitions/lmsys-chatbot-arena',
        desc: '排名 164 / 1849',
      },
    ],
  },
  blogCta: {
    title: '想看看我最近在做什么？',
    desc: '记录大模型推理加速、AI Agent 与工程实践中的思考。',
    button: '进入博客',
  },
  nav: {
    about: '关于',
    experience: '经历',
    skills: '技能',
    publications: '论文',
    competitions: '竞赛',
    blog: '博客',
  },
  footer: {
    views: '访问',
    visitors: '访客',
    blog: '博客',
    rss: 'RSS',
  },
};

const en: ResumeData = {
  siteName: 'Zhangkunlong',
  metaDescription:
    'Zhangkunlong — Engineer focused on multimodal LLM inference acceleration & AI Agent systems.',
  hero: {
    eyebrow: 'Zhangkunlong · zkl-ai',
    name: 'Zhangkunlong',
    roles: ['Multimodal LLM Inference Engineer', 'AI Agent Engineer'],
    tagline:
      'Engineer focused on shipping AI systems to production — making LLM inference faster and cheaper, and putting AI Agents into real workloads.',
    location: '📍 Shanghai, China',
    github: 'GitHub',
    contact: 'Contact',
    blog: 'Blog',
  },
  about: {
    title: 'About Me',
    paragraphs: [
      'I am Zhangkunlong, an engineer focused on <strong>shipping AI systems to production</strong>, based in Shanghai. I currently work at Meituan on <strong>multimodal LLM inference acceleration</strong> and <strong>AI Agent</strong> systems.',
      'I graduated from <a href="https://www.sustech.edu.cn" target="_blank" rel="noopener">Southern University of Science and Technology</a> (B.Eng. in Computer Science and Technology, M.Eng. in Electronic Science and Technology). I have contributed PRs to the vLLM community, published a paper on hardware-aware DNN compression for edge devices (Best Poster Award), and won several competitions on Tianchi and Kaggle.',
    ],
  },
  experience: {
    title: 'Experience',
    items: [
      {
        badge: '2026.07 - Present',
        title: 'Meituan · Multimodal LLM Inference Acceleration & Long-horizon AI Agent',
        desc: 'Inference acceleration for multimodal LLMs (Qwen-VL series); cost reduction and efficiency optimization of long-horizon AI Agents.',
      },
      {
        badge: '2025.04 - 2025.09',
        title: 'Meituan · Multimodal LLM Inference Acceleration (Intern)',
        desc: 'Contributed to multimodal LLM inference acceleration.',
      },
    ],
  },
  education: {
    title: 'Education',
    items: [
      {
        meta: '2023.09 - 2026.06 · M.Eng.',
        school: 'Southern University of Science and Technology',
        major: 'Electronic Science and Technology',
      },
      {
        meta: '2019.09 - 2023.06 · B.Eng.',
        school: 'Southern University of Science and Technology',
        major: 'Computer Science and Technology',
      },
    ],
  },
  skills: {
    title: 'Skills',
    groups: [
      {
        name: 'LLM Inference Acceleration',
        tags: ['vLLM', 'SGLang', 'Model Quantization (GPTQ / SpineQuant)', 'Vision Token Compression', 'Speculative Decoding'],
      },
      {
        name: 'AI Agent Optimization',
        tags: ['RSI (Self-Evolving Agent)', 'Memory Evolution (Experience Memory)', 'Skill Evolution (Skill Bank)', 'Context / Harness Optimization', 'Long-horizon Orchestration'],
      },
      {
        name: 'Multimodal & Generative Models',
        tags: ['Qwen-VL', 'Vision-Language Models', 'Diffusion'],
      },
      {
        name: 'Languages & Tools',
        tags: ['Python', 'PyTorch', 'TensorRT', 'Triton Inference Server', 'Hugging Face', 'Git', 'Linux'],
      },
    ],
  },
  publications: {
    title: 'Publications',
    items: [
      {
        badge: 'Best Poster Award',
        title: 'Hardware-aware DNN Compression for Homogeneous Edge Devices',
        href: 'https://doi.org/10.1109/DOCS67533.2025.11200827',
        authors:
          'Zhang K, Li G, Lu N, et al. — 2025 7th International Conference on Data-driven Optimization of Complex Systems (DOCS), IEEE, 2025, pp. 387–392.',
      },
    ],
  },
  competitions: {
    title: 'Competitions',
    items: [
      {
        badge: 'Tianchi · 3rd Place',
        title: 'AFAC2023 Financial Intelligence Challenge',
        href: 'https://cse.sustech.edu.cn/graduate/1760.html',
        desc: 'Fund trend simulation prediction track; 3rd place out of 813 teams.',
      },
      {
        badge: 'Kaggle · Silver',
        title: 'Optiver - Trading at the Close',
        href: 'https://www.kaggle.com/competitions/optiver-trading-at-the-close',
        desc: 'Ranked 205 / 4436',
      },
      {
        badge: 'Kaggle · Bronze',
        title: 'LMSYS - Chatbot Arena Human Preference Predictions',
        href: 'https://www.kaggle.com/competitions/lmsys-chatbot-arena',
        desc: 'Ranked 164 / 1849',
      },
    ],
  },
  blogCta: {
    title: 'Want to see what I am working on?',
    desc: 'Notes on LLM inference acceleration, AI Agents, and engineering practice.',
    button: 'Read Blog',
  },
  nav: {
    about: 'About',
    experience: 'Experience',
    skills: 'Skills',
    publications: 'Publications',
    competitions: 'Competitions',
    blog: 'Blog',
  },
  footer: {
    views: 'Views',
    visitors: 'Visitors',
    blog: 'Blog',
    rss: 'RSS',
  },
};

export const resume: Record<Lang, ResumeData> = { zh, en };
