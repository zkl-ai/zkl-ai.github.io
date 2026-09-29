export interface Category {
  slug: string;
  name: { zh: string; en: string };
  description: { zh: string; en: string };
}

export const categories: Category[] = [
  {
    slug: 'inference-acceleration',
    name: {
      zh: '多模态大模型推理加速',
      en: 'Multimodal LLM Inference Acceleration',
    },
    description: {
      zh: 'vLLM / SGLang、模型量化、投机解码、Visual Token Compression 等推理加速实践。',
      en: 'Inference acceleration: vLLM / SGLang, quantization, speculative decoding, visual token compression.',
    },
  },
  {
    slug: 'agent-optimization',
    name: {
      zh: '多模态长程智能体优化',
      en: 'Multimodal Long-horizon Agent Optimization',
    },
    description: {
      zh: '自进化智能体（Recursive Self-Improvement）、记忆 / Skill / Harness / 上下文进化与降本增效。',
      en: 'Self-evolving agents (RSI), memory / skill / harness / context evolution, and cost optimization.',
    },
  },
];

export function getCategory(slug: string): Category | undefined {
  return categories.find((c) => c.slug === slug);
}
