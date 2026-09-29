---
title: "你好，这里是张坤龙的博客"
description: "开篇：这个博客会写些什么、为什么用 Astro 搭建，以及如何用它记录大模型推理加速与 AI Agent 的实践。"
pubDate: 2026-09-28
tags: ["随笔", "关于本站"]
---

欢迎来到我的博客。

这个博客和我的[个人主页](/)一起，托管在 `zkl-ai.top`。我会在这里记录**大模型推理加速**（尤其是 Qwen3-VL / Qwen3.5 等系列）、**AI Agent** 以及日常工程实践中的思考和踩坑。

## 为什么用 Astro

选 Astro 的理由很简单：它默认就是静态输出，快、SEO 好、部署免费，同时又能用我熟悉的 Markdown 写文章，用组件化的方式组织页面。

- **零 JS 负担**：页面默认不往浏览器塞多余脚本
- **内容优先**：Markdown + 内容集合（Content Collections）写博客很顺手
- **部署省心**：一条 GitHub Actions 自动构建，推到 GitHub Pages 即可

## 一个例子

写代码块和平时用 Markdown 一样：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct")
params = SamplingParams(temperature=0.7, max_tokens=256)

outputs = llm.generate(["请介绍一下你自己"], params)
print(outputs[0].outputs[0].text)
```

## 接下来写什么

计划先从这几个方向开始：

1. vLLM 推理加速的入门笔记与调优经验
2. Qwen3-VL / Qwen3.5 多模态模型的部署与优化
3. 智能标注 Agent 的设计与实践

如果有想交流的话题，欢迎通过 [GitHub](https://github.com/zkl-ai) 或邮件联系我。
