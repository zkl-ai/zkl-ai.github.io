---
title: "vLLM 核心机制学习笔记（附参考资料）"
description: "基于 vLLM 官方博客 Anatomy of vLLM 整理的学习笔记：Continuous Batching、PagedAttention 与调度器。"
pubDate: 2026-10-18
tags: ["成长计划", "推理加速", "vLLM"]
draft: true
category: inference-acceleration
---

> **本文性质说明**：这是一篇**学习笔记**，主要内容整理自 vLLM 官方博客
> [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
> （作者 Aleksa Gordić，2025 年 9 月），并结合 vLLM 论文、官方文档与源码。
> 文中内容为本人阅读后的**理解与提炼，非原文翻译**，具体表述以原文为准；引用处均在下文「参考资料」列出。

## 一、这篇文章在讲什么

作者用一个「倒金字塔」的方式组织全文：先给一段最简的离线推理代码，以此为线索，逐步揭开一个高吞吐推理引擎的全貌——

1. **引擎骨架**：LLM Engine / Engine Core 是怎么组成的
2. **核心循环**：请求如何进来、被调度、完成前向
3. **高级特性**：chunked prefill、prefix caching、guided decoding、speculative decoding、P/D 分离
4. **扩展到多卡/多机**：从单 GPU 到 TP/PP/DP
5. **Benchmark**：怎么测延迟和吞吐

我这份笔记只聚焦第 1、2 部分（也就是标题里的 Continuous Batching 和 PagedAttention），其余留作后续专题。

## 二、核心概念笔记

### 1. 引擎骨架：LLM Engine 与 Engine Core

- **LLM Engine** 是对外接口：离线是 `LLM`，在线是 `AsyncLLMEngine`。它自己只能做高吞吐推理，还不能对外提供 Web 服务。
- **Engine Core** 才是干活的循环，内部有：**Model Executor**（驱动前向）、**Scheduler**（决定每步跑谁）、**KV Cache Manager**（管显存，PagedAttention 的心脏）。

我的理解：把 Engine 想成「门面」，Engine Core 想成「发动机」，KV Cache Manager 则是发动机里最关键的「供油系统」。

### 2. Continuous Batching（连续批处理）

- 传统静态批处理要等一批里最慢的请求跑完才整批结束，GPU 大量空转。
- vLLM 把调度粒度细化到 **token 级**：每生成一个 token，就把完成的请求移走、把新请求加进来，GPU 每一步尽量跑满。
- 实现上，它把这一批所有序列**拼接成一条扁平序列**，靠 position 和 attention mask 保证每个 token 只看自己序列的前文。

我的理解：Continuous Batching 解决的是「**算力利用率**」问题。

### 3. PagedAttention（分页注意力）

- KV cache 是显存大头；传统「连续分配」会带来碎片化、过度预留、无法共享三个毛病。
- PagedAttention 借用操作系统的**虚拟内存分页**：把 KV cache 切成固定大小 **block**（默认 16 token/块），每个序列用一张 **block table** 记录逻辑块到物理块的映射。
- 空闲 block 统一放在 `free_block_queue` 池子里，按需取用、用完归还；因为块大小统一，几乎没碎片；因为按需分配，不再按 max_len 预留；因为可被多序列引用，支持 KV 共享（前缀缓存的基础）。

我的理解：PagedAttention 解决的是「**显存利用率**」问题；和 Continuous Batching 其实是同一件事的两面——扁平序列 + 索引表。

### 4. Scheduler（调度器）

- 每步要决定「跑谁、给多少 token 预算」。工作负载分两类：**prefill**（对整段 prompt 前向，compute-bound）和 **decode**（只对最新 token 前向，memory-bandwidth-bound）。
- V1 调度器能在同一步里混合 prefill 和 decode；调度策略支持 **FCFS** 和 **priority**。
- `allocate_slots` 做三件事：算要几个 block → 查够不够（不够就抢占，V1 主要是 recompute）→ 从池子分配并登记到 `req_to_blocks`。

### 5. 一次前向传播的步骤

1. 更新状态（清掉完成请求、更新 block table）
2. 准备输入（算 position、`slot_mapping`）
3. 前向（扁平序列 + paged attention kernel，或用捕获好的 CUDA Graph 重放）
4. 取每个序列最后一个位置的 hidden state 算 logits
5. 采样

## 三、我的理解与疑问（待深入）

- `slot_mapping` 具体是怎么把「扁平序列里的 token」映射到「KV cache 的 slot」的？还没完全吃透。
- V1 引擎和 V0 的关键差异除了「prefill/decode 可混跑」，还有哪些？
- 多模态模型（Qwen3-VL）的 KV cache 有什么特殊？视觉 token 和文本 token 在分页时是一视同仁还是分开管理？（这条和后面的 Visual Token Compression 专题直接相关）

## 四、参考资料

1. Aleksa Gordić, *Inside vLLM: Anatomy of a High-Throughput LLM Inference System*, vLLM Blog, 2025-09-05. <https://vllm.ai/blog/2025-09-05-anatomy-of-vllm>
2. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
3. vLLM 官方文档：<https://docs.vllm.ai>
4. vLLM 源码：<https://github.com/vllm-project/vllm>

## 版权说明

本文为个人学习笔记，整理时参考并标注了上述公开资料，未复制原文大段内容；如涉及版权问题，请联系我删除。
