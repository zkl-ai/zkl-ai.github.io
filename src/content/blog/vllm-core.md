---
title: "vLLM 核心机制：Continuous Batching 与 PagedAttention"
description: "吃透 vLLM 的调度与显存管理，建立推理加速基线。"
pubDate: 2026-10-18
tags: ["成长计划", "推理加速", "vLLM"]
draft: true
category: inference-acceleration
---

> 本文是「多模态大模型推理加速」方向的第一篇：把 vLLM 的两大核心机制——Continuous Batching 和 PagedAttention——从原理讲到源码，最后落到一份可复用的推理基线报告。

## 学习目标

读完并做完实验后，应该能回答这几个问题：

1. 为什么用 HuggingFace Transformers 直接推理，吞吐会这么低？卡点在哪？
2. Continuous Batching 相比静态批处理，到底解决了什么问题？
3. PagedAttention 的「分页」思想是怎么来的？它如何解决 KV cache 的碎片化和共享问题？
4. vLLM 的调度器每步在做什么？preemption（抢占）有哪两种方式？
5. 能独立压测出一个 Qwen3-VL 服务的 P50/P99 延迟、吞吐和显存基线。

## 背景：两个「历史包袱」

### 1. KV cache 是显存的主要占用

自回归生成时，第 t 个 token 的 attention 要看到前面所有 token。如果每步都重算，复杂度是 O(n²)；所以把每层的 Key/Value 缓存下来，生成第 t 个 token 时只算新 token 的 KV 并 append。

KV cache 的显存量级：

```
KV cache ≈ 2 × num_layers × hidden_size × seq_len × batch × dtype
```

长序列 + 大 batch 时，KV cache 会远远超过模型权重本身，成为显存瓶颈。**推理服务要解决的核心问题，本质上是「如何高效管理这块 KV cache 内存」。**

### 2. 传统静态批处理浪费算力

Transformers 的 `generate` 是「一个请求一个 batch」，多个请求只能排队串行；而早期的静态批处理是「一批请求一起进来，等最慢的那个生成完，整批才结束」——木桶效应，GPU 大量时间在空转。

## 核心机制一：Continuous Batching（连续批处理）

Continuous Batching 把调度粒度从「请求级别」细化到「token 级别」：

- 每个 step 生成一个 token 后，**立即**把已完成的请求移出，把新请求加入
- 不再等整批结束，GPU 每一步都尽量跑满

效果：吞吐量相比静态批处理通常提升一个数量级。这是现在几乎所有推理框架（vLLM / SGLang / TensorRT-LLM）的标配。

> 关键理解：Continuous Batching 解决的是「**算力利用率**」问题；而下面 PagedAttention 解决的是「**显存利用率**」问题。两者独立，但配合起来才构成 vLLM 的高吞吐。

## 核心机制二：PagedAttention（分页注意力）

### 问题：连续分配的 KV cache 有三个毛病

传统做法是给每个序列预留一块**连续**的显存放 KV cache，导致：

1. **碎片化**：不同序列长度不同，释放后留下大小不一的空洞，新的序列塞不进去
2. **过度预留**：不知道序列会生成多长，只能按 `max_len` 预留，大量浪费
3. **无法共享**：并行采样 / beam search 会共享相同前缀，但连续分配没法让多个序列复用同一段 KV

### 解法：借鉴操作系统的虚拟内存分页

PagedAttention 把 KV cache 切成**固定大小的 block**（默认 16 个 token 一块），每个序列通过一张 **block table** 记录它用到哪些 block。物理上不连续，逻辑上连续。

带来的好处：

- **几乎消除碎片**：block 大小统一，随时能复用
- **按需分配**：用到多少分多少，不再按 `max_len` 预留
- **KV cache 共享**：并行采样、beam search 可以共享公共前缀的 block，显存省一大截

> 参考：vLLM 论文《Efficient Memory Management for Large Language Model Serving with PagedAttention》（SOSP 2023）。这篇论文是理解 PagedAttention 的第一手材料。

## 核心机制三：调度器（Scheduler）

调度器每步决定「哪些序列参与这次 forward」。序列大致有这几个状态：

- **WAITING**：已排队，还没进显存
- **RUNNING**：正在生成
- **SWAPPED**：被抢占，KV cache 换到了 CPU 内存
- **FINISHED**：完成

显存不够时，调度器会**抢占（preemption）**一部分序列，两种方式：

- **Swap**：把 KV cache 换到 CPU 内存，省显存但恢复慢
- **Recompute**：直接丢弃 KV，恢复时重算，快但费算力

另外还有 **chunked prefill**：把很长的 prompt 切块处理，避免 prefill 阶段显存峰值过高、也降低首 token 延迟。

## 一个请求的生命周期

```
Tokenizer → Engine/Scheduler（WAITING → RUNNING）
          → Model Runner（prefill / decode）
          → Sampler（greedy / beam / sampling）
          → 命中 EOS 或 max_tokens → FINISHED
```

> 建议自己画一张带 block table 的链路图，画完才算真正理解了。

## 源码 / 实验

### 读源码（重点文件）

- `vllm/core/scheduler.py`：调度逻辑，continuous batching + preemption 都在这里
- `vllm/attention/`：PagedAttention 的实现
- `vllm/worker/`：模型前向执行的 worker
- `vllm/engine/`：请求入口，LLM / AsyncLLM 的封装

### 动手实验

- [ ] 用 vLLM 起一个 Qwen3-VL 服务（`vllm serve`）
- [ ] 写一个压测脚本，固定 prompt 长度，测不同并发下的吞吐和延迟
- [ ] 用 `vllm` 自带的指标 / 或 `nvtop` 观察显存占用
- [ ] （TODO）记录你的具体命令和观测到的现象

## 可量化成果

填一张基线报告，作为后续所有优化的对比基准：

| 指标 | 数值（TODO 填） |
|------|----------------|
| 模型 | Qwen3-VL-xxB |
| 硬件 | TODO（如 A100 / H800） |
| 最大并发 | TODO |
| P50 延迟 | TODO |
| P99 延迟 | TODO |
| 吞吐（token/s） | TODO |
| 显存占用 | TODO |

> 这一步的基线很重要：后面做量化、投机解码、Visual Token Compression 时，都要拿它做 before/after 对比。

## 踩坑记录

- [ ] （TODO）记录你遇到的第一个坑，例如 `max_model_len` 设太大导致 OOM、`num_gpu_blocks` 相关报错、vLLM 版本与模型不兼容等

## 参考资料

- vLLM 论文：Efficient Memory Management for LLM Serving with PagedAttention（SOSP 2023）
- vLLM 官方文档：docs.vllm.ai
- 源码：github.com/vllm-project/vllm
