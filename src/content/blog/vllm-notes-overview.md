---
title: "vLLM 学习笔记（总纲）"
description: "以 vLLM 0.30.0 源码为基线，从最简的离线例子出发，逐步搭建 vLLM 的知识体系。"
pubDate: 2026-09-29
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
---

这是一份边读源码边记的笔记。材料是 [vLLM 0.30.0 的源码](https://github.com/vllm-project/vllm/tree/v0.30.0)、PagedAttention 那篇论文，以及 vLLM 官方那篇 [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)。

写法上参考 Anatomy 的思路：从一个最简单的例子出发，先把骨架看清楚，再一层层加复杂度，不追求一次讲全。重点是别一上来就扎进细节里，那样容易没头绪。

实验模型统一用 `Qwen/Qwen3.5-0.8B`，小模型，真要跑的时候省事。

## 环境说明

vLLM 0.30.0 在 PyPI 上只有 manylinux 的 wheel，没有 macOS 版本；在 Apple Silicon 上从源码编译也不现实。所以想动手跑，得有一台 Linux + NVIDIA GPU 的机器（云 GPU 或者公司机器）。

所以现阶段以读源码、建立概念为主，动手环节留到有 GPU 环境时再补。

## 学习路线

**第一篇 · 宏观（先把直觉建立起来）**

1. [最小的离线示例](/blog/vllm-notes-01-minimal-offline/) —— 一次生成到底发生了什么
2. [从单请求到批处理](/blog/vllm-notes-02-batching/) —— 为什么 GPU 不能一个一个来
3. 调度器 —— 谁先跑、显存不够怎么办
4. PagedAttention —— 显存怎么分页管

**第二篇 · 深入源码**

5. 引擎骨架与源码地图 —— 分层结构、请求状态机
6. 引擎初始化 —— KV block 的数量是怎么算出来的
7. 前向传播 —— 扁平序列 + paged kernel

**第三篇 · 逐渐加复杂度**

8. 在线服务 —— 异步、流式、多进程
9. 高级特性 —— chunked prefill、prefix caching、guided decoding、speculative decoding、P/D 分离

**第四篇 · 扩展**

10. 多卡多机（TP / PP / DP / EP）

## 源码地图（0.30.0）

| 源文件 | 职责 | 关键类 / 方法（行号） |
|--------|------|----------------------|
| `entrypoints/llm.py` | 离线推理入口 | `LLM`（L67）、`generate`（L418） |
| `v1/engine/llm_engine.py` | 引擎门面 | `LLMEngine`（L48） |
| `v1/engine/core.py` | 引擎核心循环 | `EngineCore`（L105）、`step`（L589） |
| `v1/engine/core_client.py` | 前端 ↔ 核心通信 | `EngineCoreClient` |
| `v1/core/sched/scheduler.py` | 调度 | `Scheduler`（L79） |
| `v1/core/kv_cache_manager.py` | KV cache 管理 | `KVCacheManager`（L134） |
| `v1/worker/gpu_worker.py` | 单卡 worker | `init_device`(L360)、`load_model`(L500)、`determine_available_memory`(L528)、`initialize_from_config`(L735)、`compile_or_warm_up_model`(L773) |
| `v1/worker/gpu_model_runner.py` | 模型前向执行 | `GPUModelRunner` |
| `v1/request.py` | 请求对象与状态机 | `Request`、`RequestStatus`（L364） |

## 术语表

| 术语 | 含义 |
|------|------|
| prefill | 对整段 prompt 做一次前向，计算密集（compute-bound） |
| decode | 只对最新一个 token 做前向，读显存为主（memory-bandwidth-bound） |
| KV cache | 缓存每层 Key/Value，避免自回归时重复计算历史 |
| block | PagedAttention 的显存分配单位，默认 16 个 token |
| block table | 逻辑块到物理块的映射表 |
| `free_block_queue` | 空闲 KV block 池 |
| continuous batching | 调度粒度细化到 token 级，每步动态进出请求 |
| preemption | 显存不足时抢占部分请求（V1 主要用 recompute） |
| chunked prefill | 长 prompt 分块 prefill，避免独占一步 |

## 和 Anatomy 博客的差异

对照 0.30.0 和博客（基于 2025 年 8 月的 commit `42172ad`），有几处已经变了：

1. `renderer` 是新增组件（`llm_engine.py:91`），博客里没有。输入渲染被单独拆了出来。
2. 请求状态机从粗粒度的 WAITING / RUNNING / FINISHED 细化到了 12 个状态。
3. KV 管理多了一层 `coordinator`（`kv_cache_manager.py:170`），用来支持 MHA / MLA / Mamba 这类混合结构。

博客没写错，只是框架在往"支持更复杂模型结构"的方向演进。记这些差异，比记住某个静态结构有用。
