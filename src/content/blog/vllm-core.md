---
title: "vLLM 核心机制：Continuous Batching 与 PagedAttention"
description: "从一段最简的 vLLM 代码出发，沿着引擎的调用链路拆解连续批处理、分页注意力与调度器。"
pubDate: 2026-10-18
tags: ["成长计划", "推理加速", "vLLM"]
draft: true
category: inference-acceleration
---

> 这篇文章想回答一个朴素的问题：**为什么同样一张 GPU，用 vLLM 跑大模型，吞吐能比 HuggingFace Transformers 高出一个数量级？**
>
> 答案是它做对了两件看似不相关、其实缺一不可的事：用 **Continuous Batching** 解决「算力利用率」，用 **PagedAttention** 解决「显存利用率」。本文沿着 vLLM 引擎的真实调用链路，把这两件事拆开讲。

## 从一个最简例子开始

```python
from vllm import LLM, SamplingParams

prompts = ["介绍一下你自己", "今天天气怎么样"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=128)

llm = LLM(model="Qwen/Qwen3-VL-8B-Instruct")
outputs = llm.generate(prompts, sampling_params)
```

就这么几行。但 `LLM(...)` 构造和 `generate(...)` 这两次调用背后，是一整套为「高吞吐」设计的引擎。我们先建立一个心智模型，再往下钻。

## 心智模型：引擎的三个层次

vLLM 的核心可以抽象成三个层次：

- **LLM Engine**：对外接口。离线场景就是上面的 `LLM`，在线场景是 `AsyncLLMEngine`（配合 FastAPI 暴露成 OpenAI 兼容接口）。
- **Engine Core**：真正干活的循环。它内部有三样东西——**Scheduler**（决定每步跑哪些请求）、**Model Executor**（驱动模型前向）、**KV Cache Manager**（管理显存，PagedAttention 的心脏）。
- **请求**：一条请求从进来到离开，状态在 `WAITING → RUNNING → FINISHED` 之间流转。

一个请求的完整生命周期可以概括为：

```
Tokenizer → Scheduler（WAITING → RUNNING）→ 前向传播 → 采样 → 命中 EOS 或 max_tokens → FINISHED
```

## 引擎初始化：`LLM(...)` 构造时发生了什么

构造 `LLM` 时，最关键的几件事：

1. **加载模型**：实例化模型结构、加载权重、`model.eval()`。
2. **初始化 KV Cache**：跑一次 dummy 前向，算清楚「扣除权重后还剩多少显存」，再除以每个 block 的大小，得到总共能分多少个 KV block。这些 block 统一装进一个叫 `free_block_queue` 的空闲池里。
3. **捕获 CUDA Graph**：decode 阶段每一步的算子序列是固定的，vLLM 会预先「录制」成 CUDA Graph，之后每步直接重放，省掉海量的 kernel 启动开销。

这里有一个概念要提前记住：**KV block**。PagedAttention 把 KV cache 切成固定大小的块，默认 **16 个 token 一块**。一个标准 attention 层的 block 显存是：

```
2（K/V）× block_size(16) × num_kv_heads × head_size × dtype字节数
```

整个引擎的显存管理，就是围绕「怎么分配、复用、回收这些 block」展开的。

## Continuous Batching：调度粒度从「请求」细化到「token」

### 老问题：静态批处理的木桶效应

早期的批处理是把一批请求一起塞进模型，**等最慢的那个生成完，整批才算结束**。这就像等一桌人里吃得最慢的人吃完才上下一道菜——GPU 大量时间在空转。

### vLLM 的做法：每一步都可以换人

vLLM 的引擎是一个 `step()` 循环，每步只做三件事：

1. **Schedule**：从 `running` 和 `waiting` 队列里挑出这一步要跑的请求
2. **Forward**：跑一次模型前向，每个请求采样一个 token
3. **Postprocess**：把 token 拼回序列，检查是否命中停止条件

关键在于：**每生成一个 token，就可以把已完成的请求移走、把新请求加进来**。不再有「整批必须一起结束」的约束，GPU 每一步都被尽量填满。

> 顺带厘清两个术语：**prefill** 是对整段 prompt 做前向（计算密集，compute-bound）；**decode** 是只对最新一个 token 做前向（读显存为主，memory-bandwidth-bound）。Continuous Batching 的意义主要在于把大量「单 token」的 decode 请求高效地打包在一起。

### 具体怎么实现的：把所有序列拼成一条「超长序列」

这是 Continuous Batching 最巧的地方。vLLM 不把每个请求单独 padding 成一样长（那样浪费），而是把这一批所有请求的 token **首尾拼接成一条扁平序列**，交给自定义的 attention kernel 一次算完。靠的是两样东西：

- `position_ids`：告诉 kernel 每个 token 在它自己序列里的位置
- attention mask：保证每个 token **只看到自己序列里的前文**，不会跨序列「串味」

所以从 kernel 的角度，它看到的永远是「一条序列 + 一张索引表」，天然支持任意数量的请求混在一起。

## PagedAttention：给 KV cache 做「分页」

Continuous Batching 解决算力，但显存问题还悬着。而显存的大头，是 KV cache。

### 为什么 KV cache 是瓶颈

自回归生成第 t 个 token 时，attention 要看前面所有 token 的 Key/Value。如果不缓存，每步都要重算全部历史，复杂度是 O(n²)。所以把每层的 K/V 存下来，第 t 步只算新 token 的 KV 再 append。

代价是显存：KV cache 随「层数 × 序列长度 × batch」线性增长，长序列 + 大并发时，它会轻松超过模型权重本身。

### 连续分配的三个毛病

传统做法是给每个序列预留一段**连续**显存放 KV cache，这带来三个问题：

1. **碎片化**：不同请求长度不同，释放后留下大小不一的空洞，新请求塞不进去
2. **过度预留**：不知道会生成多长，只能按 `max_len` 预留，大半浪费
3. **无法共享**：并行采样、beam search 共享同一个前缀，但连续分配没法让它们复用同一段 KV

### 分页思想：block + block table

PagedAttention 直接借用了操作系统的**虚拟内存分页**：

- 把 KV cache 切成固定大小的 **block**（默认 16 token）
- 每个序列用一张 **block table** 记录「我的第 i 个逻辑 block 存在哪个物理 block」
- 物理上不连续，逻辑上连续

调度器里那个 `free_block_queue`，就是一个装空闲 block 的池子。请求需要新 block 时从池子里取，用完还回池子。因为 block 大小统一，**几乎不再有碎片**；因为按需分配，**不再按 max_len 预留**；因为 block 可以被多个序列的 block table 同时引用，**KV cache 可以共享**（这是前缀缓存 prefix caching 的基础，留到下一篇讲）。

> 读到这里你会发现，PagedAttention 和 Continuous Batching 其实是同一个东西的两面：**扁平序列 + 索引表**。扁平序列让算力能被塞满，索引表（block table / slot mapping）让显存能被高效管理。理解了这一层，vLLM 的核心就通了。

## 调度器：每一步的「决策中枢」

Scheduler 每步要回答「这一批跑谁、给多少 token 预算」。大致逻辑：

1. **先照顾 decode**（已经在 `running` 的请求）：算它这一步要生成几个 token（投机解码时可能不止 1 个），调用 KV cache manager 分配 slot
2. **再处理 prefill**（`waiting` 队列里的新请求）：分配 prompt 所需的 block，把请求从 `waiting` 挪到 `running`
3. 每步有一个 **token budget**，超了就停止加人

而 `allocate_slots` 内部做三件事：

1. **算要几个 block**：比如 prefill 有 17 个新 token，`ceil(17/16) = 2` 个 block
2. **查够不够**：池子里 block 不够时，触发**抢占（preemption）**——V1 里主要是 recompute（丢弃部分 KV，恢复时重算），V0 里还有 swap（KV 换到 CPU 内存）
3. **分配**：从 `free_block_queue` 取 block，登记到 `req_to_blocks` 映射表

调度策略上，V1 支持 **FCFS**（先来先服务）和 **priority**（高优先级先跑）。

## 一次前向传播的内部

调度器挑好人之后，前向传播大致分五步：

1. **Update states**：清掉已完成的请求，更新 block table、采样元数据
2. **Prepare inputs**：把 CPU 侧的 token/位置拷到 GPU，算出 `slot_mapping`（每个 token 对应写到哪个 KV slot）
3. **Forward**：跑模型，attention 用自定义的 paged kernel，扁平序列 + 掩码一次算完
4. **Gather last-token states**：只取每个序列最后一个位置的 hidden state，算 logits
5. **Sample**：按采样参数（greedy / temperature / top-p / top-k）出 token

如果是 decode 且没关 CUDA Graph，第 3 步就是「重放录制好的图」，这也是 decode 低延迟的关键之一。

## 接下来可以深入的方向

到这里，Continuous Batching 和 PagedAttention 的主干就通了。再往前，还有几个「长在这套机制上」的高级特性，正好对应这个系列后续的文章：

- **Chunked prefill**：长 prompt 分块处理，避免单个长请求独占一步（对应「投机解码」之外的一篇）
- **Prefix caching**：复用相同前缀的 KV，是 PagedAttention 共享能力的直接应用
- **Speculative decoding**：小模型草稿 + 大模型校验，下一篇的主角
- **Disaggregated P/D**：把 prefill 和 decode 拆到不同实例，各自扩缩容

## 参考资料

- vLLM 论文：Efficient Memory Management for LLM Serving with PagedAttention（SOSP 2023）
- vLLM Blog：Inside vLLM: Anatomy of a High-Throughput LLM Inference System
- vLLM 官方文档：docs.vllm.ai（重点看 V1 engine 指南）
- 源码：`vllm/core/scheduler.py`、`vllm/attention/`、`vllm/worker/`、`vllm/v1/`
