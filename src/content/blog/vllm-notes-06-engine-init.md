---
title: "vLLM 学习笔记（6）：引擎初始化"
description: "LLM(...) 构造时那条链做了什么，以及 KV cache 能开多少 block 到底是怎么算出来的。"
pubDate: 2026-10-04
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
---

第 5 章说 `LLM(...)` 构造的时候在做"准备"，准备的结果决定了这台机器**最多能同时服务多少个请求**。这一章就把那条链走完，落点是一个数字：`num_blocks`。

## 构造链长什么样

调用顺序在 `EngineCore.__init__`（`v1/engine/core.py`）里：

```text
（executor 初始化，内部会调 worker 的）
    init_device                      worker_gpu_worker.py:360
    load_model                       :500
                                     
determine_available_memory()          engine_core.py:308  → worker:528
    └─ 跑一次假前向，量出「非 KV 显存」，算出可用字节数
get_kv_cache_configs(...)             engine_core.py:319  → kv_cache_utils.py
    └─ 把字节数换算成 num_blocks
initialize_from_config(...)           engine_core.py:354  → worker:735
    └─ 按这个数量真正分配 KV cache 张量
compile_or_warm_up_model()            engine_core.py:356  → worker:773
    └─ CUDA Graph 捕获、torch.compile 预热
```

两个前提先做完：`init_device` 把显卡和通信组准备好，`load_model` 把权重读进显存。之后才轮到算 KV cache 的空间——因为**算之前得先知道权重占了多少**。

## 为什么必须真的跑一次前向

算"还剩多少显存给 KV cache"，直觉上应该是个减法：

```
可用 = 总显存 × 利用率 − 权重 − 其他
```

权重这一项好算：参数量乘字节数。但真正干扰的是**激活值的峰值显存**——它取决于 batch 里有多少 token、kernel 怎么实现、算子有没有融合、CUDA Graph 捕获了多少，静态估算基本不可靠。

所以 vLLM 干脆选择**跑一遍量出来**。`determine_available_memory()` 的文档就是这么写的：

> Profiles the peak memory usage of the model to determine how much memory can be used for KV cache without OOMs.

具体是调 `profile_run()`（`worker_gpu_worker.py:572`），而且**是按 `max_num_batched_tokens` 这个上限去跑的**（源码注释：`still need a profile run which compiles the model for max_num_batched_tokens`）。要量的就是峰值，所以得按最坏情况跑。

拿到测量结果后，核心就三行（`worker_gpu_worker.py:621`）：

```python
self.available_kv_cache_memory_bytes = (
    self.requested_memory
    - profile_result.non_kv_cache_memory
    - cudagraph_memory_estimate_applied
)
```

其中 `requested_memory` 是这么来的（`vllm/v1/worker/utils.py:539`）：

```python
requested_memory = math.ceil(
    init_snapshot.total_memory * cache_config.gpu_memory_utilization
)
```

注意乘的是**总显存**，不是空闲显存。而 `gpu_memory_utilization` 默认 **0.92**（`vllm/config/cache.py:103`）——留出来的那 8% 是给别的进程、显存碎片，以及那个可选的 CUDA Graph 估算留的余量。顺带一提源码会校验 `free_memory >= requested_memory`，不满足就直接报错让你调小，而不是硬着头皮往后跑。

所以公式其实是：

```
可用 KV 显存 = 总显存 × gpu_memory_utilization − 非 KV 显存（假前向量出来的）
```

## 从字节到块数

有了字节数，换算成块数是一步除法：

```
num_blocks = 可用 KV 显存 ÷ 一块的字节数
```

`一块的字节数` 的定义在 `vllm/v1/kv_cache_interface.py:519`：

```python
unpadded_page_size_bytes = (
    self.num_heads * self.num_states * self.state_content_size_bytes
)
# state_content_size_bytes = (head_size + head_size_v) * dtype 的字节数
```

翻译成人话，一块要存的东西是：

```
一块 = block_size（16 个 token）
     × num_kv_heads（KV 头数，GQA 下比 Q 头数少）
     × (head_size + head_size_v)   ← 这两项就是 K 和 V 各一份
     × dtype 字节数
```

`head_size_v` 默认等于 `head_size`，所以常见情况就是乘 2。拿 **Llama-3-8B** 实际算一下（32 层、8 个 KV head、head_dim 128、bf16）：

- **一个 token**：32 层 × 8 头 × 128 维 × 2（K/V）× 2 字节 = **131,072 字节 = 128 KiB**
- **一块**（16 个 token）：128 KiB × 16 = **2 MiB**

顺带解释了一个常见数字：Llama-3-8B 这个模型**每个 token 的 KV cache 就是 128 KiB**，所以 8192 的上下文一个请求要 1 GiB。想不起这个数的时候，从上面这个乘法推一遍就行。

这块是**逐层求和**的——`page_size_bytes` 是单层的规格，总的一块大小要把所有层加起来。这也是为什么 `get_kv_cache_spec()`（`worker_gpu_worker.py:719`）要把每一层的规格报给引擎核心。

## 走一遍完整数字

H100（80 GiB）+ Llama-3-8B，`gpu_memory_utilization` 用默认的 0.92：

| 步骤 | 算法 | 结果 |
|---|---|---|
| 请求的显存 | 80 GiB × 0.92 | 73.6 GiB |
| 非 KV 显存 | 权重 15 GiB（80 亿参数 × 2 字节）+ 激活峰值约 2 GiB | ≈ 17 GiB |
| 可用 KV 显存 | 73.6 − 17 | **56.6 GiB** |
| 块数 | 56.6 × 1024 ÷ 2 | **28,979 块** |
| 换算成 token | 28,979 × 16 | **463,664 个** |
| 最大并发请求数 | 463,664 ÷ 8192（`max_model_len`） | **≈ 56 个** |

（第三行那个"激活峰值约 2 GiB"是我给的估值，真实数字就是那次假前向量出来的；权重 15 GiB 是可以直接算的。）

所以启动日志里那两行长这样（格式来自官方文档）：

```text
INFO [kv_cache_utils.py] GPU KV cache size: 463,664 tokens
INFO [kv_cache_utils.py] Maximum concurrency for 8192 tokens per request: 56.60x
```

第二行的意思是：如果每个请求都跑满 8192 的上下文，最多能同时伺候 56 个。**这就是第 2 章那个 batch size 的硬上限**——不是调度器不想多排，是排不下。

## 那能调什么

看到这个数字之后自然想问：怎么让它变大？几个旋钮，代价各不同。

- **`gpu_memory_utilization` 调大**：直接抢更多显存，但挤的是同卡上别的进程，调太满容易 OOM。
- **`max_model_len` 调小**：每个请求占的块变少，并发数上去——代价是能处理的上下文变短。
- **模型量化 / 换小模型**：权重少占显存，省下的全归 KV cache。
- **`block_size` 调小**：第 4 章说过最后一块最多浪费 15 个 token，块越小这个浪费越小。但总容量是由字节数定死的，改块大小主要是减少浪费，顺带让块表变大。

注意这四个里**没有"把 batch 调大"**——batch size 是结果，不是旋钮。这正好呼应第 1 章那条链：吞吐被 batch 卡住，batch 被显存卡住。

## 最后两步

`initialize_from_config`（`worker_gpu_worker.py:735`）是真正分配张量的地方。这里有个细节值得注意：

```python
# Update local config with adjusted num blocks after profiling,
# so that it's available to the warmup stage.
self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
```

**worker 不决定块数，它只是接收。** 块数是在引擎核心里算的（`get_kv_cache_configs`，`engine_core.py:319`），worker 负责量出字节数并上报，再把算好的结果落到显存上。这就回到第 5 章那张分层图：worker 管执行，引擎核心管决策。

`compile_or_warm_up_model`（`:773`）做 CUDA Graph 捕获和 `torch.compile` 预热。它也会吃显存——就是前面公式里那个 `cudagraph_memory_estimate`，vLLM 甚至会拿它反推出"如果你想保持和以前一样的 KV cache 大小，`gpu_memory_utilization` 该调到多少"。

## 下一章看什么

骨架搭好、显存分好，接下来是每个 step 真正跑前向的那一刻。第 5 章提过，vLLM 的输入是**一条扁平序列**加一个 `query_start_loc`，而注意力算的时候还要靠第 4 章那张 block table 去取散落的 KV。这两件事怎么在 kernel 层面配合，就是第 7 章的内容。

## 参考资料

- vLLM 0.30.0 源码：`v1/worker/gpu_worker.py`、`v1/worker/utils.py`、`v1/engine/core.py`、`v1/core/kv_cache_utils.py`、`v1/kv_cache_interface.py`、`config/cache.py`
- vLLM Docs, [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)（`GPU KV cache size` 与 `Maximum concurrency` 两行日志的说明）
