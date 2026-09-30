---
title: "vLLM 学习笔记（2）：从单请求到批处理"
description: "把长度不齐的请求拼进一次前向，以及静态批处理为什么会中途变空、连续批处理改了什么。"
pubDate: 2026-09-30
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
---

上一章最后算了一笔账：单请求 decode 的算术强度只有 1 FLOP/字节，而 H100 的平衡点在 300 FLOP/字节附近。差 300 倍的意思是，它绝大部分时间在等显存把权重送过来，计算单元干等着。

要把权重搬运摊薄，只有一条路：让多个请求共享同一次前向，也就是 batch。这一章不再重复"为什么要 batch"，只解决两个问题——**batch 具体是怎么塞进一次前向的**，以及**它为什么会中途变空**。

## 长度不齐的序列怎么拼成一次前向

一次前向要求输入是个规整的张量，可 prompt 的长度是乱的。

最朴素的办法是 padding：补到一样长。三条 prompt 长度分别是 3、5、2，全部补到 5：

```text
        位置 1  2  3  4  5
prompt1    A  B  C  _  _      补 2 个 pad
prompt2    D  E  F  G  H
prompt3    I  J  _  _  _      补 3 个 pad
```

能算，但代价就摆在图里：15 个位置里只有 10 个是真的 token，**三分之一的计算花在 pad 上**。而且 pad 位置还必须在注意力里屏蔽掉，否则 prompt1 会去"注意"两个毫无意义的位置。

真正躲不开 padding 的是 prefill，因为 prompt 长度天差地别。decode 反而好办——每条序列这一步都只添 1 个新 token，新 token 的个数是齐的。

顺着这个思路还能想通一件事：为什么注意力和矩阵乘不能一视同仁地批处理。矩阵乘、FFN 这类算子对每条序列的计算完全独立，拼成大矩阵一起算就行；而注意力要求每条序列只能在**自己**的 token 范围内做点积，长度不一样就不好拼。Orca 那篇论文里专门给这个现象起了个名字叫 **selective batching**——只对"能批"的那部分算子做批处理，注意力另行处理。

vLLM 干脆连 padding 都不做。它把这一步要算的所有 token 首尾相接排成一条扁平序列，另外给一个数组 `query_start_loc` 记每条序列的起点。还是刚才那三条长度 3、5、2 的 prompt：

```text
A  B  C  D  E  F  G  H  I  J
0  1  2  3  4  5  6  7  8  9
└───────┘└─────────────┘└────┘
0        3              8   10
prompt1  prompt2        prompt3
```

`query_start_loc = [0, 3, 8, 10]` 有 4 个数，但只有 3 条序列——**最后那个 10 不是谁的起点，是总长度**，也就是第 3 条序列的终点。这么存的好处是任意一条序列占多少 token 一减就出来：$3-0=3$、$8-3=5$、$10-8=2$，和三条 prompt 的长度正好对上。

这个数组在 `worker_gpu_model_runner.py:2080` 附近被填好（源码里就是把各序列长度累加起来，再在末尾补一个总数），GPU 上按这些边界把扁平序列切开。细节留到第 7 章，这里先记结论：**padding 是可以绕开的，不是必须付的代价**。

## 静态批处理：一批一起开始，一起结束

朴素地用起 batch，就是"攒够 B 个请求一起跑，等整批都结束，再开下一批"。这叫静态批处理。

规则本身没问题，跑起来是这样的。假设一次能跑 3 个请求（3 个槽位），来了 6 个请求，输出长度分别是 2、3、5、4、6、3：

| 请求 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A（长度 2） | █ | █ | | | | | | | | | |
| B（长度 3） | █ | █ | █ | | | | | | | | |
| C（长度 5） | █ | █ | █ | █ | █ | | | | | | |
| D（长度 4） | | | | | | █ | █ | █ | █ | | |
| E（长度 6） | | | | | | █ | █ | █ | █ | █ | █ |
| F（长度 3） | | | | | | █ | █ | █ | | | |
| **在跑的请求数** | 3 | 3 | 2 | 1 | 1 | 3 | 3 | 3 | 2 | 1 | 1 |

看最后一行——它从没稳定在 3 上。第 1、2 步是满的，第 3 步开始掉，第 4、5 步只剩 1 个；换到批 2 又是同样的剧本重演一遍。

掉下来的原因有两个。**批内**：A 在第 2 步就写完了，B 在第 3 步写完，它们的槽位不会让给别人，只能等 C 跑到第 5 步。**批间**：D、E、F 明明在第 3 步就已经有空位了，却要等到第 6 步才开始。

用一整批的量算一下利用率。批 1 的"容量"是 3 个槽位 × 5 步 = 15，实际用掉 2 + 3 + 5 = 10；批 2 容量是 3 × 6 = 18，实际用掉 4 + 6 + 3 = 13。合起来 23 / 33 ≈ **70%**。

写成公式更清楚。一批请求的长度记作 $L_i$，批大小为 $B$，那么这一批要跑 $\max_i L_i$ 步、产出 $\sum_i L_i$ 个 token，于是

$$
U = \frac{\sum_i L_i}{B \cdot \max_i L_i}
$$

分子是真正干的活，分母是为这批预留的槽位·步。$L_i$ 越不齐，$\max_i L_i$ 就越是被少数几条长请求撑起来，$U$ 就越低。

## 连续批处理：把换人的时机挪到每一步

连续批处理只改一件事：**换人的时机**。从"整批结束"提前到"每一步结束"。

还是那 6 个请求、3 个槽位。规则变成：每跑完一步，就把写完的请求移出去，再从排队的人里补进来。

| 请求 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A（长度 2） | █ | █ | | | | | | | |
| B（长度 3） | █ | █ | █ | | | | | | |
| C（长度 5） | █ | █ | █ | █ | █ | | | | |
| D（长度 4） | | | █ | █ | █ | █ | | | |
| E（长度 6） | | | | █ | █ | █ | █ | █ | █ |
| F（长度 3） | | | | | | █ | █ | █ | |
| **在跑的请求数** | 3 | 3 | 3 | 3 | 3 | 3 | 2 | 2 | 1 |

A 在第 2 步结束时腾出槽位，D 第 3 步就进来了；B 第 3 步腾位，E 第 4 步进来；C 第 5 步腾位，F 第 6 步进来。同样多的活（还是 23 个 token），从 11 步缩到 9 步，利用率从 70% 涨到 23 / 27 ≈ **85%**。

| | 总步数 | 产出 token | 槽位·步 用掉 / 总共 | 利用率 |
|---|---|---|---|---|
| 静态批处理 | 11 | 23 | 23 / 33 | 70% |
| 连续批处理 | 9 | 23 | 23 / 27 | 85% |

这个思路的正式名字是 **iteration-level scheduling**（迭代级调度），出自 Orca（OSDI '22）。它的摘要把静态批处理的毛病说得比我上面这段还直白：

> requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes.

一批里先做完的请求没法返回给客户端，新到的请求又必须等这一批彻底跑完。Orca 的办法是"把执行的调度粒度从请求降到迭代——调度器每次只让执行引擎跑模型的一步"。

所以连续批处理不是"多了一个 batch 的开关"，而是**换人这件事从批的边界挪到了步的边界**。vLLM 采纳了这个思路，它自己的贡献在另外一半（显存），上一章说过。

## 但它并不能填满所有空档

上面那个例子里连续批处理能赢，靠的是一个前提：**有人在排队**。

把长度改一下就看出来了。还是 3 个槽位、6 个请求，但长度是 2、3、5、4、6、**40**：

| 阶段 | 步数 | 同时在跑的请求数 |
|---|---|---|
| 第 1–6 步 | 6 | 3 |
| 第 7–9 步 | 3 | 2 |
| 第 10–45 步 | 36 | 1 |

那条 40 的请求在第 6 步进场，之后一直占着一个槽位跑到第 45 步。前面五条陆续都结束了，可后面 36 步里 3 个槽位只有 1 个在干活。

有意思的是，这个例子里静态批处理的总步数**也是 45 步**——一模一样。静态做法把 40 那条放在批 2 里，从第 6 步开始跑，同样跑到第 45 步。两者的槽位·步都是 60 / 135 ≈ 44%。

也就是说，在"所有请求同时到达、之后不再来"的场景下，连续批处理**一点吞吐都没多给**。它买到的是延迟：那五条短请求不用陪着长请求干等，能早得多地返回给客户端。

吞吐的收益要等到请求**持续到达**时才出现。稳态下，只要队列不空，连续批处理每个 step 都是 3 个槽位满的，也就是每步 3 个 token。静态批处理每批的每步产出是 $\sum L_i / \max L_i$：上面两批分别是 10/5 = 2 和 50/40 = 1.25，如果这两批交替出现，平均是 (10 + 50) / (5 + 40) ≈ 1.33。倍数是 3 ÷ 1.33 ≈ **2.3 倍**。

这个倍数不是常数，它取决于两件事：请求长度的分布（越偏越大）和队列的深度（越浅越小）。队列空了，连续批处理也只能干等。

所以更准确的说法是：**连续批处理消除的是"人为的等待"，不是"固有的不均衡"**。前者是批边界造成的，后者来自长度分布本身，得靠别的手段——比如把先算完的结果尽早返回（流式输出），或者干脆把长请求和短请求分开服务。

## 落到 vLLM 上

概念到了源码里，就是几行注释的事。`Scheduler.schedule()`（`vllm/v1/core/sched/scheduler.py:561`）开头有这么一段：

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. ... At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

调度器里没有"decode 阶段"也没有"prefill 阶段"。每个请求身上只有两个数：已经算了多少 token、一共需要算多少 token。每一步调度器做的事，就是给请求分一点 token 配额，让前者追上后者。这正是连续批处理的思路——**不按阶段想问题，只按步想问题**。

这两个变量名值得单独记一下，因为它们和"阶段"是一回事的两面说法。`num_computed_tokens` 是"这个请求已经算过多少 token"；`num_tokens_with_spec` 是"它一共有多少 token"——prompt 加已生成的部分，再加上投机解码草拟出来、还没验证的那些。`request.py:292` 的定义就是 `len(self._all_token_ids) + len(self.spec_token_ids)`，名字里的 `spec` 是 speculative 的缩写。两者之差，就是这一步还欠它多少 token。

`schedule()` 内部的顺序也照着这个来。先是第 610 行一句 `# First, schedule the RUNNING requests.`，把已经在跑的请求续上；接着是第 854 行 `# Next, schedule the WAITING requests.`，再回头看排队的能不能补进空出来的位置。请求跑完之后由 `_free_request`（第 2561 行）收尾，KV cache 的块跟着回收，下一次 `schedule()` 就能把这些位置分给别人。

V0 不是这样的。那一版的序列上挂着一个显式的 `SequenceStage`（`PREFILL` / `DECODE`），调度器也按这个划分出两套策略：默认那条 `_schedule_default()` 的文档写的是"先尽量把 prefill 塞满，再排 decode"，要混批得打开 `enable_chunked_prefill` 走另一条路径 `_schedule_chunked_prefill()`。

V1 把两条路合成了一条。官方的说法是它 "removes the traditional distinction between 'prefill' and 'decode' phases"，不再区分用户给的 prompt token 和模型生成的 output token。理由是 V0 那些特性各自独立开发，难组合（"Features were often developed independently, making it difficult to combine them effectively and cleanly"）；而 `{request_id: num_tokens}` 这种统一表示恰好能同时表达 chunked prefill、prefix caching 和投机解码。于是 V1 干脆把混批变成默认行为——V0 里 `enable_chunked_prefill` 默认是关的（只有上下文超过 32K 的模型会自动打开），V1 则直接写死 `self.enable_chunked_prefill = True`，注释只有一句 `# V1 always uses chunked prefills.`。

不过这里冒出一个新问题：一个新请求要进场，得先做 prefill；而这一步里同时还有别的请求在做 decode。prefill 是计算密集，decode 是访存密集，硬塞进同一个 step 会互相拖累。这件事留到第 9 章讲 chunked prefill 时再说。

## 还没解决的问题

上面说"把写完的请求移出去、从排队的人里补进来"，听着简单，但一到实现全是坑：每一步的 token 预算是多少、怎么分给各个请求？显存不够的时候把谁踢出去，被踢的那个已经算好的 KV cache 怎么办、回来时要不要重算？短的请求要不要优先，好让平均延迟好看一点？

再往下还有一层：既然每个 step 都要重新决定"这一批跑谁"，那这个决定就是整个系统的枢纽——它同时决定了吞吐和延迟。这是调度器（scheduler）的活，也是第 3 章的全部内容。

## 参考资料

- vLLM 0.30.0 源码：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/request.py`
- 对照 V0 用的源码（v0.9.0 这个 tag）：`vllm/core/scheduler.py`、`vllm/sequence.py`、`vllm/engine/arg_utils.py`
- vLLM Team, [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release)，2025-01-27
- Yu et al., [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)，OSDI '22
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)，2025-09-05
- Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention，SOSP 2023
