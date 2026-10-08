---
title: "vLLM 学习笔记（4）：PagedAttention"
description: "KV cache 的显存到底浪费在哪，分页怎么把浪费从 60-80% 压到 4% 以下。"
pubDate: 2026-10-02
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
---

第 3 章末尾留了个问题：调度器决定"把谁踢出去"的时候，它得先知道"还能不能给这条请求分配块"。那个判断是 KV cache 管理器给的。这一章看它凭什么能给——以及它管的东西到底有多麻烦。

## 先看浪费有多少

PagedAttention 那篇论文里有一句很不客气的判断：**现有系统浪费了 60%–80% 的显存**，原因是两点——过度预留（over-reservation）和碎片（fragmentation）。

拿个具体的例子看。一个 512 token 的 prompt，模型实际只生成了 30 个 token 就撞上 EOS：

- **传统做法**：sequence 长度事先不知道，只能按 `max_model_len` 预留连续空间。假设上限 2048，那就得先占下 2048 个 token 的 KV 位置。实际只用了 542 个——**浪费 73.5%**。
- 更糟的是，这份浪费不是一个人的事。显存被白占着，就意味着能同时跑的请求变少，而 batch size 直接决定吞吐（第 2 章那笔账）。

另外两种浪费也都是"连续分配"逼出来的：**内部碎片**（预留的连续块没用满，尾巴空着）和**外部碎片**（大小不一的连续区间之间留下的空洞，谁也塞不进去）。它们的共同根源是同一个假设：**一个序列的 KV 必须待在连续显存里**。

翻回论文摘要那句话，它点的其实是一整条因果链：

> the KV cache memory for each request is huge and grows and shrinks dynamically. When managed inefficiently, this memory can be significantly wasted by fragmentation and redundant duplication, **limiting the batch size**.

注意最后三个词——浪费显存不是终点，**卡住 batch size 才是**。

## 分页：把 KV cache 当成虚拟内存

解法是从操作系统借的。论文自己给了一组类比，我觉得这三个对应关系是理解整章的钥匙：

| 操作系统 | KV cache |
|---|---|
| 页（page） | **块（block）** |
| 字节（byte） | **token** |
| 进程（process） | **序列（sequence）** |

具体做法：把一个序列的 KV cache 切成**固定大小的块**，默认 **16 个 token** 一块（`vllm/config/cache.py:71` 的 `DEFAULT_BLOCK_SIZE = 16`）。块之间**不必连续**，另外用一张 **block table** 记录"第几个逻辑块 → 第几个物理块"。

于是那个 100 token 的序列长这样：

```text
逻辑序列被切成 7 个逻辑块

  逻辑块    [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ]
  覆盖的   0-15 16-31 32-47 48-63 64-79 80-95 96-99
  token                                          ↑ 这块只用了 4 个位置

block table 把逻辑块翻译成物理块

  逻辑块 0 → 物理块 17      逻辑块 4 → 物理块  5
  逻辑块 1 → 物理块  3      逻辑块 5 → 物理块 31
  逻辑块 2 → 物理块 42      逻辑块 6 → 物理块  8
  逻辑块 3 → 物理块 12

显存里的物理块池（位置任意）

  ... [ 3][ 4][ 5] ... [ 8] ... [12] ... [17] ... [31] ... [42] ...
      同一个序列的 7 个块散落在各处，中间夹着别人的块
```

**没有"预留"这个概念了。** 物理块是**按需分配**的：生成到第 16 个 token 就分第二块，到第 32 个就分第三块。因为块之间不需要连续，分配器也就不用去找"一块够大的连续空间"——外部碎片从根上消失了。

## 浪费被压到了哪里

按需分配之后，浪费只剩一个来源：**最后一块没用满**。

- 一个序列最多浪费 **15 个 token**（16 个位置只用了 1 个的情况）
- 用平均算大概是半块，8 个 token 左右

回到开头那个例子：512 的 prompt + 30 个输出 = 542 个 token，需要 $\lceil 542/16 \rceil = 34$ 块，也就是 544 个位置，**浪费 2 个**——从 73.5% 掉到 **0.37%**。

论文的说法是一致的：

> In PagedAttention, memory waste only happens in the last block of a sequence. In practice, this results in near-optimal memory usage, with a mere waste of **under 4%**.

从 60%–80% 到 4% 以下，省下来的全是本来被白占、拿不回来给别人的那份显存。

## 第二种浪费：重复

前面那个"浪费"只讲了显存不够用。但摘要里还提了一个词——**redundant duplication**（冗余重复）。这是另一类浪费，分页恰好也顺手解决了。

原因是有些请求之间**本来就有相同的 KV**：

- **同一个 prompt 采多个输出**（parallel sampling / best-of-n）：prompt 部分的 KV 一模一样，没必要存 N 份。
- **beam search**：不同 beam 共享前缀，只在中途分叉。

在连续分配的框架下，共享很难做——你得让两个序列的 KV 挤在同一段连续内存里，还要处理"其中一个要往后写、另一个不写"的情况。

分页之后这件事变得自然：**让两张 block table 指向同一批物理块就行**。代价是要回答两个问题：谁还在用这块？（引用计数 `ref_cnt`）如果有人要往里写怎么办？（**Copy-on-Write**——先复制一份新的，再写，别动共享的那份）。

效果论文也给了数：parallel sampling 和 beam search 的显存开销**最多降 55%**，换算成吞吐**最多 2.2×**。

## 落到代码上

前面说的东西，在 vLLM 0.30.0 里对应这两个文件：

- `vllm/v1/core/kv_cache_manager.py`——`KVCacheManager`，对外的门面。调度器调的就是它的 `allocate_slots()`（`kv_cache_manager.py:370`）。
- `vllm/v1/core/block_pool.py`——`BlockPool`，真正的块池。

`BlockPool` 的文档把核心数据结构说清楚了：

> The `free_block_queue` stores the free blocks in **eviction order** to enable allocation, free, and cache eviction. The `cached_block_hash_to_block` maps between **block hash** and cached block to support finding cached blocks by their block hash.

两句话两件事：

- **`free_block_queue`** 不只是个空闲链表，它是**按淘汰顺序**排的——所以前缀缓存要腾地方时，直接从头部拿就是最该淘汰的那个（LRU）。
- **`cached_block_hash_to_block`** 是前缀缓存的查找表：给块内容算个哈希，就能知道"这段前缀我是不是已经算过、块还在不在"。这就是 prefix caching 的落地方式，第 9 章细讲。

现在回到第 3 章那个问题了。调度器每一步都在调 `allocate_slots()`，而这个函数的返回值是：

```python
) -> KVCacheBlocks | None:
    ...
    if new_blocks is not None:
        # The request can be scheduled.
        break
    ...
    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
```

**返回 `None` 就是"没有块了"**，调度器收到这个信号才去挑人抢占。第 3 章讲的"装不下怎么办"，触发条件就在这一行——分页管理是手段，抢占是它不够用时的兜底。

（还有一层：0.30.0 里 `KVCacheManager` 把实际的块管理交给了一个 `coordinator`（`kv_cache_manager.py:170`），用来支持 MHA / MLA / Mamba 这类结构不同的层混在同一个模型里。这是比论文更新的演进，第 5 章说到骨架时再展开。）

## 效果

论文给的数字是 **2–4× 吞吐提升**，前提值得说清楚——是"**在相同延迟下**，跟当时的 SOTA 系统（FasterTransformer 和 Orca）比"：

> vLLM improves the throughput of popular LLMs by 2-4× with the same level of latency compared to the state-of-the-art systems, such as FasterTransformer and Orca.

你可能还见过 vLLM "**24×**" 那个数字，那是官方博客里的另一个对比：

> it delivers up to **24x** higher throughput than **HuggingFace Transformers**

**这两个数字的基线完全不同**，不能混着引用。24× 的对手是 HF Transformers——一个基本没做批处理和服务优化的基线；2–4× 的对手是已经做过连续批处理的专业推理系统。拿 24× 说事容易被人挑，2–4× 才是和同行比的结果。

顺带一提，论文还说了提升**在长序列、大模型、复杂解码算法上更明显**——这正好对应上面两条：序列越长、最后一块的浪费占比越低（分页的收益全在"不用预留"上）；解码算法越复杂（parallel sampling、beam search），共享能省的就越多。

## 该收一下第一篇了

到这里，第 1 章那条因果链的每一环都落地了：

```text
高吞吐要靠大 batch
  └─ batch 被 KV cache 的显存卡住
       └─ 显存被"过度预留 + 碎片 + 重复"浪费掉 60-80%
            └─ 根因：连续分配的假设
                 └─ PagedAttention：定长块 + block table + 按需分配 + 共享
                      └─ 浪费 < 4%，吞吐 2-4×
```

同时，四章下来我们已经把三个角色都见过了：**引擎循环**（第 1 章）、**调度器**（第 3 章）、**KV cache 管理器**（这一章）。

但都还停在"它该干什么"的层面。**真正去看它们在源码里长什么样、怎么串起来的**，是第二篇的事。下一章从引擎的骨架开始：从 `LLM.generate()` 那一行进去，一路到 `EngineCore.step()`。

## 参考资料

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，SOSP 2023
- vLLM Team, [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)，2023-06-20
- vLLM 0.30.0 源码：`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`、`vllm/config/cache.py`
