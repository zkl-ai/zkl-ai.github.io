# vLLM vs SGLang 推理框架对比研究

> 固定版本：**vLLM 0.30.0** vs **SGLang 0.5.20**。所有结论都基于工作区内 `sources/` 里的源码，标注了文件与行号，可复核。

## 一、研究目标

围绕「大模型推理引擎的核心机制」做系统性对比，重点回答：

- 调度器（Scheduler）的职责边界是什么？
- KV cache 管理挂在哪个层级、怎么组织？
- 前缀缓存、抢占、分页等机制的实现差异？

## 二、源码清单

### vLLM 0.30.0

| 文件 | 行数 | 说明 |
|------|------|------|
| `sources/vllm-0.30.0/scheduler.py` | 3250 | V1 调度器 |
| `sources/vllm-0.30.0/kv_cache_manager.py` | 941 | KV cache 管理器 |

### SGLang 0.5.20

| 文件 | 行数 | 说明 |
|------|------|------|
| `sources/sglang-0.5.20/scheduler.py` | 5894 | 调度器（monolithic） |
| `sources/sglang-0.5.20/radix_cache.py` | 908 | RadixCache 前缀缓存 |
| `sources/sglang-0.5.20/memory_pool.py` | 5702 | 各类 KV 内存池 |

> 两个项目均为 Apache-2.0 许可，源文件保留原始版权头，仅供本地学习研究。

## 三、核心结论一：KV cache 管理归属于调度器

两个框架在这一点上是**一致的**——KV cache 管理都是调度器的成员，因为「能不能接纳一个请求」取决于显存够不够，这是调度决策的直接输入。

### vLLM 0.30.0 的证据

文件：`vllm/v1/core/sched/scheduler.py`

| 行号 | 内容 |
|------|------|
| L79 | `class Scheduler(SchedulerInterface):` |
| L299 | `self.kv_cache_manager = KVCacheManager(...)`（在 `Scheduler.__init__` 内创建） |
| L729 | `new_blocks = self.kv_cache_manager.allocate_slots(...)`（在调度循环内调用） |
| — | `self.kv_cache_manager` 全文出现 **44 次** |

### SGLang 0.5.20 的证据

文件：`python/sglang/srt/managers/scheduler.py`

| 行号 | 内容 |
|------|------|
| L428 | `class Scheduler(...)` |
| L613 | `self.req_to_token_pool = result.req_to_token_pool` |
| L614 | `self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator` |
| L616 | `self.tree_cache = result.tree_cache`（RadixCache） |
| — | 这三者合计被引用 **108 次** |

**结论**：无论是 vLLM 还是 SGLang，调度器都直接持有 KV 管理组件。区别不在「归谁管」，而在**封装粒度**（见下）。

## 四、核心结论二：封装粒度不同

### vLLM：单一 Manager 封装

```
Scheduler
└── kv_cache_manager: KVCacheManager
    ├── coordinator（协调多种 KV cache 类型）
    └── block_pool（分页 block 池）
```

- `KVCacheManager`（`kv_cache_manager.py` L134）
  - L170：`self.coordinator = get_kv_cache_coordinator(...)`
  - L200：`self.block_pool = self.coordinator.block_pool`
- 对外方法：`get_computed_blocks`（L265）、`allocate_slots`（L370）、`free`（L603）、`cache_blocks`（L788）、`evict_blocks`（L645）、`reset_prefix_cache`（L653）等

### SGLang：三个组件协作

```
Scheduler
├── req_to_token_pool: ReqToTokenPool        # 请求 → token 槽位索引
├── token_to_kv_pool_allocator               # KV 显存分配器
└── tree_cache: RadixCache                   # 前缀共享（Radix 树）
```

- `ReqToTokenPool`（`memory_pool.py` L259）：内部是 `req_to_token`（`[size, max_context_len]` 张量）+ `free_slots`，即「请求 → token 位置」的索引表
- `RadixCache`（`radix_cache.py` L321）：`match_prefix`（L397）、`insert`（L457）、`cache_finished_req`（L479）、`cache_unfinished_req`（L561）、`evict`（L638）
- KV 池有大量变体（`memory_pool.py`）：`MHATokenToKVPool`、`MLATokenToKVPool`、`MambaPool`、`DSATokenToKVPool`、`MiniMaxSparseKVPool` 等

## 五、对比总表

| 维度 | vLLM 0.30.0 | SGLang 0.5.20 |
|------|-------------|---------------|
| KV 管理归属 | Scheduler 成员 | Scheduler 成员（三个） |
| 封装粒度 | **单一** `KVCacheManager` | **三个**协作组件 |
| 索引结构 | block table（分页，block 为最小单位） | `req_to_token`（token 级索引） |
| 显存分配 | `block_pool`（统一 block 池） | `token_to_kv_pool_allocator` + 多种池 |
| 前缀缓存 | `KVCacheManager` 内（block hash + `find_longest_cache_hit`） | `RadixCache`（Radix 树 + `match_prefix`） |
| 调度器文件规模 | 3250 行 | 5894 行（更 monolithic） |
| 并行/分布式扩展 | `coordinator` 协调多类型 cache | 池种类更细（MHA/MLA/Mamba/DSA…） |

## 六、设计哲学差异（初步判断）

- **vLLM 走「统一抽象」**：把分页块、前缀缓存、淘汰都收进一个 `KVCacheManager`，对外接口收敛（`allocate_slots` / `free` / `cache_blocks`）。调度器只跟一个对象打交道，简单清晰。
- **SGLang 走「职责拆分」**：索引（`req_to_token_pool`）、分配（allocator）、复用（`tree_cache`）各自独立，配合更灵活，池类型可以按模型结构（MHA/MLA/Mamba）细分。

这背后是两种 KV 抽象：**vLLM = block 分页**，**SGLang = token 级索引 + Radix 树**。

## 七、待深入对比的维度

- [ ] 调度策略：FCFS / priority 的具体实现差异
- [ ] 抢占（preemption）机制：触发条件、swap vs recompute
- [ ] chunked prefill 的实现
- [ ] 投机解码（speculative decoding）的集成方式
- [ ] 多模态视觉 token 的 KV 管理（与 Qwen3-VL 相关工作直接相关）
- [ ] 分布式（TP/PP/DP/EP）下的 KV 管理

## 八、参考资料

1. vLLM 源码：<https://github.com/vllm-project/vllm>（tag `v0.30.0`）
2. SGLang 源码：<https://github.com/sgl-project/sglang>（tag `v0.5.20`）
3. Aleksa Gordić, *Inside vLLM: Anatomy of a High-Throughput LLM Inference System*, vLLM Blog, 2025-09-05. <https://vllm.ai/blog/2025-09-05-anatomy-of-vllm>
4. Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023.
5. Zheng et al., *SGLang: Efficient Execution of Structured Language Model Programs*（RadixAttention）
