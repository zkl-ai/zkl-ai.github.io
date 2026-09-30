# 各章素材（已核实，待写）

给后面章节备的料。每条都标了出处，写的时候直接引，不用重新翻源码和文档。
不确定的地方一律进最后一节，不混进正文。

---

## 第 9 章 · 高级特性

### P/D 分离

**定位（最重要的一条）**
官方文档明确写着 **"Disaggregated prefill DOES NOT improve throughput."**
→ 它是延迟可调性手段，不是吞吐手段。出处：vLLM docs `features/disagg_prefill`。

官方给的两条理由：
1. 分开调 TTFT 和 ITL——可以给 prefill 实例和 decode 实例配不同并行策略（tp / pp），调一个不影响另一个。
2. 控制 tail ITL——不分离时 decode 期间会被插进 prefill 任务，尾部延迟变差。chunked prefill 也能达到同样目的，但官方原话是 "in practice it's hard to figure out the correct chunk size"，所以分离更可靠。

同出处还有一句：**"Disaggregated prefill DOES NOT improve throughput."**（放在 Note 里，专门强调）

**和 V1「调度器里没有 P/D 阶段」的关系**（这是最容易搞混的点）

| 结论 | 出处 |
|------|------|
| P/D 从「显式阶段」降级成「从计数器推出来的性质」：`request.is_prefill_chunk = request.num_computed_tokens < (request.num_tokens + request.num_output_placeholders)` | `scheduler.py:1538` |
| 实例角色是**启动配置**不是调度决策：`kv_role` = `kv_producer` / `kv_consumer` / `kv_both` | vLLM docs `features/disagg_prefill` 用法示例 |
| 远程 KV 落进同一表示：请求进 `RequestStatus.WAITING_FOR_REMOTE_KVS`，且 `num_computed_tokens` 被**预先**设成 prompt 长度。源码注释：“Set num_computed_tokens even though KVs are not yet loaded. ... will not be used anywhere until the request finished the KV transfer.” | `request.py:369`、`scheduler.py:1212` 附近 |
| 远程 KV 和本地 prefix cache **走同一个口子**：`# Get externally-cached tokens if using a KVConnector.` | `scheduler.py:927` |
| connector 分成调度端和 worker 端两个角色 | `KVConnectorRole.SCHEDULER` / `WORKER`，`scheduler.py:155` |
| 消费者侧在重叠批（异步调度 / PP）下要延迟释放块 | `defer_block_free`，`scheduler.py:170` |
| 生产者侧在 `is_kv_producer` 时走 `finalize_partial_tail_offloads` 把 block table 交给 connector | `scheduler.py:2856` |

**写法建议**：先讲「不区分阶段」是表示层的事，再讲分离是部署层的事，中间用「`num_computed_tokens` 被预先设成 prompt 长度」这个细节把两件事接起来——这是最漂亮的一环。

### FA3（attention kernel）

**为什么 V1 需要它**（V1 博客原话）
> Given the high level of dynamism in V1—such as combining prefill and decode within the same batch—a flexible and high-performance attention kernel was essential.

**FA3 vs FA2 的实质差别**（出处：Tri Dao《FlashAttention-3》博客，2024）

- FA2 在 A100 上能到 70% 峰值，但在 H100 上**只有 35%**——没用上 Hopper 新特性。
- 三个没用上的 Hopper 特性：
  - **WGMMA**：不用它，老 `mma.sync` 只能到 Hopper Tensor Core 峰值的 **2/3**
  - **TMA**：专管 global ↔ shared memory 搬运，顺带做索引计算和越界判断，省寄存器
  - **FP8**：吞吐翻倍（H100 SXM5：FP16 989 TFLOPS → FP8 1978 TFLOPS）
- 只是用上这三样：FP16 forward 从 FA2 的 ~350 TFLOPS 到 540–570。
- 真正拉开差距的是利用异步性做**重叠**：
  - **warp specialization**：producer warp 只发 TMA，consumer warp 只做 WGMMA
  - **pingpong scheduling**：两个 warpgroup 交替，1 做 GEMM 时 2 做 softmax
- **关键数字（解释为什么要重叠）**：H100 SXM5 的 FP16 矩阵乘 989 TFLOPS，但特殊函数（exp 这类）**只有 3.9 TFLOPS——慢 256 倍**。head_dim=128 时 matmul FLOPs 比 exponential 多 512 倍，所以 exp 能吃掉**一半时间**。FP8 下 matmul 又翻倍而 exp 不变，更糟。
- 结果：FP16 到 **740 TFLOPS / 75% 利用率**，比 FA2 快 1.5–2×；FP8 接近 **1.2 PFLOPS**，误差比 baseline FP8 attention 小 2.6×。

**注意别写过头**：FA3 不是 V1 的唯一后端。`initialize_attn_backend`（`worker_gpu_model_runner.py:7036`）会按模型/硬件挑后端。博客说它是"最后一块拼图"指的是性能，不是依赖。

---

## 第 10 章 · 多卡多机

### TP / PP / DP 的取舍

**官方锚点**
- 非 NVLink 节点（如 L40S）官方建议**用 PP 而不是 TP**，因为 PP 通信开销更低。出处：vLLM docs `serving/parallelism_scaling`。→ 说明 TP 的通信代价是真实存在的。
- 模型太大时的默认套路：`tensor_parallel_size` = 单节点卡数，`pipeline_parallel_size` = 节点数。同出处。

**机理（本项目推导，基于第 1 章的算术强度公式）**

还是 $m=n=4096$ 那层，$\text{强度} = \frac{Bnm}{nm + Bm + Bn}$：

| 场景 | $B$ | 强度 | 相对 H100 平衡点（300） | 缺什么 | 该动什么 |
|---|---|---|---|---|---|
| prefill，4096 token 的 chunk | 4096 | $\approx 1365$ | 4.5 倍 | 算力 | TP（堆算力）|
| decode，单请求 | 1 | $\approx 1$ | 差 300 倍 | 带宽 | 加大 batch，不是 TP |

- **prefill 上 TP**：切分权重，每卡算 $1/tp$ 的 FLOPs，TTFT 近似按比例下降；算子计算密集，通信能藏在计算后面。
- **decode 上 TP**：权重读取已按 TP 摊到各卡，但每步计算极小，每层 all-reduce 的**固定延迟**占比很高且每步都付 → 收益迅速递减。
- **PP 对 decode 尤其不友好**：每 token 顺序穿过所有 stage，一步只有一个 token，流水线全是气泡。prefill 一次几千 token 能填满流水线，PP 就合理。
- P/D 分离让两个池的**并行策略**和**规模**各自独立。

---

## 吞吐 vs 延迟 · 杠杆对照

**核心区分**：吞吐不是"能跑多快"，而是"**在给定 ITL 下能跑多大 batch**"。所以延迟稳定手段会间接变成吞吐手段。

| 你缺什么 | 杠杆 | 出处 / 章节 |
|---|---|---|
| 单步成本（延迟 + 吞吐都受益） | 更快的 kernel（FA3）、更大 TP | 第 9、10 章 |
| 延迟抖动（tail ITL） | P/D 分离、chunked prefill（chunk 调小） | 第 9 章 |
| 吞吐 | 加大 batch（**前提是显存够**）| 第 2 章 |
| 吞吐 | PagedAttention 省显存 → 换来更大 batch | 第 4 章 |
| 吞吐 | prefix caching（去掉重复 prefill） | 第 9 章 |
| 吞吐 | 投机解码（把空闲带宽换成算力）——正好对应 roofline：decode 强度 ≈1 远低于平衡点，就该这么花算力 | 第 9 章 |
| 吞吐 | DP 副本（纯横向扩，最直接） | 第 10 章 |

**第 1 章那个 $B$ 表不只是解释"为什么要 batch"，它同时是一张优化手段的导航图**：先看强度落在平衡点哪一侧，再决定力气往哪使。写在 prefill 上堆算力只降延迟、不加吞吐；写在 decode 上要省搬运。

---

## 待核实 / 暂不写

- 具体 TP 度数下 decode 收益的拐点，依赖硬件（NVLink vs PCIe）和模型结构，没有实测数据前不给数字。
- `is_kv_producer` 的完整语义（生产者侧何时算"结束"）只看了片段，写第 9 章前要把 `_connector_finished`（`scheduler.py:2842`）整段读完。
- SGLang 侧对应的机制（radix cache / P/D 分离做法）还没查，做对比时再补。
