---
title: "vLLM 学习笔记（3）：调度器"
description: "每一步的 token 预算怎么分、装不下时抢谁、抢完的请求怎么回到队列。"
pubDate: 2026-10-01
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
draft: false
---

第 2 章说连续批处理把"换人"的时机从批边界挪到了步边界。但换人是个决定，是决定就得有依据。这一章看调度器每一步到底在决定什么。

## 一个请求想要的，和调度器能给的

整个调度器的核心是一个减法。挖到源码里就一行（`scheduler.py:656`）：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

要读懂它，得先把两个很容易混的量分开：

- **`num_tokens`——"已知"**：prompt 加上已经返回的输出。注意 prompt 只要你提交了就算已知（`request.py:148` 把它整个拷进 `_all_token_ids`），所以它从第一刻起就是完整的 prompt 长度，哪怕一个 token 都还没算。
- **`num_computed_tokens`——"已排进前向"**：已经排进过前向、KV 有着落的 token 数。注意是"已排进"而不是"已算完"——`scheduler.py:1533` 是在**调度的时候**加它的，因为异步调度和 PP 下 GPU 跑在前面，等结果回来再加就晚了。`request.py:168` 那段注释管这叫 "counts them optimistically"。

**一个请求想要多少 token？已知的减去已排进前向的——它欠多少就要多少。**

拿第 1 章那组例子走一遍（prompt 是 A B C，生成 D E F G）：

| 时刻 | `num_tokens` | `num_computed_tokens` | 差 |
|---|---|---|---|
| 刚提交 | 3 | 0 | 3 → 整个 prompt 一次 prefill |
| 算出 prefill、采样出 D | 4 | 3 | 1 |
| 算出 D、采样出 E | 5 | 4 | 1 |
| 算出 E、采样出 F | 6 | 5 | 1 |

**decode 阶段这个差恒等于 1**：每算掉一个 token 的 KV，同时又生成一个新 token，账一直欠着。所以 decode 的本质是"**一直欠 1 个 token 的 KV**"——这比"每步算一个 token"准确，因为它顺带说明了为什么不存在"欠 0 个"的稳定状态。

长 prompt 那边，两个量能差得很远：

| 步骤 | `num_tokens` | `num_computed_tokens` | 差 |
|---|---|---|---|
| 初始 | 20000 | 0 | 20000 |
| 第 1 块后 | 20000 | 7992 | 12008 |
| 第 2 块后 | 20000 | 15984 | 4016 |
| 第 3 块后 | 20001 | 20000 | 1 |

中间几块 `num_tokens` 一直不动——**因为 prompt 还没算完，采样不出任何输出 token**。

有了这两个量，"它想要多少"就清楚了：把欠的一次补齐。然后这个数会被削一刀——源码紧接着就是 `min(num_new_tokens, token_budget, ...)`：**预算剩多少，就只能给多少**。

**这就是 chunked prefill 的全部秘密。** 框架里没有为长 prompt 写什么特殊逻辑，只是"它想要 20000，我只给得起 7992"。

## 公式里多出来的两项

既然是"已知减已算"，为什么公式里还多两项？

因为**两个边都得先修正到同一个口径**，否则会算出负数。看这个瞬间：

| 量 | 值 | 原因 |
|---|---|---|
| `num_tokens` | 5 | prompt 3 + 已返回的输出 2 |
| `num_output_placeholders` | 1 | 异步调度已经为下一步占了 1 个输出位置，id 还没回来 |
| `num_computed_tokens` | 6 | 而那个占位位置**已经排进前向**了 |

直接减是 $5 - 6 = -1$，荒谬。加上修正项：$5 + 1 - 6 = 0$——合理，已经有一个在飞了，这一步不用再排。

- **`+ len(spec_token_ids)`**：投机解码的草稿不在 `_all_token_ids` 里（它们还没被接受），但投机解码本来就要把 k+1 个位置一次前向全算掉再统一验证，所以它们是**真要算的工作**。
- **`+ num_output_placeholders`**：就是上面那个例子。它是**异步调度**专用的（`request.py:159` 的注释只有一句 `# Used in async scheduling.`），调度时加、输出回来时减（`async_scheduler.py:39` / `:62`）；基类 `scheduler.py` 里只有减和清零，**增量只存在于 `async_scheduler.py`**。

有一处细节挺漂亮：这个差分本身很稳。调度时两边一起加，输出回来时 `num_tokens` 加、`placeholders` 减，草稿被拒时**两边一起减**（`scheduler.py:1993`）。同增同减，所以公式不需要给投机解码写任何特例——第 2 章那句"这个表示足够通用"，代码版就在这里。

**而在最常见的场景下（不开投机解码、不开异步调度），两个修正项都是 0**，公式就退化成纯粹的减法 `num_tokens − num_computed_tokens`。

## 那 max_tokens 呢

读完公式很容易追问：设了 `max_tokens` 是不是要跟着调整？**不用。** 公式算的是"已知但还没算 KV 的"，`max_tokens` 限制的是"还要新生成多少"——一个是欠账，一个是未来额度。而且这个公式天然只会算出 1（投机解码下是 1+k），永远不会因为"`max_tokens` 还剩 100 个"就一次要 100 个：那 100 个 token 根本还不存在。

更能说明问题的是这个对比——**`max_model_len` 反而真的进了公式**（`scheduler.py:667`）：

```python
# Make sure the input position does not exceed the max model len.
num_new_tokens = min(
    num_new_tokens,
    self.max_model_len
    - request.num_computed_tokens
    - self.num_sampled_tokens_per_step,
)
```

区别在于：`max_model_len` 约束的是**位置**——"这次前向最多算到第几个位置"，这是能对单步施加的现实上限；`max_tokens` 约束的是**输出个数**，它不影响任何单步能算多少 token。**能预算的才进公式，只作终止条件的用别的方式处理。**

`max_tokens` 的生效方式是"每生成一个 token 就判一次"。判定在 `vllm/v1/core/sched/utils.py` 的 `check_stop()` 里：

```python
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

调用点在 `scheduler.py:2347`，而且是**每 append 一个 token 就调一次**：

```python
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]   # Trim new tokens if needed.
        break
```

所以是"生成 → 立即判定 → 越界就停"，不会白算下一步。唯一会真浪费的是投机解码：一步回来 1+k 个 token，如果第 3 个就把 `max_tokens` 打满，后面验证通过的草稿会被那个 `del` 丢掉——那部分算力是真丢了。细节留到第 9 章。

## 预算：不是"能跑几个请求"，而是"能算多少 token"

V1 的调度决策就是一本字典：`{request_id: num_tokens}`，说明这一步每条请求各算多少 token。它受两个独立的约束：

- `max_num_batched_tokens`——一步最多算多少 token
- `max_num_seqs`——同时最多多少条请求在跑（源码里叫 `max_num_running_reqs`，`scheduler.py:124`）

为什么 token 预算比"请求数预算"合适？因为 decode 一步只占 1 个 token，而 chunked prefill 的一块可能占几千个——用请求数当预算，这两种情况差三个数量级，根本控不住。

具体默认值跟硬件和使用场景都有关（`arg_utils.py:2726` 附近）。同一个 H100 上：

| 使用场景 | `max_num_batched_tokens` | `max_num_seqs` |
|---|---|---|
| 离线（`LLM` 类） | 16384 | 1024 |
| 在线（API server） | 8192 | 1024 |

在线给得比离线小一半。代码里没解释为什么，但方向很清楚：预算越大，单个 step 越重，延迟越容易抖。这和第 2 章结尾那个结论是一回事——吞吐和延迟得分开算。

还有个反例值得记一笔：A100 上把 `max_num_batched_tokens` 调大**反而掉吞吐**，所以源码里专门做了设备名判断把它排除在外，注释指向 PR #17885。"预算越大越好"是不成立的。

## 每一步的顺序：先续上已经跑着的

`schedule()` 里两句注释把顺序说尽了（`scheduler.py:610` 和 `scheduler.py:854`）：

```text
# First, schedule the RUNNING requests.
# Next, schedule the WAITING requests.
```

这里有个容易搞混的地方：**running 不等于 decode。** 一条长 prompt 一旦被准入就进了 running，之后每一步续一块 chunk，直到 prompt 算完。所以准确的说法是"**已准入的优先**"，"decode 优先"只是它的一个副产品。

为什么这么排？已准入的请求 KV 已经占着显存、用户已经在等它的下一个 token；让排队的请求多等一步，代价小得多。这个选择偏向**延迟稳定**，而不是公平。

不过 FCFS 也不是严格的：running 队列里如果有哪条请求这一步排不进去（比如编码器预算不够、或者预算凑不出一个对齐的块），调度器是 `continue` 而不是 `break`。源码注释把话说明白了——这样"就不严格遵守 FCFS 了，会让优先级更低的请求也拿到机会"（`scheduler.py:713`）。免得一条卡住的请求堵死后面所有人。

## 走一遍：200 条 decode + 一条 20000 token 的 prompt

用在线那组默认值，预算 8192。

1. 200 条已经在跑的请求，都是 decode，各欠 1 个 token → 拿走 200，预算剩 7992
2. 排队里那条 20000 token 的长 prompt 想要 20000，只拿到 7992 → 这一步它只算前 7992 个 token
3. 下一步继续给它一块，直到算完：7992 + 7992 + 4016 = 20000，一共三步

事办成了，代价落在这一步：总共要算 8192 个 token，而纯 decode 只要 200 个——那 200 条请求在这一步的 ITL 会明显变差（这一步从访存受限变成了计算受限）。

想压住这个抖动，就得限制一块能有多大。源码里确实有旋钮：`long_prefill_token_threshold`，prompt 超过这个长度就会被主动截断成更小的块。它的默认值是 **0，也就是不限制**。

这个阈值就是"ITL 抖动"和"prefill 效率"之间的调节点。顺便记一下：第 9 章会看到，官方自己也承认这个值不好调——那正是 P/D 分离存在的理由之一。

## 装不下怎么办：抢占

预算解决的是"算多少"，但还有另一条约束：**KV cache 是有限的**（第 1 章的障碍一）。当 `allocate_slots` 要不到块的时候，只能从已经在跑的人里挑一个出去。

**抢谁**

默认策略是 `fcfs`（配置项是 `policy`，另一个选项 `priority` 按显式优先级排，见 `vllm/config/scheduler.py:141`）。FCFS 下选的是 `self.running[-1]`——**running 队列里最后加进来的那条**，也就是最近才被准入的（`scheduler.py:753`）。

这个选择是有道理的。FCFS 下它优先级最低，而且通常也是**算得最少**的那条。recompute 会把 `num_computed_tokens` 清零，之前算过的全部作废——抢一条已经跑了两三千 token 的请求，那部分 prefill 就白算了；抢刚进来的那条，损失最小。

**怎么处置**

V1 只有一种方式：**recompute**。把请求的块全部还给池子、`num_computed_tokens` 清零、状态标成 `PREEMPTED`，然后**插到等待队列的最前面**（`scheduler.py:1517` 的 `prepend_request`）。

插到队首这一步是防饿死的关键：它已经被耽误过一次了，要是还排在队尾，就可能被后面源源不断的新请求一直挡在门外。

V0 时代还有第二种方式：**swap**——把 KV 换到 CPU 内存，回来时再搬回去，省掉重算。V1 的调度器里已经完全没有 swap 了（在 `scheduler.py` 里搜一遍 `swap`，一个都没有）。统一走 recompute 换来了实现简单，代价是长 prompt 被抢占一次就要重算一次。

**抢完那一步有条特殊规则**

只要这一步发生过抢占，waiting 那一段就整个跳过（`scheduler.py:855` 的 `if not preempted_reqs and ...`）。既然显存已经紧张到要抢人了，就别再放新请求进来火上浇油。

recompute 的代价随 prompt 长度线性涨——这正是后面要做 prefix caching 的动机之一（第 9 章）：重算的时候如果前缀命中缓存，就不用真的重算了。

## 调度器交出去的东西

调度器每步的产出是一份 `SchedulerOutput`，本质是**一份"这一步谁跑、各跑多少"的计划书，而且是增量的**：

- `num_scheduled_tokens`——核心的那本字典 `{request_id: num_tokens}`
- `scheduled_new_reqs` / `scheduled_cached_reqs`——这一步新准入的、和已经在跑的
- `finished_req_ids`——上一步到现在跑完的
- `preempted_req_ids`——被抢走的
- `scheduled_spec_decode_tokens`、`num_common_prefix_blocks`——投机解码和前缀共享需要的信息

注意后面三项：跑完的、被抢占的、新来的，都是**变化量**。V1 的设计是每一步只把增量发给 worker，而不是重发整个状态。这是它能干脆把调度器和 worker 拆到两个进程里的前提（第 5 章会回到这里）。

## 还没解决的问题

调度器现在会做决定了，但它依赖一个前提：**得知道显存还剩多少、还能不能给这条请求分配块。** 这个"能不能"是谁回答的？块又是怎么组织的、为什么会不够？

这两个问题指向同一个组件：KV cache 管理器。[下一章](/blog/vllm-notes-04-pagedattention/)看 PagedAttention 到底在管什么。

## 参考资料

- vLLM 0.30.0 源码：`vllm/v1/core/sched/scheduler.py`、`vllm/config/scheduler.py`、`vllm/engine/arg_utils.py`
- vLLM Team, [vLLM V1: A Major Upgrade to vLLM's Core Architecture](https://vllm.ai/blog/2025-01-27-v1-alpha-release)，2025-01-27
