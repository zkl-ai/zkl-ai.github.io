---
title: "vLLM 学习笔记（5）：引擎骨架与源码地图"
description: "从 LLM.generate() 往里走，看引擎分成哪几层、三个角色各自住在哪、一条请求有多少种状态。"
pubDate: 2026-10-03
tags: ["vLLM", "推理加速", "源码笔记"]
category: inference-acceleration
---

第一篇那四章都在讲"该干什么"：谁负责调度、谁负责显存、谁负责循环。从这一章开始换一个问题——**它们在源码里长什么样，又是怎么串起来的**。

## 从一行代码进去

还是第 1 章那段代码，这次看的是它的内部：

```python
from vllm import LLM, SamplingParams

prompts = ["你好，请用一句话介绍你自己", "写一个把列表倒序的 Python 函数"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="Qwen/Qwen3.5-0.8B")
outputs = llm.generate(prompts, sampling_params)
```

用 0.8B 的小模型纯粹是为了下载快、单卡能跑，方便反复改参数看现象。要观察的机制跟模型大小没关系。

## 这两行代码管的不是一回事

我一开始把 `LLM(...)` 和 `generate(...)` 当成一件事，读源码才发现它们管的东西完全不同。

`LLM(...)` 构造的时候在**准备**：读权重、跑一次假前向量一下显存、据此决定 KV cache 能开多少 block、把 decode 阶段的算子录成 CUDA Graph。这些决定了这台机器**最多能同时服务多少个请求**。

`generate(...)` 的时候在**跑**：请求进队列，引擎一步步循环——每步挑一批请求、跑一次前向、把新 token 拼回去。这些决定了**每条请求要多久**。

所以第 6 章讲构造、第 7 章讲生成，是两条独立的线。这一章先把两张地图给出来。

## 引擎的分层

顺着源码往下读，层次是这样的。缩进就是嵌套关系，右边是「文件:行号」：

```text
LLM                                     entrypoints/llm.py:67
└── LLMEngine                           v1/engine/llm_engine.py:48
    ├── renderer                        :91
    ├── input_processor                 :94
    ├── output_processor                :97
    └── engine_core ──────────────►     :109   EngineCoreClient
        └── EngineCore                  v1/engine/core.py:105
            ├── model_executor          :134
            ├── structured_output_manager  :146
            └── scheduler               :162   → v1/core/sched/scheduler.py:79
                └── kv_cache_manager           scheduler.py:299
                    └── KVCacheManager         v1/core/kv_cache_manager.py:134
```

有两处我读的时候停顿了一下。

**一处是 `LLMEngine` 和 `EngineCore` 中间夹着的那个 `EngineCoreClient`**（`llm_engine.py:109`）。它看着像多余的一层，其实是部署方式的适配层：单进程跑的时候它就是直连，多进程跑的时候它负责跨进程传消息，而 `LLMEngine` 不用改。第 8 章讲在线服务时会回到这里。

**另一处是 `kv_cache_manager` 的位置。** 我第一次画图时把它和 `model_executor`、`scheduler` 画成了三个并列的框，后来翻到 `scheduler.py:299` 才发现它是在 `Scheduler.__init__` 里创建的，**是 Scheduler 的成员**。当时觉得别扭，现在知道为什么了——第 4 章讲过，调度器判断"能不能接纳这条请求"的方式就是调 `allocate_slots()`，而它返回 `None` 就意味着没块了、该挑人抢占了。**显存状态是调度决策的直接输入**，所以它挂在调度器身上。

（等 GPU 机器到位，第一件想做的事是把 `llm.llm_engine`、`llm.llm_engine.engine_core` 这两个对象打印出来，看层次跟上面这张图对不对得上。第 1 章说过，现在这台机器装不了 vLLM。）

## 三个角色住在哪

第 1 章说引擎由三个角色配合：**引擎循环**、**调度器**、**KV cache 管理器**。现在可以给它们填上门牌号。

分界点就在 `EngineCore.step()`（`engine_core.py:589`）。它的文档字符串只有一句：

> Schedule, execute, and make output.

```python
def step(self):
    ...
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule(...)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    ...
    model_output = future.result()
```

对应关系是：

| 角色（第 1 章） | 源码位置 | 哪一章讲透的 |
|---|---|---|
| 引擎循环 | `EngineCore.step()`（`engine_core.py:589`），外层是 `run_busy_loop()`（`:1403`） | 第 7 章 |
| 调度器 | `Scheduler.schedule()`（`scheduler.py:561`） | 第 3 章 |
| KV cache 管理器 | `KVCacheManager`（`kv_cache_manager.py:134`），挂在调度器里 | 第 4 章 |

顺便注意 `execute_model(..., non_block=True)`：这一步是**异步**发出去的。引擎不等 GPU 算完就往下走，用这段时间去做 CPU 侧的事（比如准备 grammar bitmask），最后才 `future.result()` 等结果。这就是 V1 博客里说的 "Optimized Execution Loop"——尽量别让 CPU 空等 GPU。

第 3 章那个"`num_computed_tokens` 在调度时就加"的乐观计数，是同一件事的更进一步：异步调度下**调度器会跑到 GPU 前面好几步**，所以计数器只能先记上，不能等结果回来再记。

## 请求的状态机

一条请求从进来到结束会走一个状态机。0.30.0 的定义在 `v1/request.py:364`，一共 12 个状态。

主干是 `WAITING → RUNNING → FINISHED_*`，中间可能被 `PREEMPTED` 打断，之后再回到 `RUNNING`。围绕主干长出来的是几个高级特性带来的状态：`WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR`（等引导解码的文法编译完）、`WAITING_FOR_REMOTE_KVS`（等远端 KV，P/D 分离场景）、`WAITING_FOR_STREAMING_REQ`。结束状态有 6 种，区分不同终止原因：正常停止、超过长度上限、被主动中止、出错、检测到重复、以及被忽略。

这几个"被阻塞的等待状态"不是摆设。调度器里有个 `_try_promote_blocked_waiting_request()`（`scheduler.py:2963`）专门负责把它们提升回可调度的状态——远端 KV 到了就从 `WAITING_FOR_REMOTE_KVS` 恢复成 `WAITING`（如果之前被抢占过则是 `PREEMPTED`），文法编译好了就从 `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` 变回 `WAITING`。所以状态机不只是记录现状，它也是调度器判断"这条请求现在能不能跑"的依据。

这里有个细节我挺喜欢：源码把所有 `FINISHED_*` 的枚举值都排在 `PREEMPTED` 之后，于是判断请求是否结束就一行（`request.py:386`）：

```python
@staticmethod
def is_finished(status: "RequestStatus") -> bool:
    return status > RequestStatus.PREEMPTED
```

用枚举的先后顺序来表达"是否结束"，省掉了一个专门的状态字段。当然代价是**这个顺序不能随便动**——往中间插一个新状态就得先想清楚它算不算结束。

## 和 Anatomy 博客的差异

对照 0.30.0 和那篇博客（基于 2025 年 8 月的 commit `42172ad`），有三处已经不一样了：

- `renderer` 是新增的组件（`llm_engine.py:91`），博客里没有——输入渲染被单独拆了出来；
- 请求状态机从 3 个粗粒度状态细化到了 12 个；
- KV 管理多了一层 `coordinator`（`kv_cache_manager.py:170`），用来支持 MHA / MLA / Mamba 这类结构混在同一个模型里。

博客没写错，只是框架在往"支持更复杂的模型结构"这个方向演进。顺手记这些差异，比记住某个静态结构有用——毕竟下次升级，变的还是这些地方。

## 下一章看什么

骨架有了，[下一章](/blog/vllm-notes-06-engine-init/)钻进 `LLM(...)` 的构造函数。里面是一条链：`init_device` → `load_model` → `determine_available_memory` → `initialize_from_config`。

其中 `determine_available_memory` 是关键的一环——**KV cache 能开多少个 block，就是在这一步算出来的**。而它直接决定了这台机器最多能同时服务多少请求，也就决定了第 2 章那个 batch size 的上限。

## 参考资料

- vLLM 0.30.0 源码：`entrypoints/llm.py`、`v1/engine/llm_engine.py`、`v1/engine/core.py`、`v1/core/sched/scheduler.py`、`v1/request.py`
- Aleksa Gordić, [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)，2025-09-05
