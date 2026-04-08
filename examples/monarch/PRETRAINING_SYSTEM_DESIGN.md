# Designing a Large-Scale Pre-Training System

## Table of Contents

1. [Goals and Scope](#1-goals-and-scope)
2. [Requirements](#2-requirements)
3. [Model Sizing and Memory Analysis](#3-model-sizing-and-memory-analysis)
4. [Hardware Architecture and Interconnects](#4-hardware-architecture-and-interconnects)
5. [Parallelism Strategies](#5-parallelism-strategies)
6. [Communication Primitives and Libraries](#6-communication-primitives-and-libraries)
7. [The Three-Way Tradeoff: Computation, Communication, Memory](#7-the-three-way-tradeoff-computation-communication-memory)
8. [Kernels and Compute Optimization](#8-kernels-and-compute-optimization)
9. [Mixed Precision and Quantization](#9-mixed-precision-and-quantization)
10. [Data Loading Pipeline](#10-data-loading-pipeline)
11. [Fault Tolerance](#11-fault-tolerance)
12. [Checkpointing](#12-checkpointing)
13. [Monitoring and Observability](#13-monitoring-and-observability)
14. [Putting It All Together](#14-putting-it-all-together)

---

## 1. Goals and Scope

Before writing any code, define what you're building:

### What are you training?
- **Model architecture**: Transformer decoder-only (GPT-style), encoder-decoder (T5-style), mixture-of-experts (MoE)?
- **Model size**: 7B, 70B, 405B, 1T+ parameters?
- **Target dataset size**: 1T, 10T, 15T tokens?
- **Training duration**: Days, weeks, months?

### What does success look like?
- **MFU (Model FLOPS Utilization)**: What percentage of theoretical hardware FLOPS do you achieve? Good systems hit 40-55% MFU on large runs. This is the single most important efficiency metric.
- **Time to train**: Total wall-clock time to reach target tokens/loss.
- **Cost efficiency**: Total GPU-hours (and dollars) per unit of quality.
- **Reliability**: Can you sustain training for weeks without manual intervention?

### Scope decisions
- Are you training from scratch or fine-tuning?
- Single cluster or multi-cluster?
- What hardware do you have access to? (This constrains everything.)
- What's your team size? (Simpler systems are better if you have a small team.)

---

## 2. Requirements

### Compute Requirements

For a transformer model, training FLOPS are approximately:

```
FLOPS ≈ 6 * N * D
```

Where:
- `N` = number of parameters
- `D` = number of training tokens
- The factor `6` comes from: 2 (multiply-add) × 3 (forward + backward is ~3x forward)

**Example — LLaMA 70B on 15T tokens:**
```
FLOPS = 6 × 70×10⁹ × 15×10¹² = 6.3×10²⁴ FLOPS
```

On a cluster of 2048 H100s at 50% MFU:
```
Effective FLOPS/GPU = 989 TFLOPS × 0.50 = ~495 TFLOPS
Cluster FLOPS = 2048 × 495 × 10¹² = 1.01×10¹⁸ FLOPS/sec
Time = 6.3×10²⁴ / 1.01×10¹⁸ = 6.2×10⁶ seconds ≈ 72 days
```

### Memory Requirements

Per-GPU memory must hold:
1. **Model parameters**: `N × bytes_per_param`
2. **Optimizer state**: Adam stores 2 copies (momentum + variance) = `2 × N × bytes`
3. **Gradients**: `N × bytes_per_grad`
4. **Activations**: Depends on batch size, sequence length, hidden dimension, and number of layers. This is often the largest consumer.
5. **Temporary buffers**: Communication buffers, workspace memory

**Example — 70B model in BF16 (no parallelism, no sharding):**
```
Parameters:       70B × 2 bytes  = 140 GB
Optimizer (FP32): 70B × 4 bytes × 2 = 560 GB  (momentum + variance in FP32)
Gradients (BF16): 70B × 2 bytes  = 140 GB
Total (no activations):           = 840 GB
```

A single H100 has 80 GB HBM. You need at minimum 11 GPUs just for model state.
Activations add significantly more. This is why parallelism is not optional.

### Hardware Requirements
- GPU type and count
- Interconnect topology (NVLink, NVSwitch, InfiniBand, RoCE)
- CPU memory per node (for data loading, checkpointing)
- Storage (for checkpoints, datasets): throughput matters more than capacity
- Network bandwidth between nodes

---

## 3. Model Sizing and Memory Analysis

### Transformer Memory Breakdown

For a standard transformer with:
- `L` layers, `H` hidden dimension, `A` attention heads
- Sequence length `S`, micro-batch size `B`

**Parameter count:**
```
Per layer ≈ 12H² (attention: 4H², MLP: 8H² for standard FFN)
Total ≈ 12H² × L
```

Common sizes:
| Model   | L   | H     | A   | Params |
|---------|-----|-------|-----|--------|
| 7B      | 32  | 4096  | 32  | 6.7B   |
| 13B     | 40  | 5120  | 40  | 13.0B  |
| 70B     | 80  | 8192  | 64  | 64.7B  |
| 405B    | 126 | 16384 | 128 | 405B   |

**Activation memory per layer (no activation checkpointing):**
```
≈ B × S × H × (34 + 5×A×S/H) bytes  (in mixed precision)
```

Key insight: activation memory scales with `B × S × H` per layer and `L` layers total.
With 80 layers, this dominates. That's why activation checkpointing exists.

### Activation Checkpointing

Instead of storing all intermediate activations, discard them and recompute
during the backward pass.

| Mode      | Memory Saved     | Compute Overhead |
|-----------|-----------------|------------------|
| None      | 0%              | 0%               |
| Selective | ~50-70%         | ~5-10%           |
| Full      | ~90%+           | ~33%             |

**Selective checkpointing** saves only cheap-to-recompute activations (e.g.,
recompute attention but save linear outputs). This gives most of the memory
savings with minimal compute overhead. This is what most production systems use.

**Full checkpointing** saves only layer inputs and recomputes everything else.
Higher compute overhead but maximum memory savings.

---

## 4. Hardware Architecture and Interconnects

### The Memory Hierarchy

Understanding the hardware hierarchy is critical because it determines which
parallelism strategies are efficient:

```
┌─────────────────────────────────────────────────────┐
│                    CLUSTER                           │
│                                                      │
│  ┌──────────────────┐    ┌──────────────────┐       │
│  │     NODE 0        │    │     NODE 1        │       │
│  │                    │    │                    │       │
│  │  ┌────┐  ┌────┐  │    │  ┌────┐  ┌────┐  │       │
│  │  │GPU0│  │GPU1│  │    │  │GPU0│  │GPU1│  │       │
│  │  │80GB│  │80GB│  │    │  │80GB│  │80GB│  │       │
│  │  └──┬─┘  └─┬──┘  │    │  └──┬─┘  └─┬──┘  │       │
│  │     │NVLink│      │    │     │NVLink│      │       │
│  │  ┌──┴─┐  ┌─┴──┐  │    │  ┌──┴─┐  ┌─┴──┐  │       │
│  │  │GPU2│  │GPU3│  │    │  │GPU2│  │GPU3│  │       │
│  │  │80GB│  │80GB│  │    │  │80GB│  │80GB│  │       │
│  │  └──┬─┘  └─┬──┘  │    │  └──┬─┘  └─┬──┘  │       │
│  │     │NVLink│      │    │     │NVLink│      │       │
│  │  ┌──┴─┐  ┌─┴──┐  │    │  ┌──┴─┐  ┌─┴──┐  │       │
│  │  │GPU4│  │GPU5│  │    │  │GPU4│  │GPU5│  │       │
│  │  │80GB│  │80GB│  │    │  │80GB│  │80GB│  │       │
│  │  └──┬─┘  └─┬──┘  │    │  └──┬─┘  └─┬──┘  │       │
│  │     │NVLink│      │    │     │NVLink│      │       │
│  │  ┌──┴─┐  ┌─┴──┐  │    │  ┌──┴─┐  ┌─┴──┐  │       │
│  │  │GPU6│  │GPU7│  │    │  │GPU6│  │GPU7│  │       │
│  │  │80GB│  │80GB│  │    │  │80GB│  │80GB│  │       │
│  │  └────┘  └────┘  │    │  └────┘  └────┘  │       │
│  │                    │    │                    │       │
│  └────────┬───────────┘    └────────┬───────────┘       │
│           │         InfiniBand       │                   │
│           └──────────────────────────┘                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Bandwidth at Each Level

| Level                  | Technology     | Bandwidth (bidirectional) | Latency   |
|------------------------|---------------|---------------------------|-----------|
| Within GPU             | HBM3          | 3.35 TB/s (H100)          | ~ns       |
| GPU ↔ GPU (same node)  | NVLink/NVSwitch| 900 GB/s (H100)          | ~1-5 μs   |
| GPU ↔ GPU (cross-node) | InfiniBand NDR | 400 Gb/s = 50 GB/s       | ~1-5 μs   |
| GPU ↔ GPU (cross-node) | RoCE           | 100-400 Gb/s             | ~2-10 μs  |
| GPU ↔ CPU              | PCIe Gen5     | 128 GB/s                  | ~μs       |
| CPU ↔ Storage          | NVMe/Network  | 5-25 GB/s                 | ~ms       |

**Key insight**: NVLink is **18x faster** than InfiniBand. This is why:
- Tensor Parallelism (high communication) should stay within a node (NVLink)
- Data Parallelism (lower communication) can go across nodes (InfiniBand)
- Pipeline Parallelism (point-to-point) can go across nodes

### NVLink

NVLink is NVIDIA's high-bandwidth GPU-to-GPU interconnect.

- **NVLink 4.0 (H100)**: 18 links × 25 GB/s = 900 GB/s bidirectional per GPU
- Connects GPUs within a single node
- **NVSwitch**: A chip that provides all-to-all NVLink connectivity between all
  8 GPUs in a node. Without NVSwitch, GPUs would need point-to-point links
  (which don't scale — 8 GPUs would need 28 links). NVSwitch acts like a
  crossbar switch, allowing any GPU to talk to any other at full bandwidth.

**Why it matters**: Tensor parallelism requires all-reduce operations every layer
(hundreds of times per step). At 900 GB/s on NVLink, a 100 MB all-reduce takes
~0.2 ms. Over InfiniBand at 50 GB/s, it would take ~4 ms — a 20x slowdown
multiplied by hundreds of layers.

### InfiniBand (IB)

InfiniBand is a high-bandwidth, low-latency network fabric for connecting nodes.

- **NDR (Next Data Rate)**: 400 Gb/s per port
- Uses **RDMA (Remote Direct Memory Access)**: GPU memory can be read/written
  directly by a remote GPU without involving the CPU. This bypasses the OS
  kernel and TCP/IP stack.
- **GPUDirect RDMA**: Extends RDMA so that data goes directly from GPU memory
  on one node to GPU memory on another node, without staging through CPU memory.

```
Traditional:  GPU0 → CPU0 → NIC0 → Network → NIC1 → CPU1 → GPU1
GPUDirect:    GPU0 → NIC0 → Network → NIC1 → GPU1
```

**Fat-tree topology**: InfiniBand networks are typically wired as fat trees.
Leaf switches connect to nodes, spine switches connect leaf switches. A
**full bisection bandwidth** fat tree means any node can talk to any other node
at full line rate. In practice, many clusters use **2:1 or 3:1 oversubscription**
(less spine bandwidth than leaf bandwidth) to save cost.

### RoCE (RDMA over Converged Ethernet)

Alternative to InfiniBand using standard Ethernet hardware.

- Cheaper than InfiniBand
- Higher latency, less reliable congestion control
- Used in cloud environments (AWS EFA, OCI RDMA)
- Requires lossless Ethernet configuration (PFC, ECN)

### DGX / HGX Node Architecture (H100)

A standard 8-GPU training node:
- 8× H100 GPUs, each with 80 GB HBM3
- 4× NVSwitch chips connecting all 8 GPUs (full mesh at 900 GB/s)
- 8× ConnectX-7 NICs (one per GPU) for 400 Gb/s InfiniBand each
- 2× CPUs (Intel or AMD) with 1-2 TB system RAM
- Local NVMe storage (typically 8-30 TB)

The NICs are wired so that each GPU has its own dedicated NIC — this is called
**rail-optimized** topology. GPU0 on every node connects to the same network
rail (switch), GPU1 to another rail, etc. This optimizes all-reduce across nodes
because each GPU only talks to its counterpart on other nodes.

---

## 5. Parallelism Strategies

No single GPU can hold a large model. Parallelism is how you distribute the
work across many GPUs. There are four main types, and production systems use
all of them simultaneously (4D parallelism).

### Data Parallelism (DP)

**What**: Each GPU holds a complete copy of the model. Different GPUs process
different micro-batches. After the backward pass, gradients are synchronized.

**Communication**: All-reduce of gradients after each step. Volume = model size.

**When to use**: Always. It's the simplest form and scales the effective batch
size linearly with GPU count.

**Tradeoff**: Every GPU must hold the full model + optimizer state. For large
models, this doesn't fit.

### Fully Sharded Data Parallelism (FSDP / ZeRO)

**What**: Shards model parameters, gradients, and optimizer states across DP
ranks. Each GPU only stores 1/N of the model state. Parameters are gathered
(all-gather) just before use and discarded after.

**Three stages (ZeRO terminology):**
- **Stage 1**: Shard optimizer states only. Memory savings: ~4x.
- **Stage 2**: Shard optimizer states + gradients. Memory savings: ~8x.
- **Stage 3 / FSDP**: Shard everything (parameters too). Memory savings: ~N×.

**Communication**: All-gather before forward/backward, reduce-scatter for gradients.
Total volume ≈ 3× model size per step (more than vanilla DP's 1× all-reduce).

**When to use**: When model state doesn't fit on a single GPU. Standard for
models > 1B parameters.

**Tradeoff**: 3x communication volume vs vanilla DP, but enables training much
larger models. Communication can be overlapped with computation.

### Tensor Parallelism (TP)

**What**: Splits individual layers (matrices) across GPUs. For a linear layer
`Y = XW`, the weight matrix `W` is split column-wise or row-wise across GPUs.

**For a transformer layer:**
```
Attention:  Q, K, V matrices split across TP ranks (column-parallel)
            Output projection split across TP ranks (row-parallel)
MLP:        First linear split column-wise
            Second linear split row-wise
```

Each split requires a collective (all-reduce or reduce-scatter) to combine
partial results:
- Column-parallel → all-reduce after
- Row-parallel → all-reduce before (or reduce-scatter + all-gather with
  sequence parallelism)

**Communication**: 2 all-reduces per transformer layer (one for attention, one
for MLP). Very high frequency, small volume per op.

**When to use**: TP degree typically = number of GPUs per node (8 for DGX).
Must use NVLink — InfiniBand is too slow for the per-layer communication.

**Tradeoff**: High communication frequency requires fast interconnect. Beyond
8-way TP, returns diminish rapidly even on NVLink.

### Sequence Parallelism (SP)

**What**: Extension of TP. In standard TP, LayerNorm and Dropout operate on the
full hidden dimension (replicated across TP ranks). SP splits these operations
along the sequence dimension instead, so each GPU processes a portion of the
sequence.

**Communication**: Converts the all-reduce in TP to reduce-scatter + all-gather,
which can be pipelined. No additional communication volume.

**When to use**: Always paired with TP. Free memory savings on activations.

### Pipeline Parallelism (PP)

**What**: Splits the model by layers across GPUs. GPU0 has layers 0-9, GPU1 has
layers 10-19, etc.

**The bubble problem**: Naive pipeline execution has a bubble where GPUs sit idle
waiting for activations from the previous stage:

```
GPU0: [F0][F1][F2][F3]                    [B3][B2][B1][B0]
GPU1:     [F0][F1][F2][F3]            [B3][B2][B1][B0]
GPU2:         [F0][F1][F2][F3]    [B3][B2][B1][B0]
GPU3:             [F0][F1][F2][F3][B3][B2][B1][B0]
                                    ^--- bubble
```

**Schedules to reduce the bubble:**
- **GPipe**: Simple, large bubble = (PP-1)/total_microbatches
- **1F1B (One Forward One Backward)**: Interleaves forward and backward passes.
  Reduces peak memory but same bubble.
- **Interleaved 1F1B**: Each GPU handles multiple non-contiguous stages (e.g.,
  GPU0 has layers 0-9 and 40-49). Reduces bubble by V× where V = virtual stages.
- **Zero Bubble PP**: Splits backward into B (weight gradient) and W (input gradient),
  schedules them to fill the bubble. Approaches zero bubble at the cost of
  complexity.

**Communication**: Point-to-point (send/recv) between adjacent stages. Only
activation tensors (not full model). Lower volume than DP/TP.

**When to use**: When TP alone can't distribute the model enough. Typical PP
degrees: 2, 4, 8.

**Tradeoff**: Pipeline bubbles waste compute. More micro-batches reduce the
bubble but increase memory. Schedule complexity.

### Context Parallelism (CP)

**What**: Splits the input sequence across GPUs for long-context training. Each
GPU processes a chunk of the sequence. Attention requires all-to-all
communication to exchange KV values.

**When to use**: When sequence length is very long (32K, 128K+) and activation
memory for a single sequence doesn't fit on one GPU.

**Communication**: Ring attention or all-to-all for KV exchange during attention.

### Expert Parallelism (EP)

**What**: For Mixture-of-Experts models. Different experts live on different GPUs.
Tokens are routed to their assigned expert via all-to-all communication.

**Communication**: All-to-all token routing before and after expert computation.

### 4D/5D Parallelism — Putting Them Together

Production systems combine multiple parallelism types. The total number of GPUs:

```
Total GPUs = DP × TP × PP × CP
```

**Standard mapping to hardware hierarchy:**

```
TP = 8 (within node, over NVLink)  ← fastest interconnect
PP = 4-8 (across nodes, over InfiniBand)  ← point-to-point, tolerates latency
DP/FSDP = remaining GPUs (across nodes)  ← all-reduce, overlaps with compute
CP = as needed for long sequences
```

**Example — 405B model on 2048 H100s:**
```
TP = 8 (one node)
PP = 4 (4 nodes per replica)
DP = 2048 / (8 × 4) = 64 replicas
Total: 64 × 4 × 8 = 2048 GPUs
```

**Example — 70B model on 512 H100s:**
```
TP = 8 (one node)
PP = 1 (model fits in 8 GPUs with FSDP)
DP = 512 / 8 = 64 replicas
```

### How to Choose Parallelism Configuration

1. Start with TP = number of GPUs per node (8)
2. Check if model fits in one node with FSDP — if yes, PP = 1
3. If not, add PP until it fits (PP = 2, 4, 8...)
4. Fill remaining GPUs with DP/FSDP
5. Add CP only if sequence length requires it
6. For MoE, add EP for expert layers

---

## 6. Communication Primitives and Libraries

### NCCL (NVIDIA Collective Communications Library)

NCCL is the standard library for GPU-to-GPU collective communication.

**What it does**: Provides optimized implementations of collective operations
that automatically use the fastest available interconnect (NVLink within node,
InfiniBand/RoCE across nodes).

**Key collectives:**

| Operation       | What it does                                           | Used by        |
|----------------|-------------------------------------------------------|----------------|
| All-Reduce     | Sum (or other op) across all GPUs, result on all GPUs  | DP gradients, TP |
| All-Gather     | Each GPU sends its chunk, all GPUs get full tensor     | FSDP parameter gather |
| Reduce-Scatter | Reduce + scatter result chunks to different GPUs       | FSDP gradient sync |
| All-to-All     | Each GPU sends different data to each other GPU        | EP token routing |
| Broadcast      | One GPU sends data to all others                       | Parameter init  |
| Send/Recv      | Point-to-point between two GPUs                        | PP activations  |

**How NCCL optimizes communication:**

1. **Ring algorithm**: GPUs form a ring. Data flows around the ring in chunks.
   For N GPUs, each sends (N-1)/N of data in N-1 steps. Bandwidth-optimal.

2. **Tree algorithm**: GPUs form a binary tree. Better latency for small
   messages (log N steps vs N steps).

3. **NCCL automatically selects** the best algorithm based on message size,
   number of GPUs, and topology.

4. **Kernel fusion**: NCCL can fuse the reduction operation with the
   communication, reducing memory bandwidth pressure.

### Process Groups

PyTorch Distributed uses process groups to define subsets of GPUs that
communicate together. Different parallelism types have different groups:

```python
# Example: 16 GPUs, TP=4, DP=4
# GPU layout: [0,1,2,3] [4,5,6,7] [8,9,10,11] [12,13,14,15]
#              TP group   TP group   TP group     TP group

# TP groups (high-bandwidth, within node):
# [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]

# DP groups (across nodes):
# [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]
```

Each collective operates within its process group. TP all-reduces only involve
the 4 GPUs in the TP group. DP all-reduces only involve the 4 GPUs in the DP
group.

### Communication-Computation Overlap

The key to high MFU is hiding communication behind computation:

**FSDP overlap:**
```
Layer N forward:   [compute XXXXXXXXX]
Layer N+1 gather:  [all-gather XXXXX]      ← prefetch next layer's params
Layer N-1 free:    [free params]           ← discard used params
```

**DP gradient overlap:**
```
Layer N backward:    [compute grad XXXXXXX]
Layer N+1 all-reduce: [reduce-scatter XXXX]  ← sync previous layer's grads
```

The backward pass processes layers in reverse order, so by the time layer N's
backward finishes, layer N+1's gradient all-reduce has already started.

---

## 7. The Three-Way Tradeoff: Computation, Communication, Memory

Every decision in system design involves trading between these three resources.

### Computation vs Memory

| More Compute | More Memory |
|-------------|-------------|
| Activation checkpointing: recompute activations instead of storing them | Store all activations: no recomputation, faster backward |
| Gradient accumulation: more micro-steps, less memory per step | Larger micro-batch: fewer steps, but more activation memory |

### Communication vs Memory

| More Communication | More Memory |
|-------------------|-------------|
| FSDP (shard everything): all-gather params on demand | Replicate params: no gather needed, but N× memory |
| Offload optimizer to CPU: swap over PCIe | Keep optimizer on GPU: fast but limited memory |

### Computation vs Communication

| More Compute | More Communication |
|-------------|-------------------|
| Larger TP degree: less compute per GPU, but more all-reduces | Smaller TP: more compute per GPU, fewer all-reduces |
| Recompute in PP to reduce activation size sent between stages | Send full activations between PP stages |

### The Arithmetic Intensity Framework

**Arithmetic intensity** = FLOPS / bytes_moved

A GPU is **compute-bound** when arithmetic intensity is high (matrix multiplications).
It's **memory/communication-bound** when intensity is low (element-wise ops, collectives).

For a training step to be efficient:
```
Time(compute) >> Time(communication)
```

If communication takes longer than compute, GPUs sit idle. The goal is to:
1. Maximize compute per step (large batch, large matrices)
2. Minimize communication volume (right parallelism choices)
3. Overlap what communication remains with computation

---

## 8. Kernels and Compute Optimization

### What is a Kernel?

A kernel is a function that runs on the GPU. Every operation (matmul, softmax,
layer norm, attention) is a kernel launch. The GPU executes thousands of threads
in parallel.

### FlashAttention

Standard attention computes:
```
Attention(Q, K, V) = softmax(QK^T / √d) × V
```

This requires materializing the full `S × S` attention matrix, which is O(S²)
memory and O(S²) compute.

**FlashAttention** tiles the computation so it never materializes the full
attention matrix. It processes the attention in blocks, keeping running
statistics (online softmax) to compute the correct result.

- Memory: O(S) instead of O(S²)
- Speed: 2-4x faster through better GPU memory access patterns (reduces HBM reads/writes)
- Enables much longer sequences

### Fused Kernels

Instead of launching separate kernels for each operation (which requires
reading/writing to HBM between each), fuse multiple operations into one kernel:

```
Unfused:  LayerNorm → HBM → Dropout → HBM → Add → HBM
Fused:    LayerNorm+Dropout+Add → HBM  (one read, one write)
```

Common fusions:
- **Fused attention**: Q/K/V projection + attention + output projection
- **Fused optimizer**: Adam update + gradient scaling + weight decay in one kernel
- **Fused LayerNorm**: Combine with residual add and/or dropout
- **SwiGLU fusion**: Combine the gated activation with the linear layers

### Triton

Triton is a Python-based language for writing custom GPU kernels. Easier than
CUDA but still produces efficient code. Many fused kernels in modern training
frameworks are written in Triton.

### torch.compile

PyTorch's compiler that automatically fuses operations, eliminates unnecessary
memory accesses, and generates optimized kernels. Can recover 10-30% performance
on non-optimized code. Works by tracing the computation graph and applying
optimizations before generating GPU code.

---

## 9. Mixed Precision and Quantization

### Mixed Precision Training

Train with lower-precision numbers to reduce memory and increase compute throughput.

| Precision | Bytes | Range         | Used for                    |
|-----------|-------|---------------|----------------------------|
| FP32      | 4     | ±3.4×10³⁸    | Master weights, optimizer   |
| BF16      | 2     | ±3.4×10³⁸    | Forward/backward compute    |
| FP16      | 2     | ±65504       | Forward/backward (with loss scaling) |
| FP8       | 1     | ±448 (E4M3)  | Matmuls (H100 Transformer Engine) |

**Standard recipe (BF16 mixed precision):**
- Forward pass: BF16
- Backward pass: BF16
- Master weights: FP32 (for optimizer updates — accumulation in BF16 loses precision)
- Optimizer states: FP32 (Adam momentum and variance)
- Gradient all-reduce: BF16

**Why BF16 over FP16?** BF16 has the same exponent range as FP32 (8 bits), so
it doesn't overflow/underflow during training. FP16 has only 5 exponent bits
and requires loss scaling to avoid underflow in gradients.

### FP8 Training

H100s have dedicated FP8 Tensor Cores that are 2x faster than BF16.

Two FP8 formats:
- **E4M3** (4 exponent, 3 mantissa): Higher precision, used for forward pass
- **E5M2** (5 exponent, 2 mantissa): Higher range, used for backward pass

Challenges:
- Requires per-tensor scaling factors (dynamic or delayed scaling)
- Not all operations are numerically stable in FP8
- Typically only the matmuls are in FP8, everything else stays in BF16/FP32

### Quantization for Training

Different from inference quantization. Training quantization reduces memory and
communication costs:

- **Communication quantization**: Compress gradients to INT8 or FP8 before
  all-reduce, decompress after. Reduces communication volume by 2-4x.
- **Optimizer quantization**: Store Adam states in FP8 instead of FP32.
  4x memory savings on optimizer state.

These are active research areas and not standard practice yet.

---

## 10. Data Loading Pipeline

Data loading seems simple but is a common bottleneck at scale.

### Requirements
- **Throughput**: Must feed GPUs faster than they can consume. A 2048-GPU
  cluster at 100K tokens/second/GPU needs 200M tokens/second.
- **Deterministic**: Must be able to reproduce exact training data order for
  debugging and reproducibility.
- **Resumable**: After a failure, resume from the exact position in the dataset.
- **Shuffled**: Data should be well-shuffled for training quality.
- **No duplicates**: Each token should be seen exactly once per epoch (or
  according to the sampling strategy).

### Architecture

```
Storage (S3/GCS/NFS)
    ↓
Prefetch to local NVMe (async, ahead of training)
    ↓
CPU workers (tokenization, packing, shuffling)
    ↓
Pinned CPU memory (staging area)
    ↓
GPU memory (via DMA/pinned transfer)
```

### Key Design Decisions

**Tokenization**: Pre-tokenize offline (faster startup, deterministic) vs
tokenize online (more flexible, less storage).

**Sequence packing**: Concatenate short documents and split into fixed-length
sequences. Avoids wasting compute on padding tokens. Must handle document
boundaries carefully (attention masks, loss masking).

**Shuffling strategy**:
- Shuffle at the file/shard level (coarse)
- Shuffle sequences within a buffer (fine)
- Pre-shuffled dataset (offline)

**Multi-dataset blending**: Training on a mix of datasets (web, code, books,
math). Requires a sampling strategy:
- Fixed ratios (e.g., 50% web, 20% code, 15% books, 15% math)
- Dynamic upsampling/downsampling based on training progress

**State tracking**: The dataloader must save its position (current file, offset
within file, random seed state) to resume after failures. This state must be
consistent with the model checkpoint.

### Common Issues at Scale
- **I/O bottleneck**: If storage throughput < consumption rate, GPUs idle
- **Stragglers**: One slow worker delays the entire step (synchronous training)
- **Memory pressure**: Too many prefetch workers compete with GPU memory
- **Non-determinism**: Different resume points after failures break reproducibility

---

## 11. Fault Tolerance

At scale, hardware failures are not exceptional — they're expected. A 2048-GPU
cluster with ~0.1% daily GPU failure rate will see ~2 failures per day.

### Failure Modes

| Type              | Frequency    | Impact           | Detection          |
|-------------------|-------------|------------------|--------------------|
| GPU ECC error     | Common      | Single GPU/node  | Hardware interrupt  |
| NIC failure       | Moderate    | Node unreachable | Timeout            |
| Software crash    | Common      | Single process   | Process exit code  |
| Node failure      | Moderate    | 8 GPUs lost      | Heartbeat timeout  |
| Switch failure    | Rare        | Many nodes       | Multiple timeouts  |
| Silent data corruption | Rare   | Wrong gradients  | Validation loss spike |

### Recovery Strategies

**1. Checkpoint and restart (traditional)**
- Periodically save full model state
- On failure, restart all processes from last checkpoint
- Wasted work = steps since last checkpoint
- Simple but expensive at scale (restart ALL GPUs, even healthy ones)

**2. Elastic training (TorchFT approach)**
- Healthy replicas continue training
- Failed replica recovers independently
- Lighthouse coordinates quorum (who's alive)
- Gradient averaging adjusts to fewer replicas
- Minimal disruption to training

**3. Redundant computation**
- Run multiple copies, vote on results
- Detects silent data corruption
- Very expensive (2-3x compute)

### TorchFT Fault Tolerance

TorchFT provides elastic fault tolerance for PyTorch training:

```
Lighthouse Server
    ├── Replica 0 (8 GPUs) — healthy, training
    ├── Replica 1 (8 GPUs) — healthy, training
    └── Replica 2 (8 GPUs) — failed, recovering
```

- **Lighthouse**: Coordination server that maintains quorum of healthy replicas
- **Quorum**: Agreement on which replicas are alive. With `min_replicas=1`,
  training continues even if only 1 replica survives.
- **Recovery**: Failed replica loads checkpoint, syncs state, rejoins quorum.
  Other replicas never stop.
- **Gradient adjustment**: With N replicas, gradients are averaged over N. If
  one fails, average over N-1. Semantically equivalent to a smaller batch size.

### Cost of Failures

Without elastic training (checkpoint-restart):
```
Cost per failure = (time to detect + time to restart + time to reload checkpoint
                    + wasted work since checkpoint) × number of GPUs

Example: 2048 GPUs, 10-minute checkpoint interval, 5-minute restart
Cost per failure = 15 minutes × 2048 GPUs = 512 GPU-hours
At 2 failures/day over 72 days: 72 × 2 × 512 = 73,728 GPU-hours wasted
```

With elastic training:
```
Cost per failure = (recovery time for 1 replica) × (GPUs in that replica)
                   + (slightly reduced throughput during recovery)

Example: 8 GPUs per replica, 2-minute recovery
Cost per failure = 2 minutes × 8 GPUs = 0.27 GPU-hours
At 2 failures/day over 72 days: 72 × 2 × 0.27 = 39 GPU-hours wasted
```

**~1900x reduction in wasted GPU-hours.**

---

## 12. Checkpointing

### What Gets Saved

A complete checkpoint includes:
1. Model parameters (potentially sharded across FSDP ranks)
2. Optimizer states (momentum, variance — 2x model size in FP32)
3. Learning rate scheduler state
4. Dataloader state (position in dataset, RNG states)
5. Training metadata (step number, tokens seen, loss)

### Checkpoint Size

```
70B model:
  Parameters (BF16):     140 GB
  Optimizer (FP32):      560 GB
  Total:                ~700 GB

405B model:
  Parameters (BF16):     810 GB
  Optimizer (FP32):    3,240 GB
  Total:              ~4,050 GB
```

### Checkpointing Strategies

**Synchronous checkpointing:**
- All GPUs stop training, save state, resume
- Simple but creates a "checkpoint stall" (30s to minutes for large models)
- Frequency tradeoff: more frequent = less wasted work on failure, but more
  stall time

**Asynchronous checkpointing:**
- Copy model state to CPU memory (fast — PCIe)
- Resume training immediately
- Background thread writes from CPU to storage
- Requires 2x model memory on CPU side (double buffering)
- Near-zero stall time

**Distributed checkpointing (PyTorch DCP):**
- Each rank saves its own shard in parallel
- Aggregate write throughput = N × per-rank throughput
- Resharding: can load a checkpoint saved with different parallelism config

### Checkpoint Storage

| Storage         | Throughput      | Use Case                   |
|----------------|-----------------|---------------------------|
| Local NVMe     | 5-7 GB/s/node   | Fast temporary checkpoint  |
| Parallel FS    | 10-100 GB/s     | Shared checkpoint storage  |
| Object storage | Variable        | Long-term storage, backup  |

**Two-phase approach:**
1. Fast save to local NVMe (seconds)
2. Async upload to durable storage (minutes, in background)

---

## 13. Monitoring and Observability

You can't optimize what you can't measure.

### Key Metrics

**Training quality:**
- Loss curve (should decrease smoothly)
- Gradient norm (spikes indicate instability)
- Learning rate (verify schedule is correct)
- Tokens per second / samples per second

**System efficiency:**
- **MFU (Model FLOPS Utilization)**: Actual FLOPS / theoretical peak FLOPS.
  The single most important metric.
- **GPU utilization**: Should be >95%. Low utilization = pipeline bubble or
  communication bottleneck.
- **Memory utilization**: How much of 80 GB is used. Too low = batch size
  could be larger. Too high = OOM risk.

**Communication:**
- NCCL collective time per step
- Communication/computation overlap ratio
- Straggler detection (one slow GPU delays entire step)

**Hardware health:**
- GPU temperature (throttling at >83°C)
- ECC error count (uncorrectable = impending failure)
- NIC error rates
- PCIe errors

### Tools
- **TensorBoard / W&B**: Loss curves, learning rate, gradient norms
- **NVIDIA DCGM**: GPU metrics (utilization, memory, temperature, ECC)
- **PyTorch Profiler**: Kernel-level timing, communication breakdown
- **NCCL debug logs**: Communication patterns, algorithm selection
- **Custom dashboards**: Tokens/sec, MFU, per-node throughput

### Debugging Training Issues

| Symptom                    | Likely Cause                                     |
|---------------------------|--------------------------------------------------|
| Loss spike                | Bad data batch, learning rate too high, NaN       |
| Loss plateau              | Learning rate too low, data quality issue          |
| Low MFU                   | Communication bottleneck, pipeline bubble          |
| OOM                       | Batch size too large, activation memory            |
| Slow step time            | Straggler GPU, I/O bottleneck, GC pressure         |
| NaN in gradients          | Numerical instability, bad initialization          |
| Inconsistent loss across runs | Non-determinism in data loading or computation |

---

## 14. Putting It All Together

### Design Checklist

**Phase 1: Requirements**
- [ ] Define model architecture and size
- [ ] Calculate FLOPS and training time estimates
- [ ] Inventory available hardware and interconnects
- [ ] Set MFU and reliability targets

**Phase 2: Parallelism Design**
- [ ] Choose TP degree (= GPUs per node, typically 8)
- [ ] Determine if PP is needed (model too large for one node with FSDP?)
- [ ] Set DP/FSDP degree (= remaining GPUs)
- [ ] Validate memory fits with chosen config
- [ ] Run small-scale experiments to measure MFU

**Phase 3: Systems Infrastructure**
- [ ] Build container image with all dependencies
- [ ] Set up storage for checkpoints and data
- [ ] Configure NCCL environment variables
- [ ] Set up monitoring and alerting

**Phase 4: Data Pipeline**
- [ ] Pre-process and tokenize dataset
- [ ] Implement deterministic, resumable dataloader
- [ ] Validate throughput exceeds training consumption rate
- [ ] Implement multi-dataset blending if needed

**Phase 5: Fault Tolerance**
- [ ] Implement checkpointing (async preferred)
- [ ] Set checkpoint frequency based on failure rate analysis
- [ ] Implement elastic training if multi-replica
- [ ] Test recovery: kill a node, verify training resumes
- [ ] Set up automated health monitoring

**Phase 6: Optimization**
- [ ] Profile and identify bottlenecks
- [ ] Enable communication-computation overlap
- [ ] Tune micro-batch size for optimal memory/compute balance
- [ ] Enable torch.compile where beneficial
- [ ] Experiment with FP8 if on H100s

**Phase 7: Validation**
- [ ] Small-scale convergence test (does the model learn?)
- [ ] Medium-scale efficiency test (does MFU meet target?)
- [ ] Full-scale burn-in (does it run for 24h without failure?)
- [ ] Reproduce known benchmarks to validate setup

### Example Configuration: 70B Model on 512 H100s

```yaml
Model:
  architecture: LLaMA
  parameters: 70B
  layers: 80
  hidden_dim: 8192
  attention_heads: 64
  sequence_length: 8192

Parallelism:
  tensor_parallel: 8          # within node (NVLink)
  pipeline_parallel: 1        # model fits with FSDP
  data_parallel: 64           # 512 / 8 = 64 replicas
  fsdp: full_shard            # shard params + grads + optimizer
  activation_checkpoint: selective

Training:
  precision: BF16
  optimizer: AdamW
  global_batch_size: 4M tokens  # 64 replicas × 8 micro-batches × 8K seq_len
  micro_batch_size: 1
  gradient_accumulation: 8
  learning_rate: 3e-4
  warmup_steps: 2000
  total_tokens: 15T

Fault Tolerance:
  checkpoint_interval: 500 steps
  checkpoint_type: async
  elastic_training: true (TorchFT)
  min_replicas: 1

Hardware:
  gpus: 512x H100 80GB
  interconnect: NVLink (intra-node), InfiniBand NDR (inter-node)
  storage: Parallel filesystem, 50 GB/s aggregate
```

### Expected Performance

```
Theoretical peak: 512 × 989 TFLOPS = 506 PFLOPS
At 50% MFU:       253 PFLOPS sustained
Tokens/sec:       ~500K tokens/sec
Time for 15T:     ~347 days... need more GPUs or higher MFU

At 2048 GPUs, 50% MFU:
Tokens/sec:       ~2M tokens/sec
Time for 15T:     ~87 days
```

This is why large pre-training runs use thousands of GPUs. The economics only
work at scale.
