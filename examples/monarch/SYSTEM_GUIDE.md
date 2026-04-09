# Fault-Tolerant Distributed Training: TorchTitan + Monarch + TorchFT on Kubernetes

A deep-dive into how every component works, how they interact, how failures are injected and recovered from, and how the K8s design differs from SLURM.

---

## Table of Contents

1. [The Three Frameworks](#1-the-three-frameworks)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Monarch: The Actor Framework](#3-monarch-the-actor-framework)
4. [TorchFT: Fault-Tolerant Coordination](#4-torchft-fault-tolerant-coordination)
5. [TorchTitan: The Training Framework](#5-torchtitan-the-training-framework)
6. [The OrchestrationManager](#6-the-orchestrationmanager)
7. [Failure Injection: How It Works](#7-failure-injection-how-it-works)
8. [Recovery: How Replicas Spin Back Up](#8-recovery-how-replicas-spin-back-up)
9. [Checkpointing and Recovery](#9-checkpointing-and-recovery)
10. [PyTorch Components Used](#10-pytorch-components-used)
11. [Kubernetes Integration](#11-kubernetes-integration)
12. [K8s vs SLURM Design Differences](#12-k8s-vs-slurm-design-differences)
13. [Sync vs Async: What Blocks What](#13-sync-vs-async-what-blocks-what)
14. [End-to-End Flow: From Launch to Recovery](#14-end-to-end-flow-from-launch-to-recovery)

---

## 1. The Three Frameworks

Think of three layers stacked on top of each other:

```
+--------------------------------------------------+
|                   TorchTitan                      |  <-- Training logic (model, optimizer, data)
+--------------------------------------------------+
|                    Monarch                        |  <-- Actor framework (process management, supervision)
+--------------------------------------------------+
|                    TorchFT                        |  <-- Fault tolerance (quorum, checkpoint recovery)
+--------------------------------------------------+
|              Kubernetes / SLURM                   |  <-- Infrastructure (pod/job scheduling)
+--------------------------------------------------+
```

- **TorchTitan** is the training framework. It defines the model, optimizer, data pipeline, and training loop. It doesn't know about actors or K8s.
- **Monarch** is the actor framework. It manages processes, spawns actors on remote machines, and handles supervision (detecting when a child process dies). It doesn't know about training or gradients.
- **TorchFT** is the fault-tolerance layer. It coordinates replicas via a Lighthouse, manages quorum (which replicas are alive), and handles checkpoint-based recovery. It doesn't know about actors or K8s.

Each framework does one thing well. The orchestration code (`train_distributed_k8s.py`) is the glue that wires them together.

---

## 2. System Architecture Overview

Based on the Excalidraw diagram (`system_design.excalidraw`):

```
                    +-----------------------------------------+
                    |         Controller Pod (CPU)             |
                    |                                          |
                    |  +------------------+                    |
                    |  | LighthouseServer |  (TorchFT, Rust)   |
                    |  | - quorum mgmt    |                    |
                    |  | - heartbeats     |                    |
                    |  +--------+---------+                    |
                    |           |                               |
                    |  +--------+---------+                    |
                    |  | OrchestrationMgr |  (Python, asyncio)  |
                    |  | - start_training |                    |
                    |  | - _run_replica   |                    |
                    |  | - _teardown      |                    |
                    |  +--------+---------+                    |
                    |           |                               |
                    |     spawn_procs x2                        |
                    |           |                               |
                    |  +--------+--------+--------+            |
                    |  | ReplicaActor 0  | ReplicaActor 1  |   |
                    |  | (local process) | (local process)  |   |
                    |  +--------+--------+--------+---------+  |
                    +-----------|------------------|-----------+
                                |                  |
                    Monarch TCP connections (actor RPC)
                                |                  |
              +-----------------+     +------------+----------+
              |                       |                       |
    +---------v---------+   +---------v---------+
    |   GPU Pod 0       |   |   GPU Pod 1       |
    |   (replica0-0)    |   |   (replica1-0)    |
    |                   |   |                   |
    |  8x TrainingActor |   |  8x TrainingActor |
    |  8x FailureActor  |   |  8x FailureActor  |
    |                   |   |                   |
    |  TorchFT Manager  |   |  TorchFT Manager  |
    |  (rank 0 = server)|   |  (rank 0 = server)|
    |                   |   |                   |
    |  NCCL AllReduce   |   |  NCCL AllReduce   |
    |  (intra-replica)  |   |  (intra-replica)  |
    +-------------------+   +-------------------+
              |                       |
              +--- NCCL cross-replica gradient sync ---+
```

**Key insight**: The controller pod runs NO training. It only runs the Lighthouse, the OrchestrationManager, and lightweight ReplicaActor supervisor processes. All GPU work happens on the GPU pods.

---

## 3. Monarch: The Actor Framework

### What is an Actor?

An actor is a Python class that inherits from `monarch.actor.Actor`. It has:
- **State** (instance variables)
- **Endpoints** (methods decorated with `@endpoint` that can be called remotely)
- **A process** it runs in

```python
class TrainingActor(Actor):
    def __init__(self, config, replica_id):
        self.config = config              # state

    @endpoint
    async def start_training(self, addr):  # endpoint, callable from remote
        trainer = self.config.build()
        trainer.train()
```

### Key Monarch Concepts

**HostMesh**: A set of physical machines (hosts). On K8s, each GPU pod is one host.

**ProcMesh**: A set of processes spawned on a HostMesh. When you call `mesh.spawn_procs({"gpus": 8})`, Monarch creates 8 processes (one per GPU) on that host.

**Spawning**: `proc_mesh.spawn("name", ActorClass, args...)` creates actor instances across all processes in the ProcMesh. If there are 8 processes, you get 8 actor instances.

**`this_host()`**: Returns the HostMesh for the machine the code is currently running on (the controller pod).

**`spawn_procs({"gpus": 1})`**: Spawns a child process on the current host. Used to create ReplicaActors on the controller.

### Supervision Tree

Monarch has a supervision hierarchy. Every actor has a parent:

```
root (controller main process)
├── ReplicaActor 0 (local child process on controller)
│   ├── TrainingActor 0..7 (remote, on GPU pod 0)
│   ├── FailureActor 0..7 (remote, on GPU pod 0)
│   └── log_forwarder (remote, on GPU pod 0)
└── ReplicaActor 1 (local child process on controller)
    ├── TrainingActor 0..7 (remote, on GPU pod 1)
    ├── FailureActor 0..7 (remote, on GPU pod 1)
    └── log_forwarder (remote, on GPU pod 1)
```

**When a child dies, the error cascades UP.** If a TrainingActor's process exits (KILL_PROC), the `log_forwarder` on that process also dies, which causes `SupervisionError` to propagate up to the ReplicaActor, which propagates to root.

**SupervisionError**: The exception Monarch raises when you `await` an endpoint call on a dead actor. This is the mechanism that tells the OrchestrationManager "this replica died."

### Actor Communication

Actors communicate via TCP. When the controller calls:
```python
await replica.actor.start_replica.call_one()
```
This sends a message over TCP to the ReplicaActor process, which then sends messages over TCP to the GPU pods.

`.call_one()` — call on a single actor (when there's exactly one)
`.call()` — call on all actors in the mesh
`.choose()` — call on a randomly selected actor

---

## 4. TorchFT: Fault-Tolerant Coordination

TorchFT has two main components: **Lighthouse** and **Manager**.

### Lighthouse (Rust gRPC Server)

The Lighthouse is a central coordinator that runs on the controller pod. It does NOT participate in training. Its job:

1. **Receive heartbeats** from each replica's Manager (every few seconds)
2. **Form quorums** — decide which replicas are alive and should train together
3. **Broadcast quorum** to all participating replicas

**Quorum formation** (`quorum_compute` in `lighthouse.rs`):
- Collects heartbeats. A replica is "healthy" if its heartbeat is within `heartbeat_timeout_ms`.
- Waits for at least `min_replicas` healthy replicas to request quorum.
- Checks for split-brain: requires at least half of all healthy workers to be participating.
- Optionally waits `join_timeout_ms` for stragglers after minimum is met.
- Assigns a monotonically increasing `quorum_id`.

**Fast quorum path**: If all replicas from the previous quorum are still healthy and participating, the same quorum is re-issued immediately (no reconfiguration needed). This is the common case during normal training — every step gets a fast quorum.

```
Replica 0 Manager ──heartbeat──> Lighthouse
Replica 0 Manager ──quorum req──> Lighthouse ──quorum response──> Replica 0
Replica 1 Manager ──heartbeat──> Lighthouse
Replica 1 Manager ──quorum req──> Lighthouse ──quorum response──> Replica 1
```

### Manager (Python + Rust)

Each replica has one Manager (running on rank 0 of the replica). It has two parts:
- **ManagerServer** (Rust, rank 0 only): gRPC server that talks to the Lighthouse
- **ManagerClient** (Rust FFI, all ranks): talks to the ManagerServer within the replica

**The Manager does three things per training step:**

#### Step 1: `start_quorum()` — "Who is alive?"

Called at the beginning of each training step. All 8 GPU workers in a replica join quorum through rank 0's ManagerServer. The ManagerServer forwards to the Lighthouse and gets back the quorum (list of participating replicas).

If a recovering replica joins with a stale step number, the Manager detects `heal=True` and triggers checkpoint recovery (see Section 9).

The ProcessGroup is reconfigured based on the new quorum — this is how replicas dynamically join and leave.

#### Step 2: `allreduce()` — "Fault-tolerant gradient sync"

During the backward pass, FSDP calls `allreduce` to synchronize gradients. TorchFT wraps this in a `ManagedProcessGroup` that:
- If everything is fine: performs the real NCCL AllReduce
- If an error occurs (e.g., peer died): calls `report_error(e)` and returns a `_DummyWork` (a no-op). All subsequent allreduces for this step become no-ops too.

This is crucial: **a failed collective does NOT crash the training loop.** It silently swallows the error and marks the step as dirty.

#### Step 3: `should_commit()` — "Was this step clean?"

Called after backward, before optimizer step. Every worker in the replica votes:
- `enough_replicas`: are there at least `min_replica_size` replicas in the quorum?
- `errored`: did any collective fail during this step?

The ManagerServer collects all 8 workers' votes. The step is committed ONLY if ALL votes are `True` (no errors, enough replicas).

- **`should_commit=True`**: optimizer steps, step counter increments, checkpoint is staged
- **`should_commit=False`**: optimizer step is SKIPPED, the corrupted gradient is discarded, step counter stays the same. On the next step, quorum is re-formed and the step is retried.

```
    ┌────────────────────────────────────────────────────┐
    │                   Training Step N                   │
    │                                                     │
    │  start_quorum() ──> Lighthouse ──> quorum response  │
    │        │                                            │
    │  forward pass                                       │
    │        │                                            │
    │  backward pass with allreduce()                     │
    │    (if peer died: report_error, continue)           │
    │        │                                            │
    │  should_commit()?                                   │
    │    ├── True:  optimizer.step(), stage checkpoint     │
    │    └── False: skip optimizer, retry step N          │
    └────────────────────────────────────────────────────┘
```

---

## 5. TorchTitan: The Training Framework

TorchTitan provides the training loop, model definitions, optimizers, and data loading. For fault-tolerant training, it uses `FaultTolerantTrainer`.

### FaultTolerantTrainer

Defined in `torchtitan/experiments/ft/trainer.py`. Key methods:

**`__init__` / `build()`**: Creates the model, optimizer, dataloader, checkpoint manager, and FT Manager. Configures FSDP (Fully Sharded Data Parallelism).

**`init_distributed()`**: Sets up distributed training:
- Computes global ranks: `replica_id * group_size + local_rank` (e.g., replica 1, 8 GPUs → ranks 8-15)
- Creates the TorchFT Manager with these ranks
- Creates ProcessGroups for gradient communication

**`train()`**: The main loop:
```python
def train(self):
    self.checkpointer.load()          # load from checkpoint if exists
    for step in range(steps):
        self.train_step(data_iterator)  # forward + backward + optimizer
        self.checkpointer.save()       # stage async checkpoint
```

**`train_step()`**: One training step:
1. `start_quorum()` — join quorum at Lighthouse
2. Forward pass through the model
3. Backward pass — FSDP triggers `allreduce` via `ManagedProcessGroup`
4. `maybe_wait_for_staging()` — wait for previous checkpoint staging to finish
5. `should_commit()` — consensus on whether to apply gradients
6. If committed: `optimizer.step()`, update LR scheduler

### Model (`debugmodel`)

The example uses `model_registry("debugmodel")` which is a small Llama-style transformer for testing. In production you'd use `"llama3_8b"` etc.

### Configuration

All training config is bundled in `FaultTolerantTrainer.Config`:
```python
trainer_config = FaultTolerantTrainer.Config(
    model_spec=model_registry("debugmodel"),
    optimizer=FTOptimizersContainer.Config(lr=8e-4),
    training=TrainingConfig(local_batch_size=8, seq_len=2048, steps=10000),
    fault_tolerance=FaultTolerance(
        enable=True,
        group_size=8,            # 8 GPUs per replica
        process_group="nccl",    # use NCCL for collectives
        process_group_timeout_ms=60000,  # 60s timeout for NCCL ops
    ),
    comm=CommConfig(train_timeout_seconds=300),  # 5min NCCL timeout
    checkpoint=CheckpointManager.Config(),
)
```

This config is serialized and shipped to every TrainingActor on the GPU pods via Monarch's actor message passing.

---

## 6. The OrchestrationManager

The `OrchestrationManager` is the main controller class that ties everything together. It runs on the controller pod inside `asyncio.run(main())`.

### What it does

```python
class OrchestrationManager:
    def __init__(self, spec):
        self.scheduler = MonarchKubernetes(...)   # K8s job manager
        self.replicas = {}                         # active replicas
        self._job_creation_lock = asyncio.Lock()   # prevents concurrent K8s API calls
```

### Lifecycle

```
main()
  └── OrchestrationManager
        ├── start_lighthouse()          # start TorchFT Lighthouse gRPC server
        └── start_training()
              ├── get_or_create_job()    # create K8s jobs for each replica
              ├── _run_replica(0, 0)     # async task for replica 0
              ├── _run_replica(1, 0)     # async task for replica 1 (concurrent)
              └── execute_failures()     # async task for failure injection (concurrent)
```

### `start_training()` — Everything is concurrent

```python
async def start_training(self):
    # Create K8s jobs (sequential — each one talks to K8s API)
    for replica_id in range(self.spec.replica_count):
        await self.scheduler.get_or_create_job(f"replica{replica_id}")

    # Launch replicas concurrently as asyncio tasks
    mesh_futures = {}
    for i in range(self.spec.replica_count):
        mesh_futures[i] = asyncio.create_task(self._run_replica(i, 0))

    # Launch failure injector concurrently
    if self.spec.with_failures:
        failure_future = asyncio.create_task(
            FailureController.execute_failures(self.replicas, self.scheduler)
        )

    # Wait for all replicas to finish (or fail and exhaust retries)
    await asyncio.gather(*mesh_futures.values(), return_exceptions=True)
```

All replicas run as independent asyncio tasks. They don't block each other. The failure controller runs as another independent task.

### `_run_replica()` — The retry loop

```python
async def _run_replica(self, replica_id, attempt_number):
    if attempt_number >= MAX_ATTEMPT:
        return  # give up after 16 failures

    try:
        await self._spin_up_replica(replica_id, attempt_number)
        # If we get here, training finished successfully
        await self._teardown(replica_id)
    except BaseException as e:
        # Something went wrong (SupervisionError, KeyboardInterrupt, etc.)
        await self._teardown(replica_id)
        await self._run_replica(replica_id, attempt_number + 1)  # retry
```

This is a recursive retry loop. Each call to `_spin_up_replica` blocks until either:
- Training completes (success)
- A `SupervisionError` or `KeyboardInterrupt` is raised (failure → teardown → retry)

### `_spin_up_replica()` — The ReplicaActor pattern

```python
async def _spin_up_replica(self, replica_id, attempt_number):
    if attempt_number != 0:
        await self._ensure_job_alive(replica_id, attempt_number)  # check K8s job

    # 1. Spawn a LOCAL process on the controller pod
    replica_proc_mesh = this_host().spawn_procs({"gpus": 1})

    # 2. Create a ReplicaActor in that local process
    replica_actor = replica_proc_mesh.spawn(
        "replica_actor", ReplicaActor, self.spec, replica_id, self.scheduler
    )

    # 3. Tell the ReplicaActor to start (this blocks until training ends or dies)
    await replica.actor.start_replica.call_one()
```

**Why a local ReplicaActor?** This is the supervision boundary. The ReplicaActor is a child process of the controller. It OWNS the remote training actors. When a GPU process dies:
1. Monarch detects the dead process
2. `SupervisionError` propagates to the ReplicaActor
3. The ReplicaActor's `start_replica` endpoint raises the error
4. The `await replica.actor.start_replica.call_one()` in `_spin_up_replica` unblocks with `SupervisionError`
5. `_run_replica`'s `except` clause catches it and retries

Without this pattern, you'd need hacky workarounds (threading.Event, fault hooks, etc.) to detect remote process death from the main controller process.

---

## 7. Failure Injection: How It Works

The failure injection system has three components:

### FailureActor (runs on GPU pods)

One `FailureActor` per GPU process (8 per replica). It's spawned alongside `TrainingActor` on the same ProcMesh:

```python
class FailureActor(Actor):
    @endpoint
    def fail(self, failure: Failure):
        match failure:
            case Failure.SEGFAULT:    # trigger SIGSEGV via ctypes
            case Failure.KILL_PROC:   # os._exit(1) — immediate process death
            case Failure.COMMS:       # abort NCCL process group
            case Failure.DEADLOCK:    # hold GIL for 70s via libc.sleep()
```

### ReplicaActor.inject_failure (runs on controller)

The ReplicaActor has an `inject_failure` endpoint that picks a random FailureActor:

```python
@endpoint
async def inject_failure(self, failure_type):
    await self.failure_actors.fail.choose(failure_type)  # .choose() = random one
```

### FailureController (runs on controller)

An async loop that periodically picks a random replica and failure type:

```python
async def execute_failures(replicas, scheduler, startup_wait=30, rest_time=30):
    await asyncio.sleep(startup_wait)  # let training stabilize
    while replicas:
        replica = random.choice(list(replicas.values()))
        failure = random.choice([SEGFAULT, KILL_PROC, COMMS, DEADLOCK])
        await replica.actor.inject_failure.call_one(failure)
        await asyncio.sleep(rest_time)  # wait before next injection
```

### The failure types and their effects

| Failure | What happens | Process dies? | Detection speed |
|---------|-------------|---------------|-----------------|
| `SEGFAULT` | `ctypes.CFUNCTYPE(None)()` triggers SIGSEGV | Yes, immediately | Instant |
| `KILL_PROC` | `os._exit(1)` kills process | Yes, immediately | Instant |
| `COMMS` | `_abort_process_group()` destroys NCCL | No, process lives | Next allreduce fails |
| `DEADLOCK` | `libc.sleep(70)` holds GIL for 70s | No, process lives | NCCL timeout (300s) |
| `KILL_JOB` | Delete K8s CRD | **No-op on K8s** | N/A |

**KILL_JOB is excluded** on K8s because deleting the MonarchMesh CRD doesn't kill already-connected actor TCP connections — the pod and processes keep running.

### Flow of a KILL_PROC failure

```
FailureController                    ReplicaActor 0              GPU Pod 0
     |                                    |                         |
     |-- inject_failure(KILL_PROC) ------>|                         |
     |                                    |-- fail.choose() ------->|
     |                                    |                    FailureActor.fail()
     |                                    |                    os._exit(1)
     |                                    |                    PROCESS DIES
     |                                    |                         X
     |                                    |<--- SupervisionError ---X
     |                                    X (ReplicaActor dies)
     |<---------- SupervisionError -------X
     |
     | (_run_replica catches it, calls _teardown, then retries)
```

---

## 8. Recovery: How Replicas Spin Back Up

When a replica fails, the recovery happens at two levels: **Monarch level** (process respawning) and **TorchFT level** (checkpoint recovery).

### Monarch Level: Process Respawning

1. `_run_replica` catches the exception
2. `_teardown(replica_id)` stops the old ReplicaActor's proc_mesh
3. `_run_replica` recurses with `attempt_number + 1`
4. `_ensure_job_alive` checks if the K8s job is still running. If not, recreates it.
5. `_spin_up_replica` spawns a NEW ReplicaActor and NEW training actors

```python
# Retry hierarchy:
# attempt 0..3:  reuse existing K8s job (just respawn actors)
# attempt 4:     kill K8s job, create new one (fresh pod)
# attempt 8:     kill and recreate again
# ...
# attempt 16:    give up (MAX_ATTEMPT)
```

### TorchFT Level: Checkpoint Recovery

Once the new training actors start, they call `trainer.train()` which calls `start_quorum()`:

1. The recovering replica joins quorum at the Lighthouse with `step=0`
2. The Lighthouse includes it in the next quorum alongside the healthy replica
3. The Manager computes `heal=True` for the recovering replica (its step < max_step)
4. The healthy replica serves its state dict (model, optimizer, LR scheduler) via HTTP
5. The recovering replica receives and loads the state dict
6. The ProcessGroup is reconfigured to include both replicas
7. Training continues from the healthy replica's step

```
Healthy Replica (step 55)     Lighthouse           Recovering Replica (step 0)
       |                          |                         |
       |-- quorum(step=55) ------>|<--- quorum(step=0) ----|
       |                          |                         |
       |                     forms quorum                   |
       |                     heal=True for recovering       |
       |                          |                         |
       |<-- quorum response ------|---- quorum response --->|
       |   (recover_dst=[1])      |   (heal=True,           |
       |                          |    recover_src=0)        |
       |                          |                         |
       |<----------- HTTP: send state_dict ----------------|
       |                          |                         |
       | (continue at step 55)    |   (load state_dict,    |
       |                          |    continue at step 55) |
```

---

## 9. Checkpointing and Recovery

There are **two types of checkpoints** operating simultaneously:

### Type 1: Per-Replica FT Checkpoint (every step)

- **What**: Only the dataloader state (which data samples have been consumed)
- **Who**: Every replica, independently
- **When**: Every training step, async
- **Where**: `{folder}/ft-replica-{replica_id}/step-{N}`
- **Purpose**: When a replica recovers, it needs to resume the dataloader from the right position to avoid retraining on the same data

This is the "Staging ft checkpoint took 0.001s" log message you see. "Staging" means the data has been copied to CPU memory and the async write has been kicked off.

### Type 2: Full Persistent Checkpoint (periodic)

- **What**: Model weights, optimizer state, LR scheduler state, training state
- **Who**: Only the replica with `participating_rank == 0`
- **When**: At configured intervals (e.g., every 100 steps) and on the last step
- **Where**: Standard DCP checkpoint directory
- **Purpose**: Durable checkpoint for resuming from scratch (not just from a peer)

### Type 3: In-Memory Recovery (on-demand)

- **What**: Full state dict (model, optimizer, LR scheduler, train state)
- **Who**: Healthy replica serves to recovering replica
- **When**: Only during quorum with `heal=True`
- **How**: HTTP transfer via `CheckpointTransport`
- **Purpose**: Fast recovery without touching disk

### The `should_commit` and checkpoint interaction

```
Step N begins
  │
  ├── start_quorum()          ← join quorum, maybe recover from peer
  ├── forward()
  ├── backward() + allreduce() ← might fail silently
  ├── maybe_wait_for_staging() ← wait for step N-1's checkpoint to finish staging
  ├── should_commit()?
  │     ├── True:
  │     │     ├── optimizer.step()
  │     │     ├── stage FT checkpoint (async, dataloader state only)
  │     │     └── maybe save full checkpoint (if at interval)
  │     └── False:
  │           └── skip everything, retry step N
  └── next step
```

**Critical**: `maybe_wait_for_staging()` is called BEFORE `optimizer.step()`. This ensures the model weights aren't modified while they're being asynchronously written to a checkpoint. Without this, you could write a half-updated checkpoint.

### How recovery actually works step by step

1. Replica 0 is at step 55, Replica 1 dies
2. Replica 0 continues training alone (quorum with 1 participant)
3. New Replica 1 starts, calls `trainer.train()` → `start_quorum()` with step 0
4. Lighthouse forms quorum: [Replica 0 at step 55, Replica 1 at step 0]
5. Manager computes: Replica 1 needs to heal, source = Replica 0
6. Replica 0's Manager calls `state_dict()` → captures model, optimizer, LR, train state
7. State dict is sent via HTTP to Replica 1
8. Replica 1's Manager calls `load_state_dict()` → restores everything
9. Both replicas are now at step 55 with identical state
10. Training continues with both replicas

---

## 10. PyTorch Components Used

### FSDP (Fully Sharded Data Parallelism)

The model is wrapped with FSDP, which shards model parameters across GPUs within a replica. Each GPU holds 1/8 of the model. During forward/backward, FSDP performs all-gather and reduce-scatter collectives to reconstruct and synchronize parameters.

### ProcessGroup / NCCL

PyTorch's `torch.distributed` provides the communication backend:
- **NCCL**: GPU-to-GPU communication for AllReduce, AllGather, etc.
- **ProcessGroup**: Abstraction over NCCL that TorchFT wraps as `ManagedProcessGroup`

TorchFT's `ManagedProcessGroup` intercepts every collective call and:
- Routes it through the fault-tolerant Manager
- Catches NCCL errors instead of crashing
- Reconfigures the ProcessGroup when replicas join/leave

### Distributed Data Parallel concepts

- **Gradient AllReduce**: After backward pass, gradients are averaged across all GPUs in a replica
- **Cross-replica sync**: Gradients or model updates are periodically synchronized between replicas (if using DiLoCo/LocalSGD)

### DCP (Distributed Checkpoint)

PyTorch's `torch.distributed.checkpoint` handles parallel saving/loading of sharded model state across multiple GPUs. Used for both FT per-step checkpoints and full periodic checkpoints.

### Activation Checkpointing

`ActivationCheckpointConfig(mode="full")` — during backward pass, recompute activations instead of storing them in memory. Saves GPU memory at the cost of compute.

---

## 11. Kubernetes Integration

### How K8s fits in

K8s provides the infrastructure layer: it schedules pods onto nodes and manages their lifecycle.

```
K8s Cluster
├── Namespace: monarch-tests
│   ├── Pod: hossein-controller (CPU, manually created)
│   │   └── runs: OrchestrationManager + Lighthouse
│   ├── MonarchMesh CRD: replica0 (created by KubernetesJob)
│   │   └── Pod: replica0-0 (8x A100 GPUs)
│   └── MonarchMesh CRD: replica1 (created by KubernetesJob)
│       └── Pod: replica1-0 (8x A100 GPUs)
```

### MonarchKubernetes scheduler

The `MonarchKubernetes` class manages `KubernetesJob` objects from the Monarch library:

```python
class MonarchKubernetes:
    async def get_or_create_job(self, mesh_name):
        job = KubernetesJob(namespace=self.namespace)
        pod_spec = build_gpu_pod_spec(self.image, self.gpus_per_host)
        job.add_mesh(mesh_name, num_replicas=1, pod_spec=pod_spec)

    def proc_mesh(self, mesh_name, num_procs):
        job = self.job_handles[mesh_name]
        mesh = getattr(job.state(), mesh_name)  # HostMesh
        return mesh.spawn_procs({"gpus": num_procs})  # ProcMesh
```

### What `KubernetesJob` does

When `job.add_mesh("replica0", num_replicas=1, pod_spec=pod_spec)` is called:
1. Creates a `MonarchMesh` Custom Resource Definition (CRD) in the K8s namespace
2. The Monarch K8s operator (running as a controller in the cluster) sees the CRD
3. The operator creates the Pod described by `pod_spec`
4. The Pod runs `run_worker_loop_forever()` — a Monarch worker that listens for actor spawn requests
5. `job.state()` returns the HostMesh handle once the pod is ready

### The worker bootstrap script

Every GPU pod runs this on startup:

```python
from monarch.actor import run_worker_loop_forever
port = os.environ.get("MONARCH_PORT", "26600")
hostname = socket.getfqdn()
address = f"tcp://{hostname}:{port}"
run_worker_loop_forever(address=address, ca="trust_all_connections")
```

This starts a Monarch worker that:
1. Binds to TCP port 26600
2. Waits for the controller to connect and spawn actors
3. Runs actors (TrainingActor, FailureActor) when told to

### RBAC

The controller pod needs K8s permissions to create/delete MonarchMesh CRDs:
```yaml
rules:
  - apiGroups: ["monarch.pytorch.org"]
    resources: ["monarchmeshes"]
    verbs: ["create", "get", "patch", "delete"]
```

### Headless Service

A headless Service (`clusterIP: None`) is created for the controller pod so that GPU pods can resolve its hostname via DNS. This is needed because the Lighthouse address uses the controller's FQDN (e.g., `hossein-controller.hossein-controller-svc.monarch-tests.svc.cluster.local`).

---

## 12. K8s vs SLURM Design Differences

### Job management

| Aspect | K8s (`train_distributed_k8s.py`) | SLURM (`train_distributed.py`) |
|--------|------|------|
| **Scheduler class** | `MonarchKubernetes` | `MonarchSlurm` |
| **Job abstraction** | `KubernetesJob` → MonarchMesh CRD → Pod | `SlurmJob` → sbatch → SLURM allocation |
| **One job per replica?** | Yes (independent kill/recreate) | Depends: 1 replica → 1 job; >1 replica → 1 shared job |
| **Kill on failure** | Kill only the failed replica's CRD | Kill ALL jobs (shared SLURM allocation) |
| **Pod spec** | Explicit `V1PodSpec` with GPU resources, `/dev/shm` mount | Implicit via SLURM `--gpus-per-node` |
| **DNS resolution** | Headless Service + FQDN | SLURM provides hostnames |

### Lighthouse placement

| Aspect | K8s | SLURM |
|--------|-----|-------|
| **Where** | Directly on controller pod (same process) | In a separate `LighthouseActor` on a local or remote proc_mesh |
| **Sync/Async** | `start_lighthouse()` is synchronous | `start_lighthouse()` is async (awaits actor endpoint) |
| **Lifecycle** | Managed directly (no actor wrapper) | Managed via actor pattern |

The K8s version runs the Lighthouse directly in the main process because the controller pod is a stable, always-on pod. The SLURM version wraps it in an actor because on SLURM the controller might be running on a compute node that could get preempted.

### Job recreation on failure

**K8s**: Each replica has its own `KubernetesJob`. When replica 1 fails:
```python
# Only kill and recreate replica 1's job
self.scheduler.kill_job("replica1")
await self.scheduler.get_or_create_job("replica1")
```

**SLURM**: When replica count > 1, all replicas share one SLURM job. When any replica fails:
```python
# Kill ALL jobs and recreate everything
self.scheduler.kill_jobs()
await self._create_all_jobs()
```

This is because SLURM allocations are tied together — you can't kill one node in a multi-node allocation without affecting the others. K8s pods are independent.

### Configuration transport

**K8s**: Uses `monarch.config.configure(enable_log_forwarding=True, ...)` — Python-level config.

**SLURM**: Uses `configure(default_transport=ChannelTransport.TcpWithHostname)` — Rust-level config to ensure cross-node TCP communication works on SLURM's network.

### KILL_JOB failure type

**K8s**: Excluded because deleting the MonarchMesh CRD doesn't kill running actor TCP connections — the pod stays alive.

**SLURM**: Works because killing the SLURM job terminates all processes.

---

## 13. Sync vs Async: What Blocks What

### The asyncio event loop

The controller runs a single Python asyncio event loop. Everything that says `async def` or `await` is cooperative — it yields control back to the event loop when waiting.

```python
asyncio.run(main())  # starts the event loop
```

### What runs concurrently (async tasks)

```
Event Loop
├── Task: _run_replica(0, 0)       # blocks on await start_replica.call_one()
├── Task: _run_replica(1, 0)       # blocks on await start_replica.call_one()
└── Task: execute_failures()       # blocks on await asyncio.sleep(rest_time)
```

These three tasks run concurrently. While `_run_replica(0)` is waiting for replica 0 to finish training, `_run_replica(1)` and `execute_failures()` can proceed independently.

### What blocks within a task

| Operation | Blocking? | What waits? |
|-----------|-----------|-------------|
| `await start_replica.call_one()` | Yes (async) | Controller waits for training to finish or crash |
| `await scheduler.get_or_create_job()` | Yes (async) | Waits for K8s to create pod and become ready |
| `await asyncio.sleep(rest_time)` | Yes (async) | Failure controller pauses between injections |
| `trainer.train()` | Yes (sync, on GPU pod) | TrainingActor blocks running the training loop |
| `start_quorum()` → Lighthouse | Yes (sync, on GPU pod) | All GPUs in replica wait for quorum |
| `allreduce()` | Yes (sync, on GPU pod) | All GPUs wait for NCCL collective |
| `should_commit()` | Yes (sync, on GPU pod) | All GPUs wait for commit decision |

### Do K8s and Monarch block each other?

**No.** They operate at different levels:

- **K8s API calls** (`get_or_create_job`, `kill_job`) are async and only happen during job creation/teardown. They complete in seconds.
- **Monarch actor communication** (TCP messages to GPU pods) happens continuously during training. It's independent of K8s.
- **K8s pod scheduling** happens before Monarch connects. Once the pod is running and Monarch has established TCP connections, K8s is out of the picture for training.

The only interaction is:
1. K8s creates/deletes pods → Monarch connects/disconnects from them
2. A `_job_creation_lock` prevents concurrent K8s API calls during replica respawning

### Do replicas block each other?

**At the Lighthouse**: Yes, briefly. Quorum formation waits for `min_replicas` to join. If only 1 replica is alive, quorum proceeds with 1 participant (because `min_replicas=1`).

**At NCCL**: Only within a replica. The 8 GPUs in a replica synchronize via NCCL. Different replicas don't directly block each other's NCCL collectives (unless using cross-replica sync like DiLoCo).

**At the controller**: No. Each replica's `_run_replica` is an independent asyncio task.

---

## 14. End-to-End Flow: From Launch to Recovery

### Phase 1: Startup

```
1. User runs: python train_distributed_k8s.py --namespace monarch-tests --image ... --with-failures
2. asyncio.run(main()) starts
3. OrchestrationManager.__init__() creates MonarchKubernetes scheduler
4. start_lighthouse() starts TorchFT Lighthouse gRPC server on controller pod
5. start_training():
   a. get_or_create_job("replica0") → K8s creates MonarchMesh CRD → Pod replica0-0 starts
   b. get_or_create_job("replica1") → K8s creates MonarchMesh CRD → Pod replica1-0 starts
   c. asyncio.create_task(_run_replica(0, 0)) → spawns ReplicaActor 0
   d. asyncio.create_task(_run_replica(1, 0)) → spawns ReplicaActor 1
   e. asyncio.create_task(execute_failures()) → starts failure injection timer
```

### Phase 2: Normal Training

```
6. ReplicaActor 0 connects to GPU Pod 0:
   a. scheduler.proc_mesh("replica0", 8) → gets ProcMesh with 8 GPUs
   b. Spawns 8 TrainingActors + 8 FailureActors
   c. Calls training_actors.start_training.call(lighthouse_address)
7. Each TrainingActor:
   a. Creates FaultTolerantTrainer
   b. Joins quorum at Lighthouse (both replicas join → quorum formed)
   c. Runs training loop: forward → backward → allreduce → should_commit → optimizer.step
```

### Phase 3: Failure Injection

```
8. After 30 seconds, FailureController picks replica 0, picks KILL_PROC
9. Calls replica_0_actor.inject_failure(KILL_PROC)
10. ReplicaActor 0 calls failure_actors.fail.choose(KILL_PROC)
11. Random FailureActor on GPU Pod 0 calls os._exit(1)
12. The process dies immediately
```

### Phase 4: Error Detection

```
13. Monarch detects the dead process on GPU Pod 0
14. SupervisionError cascades: log_forwarder → ReplicaActor 0 → root
15. "Unhandled monarch error on root actor" delivers KeyboardInterrupt to main thread
16. The await in _run_replica(0) unblocks with the exception
```

### Phase 5: Recovery

```
17. _run_replica(0) catches the exception in except BaseException
18. _teardown(0): stops the old ReplicaActor's proc_mesh, cleans up
19. _run_replica(0, attempt_number=1):
    a. _ensure_job_alive(0, 1): checks if K8s job is still active
    b. If pod died: kill_job + get_or_create_job → new pod
    c. If pod alive: reuse existing pod
    d. Waits PROC_ATTEMPT_DELAY (5s) for Monarch cleanup
    e. Spawns NEW ReplicaActor 0 → connects to GPU Pod 0
    f. Spawns new TrainingActors
    g. TrainingActors call trainer.train() → start_quorum()
```

### Phase 6: Checkpoint Recovery via TorchFT

```
20. New replica 0 joins quorum at Lighthouse with step=0
21. Healthy replica 1 is at step=55
22. Lighthouse forms quorum: [replica0 step=0, replica1 step=55]
23. Manager computes: replica 0 needs healing, source = replica 1
24. Replica 1 serializes state dict (model, optimizer, LR, train state)
25. Sends via HTTP to replica 0
26. Replica 0 loads state dict, is now at step 55
27. ProcessGroup reconfigured with both replicas
28. Training continues at step 55 with both replicas in sync
```

### Phase 7: Continued Training

```
29. Both replicas train together, quorum has 2 participants
30. 30 seconds later, FailureController injects another failure
31. Cycle repeats: detect → teardown → respin → recover → continue
32. After 10000 steps (or MAX_ATTEMPT failures), training ends
```
