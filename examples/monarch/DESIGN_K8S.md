# Kubernetes Adaptation of train_distributed.py

## Overview

`train_distributed_k8s.py` adapts the SLURM-based `train_distributed.py` to run
fault-tolerant distributed training on Kubernetes. The core training logic
(actors, TorchFT coordination, failure recovery) remains the same. The changes
are focused on replacing the SLURM scheduler abstraction with a Kubernetes one
and handling K8s-specific constraints.

## Reference: Working DDP Example

The Monarch DDP K8s example (`kubernetes_ddp_v0.4.0.py`) served as the reference
for how to correctly use `KubernetesJob`. Key patterns borrowed:

- `KubernetesJob(namespace=...)` for job creation
- `job.add_mesh(name, num_replicas=N, ...)` for mesh definition
- `job.state()` to connect to running pods and get `HostMesh`
- `host_mesh.spawn_procs({"gpus": N})` to create GPU processes
- Controller pod + MonarchMesh CRD pattern (no worker YAML needed)

## Changes from SLURM to Kubernetes

### 1. Scheduler Abstraction: `MonarchSlurm` -> `MonarchKubernetes`

**SLURM (original):**
```python
class MonarchSlurm:
    def __init__(self):
        self.job_handles: Dict[str, SlurmJob] = {}

    async def get_or_create_job(self, mesh_name, nodes_per_mesh=1, gpus_per_node=8):
        job = SlurmJob(
            meshes={mesh_name: nodes_per_mesh},
            gpus_per_node=gpus_per_node,
            job_name=f"{self.job_name_prefix}-{mesh_name}",
        )
        job.apply()
        self.job_handles[mesh_name] = job
```

**Kubernetes (new):**
```python
class MonarchKubernetes:
    def __init__(self, namespace, image_spec=None, timeout=None):
        self.namespace = namespace
        self.image_spec = image_spec
        self.timeout = timeout
        self.job_handles: Dict[str, KubernetesJob] = {}
        self._is_owner = True
        atexit.register(self.kill_jobs)

    async def get_or_create_job(self, mesh_name):
        job = KubernetesJob(namespace=self.namespace, timeout=self.timeout)
        job.add_mesh(mesh_name, num_replicas=1, image_spec=self.image_spec)
        job.apply()
        self.job_handles[mesh_name] = job
```

**What changed and why:**
- `SlurmJob` replaced with `KubernetesJob` which creates MonarchMesh CRDs
  instead of submitting `srun` jobs.
- Constructor takes `namespace` (required for K8s), `image_spec` (container
  image for worker pods), and `timeout` (pod readiness wait time).
- Each replica gets its own `KubernetesJob` (one-to-one mapping), same as
  SLURM. This is critical for independent failure recovery.
- `image_spec` supports two modes:
  - **Provisioning mode** (`--image` flag): Monarch operator creates worker pods
    with the specified image.
  - **Attach mode** (no `--image`): Connects to pre-existing pods.

### 2. Mesh Naming: Underscores Removed

**SLURM:** `replica_0`, `replica_1`, ...
**Kubernetes:** `replica0`, `replica1`, ...

K8s resource names must conform to RFC 1123 (lowercase alphanumeric and hyphens
only). Underscores are not allowed. This affected every reference to mesh names
throughout the script:
- `get_or_create_job(f"replica{replica_id}")`
- `self.scheduler.proc_mesh(f"replica{self.replica_id}", ...)`
- `_ensure_job_alive` mesh name construction

### 3. Pickle Protection: `__getstate__` / `__setstate__`

**SLURM:** Not needed.
**Kubernetes:** Added to `MonarchKubernetes`.

```python
def __getstate__(self):
    state = self.__dict__.copy()
    state["_is_owner"] = False
    return state
```

**Why:** When `MonarchKubernetes` is serialized (pickled) and sent to actor
subprocesses via Monarch RPC, the deserialized copy in the subprocess would
register its own `atexit` handler. When the subprocess exits, it would
accidentally call `kill_jobs()` and destroy all worker pods. Setting
`_is_owner = False` on pickle ensures only the original controller process can
kill jobs.

### 4. Concurrency Lock: `_job_creation_lock`

**SLURM:** No lock.
**Kubernetes:** Added `asyncio.Lock` in `OrchestrationManager`.

```python
self._job_creation_lock = asyncio.Lock()

async def _spin_up_replica(self, replica_id, attempt_number=0):
    if attempt_number != 0:
        async with self._job_creation_lock:
            await self._ensure_job_alive(replica_id, attempt_number)
```

**Why:** In K8s, pod failures are isolated -- replica 0 can crash independently
of replica 2. If both crash simultaneously, their `_run_replica` asyncio tasks
run concurrently and both call `_ensure_job_alive`, which reads and modifies the
shared `job_handles` dictionary. The lock serializes these operations to prevent
race conditions (e.g., one coroutine deleting a key while another reads it).

The SLURM version didn't need this because `srun` propagates failures to all
processes, so replicas effectively fail and recover sequentially. However, this
is arguably a bug in the SLURM version -- concurrent failures are still possible
in theory.

### 5. Independent Replica Recovery: `_ensure_job_alive`

**SLURM (original, inside `_spin_up_replica`):**
```python
if attempt_number != 0 and attempt_number % PROC_ATTEMPTS == 0:
    self.scheduler.kill_job(f"replica_{replica_id}")
    await self.scheduler.get_or_create_job(f"replica_{replica_id}", ...)
```

**Kubernetes (new, extracted to `_ensure_job_alive`):**
```python
async def _ensure_job_alive(self, replica_id, attempt_number):
    mesh_name = f"replica{replica_id}"
    if attempt_number % PROC_ATTEMPTS == 0:
        self.scheduler.kill_job(mesh_name)
        await self.scheduler.get_or_create_job(mesh_name)
    else:
        job = self.scheduler.job_handles.get(mesh_name)
        if job is None or not job.active:
            self.scheduler.kill_job(mesh_name)
            await self.scheduler.get_or_create_job(mesh_name)
```

**What changed:**
- Extracted to a separate method (called under the lock).
- Added an `else` branch that checks `job.active` -- in K8s, a pod can
  terminate without triggering a full job recreation cycle. This check detects
  dead pods between the periodic `PROC_ATTEMPTS` threshold and recreates them
  immediately rather than waiting.

### 6. Exception Handling: `BaseException` instead of `Exception`

**SLURM:**
```python
except Exception as e:
    await self._teardown(replica_id)
```

**Kubernetes:**
```python
except BaseException as e:
    if isinstance(e, KeyboardInterrupt):
        raise
    await self._teardown(replica_id)
```

**Why:** Monarch's async `CancelledError` during K8s pod cleanup inherits from
`BaseException`, not `Exception`. Without catching `BaseException`, cleanup
failures during teardown would propagate unhandled and crash the orchestrator.
`KeyboardInterrupt` is re-raised to allow manual script termination.

### 7. Configuration System: `ConfigManager` -> Direct Dataclass Construction

**SLURM:** Used TorchTitan's `ConfigManager` with CLI-style argument parsing:
```python
default_args = [
    "--job.config_file", os.path.join(script_dir, args.model_config),
    "--fault_tolerance.enable",
    "--fault_tolerance.group_size", str(args.replica_count),
    ...
]
config_manager = ConfigManager()
job_config = config_manager.parse_args(default_args)
```

**Kubernetes:** Uses typed dataclass configs directly:
```python
trainer_config = FaultTolerantTrainer.Config(
    model_spec=model_registry("debugmodel"),
    training=TrainingConfig(local_batch_size=8, seq_len=2048, steps=args.training_steps),
    fault_tolerance=FaultTolerance(
        enable=True,
        group_size=data_parallel_shard_degree,
        process_group="nccl",
    ),
    ...
)
```

**Why:** The K8s version uses TorchTitan's `experiments.ft` module which provides
`FaultTolerantTrainer` with typed `Config` dataclasses. This is cleaner than
constructing fake CLI arguments and avoids dependency on TOML config files that
would need to be present inside the container.

### 8. CLI Arguments

**Removed (SLURM-specific):**
- `--host-per-replica` (K8s uses single-pod replicas)
- `--model-config` (replaced by direct config construction)
- `--dataset-path` as relative path (K8s uses HuggingFace download or absolute path)

**Added (K8s-specific):**
- `--namespace` (required): K8s namespace for all resources
- `--image` (optional): Container image for worker pods. Enables provisioning
  mode. Without it, attaches to pre-existing pods.
- `--timeout` (optional): Seconds to wait for pod readiness

**Renamed:**
- `--gpu-per-node` -> `--gpus-per-host` (K8s terminology, pods not nodes)

### 9. Imports

**Removed:**
- `from monarch.job import SlurmJob`
- `from monarch.utils import setup_env_for_distributed`
- `from torchtitan.config import ConfigManager, JobConfig`
- `from torchtitan.train import Trainer`

**Added:**
- `from monarch.job.kubernetes import KubernetesJob, ImageSpec`
- `from monarch.spmd import setup_torch_elastic_env_async`
- `from torchtitan.experiments.ft.trainer import FaultTolerantTrainer`
- `from torchtitan.experiments.ft.config.job_config import FaultTolerance`
- Various TorchTitan config imports for direct dataclass construction

## Launcher YAML (`launcher.yaml`)

The controller pod manifest was created from scratch for K8s. It deploys
the orchestration environment that the training script runs inside.

### Components

1. **Namespace** (`monarch-tests`): Isolates all resources.

2. **Controller ServiceAccount + Role + RoleBinding**: Grants the controller pod
   permissions to:
   - `get`, `list`, `watch` pods (monitor worker pod status)
   - `create`, `get`, `patch`, `delete` MonarchMesh CRDs (manage worker lifecycle)

3. **Controller Pod** (`hossein-controller`):
   - Runs on GPU nodes via `nodeAffinity` (`BM.GPU.H100.8`, `BM.GPU.A100-v2.8`)
   - Tolerates GPU node taints (`nvidia.com/gpu`)
   - Mounts `/dev/shm` (16Gi) for NCCL shared memory
   - Uses custom image with TorchFT + TorchTitan installed
   - Runs `sleep infinity` -- user execs in and runs the training script manually

4. **monarch-client ServiceAccount + RoleBinding**: Worker pods created by the
   Monarch operator need permissions to register with the operator. References
   the pre-existing `monarch-client-role` ClusterRole.

### Why a Controller Pod?

The training script must run **inside** the K8s cluster because
`KubernetesJob` requires in-cluster config (`config.load_incluster_config()`).
It cannot run from the jump server or a local machine. The controller pod
provides this environment.

## Architecture Diagram

```
Jump Server (kubectl)
    |
    | kubectl exec
    v
Controller Pod (hossein-controller)
    |
    | runs train_distributed_k8s.py
    | OrchestrationManager
    |     |
    |     | KubernetesJob.apply() -> creates MonarchMesh CRD
    |     |                              |
    |     |                              v
    |     |                    Monarch Operator
    |     |                              |
    |     |                              v
    |     |                    Worker Pod (replica0-0)
    |     |                         8x GPU processes
    |     |                         TrainingActor per GPU
    |     |
    |     | LighthouseActor (local or remote)
    |     |     coordinates TorchFT quorum
    |     |
    |     | On failure:
    |     |   1. Detect (Monarch RPC exception)
    |     |   2. Teardown (stop proc mesh)
    |     |   3. Recreate (kill CRD -> new CRD -> new pod)
    |     |   4. Rejoin (TorchFT syncs state)
```

## Known Issue: `job.apply()` Timing

The current script calls `job.apply()` in `get_or_create_job()`, then calls
`job.state()` later when spawning processes. The DDP reference example does NOT
call `apply()` -- it lets `state()` handle both creation and waiting for pod
readiness. This causes a race condition where `spawn_procs()` times out (30s)
because the worker pod isn't ready when the controller tries to connect.

**Fix:** Remove the explicit `job.apply()` call and let `state()` handle it, or
ensure `state()` is called immediately after `apply()` to block until pods are
ready.
