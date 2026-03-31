# TorchFT + Monarch + TorchTitan: Distributed Fault-Tolerant Training Progress Report

**Date:** March 31, 2026
**Cluster:** 2x bare-metal EC2 instances with 8x H100 GPUs each
**Nodes:** `ip-10-0-85-22` (node 1) and `ip-10-0-91-197` (node 2)
**Dev machine:** `hosseinkh` (code editing) -> deploys to `dev@ip-10-0-85-22` (execution)

---

## 1. Infrastructure Setup

### 1.1 SLURM + Munge Installation

We installed SLURM 21.08.5 with Munge authentication on both bare-metal EC2 nodes to provide workload management for Monarch's `SlurmJob` scheduler.

**Munge setup:**
- Installed `munge` on both nodes
- Generated a shared Munge key and copied it to both nodes at `/etc/munge/munge.key`
- Ensured consistent permissions (`chmod 400`, owned by `munge:munge`)
- Started `munged` daemon on both nodes
- Verified cross-node authentication with `munge -n | ssh node2 unmunge`

**SLURM setup:**
- Installed `slurmd` (worker daemon) on both nodes
- Installed `slurmctld` (controller daemon) on node 1 (`ip-10-0-85-22`)
- Configured `slurm.conf` with both nodes listed, GPU resources (8 GPUs per node)
- Started `slurmctld` on node 1, `slurmd` on both nodes
- Verified with `sinfo` showing both nodes in `idle` state

### 1.2 Software Installation

**On both nodes:**
- Python 3.11 via `/opt/conda/bin/python3`
- Monarch (Meta's distributed actor framework) - editable install from source at `/home/dev/monarch`
- TorchFT (fault-tolerant training) - editable install from source at `/home/dev/dev-torchft`
- TorchTitan (PyTorch training framework)
- PyTorch with CUDA support for H100 GPUs

---

## 2. Issues Encountered and Fixes

### 2.1 Python Version Mismatch Between Nodes

**Problem:** Node 1's `python3` resolved to `/opt/conda/bin/python3` (Python 3.11), while node 2's `python3` resolved to `/usr/bin/python3` (Python 3.10). The Monarch editable install `.pth` file was at `~/.local/lib/python3.11/site-packages/`, which Python 3.10 never reads.

**Symptom:** `ModuleNotFoundError: No module named 'monarch.actor'` on node 2 when SLURM ran worker processes.

**Fix:** Changed `python_exe` in both `SlurmJob` constructors from `"python3"` to `"/opt/conda/bin/python3"` (absolute path), ensuring SLURM's `srun` uses the correct Python on all nodes.

### 2.2 SLURM 21.x `squeue --json` Returns All Jobs

**Problem:** SLURM 21.08.5's `squeue --job <ID> --json` ignores the `--job` filter and returns ALL jobs in the queue. Monarch's `_get_job_info_json()` used `jobs[0]` which would pick up a previously CANCELLED job's status instead of the requested one.

**Symptom:** Newly submitted SLURM jobs appeared to be immediately CANCELLED because the code read a stale CANCELLED job from a previous run.

**Discovery:** Diagnostic command `squeue --job 78 --json | python3 -c "import sys,json; jobs=json.load(sys.stdin)['jobs']; print([(j['job_id'],j['job_state']) for j in jobs])"` revealed `[(77, 'CANCELLED'), (78, 'RUNNING')]`.

**Fix:** Created `fix_squeue.py` patch script that modifies Monarch's `_get_job_info_json()` to filter jobs by `job_id` before falling back to `jobs[0]`:
```python
for job in jobs:
    if str(job.get("job_id")) == str(job_id):
        return job
return jobs[0] if jobs else None
```

### 2.3 Cross-Node Unix Socket Communication Failure

**Problem:** Monarch defaulted to Unix domain sockets for actor communication. When `ReplicaActor` was spawned on `this_host()` (node 1), it used a Unix socket address. Workers on node 2 couldn't connect back via Unix sockets across the network.

**Symptom:** `Timeout(30.001484779s)` errors when spawning proc meshes. Monarch log showed: `session unix:@5lpzh6ss0gRlS7F6xBjUXiJn: failed to deliver message within timeout 30s; never connected`

**Fix:** Added TCP transport configuration at the module level in `train_distributed.py`:
```python
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
configure(default_transport=ChannelTransport.TcpWithHostname)
```

### 2.4 MonarchSlurm atexit Handler Killing Jobs in Subprocesses

**Problem:** When `MonarchSlurm` was pickled and sent to `ReplicaActor` (which runs in a subprocess), the `atexit` handler would fire on subprocess exit and cancel the SLURM jobs.

**Fix:** Added `__getstate__`/`__setstate__` to `MonarchSlurm`:
```python
def __getstate__(self):
    state = self.__dict__.copy()
    state["_is_owner"] = False
    return state
```
The `kill_jobs()` method checks `_is_owner` before cancelling.

### 2.5 Shared Multi-Mesh SlurmJob Retry Logic

**Problem:** For 2-replica training, both replicas share a single `SlurmJob` with meshes `{"replica_0": 1, "replica_1": 1}`. The original retry logic called `kill_job("replica_X")` which killed the shared job for ALL replicas.

**Fix:** Updated `_spin_up_replica` to check `replica_count`: for multi-replica, recreate the entire shared job via `_create_all_jobs()` instead of killing individual mesh entries.

### 2.6 CancelledError Crashing the Retry Loop

**Problem:** When a process-level failure (SEGFAULT, KILL_PROC) killed a trainer, Monarch's `proc_mesh.stop()` raised `asyncio.CancelledError`. In Python 3.9+, `CancelledError` inherits from `BaseException`, not `Exception`, so it bypassed all `except Exception` handlers and crashed the entire program.

**Fix:** Changed `except Exception` to `except BaseException` in both `_run_replica` and `_teardown`, with a re-raise for `KeyboardInterrupt`:
```python
except BaseException as e:
    if isinstance(e, KeyboardInterrupt):
        raise
    await self._teardown(replica_id)
    await self._run_replica(replica_id, attempt_number + 1)
```

### 2.7 Stale In-Memory SLURM Job State

**Problem:** After a process crash, SLURM marks the job as FAILED, but Monarch's `SlurmJob.active` property only checks the in-memory `_status` flag (still "running"). The retry logic thought the job was alive and kept trying to spawn trainers on a dead job, wasting 30 seconds per attempt on proc mesh timeouts.

**Fix:** Replaced `job.active` with `job._jobs_active()` which actually queries `squeue --json` to check the real SLURM job state. Also added a 5-second delay before checking to let SLURM propagate the state change:
```python
await asyncio.sleep(5)
if job is None or not job._jobs_active():
    self.scheduler.kill_jobs()
    await self._create_all_jobs()
```

### 2.8 Concurrent Job Creation Race Condition

**Problem:** When both replicas fail simultaneously (common after any failure since both share the same SLURM job), both retry loops would concurrently call `_create_all_jobs()`, creating duplicate SLURM jobs.

**Fix:** Added `asyncio.Lock` to serialize job creation:
```python
self._job_creation_lock = asyncio.Lock()

async def _spin_up_replica(self, replica_id, attempt_number):
    if attempt_number != 0:
        async with self._job_creation_lock:
            await self._ensure_jobs_alive(replica_id, attempt_number)
```

---

## 3. Architecture

### 3.1 System Components

```
Controller Process (ip-10-0-85-22)
├── OrchestrationManager
│   ├── LighthouseActor (torchft quorum coordination)
│   ├── MonarchSlurm (SLURM job management)
│   └── ReplicaActor x2 (one per replica)
│       ├── TrainingActor x8 (one per GPU)
│       └── FailureActor x8 (one per GPU, for --with-failures)
│
SLURM Job (shared, 2 nodes)
├── Node 1 (ip-10-0-85-22): Monarch worker + 8 GPU processes (replica_0)
└── Node 2 (ip-10-0-91-197): Monarch worker + 8 GPU processes (replica_1)
```

### 3.2 Communication

- **Controller <-> Workers:** TCP via `ChannelTransport.TcpWithHostname`
- **Lighthouse:** gRPC on `[::]:0` (random port), address shared to all trainers
- **Trainer <-> Trainer (intra-replica):** NCCL over GPU interconnect
- **Trainer <-> Lighthouse:** HTTP for quorum requests and heartbeats

### 3.3 Fault Tolerance Flow

1. Lighthouse maintains quorum of all replicas (min_replicas=1)
2. Each training step: all replicas request quorum, get assigned same step
3. On failure: lighthouse detects missing heartbeats, forms new quorum without dead replica
4. Controller detects replica failure, tears down, checks SLURM state, recreates job if needed, respawns replica
5. New replica joins lighthouse with fresh UUID, resumes from current step

---

## 4. Successful Runs

### 4.1 Basic 2-Replica Training (No Failures)

**Command:**
```bash
python train_distributed.py \
  --tokenizer-path /home/dev/torchtitan/tests/assets/tokenizer \
  --dataset-path /home/dev/torchtitan/tests/assets/c4_test \
  --replica-count 2 --gpu-per-node 8 --host-per-replica 1 \
  --training-steps 50
```

**Result:** Both replicas completed all 50 steps with full quorum (2/2 participants) on both nodes. Clean shutdown with "Controller replica 0 done", "Controller replica 1 done", "Lighthouse stopped".

### 4.2 Fault Tolerance Testing

**Command:**
```bash
python train_distributed.py \
  --tokenizer-path /home/dev/torchtitan/tests/assets/tokenizer \
  --dataset-path /home/dev/torchtitan/tests/assets/c4_test \
  --replica-count 2 --gpu-per-node 8 --host-per-replica 1 \
  --training-steps 8000 --with-failures
```

**Failure types tested:**
| Failure Type | What It Does | Recovery Status |
|---|---|---|
| `KILL_SLURM` | `scancel` the SLURM job | Controller detects dead job, creates new allocation, respawns replicas |
| `KILL_PROC` | `os._exit(1)` on a random trainer | Controller catches error, detects SLURM FAILED after 5s delay, recreates job |
| `SEGFAULT` | Triggers SIGSEGV via ctypes | Same as KILL_PROC - crashes srun, SLURM marks job FAILED |
| `COMMS` | Aborts NCCL ProcessGroup | **Recovers successfully.** Gentlest failure — doesn't kill process or SLURM job. Surviving replica continues training alone with shrunk quorum (1/2). Failed replica respawns on existing allocation without job recreation. |
| `DEADLOCK` | Deadlocks GIL for 70s | Not yet tested |

**Key observations:**
- Process-level failures (KILL_PROC, SEGFAULT) cause SLURM to mark the entire job as FAILED because `srun` exits when any child process dies
- Each recovery cycle takes ~10-15 seconds (5s SLURM state delay + job submission + worker startup + trainer init)
- The 120-second interval between failures is sufficient for recovery
- Recovery successfully creates new SLURM allocations and respawns all replicas
- COMMS failure is the gentlest — the ProcessGroup is aborted but the process stays alive, so no SLURM job recreation is needed. The surviving replica continues training with a shrunk quorum while the failed replica respawns.
- Multiple back-to-back failure types (COMMS at step ~1050, then KILL_PROC at step ~2244) both recovered successfully in the same run

---

## 5. Key Files Modified

| File | Changes |
|---|---|
| `examples/monarch/train_distributed.py` | TCP transport, `_create_all_jobs()`, `_ensure_jobs_alive()`, `BaseException` handling, job creation lock, `_jobs_active()` check, 5s delay |
| `examples/monarch/utils/failure.py` | No permanent changes (startup_wait/rest_time remain at 120s) |
| `fix_squeue.py` | Patch script for Monarch's `slurm.py` to fix SLURM 21.x squeue --json bug (applied on gpu-dev only) |

---

## 6. Known Limitations and Next Steps

### Current Limitations
1. **Process crashes kill entire SLURM job:** SLURM's `srun` propagates any child process failure to all tasks, meaning a single trainer crash kills all workers on both nodes. This forces a full job recreation on every process-level failure.
2. **Recovery takes ~10-15s:** The combination of SLURM state propagation delay, job submission, and trainer initialization means there's a window where training is not progressing.
3. **PyO3 panics on cleanup:** When a Monarch worker process crashes, the parent process's Rust/PyO3 runtime logs numerous thread panic errors during cleanup. These are cosmetic but noisy.
4. **SLURM 21.x squeue patch:** The fix for `squeue --json` returning all jobs is applied as a monkey-patch on the gpu-dev machine, not upstreamed to Monarch.

### Next Steps
1. **Test DEADLOCK failure:** COMMS is confirmed working. DEADLOCK (GIL held for 70s) should trigger NCCL timeout — may behave differently.
2. **Upstream squeue fix:** Submit a PR to Monarch to fix `_get_job_info_json()` for SLURM 21.x compatibility.
3. **Explore per-task srun:** Using `srun --ntasks=1` per node instead of a single multi-node srun might isolate failures to a single node.
4. **Checkpoint-based recovery:** Currently replicas restart from scratch. Adding checkpoint save/load would allow resuming from the last saved step.
5. **Test with larger models:** Current testing uses `debugmodel`. Testing with actual LLaMA-3 configurations would validate real-world performance.
