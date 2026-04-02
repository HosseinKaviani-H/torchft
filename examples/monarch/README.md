### Monarch-TorchFT-TorchTitan Distributed Training Orchestrator

#### Overview
This directory contains scripts for orchestrating fault-tolerant distributed training using
TorchTitan and Monarch. Two scheduler backends are supported:

- **`train_distributed.py`** — SLURM-based (bare-metal / HPC clusters)
- **`train_distributed_k8s.py`** — Kubernetes-based (containerized environments)

Both scripts manage multiple training replicas with automatic failure recovery and
TorchFT lighthouse coordination.

##### PREREQUISITES

**Common:**
- TorchTitan training configuration file in script directory (debug_model.toml)
- A training dataset (c4_test) and tokenizer in script directory

**SLURM (`train_distributed.py`):**
- Access to a SLURM cluster with GPU nodes
- Munge authentication configured across nodes

**Kubernetes (`train_distributed_k8s.py`):**
- Access to a Kubernetes cluster with GPU nodes
- Monarch operator installed (for provisioning mode), or pre-existing pods (for attach mode)
- A container image with Monarch, TorchFT, and TorchTitan installed

##### USAGE

**SLURM:**

    python train_distributed.py --help

    # Basic usage with 2 replicas, each with 1 node and 8 GPUs:
    python train_distributed.py

    # Custom configuration:
    python train_distributed.py --replica-count 3 --gpu-per-node 8 \
        --host-per-replica 2 --training-steps 100

    # With remote TorchFT lighthouse:
    python train_distributed.py --remote-lighthouse

    # With failure injection testing:
    python train_distributed.py --training-steps 8000 --with-failures

**Kubernetes:**

    python train_distributed_k8s.py --help

    # Provisioning mode (Monarch operator creates pods):
    python train_distributed_k8s.py --namespace my-ns \
        --image ghcr.io/org/monarch-torchft:latest \
        --replica-count 2 --gpus-per-host 8

    # Attach mode (connect to pre-existing pods):
    python train_distributed_k8s.py --namespace my-ns \
        --replica-count 2 --gpus-per-host 8

    # With timeout for pod readiness:
    python train_distributed_k8s.py --namespace my-ns \
        --image ghcr.io/org/monarch-torchft:latest \
        --timeout 300

##### KEY COMPONENTS
- **LighthouseActor**: TorchFT coordination server for quorum-based fault tolerance
- **TrainingActor**: Individual trainer processes (one per GPU)
- **ReplicaActor**: Manages a group of trainers on a single allocation
- **OrchestrationManager**: Top-level orchestration and failure recovery
- **MonarchSlurm / MonarchKubernetes**: Scheduler abstraction for job lifecycle management
- **FailureController**: Optional, periodically injects random failures for testing

##### FAILURE RECOVERY
- Automatic retry with configurable delays (PROC_ATTEMPT_DELAY)
- New allocations after repeated failures (every PROC_ATTEMPTS failures)
- Maximum attempts per replica before giving up (MAX_ATTEMPT)
- BaseException handling for Monarch CancelledError during cleanup
- Concurrency lock to prevent duplicate job creation on simultaneous failures

**SLURM-specific:** Process crashes kill the entire SLURM job (srun propagates failures),
so all replicas must be recreated. A 5-second delay is used to let SLURM propagate job
state before health checking.

**Kubernetes-specific:** Pod crashes are isolated — only the failed replica's job is
recreated while healthy replicas continue training uninterrupted.

##### OUTPUT
- Training outputs saved to ./outputs directory
- Logs streamed from all distributed processes
- TensorBoard metrics enabled by default

##### CLEANUP
All jobs (SLURM or Kubernetes) are automatically terminated at script exit via atexit handlers.
The `__getstate__`/`__setstate__` pickle protection ensures that subprocesses do not
accidentally kill jobs when they exit.
