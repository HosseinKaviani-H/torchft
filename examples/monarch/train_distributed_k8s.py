# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import atexit
import os
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import torch
from monarch.config import configure

configure(
    enable_log_forwarding=True,
    message_delivery_timeout="2m",
    host_spawn_ready_timeout="2m",
)

from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1PodSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)
from monarch.actor import Actor, current_rank, endpoint, HostMesh, ProcMesh
from monarch.job.kubernetes import KubernetesJob
from monarch.spmd import setup_torch_elastic_env_async
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import (
    ActivationCheckpointConfig,
    CommConfig,
    TrainingConfig,
)
from torchtitan.experiments.ft.config.job_config import FaultTolerance
from torchtitan.experiments.ft.llama3 import model_registry
from torchtitan.experiments.ft.optimizer import FTOptimizersContainer
from torchtitan.experiments.ft.trainer import FaultTolerantTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import ProfilingConfig
try:
    from utils.failure import Failure, FailureActor, FailureController
except ModuleNotFoundError:
    FailureActor = None
    FailureController = None
    Failure = None


# ==== Allocation boilerplate ====

_WORKER_BOOTSTRAP_SCRIPT: str = textwrap.dedent("""\
    import os
    import socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    hostname = socket.getfqdn()
    address = f"tcp://{hostname}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


def build_gpu_pod_spec(image: str, gpus_per_host: int) -> V1PodSpec:
    """Build a V1PodSpec with GPU resources and shared memory for NCCL."""
    gpu_resources = {"nvidia.com/gpu": str(gpus_per_host)}
    return V1PodSpec(
        containers=[
            V1Container(
                name="worker",
                image=image,
                command=["python", "-u", "-c", _WORKER_BOOTSTRAP_SCRIPT],
                env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
                resources=V1ResourceRequirements(
                    limits=gpu_resources,
                    requests=gpu_resources,
                ),
                volume_mounts=[
                    V1VolumeMount(name="dshm", mount_path="/dev/shm"),
                ],
            )
        ],
        volumes=[
            V1Volume(
                name="dshm",
                empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"),
            )
        ],
    )


class MonarchKubernetes:
    """Manages KubernetesJob lifecycle for fault-tolerant training.

    Creates one KubernetesJob per replica so that individual replicas can be
    killed and recreated independently — unlike a shared job where one failure
    would require tearing down all replicas.
    """

    def __init__(
        self,
        namespace: str,
        image: str | None = None,
        gpus_per_host: int = 8,
        timeout: int | None = None,
    ):
        self.namespace = namespace
        self.image = image
        self.gpus_per_host = gpus_per_host
        self.timeout = timeout
        self.job_handles: Dict[str, KubernetesJob] = {}
        self._is_owner = True
        atexit.register(self.kill_jobs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_is_owner"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def get_or_create_job(self, mesh_name: str) -> None:
        job = KubernetesJob(namespace=self.namespace, timeout=self.timeout)
        if self.image is not None:
            pod_spec = build_gpu_pod_spec(self.image, self.gpus_per_host)
            job.add_mesh(mesh_name, num_replicas=1, pod_spec=pod_spec)
        else:
            job.add_mesh(mesh_name, num_replicas=1)
        self.job_handles[mesh_name] = job

    def kill_jobs(self):
        if not self._is_owner:
            return
        for mesh_name in list(self.job_handles.keys()):
            self.kill_job(mesh_name)

    def kill_job(self, mesh_name: str):
        try:
            job = self.job_handles.pop(mesh_name, None)
            if job is not None:
                logger.info(f"Destroying job for mesh {mesh_name}")
                job.kill()
        except Exception as e:
            logger.exception(f"Failed to destroy job for {mesh_name}: {e}")

    def proc_mesh(self, mesh_name: str, num_procs: int) -> ProcMesh:
        job = self.job_handles[mesh_name]
        mesh: HostMesh = getattr(job.state(), mesh_name)
        proc_mesh = mesh.spawn_procs({"gpus": num_procs})
        return proc_mesh


# ==== allocation boilerplate ====


class LighthouseActor(Actor):
    def __init__(self) -> None:
        self.lighthouse = None

    @endpoint
    def start_lighthouse(self) -> str:
        from torchft.coordination import LighthouseServer

        self.lighthouse = LighthouseServer(
            bind="[::]:0", min_replicas=1, join_timeout_ms=60000
        )
        return self.lighthouse.address()

    @endpoint
    def stop_lighthouse(self) -> None:
        if not self.lighthouse:
            raise RuntimeError("Lighthouse not started!")
        self.lighthouse.shutdown()


class TrainingActor(Actor):
    def __init__(self, trainer_config: FaultTolerantTrainer.Config, replica_id: int) -> None:
        self.trainer_config = trainer_config
        rank = current_rank().rank
        self.uid = f"[replica_{replica_id}_trainer_{rank}]"

    @endpoint
    async def start_training(self, lighthouse_address: str) -> None:
        init_logger()

        os.environ["TORCHFT_LIGHTHOUSE"] = lighthouse_address
        trainer = self.trainer_config.build()
        logger.info(f"{self.uid} initialized successfully on {os.getpid()}")

        try:
            logger.info(f"{self.uid} starting training")
            trainer.train()
        except Exception:
            if trainer:
                trainer.close()
            raise
        else:
            trainer.close()
        finally:
            torch.distributed.destroy_process_group()
            logger.info(f"{self.uid} trainer cleaned up")


@dataclass
class JobSpec:
    trainer_config: FaultTolerantTrainer.Config
    remote_lighthouse: bool
    replica_count: int
    gpus_per_host: int
    with_failures: bool
    namespace: str = ""
    image: str | None = None
    timeout: int | None = None
    lighthouse_address: str = ""


@dataclass
class Replica:
    rid: int
    proc_mesh: ProcMesh
    failure_actors: "FailureActor | None" = None
    attempt_number: int = 0


# delay before re-creating proc mesh on existing job. change as needed.
PROC_ATTEMPT_DELAY = 0
# proc attempts before getting a new scheduler allocation. change as needed.
PROC_ATTEMPTS = 4
# attempts before failing training on replica. change as needed.
MAX_ATTEMPT = PROC_ATTEMPTS * 4


class OrchestrationManager:
    def __init__(self, spec: JobSpec) -> None:
        self.spec = spec
        self.replicas: Dict[int, Replica] = {}
        self.lighthouse = None

        self.scheduler = MonarchKubernetes(
            namespace=spec.namespace,
            image=spec.image,
            gpus_per_host=spec.gpus_per_host,
            timeout=spec.timeout,
        )
        self._job_creation_lock = asyncio.Lock()

    async def start_training(self) -> None:
        logger.info(
            f"[Controller] Creating training system with {self.spec.replica_count} replicas"
        )

        for replica_id in range(self.spec.replica_count):
            await self.scheduler.get_or_create_job(f"replica{replica_id}")

        mesh_futures = {}
        for i in range(self.spec.replica_count):
            mesh_futures[i] = asyncio.create_task(self._run_replica(i, 0))

        failure_future = None
        if self.spec.with_failures:
            failure_future = asyncio.create_task(
                FailureController.execute_failures(self.replicas, self.scheduler)
            )

        await asyncio.gather(*mesh_futures.values(), return_exceptions=True)

        if failure_future:
            failure_future.cancel()

    def start_lighthouse(self) -> None:
        import socket as _socket

        from torchft.coordination import LighthouseServer

        self.lighthouse = LighthouseServer(
            bind="[::]:0", min_replicas=1, join_timeout_ms=60000
        )
        # Replace hostname with IP so worker pods can resolve it
        addr = self.lighthouse.address()
        hostname = _socket.gethostname()
        pod_ip = _socket.gethostbyname(hostname)
        self.spec.lighthouse_address = addr.replace(hostname, pod_ip)
        logger.info(
            f"[Controller] Lighthouse started at {self.spec.lighthouse_address}"
        )

    def stop_lighthouse(self) -> None:
        try:
            if self.lighthouse:
                self.lighthouse.shutdown()
            logger.info("[Controller] Lighthouse stopped")
        except Exception as e:
            logger.exception(f"[Controller] Failed to stop lighthouse: {e}")

    async def _run_replica(self, replica_id: int, attempt_number: int) -> None:
        if attempt_number >= MAX_ATTEMPT:
            logger.info(f"[Controller] Replica {replica_id} has failed too many times.")
            return

        try:
            await self._spin_up_replica(replica_id, attempt_number)
            logger.info(f"[Controller] replica {replica_id} done")
            await self._teardown(replica_id)
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            await self._teardown(replica_id)
            logger.exception(f"[Controller] replica {replica_id} failed: {e}")
            await self._run_replica(replica_id, attempt_number + 1)

    async def _ensure_job_alive(self, replica_id: int, attempt_number: int) -> None:
        """Ensure the K8s job for this replica is alive before respawning.
        Must be called under self._job_creation_lock.

        In K8s, each replica has its own independent job, so we only need to
        recreate the failed replica's job — not all jobs like in SLURM.
        """
        mesh_name = f"replica{replica_id}"

        if attempt_number % PROC_ATTEMPTS == 0:
            logger.info(
                f"[Controller] Replica {replica_id} has failed {attempt_number} times. Getting new allocation."
            )
            self.scheduler.kill_job(mesh_name)
            await self.scheduler.get_or_create_job(mesh_name)
        else:
            job = self.scheduler.job_handles.get(mesh_name)
            if job is None or not job.active:
                logger.info(
                    f"[Controller] K8s job for {mesh_name} is no longer active, recreating."
                )
                self.scheduler.kill_job(mesh_name)
                await self.scheduler.get_or_create_job(mesh_name)

    async def _spin_up_replica(self, replica_id: int, attempt_number: int = 0) -> None:
        if attempt_number != 0:
            async with self._job_creation_lock:
                await self._ensure_job_alive(replica_id, attempt_number)

        delay = 0 if not attempt_number else PROC_ATTEMPT_DELAY
        logger.info(
            f"[Controller] Spinning up replica with ID {replica_id} in {delay} seconds"
        )
        await asyncio.sleep(delay)

        # Spawn trainers proc mesh directly from the main process.
        # This avoids the Timeout(30s) that occurs when spawn_procs is called
        # from inside a subprocess (ReplicaActor).
        mesh_name = f"replica{replica_id}"
        trainers_proc_mesh = self.scheduler.proc_mesh(
            mesh_name, num_procs=self.spec.gpus_per_host
        )

        spec = deepcopy(self.spec)
        spec.trainer_config.fault_tolerance.replica_id = replica_id

        await trainers_proc_mesh.logging_option(stream_to_client=True)
        await setup_torch_elastic_env_async(trainers_proc_mesh)

        training_actors = trainers_proc_mesh.spawn(
            "training_actors", TrainingActor, spec.trainer_config, replica_id
        )
        failure_actors = None
        if FailureActor is not None and self.spec.with_failures:
            failure_actors = trainers_proc_mesh.spawn("failure_actors", FailureActor)

        replica = Replica(replica_id, trainers_proc_mesh, failure_actors, attempt_number)
        self.replicas[replica_id] = replica

        logger.info(f"[Controller] Replica {replica_id} starting training")
        await training_actors.start_training.call(spec.lighthouse_address)

    async def _teardown(self, replica_id: int) -> None:
        try:
            replica = self.replicas[replica_id]
            try:
                await replica.proc_mesh.stop()
            except BaseException as e:
                logger.exception(
                    f"[Controller] Failed to stop replica {replica_id}, it may already be stopped. {e}"
                )
            del self.replicas[replica_id]
            del replica.proc_mesh
        except BaseException as e:
            logger.exception(
                f"[Controller] Failed to teardown replica {replica_id}: {e}"
            )


# === CLI / CONFIG === #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monarch-TorchFT Kubernetes Distributed Training"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--replica-count", type=int, default=2, help="Number of replicas (default: 2)"
    )
    parser.add_argument(
        "--gpus-per-host", type=int, default=8, help="GPUs per pod (default: 8)"
    )
    parser.add_argument(
        "--remote-lighthouse",
        action="store_true",
        help="Run the LighthouseServer on a worker pod (default: False)",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=50,
        help="Number of training steps (default: 50)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/opt/torchtitan/tests/assets/tokenizer",
        help="Path to tokenizer directory (must exist on worker pods)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Absolute path to the dataset directory (default: None, downloads from HuggingFace)",
    )
    parser.add_argument(
        "--with-failures",
        action="store_true",
        help="Enable the failure injector utility (default: False)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Kubernetes namespace for job scheduling",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Container image for provisioning mode (e.g., ghcr.io/org/image:tag). "
        "If not set, uses attach-only mode (pods must already exist).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximum seconds to wait for pods to be ready (default: wait indefinitely)",
    )

    return parser.parse_args()


def make_job_spec(args: argparse.Namespace) -> JobSpec:
    data_parallel_shard_degree = args.gpus_per_host

    script_dir = os.path.dirname(os.path.abspath(__file__))

    trainer_config = FaultTolerantTrainer.Config(
        hf_assets_path=args.tokenizer_path,
        profiling=ProfilingConfig(),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model_spec=model_registry("debugmodel"),
        optimizer=FTOptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=args.training_steps,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset_path=args.dataset_path,
        ),
        checkpoint=CheckpointManager.Config(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        comm=CommConfig(train_timeout_seconds=300),
        fault_tolerance=FaultTolerance(
            enable=True,
            group_size=data_parallel_shard_degree,
            process_group="nccl",
            process_group_timeout_ms=60000,
        ),
    )

    return JobSpec(
        trainer_config=trainer_config,
        remote_lighthouse=args.remote_lighthouse,
        replica_count=args.replica_count,
        gpus_per_host=args.gpus_per_host,
        with_failures=args.with_failures,
        namespace=args.namespace,
        image=args.image,
        timeout=args.timeout,
    )


# === CLI / CONFIG === #


async def main() -> None:
    init_logger()

    args = parse_args()
    job_spec = make_job_spec(args)

    orchestrator = OrchestrationManager(job_spec)
    try:
        orchestrator.start_lighthouse()
        await orchestrator.start_training()
    finally:
        orchestrator.stop_lighthouse()


if __name__ == "__main__":
    asyncio.run(main())
