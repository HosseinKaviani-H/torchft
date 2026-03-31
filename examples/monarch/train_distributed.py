# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import asyncio
import atexit
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict

import torch
from monarch._rust_bindings.monarch_hyperactor.channel import ChannelTransport
from monarch._rust_bindings.monarch_hyperactor.config import configure
from monarch.actor import Actor, current_rank, endpoint, HostMesh, ProcMesh, this_host
from monarch.job import SlurmJob

# Use TCP transport globally so cross-node actors can communicate
configure(default_transport=ChannelTransport.TcpWithHostname)
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
from utils.failure import Failure, FailureActor, FailureController


# ==== Allocation boilerplate ====
class MonarchSlurm:
    job_name_prefix: str = "monarch-torchft"

    def __init__(self):
        self.job_handles: Dict[str, SlurmJob] = {}
        self._is_owner = True
        atexit.register(self.kill_jobs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_is_owner"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    async def get_or_create_job(
        self, mesh_name: str, nodes_per_mesh: int = 1, gpus_per_node: int = 8
    ) -> None:
        job = SlurmJob(
            meshes={mesh_name: nodes_per_mesh},
            gpus_per_node=gpus_per_node,
            job_name=f"{self.job_name_prefix}-{mesh_name}",
            python_exe="/opt/conda/bin/python3",
        )
        job.apply()
        self.job_handles[mesh_name] = job

    async def get_or_create_multi_job(
        self,
        mesh_names: list,
        nodes_per_mesh: int = 1,
        gpus_per_node: int = 8,
    ) -> None:
        meshes = {name: nodes_per_mesh for name in mesh_names}
        job = SlurmJob(
            meshes=meshes,
            gpus_per_node=gpus_per_node,
            job_name=f"{self.job_name_prefix}-all",
            python_exe="/opt/conda/bin/python3",
        )
        job.apply()
        for name in mesh_names:
            self.job_handles[name] = job

    def kill_jobs(self):
        if not self._is_owner:
            return
        killed = set()
        for mesh_name, job in self.job_handles.items():
            if id(job) not in killed:
                killed.add(id(job))
                self.kill_job(mesh_name)

    def kill_job(self, mesh_name: str):
        try:
            job = self.job_handles[mesh_name]
            logger.info(f"Destroying job for mesh {mesh_name}")
            job.kill()
        except Exception as e:
            logger.exception(f"Failed to destroy job for {mesh_name}: {e}")

    def proc_mesh(self, mesh_name: str, num_procs: int) -> ProcMesh:
        job = self.job_handles[mesh_name]
        mesh: HostMesh = getattr(job.state(cached_path=None), mesh_name)
        proc_mesh = mesh.spawn_procs({"gpus": num_procs})
        return proc_mesh


# ==== allocation boilerplate ====


class LighthouseActor(Actor):
    def __init__(self) -> None:
        self.lighthouse = None

    @endpoint
    def start_lighthouse(self) -> str:
        # inline import because of https://github.com/meta-pytorch/monarch/issues/804
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
    hosts_per_replica: int
    gpus_per_node: int
    with_failures: bool
    lighthouse_address: str = ""


@dataclass
class Replica:
    rid: int
    proc_mesh: ProcMesh
    actor: "ReplicaActor"
    attempt_number: int = 0


# This does not currently benefit from being an actor, but will once
# Monarch supervision APIs are fleshed out.
class ReplicaActor(Actor):
    def __init__(self, spec: JobSpec, replica_id: int, scheduler: MonarchSlurm) -> None:
        self.spec = deepcopy(spec)
        self.replica_id = replica_id

        self.uid = f"[replica_{replica_id}]"
        self.spec.trainer_config.fault_tolerance.replica_id = self.replica_id
        self.scheduler = scheduler

        self.failure_actors: FailureActor | None = None

    @endpoint
    async def start_replica(self) -> None:
        init_logger()
        logger.info(f"{self.uid} Spawning trainers")

        trainers_proc_mesh = self.scheduler.proc_mesh(
            f"replica_{self.replica_id}",
            num_procs=self.spec.gpus_per_node,
        )

        async with trainers_proc_mesh:
            await trainers_proc_mesh.logging_option(stream_to_client=True)
            await setup_torch_elastic_env_async(trainers_proc_mesh)

            training_actors = trainers_proc_mesh.spawn(
                "training_actors",
                TrainingActor,
                self.spec.trainer_config,
                self.replica_id,
            )

            self.failure_actors = trainers_proc_mesh.spawn(
                "failure_actors", FailureActor
            )

            logger.info(f"{self.uid} Starting trainers")
            await training_actors.start_training.call(self.spec.lighthouse_address)

    @endpoint
    async def inject_failure(self, failure_type: Failure):
        if self.failure_actors:
            try:
                logger.info(
                    f"{self.uid} Injecting failure ({failure_type}) into random trainer"
                )

                await self.failure_actors.fail.choose(failure_type)
            except Exception as e:
                logger.exception(f"{self.uid} Injected failure: {e}")
        else:
            error_msg = f"{self.uid} No failure actors available"
            logger.error(error_msg)


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
        self.lighthouse_actor: LighthouseActor | None = None
        self.lighthouse_mesh: ProcMesh | None = None

        self.scheduler = MonarchSlurm()

    async def _create_all_jobs(self, max_retries: int = 3) -> None:
        """Create SLURM jobs and wait for them to be RUNNING before returning."""
        mesh_names = [f"replica_{i}" for i in range(self.spec.replica_count)]

        for attempt in range(max_retries):
            try:
                if self.spec.replica_count > 1:
                    await self.scheduler.get_or_create_multi_job(
                        mesh_names, self.spec.hosts_per_replica, self.spec.gpus_per_node
                    )
                else:
                    await self.scheduler.get_or_create_job(
                        mesh_names[0], self.spec.hosts_per_replica, self.spec.gpus_per_node
                    )

                # Wait for SLURM job to be RUNNING and populate hostnames
                # before replicas are started.
                job = self.scheduler.job_handles[mesh_names[0]]
                total_nodes = sum(job._meshes.values())
                job._all_hostnames = job._wait_for_job_start(
                    job._slurm_job_id, total_nodes
                )
                logger.info(
                    f"[Controller] SLURM job is RUNNING on nodes: {job._all_hostnames}"
                )
                return
            except RuntimeError as e:
                logger.warning(
                    f"[Controller] Job creation attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                # Reset the scheduler's job handles so a fresh job can be created
                for name in mesh_names:
                    self.scheduler.job_handles.pop(name, None)
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                else:
                    raise

    async def start_training(self) -> None:
        logger.info(
            f"[Controller] Creating training system with {self.spec.replica_count} replicas"
        )

        await self._create_all_jobs()

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

    async def start_lighthouse(self) -> None:
        if self.spec.remote_lighthouse:
            await self.scheduler.get_or_create_job("lighthouse")
            self.lighthouse_mesh = self.scheduler.proc_mesh("lighthouse", num_procs=1)
        else:
            self.lighthouse_mesh = this_host().spawn_procs({"gpus": 1})

        await self.lighthouse_mesh.logging_option(stream_to_client=True)
        self.lighthouse_actor = self.lighthouse_mesh.spawn(
            "lighthouse_actor", LighthouseActor
        )
        self.spec.lighthouse_address = (
            await self.lighthouse_actor.start_lighthouse.call_one()
        )

    async def stop_lighthouse(self) -> None:
        try:
            if self.lighthouse_mesh:
                await self.lighthouse_actor.stop_lighthouse.call_one()
                await self.lighthouse_mesh.stop()
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
        except Exception as e:
            await self._teardown(replica_id)
            logger.exception(f"[Controller] replica {replica_id} failed: {e}")
            await self._run_replica(replica_id, attempt_number + 1)

    async def _spin_up_replica(self, replica_id: int, attempt_number: int = 0) -> None:
        if attempt_number != 0 and attempt_number % PROC_ATTEMPTS == 0:
            logger.info(
                f"[Controller] Replica {replica_id} has failed {attempt_number} times. Getting new allocation."
            )
            if self.spec.replica_count == 1:
                self.scheduler.kill_job(f"replica_{replica_id}")
                await self.scheduler.get_or_create_job(
                    f"replica_{replica_id}", self.spec.hosts_per_replica
                )
            else:
                # Recreate the shared multi-mesh job
                self.scheduler.kill_jobs()
                await self._create_all_jobs()
        delay = 0 if not attempt_number else PROC_ATTEMPT_DELAY
        logger.info(
            f"[Controller] Spinning up replica with ID {replica_id} in {delay} seconds"
        )
        await asyncio.sleep(delay)

        replica_proc_mesh = this_host().spawn_procs({"gpus": 1})
        await replica_proc_mesh.logging_option(aggregate_window_sec=None)

        replica_actor = replica_proc_mesh.spawn(
            "replica_actor", ReplicaActor, self.spec, replica_id, self.scheduler
        )

        replica = Replica(replica_id, replica_proc_mesh, replica_actor, attempt_number)
        self.replicas[replica_id] = replica
        await replica.actor.start_replica.call_one()

    async def _teardown(self, replica_id: int) -> None:
        try:
            replica = self.replicas[replica_id]
            try:
                await replica.proc_mesh.stop()
            except Exception as e:
                logger.exception(
                    f"[Controller] Failed to stop replica {replica_id}, it may already be stopped. {e}"
                )
            del self.replicas[replica_id]
            del replica.proc_mesh
        except Exception as e:
            logger.exception(
                f"[Controller] Failed to teardown replica {replica_id}: {e}"
            )


# === CLI / CONFIG === #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monarch-TorchFT Distributed Training Example"
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--replica-count", type=int, default=2, help="Number of replicas (default: 2)"
    )
    parser.add_argument(
        "--gpu-per-node", type=int, default=8, help="GPUs per replica (default: 8)"
    )
    parser.add_argument(
        "--host-per-replica", type=int, default=1, help="Hosts per replica (default: 1)"
    )
    parser.add_argument(
        "--remote-lighthouse",
        action="store_true",
        help="Run the LighthouseServer on a worker node (default: False)",
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
        default="debug_tokenizer",
        help=f"Relative path to tokenizer (default: {os.path.join(script_dir, 'debug_tokenizer')})",
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

    return parser.parse_args()


def make_job_spec(args: argparse.Namespace) -> JobSpec:
    data_parallel_shard_degree = args.gpu_per_node * args.host_per_replica

    script_dir = os.path.dirname(os.path.abspath(__file__))

    trainer_config = FaultTolerantTrainer.Config(
        hf_assets_path=os.path.join(script_dir, args.tokenizer_path),
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
        hosts_per_replica=args.host_per_replica,
        gpus_per_node=args.gpu_per_node,
        with_failures=args.with_failures,
    )


# === CLI / CONFIG === #


async def main() -> None:
    init_logger()

    args = parse_args()
    job_spec = make_job_spec(args)

    orchestrator = OrchestrationManager(job_spec)
    try:
        await orchestrator.start_lighthouse()
        await orchestrator.start_training()
    finally:
        await orchestrator.stop_lighthouse()


if __name__ == "__main__":
    asyncio.run(main())
