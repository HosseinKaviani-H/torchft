import asyncio
import textwrap

from kubernetes.client import (
    V1Container,
    V1EmptyDirVolumeSource,
    V1EnvVar,
    V1PodSpec,
    V1ResourceRequirements,
    V1Volume,
    V1VolumeMount,
)
from monarch.actor import this_host
from monarch.job.kubernetes import KubernetesJob

_BOOTSTRAP = textwrap.dedent("""\
    import os, socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    address = f"tcp://{socket.getfqdn()}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


async def main():
    # Step 1: local spawn — same as what the train script does for lighthouse
    print("Spawning local proc mesh (like lighthouse)...")
    local_pm = this_host().spawn_procs({"gpus": 1})
    print("Local proc mesh spawned")

    # Step 2: remote spawn — same as test_spawn.py
    pod_spec = V1PodSpec(
        containers=[
            V1Container(
                name="worker",
                image="ocir.ap-sydney-1.oci.oraclecloud.com/iduyx1qnmway/meta-pytorch/monarch:hossein-ft-v1",
                command=["python", "-u", "-c", _BOOTSTRAP],
                env=[V1EnvVar(name="MONARCH_PORT", value="26600")],
                resources=V1ResourceRequirements(
                    limits={"nvidia.com/gpu": "1"},
                    requests={"nvidia.com/gpu": "1"},
                ),
                volume_mounts=[V1VolumeMount(name="dshm", mount_path="/dev/shm")],
            )
        ],
        volumes=[
            V1Volume(
                name="dshm",
                empty_dir=V1EmptyDirVolumeSource(medium="Memory", size_limit="16Gi"),
            )
        ],
    )

    job = KubernetesJob(namespace="monarch-tests")
    job.add_mesh("testmesh2", num_replicas=1, pod_spec=pod_spec)

    print("Getting state...")
    state = job.state()
    mesh = state.testmesh2

    print("Spawning remote procs...")
    proc_mesh = mesh.spawn_procs({"gpus": 1})

    print("SUCCESS! Remote proc mesh spawned after local spawn")
    await proc_mesh.stop()
    await local_pm.stop()
    job.kill()
    print("Done")


asyncio.run(main())
