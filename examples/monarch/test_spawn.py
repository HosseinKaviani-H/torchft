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
from monarch.job.kubernetes import KubernetesJob

_BOOTSTRAP = textwrap.dedent("""\
    import os, socket
    from monarch.actor import run_worker_loop_forever
    port = os.environ.get("MONARCH_PORT", "26600")
    address = f"tcp://{socket.getfqdn()}:{port}"
    run_worker_loop_forever(address=address, ca="trust_all_connections")
""")


async def main():
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
    job.add_mesh("testmesh", num_replicas=1, pod_spec=pod_spec)

    print("Getting state...")
    state = job.state()
    mesh = state.testmesh

    print("Spawning procs...")
    proc_mesh = mesh.spawn_procs({"gpus": 1})

    print("SUCCESS! Proc mesh spawned")
    await proc_mesh.stop()
    job.kill()
    print("Done")


asyncio.run(main())
