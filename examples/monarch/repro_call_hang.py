"""
Repro: After killing one process in a multi-process mesh, do the surviving
processes (stuck in C-level blocking calls) get cleaned up?

This verifies:
1. Are workers in SEPARATE processes? (check PIDs)
2. Does call() raise when one process dies?
3. Does the surviving process (stuck in libc.sleep) get killed by proc_mesh cleanup?
4. Or does it stay alive as a zombie spamming errors?

GitHub issue: https://github.com/meta-pytorch/monarch/issues/3435
"""
import asyncio
import os

from monarch.actor import Actor, current_rank, endpoint, this_host
from monarch.config import configure

configure(enable_log_forwarding=True)


class WorkerActor(Actor):
    @endpoint(instrument=False)
    async def do_work(self) -> None:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        rank = current_rank().rank
        pid = os.getpid()
        print(f"[Worker rank={rank} pid={pid}] Starting blocking C-level work (300s)")
        libc.sleep(300)
        print(f"[Worker rank={rank} pid={pid}] Finished (should not reach here)")


class KillerActor(Actor):
    @endpoint(instrument=False)
    async def kill_if_rank_0(self) -> None:
        rank = current_rank().rank
        pid = os.getpid()
        if rank == 0:
            print(f"[Killer rank={rank} pid={pid}] I am rank 0 — killing myself")
            os._exit(1)
        else:
            print(f"[Killer rank={rank} pid={pid}] I am rank {rank} — staying alive")


class SupervisorActor(Actor):
    def __init__(self):
        self._proc_mesh = None

    async def __supervise__(self, failure) -> bool:
        print(f"[Supervisor] __supervise__ fired: {failure}")
        return True

    @endpoint(instrument=False)
    async def run(self) -> None:
        host_mesh = this_host()
        self._proc_mesh = host_mesh.spawn_procs({"gpus": 2})

        async with self._proc_mesh:
            workers = self._proc_mesh.spawn("workers", WorkerActor)
            killers = self._proc_mesh.spawn("killers", KillerActor)

            async def delayed_kill():
                await asyncio.sleep(5)
                print("[Supervisor] Telling all ranks to kill if rank 0...")
                try:
                    await killers.kill_if_rank_0.call()
                except Exception as e:
                    print(f"[Supervisor] Kill call raised: {e}")

            asyncio.get_running_loop().create_task(delayed_kill())

            print("[Supervisor] Calling workers.do_work.call() on 2 processes")
            print("[Supervisor] Watch the PIDs — are they different?")
            print("[Supervisor] After kill: does call() raise? Does rank 1 get killed?")
            try:
                await workers.do_work.call()
                print("[Supervisor] ERROR: call() returned normally")
            except Exception as e:
                print(f"[Supervisor] SUCCESS: call() raised: {e}")

        print("[Supervisor] async with block exited — proc_mesh cleaned up")
        print("[Supervisor] If rank 1 is still spamming, proc_mesh.stop() didn't SIGKILL it")


async def main():
    print("=== Multi-process repro ===")
    print("Spawns 2 SEPARATE processes (check PIDs).")
    print("Kills rank 0. Rank 1 is stuck in libc.sleep(300).")
    print("Question: does proc_mesh cleanup kill rank 1?\n")

    sup_mesh = this_host().spawn_procs({"gpus": 1})
    sup = sup_mesh.spawn("supervisor", SupervisorActor)

    try:
        await sup.run.call_one()
    except Exception as e:
        print(f"[Main] run() raised: {e}")

    print("=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
