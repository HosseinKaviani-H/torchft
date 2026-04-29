"""
Minimal repro: call() does not raise when one process in a multi-process
mesh is killed while the surviving processes are blocked in C-level calls.

Setup: 2 processes, each with a WorkerActor (blocks in libc.sleep) and a
KillerActor. Kill process 0 → process 1 survives but is stuck in C sleep.
call() should raise SupervisionError but hangs because process 1 is alive
and blocked.

This simulates NCCL allreduce: 8 GPUs in a collective, 1 dies, 7 are stuck
waiting in C++ code and can't respond to Monarch's stop signal.

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
        print(f"[Worker rank={rank} pid={pid}] Starting blocking C-level work")
        libc.sleep(300)


class KillerActor(Actor):
    @endpoint(instrument=False)
    async def kill_self(self) -> None:
        rank = current_rank().rank
        pid = os.getpid()
        print(f"[Killer rank={rank} pid={pid}] Killing process")
        os._exit(1)


class SupervisorActor(Actor):
    def __init__(self):
        self._proc_mesh = None

    async def __supervise__(self, failure) -> bool:
        print(f"[Supervisor] __supervise__ fired: {failure}")
        if self._proc_mesh is not None:
            print("[Supervisor] Stopping proc_mesh...")
            pm = self._proc_mesh
            self._proc_mesh = None
            await pm.stop()
            print("[Supervisor] proc_mesh stopped")
        return True

    @endpoint(instrument=False)
    async def run(self) -> None:
        host_mesh = this_host()
        # Spawn 2 processes (2 ranks) — kill rank 0, rank 1 survives but blocks
        self._proc_mesh = host_mesh.spawn_procs({"gpus": 2})

        async with self._proc_mesh:
            workers = self._proc_mesh.spawn("workers", WorkerActor)
            killers = self._proc_mesh.spawn("killers", KillerActor)

            async def delayed_kill():
                await asyncio.sleep(5)
                print("[Supervisor] Killing rank 0 only...")
                try:
                    # choose() picks a random rank — with 2 ranks this kills one
                    await killers.kill_self.choose()
                except Exception as e:
                    print(f"[Supervisor] Kill command raised (expected): {e}")

            asyncio.get_running_loop().create_task(delayed_kill())

            print("[Supervisor] Calling workers.do_work.call() on 2 processes")
            print("[Supervisor] After killing rank 0, rank 1 will be stuck in libc.sleep()")
            print("[Supervisor] call() should raise but will likely hang...")
            try:
                await workers.do_work.call()
                print("[Supervisor] ERROR: call() returned normally — should not happen")
            except Exception as e:
                print(f"[Supervisor] call() raised (expected): {e}")


async def main():
    print("Starting multi-process repro...")
    print("This spawns 2 processes. Kills 1. The other blocks in C sleep.")
    print("Expected: call() raises. Actual: call() hangs.\n")

    sup_mesh = this_host().spawn_procs({"gpus": 1})
    sup = sup_mesh.spawn("supervisor", SupervisorActor)

    try:
        await sup.run.call_one()
    except Exception as e:
        print(f"[Main] run() raised: {e}")

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
