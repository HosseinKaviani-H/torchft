"""
Minimal repro: call() does not raise when a child process is killed
while the surviving process is blocked in an uninterruptible C-level call.

With time.sleep(1) in a Python loop: call() raises correctly.
With libc.sleep(300) (simulating NCCL allreduce): call() hangs forever.

This simulates what happens in training: NCCL allreduce is a blocking C++
call that prevents the process from receiving Monarch's stop signal.

GitHub issue: https://github.com/meta-pytorch/monarch/issues/3435
"""
import asyncio
import ctypes
import os

from monarch.actor import Actor, endpoint, this_host
from monarch.config import configure

configure(enable_log_forwarding=True)

libc = ctypes.CDLL("libc.so.6")


class WorkerActor(Actor):
    @endpoint(instrument=False)
    async def do_work(self) -> None:
        print(f"[Worker {os.getpid()}] Starting blocking C-level work (simulates NCCL allreduce)")
        libc.sleep(300)


class KillerActor(Actor):
    @endpoint(instrument=False)
    async def kill_self(self) -> None:
        print(f"[Killer {os.getpid()}] Killing process")
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
        self._proc_mesh = host_mesh.spawn_procs({"gpus": 1})

        async with self._proc_mesh:
            workers = self._proc_mesh.spawn("workers", WorkerActor)
            killers = self._proc_mesh.spawn("killers", KillerActor)

            async def delayed_kill():
                await asyncio.sleep(5)
                print("[Supervisor] Sending kill command...")
                try:
                    await killers.kill_self.choose()
                except Exception as e:
                    print(f"[Supervisor] Kill command raised (expected): {e}")

            asyncio.get_running_loop().create_task(delayed_kill())

            print("[Supervisor] Calling workers.do_work.call() — should raise after kill")
            try:
                await workers.do_work.call()
                print("[Supervisor] ERROR: call() returned normally — should not happen")
            except Exception as e:
                print(f"[Supervisor] call() raised (expected): {e}")


async def main():
    print("Starting repro...")
    sup_mesh = this_host().spawn_procs({"gpus": 1})
    sup = sup_mesh.spawn("supervisor", SupervisorActor)

    try:
        await sup.run.call_one()
    except Exception as e:
        print(f"[Main] run() raised: {e}")

    print("Done")


if __name__ == "__main__":
    asyncio.run(main())
