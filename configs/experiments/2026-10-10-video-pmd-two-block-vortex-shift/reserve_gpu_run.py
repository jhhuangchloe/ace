"""Run a module under torchrun after reserving GPU memory for this rank.

Usage (as a torchrun script):  torchrun ... reserve_gpu_run.py MODULE [ARGS...]
Env: RESERVE_GPU_GB = GB to reserve on this rank's GPU (0/unset = no reservation).

PyTorch's caching allocator keeps freed memory, so allocating a big block and deleting it
leaves the memory reserved *by this process* while the model is still small -- other jobs
on the host cannot take it during the start-up ramp. Later allocations are carved out of
that cached block. torch.cuda.empty_cache() would hand it back, so it is disabled here.
The reservation is only a cache: actual use may grow past it (extra memory is requested
from the driver as usual) and CUDA-context / NCCL memory is allocated outside of it.
"""
import os
import runpy
import sys


def main() -> None:
    reserve_gb = float(os.environ.get("RESERVE_GPU_GB", "0") or 0)
    if reserve_gb > 0:
        import torch

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        try:
            block = torch.empty(int(reserve_gb * 2**30), dtype=torch.uint8, device=device)
        except torch.OutOfMemoryError:
            free, total = torch.cuda.mem_get_info(device)
            sys.exit(
                f"[rank {rank}] cannot reserve {reserve_gb:.0f} GB on GPU: only "
                f"{free / 2**30:.1f} of {total / 2**30:.0f} GB free (another job is using it)."
            )
        del block  # stays in this process's allocator cache
        torch.cuda.empty_cache = lambda: None  # keep the reservation
        torch.cuda.memory.empty_cache = torch.cuda.empty_cache
        print(
            f"[rank {rank}] reserved {torch.cuda.memory_reserved(device) / 2**30:.1f} GB on {device}",
            flush=True,
        )
    module, *args = sys.argv[1:]
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":  # guard: DataLoader worker processes re-import this file
    main()
