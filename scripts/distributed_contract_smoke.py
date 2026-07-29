"""CPU multiprocess contract smoke; this is not a multi-GPU/NCCL test."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from queue import Empty
from typing import Any


def _worker(rank: int, world_size: int, queue: Any) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    queue.put(
        {
            "rank": int(os.environ["RANK"]),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "world_size": int(os.environ["WORLD_SIZE"]),
            "backend": "stdlib-multiprocessing",
            "gpu_collective_executed": False,
        }
    )


def run_contract_smoke(world_size: int) -> dict[str, Any]:
    if not 2 <= world_size <= 8:
        raise ValueError("CPU contract smoke world_size must be in [2, 8]")
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_worker, args=(rank, world_size, queue))
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    records = []
    for _ in processes:
        try:
            records.append(queue.get(timeout=15))
        except Empty as exc:
            raise RuntimeError("Timed out waiting for a contract worker") from exc
    for process in processes:
        process.join(timeout=15)
        if process.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {process.exitcode}")
    ranks = sorted(record["rank"] for record in records)
    if ranks != list(range(world_size)):
        raise RuntimeError(f"Rank contract failed: {ranks}")
    if any(record["world_size"] != world_size for record in records):
        raise RuntimeError("WORLD_SIZE was not propagated consistently")
    return {
        "status": "validated",
        "world_size": world_size,
        "ranks": ranks,
        "transport": "local CPU multiprocessing only",
        "multi_gpu_claim": False,
        "records": sorted(records, key=lambda item: item["rank"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run_contract_smoke(args.world_size), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
