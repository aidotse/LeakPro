"""Helpers for distributed training."""

import io
import os
import socket
from typing import Iterable

import blobfile as bf
import torch as th
import torch.distributed as dist

try:
    from mpi4py import MPI
except Exception:  # pragma: no cover - optional dependency
    MPI = None

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 1

SETUP_RETRY_COUNT = 3


def setup_dist() -> None:
    """Setup a distributed process group."""
    if dist.is_initialized():
        return
    if MPI is None:
        return

    comm = MPI.COMM_WORLD
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    hostname = "localhost" if backend == "gloo" else socket.gethostbyname(socket.getfqdn().split(".")[0])
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev() -> th.device:
    """Get the device to use for torch.distributed."""
    if th.cuda.is_available():
        if MPI is not None:
            return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
        if dist.is_initialized():
            return th.device(f"cuda:{th.cuda.current_device()}")
        return th.device("cuda:0")
    return th.device("cpu")


def load_state_dict(path: str, **kwargs: object) -> object:
    """Load a PyTorch file without redundant fetches across MPI ranks."""
    if MPI is None:
        return th.load(path, **kwargs)
    chunk_size = 2 ** 30  # MPI has a relatively small size limit
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = len(data) // chunk_size
        if len(data) % chunk_size:
            num_chunks += 1
        MPI.COMM_WORLD.bcast(num_chunks)
        for i in range(0, len(data), chunk_size):
            MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
    else:
        num_chunks = MPI.COMM_WORLD.bcast(None)
        data = bytes()
        for _ in range(num_chunks):
            data += MPI.COMM_WORLD.bcast(None)

    return th.load(io.BytesIO(data), **kwargs)


def get_rank() -> int:
    """Get the current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    if MPI is not None:
        return MPI.COMM_WORLD.Get_rank()
    return 0


def get_world_size() -> int:
    """Get the world size for distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    if MPI is not None:
        return MPI.COMM_WORLD.Get_size()
    return 1


def barrier() -> None:
    """Synchronize processes if distributed is initialized."""
    if dist.is_initialized():
        dist.barrier()


def sync_params(params: Iterable[th.Tensor]) -> None:
    """Synchronize a sequence of Tensors across ranks from rank 0."""
    if not dist.is_initialized():
        return
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port() -> int:
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        if s is not None:
            s.close()
