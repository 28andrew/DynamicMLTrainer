import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import time
import sys
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed GPU Benchmarking Script")
    parser.add_argument(
        "--master_addr", type=str, required=True, help="Master node address"
    )
    parser.add_argument(
        "--master_port", type=str, required=True, help="Master node port"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        required=True,
        help="Total number of processes across all nodes",
    )
    parser.add_argument(
        "--starting_rank", type=int, required=True, help="Starting rank for this node"
    )
    return parser.parse_args()


def setup_process(
    rank, world_size, master_addr, master_port, local_rank
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )

    dist.barrier()

    tensor_size = 1024 * 1024  # 1MB
    send_tensor = torch.ones(tensor_size, device=device)
    recv_tensor = torch.zeros(tensor_size, device=device)

    # WARM UP
    warmup_iterations = 10
    for target_rank in range(world_size):
        if target_rank == rank:
            continue

        for _ in range(warmup_iterations):
            if rank < target_rank:
                torch.cuda.synchronize()
                dist.send(tensor=send_tensor, dst=target_rank)
                dist.recv(tensor=recv_tensor, src=target_rank)
                torch.cuda.synchronize()
                dist.recv(tensor=recv_tensor, src=target_rank)
                dist.send(tensor=send_tensor, dst=target_rank)
            else:
                torch.cuda.synchronize()
                dist.recv(tensor=recv_tensor, src=target_rank)
                dist.send(tensor=send_tensor, dst=target_rank)
                torch.cuda.synchronize()
                dist.send(tensor=send_tensor, dst=target_rank)
                dist.recv(tensor=recv_tensor, src=target_rank)

    dist.barrier()
    # --------------------------------------------------------

    bandwidths = {}

    ITERS = 1000

    for target_rank in range(world_size):
        if target_rank == rank:
            continue

        if rank < target_rank:
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(ITERS):
                dist.send(tensor=send_tensor, dst=target_rank)
                dist.recv(tensor=recv_tensor, src=target_rank)
                torch.cuda.synchronize()
                dist.recv(tensor=recv_tensor, src=target_rank)
                dist.send(tensor=send_tensor, dst=target_rank)
                torch.cuda.synchronize()
            end_time = time.time()
        else:
            # Receive first, then send
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(ITERS):
                dist.recv(tensor=recv_tensor, src=target_rank)
                dist.send(tensor=send_tensor, dst=target_rank)
                torch.cuda.synchronize()
                dist.send(tensor=send_tensor, dst=target_rank)
                dist.recv(tensor=recv_tensor, src=target_rank)
                torch.cuda.synchronize()
            end_time = time.time()

        # Calculate bandwidth in MB/s
        elapsed_time = end_time - start_time
        # Each send and receive involves tensor_size * 4 bytes (float32) * (2 tensors/iter * ITERS)
        bandwidth = (
            (tensor_size * 4 * 2) / (elapsed_time * 1024 * 1024) * 2 * ITERS
        )  # MB/s
        bandwidths[target_rank] = bandwidth

        print(f"Rank {rank} <-> Rank {target_rank}: {bandwidth:.2f} MB/s")

    gather_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)
    for target_rank, bw in bandwidths.items():
        gather_tensor[target_rank] = bw

    if rank == 0:
        gather_list = [
            torch.zeros(world_size, dtype=torch.float32, device=device)
            for _ in range(world_size)
        ]
    else:
        gather_list = None

    dist.gather(tensor=gather_tensor, gather_list=gather_list, dst=0)

    if rank == 0:
        full_bandwidth = []
        for proc in range(world_size):
            # Copy data from gather_list
            proc_bandwidth = gather_list[proc].cpu().numpy()
            full_bandwidth.append(proc_bandwidth)

        bandwidth_matrix = np.array(full_bandwidth)

        np.fill_diagonal(bandwidth_matrix, 0)

        min_bw = bandwidth_matrix.min()
        max_bw = bandwidth_matrix.max()
        if max_bw > min_bw:
            normalized_matrix = (bandwidth_matrix - min_bw) / (max_bw - min_bw)
        else:
            normalized_matrix = (
                bandwidth_matrix
            )

        print("\nNormalized Bandwidth Matrix (0 to 1):")
        print(normalized_matrix)

        np.save("gpu_speed.npy", normalized_matrix)

    dist.barrier()
    dist.destroy_process_group()


def worker(local_rank, args, results_queue):
    global_rank = args.starting_rank + local_rank
    setup_process(
        global_rank,
        args.world_size,
        args.master_addr,
        args.master_port,
        local_rank,
        results_queue,
    )


def main():
    args = parse_args()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found on this node.")
        sys.exit(1)

    if args.starting_rank + num_gpus > args.world_size:
        print(
            f"Error: starting_rank ({args.starting_rank}) + num_gpus ({num_gpus}) exceeds world_size ({args.world_size})."
        )
        sys.exit(1)

    # Spawn one process per GPU
    mp.spawn(worker, args=(args), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()
