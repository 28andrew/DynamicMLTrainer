import argparse
import os
import random
import time
from collections import deque
from itertools import cycle

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch_geometric.datasets import Amazon
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torchinfo import summary

from model import CustomGraphLinearModel2

# CONSTANTS
SEED = 42
BATCH_SIZE = 8192
HIDDEN_CHANNELS = 4096


def parse_args():
    parser = argparse.ArgumentParser(
        description="FSDP-based Distributed Training Script"
    )
    parser.add_argument(
        "--master_addr", type=str, default="localhost", help="Master node address"
    )
    parser.add_argument(
        "--master_port", type=str, default="4000", help="Master node port"
    )
    parser.add_argument(
        "--world_size", type=int, default=3, help="Number of processes across nodes"
    )
    parser.add_argument(
        "--starting_rank", type=int, default=0, help="Starting rank for this node"
    )
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
    dataset = Amazon(root="data/Amazon", name="Computers", transform=transform)
    data = dataset[0]

    train_loader = cycle(
        iter(
            NeighborLoader(
                data,
                input_nodes=data.train_mask,
                num_neighbors=[15, 15],
                shuffle=True,
                batch_size=BATCH_SIZE,
            )
        )
    )
    val_loader = iter(
        NeighborLoader(
            data,
            input_nodes=data.val_mask,
            num_neighbors=[15, 15],
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    )
    return train_loader, val_loader, dataset


def fsdp_worker(rank, args):
    set_seed(SEED)

    # Set up distributed environment
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Data and Model
    train_loader, _, dataset = load_data()
    model = CustomGraphLinearModel2(
        dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes
    ).to(device)
    model = FSDP(wrap(model), device_id=device)
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    iteration = 0
    rolling_time = deque(maxlen=100)
    while True:
        start_time = time.time()
        batch = next(train_loader)
        x, y = batch.x, batch.y
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            duration = (time.time() - start_time)
            rolling_time.append(duration)
            print(
                f"Iteration {iteration}: Loss={loss.item():.3f}. Time={np.mean(rolling_time):.6f}s"
            )
        iteration += 1


def main():
    args = parse_args()
    num_gpus = min(torch.cuda.device_count(), args.world_size)
    set_seed(SEED)

    if args.world_size == 1:
        fsdp_worker(0, args)
    else:
        mp.spawn(fsdp_worker, args=(args,), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()
