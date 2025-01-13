import argparse
import functools
import os
import random
import time
from collections import defaultdict, deque
from itertools import cycle

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from torch_geometric.datasets import Amazon
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import RandomNodeSplit
from torchinfo import summary

from algo_bash import partition_graph
from graph import create_graph, print_graph
from model import CustomGraphLinearModel2

# CONSTANTS
SEED = 42
BATCH_SIZE = 8192
HIDDEN_CHANNELS = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Model Training Script")
    parser.add_argument(
        "--master_addr", type=str, help="Master node address", default="localhost"
    )
    parser.add_argument(
        "--master_port", type=str, help="Master node port", default=4000
    )
    parser.add_argument(
        "--world_size",
        type=int,
        help="Total number of processes across all nodes",
        default=3,
    )
    parser.add_argument(
        "--starting_rank", type=int, help="Starting rank for this node", default=0
    )
    parser.add_argument(
        "--num_microbatches", type=int, help="Number of microbatches", default=1
    )
    return parser.parse_args()


def print_rank0_def(global_rank, msg):
    if global_rank == 0:
        print(msg)


dataset = None
TRAIN_LOADER = None
VAL_LOADER = None
TEST_LOADER = None


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_loaders_if_needed():
    global TRAIN_LOADER, VAL_LOADER, TEST_LOADER

    if TRAIN_LOADER:
        return

    transform = RandomNodeSplit(num_val=0.1, num_test=0.1)
    dataset.transform = transform

    data = dataset[0]

    TRAIN_LOADER = cycle(
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

    VAL_LOADER = iter(
        NeighborLoader(
            data,
            input_nodes=data.val_mask,
            num_neighbors=[15, 15],
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    )

    TEST_LOADER = iter(
        NeighborLoader(
            data,
            input_nodes=data.test_mask,
            num_neighbors=[15, 15],
            shuffle=False,
            batch_size=BATCH_SIZE,
        )
    )


def next_data():
    global TRAIN_LOADER

    init_loaders_if_needed()
    batch = next(TRAIN_LOADER)
    return batch.x, batch.edge_index


def next_y():
    global TRAIN_LOADER

    init_loaders_if_needed()
    batch = next(TRAIN_LOADER)
    return batch.y


def setup_process(global_rank, local_rank, args):
    global dataset

    print(f"Process for global rank {global_rank}")

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    # Set the device for this process
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(
        backend="nccl", rank=global_rank, world_size=args.world_size, device_id=device
    )
    dist.barrier()

    dataset = Amazon(root="data/Amazon", name="Computers")
    model = CustomGraphLinearModel2(
        dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes
    )
    summary(model)
    # model = model.to(device)

    print_rank0 = functools.partial(print_rank0_def, global_rank)

    # Computation graph
    graph = create_graph(model, False)
    # print_graph(graph)

    # Partition the model
    modules_by_gpu = partition_graph(graph, args.world_size, np.load("gpu_speed.npy"))
    print_rank0(f"Modules by GPU: {modules_by_gpu}")
    module_to_gpu = {}
    for gpu, modules in enumerate(modules_by_gpu):
        for module in modules:
            module_to_gpu[module] = gpu

    # submodules_to_split_before = set()
    submodules_to_split_after = set()
    for node in graph.nodes.values():
        node_gpu = module_to_gpu[node.module_fqdn]
        #
        # for in_node in node.in_neighbors:
        #     in_node_gpu = module_to_gpu[in_node.module_fqdn]
        #     if node_gpu != in_node_gpu:
        #         submodules_to_split_before.add(node.module_fqdn)

        for out_node in node.out_neighbors:
            out_node_gpu = module_to_gpu[out_node.module_fqdn]
            if node_gpu != out_node_gpu:
                submodules_to_split_after.add(node.module_fqdn)

    # print_rank0(f'Split before points: {submodules_to_split_before}')
    print_rank0(f"Split after points: {submodules_to_split_after}")

    split_spec = {}
    # for before in submodules_to_split_before:
    #     split_spec[before] = SplitPoint.BEGINNING
    for after in submodules_to_split_after:
        split_spec[after] = SplitPoint.END

    num_nodes = BATCH_SIZE // args.num_microbatches
    x_fake = torch.randint(
        0,
        2,
        (
            num_nodes,
            dataset.num_features,
        ),
        dtype=torch.float,
    )

    pipe = pipeline(
        module=model,
        mb_args=(),
        mb_kwargs={
            "x": x_fake,
        },
        split_spec=split_spec,
    )
    print_rank0(pipe)
    torch.cuda.set_device(local_rank)

    # Make map from GPU => (stage_idx, stage_Mod)
    gpu_to_stage_idxs_and_mods = defaultdict(list)
    for stage_idx in range(pipe.num_stages):
        stage_mod = pipe.get_stage_module(stage_idx)
        # stage_mod.print_readable()
        try:
            first_child_name = next(iter(stage_mod.named_children()))[0]
            gpu = module_to_gpu[first_child_name]
        except StopIteration:
            gpu = 0

        gpu_to_stage_idxs_and_mods[gpu].append((stage_idx, stage_mod))

    # Swap submod_0 to be on GPU 0, not sure why this is necessary
    def get_module_name(parent_module, inner_object):
        for name, module in parent_module.named_children():
            if module == inner_object:
                return name
        return None

    for gpu, pairs in list(gpu_to_stage_idxs_and_mods.items())[:]:
        for stage_idx, mod in pairs:
            if get_module_name(pipe.split_gm, mod) == "submod_0":
                gpu_to_stage_idxs_and_mods[0], gpu_to_stage_idxs_and_mods[gpu] = (
                    gpu_to_stage_idxs_and_mods[gpu],
                    gpu_to_stage_idxs_and_mods[0],
                )
                break

    print_rank0(f"\nGPU to stage mapping:")
    for gpu, idxs_and_mods in gpu_to_stage_idxs_and_mods.items():
        print_rank0(f"  GPU {gpu}: {[idx for idx, _ in idxs_and_mods]}")

    # Build stages
    stage_idx_and_mods = gpu_to_stage_idxs_and_mods[global_rank]
    stage_idx_and_stages = []
    for stage_idx, stage_mod in stage_idx_and_mods:
        # stage = build_stage(stage_mod, stage_idx, pipe.info(), device)
        stage = pipe.build_stage(stage_idx, device)
        stage_idx_and_stages.append((stage_idx, stage))

    # Turn them into schedule pipes
    stage_idx_and_schedule_pipes = []
    optimizers = {}
    for stage_idx, stage in stage_idx_and_stages:
        schedule_pipe = ScheduleGPipe(
            stage,
            n_microbatches=args.num_microbatches,
            loss_fn=F.nll_loss,
            # kwargs_chunk_spec={
            #     'x': TensorChunkSpec(0),
            # 'edge_index': _Replicate
            # }
        )
        stage_idx_and_schedule_pipes.append((stage_idx, schedule_pipe))

        optimizer = torch.optim.Adam(stage.submod.parameters(), lr=1e-4)
        optimizers[stage_idx] = optimizer

    # Train loop
    iteration = 0
    rolling_time = deque(maxlen=100)
    while True:
        # Reset gradients before processing micro-batches
        for optimizer in optimizers.values():
            optimizer.zero_grad()

        start_time = time.time()

        for microbatch_idx in range(args.num_microbatches):
            for stage_idx, schedule_pipe in stage_idx_and_schedule_pipes:
                stage = schedule_pipe._stage

                if stage.is_first:
                    x, edge_index = next_data()
                    x = x[:BATCH_SIZE].to(torch.float).to(device)
                    schedule_pipe.step(x=x)
                elif stage.is_last:
                    y = next_y().to(device)[:BATCH_SIZE]
                    losses = []
                    schedule_pipe.step(target=y, losses=losses)
                    if microbatch_idx == args.num_microbatches - 1:
                        duration = (time.time() - start_time)
                        rolling_time.append(duration)
                        print(
                            f"Iteration {iteration}: Loss={losses[0]:.3f}. Avg time={np.mean(rolling_time):.6f}s"
                        )
                else:
                    schedule_pipe.step()

        for stage_idx, optimizer in optimizers.items():
            optimizer.step()

        iteration += 1

    dist.destroy_process_group()


def worker(local_rank, args):
    global_rank = args.starting_rank + local_rank
    if global_rank == 0:
        main_rank0(args)

    setup_process(global_rank, local_rank, args)


def main_rank0(args):
    pass


def main():
    set_seed(SEED)

    args = parse_args()
    num_gpus = min(torch.cuda.device_count(), args.world_size)

    if args.world_size == 1:
        print("Direct")
        worker(0, args)
    else:
        mp.spawn(worker, args=(args,), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()
