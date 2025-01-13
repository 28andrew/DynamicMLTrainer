import csv
import itertools
import time
from collections import defaultdict

import math
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import Amazon
from tqdm import tqdm

from graph import Graph, create_graph, print_graph
from model import CustomGraphLinearModel2


def generate_k_partitions(nodes, k):
    if k == 1:
        yield [set(nodes)]
        return
    if k > len(nodes):
        return

    first = nodes[0]
    for partition in generate_k_partitions(nodes[1:], k - 1):
        yield [{first}] + partition
    for partition in generate_k_partitions(nodes[1:], k):
        for i in range(len(partition)):
            new_partition = [s.copy() for s in partition]
            new_partition[i].add(first)
            yield new_partition


def generate_label_assignments(k):
    return itertools.permutations(range(1, k + 1))

def compute_metric(
    partition,
    label_assignment,
    node_weights,
    edges,
    matrix,
    best_metric_so_far=math.inf,
):
    component_labels = {idx: label for idx, label in enumerate(label_assignment)}

    # Compute total variance
    component_weights = []
    for idx, component in enumerate(partition):
        if not component:
            return math.inf 
        component_weights.append(sum(node_weights[node] for node in component))
    component_weights = np.array(component_weights)
    component_weights_mean = np.mean(component_weights)
    total_variance = np.sum((component_weights - component_weights_mean) ** 2)
    
    node_to_label = {}
    for idx, component in enumerate(partition):
        label = component_labels[idx]
        for node in component:
            node_to_label[node] = label

    # Count edges between labeled components
    edge_counts = defaultdict(int)
    for u, v in edges:
        label_u = node_to_label[u]
        label_v = node_to_label[v]
        if label_u != label_v:
            edge_counts[(label_u, label_v)] += 1
            partial_cross_weight = (
                edge_counts[(label_u, label_v)] * matrix[label_u - 1][label_v - 1]
            )
            if total_variance + partial_cross_weight > best_metric_so_far:
                return math.inf

    total_cross_edge_weight = 0.0
    for (i, j), count in edge_counts.items():
        total_cross_edge_weight += count * matrix[i - 1][j - 1]
        if total_variance + total_cross_edge_weight > best_metric_so_far:
            return math.inf
    
    return total_variance + total_cross_edge_weight


def find_optimal_partition_and_labeling(
    nodes, node_weights, edges, k, matrix, total_time=None
):
    best_metric = math.inf
    best_partition = None
    best_label_assignment = None
    total_partitions = 0
    total_labelings_evaluated = 0

    timestamps = []
    metrics_over_time = []

    start_time = time.time()

    # Generate all k-partitions
    for partition in tqdm(generate_k_partitions(nodes, k)):
        total_partitions += 1
        # Generate all possible label assignments for this partition
        for label_assignment in generate_label_assignments(k):
            total_labelings_evaluated += 1
            metric = compute_metric(
                partition, label_assignment, node_weights, edges, matrix, best_metric
            )
            if metric < best_metric:
                best_metric = metric
                best_partition = partition
                best_label_assignment = label_assignment
                current_time = time.time() - start_time
                timestamps.append(current_time)
                metrics_over_time.append(best_metric)

            if total_time is not None and (time.time() - start_time > total_time):
                print(f"Total partitions evaluated: {total_partitions}")
                print(f"Total label assignments evaluated: {total_labelings_evaluated}")
                return (
                    best_partition,
                    best_label_assignment,
                    best_metric,
                    timestamps,
                    metrics_over_time,
                )

    print(f"Total partitions evaluated: {total_partitions}")
    print(f"Total label assignments evaluated: {total_labelings_evaluated}")
    return (
        best_partition,
        best_label_assignment,
        best_metric,
        timestamps,
        metrics_over_time,
    )


def partition_graph(graph: Graph, gpus: int, gpu_speeds: np.ndarray) -> list[list[str]]:
    nodes = []
    node_weights = {}
    edges = []
    for _, node in graph.nodes.items():
        nodes.append(node.module_fqdn)
        node_weights[node.module_fqdn] = node.weight
        for other in node.out_neighbors:
            edges.append((node.module_fqdn, other.module_fqdn))
    matrix = 1 - gpu_speeds
    matrix = matrix.tolist()

    best_partition, best_label_assignment, best_metric, _, _ = (
        find_optimal_partition_and_labeling(
            nodes, node_weights, edges, gpus, matrix, total_time=None
        )
    )

    print(best_metric)

    ans = []
    for partition in best_partition:
        ans.append(list(partition))

    sorted_ans = [None] * len(ans)
    for i, partition in enumerate(ans):
        new_idx = best_label_assignment[i] - 1
        sorted_ans[new_idx] = partition

    return sorted_ans


def main():
    # Example computation graph with 201 nodes
    n = 201
    nodes = [i for i in range(n)]
    # All nodes are equally weighted in this case
    node_weights = {i: 2.0 for i in range(n)}
    edges = []
    for i in range(math.floor(n / 2)):
        edges.append((i, i + 1))
    for i in range(math.floor(n / 2) + 1, n - 1):
        edges.append((i, i + 1))
    edges.append((n - 1, math.floor(n / 2)))

    k = 3
    matrix = np.load("gpu_speed.npy")
    matrix = np.max(matrix) - matrix
    matrix = matrix.tolist()

    (
        best_partition,
        best_label_assignment,
        best_metric,
        timestamps,
        metrics_over_time,
    ) = find_optimal_partition_and_labeling(
        nodes, node_weights, edges, k, matrix, total_time=120
    )

    # Display the results
    print("\nBest Partition and Labeling:")
    for idx, component in enumerate(best_partition):
        label = best_label_assignment[idx]
        sorted_component = sorted(component)
        print(f"Component {label}: {sorted_component}")
    print(f"\nTotal Metric: {best_metric}")


    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, metrics_over_time, marker="o", linestyle="-")
    plt.title("Best Metric Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Metric")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.savefig("metric_over_time_bash.png")

    with open("metric_over_time_bash.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time (seconds)", "Best Metric"])
        writer.writerows(zip(timestamps, metrics_over_time))


if __name__ == "__main__":
    main()
    # dataset = Amazon(root="data/Amazon", name="Computers")
    # model = CustomGraphLinearModel2(dataset.num_features, 4096, dataset.num_classes)
    # # Computation graph
    # graph = create_graph(model)
    # print_graph(graph)
    # gpu_speeds = np.load("gpu_speed.npy")
    # print(partition_graph(graph, 3, gpu_speeds))
