from collections import defaultdict
import math
from tqdm import tqdm
import numpy as np
import random
import copy
import warnings
import time
import matplotlib.pyplot as plt
import csv

timestamps = []
metrics_over_time = []

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="A class named 'FitnessMin' has already been created*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="A class named 'Individual' has already been created*",
)


def compute_total_variance(partition, node_weights):
    component_weights = []
    for component in partition:
        if not component:
            return math.inf  # Invalid partition
        component_weights.append(sum(node_weights[node] for node in component))
    component_weights = np.array(component_weights)
    component_weights_mean = np.mean(component_weights)
    total_variance = np.sum((component_weights - component_weights_mean) ** 2)
    return total_variance


def compute_cross_edge_weight(partition, label_assignment, edges, matrix):
    # Map each node to its component label
    node_to_label = {}
    for idx, component in enumerate(partition):
        label = label_assignment[idx]
        for node in component:
            node_to_label[node] = label

    # Count edges between labeled components
    edge_counts = defaultdict(int)
    for u, v in edges:
        label_u = node_to_label[u]
        label_v = node_to_label[v]
        if label_u != label_v:
            edge_counts[(label_u, label_v)] += 1

    # Compute total cross edge weight
    total_cross_edge_weight = 0.0
    for (i, j), count in edge_counts.items():
        total_cross_edge_weight += count * matrix[i - 1][j - 1]
    return total_cross_edge_weight


def compute_metric(partition, label_assignment, node_weights, edges, matrix):
    total_variance = compute_total_variance(partition, node_weights)
    if total_variance == math.inf:
        return math.inf
    total_cross_edge_weight = compute_cross_edge_weight(
        partition, label_assignment, edges, matrix
    )
    return total_variance + total_cross_edge_weight


def initialize_partition_randomly(nodes, k):
    partition = [set() for _ in range(k)]
    nodes_copy = list(nodes)
    for component in partition:
        random_node = random.choice(nodes_copy)
        component.add(random_node)
        nodes_copy.remove(random_node)
    for node in nodes_copy:
        chosen_component = random.randint(0, k - 1)
        partition[chosen_component].add(node)
    return partition


def assign_labels_greedy(partition, edges, matrix):
    k = len(partition)
    # Initialize labels as None
    label_assignment = [0] * k

    # Compute the connection profile for each component
    connection_profiles = [defaultdict(int) for _ in range(k)]
    for u, v in edges:
        # Find the components of u and v
        comp_u = None
        comp_v = None
        for idx, component in enumerate(partition):
            if u in component:
                comp_u = idx
            if v in component:
                comp_v = idx
            if comp_u is not None and comp_v is not None:
                break
        if comp_u is not None and comp_v is not None and comp_u != comp_v:
            connection_profiles[comp_u][comp_v] += 1

    # Order components by the number of connections (descending)
    component_order = sorted(
        range(k), key=lambda x: sum(connection_profiles[x].values()), reverse=True
    )

    # Assign labels iteratively
    for comp_idx in component_order:
        best_label = None
        best_cost = math.inf
        for label in range(1, k + 1):
            if label in label_assignment:
                continue  # Each label must be unique
            # Tentatively assign the label
            temp_assignment = label_assignment.copy()
            temp_assignment[comp_idx] = label
            # Compute the incremental cross edge weight
            incremental_cost = 0.0
            for neighbor_idx, edge_count in connection_profiles[comp_idx].items():
                neighbor_label = temp_assignment[neighbor_idx]
                if neighbor_label > 0:
                    incremental_cost += (
                            edge_count * matrix[label - 1][neighbor_label - 1]
                    )
            if incremental_cost < best_cost:
                best_cost = incremental_cost
                best_label = label
        if best_label > 0:
            label_assignment[comp_idx] = best_label
        else:
            # If all labels are used, assign the next available label (this shouldn't happen if k labels are used)
            label_assignment[comp_idx] = k
    return tuple(label_assignment)


def perform_kl_heuristic(
        partition,
        label_assignment,
        node_weights,
        edges,
        matrix,
        max_iterations=1000,
        start_time=0.0,
):
    """
    Perform a Kernighan-Lin-like heuristic to improve the partition.
    """
    k = len(partition)
    best_partition = copy.deepcopy(partition)
    best_label_assignment = label_assignment
    best_metric = compute_metric(
        best_partition, best_label_assignment, node_weights, edges, matrix
    )

    for _ in range(max_iterations):
        improved = False
        # Iterate through all possible pairs of components
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                # Iterate through all nodes in component i
                for node in list(partition[i]):
                    # Attempt to move node from i to j
                    partition[i].remove(node)
                    partition[j].add(node)
                    # Assign labels (could use a smarter label assignment strategy)
                    new_label_assignment = assign_labels_greedy(
                        partition, edges, matrix
                    )
                    # Compute the new metric
                    new_metric = compute_metric(
                        partition, new_label_assignment, node_weights, edges, matrix
                    )
                    if new_metric < best_metric:
                        best_metric = new_metric
                        best_partition = copy.deepcopy(partition)
                        best_label_assignment = new_label_assignment
                        improved = True
                        current_time = time.time() - start_time
                        timestamps.append(current_time)
                        if len(metrics_over_time) > 0:
                            metrics_over_time.append(
                                min(best_metric, metrics_over_time[-1])
                            )
                        else:
                            metrics_over_time.append(best_metric)
                        break  # Exit after first improvement
                    else:
                        # Revert the move
                        partition[j].remove(node)
                        partition[i].add(node)
            if improved:
                break  # Proceed to next iteration after an improvement
        if not improved:
            break  # No improvement found; terminate early

    return best_partition, best_label_assignment, best_metric


def find_optimal_partition_and_labeling_heuristic(
        nodes, node_weights, edges, k, matrix, max_iterations=1000, num_restarts=10
):
    best_metric = math.inf
    best_partition = None
    best_label_assignment = None
    total_partitions_evaluated = 0

    # Record the start time
    start_time = time.time()

    for _ in tqdm(range(num_restarts), desc="Heuristic Restarts"):
        # Initialize partition
        partition = initialize_partition_randomly(nodes, k)

        label_assignment = assign_labels_greedy(partition, edges, matrix)
        # current_metric = compute_metric(partition, label_assignment, node_weights, edges, matrix)

        # Perform KL heuristic
        improved_partition, improved_label_assignment, improved_metric = (
            perform_kl_heuristic(
                partition,
                label_assignment,
                node_weights,
                edges,
                matrix,
                max_iterations,
                start_time,
            )
        )

        total_partitions_evaluated += 1

        # Update the best solution found
        if improved_metric < best_metric:
            best_metric = improved_metric
            best_partition = improved_partition
            best_label_assignment = improved_label_assignment

    print(
        f"Total initial partitions evaluated (restarts): {total_partitions_evaluated}"
    )
    return best_partition, best_label_assignment, best_metric


def main():
    # Example DAG
    # Nodes are labeled with unique identifiers (e.g., integers)
    n = 201
    nodes = [i for i in range(n)]
    node_weights = {i: 2.0 for i in range(n)}
    # Edges are represented as tuples (source, target)
    edges = []
    for i in range(math.floor(n / 2)):
        edges.append((i, i + 1))
    for i in range(math.floor(n / 2) + 1, n - 1):
        edges.append((i, i + 1))
    edges.append((n - 1, math.floor(n / 2)))
    # Define the k x k matrix
    k = 96
    # matrix = np.load("gpu_speed.npy")
    # matrix = np.max(matrix) - matrix
    # matrix = matrix.tolist()

    k_values = [2, 3, 4, 4]
    matrix = np.zeros((96, 96))
    for i in range(96):
        for j in range(96):
            tmp_i = i
            tmp_j = j
            for k_val in k_values:
                if math.floor(tmp_i / k_val) != math.floor(tmp_j / k_val):
                    matrix[i][j] += 0.5
                tmp_i %= k_val
                tmp_j %= k_val

    # Find the optimal partition and labeling using heuristic
    best_partition, best_label_assignment, best_metric = (
        find_optimal_partition_and_labeling_heuristic(
            nodes, node_weights, edges, k, matrix, max_iterations=10, num_restarts=20
        )
    )
    # Display the results
    print("\nBest Partition and Labeling:")
    for idx, component in enumerate(best_partition):
        label = best_label_assignment[idx]
        sorted_component = sorted(component)
        print(f"Component {label}: {sorted_component}")
    print(f"\nTotal Metric: {best_metric}")

    # Plot the metric over time
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, metrics_over_time, marker="o", linestyle="-")
    plt.title("Best Metric Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Best Metric")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally, save the plot
    plt.savefig("metric_over_time_heuristic.png")

    with open("metric_over_time_heuristic.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time (seconds)", "Best Metric"])
        writer.writerows(zip(timestamps, metrics_over_time))


if __name__ == "__main__":
    main()
