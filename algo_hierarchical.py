from cProfile import label
from collections import defaultdict
import math
from tqdm import tqdm
import numpy as np
import random
import copy
import warnings
import time

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
            return math.inf  
        component_weights.append(sum(node_weights[node] for node in component))
    component_weights = np.array(component_weights)
    component_weights_mean = np.mean(component_weights)
    total_variance = np.sum((component_weights - component_weights_mean) ** 2)
    return total_variance


def compute_cross_edge_weight(partition, label_assignment, edges, matrix):
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
    label_assignment = [0] * k

    connection_profiles = [defaultdict(int) for _ in range(k)]
    for u, v in edges:
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

    component_order = sorted(
        range(k), key=lambda x: sum(connection_profiles[x].values()), reverse=True
    )

    # Assign labels greedily
    for comp_idx in component_order:
        best_label = None
        best_cost = math.inf
        for label in range(1, k + 1):
            if label in label_assignment:
                continue  

            temp_assignment = label_assignment.copy()
            temp_assignment[comp_idx] = label

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
            label_assignment[comp_idx] = k
    return tuple(label_assignment)


def perform_kl_heuristic(
    partition, label_assignment, node_weights, edges, matrix, max_iterations=1000
):
    k = len(partition)
    best_partition = copy.deepcopy(partition)
    best_label_assignment = label_assignment
    best_metric = compute_metric(
        best_partition, best_label_assignment, node_weights, edges, matrix
    )

    for _ in range(max_iterations):
        improved = False
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                for node in list(partition[i]):
                    # Attempt to move node from i to j
                    partition[i].remove(node)
                    partition[j].add(node)

                    new_label_assignment = assign_labels_greedy(
                        partition, edges, matrix
                    )

                    new_metric = compute_metric(
                        partition, new_label_assignment, node_weights, edges, matrix
                    )
                    if new_metric < best_metric:
                        best_metric = new_metric
                        best_partition = copy.deepcopy(partition)
                        best_label_assignment = new_label_assignment
                        improved = True
                        break 
                    else:
                        # Revert the move
                        partition[j].remove(node)
                        partition[i].add(node)
            if improved:
                break  
        if not improved:
            break 

    return best_partition, best_label_assignment, best_metric


def find_optimal_partition_and_labeling_heuristic(
    nodes, node_weights, edges, k, matrix, max_iterations=1000, num_restarts=10
):
    best_metric = math.inf
    best_partition = None
    best_label_assignment = None
    total_partitions_evaluated = 0

    for _ in tqdm(range(num_restarts), desc="Heuristic Restarts"):
        partition = initialize_partition_randomly(nodes, k)

        label_assignment = assign_labels_greedy(partition, edges, matrix)
        # current_metric = compute_metric(partition, label_assignment, node_weights, edges, matrix)

        improved_partition, improved_label_assignment, improved_metric = (
            perform_kl_heuristic(
                partition, label_assignment, node_weights, edges, matrix, max_iterations
            )
        )

        total_partitions_evaluated += 1

        if improved_metric < best_metric:
            best_metric = improved_metric
            best_partition = improved_partition
            best_label_assignment = improved_label_assignment

    print(
        f"Total initial partitions evaluated (restarts): {total_partitions_evaluated}"
    )
    return best_partition, best_label_assignment, best_metric


def find_partition_and_labeling_hierarchical(
    nodes, node_weights, edges, k_values, matrix, max_iterations, num_restarts
):

    if len(k_values) == 1:
        return find_optimal_partition_and_labeling_heuristic(
            nodes,
            node_weights,
            edges,
            k_values[0],
            matrix,
            max_iterations,
            num_restarts,
        )

    cur_k = k_values[0]
    matrix = np.array(matrix)
    n = int(len(matrix) / cur_k)
    cur_matrix = np.zeros((cur_k, cur_k))

    for i in range(cur_k):
        for j in range(cur_k):
            # Extract the n x n subgrid
            subgrid = matrix[i * n : (i + 1) * n, j * n : (j + 1) * n]
            cur_matrix[i, j] = np.mean(subgrid)

    cur_matrix = cur_matrix.tolist()

    best_partition, best_label_assignment, _ = (
        find_optimal_partition_and_labeling_heuristic(
            nodes, node_weights, edges, cur_k, cur_matrix, max_iterations, num_restarts
        )
    )
    partition_ans = []
    label_assignment_ans = []
    K = np.prod(np.array(k_values[1:]))
    for idx, partition in enumerate(best_partition):
        cur_nodes = list(partition)
        cur_node_weights = {k: v for k, v in node_weights.items() if k in cur_nodes}
        cur_edges = [(u, v) for (u, v) in edges if u in cur_nodes and v in cur_nodes]
        i = best_label_assignment[idx] - 1
        cur_partition, cur_label_assignment, _ = (
            find_partition_and_labeling_hierarchical(
                cur_nodes,
                cur_node_weights,
                cur_edges,
                k_values[1:],
                matrix[i * n : (i + 1) * n, i * n : (i + 1) * n],
                max_iterations,
                num_restarts,
            )
        )
        partition_ans += cur_partition
        for lbl in cur_label_assignment:
            label_assignment_ans.append(i * K + lbl)

    return partition_ans, label_assignment_ans, 0


def main():
    # Example DAG as in algo_bash.py
    start_time = time.time()
    n = 201
    nodes = [i for i in range(n)]
    node_weights = {i: 2.0 for i in range(n)}
    edges = []
    for i in range(math.floor(n / 2)):
        edges.append((i, i + 1))
    for i in range(math.floor(n / 2) + 1, n - 1):
        edges.append((i, i + 1))
    edges.append((n - 1, math.floor(n / 2)))

    # Define the hierarchy of communication costs
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

    best_partition, best_label_assignment, _ = find_partition_and_labeling_hierarchical(
        nodes, node_weights, edges, k_values, matrix, max_iterations=10, num_restarts=20
    )

    # Display the results
    print('=' * 100)
    print(f"Time taken: {time.time() - start_time}")
    print("\nBest Partition and Labeling:")
    for idx, component in enumerate(best_partition):
        label = best_label_assignment[idx]
        sorted_component = sorted(component)
        print(f"Component {label}: {sorted_component}")
    print(f"Metric: {compute_metric(best_partition, best_label_assignment, node_weights, edges, matrix)}")


if __name__ == "__main__":
    main()
