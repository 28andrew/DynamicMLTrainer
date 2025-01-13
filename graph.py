from dataclasses import dataclass, field
from queue import Queue
from typing import List, Iterable

import torch.nn as nn
from torch.fx.node import Node as FxNode
from torch_geometric.nn.fx import symbolic_trace


def mark_as_pipeline_stage(obj):
    setattr(obj, "__stage__", True)
    return obj


def mark_as_ghost_stage(obj):
    setattr(obj, "__ghost__", True)
    return obj


@dataclass
class Node:
    module_fqdn: str
    staged: bool = False
    weight: int = 0
    in_neighbors: List["Node"] = field(default_factory=list)
    out_neighbors: List["Node"] = field(default_factory=list)

    def get_display_name(self):
        return f"{self.module_fqdn} ({self.weight})"


@dataclass
class Graph:
    nodes: dict[str, Node] = field(default_factory=dict)

    def add_edge(self, u: str, v: str):
        self.nodes[u].out_neighbors.append(self.nodes[v])
        self.nodes[v].in_neighbors.append(self.nodes[u])

    def bfs_shortest_path(self, start: str, end: str) -> tuple[int, List[str]]:
        """
        BFS to compute shortest path between two nodes.
        Returns the distance and the path (list of node names).
        """
        queue = Queue()
        visited = {node: False for node in self.nodes}
        parent = {node: None for node in self.nodes}
        distance = {node: float("inf") for node in self.nodes}

        queue.put(start)
        visited[start] = True
        distance[start] = 0

        while not queue.empty():
            current = queue.get()

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return distance[end], path[::-1]

            for neighbor in self.nodes[current].out_neighbors:
                neighbor_name = neighbor.module_fqdn
                if not visited[neighbor_name]:
                    visited[neighbor_name] = True
                    parent[neighbor_name] = current
                    distance[neighbor_name] = distance[current] + 1
                    queue.put(neighbor_name)
        return float("inf"), []

    def find_mst_between_whitelist(self, whitelist: set[str]) -> "Graph":
        edges = []
        for u in whitelist:
            for v in whitelist:
                if u != v:
                    weight, _ = self.bfs_shortest_path(u, v)
                    edges.append((weight, u, v))

        # Kruskal's
        parent = {node: node for node in whitelist}
        rank = {node: 0 for node in whitelist}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                if rank[root_x] > rank[root_y]:
                    parent[root_y] = root_x
                elif rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                else:
                    parent[root_y] = root_x
                    rank[root_x] += 1

        edges.sort()
        mst_edges = []
        for weight, u, v in edges:
            if find(u) != find(v):
                union(u, v)
                mst_edges.append((u, v, weight))

        mst_graph = Graph()
        for u, v, _ in mst_edges:
            if u not in mst_graph.nodes:
                mst_graph.nodes[u] = Node(u)
            if v not in mst_graph.nodes:
                mst_graph.nodes[v] = Node(v)
            mst_graph.add_edge(u, v)
        return mst_graph


def get_parameter_count(module: nn.Module) -> int:
    total_params = 0

    def recursive_count(m, depth=0):
        nonlocal total_params

        if depth > 0 and (getattr(m, "__stage__", False) or hasattr(m, "__ghost__")):
            return

        total_params += sum(p.numel() for p in m.parameters(recurse=False))

        for child in m.children():
            recursive_count(child, depth + 1)

    recursive_count(module)
    return total_params


def create_graph(model: nn.Module, use_weights=True):
    graph = Graph()

    mod = symbolic_trace(model)
    fx_graph = mod.graph
    for fx_node in fx_graph.nodes:
        name = fx_node.name
        graph.nodes[name] = Node(name)

    for fx_node in fx_graph.nodes:
        node = graph.nodes[fx_node.name]

        out_fx_node_names = []
        for arg in fx_node.args:
            if isinstance(arg, FxNode):
                out_fx_node_names.append(arg.name)
            elif isinstance(arg, Iterable):
                for item in arg:
                    if isinstance(item, FxNode):
                        out_fx_node_names.append(item.name)

        for out_fx_node_name in out_fx_node_names:
            in_node = graph.nodes[out_fx_node_name]

            in_node.out_neighbors.append(node)
            node.in_neighbors.append(in_node)

    # ==== Post-processing

    # Rename output to ""
    output_node = graph.nodes["output"]
    output_node.module_fqdn = ""
    graph.nodes[""] = output_node
    del graph.nodes["output"]

    # Remove nodes that are not staged FQDN submodules, make into DAG
    fqdn_set = set()
    name_to_module = {}
    for fqdn, module in model.named_modules():
        name_to_module[fqdn] = module

        if fqdn == "" or getattr(module, "__stage__", False):
            fqdn_set.add(fqdn)

    graph = graph.find_mst_between_whitelist(fqdn_set)

    # Assign weights
    for node in graph.nodes.values():
        node.weight = get_parameter_count(name_to_module[node.module_fqdn]) if use_weights else 0

    return graph


# Not marked as stage
class C(nn.Module):
    def forward(self, x):
        return x * 3


@mark_as_pipeline_stage
class B(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = C()

    def forward(self, x):
        return self.c(x) + 1


@mark_as_pipeline_stage
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = B()

    def forward(self, x):
        return self.b(x) + 1


@mark_as_pipeline_stage
class RootModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = A()
        self.other = B()

    def forward(self, x):
        return self.a(x) + self.other(x)


def print_graph(graph: Graph):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()

    for node in graph.nodes.values():
        G.add_node(node.module_fqdn, label=node.get_display_name().replace("_", "\n_"))

    for node in graph.nodes.values():
        for out_neighbor in node.out_neighbors:
            G.add_edge(node.module_fqdn, out_neighbor.module_fqdn)

    # Define positions for two straight chains
    pos = nx.S
    chain1 = [node.module_fqdn for node in graph.nodes.values() if "chain1" in node.module_fqdn]
    chain2 = [node.module_fqdn for node in graph.nodes.values() if "chain2" in node.module_fqdn]

    # for i, node in enumerate(sorted(chain1)):
    #     pos[node] = (1.5 * i, 0.75)
    #
    # for i, node in enumerate(sorted(chain2)):
    #     pos[node] = (1.5 * i, 1.25)
    #     last_i = i
    #
    # for node in graph.nodes:
    #     if node not in pos:
    #         pos[node] = (last_i := last_i + 1.5, 1)

    num_nodes = len(G.nodes)
    node_size = max(1000, 3000 - num_nodes * 5)
    font_size = max(6, 12 - num_nodes // 20)

    node_colors = []
    for node in G.nodes:
        if G.in_degree(node) == 0:
            node_colors.append("lightgreen")  # Input
        elif G.out_degree(node) == 0:
            node_colors.append("salmon")  # Output
        else:
            node_colors.append("lightblue")  # Intermediate

    plt.figure(figsize=(9, 6))
    plt.tight_layout()

    nx.draw_networkx_nodes(
        G, pos, node_size=node_size, node_color=node_colors, edgecolors="black"
    )
    nx.draw_networkx_labels(
        G,
        pos,
        labels=nx.get_node_attributes(G, "label"),
        font_size=font_size,
        font_weight="bold",
    )
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="gray")

    plt.axis("off")

    plt.savefig("graph_linear.png", format="PNG", dpi=300)
    plt.close()

if __name__ == "__main__":
    from model import CustomGraphSAGEModel
    mod = CustomGraphSAGEModel(768, 4096, 20)
    graph = create_graph(mod)
    print_graph(graph)
