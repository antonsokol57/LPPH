from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import phat  # type: ignore
from persim import PersistenceImager  # type: ignore
from tqdm.contrib.concurrent import process_map  # type: ignore


class ListDigraph:
    def __init__(self, num_nodes: int, num_edges: int) -> None:
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.out_neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
        self.in_neighbors: list[set[int]] = [set() for _ in range(num_nodes)]

    def add_edge(self, u: int, v: int) -> None:
        self.out_neighbors[u].add(v)
        self.in_neighbors[v].add(u)

    def iter_out(self, u: int):
        return self.out_neighbors[u]

    def iter_in(self, v: int):
        return self.in_neighbors[v]

    def iter_edges(self):
        for u in range(self.num_nodes):
            for v in self.iter_out(u):
                yield u, v


class DictDigraph:
    def __init__(self) -> None:
        self.out_neighbors: defaultdict[int, set[int]] = defaultdict(set)
        self.in_neighbors: defaultdict[int, set[int]] = defaultdict(set)

    def add_node(self, node: int) -> None:
        self.out_neighbors[node] = set()
        self.in_neighbors[node] = set()

    def add_edge(self, u: int, v: int) -> None:
        self.out_neighbors[u].add(v)
        self.in_neighbors[v].add(u)

    def iter_out(self, u: int):
        return self.out_neighbors[u]

    def iter_in(self, v: int):
        return self.in_neighbors[v]

    @property
    def nodes(self):
        return self.out_neighbors.keys()


# def compute_boundary_matrix(
#     paths: list[tuple[int, ...]],
# ):
#     vert_paths: list[tuple[int, ...]] = []
#     vert_path_rows: list[np.ndarray[tuple[int, int], np.dtype[np.bool]]] = []
#     boundary_matrix = np.zeros((len(paths), len(paths)), dtype=np.bool)

#     path_indexes: dict[tuple[int, ...], int] = {}
#     for path_idx, path in enumerate(paths):
#         if len(path) > 1:
#             for path_term in itertools.combinations(path, len(path) - 1):
#                 path_term_idx = path_indexes.get(path_term)
#                 if path_term_idx is None:
#                     path_term_idx = len(paths) + len(vert_paths)
#                     path_indexes[path_term] = path_term_idx
#                     path_term_row = np.zeros((1, len(paths)), dtype=np.bool)
#                     path_term_row[0, path_idx] = True
#                     vert_path_rows.append(path_term_row)
#                     vert_paths.append(path_term)
#                 elif path_term_idx >= len(paths):
#                     vert_path_rows[path_term_idx - len(paths)][0, path_idx] = True
#                 else:
#                     boundary_matrix[path_term_idx, path_idx] = True
#         path_indexes[path] = path_idx
#     vert_path_rows = [boundary_matrix] + vert_path_rows
#     boundary_matrix = np.vstack(vert_path_rows, dtype=np.bool)
#     return boundary_matrix


def compute_boundary_matrix_phat(paths: list[tuple[int, ...]]):
    columns: list[tuple[int, list[int]]] = []
    path_indexes: dict[tuple[int, ...], int] = {}
    forbidden_path_cnt = 0
    for path_idx, path in enumerate(paths):
        column: list[int] = []
        if len(path) > 1:
            for path_term in itertools.combinations(path, len(path) - 1):
                if len(set(path_term)) == 1:
                    continue
                path_term_idx = path_indexes.get(path_term)
                if path_term_idx is None:
                    path_term_idx = len(paths) + forbidden_path_cnt
                    path_indexes[path_term] = path_term_idx
                    forbidden_path_cnt += 1
                column.append(path_term_idx)
        columns.append((len(path) - 1, column))
        path_indexes[path] = path_idx
    return columns + [(0, [])] * forbidden_path_cnt


# def compute_PPH(paths: list[tuple[int, ...]], times: list[int], dim: int = 4):
#     D = compute_boundary_matrix(paths)
#     row_indices = np.arange(D.shape[0], dtype=np.int32)[:, None] * D
#     lows = row_indices.max(axis=0, where=D, initial=-1)
#     for column_num in range(len(paths)):
#         while lows[column_num] != -1 and lows[column_num] in lows[:column_num]:
#             column_1 = D[:, column_num]
#             number_of_prev = np.argmax(lows[:column_num] == lows[column_num])
#             column_2 = D[:, number_of_prev]
#             mask = np.logical_xor(column_1, column_2)
#             D[:, column_num] = mask
#             (nonzero,) = np.nonzero(mask)
#             if nonzero.size > 0:
#                 lows[column_num] = nonzero[-1]
#             else:
#                 lows[column_num] = -1

#     PPH_ordered: list[list[tuple[int, int]]] = [[] for _ in range(dim + 1)]
#     for i in range(len(paths)):
#         if lows[i] == -1:
#             birth = times[i]
#             row = i
#             death = np.where(lows == row)[0]
#             if len(death) == 0:
#                 death = times[-1] + 1
#             else:
#                 death = times[int(death)]
#             PPH_ordered[len(paths[i]) - 1].append((birth, death))
#     return PPH_ordered


def compute_pph_phat(
    paths: list[tuple[int, ...]],
    columns: list[tuple[int, list[int]]],
    times: list[int],
    max_dim: int,
):
    bm = phat.boundary_matrix(columns=columns)
    b: int
    d: int
    pph_ordered: list[list[tuple[int, int]]] = [[] for _ in range(max_dim + 1)]
    for b, d in bm.compute_persistence_pairs():
        if b >= len(paths):
            continue
        if d == -1:
            d_time = times[-1] + 1
        else:
            d_time = times[d]
        pph_ordered[len(paths[b]) - 1].append((times[b], d_time))
    return pph_ordered


def edge_filtration(
    graph: ListDigraph,
    edge: tuple[int, int],
    max_depth: int,
    max_dim: int,
):
    node_1, node_2 = edge
    nodes_0 = {node_1}
    old_0_nodes = {node_1}
    nodes_1 = {node_1}
    nodes_2 = {node_2}
    old_2_nodes = {node_2}
    nodes_3 = {node_2}

    subgraph = DictDigraph()
    subgraph.add_node(node_1)
    subgraph.add_node(node_2)
    paths: list[tuple[int, ...]] = []
    times: list[int] = []

    for step in range(max_depth - 1):
        ost = step % 4
        if ost == 0:
            new_edges: set[tuple[int, int]] = set()

            new_nodes: set[(int)] = set()

            for u in nodes_0:
                for v in graph.iter_out(u):
                    if v not in subgraph.iter_out(u) and (u, v) != (node_1, node_2):
                        new_edges.add((u, v))

                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            old_0_nodes = nodes_0.copy()
            for _, v in new_edges:
                nodes_0.add(v)
            nodes_0.update(nodes_1)

            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths: set[tuple[int, ...]] = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths: set[tuple[int, ...]] = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
        elif ost == 1:
            new_edges = set()
            new_nodes: set[(int)] = set()
            for v in nodes_1:
                for u in graph.iter_in(v):
                    if v not in subgraph.iter_out(u) and (u, v) != (node_1, node_2):
                        new_edges.add((u, v))
                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            for u, _ in new_edges:
                nodes_1.add(u)
            nodes_1.update(old_0_nodes)

            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
        elif ost == 2:
            new_edges = set()
            new_nodes: set[(int)] = set()
            for u in nodes_2:
                for v in graph.iter_out(u):
                    if v not in subgraph.iter_out(u) and (u, v) != (node_1, node_2):
                        new_edges.add((u, v))
                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            old_2_nodes = nodes_2.copy()
            for _, v in new_edges:
                nodes_2.add(v)
            nodes_2.update(nodes_3)

            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
        elif ost == 3:
            new_edges = set()
            new_nodes: set[(int)] = set()
            for v in nodes_3:
                for u in graph.iter_in(v):
                    if v not in subgraph.iter_out(u) and (u, v) != (node_1, node_2):
                        new_edges.add((u, v))
                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            for u, _ in new_edges:
                nodes_3.add(u)
            nodes_3.update(old_2_nodes)

            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
    paths = [(node_1,), (node_2,)] + paths
    times = [0, 0] + times
    # paths = [(node,) for node in subgraph.nodes] + paths
    # times = [0 for _ in range(len(subgraph.nodes))] + times
    return paths, times


def compute_edge_feat(
    graph: ListDigraph,
    edge: tuple[int, int],
    max_depth: int,
    pixel_size: float,
    max_dim: int,
    sigma: float,
):
    paths, times = edge_filtration(graph, edge, max_depth, max_dim)
    bm = compute_boundary_matrix_phat(paths)
    pph = compute_pph_phat(paths, bm, times, max_dim)
    persistence_image_calculator = PersistenceImager(
        birth_range=(0, max_depth),
        pers_range=(0, max_depth),
        pixel_size=pixel_size,
        kernel_params={"sigma": np.diag((sigma, sigma))},
    )
    diagrams = [
        (np.array(diagram) if len(diagram) > 0 else np.empty((0, 2))) for diagram in pph
    ]
    features: list[np.ndarray[tuple[int, int], np.float64]] = persistence_image_calculator.transform(diagrams)  # type: ignore
    return np.concatenate([f.flatten() for f in features])


def compute_edge_features_parallel(
    graph: ListDigraph,
    edges: Iterable[tuple[int, int]],
    max_depth: int = 9,
    resolution: int = 5,
    max_dim: int = 2,
    sigma: float = 1,
    max_workers: int | None = None,
    chunksize: int = 1024,
    total: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(
        process_map(  # type: ignore
            compute_edge_feat,
            itertools.repeat(graph),
            edges,
            itertools.repeat(max_depth),
            itertools.repeat(max_depth / resolution),
            itertools.repeat(max_dim),
            itertools.repeat(sigma),
            max_workers=max_workers,
            chunksize=chunksize,
            total=total,
        )
    )


def node_filtration(
    graph: ListDigraph,
    node: int,
    max_depth: int,
    max_dim: int,
):

    nodes_0 = {node}
    old_0_nodes = {node}
    nodes_1 = {node}

    subgraph = DictDigraph()
    subgraph.add_node(node)
    paths: list[tuple[int, ...]] = []
    times: list[int] = []

    for step in range(max_depth - 1):
        ost = step % 2
        if ost == 0:
            new_edges: set[tuple[int, int]] = set()
            new_nodes: set[(int)] = set()

            for u in nodes_0:
                for v in graph.iter_out(u):
                    if v not in subgraph.iter_out(u):
                        new_edges.add((u, v))
                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            old_0_nodes = nodes_0.copy()
            for _, v in new_edges:
                nodes_0.add(v)
            nodes_0.update(nodes_1)
            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths: set[tuple[int, ...]] = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths: set[tuple[int, ...]] = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
        elif ost == 1:
            new_edges = set()
            new_nodes: set[(int)] = set()

            for v in nodes_1:
                for u in graph.iter_in(v):
                    if v not in subgraph.iter_out(u):
                        new_edges.add((u, v))
                        if v not in subgraph.nodes:
                            new_nodes.add((v,))

                        subgraph.add_edge(u, v)
            for u, _ in new_edges:
                nodes_1.add(u)
            nodes_1.update(old_0_nodes)
            paths.extend(new_nodes)
            times.extend(itertools.repeat(step + 1, len(new_nodes)))

            paths.extend(new_edges)
            times.extend(itertools.repeat(step + 1, len(new_edges)))
            new_paths = {i for i in new_edges}
            for _ in range(2, max_dim + 1):
                new_new_paths = set()
                for path in new_paths:
                    for v in subgraph.iter_out(path[-1]):
                        new_new_paths.add((*path, v))
                    for u in subgraph.iter_in(path[0]):
                        new_new_paths.add((u, *path))
                new_paths = new_new_paths
                paths.extend(new_paths)
                times.extend(itertools.repeat(step + 1, len(new_paths)))
    paths = [(node,)] + paths
    times = [0] + times
    # paths = [(node,) for node in subgraph.nodes] + paths
    # times = [0 for _ in range(len(subgraph.nodes))] + times
    return paths, times


def compute_node_feat(
    graph: ListDigraph,
    node: int,
    max_depth: int,
    pixel_size: float,
    max_dim: int,
    sigma: float,
):
    paths, times = node_filtration(graph, node, max_depth, max_dim)
    bm = compute_boundary_matrix_phat(paths)
    pph = compute_pph_phat(paths, bm, times, max_dim)
    persistence_image_calculator = PersistenceImager(
        birth_range=(0, max_depth),
        pers_range=(0, max_depth),
        pixel_size=pixel_size,
        kernel_params={"sigma": np.diag((sigma, sigma))},
    )
    diagrams = [
        (np.array(diagram) if len(diagram) > 0 else np.empty((0, 2))) for diagram in pph
    ]
    features: list[np.ndarray[tuple[int, int], np.float64]] = persistence_image_calculator.transform(diagrams)  # type: ignore
    return np.concatenate([f.flatten() for f in features])


def compute_node_features_parallel(
    graph: ListDigraph,
    nodes: Iterable[int],
    max_depth: int = 13,
    resolution: int = 10,
    max_dim: int = 2,
    sigma: float = 1,
    max_workers: int | None = None,
    chunksize: int = 1024,
    total: int | None = None,
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    return np.array(
        process_map(  # type: ignore
            compute_node_feat,
            itertools.repeat(graph),
            nodes,
            itertools.repeat(max_depth),
            itertools.repeat(max_depth / resolution),
            itertools.repeat(max_dim),
            itertools.repeat(sigma),
            max_workers=max_workers,
            chunksize=chunksize,
            total=total,
        )
    )


def read_edgelist(path: str):
    with open(path) as data:
        num_edges = 0
        nodes: set[str] = set()
        for line in data:
            if line.startswith("% "):
                continue
            u_label, v_label, *_ = line.split()
            if u_label != v_label:
                nodes.add(u_label)
                nodes.add(v_label)
                num_edges += 1
        data.seek(0)
        graph = ListDigraph(len(nodes), num_edges)
        for line in data:
            if line.startswith("% "):
                continue
            u, v, *_ = map(
                lambda x: int(x) - 1,
                line.split(),
            )
            if u == v:
                continue
            graph.add_edge(u, v)
    return graph


if __name__ == "__main__":
    graph = read_edgelist(r"data/moreno_highschool/out.moreno_highschool_highschool")
    features = compute_edge_features_parallel(
        graph,
        graph.iter_edges(),
        max_workers=8,
        total=graph.num_edges,
    )
    print(features.shape)
    np.save("features.npy", features)
