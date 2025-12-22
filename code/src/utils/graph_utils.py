"""
Graph utilities for DAA Project - MCCPP
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple


def generate_erdos_renyi_graph(n: int, p: float, seed: int = None) -> nx.Graph:
    """
    Generate Erdős-Rényi random graph G(n, p)

    Args:
        n: number of vertices
        p: probability of edge creation
        seed: random seed for reproducibility

    Returns:
        networkx Graph object
    """
    return nx.erdos_renyi_graph(n, p, seed=seed, directed=False)


def generate_interval_graph(
    n: int, max_length: float = 1.0, seed: int = None
) -> nx.Graph:
    """
    Generate interval graph for FISP-MDC problems

    Args:
        n: number of intervals
        max_length: maximum interval length
        seed: random seed

    Returns:
        interval graph and interval data
    """
    if seed:
        np.random.seed(seed)

    intervals = []
    for i in range(n):
        start = np.random.uniform(0, max_length * 0.8)
        end = start + np.random.uniform(0.1, max_length * 0.2)
        intervals.append((start, end))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges for overlapping intervals
    for i in range(n):
        for j in range(i + 1, n):
            if (
                intervals[i][1] >= intervals[j][0]
                and intervals[j][1] >= intervals[i][0]
            ):
                G.add_edge(i, j)

    return G, intervals


def is_proper_coloring(graph: nx.Graph, coloring: Dict[int, int]) -> bool:
    """
    Verify if a coloring is proper (no adjacent vertices have same color)

    Args:
        graph: networkx Graph
        coloring: dictionary mapping vertex -> color

    Returns:
        True if coloring is proper, False otherwise
    """
    for u, v in graph.edges():
        # if not coloring.get(u) or not coloring.get(v):
        #     return False
        if coloring.get(u) == coloring.get(v):
            return False
    return True


def calculate_degrees(graph: nx.Graph) -> Dict[int, int]:
    """
    Calculate degrees of all vertices

    Args:
        graph: networkx Graph

    Returns:
        dictionary of vertex -> degree
    """
    return dict(graph.degree())


def calculate_saturation_degrees(
    graph: nx.Graph, colored_vertices: Set[int]
) -> Dict[int, int]:
    """
    Calculate saturation degrees (number of different colors in neighborhood)

    Args:
        graph: networkx Graph
        colored_vertices: set of already colored vertices

    Returns:
        dictionary of vertex -> saturation degree
    """
    saturation = {}
    for node in graph.nodes():
        if node in colored_vertices:
            saturation[node] = -1  # Already colored
            continue

        neighbor_colors = set()
        for neighbor in graph.neighbors(node):
            if neighbor in colored_vertices:
                # In actual implementation, we'd track colors
                # For now, we'll return basic saturation
                neighbor_colors.add(1)  # Placeholder
        saturation[node] = len(neighbor_colors)

    return saturation


def get_maximal_independent_set(
    graph: nx.Graph, available_vertices: Set[int] = None
) -> Set[int]:
    """
    Find a maximal independent set using greedy approach

    Args:
        graph: networkx Graph
        available_vertices: subset of vertices to consider

    Returns:
        set of vertices in the independent set
    """
    if available_vertices is None:
        available_vertices = set(graph.nodes())

    independent_set = set()
    remaining = available_vertices.copy()

    while remaining:
        # Select vertex with minimum degree in remaining graph
        node = min(remaining, key=lambda x: len(set(graph.neighbors(x)) & remaining))
        independent_set.add(node)

        # Remove selected node and its neighbors
        remaining.discard(node)
        remaining -= set(graph.neighbors(node))

    return independent_set


def graph_density(graph: nx.Graph) -> float:
    """
    Calculate graph density

    Args:
        graph: networkx Graph

    Returns:
        density value between 0 and 1
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    return (2 * graph.number_of_edges()) / (n * (n - 1))
