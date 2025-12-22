"""
Largest First (LF) Cost-Aware heuristic for DAA Project - MCCPP
"""

import time
from typing import Dict, Any
import networkx as nx
import numpy as np
from src.utils.io_utils import convert_numpy_types
from src.utils.graph_utils import calculate_degrees
from src.utils.cost_utils import evaluate_solution


def largest_first_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Largest First (LF) Cost-Aware heuristic for DAA Project - MCCPP.

    Orders vertices by decreasing degree and assigns the cheapest available color
    to each vertex. This is the fastest heuristic but often produces lower quality
    solutions due to its static ordering.

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix

    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()

    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())

    # Calculate degrees and sort vertices by decreasing degree
    degrees = calculate_degrees(graph)
    sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)

    coloring = {}
    color_usage = {}  # Track which colors are used by neighbors for each vertex

    # Initialize color usage tracking
    for vertex in vertices:
        color_usage[vertex] = set()

    # Color vertices in order of decreasing degree
    for vertex in sorted_vertices:
        # Get colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        # Find available colors (colors not used by neighbors)
        available_colors = set(range(n_colors)) - neighbor_colors

        if available_colors:
            # Choose the available color with minimum cost for this vertex
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            # If no available color, use the color with minimum cost (will cause conflict)
            best_color = np.argmin(cost_matrix[vertex, :])

        coloring[vertex] = best_color

        # Update color usage for neighbors
        for neighbor in graph.neighbors(vertex):
            color_usage[neighbor].add(best_color)

    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()

    # Convertir tipos de NumPy a tipos nativos de Python para serialización JSON
    result = {
        "solution": {
            int(k): int(v) for k, v in coloring.items()
        },  # Convertir a int nativos
        "cost": float(cost),  # Convertir a float nativo
        "execution_time": float(end_time - start_time),
        "optimal": False,
        "algorithm": "largest_first_cost_aware",
        "colors_used": int(len(set(coloring.values()))),  # Convertir a int nativo
        "vertex_ordering": "decreasing_degree",
    }

    return convert_numpy_types(result)


def largest_first_variant(
    graph: nx.Graph, cost_matrix: np.ndarray, ordering: str = "degree"
) -> Dict[str, Any]:
    """
    Variant of Largest First with different vertex ordering strategies for DAA Project - MCCPP

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        ordering: vertex ordering strategy
            - "degree": decreasing degree (standard LF)
            - "cost_variance": vertices with highest cost variance first
            - "weighted_degree": degree weighted by cost factors
            - "random": random ordering

    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()

    vertices = list(graph.nodes())

    # Determine vertex ordering based on strategy
    if ordering == "degree":
        degrees = calculate_degrees(graph)
        sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)

    elif ordering == "cost_variance":
        # Calculate cost variance for each vertex
        cost_variances = {}
        for vertex in vertices:
            costs = cost_matrix[vertex, :]
            cost_variances[vertex] = float(np.var(costs))  # Convertir a float nativo
        sorted_vertices = sorted(
            vertices, key=lambda v: cost_variances[v], reverse=True
        )

    elif ordering == "weighted_degree":
        # Weight degree by average cost (higher weight for expensive vertices)
        degrees = calculate_degrees(graph)
        avg_costs = {
            v: float(np.mean(cost_matrix[v, :])) for v in vertices
        }  # Convertir a float nativo
        # Combine degree and cost (higher values mean more constrained/expensive)
        weights = {
            v: float(degrees[v] * avg_costs[v]) for v in vertices
        }  # Convertir a float nativo
        sorted_vertices = sorted(vertices, key=lambda v: weights[v], reverse=True)

    elif ordering == "random":
        np.random.shuffle(vertices)
        sorted_vertices = vertices

    else:
        # Default to degree ordering
        degrees = calculate_degrees(graph)
        sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)

    coloring = {}

    # Color vertices in the determined order
    for vertex in sorted_vertices:
        # Get colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        # Find available colors
        available_colors = set(range(cost_matrix.shape[1])) - neighbor_colors

        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            best_color = np.argmin(cost_matrix[vertex, :])

        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()

    result = {
        "solution": {
            int(k): int(v) for k, v in coloring.items()
        },  # Convertir a int nativos
        "cost": float(cost),  # Convertir a float nativo
        "execution_time": float(end_time - start_time),
        "optimal": False,
        "algorithm": f"largest_first_{ordering}",
        "colors_used": int(len(set(coloring.values()))),  # Convertir a int nativo
        "vertex_ordering": ordering,
    }

    return convert_numpy_types(result)
