"""
Recursive Largest First (RLF) Cost-Aware heuristic for DAA Project - MCCPP
"""

import time
import numpy as np
import networkx as nx
from typing import Dict, Set, Any, Tuple
from src.utils.io_utils import convert_numpy_types
from src.utils.cost_utils import evaluate_solution
from src.utils.graph_utils import get_maximal_independent_set
from src.algorithms.heuristic.dsatur import dsatur_heuristic
from src.algorithms.heuristic.largest_first import largest_first_heuristic


def recursive_largest_first_heuristic(
    graph: nx.Graph, cost_matrix: np.ndarray
) -> Dict[str, Any]:
    """
    Recursive Largest First (RLF) Cost-Aware heuristic for DAA Project - MCCPP.

    Constructs color classes (independent sets) one by one, assigning each class
    a color and selecting vertices to minimize the cost for that color. This
    heuristic typically produces the highest quality solutions among greedy
    approaches but is more computationally expensive.

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix

    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()

    n_colors = cost_matrix.shape[1]

    coloring = {}
    uncolored = set(graph.nodes())
    color_classes = []  # List of independent sets, each will get a color

    # We'll build color classes until all vertices are colored
    # or we run out of colors (shouldn't happen with proper coloring)
    current_color = 0

    while uncolored and current_color < n_colors:
        # Build a maximal independent set from uncolored vertices
        # with cost optimization for the current color

        independent_set = _build_cost_optimized_is(
            graph, uncolored, cost_matrix, current_color
        )

        if not independent_set:
            break

        # Assign current color to all vertices in the independent set
        for vertex in independent_set:
            coloring[vertex] = current_color

        color_classes.append(independent_set)
        uncolored -= independent_set
        current_color += 1

    # Handle any remaining uncolored vertices (shouldn't happen with proper RLF)
    # But if it does, assign them the cheapest available color
    for vertex in uncolored:
        # Find available colors (not used by neighbors)
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        available_colors = set(range(n_colors)) - neighbor_colors

        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            # If no available color, use the one with minimum cost
            best_color = np.argmin(cost_matrix[vertex, :])

        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": cost,
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": "rlf_cost_aware",
        "colors_used": len(set(coloring.values())),
        "color_classes_built": len(color_classes),
        "largest_class_size": max([len(cls) for cls in color_classes])
        if color_classes
        else 0,
    }

    return convert_numpy_types(result)


def _build_cost_optimized_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    current_color: int,
) -> Set[int]:
    """
    Build a cost-optimized maximal independent set for RLF in DAA Project - MCCPP

    Args:
        graph: networkx Graph
        available_vertices: set of vertices available for inclusion
        cost_matrix: cost matrix
        current_color: color being assigned to this independent set

    Returns:
        maximal independent set optimized for the given color cost
    """
    if not available_vertices:
        return set()

    independent_set = set()
    candidates = available_vertices.copy()

    # Order candidates by cost for the current color (cheapest first)
    # and by degree in the subgraph induced by available_vertices
    subgraph = graph.subgraph(available_vertices)
    subgraph_degrees = dict(subgraph.degree())

    # Create candidate list sorted by cost and degree
    sorted_candidates = sorted(
        available_vertices,
        key=lambda v: (cost_matrix[v, current_color], -subgraph_degrees[v]),
    )

    for vertex in sorted_candidates:
        if vertex not in candidates:
            continue

        # Check if vertex can be added to independent set (no conflicts)
        can_add = True
        for neighbor in graph.neighbors(vertex):
            if neighbor in independent_set:
                can_add = False
                break

        if can_add:
            independent_set.add(vertex)
            # Remove vertex and its neighbors from candidates
            candidates.discard(vertex)
            for neighbor in graph.neighbors(vertex):
                candidates.discard(neighbor)

    return independent_set


def adaptive_rlf_heuristic(
    graph: nx.Graph, cost_matrix: np.ndarray, strategy: str = "cost_based"
) -> Dict[str, Any]:
    """
    Adaptive RLF with different strategies for color assignment in DAA Project - MCCPP

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        strategy: color assignment strategy
            - "cost_based": assign colors based on cost optimization
            - "size_based": prioritize larger independent sets
            - "balanced": balance between cost and set size

    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()

    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]

    coloring = {}
    uncolored = set(graph.nodes())
    color_assignments = []  # Track (color, independent_set_size, total_cost)

    # Precompute which color is best for each vertex (ignoring constraints)
    vertex_best_colors = {}
    for vertex in graph.nodes():
        vertex_best_colors[vertex] = np.argmin(cost_matrix[vertex, :])

    color_index = 0
    iteration = 0
    max_iterations = n_vertices * 2  # Safety limit

    while uncolored and color_index < n_colors and iteration < max_iterations:
        iteration += 1

        if strategy == "cost_based":
            independent_set, assigned_color = _build_cost_based_is(
                graph, uncolored, cost_matrix, color_index, n_colors
            )
        elif strategy == "size_based":
            independent_set, assigned_color = _build_size_based_is(
                graph, uncolored, cost_matrix, color_index, n_colors
            )
        elif strategy == "balanced":
            independent_set, assigned_color = _build_balanced_is(
                graph, uncolored, cost_matrix, color_index, n_colors
            )
        else:
            independent_set, assigned_color = _build_cost_based_is(
                graph, uncolored, cost_matrix, color_index, n_colors
            )

        if not independent_set:
            break

        # Assign color to independent set
        for vertex in independent_set:
            coloring[vertex] = assigned_color

        set_cost = sum(cost_matrix[v, assigned_color] for v in independent_set)
        color_assignments.append((assigned_color, len(independent_set), set_cost))

        uncolored -= independent_set

        # Move to next color
        color_index += 1

    # Handle remaining vertices
    for vertex in uncolored:
        # Find available color with minimum cost
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        available_colors = set(range(n_colors)) - neighbor_colors

        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            best_color = vertex_best_colors[vertex]

        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": cost,
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": f"adaptive_rlf_{strategy}",
        "colors_used": len(set(coloring.values())),
        "color_assignments": color_assignments,
        "iterations": iteration,
    }

    return convert_numpy_types(result)


def _build_cost_based_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    color_index: int,
    n_colors: int,
) -> Tuple[Set[int], int]:
    """Build IS optimized for cost"""
    # Try all colors to find the best one for current available vertices
    best_set = None
    best_color = color_index
    best_cost = float("inf")

    for color in range(n_colors):
        candidate_set = set()
        candidates = available_vertices.copy()

        # Sort by cost for this color
        sorted_candidates = sorted(candidates, key=lambda v: cost_matrix[v, color])

        for vertex in sorted_candidates:
            if vertex not in candidates:
                continue

            # Check if vertex can be added
            can_add = True
            for neighbor in graph.neighbors(vertex):
                if neighbor in candidate_set:
                    can_add = False
                    break

            if can_add:
                candidate_set.add(vertex)
                # Remove vertex and its neighbors
                candidates.discard(vertex)
                for neighbor in graph.neighbors(vertex):
                    candidates.discard(neighbor)

        if candidate_set:
            set_cost = sum(cost_matrix[v, color] for v in candidate_set)
            if set_cost < best_cost:
                best_cost = set_cost
                best_set = candidate_set
                best_color = color

    return best_set if best_set else set(), best_color


def _build_size_based_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    color_index: int,
    n_colors: int,
) -> Tuple[Set[int], int]:
    """Build IS optimized for size"""
    # Build the largest possible IS first, then find best color for it
    independent_set = get_maximal_independent_set(graph, available_vertices)

    if not independent_set:
        return set(), color_index

    # Find the color that minimizes total cost for this IS
    best_color = color_index
    best_cost = float("inf")

    for color in range(n_colors):
        set_cost = sum(cost_matrix[v, color] for v in independent_set)
        if set_cost < best_cost:
            best_cost = set_cost
            best_color = color

    return independent_set, best_color


def _build_balanced_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    color_index: int,
    n_colors: int,
) -> Tuple[Set[int], int]:
    """Build IS with balance between size and cost"""
    best_score = -float("inf")
    best_set = None
    best_color = color_index

    for color in range(n_colors):
        candidate_set = set()
        candidates = available_vertices.copy()

        # Sort by a balanced metric: cost efficiency
        def cost_efficiency(v):
            return -cost_matrix[
                v, color
            ]  # Negative because we want lower cost to be better

        sorted_candidates = sorted(candidates, key=cost_efficiency)

        for vertex in sorted_candidates:
            if vertex not in candidates:
                continue

            can_add = True
            for neighbor in graph.neighbors(vertex):
                if neighbor in candidate_set:
                    can_add = False
                    break

            if can_add:
                candidate_set.add(vertex)
                candidates.discard(vertex)
                for neighbor in graph.neighbors(vertex):
                    candidates.discard(neighbor)

        if candidate_set:
            set_size = len(candidate_set)
            set_cost = sum(cost_matrix[v, color] for v in candidate_set)
            avg_cost = set_cost / set_size if set_size > 0 else float("inf")

            # Balanced score: favor large sets with low average cost
            score = set_size / (avg_cost + 1)  # +1 to avoid division by zero

            if score > best_score:
                best_score = score
                best_set = candidate_set
                best_color = color

    return best_set if best_set else set(), best_color


def adaptive_greedy_heuristic(
    graph: nx.Graph, cost_matrix: np.ndarray, method: str = "auto"
) -> Dict[str, Any]:
    """
    Adaptive greedy heuristic that chooses the best method based on graph properties
    for DAA Project - MCCPP

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        method: heuristic method to use, or "auto" for automatic selection

    Returns:
        dictionary with solution and metrics
    """
    n_vertices = graph.number_of_nodes()
    density = nx.density(graph)

    if method == "auto":
        # Choose method based on graph properties
        if n_vertices <= 50:
            # For small graphs, use RLF for better quality
            method = "rlf"
        elif density < 0.3:
            # For sparse graphs, DSATUR works well
            method = "dsatur"
        else:
            # For dense graphs or large instances, use LF for speed
            method = "lf"

    if method == "rlf" or method == "rlf_cost_aware":
        return recursive_largest_first_heuristic(graph, cost_matrix)
    elif method == "dsatur" or method == "dsatur_cost_aware":
        return dsatur_heuristic(graph, cost_matrix)
    elif method == "lf" or method == "largest_first":
        return largest_first_heuristic(graph, cost_matrix)
    else:
        # Default to DSATUR
        return dsatur_heuristic(graph, cost_matrix)
