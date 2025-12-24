import time
import heapq
from typing import Dict, Set, Any, Tuple
import networkx as nx
import numpy as np
from utils.utils import convert_numpy_types, is_proper_coloring, calculate_degrees, evaluate_solution, get_maximal_independent_set
from utils.timeout_handler import check_global_timeout


# LARGEST FIRST

def largest_first_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Largest First (LF) Cost-Aware heuristic.
    """
    start_time = time.time()
    operations = 0

    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())

    # Calculate degrees
    degrees = calculate_degrees(graph)
    operations += len(vertices) # Accessing degrees

    # Sort vertices
    sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)
    operations += len(vertices) * np.log(len(vertices)) if len(vertices) > 0 else 0 # Sorting cost estimate

    coloring = {}
    
    # Color vertices in order of decreasing degree
    for vertex in sorted_vertices:
        operations += 1
        
        # Get colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            operations += 1
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        # Find available colors
        available_colors = set(range(n_colors)) - neighbor_colors
        operations += len(available_colors) # Set operations

        if available_colors:
            # Choose the available color with minimum cost
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
            operations += len(available_colors) # Min comparison
        else:
            # Conflict handling
            best_color = np.argmin(cost_matrix[vertex, :])
            operations += n_colors

        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": float(cost),
        "execution_time": float(end_time - start_time),
        "optimal": False,
        "algorithm": "largest_first",
        "feasible": feasible,
        "colors_used": int(len(set(coloring.values()))),
        "operations": int(operations)
    }

    return convert_numpy_types(result)

# def largest_first_variant(
#     graph: nx.Graph, cost_matrix: np.ndarray, ordering: str = "degree"
# ) -> Dict[str, Any]:
#     """
#     Variant of Largest First with different vertex ordering strategies for DAA Project - MCCPP

#     Args:
#         graph: networkx Graph
#         cost_matrix: n_vertices x n_colors cost matrix
#         ordering: vertex ordering strategy
#             - "degree": decreasing degree (standard LF)
#             - "cost_variance": vertices with highest cost variance first
#             - "weighted_degree": degree weighted by cost factors
#             - "random": random ordering

#     Returns:
#         dictionary with solution and metrics
#     """
#     start_time = time.time()

#     vertices = list(graph.nodes())

#     # Determine vertex ordering based on strategy
#     if ordering == "degree":
#         degrees = calculate_degrees(graph)
#         sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)

#     elif ordering == "cost_variance":
#         # Calculate cost variance for each vertex
#         cost_variances = {}
#         for vertex in vertices:
#             costs = cost_matrix[vertex, :]
#             cost_variances[vertex] = float(np.var(costs))  # Convertir a float nativo
#         sorted_vertices = sorted(
#             vertices, key=lambda v: cost_variances[v], reverse=True
#         )

#     elif ordering == "weighted_degree":
#         # Weight degree by average cost (higher weight for expensive vertices)
#         degrees = calculate_degrees(graph)
#         avg_costs = {
#             v: float(np.mean(cost_matrix[v, :])) for v in vertices
#         }  # Convertir a float nativo
#         # Combine degree and cost (higher values mean more constrained/expensive)
#         weights = {
#             v: float(degrees[v] * avg_costs[v]) for v in vertices
#         }  # Convertir a float nativo
#         sorted_vertices = sorted(vertices, key=lambda v: weights[v], reverse=True)

#     elif ordering == "random":
#         np.random.shuffle(vertices)
#         sorted_vertices = vertices

#     else:
#         # Default to degree ordering
#         degrees = calculate_degrees(graph)
#         sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)

#     coloring = {}

#     # Color vertices in the determined order
#     for vertex in sorted_vertices:
#         # Get colors used by neighbors
#         neighbor_colors = set()
#         for neighbor in graph.neighbors(vertex):
#             if neighbor in coloring:
#                 neighbor_colors.add(coloring[neighbor])

#         # Find available colors
#         available_colors = set(range(cost_matrix.shape[1])) - neighbor_colors

#         if available_colors:
#             best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
#         else:
#             best_color = np.argmin(cost_matrix[vertex, :])

#         coloring[vertex] = best_color

#     cost = evaluate_solution(coloring, cost_matrix)
#     end_time = time.time()

#     result = {
#         "solution": {
#             int(k): int(v) for k, v in coloring.items()
#         },  # Convertir a int nativos
#         "cost": float(cost),  # Convertir a float nativo
#         "execution_time": float(end_time - start_time),
#         "optimal": False,
#         "algorithm": f"largest_first_{ordering}",
#         "colors_used": int(len(set(coloring.values()))),  # Convertir a int nativo
#         "vertex_ordering": ordering,
#     }

#     return convert_numpy_types(result)

# DEGREE OF SATURATION

class DSATURVertex:
    def __init__(self, vertex_id: int, degree: int):
        self.vertex_id = vertex_id
        self.saturation = 0
        self.degree = degree
        self.colored = False
    
    def __lt__(self, other):
        if self.saturation != other.saturation:
            return self.saturation > other.saturation
        else:
            return self.degree > other.degree

def dsatur_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    DSATUR (Degree of Saturation) Cost-Aware heuristic.
    """
    start_time = time.time()
    operations = 0
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())
    
    dsatur_vertices = {}
    degrees = calculate_degrees(graph)
    operations += len(vertices)
    
    for vertex in vertices:
        dsatur_vertices[vertex] = DSATURVertex(vertex, degrees[vertex])
    
    heap = []
    for vertex in vertices:
        heapq.heappush(heap, (-dsatur_vertices[vertex].saturation, -degrees[vertex], vertex))
        operations += 1
    
    coloring = {}
    color_adjacency = {v: set() for v in vertices}
    
    while heap:
        operations += 1
        # Check global timeout frequently
        try:
            check_global_timeout()
        except Exception:
            break
        _, _, current_vertex = heapq.heappop(heap)
        vertex_obj = dsatur_vertices[current_vertex]
        
        if vertex_obj.colored:
            continue
        
        available_colors = set(range(n_colors)) - color_adjacency[current_vertex]
        
        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[current_vertex, c])
            operations += len(available_colors)
        else:
            best_color = np.argmin(cost_matrix[current_vertex, :])
            operations += n_colors
        
        coloring[current_vertex] = best_color
        vertex_obj.colored = True
        
        for neighbor in graph.neighbors(current_vertex):
            operations += 1
            if not dsatur_vertices[neighbor].colored:
                color_adjacency[neighbor].add(best_color)
                
                old_saturation = dsatur_vertices[neighbor].saturation
                dsatur_vertices[neighbor].saturation = len(color_adjacency[neighbor])
                
                if dsatur_vertices[neighbor].saturation != old_saturation:
                    heapq.heappush(heap, (-dsatur_vertices[neighbor].saturation, 
                                         -dsatur_vertices[neighbor].degree, 
                                         neighbor))
                    operations += 1 # Heap push cost proxy
    
    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': 'dsatur',
        'feasible': feasible,
        'colors_used': len(set(coloring.values())),
        'operations': int(operations)
    }
    
    return convert_numpy_types(result)

# def enhanced_dsatur(graph: nx.Graph, cost_matrix: np.ndarray,
#                    tie_breaking: str = "degree") -> Dict[str, Any]:
#     """
#     Enhanced DSATUR with improved tie-breaking and cost considerations for DAA Project - MCCPP
    
#     Args:
#         graph: networkx Graph
#         cost_matrix: n_vertices x n_colors cost matrix
#         tie_breaking: method for breaking ties when saturation is equal
#             - "degree": use vertex degree
#             - "cost_sensitivity": use cost variance or sensitivity
#             - "composite": combination of degree and cost factors
    
#     Returns:
#         dictionary with solution and metrics
#     """
#     start_time = time.time()
    
#     n_vertices = graph.number_of_nodes()
#     n_colors = cost_matrix.shape[1]
#     vertices = list(graph.nodes())
    
#     # Initialize data structures
#     coloring = {}
#     color_adjacency = {v: set() for v in vertices}
#     degrees = calculate_degrees(graph)
    
#     # Calculate additional metrics for tie-breaking
#     if tie_breaking == "cost_sensitivity":
#         cost_variances = {v: np.var(cost_matrix[v, :]) for v in vertices}
#     elif tie_breaking == "composite":
#         cost_variances = {v: np.var(cost_matrix[v, :]) for v in vertices}
#         avg_costs = {v: np.mean(cost_matrix[v, :]) for v in vertices}
    
#     uncolored = set(vertices)
    
#     while uncolored:
#         # Find uncolored vertex with maximum saturation
#         max_saturation = -1
#         candidates = []
        
#         for vertex in uncolored:
#             saturation = len(color_adjacency[vertex])
#             if saturation > max_saturation:
#                 max_saturation = saturation
#                 candidates = [vertex]
#             elif saturation == max_saturation:
#                 candidates.append(vertex)
        
#         # Tie-breaking
#         if tie_breaking == "degree":
#             current_vertex = max(candidates, key=lambda v: degrees[v])
#         elif tie_breaking == "cost_sensitivity":
#             current_vertex = max(candidates, key=lambda v: cost_variances[v])
#         elif tie_breaking == "composite":
#             # Combine degree and cost factors
#             def composite_score(v):
#                 return degrees[v] * cost_variances[v] * avg_costs[v]
#             current_vertex = max(candidates, key=composite_score)
#         else:
#             current_vertex = max(candidates, key=lambda v: degrees[v])
        
#         # Choose color for current vertex
#         available_colors = set(range(n_colors)) - color_adjacency[current_vertex]
        
#         if available_colors:
#             # Enhanced color selection: consider both immediate cost and potential impact
#             def color_score(color):
#                 base_cost = cost_matrix[current_vertex, color]
#                 # Consider how many uncolored neighbors could still use this color
#                 neighbor_impact = 0
#                 for neighbor in graph.neighbors(current_vertex):
#                     if neighbor in uncolored:
#                         # If neighbor has high cost for this color, it's less likely to use it
#                         neighbor_cost = cost_matrix[neighbor, color]
#                         neighbor_impact += neighbor_cost
#                 return base_cost + (neighbor_impact * 0.1)  # Weighted impact
            
#             best_color = min(available_colors, key=color_score)
#         else:
#             best_color = np.argmin(cost_matrix[current_vertex, :])
        
#         coloring[current_vertex] = best_color
#         uncolored.remove(current_vertex)
        
#         # Update color adjacency for neighbors
#         for neighbor in graph.neighbors(current_vertex):
#             if neighbor in uncolored:
#                 color_adjacency[neighbor].add(best_color)
    
#     cost = evaluate_solution(coloring, cost_matrix)
#     end_time = time.time()
    
#     result = {
#         'solution': coloring,
#         'cost': cost,
#         'execution_time': end_time - start_time,
#         'optimal': False,
#         'algorithm': f'enhanced_dsatur_{tie_breaking}',
#         'colors_used': len(set(coloring.values())),
#         'tie_breaking_method': tie_breaking
#     }
    
#     return convert_numpy_types(result)

# RECURSIVE LARGEST FIRST

def recursive_largest_first_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Recursive Largest First (RLF) Cost-Aware heuristic.
    """
    start_time = time.time()
    operations = 0

    n_colors = cost_matrix.shape[1]
    coloring = {}
    uncolored = set(graph.nodes())
    
    current_color = 0

    while uncolored and current_color < n_colors:
        operations += 1
        # Check global timeout frequently
        try:
            check_global_timeout()
        except Exception:
            break
        
        # Construir conjunto independiente
        independent_set, ops_in_step = _build_cost_optimized_is(
            graph, uncolored, cost_matrix, current_color
        )
        operations += ops_in_step

        if not independent_set:
            break

        for vertex in independent_set:
            coloring[vertex] = current_color
            operations += 1

        uncolored -= independent_set
        current_color += 1

    # Rellenar restantes
    for vertex in uncolored:
        operations += 1
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        
        available_colors = set(range(n_colors)) - neighbor_colors
        
        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            best_color = np.argmin(cost_matrix[vertex, :])
        
        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": cost,
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": "rlf",
        "feasible": feasible,
        "colors_used": len(set(coloring.values())),
        "operations": int(operations)
    }

    return convert_numpy_types(result)

def _build_cost_optimized_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    current_color: int,
) -> Tuple[Set[int], int]:
    """Helper for RLF. Returns set and operation count."""
    ops = 0
    if not available_vertices:
        return set(), ops

    independent_set = set()
    candidates = available_vertices.copy()

    subgraph = graph.subgraph(available_vertices)
    subgraph_degrees = dict(subgraph.degree())
    ops += len(available_vertices)

    sorted_candidates = sorted(
        available_vertices,
        key=lambda v: (cost_matrix[v, current_color], -subgraph_degrees[v]),
    )
    ops += len(available_vertices) * np.log(len(available_vertices)) if len(available_vertices) > 0 else 0

    for vertex in sorted_candidates:
        ops += 1
        if vertex not in candidates:
            continue

        can_add = True
        for neighbor in graph.neighbors(vertex):
            ops += 1
            if neighbor in independent_set:
                can_add = False
                break

        if can_add:
            independent_set.add(vertex)
            candidates.discard(vertex)
            for neighbor in graph.neighbors(vertex):
                candidates.discard(neighbor)
                ops += 1

    return independent_set, ops

# def adaptive_rlf_heuristic(
#     graph: nx.Graph, cost_matrix: np.ndarray, strategy: str = "cost_based"
# ) -> Dict[str, Any]:
#     """
#     Adaptive RLF with different strategies for color assignment in DAA Project - MCCPP

#     Args:
#         graph: networkx Graph
#         cost_matrix: n_vertices x n_colors cost matrix
#         strategy: color assignment strategy
#             - "cost_based": assign colors based on cost optimization
#             - "size_based": prioritize larger independent sets
#             - "balanced": balance between cost and set size

#     Returns:
#         dictionary with solution and metrics
#     """
#     start_time = time.time()

#     n_vertices = graph.number_of_nodes()
#     n_colors = cost_matrix.shape[1]

#     coloring = {}
#     uncolored = set(graph.nodes())
#     color_assignments = []  # Track (color, independent_set_size, total_cost)

#     # Precompute which color is best for each vertex (ignoring constraints)
#     vertex_best_colors = {}
#     for vertex in graph.nodes():
#         vertex_best_colors[vertex] = np.argmin(cost_matrix[vertex, :])

#     color_index = 0
#     iteration = 0
#     max_iterations = n_vertices * 2  # Safety limit

#     while uncolored and color_index < n_colors and iteration < max_iterations:
#         iteration += 1

#         if strategy == "cost_based":
#             independent_set, assigned_color = _build_cost_based_is(
#                 graph, uncolored, cost_matrix, color_index, n_colors
#             )
#         elif strategy == "size_based":
#             independent_set, assigned_color = _build_size_based_is(
#                 graph, uncolored, cost_matrix, color_index, n_colors
#             )
#         elif strategy == "balanced":
#             independent_set, assigned_color = _build_balanced_is(
#                 graph, uncolored, cost_matrix, color_index, n_colors
#             )
#         else:
#             independent_set, assigned_color = _build_cost_based_is(
#                 graph, uncolored, cost_matrix, color_index, n_colors
#             )

#         if not independent_set:
#             break

#         # Assign color to independent set
#         for vertex in independent_set:
#             coloring[vertex] = assigned_color

#         set_cost = sum(cost_matrix[v, assigned_color] for v in independent_set)
#         color_assignments.append((assigned_color, len(independent_set), set_cost))

#         uncolored -= independent_set

#         # Move to next color
#         color_index += 1

#     # Handle remaining vertices
#     for vertex in uncolored:
#         # Find available color with minimum cost
#         neighbor_colors = set()
#         for neighbor in graph.neighbors(vertex):
#             if neighbor in coloring:
#                 neighbor_colors.add(coloring[neighbor])

#         available_colors = set(range(n_colors)) - neighbor_colors

#         if available_colors:
#             best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
#         else:
#             best_color = vertex_best_colors[vertex]

#         coloring[vertex] = best_color

#     cost = evaluate_solution(coloring, cost_matrix)
#     end_time = time.time()

#     result = {
#         "solution": coloring,
#         "cost": cost,
#         "execution_time": end_time - start_time,
#         "optimal": False,
#         "algorithm": f"adaptive_rlf_{strategy}",
#         "colors_used": len(set(coloring.values())),
#         "color_assignments": color_assignments,
#         "iterations": iteration,
#     }

#     return convert_numpy_types(result)

# def _build_cost_based_is(
#     graph: nx.Graph,
#     available_vertices: Set[int],
#     cost_matrix: np.ndarray,
#     color_index: int,
#     n_colors: int,
# ) -> Tuple[Set[int], int]:
#     """Build IS optimized for cost"""
#     # Try all colors to find the best one for current available vertices
#     best_set = None
#     best_color = color_index
#     best_cost = float("inf")

#     for color in range(n_colors):
#         candidate_set = set()
#         candidates = available_vertices.copy()

#         # Sort by cost for this color
#         sorted_candidates = sorted(candidates, key=lambda v: cost_matrix[v, color])

#         for vertex in sorted_candidates:
#             if vertex not in candidates:
#                 continue

#             # Check if vertex can be added
#             can_add = True
#             for neighbor in graph.neighbors(vertex):
#                 if neighbor in candidate_set:
#                     can_add = False
#                     break

#             if can_add:
#                 candidate_set.add(vertex)
#                 # Remove vertex and its neighbors
#                 candidates.discard(vertex)
#                 for neighbor in graph.neighbors(vertex):
#                     candidates.discard(neighbor)

#         if candidate_set:
#             set_cost = sum(cost_matrix[v, color] for v in candidate_set)
#             if set_cost < best_cost:
#                 best_cost = set_cost
#                 best_set = candidate_set
#                 best_color = color

#     return best_set if best_set else set(), best_color

# def _build_size_based_is(
#     graph: nx.Graph,
#     available_vertices: Set[int],
#     cost_matrix: np.ndarray,
#     color_index: int,
#     n_colors: int,
# ) -> Tuple[Set[int], int]:
#     """Build IS optimized for size"""
#     # Build the largest possible IS first, then find best color for it
#     independent_set = get_maximal_independent_set(graph, available_vertices)

#     if not independent_set:
#         return set(), color_index

#     # Find the color that minimizes total cost for this IS
#     best_color = color_index
#     best_cost = float("inf")

#     for color in range(n_colors):
#         set_cost = sum(cost_matrix[v, color] for v in independent_set)
#         if set_cost < best_cost:
#             best_cost = set_cost
#             best_color = color

#     return independent_set, best_color

# def _build_balanced_is(
#     graph: nx.Graph,
#     available_vertices: Set[int],
#     cost_matrix: np.ndarray,
#     color_index: int,
#     n_colors: int,
# ) -> Tuple[Set[int], int]:
#     """Build IS with balance between size and cost"""
#     best_score = -float("inf")
#     best_set = None
#     best_color = color_index

#     for color in range(n_colors):
#         candidate_set = set()
#         candidates = available_vertices.copy()

#         # Sort by a balanced metric: cost efficiency
#         def cost_efficiency(v):
#             return -cost_matrix[
#                 v, color
#             ]  # Negative because we want lower cost to be better

#         sorted_candidates = sorted(candidates, key=cost_efficiency)

#         for vertex in sorted_candidates:
#             if vertex not in candidates:
#                 continue

#             can_add = True
#             for neighbor in graph.neighbors(vertex):
#                 if neighbor in candidate_set:
#                     can_add = False
#                     break

#             if can_add:
#                 candidate_set.add(vertex)
#                 candidates.discard(vertex)
#                 for neighbor in graph.neighbors(vertex):
#                     candidates.discard(neighbor)

#         if candidate_set:
#             set_size = len(candidate_set)
#             set_cost = sum(cost_matrix[v, color] for v in candidate_set)
#             avg_cost = set_cost / set_size if set_size > 0 else float("inf")

#             # Balanced score: favor large sets with low average cost
#             score = set_size / (avg_cost + 1)  # +1 to avoid division by zero

#             if score > best_score:
#                 best_score = score
#                 best_set = candidate_set
#                 best_color = color

#     return best_set if best_set else set(), best_color

# def adaptive_greedy_heuristic(
#     graph: nx.Graph, cost_matrix: np.ndarray, method: str = "auto"
# ) -> Dict[str, Any]:
#     """
#     Adaptive greedy heuristic that chooses the best method based on graph properties
#     for DAA Project - MCCPP

#     Args:
#         graph: networkx Graph
#         cost_matrix: n_vertices x n_colors cost matrix
#         method: heuristic method to use, or "auto" for automatic selection

#     Returns:
#         dictionary with solution and metrics
#     """
#     n_vertices = graph.number_of_nodes()
#     density = nx.density(graph)

#     if method == "auto":
#         # Choose method based on graph properties
#         if n_vertices <= 50:
#             # For small graphs, use RLF for better quality
#             method = "rlf"
#         elif density < 0.3:
#             # For sparse graphs, DSATUR works well
#             method = "dsatur"
#         else:
#             # For dense graphs or large instances, use LF for speed
#             method = "lf"

#     if method == "rlf" or method == "rlf_cost_aware":
#         return recursive_largest_first_heuristic(graph, cost_matrix)
#     elif method == "dsatur" or method == "dsatur_cost_aware":
#         return dsatur_heuristic(graph, cost_matrix)
#     elif method == "lf" or method == "largest_first":
#         return largest_first_heuristic(graph, cost_matrix)
#     else:
#         # Default to DSATUR
#         return dsatur_heuristic(graph, cost_matrix)
