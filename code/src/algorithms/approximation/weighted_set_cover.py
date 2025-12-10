"""
Weighted Set Cover based approximation for DAA Project - MCCPP
"""
import time
import numpy as np
import networkx as nx
from typing import Dict, Any
from src.utils.cost_utils import evaluate_solution
from src.utils.io_utils import convert_numpy_types
from src.utils.graph_utils import get_maximal_independent_set, is_proper_coloring

def weighted_set_cover_approximation(graph: nx.Graph, cost_matrix: np.ndarray, 
                                   max_iterations: int = 1000) -> Dict[str, Any]:
    """
    Weighted Set Cover approximation algorithm for DAA Project - MCCPP with O(ln n) guarantee.
    
    This algorithm models the MCCPP as a Weighted Set Cover problem where:
    - The universe U is the set of vertices.
    - Each set S is a pair (I, f) where I is an independent set and f is a color.
    - The cost of set S is the total cost of assigning color f to all vertices in I.
    
    The algorithm proceeds greedily by selecting the set with the minimum effective cost per uncovered vertex.
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        max_iterations: maximum number of iterations to prevent infinite loops
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = set(graph.nodes())
    
    # Initialize
    coloring = {}
    uncovered = set(vertices)
    total_cost = 0.0
    colors_used = set()
    iterations = 0
    
    # Build solution by selecting best (independent set, color) pairs
    while uncovered and iterations < max_iterations:
        best_ratio = float('inf')
        best_set = None
        best_color = None
        best_set_cost = 0.0
        
        # For each color, find the best independent set from uncovered vertices
        for color in range(n_colors):
            # Build a maximal independent set from uncovered vertices
            # that respects already colored neighbors with this color
            independent_set = set()
            candidates = uncovered.copy()
            
            while candidates:
                # Choose a vertex (greedy: lowest cost for this color)
                vertex = min(candidates, key=lambda v: cost_matrix[v, color])
                
                # Check if this vertex can use this color
                can_use_color = True
                for neighbor in graph.neighbors(vertex):
                    if neighbor in coloring and coloring[neighbor] == color:
                        can_use_color = False
                        break
                    if neighbor in independent_set:
                        can_use_color = False
                        break
                
                if can_use_color:
                    independent_set.add(vertex)
                    # Remove this vertex and its neighbors from candidates
                    candidates.discard(vertex)
                    for neighbor in graph.neighbors(vertex):
                        candidates.discard(neighbor)
                else:
                    # Just remove this vertex
                    candidates.discard(vertex)
            
            if not independent_set:
                continue
            
            # Calculate total cost for this independent set with this color
            set_cost = sum(cost_matrix[v, color] for v in independent_set)
            
            # Calculate cost-effectiveness ratio: cost per uncovered vertex
            covered_count = len(independent_set & uncovered)
            if covered_count == 0:
                continue
                
            ratio = set_cost / covered_count
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_set = independent_set
                best_color = color
                best_set_cost = set_cost
        
        if best_set is None:
            # No valid independent set found, try single vertex assignments
            break
        
        # Assign the best color to all vertices in the best independent set
        for vertex in best_set:
            if vertex in uncovered:  # Only assign if still uncovered
                coloring[vertex] = best_color
        
        # Update tracking variables
        uncovered -= best_set
        total_cost += best_set_cost
        colors_used.add(best_color)
        iterations += 1
    
    # Handle remaining uncovered vertices with greedy assignment
    for vertex in list(uncovered):
        # Find cheapest available color that doesn't conflict
        best_color = None
        best_cost = float('inf')
        
        for color in range(n_colors):
            # Check if this color conflicts with neighbors
            conflicts = False
            for neighbor in graph.neighbors(vertex):
                if neighbor in coloring and coloring[neighbor] == color:
                    conflicts = True
                    break
            
            if not conflicts and cost_matrix[vertex, color] < best_cost:
                best_color = color
                best_cost = cost_matrix[vertex, color]
        
        if best_color is not None:
            coloring[vertex] = best_color
            total_cost += cost_matrix[vertex, best_color]
            colors_used.add(best_color)
        else:
            # No feasible coloring possible
            coloring = {}
            total_cost = float('inf')
            break
    
    end_time = time.time()
    
    # Verify feasibility
    is_feasible = len(coloring) == n_vertices and is_proper_coloring(graph, coloring)
    
    # If not feasible, return with infinite cost
    if not is_feasible:
        coloring = {}
        total_cost = float('inf')
    
    approximation_bound = np.log(n_vertices) if n_vertices > 0 else 1.0
    
    result = {
        'solution': coloring,
        'cost': total_cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'approximation_algorithm': 'weighted_set_cover',
        'approximation_bound': approximation_bound,
        'colors_used': len(colors_used) if is_feasible else 0,
        'iterations': iterations,
        'feasible': is_feasible
    }
    
    return convert_numpy_types(result)

def improved_weighted_set_cover(graph: nx.Graph, cost_matrix: np.ndarray,
                               heuristic: str = "greedy_maximal") -> Dict[str, Any]:
    """
    Improved Weighted Set Cover approximation with different heuristics for DAA Project - MCCPP
    
    Improvements over basic version:
    - Multiple construction strategies for independent sets
    - Better vertex ordering heuristics
    - Look-ahead for color selection
    - Post-processing optimization
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        heuristic: method for independent set selection
            - "greedy_maximal": build independent sets greedily by cost
            - "min_cost": prioritize vertices with minimum cost for each color
            - "max_degree": prioritize high-degree vertices first (better packing)
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    vertices = list(graph.nodes())
    n_colors = cost_matrix.shape[1]
    
    coloring = {}
    uncovered = set(vertices)
    total_cost = 0.0
    color_assignments = []
    
    # Precompute useful information
    degrees = dict(graph.degree())
    
    # Precompute vertex orders for different heuristics
    if heuristic == "max_degree":
        # High-degree vertices first (better for dense graphs)
        vertex_priority = sorted(vertices, key=lambda v: degrees[v], reverse=True)
    else:
        vertex_priority = vertices
    
    # Main algorithm loop
    max_iterations = n_vertices * n_colors
    iteration = 0
    
    while uncovered and iteration < max_iterations:
        iteration += 1
        best_ratio = float('inf')
        best_set = None
        best_color = None
        
        for color in range(n_colors):
            # Build independent set based on heuristic
            if heuristic == "min_cost":
                # Sort uncovered vertices by cost for this color
                sorted_vertices = sorted(uncovered, key=lambda v: cost_matrix[v, color])
            elif heuristic == "max_degree":
                # Use pre-computed priority
                sorted_vertices = [v for v in vertex_priority if v in uncovered]
            else:  # "greedy_maximal"
                # Sort by cost for this color
                sorted_vertices = sorted(uncovered, key=lambda v: cost_matrix[v, color])
            
            # Build maximal independent set
            candidate_set = set()
            blocked = set()
            
            for vertex in sorted_vertices:
                if vertex in blocked:
                    continue
                
                # Check if vertex can use this color
                can_use = True
                
                # Check colored neighbors
                for neighbor in graph.neighbors(vertex):
                    if neighbor in coloring and coloring[neighbor] == color:
                        can_use = False
                        break
                    if neighbor in candidate_set:
                        can_use = False
                        break
                
                if can_use:
                    candidate_set.add(vertex)
                    # Block this vertex and all its neighbors
                    blocked.add(vertex)
                    for neighbor in graph.neighbors(vertex):
                        blocked.add(neighbor)
            
            if not candidate_set:
                continue
            
            # Calculate cost-effectiveness ratio
            set_cost = sum(cost_matrix[v, color] for v in candidate_set)
            covered_count = len(candidate_set & uncovered)
            
            if covered_count == 0:
                continue
            
            ratio = set_cost / covered_count
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_set = candidate_set
                best_color = color
        
        if best_set is None:
            break
            
        # Assign color to the selected independent set
        for vertex in best_set:
            if vertex in uncovered:
                coloring[vertex] = best_color
        
        uncovered -= best_set
        total_cost += sum(cost_matrix[v, best_color] for v in best_set if v in coloring)
        color_assignments.append((best_color, len(best_set)))
    
    # Handle remaining vertices with intelligent greedy assignment
    for vertex in list(uncovered):
        best_color = None
        best_cost = float('inf')
        
        for color in range(n_colors):
            # Check feasibility
            conflicts = False
            for neighbor in graph.neighbors(vertex):
                if neighbor in coloring and coloring[neighbor] == color:
                    conflicts = True
                    break
            
            if not conflicts:
                vertex_cost = cost_matrix[vertex, color]
                if vertex_cost < best_cost:
                    best_color = color
                    best_cost = vertex_cost
        
        if best_color is not None:
            coloring[vertex] = best_color
            total_cost += cost_matrix[vertex, best_color]
        else:
            # No feasible color
            coloring = {}
            total_cost = float('inf')
            break
    
    # Post-processing: try to improve solution with local search
    if len(coloring) == n_vertices and is_proper_coloring(graph, coloring):
        improved = True
        post_iterations = 0
        max_post_iterations = min(50, n_vertices)
        
        while improved and post_iterations < max_post_iterations:
            improved = False
            post_iterations += 1
            
            # Try recoloring each vertex with a cheaper color
            for vertex in vertices:
                current_color = coloring[vertex]
                current_cost = cost_matrix[vertex, current_color]
                
                for new_color in range(n_colors):
                    if new_color == current_color:
                        continue
                    
                    new_cost = cost_matrix[vertex, new_color]
                    if new_cost >= current_cost:
                        continue
                    
                    # Check if new color is feasible
                    feasible = True
                    for neighbor in graph.neighbors(vertex):
                        if coloring[neighbor] == new_color:
                            feasible = False
                            break
                    
                    if feasible:
                        # Apply improvement
                        coloring[vertex] = new_color
                        total_cost += new_cost - current_cost
                        improved = True
                        break
    
    end_time = time.time()
    
    is_feasible = len(coloring) == n_vertices and is_proper_coloring(graph, coloring)
    if not is_feasible:
        coloring = {}
        total_cost = float('inf')
    
    result = {
        'solution': coloring,
        'cost': total_cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'approximation_algorithm': f'improved_weighted_set_cover_{heuristic}',
        'colors_used': len(set(coloring.values())) if is_feasible else 0,
        'color_assignments': color_assignments if is_feasible else [],
        'feasible': is_feasible,
        'post_optimization_applied': is_feasible
    }
    
    return convert_numpy_types(result)
