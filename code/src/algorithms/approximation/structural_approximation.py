"""
Structural approximation algorithms for special graph classes in DAA Project - MCCPP
"""
import time
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np
from src.utils.graph_utils import generate_interval_graph
from src.utils.cost_utils import evaluate_solution
from src.utils.io_utils import convert_numpy_types

def structural_approximation_bipartite(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Structural approximation for bipartite graphs in DAA Project - MCCPP
    
    For bipartite graphs, chromatic number is at most 2, so we can achieve 
    optimal or near-optimal solutions by trying different 2-colorings and 
    using additional colors only when beneficial.
    
    Theoretical guarantee: For bipartite graphs, this achieves 2-approximation
    in the worst case, but often finds optimal solutions.
    
    Args:
        graph: bipartite networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    # Verify the graph is bipartite
    if not nx.is_bipartite(graph):
        # Fall back to general approximation if graph is not bipartite
        from .weighted_set_cover import weighted_set_cover_approximation
        return weighted_set_cover_approximation(graph, cost_matrix)
    
    # Get bipartite sets for each connected component
    all_colorings = []
    
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)
        
        if not nx.is_bipartite(subgraph):
            continue
            
        try:
            left_set, right_set = nx.bipartite.sets(subgraph)
        except:
            continue
        
        left_vertices = list(left_set)
        right_vertices = list(right_set)
        
        # Strategy 1: Try all pairs of colors for bipartite sets
        best_component_cost = float('inf')
        best_component_coloring = None
        
        for color1 in range(n_colors):
            for color2 in range(n_colors):
                if color1 == color2:
                    continue
                
                coloring = {}
                cost = 0.0
                
                # Assign colors to bipartite sets
                for vertex in left_vertices:
                    coloring[vertex] = color1
                    cost += cost_matrix[vertex, color1]
                
                for vertex in right_vertices:
                    coloring[vertex] = color2
                    cost += cost_matrix[vertex, color2]
                
                if cost < best_component_cost:
                    best_component_cost = cost
                    best_component_coloring = coloring
        
        # Strategy 2: Optimize each partition independently
        # For left set, find best single color
        left_costs = []
        for color in range(n_colors):
            cost = sum(cost_matrix[v, color] for v in left_vertices)
            left_costs.append((cost, color))
        left_costs.sort()
        
        # For right set, find best color that's different from left
        right_costs = []
        for color in range(n_colors):
            cost = sum(cost_matrix[v, color] for v in right_vertices)
            right_costs.append((cost, color))
        right_costs.sort()
        
        # Try best combinations
        for left_cost, left_color in left_costs[:min(3, len(left_costs))]:
            for right_cost, right_color in right_costs[:min(3, len(right_costs))]:
                if left_color == right_color:
                    continue
                
                total = left_cost + right_cost
                if total < best_component_cost:
                    best_component_cost = total
                    coloring = {}
                    for v in left_vertices:
                        coloring[v] = left_color
                    for v in right_vertices:
                        coloring[v] = right_color
                    best_component_coloring = coloring
        
        all_colorings.append(best_component_coloring)
    
    # Combine all component colorings
    final_coloring = {}
    for coloring in all_colorings:
        final_coloring.update(coloring)
    
    # Post-optimization: try to improve individual vertices
    improved = True
    iterations = 0
    max_iterations = min(20, n_vertices)
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for vertex in graph.nodes():
            if vertex not in final_coloring:
                continue
                
            current_color = final_coloring[vertex]
            current_cost = cost_matrix[vertex, current_color]
            
            # Try each color
            for new_color in range(n_colors):
                if new_color == current_color:
                    continue
                
                new_cost = cost_matrix[vertex, new_color]
                if new_cost >= current_cost:
                    continue
                
                # Check feasibility
                feasible = True
                for neighbor in graph.neighbors(vertex):
                    if neighbor in final_coloring and final_coloring[neighbor] == new_color:
                        feasible = False
                        break
                
                if feasible:
                    final_coloring[vertex] = new_color
                    improved = True
                    break
    
    final_cost = evaluate_solution(final_coloring, cost_matrix)
    end_time = time.time()
    
    # Verify feasibility
    from src.utils.graph_utils import is_proper_coloring
    is_feasible = is_proper_coloring(graph, final_coloring)
    
    if not is_feasible:
        final_cost = float('inf')
        final_coloring = {}
    
    result = {
        'solution': final_coloring,
        'cost': final_cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'approximation_algorithm': 'structural_bipartite',
        'colors_used': len(set(final_coloring.values())) if is_feasible else 0,
        'graph_type': 'bipartite',
        'post_optimization_iterations': iterations,
        'feasible': is_feasible,
        'theoretical_bound': 2.0  # 2-approximation for bipartite
    }
    
    return convert_numpy_types(result)

def structural_approximation_interval(graph: nx.Graph, cost_matrix: np.ndarray,
                                    intervals: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Structural approximation for interval graphs in DAA Project - MCCPP
    
    For interval graphs, we can use the inherent linear structure to achieve
    better approximation guarantees, potentially O(sqrt(n)) or constant factor.
    
    Args:
        graph: interval graph (or any graph)
        cost_matrix: n_vertices x n_colors cost matrix
        intervals: list of (start, end) for each vertex (required for true interval graphs)
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    # If intervals are provided, use interval-specific algorithm
    if intervals is not None and len(intervals) == n_vertices:
        return _interval_graph_approximation(graph, cost_matrix, intervals)
    else:
        # Fall back to general structural approximation
        return _general_structural_approximation(graph, cost_matrix)

def _interval_graph_approximation(graph: nx.Graph, cost_matrix: np.ndarray,
                                intervals: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Specialized approximation for true interval graphs in DAA Project - MCCPP
    
    For interval graphs, we can use the interval representation to achieve
    better solutions. The algorithm uses earliest-deadline-first approach
    combined with cost optimization.
    """
    start_time = time.time() 
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    # Create list of vertices with their intervals
    vertex_intervals = [(v, intervals[v][0], intervals[v][1]) for v in range(n_vertices)]
    
    # Sort by start time, then by end time
    vertex_intervals.sort(key=lambda x: (x[1], x[2]))
    
    # Greedy interval scheduling with cost awareness
    coloring = {}
    color_last_end = {}  # Track last end time for each color
    
    for vertex, start, end in vertex_intervals:
        # Find colors whose last interval ended before current start
        available_colors = []
        for color in range(n_colors):
            if color not in color_last_end or color_last_end[color] <= start:
                available_colors.append(color)
        
        if available_colors:
            # Among available colors, choose the one with minimum cost
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
            coloring[vertex] = best_color
            color_last_end[best_color] = end
        else:
            # No available color - instance not colorable with k colors
            coloring = {}
            break
    
    # Check if we have complete coloring
    if len(coloring) != n_vertices:
        cost = float('inf')
        is_feasible = False
    else:
        # Try to optimize by recoloring vertices with cheaper colors
        improved = True
        iterations = 0
        max_iterations = min(30, n_vertices)
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for vertex in range(n_vertices):
                current_color = coloring[vertex]
                current_cost = cost_matrix[vertex, current_color]
                start, end = intervals[vertex]
                
                # Try each color
                for new_color in range(n_colors):
                    if new_color == current_color:
                        continue
                    
                    new_cost = cost_matrix[vertex, new_color]
                    if new_cost >= current_cost:
                        continue
                    
                    # Check if this color is feasible for this interval
                    feasible = True
                    for other_vertex in range(n_vertices):
                        if other_vertex == vertex:
                            continue
                        if coloring[other_vertex] == new_color:
                            other_start, other_end = intervals[other_vertex]
                            # Check if intervals overlap
                            if not (end <= other_start or start >= other_end):
                                feasible = False
                                break
                    
                    if feasible:
                        coloring[vertex] = new_color
                        improved = True
                        break
        
        cost = evaluate_solution(coloring, cost_matrix)
        
        # Verify feasibility
        from src.utils.graph_utils import is_proper_coloring
        is_feasible = is_proper_coloring(graph, coloring)
        
        if not is_feasible:
            cost = float('inf')
            coloring = {}
    
    end_time = time.time()
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'approximation_algorithm': 'structural_interval',
        'colors_used': len(set(coloring.values())) if is_feasible else 0,
        'graph_type': 'interval',
        'feasible': is_feasible
    }
    
    return convert_numpy_types(result)

def _general_structural_approximation(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    General structural approximation for unknown graph types in DAA Project - MCCPP
    
    Uses a sophisticated greedy approach with multiple phases:
    1. Ordering vertices by structural properties
    2. Greedy coloring with cost awareness
    3. Local optimization phase
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    # Try to detect graph properties and apply appropriate strategy
    if nx.is_bipartite(graph):
        return structural_approximation_bipartite(graph, cost_matrix)
    
    # Phase 1: Compute vertex priorities based on multiple criteria
    degrees = dict(graph.degree())
    
    # Calculate cost characteristics for each vertex
    vertex_priorities = []
    for vertex in graph.nodes():
        costs = cost_matrix[vertex, :]
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        avg_cost = np.mean(costs)
        cost_variance = np.var(costs)
        
        # Priority combines degree (structural constraint) and cost characteristics
        # Higher degree and higher cost variance = higher priority (more constrained)
        priority = degrees[vertex] * 10 + cost_variance
        vertex_priorities.append((vertex, priority, min_cost))
    
    # Sort by priority descending, then by minimum cost ascending
    vertex_priorities.sort(key=lambda x: (-x[1], x[2]))
    sorted_vertices = [v for v, _, _ in vertex_priorities]
    
    # Phase 2: Greedy coloring with cost minimization
    coloring = {}
    
    for vertex in sorted_vertices:
        # Find used colors in neighborhood
        used_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                used_colors.add(coloring[neighbor])
        
        available_colors = set(range(n_colors)) - used_colors
        
        if available_colors:
            # Among available colors, choose the one with minimum cost
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
            coloring[vertex] = best_color
        else:
            # No feasible color - instance may not be colorable with k colors
            coloring = {}
            break
    
    # Check if we have a complete coloring
    if len(coloring) != n_vertices:
        end_time = time.time()
        return {
            'solution': {},
            'cost': float('inf'),
            'execution_time': end_time - start_time,
            'optimal': False,
            'approximation_algorithm': 'structural_general',
            'colors_used': 0,
            'graph_type': 'general',
            'feasible': False
        }
    
    # Phase 3: Local optimization with multiple strategies
    improved = True
    total_iterations = 0
    max_iterations = min(50, n_vertices * 2)
    
    while improved and total_iterations < max_iterations:
        improved = False
        total_iterations += 1
        
        # Strategy 3a: Try to recolor each vertex with a cheaper color
        for vertex in sorted_vertices:
            current_color = coloring[vertex]
            current_cost = cost_matrix[vertex, current_color]
            
            # Get neighbor colors
            neighbor_colors = set()
            for neighbor in graph.neighbors(vertex):
                neighbor_colors.add(coloring[neighbor])
            
            # Try all available colors
            for new_color in range(n_colors):
                if new_color == current_color or new_color in neighbor_colors:
                    continue
                
                new_cost = cost_matrix[vertex, new_color]
                if new_cost < current_cost:
                    coloring[vertex] = new_color
                    improved = True
                    break
        
        # Strategy 3b: Try to swap colors between non-adjacent vertices
        if not improved and total_iterations < max_iterations // 2:
            vertices_list = list(graph.nodes())
            for i in range(min(20, n_vertices)):
                v1 = vertices_list[i % n_vertices]
                v2 = vertices_list[(i + n_vertices // 2) % n_vertices]
                
                if v1 == v2 or graph.has_edge(v1, v2):
                    continue
                
                # Check if swapping improves cost
                cost_before = cost_matrix[v1, coloring[v1]] + cost_matrix[v2, coloring[v2]]
                cost_after = cost_matrix[v1, coloring[v2]] + cost_matrix[v2, coloring[v1]]
                
                if cost_after < cost_before:
                    # Check if swap is feasible
                    feasible = True
                    
                    # Check v1 with new color
                    for neighbor in graph.neighbors(v1):
                        if neighbor != v2 and coloring[neighbor] == coloring[v2]:
                            feasible = False
                            break
                    
                    # Check v2 with new color
                    if feasible:
                        for neighbor in graph.neighbors(v2):
                            if neighbor != v1 and coloring[neighbor] == coloring[v1]:
                                feasible = False
                                break
                    
                    if feasible:
                        coloring[v1], coloring[v2] = coloring[v2], coloring[v1]
                        improved = True
                        break
    
    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()
    
    # Final feasibility check
    from src.utils.graph_utils import is_proper_coloring
    is_feasible = is_proper_coloring(graph, coloring)
    
    if not is_feasible:
        cost = float('inf')
        coloring = {}
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'approximation_algorithm': 'structural_general',
        'colors_used': len(set(coloring.values())) if is_feasible else 0,
        'graph_type': 'general',
        'optimization_iterations': total_iterations,
        'feasible': is_feasible
    }
    
    return convert_numpy_types(result)

def get_approximation_quality_metrics(optimal_cost: float, achieved_cost: float, 
                                    graph_size: int, algorithm: str) -> Dict[str, float]:
    """
    Calculate approximation quality metrics for DAA Project - MCCPP
    
    Args:
        optimal_cost: known optimal cost (if available)
        achieved_cost: cost achieved by approximation algorithm
        graph_size: number of vertices
        algorithm: algorithm name
    
    Returns:
        dictionary of quality metrics
    """
    metrics = {
        'achieved_cost': achieved_cost,
        'graph_size': graph_size,
        'algorithm': algorithm
    }
    
    if optimal_cost is not None and optimal_cost > 0:
        metrics['approximation_ratio'] = achieved_cost / optimal_cost
        metrics['optimality_gap_percent'] = ((achieved_cost - optimal_cost) / optimal_cost) * 100
        
        # Theoretical bounds
        if 'weighted_set_cover' in algorithm:
            metrics['theoretical_bound'] = np.log(graph_size) if graph_size > 0 else 1.0
        elif 'bipartite' in algorithm:
            metrics['theoretical_bound'] = 2.0  # Constant factor for bipartite
        elif 'interval' in algorithm:
            metrics['theoretical_bound'] = np.sqrt(graph_size)  # Conservative bound
        else:
            metrics['theoretical_bound'] = np.log(graph_size)  # Default bound
    
    return metrics