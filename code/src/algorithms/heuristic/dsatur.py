"""
DSATUR (Degree of Saturation) Cost-Aware heuristic for DAA Project - MCCPP
"""
import time
import heapq
from typing import Dict, Any
import numpy as np
import networkx as nx
from src.utils.io_utils import convert_numpy_types
from src.utils.cost_utils import evaluate_solution
from src.utils.graph_utils import calculate_degrees

class DSATURVertex:
    """Helper class to manage vertex properties for DSATUR"""
    def __init__(self, vertex_id: int, degree: int):
        self.vertex_id = vertex_id
        self.saturation = 0  # number of different colors in neighborhood
        self.degree = degree
        self.colored = False
    
    def __lt__(self, other):
        # For max-heap: higher saturation first, then higher degree
        if self.saturation != other.saturation:
            return self.saturation > other.saturation
        else:
            return self.degree > other.degree

def dsatur_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    DSATUR (Degree of Saturation) Cost-Aware heuristic for DAA Project - MCCPP.
    
    Dynamically selects the vertex with the highest saturation degree (number of
    different colors in its neighborhood) and assigns the cheapest available color.
    This heuristic typically produces better solutions than LF due to its dynamic
    prioritization of constrained vertices.
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())
    
    # Initialize DSATUR data structures
    dsatur_vertices = {}
    degrees = calculate_degrees(graph)
    
    for vertex in vertices:
        dsatur_vertices[vertex] = DSATURVertex(vertex, degrees[vertex])
    
    # Use a max-heap (simulated with negative values in min-heap)
    heap = []
    for vertex in vertices:
        heapq.heappush(heap, (-dsatur_vertices[vertex].saturation, -degrees[vertex], vertex))
    
    coloring = {}
    color_adjacency = {v: set() for v in vertices}  # Track which colors are adjacent to each vertex
    
    # Main DSATUR loop
    while heap:
        # Get vertex with maximum saturation (and degree as tie-breaker)
        _, _, current_vertex = heapq.heappop(heap)
        vertex_obj = dsatur_vertices[current_vertex]
        
        if vertex_obj.colored:
            continue
        
        # Get available colors (colors not in adjacent colors set)
        available_colors = set(range(n_colors)) - color_adjacency[current_vertex]
        
        if available_colors:
            # Choose available color with minimum cost
            best_color = min(available_colors, key=lambda c: cost_matrix[current_vertex, c])
        else:
            # If no available color, use minimum cost color
            best_color = np.argmin(cost_matrix[current_vertex, :])
        
        # Assign color
        coloring[current_vertex] = best_color
        vertex_obj.colored = True
        
        # Update saturation degrees of neighbors
        for neighbor in graph.neighbors(current_vertex):
            if not dsatur_vertices[neighbor].colored:
                # Add this color to neighbor's adjacent colors
                color_adjacency[neighbor].add(best_color)
                # Update saturation degree
                old_saturation = dsatur_vertices[neighbor].saturation
                dsatur_vertices[neighbor].saturation = len(color_adjacency[neighbor])
                
                # If saturation changed, push updated vertex to heap
                if dsatur_vertices[neighbor].saturation != old_saturation:
                    heapq.heappush(heap, (-dsatur_vertices[neighbor].saturation, 
                                         -dsatur_vertices[neighbor].degree, 
                                         neighbor))
    
    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': 'dsatur_cost_aware',
        'colors_used': len(set(coloring.values())),
        'max_saturation_used': max([dsatur_vertices[v].saturation for v in vertices]) if vertices else 0
    }
    
    return convert_numpy_types(result)


def enhanced_dsatur(graph: nx.Graph, cost_matrix: np.ndarray,
                   tie_breaking: str = "degree") -> Dict[str, Any]:
    """
    Enhanced DSATUR with improved tie-breaking and cost considerations for DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        tie_breaking: method for breaking ties when saturation is equal
            - "degree": use vertex degree
            - "cost_sensitivity": use cost variance or sensitivity
            - "composite": combination of degree and cost factors
    
    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())
    
    # Initialize data structures
    coloring = {}
    color_adjacency = {v: set() for v in vertices}
    degrees = calculate_degrees(graph)
    
    # Calculate additional metrics for tie-breaking
    if tie_breaking == "cost_sensitivity":
        cost_variances = {v: np.var(cost_matrix[v, :]) for v in vertices}
    elif tie_breaking == "composite":
        cost_variances = {v: np.var(cost_matrix[v, :]) for v in vertices}
        avg_costs = {v: np.mean(cost_matrix[v, :]) for v in vertices}
    
    uncolored = set(vertices)
    
    while uncolored:
        # Find uncolored vertex with maximum saturation
        max_saturation = -1
        candidates = []
        
        for vertex in uncolored:
            saturation = len(color_adjacency[vertex])
            if saturation > max_saturation:
                max_saturation = saturation
                candidates = [vertex]
            elif saturation == max_saturation:
                candidates.append(vertex)
        
        # Tie-breaking
        if tie_breaking == "degree":
            current_vertex = max(candidates, key=lambda v: degrees[v])
        elif tie_breaking == "cost_sensitivity":
            current_vertex = max(candidates, key=lambda v: cost_variances[v])
        elif tie_breaking == "composite":
            # Combine degree and cost factors
            def composite_score(v):
                return degrees[v] * cost_variances[v] * avg_costs[v]
            current_vertex = max(candidates, key=composite_score)
        else:
            current_vertex = max(candidates, key=lambda v: degrees[v])
        
        # Choose color for current vertex
        available_colors = set(range(n_colors)) - color_adjacency[current_vertex]
        
        if available_colors:
            # Enhanced color selection: consider both immediate cost and potential impact
            def color_score(color):
                base_cost = cost_matrix[current_vertex, color]
                # Consider how many uncolored neighbors could still use this color
                neighbor_impact = 0
                for neighbor in graph.neighbors(current_vertex):
                    if neighbor in uncolored:
                        # If neighbor has high cost for this color, it's less likely to use it
                        neighbor_cost = cost_matrix[neighbor, color]
                        neighbor_impact += neighbor_cost
                return base_cost + (neighbor_impact * 0.1)  # Weighted impact
            
            best_color = min(available_colors, key=color_score)
        else:
            best_color = np.argmin(cost_matrix[current_vertex, :])
        
        coloring[current_vertex] = best_color
        uncolored.remove(current_vertex)
        
        # Update color adjacency for neighbors
        for neighbor in graph.neighbors(current_vertex):
            if neighbor in uncolored:
                color_adjacency[neighbor].add(best_color)
    
    cost = evaluate_solution(coloring, cost_matrix)
    end_time = time.time()
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': f'enhanced_dsatur_{tie_breaking}',
        'colors_used': len(set(coloring.values())),
        'tie_breaking_method': tie_breaking
    }
    
    return convert_numpy_types(result)