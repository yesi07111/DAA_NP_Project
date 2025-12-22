"""
Cost matrix utilities for DAA Project - MCCPP
"""
import numpy as np
import networkx as nx
from typing import Dict, Tuple
# from utils.graph_utils import is_proper_coloring
from .graph_utils import is_proper_coloring


def generate_cost_matrix(n_vertices: int, n_colors: int, cost_range: Tuple[float, float] = (1, 100), 
                        seed: int = None) -> np.ndarray:
    """
    Generate random cost matrix
    
    Args:
        n_vertices: number of vertices
        n_colors: number of colors/frequencies
        cost_range: tuple of (min_cost, max_cost)
        seed: random seed
    
    Returns:
        n_vertices x n_colors cost matrix
    """
    if seed:
        np.random.seed(seed)
    
    min_cost, max_cost = cost_range
    return np.random.uniform(min_cost, max_cost, size=(n_vertices, n_colors))

def generate_structured_cost_matrix(n_vertices: int, n_colors: int, 
                                  cost_pattern: str = "uniform",
                                  seed: int = None) -> np.ndarray:
    """
    Generate cost matrix with specific patterns for testing
    
    Args:
        n_vertices: number of vertices
        n_colors: number of colors
        cost_pattern: type of cost structure
            - "uniform": random uniform costs
            - "preferential": some colors are generally cheaper
            - "vertex_specific": some vertices have generally lower costs
            - "binary": costs are either 0 or 1 (for MVC reduction tests)
        seed: random seed
    
    Returns:
        structured cost matrix
    """
    if seed:
        np.random.seed(seed)
    
    if cost_pattern == "uniform":
        return generate_cost_matrix(n_vertices, n_colors, (1, 100), seed)
    
    elif cost_pattern == "preferential":
        base_costs = np.linspace(1, 100, n_colors)
        noise = np.random.normal(0, 10, (n_vertices, n_colors))
        costs = base_costs + noise
        return np.clip(costs, 1, 100)
    
    elif cost_pattern == "vertex_specific":
        vertex_factors = np.random.uniform(0.5, 2.0, n_vertices)
        color_factors = np.random.uniform(0.5, 2.0, n_colors)
        base = np.random.uniform(10, 50, (n_vertices, n_colors))
        costs = base * vertex_factors.reshape(-1, 1) * color_factors.reshape(1, -1)
        return costs
    
    elif cost_pattern == "binary":
        # For MVC reduction: first color costs 0, others cost 1
        costs = np.ones((n_vertices, n_colors))
        costs[:, 0] = 0  # First color is "preferred" with zero cost
        return costs
    
    else:
        raise ValueError(f"Unknown cost pattern: {cost_pattern}")

def evaluate_solution(coloring: Dict[int, int], cost_matrix: np.ndarray) -> float:
    """
    Calculate total cost of a coloring solution
    
    Args:
        coloring: dictionary mapping vertex -> color index
        cost_matrix: n_vertices x n_colors cost matrix
    
    Returns:
        total cost of the coloring
    """
    total_cost = 0.0
    for vertex, color in coloring.items():
        total_cost += cost_matrix[vertex, color]
    return total_cost

def validate_solution(graph: nx.Graph, coloring: Dict[int, int], cost_matrix: np.ndarray) -> Tuple[bool, float, str]:
    """
    Comprehensive solution validation
    
    Args:
        graph: networkx Graph
        coloring: proposed coloring
        cost_matrix: cost matrix
    
    Returns:
        tuple of (is_valid, total_cost, error_message)
    """
    # Check if all vertices are colored
    if set(coloring.keys()) != set(graph.nodes()):
        missing = set(graph.nodes()) - set(coloring.keys())
        return False, 0.0, f"Missing coloring for vertices: {missing}"
    
    # Check if coloring is proper
    if not is_proper_coloring(graph, coloring):
        return False, 0.0, "Coloring is not proper (adjacent vertices have same color)"
    
    # Calculate cost
    total_cost = evaluate_solution(coloring, cost_matrix)
    
    return True, total_cost, "Solution is valid"

def get_solution_quality_metrics(optimal_cost: float, achieved_cost: float, 
                               execution_time: float, graph_size: Tuple[int, int]) -> Dict:
    """
    Calculate solution quality metrics
    
    Args:
        optimal_cost: known optimal cost (if available)
        achieved_cost: cost achieved by algorithm
        execution_time: algorithm runtime
        graph_size: tuple of (n_vertices, n_edges)
    
    Returns:
        dictionary of quality metrics
    """
    metrics = {
        'achieved_cost': achieved_cost,
        'execution_time': execution_time,
        'n_vertices': graph_size[0],
        'n_edges': graph_size[1]
    }
    
    if optimal_cost is not None and optimal_cost > 0:
        metrics['approximation_ratio'] = achieved_cost / optimal_cost
        metrics['optimality_gap'] = (achieved_cost - optimal_cost) / optimal_cost * 100
    
    return metrics