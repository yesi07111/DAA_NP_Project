"""
Integer Linear Programming solver for DAA Project - MCCPP
"""
import time
import numpy as np
import networkx as nx
from typing import Dict
# from src.utils.io_utils import convert_numpy_types
from ...utils.io_utils import convert_numpy_types


try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

def ilp_solver(graph: nx.Graph, cost_matrix: np.ndarray, 
               time_limit: float = None) -> Dict[str, any]:
    """
    ILP solver using PuLP for DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        time_limit: maximum time to run in seconds
    
    Returns:
        dictionary with solution and metrics
    """
    if not PULP_AVAILABLE:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': 0,
            'optimal': False,
            'error': 'PuLP not installed'
        }
    
    start_time = time.time()
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())
    
    # Create the problem
    prob = pulp.LpProblem("DAA_Project_MCCPP", pulp.LpMinimize)
    
    # Decision variables: x_{v,f} = 1 if vertex v is assigned color f
    x = pulp.LpVariable.dicts("x", 
                             [(v, f) for v in vertices for f in range(n_colors)],
                             cat='Binary')
    
    # Objective function: minimize total cost
    prob += pulp.lpSum(cost_matrix[v, f] * x[(v, f)] for v in vertices for f in range(n_colors))
    
    # Constraints: each vertex must be assigned exactly one color
    for v in vertices:
        prob += pulp.lpSum(x[(v, f)] for f in range(n_colors)) == 1
    
    # Constraints: no two adjacent vertices have the same color
    for u, v in graph.edges():
        for f in range(n_colors):
            prob += x[(u, f)] + x[(v, f)] <= 1
    
    # Solve the problem with time limit or without
    if time_limit is None:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    else:
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
    
    end_time = time.time()
    
    # Extract solution
    coloring = {}
    for v in vertices:
        for f in range(n_colors):
            if pulp.value(x[(v, f)]) == 1:
                coloring[v] = f
                break
    
    cost = pulp.value(prob.objective) if prob.status == pulp.LpStatusOptimal else float('inf')
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': (prob.status == pulp.LpStatusOptimal),
        'status': pulp.LpStatus[prob.status],
        'feasible': (prob.status == pulp.LpStatusOptimal)
    }
    
    return convert_numpy_types(result)