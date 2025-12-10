"""
Simulated Annealing metaheuristic for DAA Project - MCCPP
"""
import time
import math
import random
from typing import Dict, List, Set, Any, Tuple, Callable
import networkx as nx
import numpy as np
from src.utils.graph_utils import is_proper_coloring
from src.utils.cost_utils import evaluate_solution

def simulated_annealing(graph: nx.Graph, cost_matrix: np.ndarray,
                       initial_solution: Dict[int, int] = None,
                       initial_temperature: float = 1000.0,
                       cooling_rate: float = 0.95,
                       min_temperature: float = 1e-6,
                       max_iterations: int = 10000,
                       max_non_improving: int = 1000,
                       penalty_weight: float = 1000.0,
                       seed: int = None) -> Dict[str, Any]:
    """
    Simulated Annealing metaheuristic for DAA Project - MCCPP.
    
    Inspired by the metallurgical process of annealing, this algorithm explores
    the solution space by accepting worse solutions with a probability that
    decreases over time, allowing it to escape local optima.
    
    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        initial_solution: starting solution (if None, generates one)
        initial_temperature: starting temperature
        cooling_rate: geometric cooling rate (0 < rate < 1)
        min_temperature: minimum temperature to stop
        max_iterations: maximum number of iterations
        max_non_improving: maximum iterations without improvement
        penalty_weight: weight for constraint violations in objective
        seed: random seed for reproducibility
    
    Returns:
        dictionary with solution and metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    # Generate initial solution if not provided
    if initial_solution is None:
        from src.algorithms.heuristic.dsatur import dsatur_heuristic
        initial_result = dsatur_heuristic(graph, cost_matrix)
        current_solution = initial_result['solution']
    else:
        current_solution = initial_solution.copy()
    
    # Evaluate initial solution
    current_cost = _evaluate_solution_with_penalties(graph, current_solution, cost_matrix, penalty_weight)
    best_solution = current_solution.copy()
    best_cost = current_cost

    # Check if initial solution is valid
    if not is_proper_coloring(graph, current_solution):
        # Try to fix the initial solution
        for vertex in graph.nodes():
            available_colors = set(range(n_colors))
            for neighbor in graph.neighbors(vertex):
                if neighbor in current_solution:
                    available_colors.discard(current_solution[neighbor])
            if available_colors:
                current_solution[vertex] = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        
        # Re-evaluate
        current_cost = _evaluate_solution_with_penalties(graph, current_solution, cost_matrix, penalty_weight)
        best_solution = current_solution.copy()
        best_cost = current_cost
    
    # Initialize tracking variables
    temperature = initial_temperature
    iteration = 0
    non_improving_count = 0
    accepted_worse = 0
    accepted_better = 0
    cost_history = [current_cost]
    temperature_history = [temperature]
    
    # Main SA loop
    while (temperature > min_temperature and 
           iteration < max_iterations and 
           non_improving_count < max_non_improving):
        
        iteration += 1
        
        # Generate neighbor solution
        neighbor_solution = _generate_neighbor(graph, current_solution, n_colors)
        neighbor_cost = _evaluate_solution_with_penalties(graph, neighbor_solution, cost_matrix, penalty_weight)
        
        # Calculate cost difference
        cost_delta = neighbor_cost - current_cost
        
        # Acceptance criterion
        if cost_delta < 0:
            # Always accept improving moves
            current_solution = neighbor_solution
            current_cost = neighbor_cost
            accepted_better += 1
            
            # Update best solution
            if neighbor_cost < best_cost:
                best_solution = neighbor_solution.copy()
                best_cost = neighbor_cost
                non_improving_count = 0
            else:
                non_improving_count += 1
                
        else:
            # Accept worse moves with probability exp(-ΔC / T)
            acceptance_prob = math.exp(-cost_delta / temperature)
            if random.random() < acceptance_prob:
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                accepted_worse += 1
                non_improving_count += 1
            else:
                non_improving_count += 1
        
        # Cooling schedule
        temperature *= cooling_rate
        
        # Record history
        cost_history.append(current_cost)
        temperature_history.append(temperature)
    
    # Calculate actual cost without penalties
    actual_cost = evaluate_solution(best_solution, cost_matrix)
    is_feasible = is_proper_coloring(graph, best_solution)
    
    end_time = time.time()
    
    result = {
        'solution': best_solution,
        'cost': actual_cost,
        'feasible': is_feasible,
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': 'simulated_annealing',
        'iterations': iteration,
        'final_temperature': temperature,
        'accepted_better_moves': accepted_better,
        'accepted_worse_moves': accepted_worse,
        'cooling_rate': cooling_rate,
        'cost_history': cost_history,
        'temperature_history': temperature_history,
        'penalty_weight_used': penalty_weight,
    }
    
    return result

def _evaluate_solution_with_penalties(graph: nx.Graph, coloring: Dict[int, int],
                                    cost_matrix: np.ndarray, penalty_weight: float) -> float:
    """
    Evaluate solution with penalties for constraint violations for DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        coloring: proposed coloring
        cost_matrix: cost matrix
        penalty_weight: weight for constraint violations
    
    Returns:
        penalized cost value
    """
    # Base cost
    base_cost = evaluate_solution(coloring, cost_matrix)
    
    # Penalty for constraint violations
    penalty = 0
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            penalty += penalty_weight
    
    return base_cost + penalty

def _generate_neighbor(graph: nx.Graph, current_solution: Dict[int, int],
                      n_colors: int) -> Dict[int, int]:
    """
    Generate a neighbor solution for Simulated Annealing in DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        current_solution: current coloring
        n_colors: number of available colors
    
    Returns:
        neighbor solution
    """
    neighbor = current_solution.copy()
    
    # Choose a random vertex
    vertex = random.choice(list(graph.nodes()))
    
    # Choose a mutation strategy
    strategy = random.choice(["recolor", "swap", "conflict_resolution"])
    
    if strategy == "recolor":
        # Simple recolor: assign a random color
        new_color = random.randint(0, n_colors - 1)
        neighbor[vertex] = new_color
        
    elif strategy == "swap":
        # Swap colors between two vertices (if they have different colors)
        vertices = list(graph.nodes())
        v1, v2 = random.sample(vertices, 2)
        if neighbor[v1] != neighbor[v2]:
            neighbor[v1], neighbor[v2] = neighbor[v2], neighbor[v1]
    
    elif strategy == "conflict_resolution":
        # Focus on resolving conflicts: if vertex has conflicts, try to fix them
        conflicting = False
        for neighbor_vertex in graph.neighbors(vertex):
            if neighbor[vertex] == neighbor[neighbor_vertex]:
                conflicting = True
                break
        
        if conflicting:
            # Find a color that minimizes conflicts
            neighbor_colors = set()
            for neighbor_vertex in graph.neighbors(vertex):
                neighbor_colors.add(neighbor[neighbor_vertex])
            
            available_colors = set(range(n_colors)) - neighbor_colors
            if available_colors:
                neighbor[vertex] = random.choice(list(available_colors))
            else:
                # If no conflict-free color, choose randomly
                neighbor[vertex] = random.randint(0, n_colors - 1)
    
    return neighbor

def adaptive_simulated_annealing(graph: nx.Graph, cost_matrix: np.ndarray,
                                adaptive_cooling: bool = True,
                                restart_strategy: bool = True,
                                seed: int = None) -> Dict[str, Any]:
    """
    Adaptive Simulated Annealing with enhanced strategies for DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        cost_matrix: cost matrix
        adaptive_cooling: whether to use adaptive cooling schedule
        restart_strategy: whether to use restart strategy
        seed: random seed
    
    Returns:
        dictionary with solution and metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    
    # Generate multiple initial solutions using different heuristics
    from src.algorithms.heuristic.largest_first import largest_first_heuristic
    from src.algorithms.heuristic.dsatur import dsatur_heuristic
    from src.algorithms.heuristic.recursive_largest_first import recursive_largest_first_heuristic
    
    initial_solutions = []
    initial_solutions.append(largest_first_heuristic(graph, cost_matrix)['solution'])
    initial_solutions.append(dsatur_heuristic(graph, cost_matrix)['solution'])
    initial_solutions.append(recursive_largest_first_heuristic(graph, cost_matrix)['solution'])
    
    best_global_solution = None
    best_global_cost = float('inf')
    all_results = []
    
    # Try SA with different initial solutions
    for i, initial_sol in enumerate(initial_solutions):
        if adaptive_cooling:
            # Adaptive parameters based on problem size
            initial_temp = 100 * n_vertices
            cooling_rate = 0.99 if n_vertices > 100 else 0.95
        else:
            initial_temp = 1000.0
            cooling_rate = 0.95
        
        sa_result = simulated_annealing(
            graph=graph,
            cost_matrix=cost_matrix,
            initial_solution=initial_sol,
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            max_iterations=5000,
            max_non_improving=500,
            penalty_weight=1000.0,
            seed=seed + i if seed else None
        )
        
        all_results.append(sa_result)
        
        if sa_result['cost'] < best_global_cost:
            best_global_cost = sa_result['cost']
            best_global_solution = sa_result['solution']
    
    # If restart strategy is enabled, do additional runs from the best solution
    if restart_strategy and best_global_solution:
        for restart in range(2):  # 2 additional restarts
            sa_result = simulated_annealing(
                graph=graph,
                cost_matrix=cost_matrix,
                initial_solution=best_global_solution,
                initial_temperature=500.0,  # Lower temperature for intensification
                cooling_rate=0.98,
                max_iterations=2000,
                max_non_improving=200,
                penalty_weight=1000.0,
                seed=seed + 100 + restart if seed else None
            )
            
            all_results.append(sa_result)
            
            if sa_result['cost'] < best_global_cost:
                best_global_cost = sa_result['cost']
                best_global_solution = sa_result['solution']
    
    end_time = time.time()
    
    # Collect statistics from all runs
    costs = [r['cost'] for r in all_results]
    times = [r['execution_time'] for r in all_results]
    
    result = {
        'solution': best_global_solution,
        'cost': best_global_cost,
        'feasible': is_proper_coloring(graph, best_global_solution),
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': 'adaptive_simulated_annealing',
        'total_runs': len(all_results),
        'min_cost': min(costs),
        'max_cost': max(costs),
        'avg_cost': sum(costs) / len(costs),
        'total_time': sum(times),
        'adaptive_cooling': adaptive_cooling,
        'restart_strategy': restart_strategy
    }
    
    return result