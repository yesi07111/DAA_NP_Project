import time
import random
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, List, Any, Deque
from src.utils.graph_utils import is_proper_coloring
from src.utils.cost_utils import evaluate_solution
from src.algorithms.metaheuristic.simulated_annealing import (
    simulated_annealing,
    adaptive_simulated_annealing,
)
from src.algorithms.metaheuristic.trajectory_search import trajectory_search_heuristic

def hybrid_metaheuristic(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    method: str = "auto",
    time_limit: float = 10000.0,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Hybrid metaheuristic that adapts based on problem characteristics for DAA Project - MCCPP

    Args:
        graph: networkx Graph
        cost_matrix: cost matrix
        method: metaheuristic method or "auto" for automatic selection
        time_limit: maximum execution time in seconds
        seed: random seed

    Returns:
        dictionary with solution and metrics
    """
    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    density = nx.density(graph)

    if method == "auto":
        # Choose method based on problem characteristics
        if n_vertices <= 100:
            # For small instances, use TSH+PR for best quality
            method = "tsh_pr"
        elif density < 0.2:
            # For sparse graphs, SA works well
            method = "sa"
        else:
            # For large or dense instances, use adaptive SA for speed
            method = "adaptive_sa"

    # Apply time limit to each method
    remaining_time = time_limit

    if method == "tsh_pr" or method == "trajectory_search":
        result = trajectory_search_heuristic(
            graph,
            cost_matrix,
            max_iterations=min(2000, int(remaining_time * 10)),
            seed=seed,
        )
    elif method == "sa" or method == "simulated_annealing":
        result = simulated_annealing(
            graph,
            cost_matrix,
            max_iterations=min(10000, int(remaining_time * 20)),
            seed=seed,
        )
    elif method == "adaptive_sa":
        result = adaptive_simulated_annealing(graph, cost_matrix, seed=seed)
    else:
        # Default to trajectory search
        result = trajectory_search_heuristic(graph, cost_matrix, seed=seed)

    # Ensure we respect time limit
    if time.time() - start_time > time_limit:
        result["time_limit_exceeded"] = True
    else:
        result["time_limit_exceeded"] = False

    result["algorithm"] = f"hybrid_metaheuristic_{method}"
    result["total_time"] = time.time() - start_time

    return result

def adaptive_metaheuristic(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    strategies: List[str] = None,
    time_budget: float = 10000.0,
    seed: int = None,
) -> Dict[str, Any]:
    """
    Adaptive metaheuristic that tries multiple strategies within time budget for DAA Project - MCCPP

    Args:
        graph: networkx Graph
        cost_matrix: cost matrix
        strategies: list of strategies to try
        time_budget: total time budget in seconds
        seed: random seed

    Returns:
        dictionary with solution and metrics
    """
    if strategies is None:
        strategies = ["trajectory_search", "adaptive_sa", "hybrid"]

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()
    best_solution = None
    best_cost = float("inf")
    all_results = []

    # Distribute time among strategies
    time_per_strategy = time_budget / len(strategies)

    for i, strategy in enumerate(strategies):
        strategy_start = time.time()

        if strategy == "trajectory_search":
            result = trajectory_search_heuristic(
                graph, cost_matrix, max_iterations=1000, seed=seed + i if seed else None
            )
        elif strategy == "adaptive_sa":
            result = adaptive_simulated_annealing(
                graph, cost_matrix, seed=seed + i if seed else None
            )
        elif strategy == "hybrid":
            result = hybrid_metaheuristic(
                graph,
                cost_matrix,
                method="auto",
                time_limit=time_per_strategy * 0.8,
                seed=seed + i if seed else None,
            )
        else:
            continue

        if result.get('feasible', False):
            all_results.append(result)
            
            if result["cost"] < best_cost:
                best_cost = result["cost"]
                best_solution = result["solution"]

        # Adjust remaining time
        elapsed = time.time() - strategy_start
        if elapsed < time_per_strategy:
            time.sleep(0.01)  # Brief pause

    end_time = time.time()

    # Collect statistics
    costs = [r["cost"] for r in all_results]
    times = [r["execution_time"] for r in all_results]

    # Check if we found any feasible solution
    if best_solution is None or best_cost == float('inf'):
        return {
            "solution": {},
            "cost": float('inf'),
            "feasible": False,
            "execution_time": end_time - start_time,
            "optimal": False,
            "algorithm": "adaptive_metaheuristic",
            "strategies_tried": strategies,
            "total_strategies": len(strategies),
        }
    
    result = {
        "solution": best_solution,
        "cost": best_cost,
        "feasible": is_proper_coloring(graph, best_solution),
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": "adaptive_metaheuristic",
        "strategies_tried": strategies,
        "total_strategies": len(strategies),
        "best_strategy_cost": min(costs),
        "worst_strategy_cost": max(costs),
        "average_strategy_cost": sum(costs) / len(costs),
        "total_computation_time": sum(times),
    }

    return result
