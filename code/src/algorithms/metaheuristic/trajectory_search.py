"""
Trajectory Search Heuristic with Path Relinking for DAA Project - MCCPP
"""

import time
import random
import numpy as np
import networkx as nx
from collections import deque
from typing import Dict, List, Any, Deque
from src.utils.graph_utils import is_proper_coloring
from src.utils.cost_utils import evaluate_solution

def trajectory_search_heuristic(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    population_size: int = 10,
    max_iterations: int = 10000,
    tabu_tenure: int = 50,
    path_relinking_frequency: int = 10,
    elite_pool_size: int = 5,
    diversification_frequency: int = 100,
    seed: int = None,
    initial_algorithm: str = "improved_weighted_set_cover",
) -> Dict[str, Any]:
    """
    Trajectory Search Heuristic with Path Relinking for DAA Project - MCCPP.

    Advanced metaheuristic combining local search with strategic diversification
    and intensification through Path Relinking. Based on the TSH+PR approach
    specifically designed for MCCPP.

    Args:
        graph: networkx Graph
        cost_matrix: n_vertices x n_colors cost matrix
        population_size: size of the solution population
        max_iterations: maximum number of iterations
        tabu_tenure: tenure for tabu search component
        path_relinking_frequency: how often to apply path relinking
        elite_pool_size: number of elite solutions to maintain
        diversification_frequency: how often to apply diversification
        seed: random seed for reproducibility

    Returns:
        dictionary with solution and metrics
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    start_time = time.time()

    # Initialize population
    population = _initialize_population(
    graph, cost_matrix, population_size, initial_algorithm, seed
    )
    elite_pool = _select_elite_solutions(
        population, elite_pool_size, graph, cost_matrix
    )

    # Check if we have any valid solutions
    if len(elite_pool) == 0:
        end_time = time.time()
        return {
            "solution": {},
            "cost": float('inf'),
            "feasible": False,
            "execution_time": end_time - start_time,
            "optimal": False,
            "algorithm": "trajectory_search_path_relinking",
            "iterations": 0,
            "improvements": 0,
            "elite_pool_size": 0,
            "relinking_applied": 0,
            "diversification_applied": 0,
            "final_population_diversity": 0.0,
            "cost_history": [],
        }

    # Initialize tabu memory
    tabu_list = deque(maxlen=tabu_tenure)

    # Track best solution
    best_solution = elite_pool[0]["solution"].copy()
    best_cost = elite_pool[0]["cost"]

    # Statistics
    iteration = 0
    improvements = 0
    relinking_applied = 0
    diversification_applied = 0
    cost_history = [best_cost]

    # Main TSH+PR loop
    while iteration < max_iterations:
        iteration += 1

        # Local search phase
        current_solution = random.choice(elite_pool)["solution"]
        improved_solution = _local_search(
            graph, cost_matrix, current_solution, tabu_list
        )

        improved_cost = evaluate_solution(improved_solution, cost_matrix)

        # Update elite pool
        if improved_cost < best_cost:
            best_solution = improved_solution.copy()
            best_cost = improved_cost
            improvements += 1

        _update_elite_pool(
            elite_pool,
            improved_solution,
            improved_cost,
            graph,
            cost_matrix,
            elite_pool_size,
        )

        # Path Relinking phase
        if iteration % path_relinking_frequency == 0:
            relinking_applied += 1
            guiding_solution = random.choice(elite_pool)["solution"]
            pr_solution = _path_relinking(
                graph, cost_matrix, current_solution, guiding_solution
            )
            pr_cost = evaluate_solution(pr_solution, cost_matrix)

            if pr_cost < best_cost:
                best_solution = pr_solution.copy()
                best_cost = pr_cost
                improvements += 1

            _update_elite_pool(
                elite_pool, pr_solution, pr_cost, graph, cost_matrix, elite_pool_size
            )

        # Diversification phase
        if iteration % diversification_frequency == 0:
            diversification_applied += 1
            diversified_solution = _diversify_solution(
                graph, cost_matrix, best_solution
            )
            diversified_cost = evaluate_solution(diversified_solution, cost_matrix)

            _update_elite_pool(
                elite_pool,
                diversified_solution,
                diversified_cost,
                graph,
                cost_matrix,
                elite_pool_size,
            )

        # Record history
        cost_history.append(best_cost)

        # Early termination if no improvement for too long
        if iteration > 100 and len(set(cost_history[-50:])) == 1:
            break

    end_time = time.time()

    result = {
        "solution": best_solution,
        "cost": best_cost,
        "feasible": is_proper_coloring(graph, best_solution),
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": "trajectory_search_path_relinking",
        "iterations": iteration,
        "improvements": improvements,
        "elite_pool_size": len(elite_pool),
        "relinking_applied": relinking_applied,
        "diversification_applied": diversification_applied,
        "final_population_diversity": _calculate_population_diversity(elite_pool),
        "cost_history": cost_history,
    }

    return result

# def _initialize_population(
#     graph: nx.Graph,
#     cost_matrix: np.ndarray,
#     population_size: int,
#     seed: int = None,
# ) -> List[Dict[int, int]]:
#     """
#     Initialize diverse population for TSH+PR in DAA Project - MCCPP
#     """
#     population = []

#     # Use different heuristics to generate initial solutions
#     from src.algorithms.heuristic.largest_first import largest_first_heuristic
#     from src.algorithms.heuristic.dsatur import dsatur_heuristic
#     from src.algorithms.heuristic.recursive_largest_first import (
#         recursive_largest_first_heuristic,
#     )

#     # Generate solutions using different methods
#     methods = [
#         lambda: largest_first_heuristic(graph, cost_matrix)["solution"],
#         lambda: dsatur_heuristic(graph, cost_matrix)["solution"],
#         lambda: recursive_largest_first_heuristic(graph, cost_matrix)["solution"],
#     ]

#     for method in methods:
#         if len(population) < population_size:
#             population.append(method())
#             if not is_proper_coloring(graph, population[-1]):
#                 print(">>>", len(population) % 3, "produced an improper coloring.")
#                 population.pop()

#     # Fill remaining population with random solutions
#     while len(population) < population_size:
#         random_solution = {}
#         for vertex in graph.nodes():
#             random_solution[vertex] = random.randint(0, cost_matrix.shape[1] - 1)
#         population.append(random_solution)

#     return population

def _initialize_population(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    population_size: int,
    initial_algorithm: str,
    seed: int = None,
) -> List[Dict[int, int]]:

    from src.algorithms.heuristic.largest_first import largest_first_heuristic
    from src.algorithms.heuristic.dsatur import dsatur_heuristic
    from src.algorithms.heuristic.recursive_largest_first import (
        recursive_largest_first_heuristic,
    )
    from src.algorithms.approximation.weighted_set_cover import (
        improved_weighted_set_cover,
        weighted_set_cover_approximation, 
    )
    from src.utils.graph_utils import is_proper_coloring

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    methods = {
        "improved_weighted_set_cover": lambda: improved_weighted_set_cover(graph, cost_matrix)["solution"],
        "weighted_set_cover": lambda: weighted_set_cover_approximation(graph, cost_matrix)["solution"],
        "largest_first": lambda: largest_first_heuristic(graph, cost_matrix)["solution"],
        "dsatur": lambda: dsatur_heuristic(graph, cost_matrix)["solution"],
        "recursive_largest_first": lambda: recursive_largest_first_heuristic(graph, cost_matrix)["solution"],
    }

    population = []

    # Priority: try the requested initial heuristic first
    heuristic_order = [initial_algorithm] + [h for h in methods if h != initial_algorithm]

    for hname in heuristic_order:
        try:
            sol = methods[hname]()
            if is_proper_coloring(graph, sol):
                population.append(sol)
                break
        except Exception:
            continue

    # If no heuristic produced valid solution → random fallback
    if len(population) == 0:
        return [{}]  # same behavior as before (fail-safe)

    # Fill remaining population with random valid solutions
    while len(population) < population_size:
        sol = {}
        n_colors = cost_matrix.shape[1]
        for v in graph.nodes():
            sol[v] = random.randint(0, n_colors - 1)
        if is_proper_coloring(graph, sol):
            population.append(sol)

    return population


def _select_elite_solutions(
    population: List[Dict[int, int]],
    elite_size: int,
    graph: nx.Graph,
    cost_matrix: np.ndarray,
) -> List[Dict[str, Any]]:
    """
    Select elite solutions based on cost and feasibility for DAA Project - MCCPP
    """
    evaluated = []
    for solution in population:
        feasible = is_proper_coloring(graph, solution)
        if not feasible:
            continue

        cost = evaluate_solution(solution, cost_matrix)
        evaluated.append({"solution": solution, "cost": cost, "feasible": feasible})

    # Sort by cost (lower is better), prioritizing feasible solutions
    evaluated.sort(key=lambda x: x["cost"])
    return evaluated[:elite_size]

def _local_search(
    graph: nx.Graph, cost_matrix: np.ndarray, solution: Dict[int, int], tabu_list: Deque
) -> Dict[int, int]:
    """
    Local search with tabu component for TSH+PR in DAA Project - MCCPP
    """
    current_solution = solution.copy()
    current_cost = evaluate_solution(current_solution, cost_matrix)
    n_colors = cost_matrix.shape[1]

    # Try improving moves
    for _ in range(100):  # Limit local search iterations
        best_neighbor = None
        best_cost = current_cost

        # Generate neighborhood by recolorings
        for vertex in graph.nodes():
            current_color = current_solution[vertex]

            # Try different colors
            for new_color in range(n_colors):
                if new_color == current_color:
                    continue

                # Check if move is tabu
                move = (vertex, current_color, new_color)
                if move in tabu_list:
                    continue

                # Create neighbor
                neighbor = current_solution.copy()
                neighbor[vertex] = new_color

                if not is_proper_coloring(graph, neighbor):
                    continue

                neighbor_cost = evaluate_solution(neighbor, cost_matrix)
                if neighbor_cost < best_cost:
                    best_neighbor = neighbor
                    best_cost = neighbor_cost
                    best_move = move

        if best_neighbor is None:
            break  # No improving move found

        # Accept the best move
        current_solution = best_neighbor
        current_cost = best_cost

        # Update tabu list
        tabu_list.append(best_move)

    return current_solution

def _path_relinking(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    initial_solution: Dict[int, int],
    guiding_solution: Dict[int, int],
) -> Dict[int, int]:
    """
    Path Relinking between two solutions for DAA Project - MCCPP
    """
    current_solution = initial_solution.copy()
    best_solution = current_solution.copy()
    best_cost = evaluate_solution(current_solution, cost_matrix)

    # Find differences between solutions
    differences = []
    for vertex in graph.nodes():
        if current_solution[vertex] != guiding_solution[vertex]:
            differences.append(vertex)

    # Shuffle differences to explore different paths
    random.shuffle(differences)

    # Create path from initial to guiding solution
    for vertex in differences:
        # Store current color
        tmp = current_solution[vertex]

        # Move current solution toward guiding solution
        current_solution[vertex] = guiding_solution[vertex]

        if not is_proper_coloring(graph, current_solution):
            current_solution[vertex] = tmp  # Revert if not feasible
            continue

        current_cost = evaluate_solution(current_solution, cost_matrix)

        # Update best solution found along the path
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost

    return best_solution

def _diversify_solution(
    graph: nx.Graph, cost_matrix: np.ndarray, solution: Dict[int, int]
) -> Dict[int, int]:
    """
    Apply diversification to escape local optima in DAA Project - MCCPP
    """
    diversified = solution.copy()
    n_colors = cost_matrix.shape[1]

    # Identify problematic vertices (those with conflicts or high cost)
    conflicts = []
    for u, v in graph.edges():
        if diversified[u] == diversified[v]:
            conflicts.extend([u, v])

    # If no conflicts, diversify based on cost
    if not conflicts:
        # Find vertices with highest cost contribution
        vertex_costs = []
        for vertex in graph.nodes():
            cost = cost_matrix[vertex, diversified[vertex]]
            vertex_costs.append((vertex, cost))

        vertex_costs.sort(key=lambda x: x[1], reverse=True)
        conflicts = [v for v, _ in vertex_costs[: len(graph.nodes()) // 4]]  # Top 25%

    # Perturb the solution
    for vertex in conflicts:
        # Try a few random colors
        for _ in range(3):
            new_color = random.randint(0, n_colors - 1)
            if new_color != diversified[vertex]:
                diversified[vertex] = new_color
                break

    return diversified

def _update_elite_pool(
    elite_pool: List[Dict[str, Any]],
    new_solution: Dict[int, int],
    new_cost: float,
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    elite_size: int,
):
    """
    Update elite pool with new solution for DAA Project - MCCPP
    """
    new_feasible = is_proper_coloring(graph, new_solution)
    if not new_feasible:
        return

    new_entry = {"solution": new_solution, "cost": new_cost, "feasible": new_feasible}

    # Check if solution is already in pool
    for entry in elite_pool:
        if entry["solution"] == new_solution:
            return

    # Add to pool and sort
    elite_pool.append(new_entry)
    elite_pool.sort(key=lambda x: (not x["feasible"], x["cost"]))

    # Maintain pool size
    if len(elite_pool) > elite_size:
        elite_pool.pop()

def _calculate_population_diversity(elite_pool: List[Dict[str, Any]]) -> float:
    """
    Calculate diversity of elite pool for DAA Project - MCCPP
    """
    if len(elite_pool) <= 1:
        return 0.0

    diversity = 0.0
    count = 0

    for i in range(len(elite_pool)):
        for j in range(i + 1, len(elite_pool)):
            sol1 = elite_pool[i]["solution"]
            sol2 = elite_pool[j]["solution"]

            # Calculate Hamming distance between solutions
            distance = 0
            for vertex in sol1.keys():
                if sol1[vertex] != sol2[vertex]:
                    distance += 1

            diversity += distance
            count += 1

    return diversity / count if count > 0 else 0.0

