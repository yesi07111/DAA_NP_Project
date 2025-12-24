import time
import math
import random
from collections import deque
from typing import Dict, List, Any, Deque
import networkx as nx
import numpy as np
from utils.utils import is_proper_coloring
from utils.utils import evaluate_solution
from utils.timeout_handler import check_global_timeout

# SIMULATED ANNEALING

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
        from algorithms.heuristic_algorithms import dsatur_heuristic
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
        # Check timeout every 10 iterations
        if iteration % 10 == 0:
            try:
                check_global_timeout()
            except Exception:
                break
        
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
    from algorithms.heuristic_algorithms import largest_first_heuristic
    from algorithms.heuristic_algorithms import dsatur_heuristic
    from algorithms.heuristic_algorithms import recursive_largest_first_heuristic
    
    initial_solutions = []
    initial_solutions.append(largest_first_heuristic(graph, cost_matrix)['solution'])
    initial_solutions.append(dsatur_heuristic(graph, cost_matrix)['solution'])
    initial_solutions.append(recursive_largest_first_heuristic(graph, cost_matrix)['solution'])
    
    best_global_solution = None
    best_global_cost = float('inf')
    all_results = []
    
    # Try SA with different initial solutions
    for i, initial_sol in enumerate(initial_solutions):
        # Check global timeout before each SA run
        try:
            check_global_timeout()
        except Exception:
            break
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
            # Check global timeout before restart run
            try:
                check_global_timeout()
            except Exception:
                break
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
    result['operations'] = sum(r.get('iterations', 0) for r in all_results)
    
    return result

# TRAJECTORY SEARCH

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
        # Check timeout every 5 iterations
        if iteration % 5 == 0:
            try:
                check_global_timeout()
            except Exception:
                break

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

def _initialize_population(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    population_size: int,
    initial_algorithm: str,
    seed: int = None,
) -> List[Dict[int, int]]:

    from algorithms.heuristic_algorithms import largest_first_heuristic
    from algorithms.heuristic_algorithms import dsatur_heuristic
    from algorithms.heuristic_algorithms import (
        recursive_largest_first_heuristic,
    )
    from algorithms.approximation_algorithms import (
        improved_weighted_set_cover,
        weighted_set_cover_approximation, 
    )
    from utils.utils import is_proper_coloring

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

    # Fill remaining population with small perturbations of the first solution
    base_solution = population[0].copy()
    attempts = 0
    while len(population) < population_size:
        attempts += 1
        # Check global timeout every iteration
        try:
            check_global_timeout()
        except Exception:
            break

        # Create a perturbation of the base solution
        sol = base_solution.copy()
        n_colors = cost_matrix.shape[1]
        
        # Randomly change colors of a few vertices
        vertices = list(graph.nodes())
        random.shuffle(vertices)
        for v in vertices[:random.randint(1, max(2, len(vertices) // 5))]:
            sol[v] = random.randint(0, n_colors - 1)
        
        if is_proper_coloring(graph, sol):
            population.append(sol)
        elif attempts > 100:
            # If we can't find random solutions, just use base solution again
            population.append(base_solution.copy())

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
    color_attempts = 0
    for it in range(100):  # Limit local search iterations
        # Check timeout at each local search iteration
        try:
            check_global_timeout()
        except Exception:
            return current_solution
        best_neighbor = None
        best_cost = current_cost

        # Generate neighborhood by recolorings (limit to first 10 vertices for speed)
        vertices_to_try = list(graph.nodes())[:10]
        for vertex in vertices_to_try:
            current_color = current_solution[vertex]

            # Try different colors
            for new_color in range(min(n_colors, 10)):  # Limit colors tried to first 10
                color_attempts += 1
                # Check timeout frequently in inner loop (every 2 attempts)
                if color_attempts % 2 == 0:
                    try:
                        check_global_timeout()
                    except Exception:
                        return current_solution

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
    for idx, vertex in enumerate(differences):
        # Periodic timeout check (every 1-2 vertices)
        if idx % 2 == 0:
            try:
                check_global_timeout()
            except Exception:
                return best_solution
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
    for idx, vertex in enumerate(conflicts):
        # Periodic timeout check (every 1-2 vertices)
        if idx % 2 == 0:
            try:
                check_global_timeout()
            except Exception:
                return diversified
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

# HYBRID METAHEURISTIC

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

    # Check global timeout before invoking chosen method
    try:
        check_global_timeout()
    except Exception:
        return {
            "solution": {},
            "cost": float('inf'),
            "feasible": False,
            "execution_time": time.time() - start_time,
            "optimal": False,
            "algorithm": f"hybrid_metaheuristic_{method}",
            "time_limit_exceeded": True,
        }

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

        # Check global timeout before starting this strategy
        try:
            check_global_timeout()
        except Exception:
            break

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
            "operations": 0,
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
        "operations": sum(r.get("operations", r.get("iterations", 1)) for r in all_results),
    }

    return result
