"""
Metrics calculation for DAA Project - MCCPP
"""
import numpy as np
from typing import Dict, List, Any, Optional
import networkx as nx

def calculate_approximation_ratio(achieved_cost: float,
                                  optimal_cost: Optional[float],
                                  graph_size: int,
                                  algorithm: str) -> Dict[str, float]:
    """
    Calculate approximation ratio and related metrics for DAA Project - MCCPP
    
    Args:
        achieved_cost: cost achieved by the algorithm
        optimal_cost: known optimal cost
        graph_size: number of vertices in the graph
        algorithm: algorithm name
    
    Returns:
        dictionary of approximation metrics
    """

    metrics = {
        "achieved_cost": achieved_cost,
        "optimal_cost": optimal_cost,
        "graph_size": graph_size,
        "algorithm": algorithm
    }

    # Si no hay coste óptimo, no hay análisis de aproximación
    if optimal_cost is None or optimal_cost <= 0:
        metrics.update({
            "approximation_ratio": None,
            "optimality_gap": None,
            "optimality_gap_percent": None,
            "theoretical_bound": None,
            "bound_ratio": None
        })
        return metrics

    ratio = achieved_cost / optimal_cost
    gap = (achieved_cost - optimal_cost) / optimal_cost

    metrics["approximation_ratio"] = ratio
    metrics["optimality_gap"] = gap
    metrics["optimality_gap_percent"] = gap * 100

    # Identificar tipo de algoritmo
    algo = algorithm.lower()

    if "weighted_set_cover" in algo or "improved_weighted_set_cover" in algo:
        # Lema clásico de set-cover
        theoretical = np.log(max(graph_size, 2))
    elif "bipartite" in algo:
        theoretical = 2.0  # construcciones bipartitas tienen factor 2
    elif "interval" in algo:
        theoretical = np.sqrt(max(graph_size, 1))
    else:
        theoretical = np.log(max(graph_size, 2))  # bound genérico

    metrics["theoretical_bound"] = theoretical
    metrics["bound_ratio"] = ratio / theoretical

    return metrics

def calculate_solution_quality_metrics(
    graph: nx.Graph,
    coloring: Dict[int, int],
    cost_matrix: np.ndarray,
    optimal_cost: Optional[float] = None,
    algorithm: str = ""
) -> Dict[str, Any]:
    """
    Calculate comprehensive solution quality metrics for DAA Project - MCCPP
    
    Args:
        graph: networkx Graph
        coloring: solution coloring
        cost_matrix: cost matrix
        optimal_cost: known optimal cost (if available)
    
    Returns:
        dictionary of quality metrics
    """
    from src.utils.cost_utils import evaluate_solution
    from src.utils.graph_utils import is_proper_coloring

    n = graph.number_of_nodes()
    if n == 0:
        return {
            "total_cost": 0,
            "feasible": True,
            "colors_used": 0,
            "graph_density": 0,
            "n_vertices": 0,
            "n_edges": 0,
            "vertex_cost_stats": {},
            "color_cost_stats": {},
            "approximation_ratio": None,
            "optimality_gap": None,
            "theoretical_bound": None
        }

    # Coste total
    total_cost = evaluate_solution(coloring, cost_matrix)

    # Checking feasibility
    feasible = is_proper_coloring(graph, coloring)

    colors_used = len(set(coloring.values()))

    # Vertex costs
    vertex_costs = [cost_matrix[v, coloring[v]] for v in graph.nodes()]

    metrics = {
        "total_cost": total_cost,
        "feasible": feasible,
        "colors_used": colors_used,
        "graph_density": nx.density(graph),
        "n_vertices": n,
        "n_edges": graph.number_of_edges(),
        "vertex_cost_stats": {
            "min_cost": float(np.min(vertex_costs)),
            "max_cost": float(np.max(vertex_costs)),
            "avg_cost": float(np.mean(vertex_costs)),
            "std_cost": float(np.std(vertex_costs)),
        },
    }

    # Por-color analysis
    color_costs = {}
    for v, c in coloring.items():
        color_costs.setdefault(c, []).append(cost_matrix[v, c])

    metrics["color_cost_stats"] = {
        c: {
            "min_cost": float(np.min(vals)),
            "max_cost": float(np.max(vals)),
            "avg_cost": float(np.mean(vals)),
            "total_cost": float(np.sum(vals)),
            "vertices": len(vals),
        }
        for c, vals in color_costs.items()
    }

    # Approximation-specific metrics
    approx = calculate_approximation_ratio(
        achieved_cost=total_cost,
        optimal_cost=optimal_cost,
        graph_size=n,
        algorithm=algorithm
    )
    metrics.update(approx)

    return metrics


def analyze_optimality_vs_bruteforce(comparison_group: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae métricas que comparan algoritmos eficientes contra fuerza bruta.
    Requiere que brute_force o brute_force_backtracking estén presentes.
    """
    bf_keys = [k for k in comparison_group.keys() if "brute" in k]
    if not bf_keys:
        return {}

    bf = comparison_group[bf_keys[0]]
    bf_costs = bf["costs"]

    results = {}

    for algo, stats in comparison_group.items():
        if "brute" in algo:
            continue

        algo_costs = stats["costs"]
        paired = list(zip(bf_costs, algo_costs))

        errors = [(a - b) for (b, a) in paired]
        rel_errors = [((a - b) / b) for (b, a) in paired if b != 0]

        results[algo] = {
            "optimal_matches": sum(1 for (b, a) in paired if a == b),
            "optimal_rate": sum(1 for (b, a) in paired if a == b) / len(paired),
            "avg_absolute_error": float(np.mean(errors)),
            "avg_relative_error": float(np.mean(rel_errors)),
            "worst_error": float(np.max(errors)),
            "best_case": float(np.min(errors)),
        }

    return results

def analyze_time_vs_bruteforce(comparison_group: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara tiempos de algoritmos eficientes contra fuerza bruta.
    """
    bf_keys = [k for k in comparison_group.keys() if "brute" in k]
    if not bf_keys:
        return {}

    bf = comparison_group[bf_keys[0]]
    bf_times = bf["times"]

    results = {}

    for algo, stats in comparison_group.items():
        if "brute" in algo:
            continue

        algo_times = stats["times"]
        paired = list(zip(bf_times, algo_times))

        ratios = [(b / a) if a > 0 else float("inf") for (b, a) in paired]

        results[algo] = {
            "mean_time_ratio": float(np.mean(ratios)),
            "max_ratio": float(np.max(ratios)),
            "min_ratio": float(np.min(ratios)),
            "bf_slowdown_factor": float(np.mean([b - a for (b, a) in paired])),
        }

    return results

def detect_bruteforce_limits(benchmark_results: Dict[str, Any], time_threshold: float = 60.0):
    """
    Determina el tamaño máximo de instancia para el cual brute force
    se mantiene bajo un tiempo razonable.
    """
    bf_times = []

    for instance, data in benchmark_results.items():
        if instance.startswith("_"):
            continue
        if "brute_force" not in data:
            continue

        t = data["brute_force"]["execution_time"]
        n = data["brute_force"]["solution_size"] if "solution_size" in data["brute_force"] else None
        bf_times.append((instance, n, t))

    under_limit = [(i, n, t) for (i, n, t) in bf_times if t <= time_threshold]
    over_limit = [(i, n, t) for (i, n, t) in bf_times if t > time_threshold]

    return {
        "threshold": time_threshold,
        "max_size_under_limit": max([n for _, n, t in under_limit], default=None),
        "first_instance_over_limit": over_limit[0] if over_limit else None,
        "full_table": bf_times,
    }

def analyze_behavior_by_instance_type(benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detecta en qué tipo de instancias los algoritmos heurísticos
    funcionan mejor o peor comparando contra brute force.
    """
    grouped = {}

    for instance, data in benchmark_results.items():
        if instance.startswith("_"):
            continue

        meta = data.get("metadata", {})
        inst_type = meta.get("instance_type", "unknown")

        if inst_type not in grouped:
            grouped[inst_type] = []

        grouped[inst_type].append(data)

    summary = {}

    for t, entries in grouped.items():
        bf_costs = []
        heur_best = []

        for e in entries:
            if "brute_force" not in e:
                continue

            bf_cost = e["brute_force"]["cost"]
            bf_costs.append(bf_cost)

            heur_cost = min([v["cost"] for k, v in e.items() if "brute" not in k])
            heur_best.append(heur_cost)

        diffs = [h - b for (h, b) in zip(heur_best, bf_costs)]

        summary[t] = {
            "mean_error": float(np.mean(diffs)),
            "max_error": float(np.max(diffs)),
            "optimal_rate": sum(1 for x in diffs if x == 0) / len(diffs),
            "num_instances": len(entries)
        }

    return summary

def empirical_initial_solution_analysis(
    instances: Dict[str, Any],
    output_dir: str,
    seed: int = 123,
):
    """
    Empirical evaluation of TSH+PR starting from different initial solution generators.
    """

    import json
    import os
    import time
    import numpy as np

    from src.algorithms.heuristic.largest_first import largest_first_heuristic
    from src.algorithms.heuristic.dsatur import dsatur_heuristic
    from src.algorithms.heuristic.recursive_largest_first import (
        recursive_largest_first_heuristic,
    )

    from src.algorithms.approximation.weighted_set_cover import (
        improved_weighted_set_cover,
        weighted_set_cover_approximation,
    )
  
    from src.algorithms.metaheuristic.trajectory_search import trajectory_search_heuristic
    from src.utils.graph_utils import is_proper_coloring
    from src.utils.cost_utils import evaluate_solution

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    for instance_name, data in instances.items():

        graph = data["graph"]
        cost_matrix = data["cost_matrix"]
        results[instance_name] = {}

        heuristics = {
            "improved_weighted_set_cover": lambda: improved_weighted_set_cover(graph, cost_matrix),
            "weighted_set_cover": lambda: weighted_set_cover_approximation(graph, cost_matrix),
            "largest_first": lambda: largest_first_heuristic(graph, cost_matrix),
            "dsatur": lambda: dsatur_heuristic(graph, cost_matrix),
            "recursive_largest_first": lambda: recursive_largest_first_heuristic(graph, cost_matrix),
        }

        for hname, hfunc in heuristics.items():

            # ---- Step 1: generate initial solution ----
            h_start = time.time()
            try:
                res = hfunc()
            except Exception:
                continue
            h_end = time.time()

            if res is None or "solution" not in res:
                continue

            sol0 = res["solution"]
            feasible0 = is_proper_coloring(graph, sol0)
            cost0 = evaluate_solution(sol0, cost_matrix) if feasible0 else np.inf

            if not feasible0:
                continue  # skip these

            # ---- Step 2: run trajectory using this initial heuristic ----
            tsh_start = time.time()

            tsh_result = trajectory_search_heuristic(
                graph,
                cost_matrix,
                population_size=10,
                max_iterations=5000,
                seed=seed,
                initial_algorithm=hname,    # NEW PARAMETER
            )

            tsh_end = time.time()

            results[instance_name][hname] = {
                "initial_feasible": feasible0,
                "initial_cost": cost0,
                "initial_time": h_end - h_start,
                "final_cost": tsh_result["cost"],
                "final_time": tsh_end - tsh_start,
                "improvement": cost0 - tsh_result["cost"],
                "tsh_iterations": tsh_result["iterations"],
            }

    # Save JSON
    outfile = os.path.join(output_dir, "empirical_initial_solution_analysis.json")
    with open(outfile, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Empirical initial solution analysis saved in {outfile}")
    return results

def generate_extended_empirical_report(
        comparison_group,
        optimality_analysis,
        time_analysis,
        bf_limits,
        type_behavior,
        approximation_stats, 
        initial_solution_analysis=None,     
        output_file="empirical_analysis.md"
    ):
    with open(output_file, "w") as f:

        f.write("# Empirical Performance Analysis\n")
        f.write("Este reporte analiza calidad, tiempos y límites computacionales.\n\n")

        # ==========================================================
        # 1 — CALIDAD VS BRUTE FORCE
        # ==========================================================
        f.write("## 1. Solution Quality vs Brute Force\n")
        f.write("Comparación entre algoritmos eficientes y fuerza bruta.\n\n")

        for algo, stats in optimality_analysis.items():
            f.write(f"### {algo}\n")
            f.write(f"- Optimal matches: {stats['optimal_rate']*100:.1f}%\n")
            f.write(f"- Avg absolute error: {stats['avg_absolute_error']:.3f}\n")
            f.write(f"- Avg relative error: {stats['avg_relative_error']:.3f}\n")
            f.write(f"- Worst-case error: {stats['worst_error']:.3f}\n\n")

        # ==========================================================
        # 2 — TIEMPOS VS BRUTE FORCE
        # ==========================================================
        f.write("## 2. Runtime Analysis\n")
        f.write("Tiempos relativos heurística vs fuerza bruta.\n\n")

        for algo, stats in time_analysis.items():
            f.write(f"### {algo}\n")
            f.write(f"- Mean time ratio (BF / algo): {stats['mean_time_ratio']:.2f}\n")
            f.write(f"- BF slowdown factor: {stats['bf_slowdown_factor']:.2f}s\n\n")

        # ==========================================================
        # 3 — LÍMITE DE TAMAÑO PARA BRUTE FORCE
        # ==========================================================
        if bf_limits is not None:
            f.write("## 3. Maximum Instance Size Brute Force Can Solve\n\n")
            f.write(f"- Time threshold: {bf_limits['threshold']}s\n")
            f.write(f"- Max size solvable under threshold: {bf_limits['max_size_under_limit']}\n")
            f.write(f"- First failing instance: {bf_limits['first_instance_over_limit']}\n\n")
        else:
            f.write("## 3. Maximum Instance Size Brute Force Can Solve\n\n")
            f.write("Brute force is not available in this experiment group.\n\n")

        # ==========================================================
        # 4 — COMPORTAMIENTO POR TIPO DE INSTANCIA
        # ==========================================================
        f.write("## 4. Behavior by Instance Type\n\n")

        for inst_type, stats in type_behavior.items():
            f.write(f"### Type: {inst_type}\n")
            f.write(f"- Optimal rate: {stats['optimal_rate']*100:.1f}%\n")
            f.write(f"- Mean error: {stats['mean_error']:.3f}\n")
            f.write(f"- Max error: {stats['max_error']:.3f}\n")
            f.write(f"- Num instances: {stats['num_instances']}\n\n")

        # ==========================================================
        # 5 — MÉTRICAS DE APROXIMACIÓN
        # ==========================================================
        f.write("## 5. Approximation and Quality Metrics\n\n")

        for algo, entries in approximation_stats.items():
            f.write(f"### {algo}\n")

            approx_ratios = [
                e["approximation_ratio"] for e in entries
                if e["approximation_ratio"] is not None
            ]

            if approx_ratios:
                f.write(f"- Mean approximation ratio: {np.mean(approx_ratios):.3f}\n")
                f.write(f"- Worst ratio: {np.max(approx_ratios):.3f}\n")
                f.write(f"- Best ratio: {np.min(approx_ratios):.3f}\n\n")
            else:
                f.write("- No approximation data available.\n\n")

        # ==========================================================
        # 6 — TRAJECTORY SEARCH: IMPACTO DE LA SOLUCIÓN INICIAL
        # ==========================================================
        if initial_solution_analysis:
            f.write("\n## 6. Trajectory Search — Impacto de la Solución Inicial\n")
            f.write("Este análisis evalúa cómo cambia el desempeño de TSH+PR dependiendo de la heurística que genera la solución inicial.\n\n")

            for inst, heur_data in initial_solution_analysis.items():
                f.write(f"### Instance: {inst}\n")
                for hname, stats in heur_data.items():
                    f.write(f"#### Heuristic: {hname}\n")
                    f.write(f"- Initial cost: {stats['initial_cost']:.2f}\n")
                    f.write(f"- Final cost: {stats['final_cost']:.2f}\n")
                    f.write(f"- Improvement: {stats['improvement']:.2f}\n")
                    f.write(f"- Initial time: {stats['initial_time']:.4f}s\n")
                    f.write(f"- Final time (TSH): {stats['final_time']:.4f}s\n")
                    f.write(f"- Iterations: {stats['tsh_iterations']}\n\n")

        f.write("\n---\nEnd of Report\n")

