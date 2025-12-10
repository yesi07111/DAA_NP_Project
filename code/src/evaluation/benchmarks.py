"""
Suite de benchmarks para el Proyecto DAA - MCCPP
✨ Logging colorido con emojis
"""

import time
import json
import os
from typing import Dict, List, Any

import numpy as np
from src.algorithms.metaheuristic.hybrid_metaherusitics import adaptive_metaheuristic, hybrid_metaheuristic
from src.utils.io_utils import (
    load_instance,
    save_solution,
    ensure_directory,
)
from src.algorithms.exact.brute_force import (
    brute_force_solver,
    brute_force_with_backtracking,
    intelligent_brute_force,
)
from src.algorithms.exact.ilp_solver import ilp_solver
from src.algorithms.exact.dynamic_programming import dynamic_programming_tree
from src.algorithms.approximation.weighted_set_cover import (
    improved_weighted_set_cover,
    weighted_set_cover_approximation,
)
from src.algorithms.approximation.structural_approximation import (
    structural_approximation_bipartite,
    structural_approximation_interval,
)
from src.algorithms.heuristic.largest_first import largest_first_heuristic
from src.algorithms.heuristic.dsatur import dsatur_heuristic
from src.algorithms.heuristic.recursive_largest_first import (
    recursive_largest_first_heuristic,
)
from src.algorithms.metaheuristic.simulated_annealing import adaptive_simulated_annealing, simulated_annealing
from src.algorithms.metaheuristic.trajectory_search import trajectory_search_heuristic


# 🎨 ANSI COLORS
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def run_benchmark_suite(
    instance_files: List[str],
    algorithms: List[str],
    output_dir: str = "results",
    time_limit: float = 1000000.0,
) -> Dict[str, Any]:
    results = {}
    benchmark_start = time.time()

    # Crear directorio
    ensure_directory(output_dir)
    print(f"{GREEN}📁 Directorio de salida listo: {output_dir}{RESET}")

    for instance_file in instance_files:
        print(f"\n{BOLD}{BLUE}📌 Procesando instancia:{RESET} {instance_file}")
        instance_name = os.path.basename(instance_file).replace(".json", "")
        results[instance_name] = {}

        # Cargar instancia
        print(f"{CYAN}   📥 Cargando instancia...{RESET}")
        graph, cost_matrix, metadata = load_instance(instance_file)

        for algorithm in algorithms:
            print(f"{MAGENTA}   🚀 Ejecutando algoritmo:{RESET} {algorithm}")
            algorithm_start = time.time()

            try:
                result = None

                # Selección de algoritmo
                if algorithm == "brute_force":
                    if len(graph.nodes) < 7:
                        print(f"{YELLOW}      🧮 Fuerza bruta UNGA BUNGA...{RESET}")
                        result = brute_force_solver(
                            graph, cost_matrix, time_limit=time_limit
                        )
                    else:
                        print(
                            f"{RED}      ⛔ Grafo > 7 nodos, saltando fuerza bruta{RESET}"
                        )
                elif algorithm == "brute_force_backtracking":
                    if len(graph.nodes) < 10:
                        print(
                            f"{YELLOW}      🧮 Fuerza bruta con backtracking...{RESET}"
                        )
                        result = brute_force_with_backtracking(
                            graph, cost_matrix, time_limit=time_limit
                        )
                    else:
                        print(
                            f"{RED}      ⛔ Grafo > 10 nodos, saltando fuerza bruta{RESET}"
                        )
                elif algorithm == "brute_force_intelligent":
                    if len(graph.nodes) < 20:
                        print(
                            f"{YELLOW}      🧮 Fuerza bruta con backtracking y poda...{RESET}"
                        )
                        result = intelligent_brute_force(
                            graph, cost_matrix, time_limit=time_limit
                        )
                    else:
                        print(
                            f"{RED}      ⛔ Grafo > 22 nodos, saltando fuerza bruta{RESET}"
                        )

                elif algorithm == "ilp_solver":
                    if len(graph.nodes) < 50:
                        print(f"{YELLOW}      🔧 Solucionador ILP...{RESET}")
                        result = ilp_solver(graph, cost_matrix, time_limit=time_limit)
                    else:
                        print(
                            f"{RED}      ⛔ Grafo > 50 nodos, saltando ILP{RESET}"
                        )

                elif algorithm == "dynamic_programming_tree":
                    if metadata.get("instance_type", "not_a_tree") != "tree":
                        print(
                            f"{RED}      ⏭  [SKIP] No apto para DP (instance_type={metadata.get('instance_type', 'not_a_tree')}){RESET}"
                        )
                        continue

                    result = dynamic_programming_tree(graph, cost_matrix)

                elif algorithm == "weighted_set_cover":
                    print(
                        f"{YELLOW}      🧩 Aproximación Set Cover ponderado...{RESET}"
                    )
                    result = weighted_set_cover_approximation(graph, cost_matrix)

                elif algorithm == "improved_weighted_set_cover":
                    print(
                        f"{RED}      🧩 Aproximación Set Cover ponderado mejorado...{RESET}"
                    )
                    result = improved_weighted_set_cover(graph, cost_matrix)

                elif algorithm == "structural_bipartite":
                    print(
                        f"{YELLOW}      🏗️ Aproximación estructural (bipartito)...{RESET}"
                    )
                    result = structural_approximation_bipartite(graph, cost_matrix)

                elif algorithm == "structural_interval":
                    print(
                        f"{YELLOW}      🏗️ Aproximación estructural (intervalos)...{RESET}"
                    )
                    intervals = metadata.get("intervals")
                    result = structural_approximation_interval(
                        graph, cost_matrix, intervals
                    )

                elif algorithm == "largest_first":
                    print(f"{YELLOW}      🔠 Heurística Largest First...{RESET}")
                    result = largest_first_heuristic(graph, cost_matrix)

                elif algorithm == "dsatur":
                    print(f"{YELLOW}      🎨 Heurística DSATUR...{RESET}")
                    result = dsatur_heuristic(graph, cost_matrix)

                elif algorithm == "rlf":
                    print(
                        f"{YELLOW}      🔁 Heurística Recursive Largest First...{RESET}"
                    )
                    result = recursive_largest_first_heuristic(graph, cost_matrix)

                elif algorithm == "simulated_annealing":
                    print(f"{YELLOW}      🔥 Simulated Annealing...{RESET}")
                    result = simulated_annealing(
                        graph, cost_matrix, max_iterations=10000
                    )
                
                elif algorithm == "adaptive_simmulated_annealing":
                    print(f"{RED}      🔥 Adaptive Simulated Annealing...{RESET}")
                    result = adaptive_simulated_annealing(
                        graph, cost_matrix
                    )

                elif algorithm == "trajectory_search":
                    print(f"{YELLOW}      🧭 Trajectory Search Heuristic...{RESET}")
                    result = trajectory_search_heuristic(
                        graph, cost_matrix, max_iterations=10000
                    )

                elif algorithm == "hybrid_metaheuristic":
                    print(f"{RED}      🧭 Hybrid Heuristic...{RESET}")
                    result = hybrid_metaheuristic(
                        graph, cost_matrix
                    )
                
                elif algorithm == "adaptive_metaheuristic":
                    print(f"{RED}      🧭 Hybrid Heuristic...{RESET}")
                    result = adaptive_metaheuristic(
                        graph, cost_matrix
                    )

                else:
                    print(f"{RED}      ⚠️ Algoritmo desconocido: {algorithm}{RESET}")
                    continue

                # Si hay resultado
                if result is not None:
                    algorithm_time = time.time() - algorithm_start
                    result["execution_time"] = algorithm_time

                    # Guardar solución
                    solution_file = os.path.join(
                        output_dir, f"{instance_name}_{algorithm}_solution.json"
                    )
                    save_solution(result["solution"], solution_file, result)

                    # Guardar datos para el reporte
                    results[instance_name][algorithm] = {
                        "cost": result["cost"],
                        "execution_time": algorithm_time,
                        "feasible": result.get("feasible", False),
                        "optimal": result.get("optimal", False),
                        "solution_file": solution_file,
                    }

                    print(
                        f"{GREEN}      ✔ Resultado → Costo: {result['cost']}, "
                        f"Tiempo: {algorithm_time:.2f}s, "
                        f"Factible: {result.get('feasible', True)}{RESET}"
                    )
                else:
                    print(
                        f"{RED}      ❌ No se obtuvo resultado para {algorithm}{RESET}"
                    )

            except Exception as e:
                print(
                    f"{RED}      💥 Error ejecutando {algorithm} en {instance_name}: {e}{RESET}"
                )
                results[instance_name][algorithm] = {
                    "error": str(e),
                    "cost": float("inf"),
                    "execution_time": 0.0,
                    "feasible": False,
                    "optimal": False,
                }

    total_time = time.time() - benchmark_start
    results["_metadata"] = {
        "total_benchmark_time": total_time,
        "instances_evaluated": len(instance_files),
        "algorithms_tested": algorithms,
        "time_limit_per_instance": time_limit,
    }

    print(f"\n{BOLD}{GREEN}🏁 Benchmark completado en {total_time:.2f}s{RESET}")

    benchmark_file = os.path.join(output_dir, "benchmark_results.json")
    with open(benchmark_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"{GREEN}📝 Resultados guardados en:{RESET} {benchmark_file}")

    return results


def compare_algorithms(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara algoritmos dentro de un conjunto homogéneo de resultados.
    Filtra valores NaN y evita warnings estadísticos.
    """
    comparison = {}
    algorithms = set()

    # Descubrir todos los algoritmos usados
    for instance in results:
        if instance.startswith("_"):
            continue
        algorithms.update(results[instance].keys())

    for algorithm in algorithms:
        raw_costs = []
        raw_times = []
        feasible, optimal = 0, 0
        count = 0

        # Extraer valores sin filtrar
        for instance in results:
            if instance.startswith("_"):
                continue
            if algorithm not in results[instance]:
                continue

            data = results[instance][algorithm]

            raw_costs.append(data.get("cost", np.nan))
            raw_times.append(data.get("execution_time", np.nan))

            feasible += 1 if data.get("feasible", True) else 0
            optimal += 1 if data.get("optimal", False) else 0
            count += 1

        # Filtrar NaN y no numéricos
        costs = [c for c in raw_costs if isinstance(c, (int, float)) and not np.isnan(c)]
        times = [t for t in raw_times if isinstance(t, (int, float)) and not np.isnan(t)]

        # Si no queda nada válido → descartar el algoritmo
        if len(costs) == 0 or len(times) == 0:
            continue

        comparison[algorithm] = {
            "costs": costs,
            "times": times,
            "feasible_rate": feasible / count if count else 0,
            "optimal_rate": optimal / count if count else 0,
            "mean_cost": np.mean(costs) if costs else float('inf'),
            "std_cost": np.std(costs) if len(costs) > 1 else 0.0,
            "mean_time": np.mean(times) if times else float('0.0'),
            "std_time": np.std(times) if len(times) > 1 else 0.0,
            "min_cost": np.min(costs if costs else [float('inf')]),
            "max_cost": np.max(costs) if costs else [float('inf')],
            "instances": count,
        }

    return comparison

def generate_benchmark_report(
    comparison: Dict[str, Any], output_file: str = "benchmark_report.md"
):
    """
    Genera un reporte en markdown a partir de las comparaciones de benchmarks
    usando las claves actualizadas de compare_algorithms.
    """
    with open(output_file, "w") as f:
        f.write("# Proyecto DAA - MCCPP Reporte de Benchmarks\n\n")
        f.write("## Comparación de Algoritmos\n\n")
        f.write(
            "| Algoritmo | Costo Promedio | Desv. Costo | Tiempo Promedio (s) | Tasa Factible | Tasa Óptima |\n"
        )
        f.write(
            "|-----------|----------------|-------------|----------------------|---------------|-------------|\n"
        )

        for algo, stats in comparison.items():
            f.write(
                f"| {algo} | {stats['mean_cost']:.2f} | {stats['std_cost']:.2f} | "
                f"{stats['mean_time']:.2f} | {stats['feasible_rate']:.2f} | "
                f"{stats['optimal_rate']:.2f} |\n"
            )

        f.write("\n## Estadísticas Detalladas\n\n")
        for algo, stats in comparison.items():
            f.write(f"### {algo}\n")
            f.write(f"- Instancias evaluadas: {stats['instances']}\n")

            # Rango de costos
            f.write(
                f"- Costo: {stats['min_cost']:.2f} - {stats['max_cost']:.2f} "
                f"(prom: {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f})\n"
            )

            # Rango de tiempos — corregido
            min_time = min(stats["times"])
            max_time = max(stats["times"])
            f.write(
                f"- Tiempo: {min_time:.2f}s - {max_time:.2f}s "
                f"(prom: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s)\n"
            )

            f.write(f"- Soluciones factibles: {stats['feasible_rate']:.2%}\n")
            f.write(f"- Soluciones óptimas: {stats['optimal_rate']:.2%}\n\n")


# def compare_algorithms(results: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Compara algoritmos a través de múltiples instancias para el Proyecto DAA - MCCPP

#     Args:
#         results: resultados de benchmarks de run_benchmark_suite

#     Returns:
#         estadísticas de comparación
#     """
#     comparison = {}
#     algorithms = set()

#     # Recolectar todos los algoritmos utilizados
#     for instance in results:
#         if instance.startswith("_"):
#             continue
#         algorithms.update(results[instance].keys())

#     for algorithm in algorithms:
#         costs = []
#         times = []
#         feasible_count = 0
#         optimal_count = 0
#         instance_count = 0

#         for instance in results:
#             if instance.startswith("_"):
#                 continue
#             if algorithm in results[instance]:
#                 instance_data = results[instance][algorithm]
#                 costs.append(instance_data["cost"])
#                 times.append(instance_data["execution_time"])
#                 if instance_data.get("feasible", True):
#                     feasible_count += 1
#                 if instance_data.get("optimal", False):
#                     optimal_count += 1
#                 instance_count += 1

#         if costs:
#             comparison[algorithm] = {
#                 "average_cost": np.mean(costs),
#                 "std_cost": np.std(costs),
#                 "min_cost": np.min(costs),
#                 "max_cost": np.max(costs),
#                 "average_time": np.mean(times),
#                 "std_time": np.std(times),
#                 "min_time": np.min(times),
#                 "max_time": np.max(times),
#                 "feasible_rate": feasible_count / instance_count,
#                 "optimal_rate": optimal_count / instance_count,
#                 "instances_evaluated": instance_count,
#             }

#     return comparison


# def generate_benchmark_report(
#     comparison: Dict[str, Any], output_file: str = "benchmark_report.md"
# ):
#     """
#     Genera un reporte en markdown a partir de las comparaciones de benchmarks para el Proyecto DAA - MCCPP

#     Args:
#         comparison: estadísticas de comparación de compare_algorithms
#         output_file: ruta del archivo markdown de salida
#     """
#     with open(output_file, "w") as f:
#         f.write("# Proyecto DAA - MCCPP Reporte de Benchmarks\n\n")
#         f.write("## Comparación de Algoritmos\n\n")
#         f.write(
#             "| Algoritmo | Costo Promedio | Desv. Costo | Tiempo Promedio (s) | Tasa Factible | Tasa Óptima |\n"
#         )
#         f.write(
#             "|-----------|----------------|-------------|---------------------|---------------|-------------|\n"
#         )

#         for algo, stats in comparison.items():
#             f.write(
#                 f"| {algo} | {stats['average_cost']:.2f} | {stats['std_cost']:.2f} | "
#                 f"{stats['average_time']:.2f} | {stats['feasible_rate']:.2f} | "
#                 f"{stats['optimal_rate']:.2f} |\n"
#             )

#         f.write("\n## Estadísticas Detalladas\n\n")
#         for algo, stats in comparison.items():
#             f.write(f"### {algo}\n")
#             f.write(f"- Instancias evaluadas: {stats['instances_evaluated']}\n")
#             f.write(
#                 f"- Costo: {stats['min_cost']:.2f} - {stats['max_cost']:.2f} "
#                 f"(prom: {stats['average_cost']:.2f} ± {stats['std_cost']:.2f})\n"
#             )
#             f.write(
#                 f"- Tiempo: {stats['min_time']:.2f}s - {stats['max_time']:.2f}s "
#                 f"(prom: {stats['average_time']:.2f}s ± {stats['std_time']:.2f}s)\n"
#             )
#             f.write(f"- Soluciones factibles: {stats['feasible_rate']:.2%}\n")
#             f.write(f"- Soluciones óptimas: {stats['optimal_rate']:.2%}\n\n")
