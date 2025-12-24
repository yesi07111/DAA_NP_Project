import time
import json
import os
from typing import Dict, List, Any
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils.utils import load_instance, save_solution, ensure_directory
from algorithms.metaheuristic_algorithms import adaptive_metaheuristic, hybrid_metaheuristic, adaptive_simulated_annealing, simulated_annealing, trajectory_search_heuristic
from algorithms.exacts_algorithms import brute_force_solver, backtracking_solver, intelligent_backtracking, ilp_solver, dynamic_programming_tree, dp_interval_graph_solver
from algorithms.approximation_algorithms import weighted_set_cover_approximation, improved_weighted_set_cover, interval_graph_approximation
from algorithms.heuristic_algorithms import largest_first_heuristic, dsatur_heuristic, recursive_largest_first_heuristic


# 🎨 ANSI COLORS
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"


# BENCHMARKS

def run_benchmark_suite(instance_files: List[str], algorithms: List[str], output_dir: str = "results", time_limit: float = 1000000.0,) -> Dict[str, Any]:
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
        try:
            graph, cost_matrix, metadata = load_instance(instance_file)
        except Exception as e:
            print(f"{RED}   💥 Error cargando instancia: {e}{RESET}")
            continue

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
                        print(f"{RED}      ⛔ Grafo > 7 nodos, saltando fuerza bruta{RESET}")
                        
                elif algorithm == "brute_force_backtracking":
                    if len(graph.nodes) < 10:
                        print(f"{YELLOW}      🧮 Fuerza bruta con backtracking...{RESET}")
                        result = backtracking_solver(
                            graph, cost_matrix, time_limit=time_limit
                        )
                    else:
                        print(f"{RED}      ⛔ Grafo > 10 nodos, saltando fuerza bruta{RESET}")
                        
                elif algorithm == "brute_force_intelligent":
                    if len(graph.nodes) < 20:
                        print(f"{YELLOW}      🧮 Fuerza bruta con backtracking y poda...{RESET}")
                        result = intelligent_backtracking(
                            graph, cost_matrix, time_limit=time_limit
                        )
                    else:
                        print(f"{RED}      ⛔ Grafo > 22 nodos, saltando fuerza bruta{RESET}")

                elif algorithm == "ilp_solver":
                    if len(graph.nodes) < 50:
                        print(f"{YELLOW}      🔧 Solucionador ILP...{RESET}")
                        result = ilp_solver(graph, cost_matrix, time_limit=time_limit)
                    else:
                        print(f"{RED}      ⛔ Grafo > 50 nodos, saltando ILP{RESET}")

                elif algorithm == "dynamic_programming_tree":
                    # Verificar que sea un árbol ANTES de ejecutar
                    if not nx.is_tree(graph):
                        print(f"{RED}      ⭕ [SKIP] Grafo no es un árbol{RESET}")
                        result = {
                            'cost': float('inf'),
                            'feasible': False,
                            'execution_time': 0,
                            'skipped': True,
                            'skip_reason': 'Grafo no es un árbol',
                            'algorithm': algorithm
                        }
                    else:
                        print(f"{YELLOW}      🌳 DP para árboles...{RESET}")
                        result = dynamic_programming_tree(graph, cost_matrix)
                    
                    # Verificar que el resultado sea válido
                    if result is None:
                        print(f"{RED}      ❌ DP retornó None{RESET}")
                        result = {
                            'cost': float('inf'),
                            'feasible': False,
                            'execution_time': time.time() - algorithm_start,
                            'error': 'Algoritmo retornó None',
                            'algorithm': 'dynamic_programming_tree'
                        }

                elif algorithm == "weighted_set_cover":
                    print(f"{YELLOW}      🧩 Aproximación Set Cover ponderado...{RESET}")
                    result = weighted_set_cover_approximation(graph, cost_matrix)

                elif algorithm == "improved_weighted_set_cover":
                    print(f"{RED}      🧩 Aproximación Set Cover ponderado mejorado...{RESET}")
                    result = improved_weighted_set_cover(graph, cost_matrix)

                elif algorithm == "interval_approximation":
                    print(f"{YELLOW}      🗂️ Aproximación O(√n) para intervalos...{RESET}")
                    result = interval_graph_approximation(graph, cost_matrix)

                elif algorithm == "dp_interval":
                    if nx.is_chordal(graph):
                        print(f"{YELLOW}      🧮 DP exacto para grafos de intervalo...{RESET}")
                        result = dp_interval_graph_solver(graph, cost_matrix)
                    else:
                        print(f"{RED}      ⛔ Grafo no es cordal, saltando DP intervalo{RESET}")
                        result = {
                            'cost': float('inf'),
                            'feasible': False,
                            'execution_time': 0,
                            'skipped': True,
                            'skip_reason': 'Grafo no es cordal',
                            'algorithm': algorithm
                        }

                elif algorithm == "largest_first":
                    print(f"{YELLOW}      📊 Heurística Largest First...{RESET}")
                    result = largest_first_heuristic(graph, cost_matrix)

                elif algorithm == "dsatur":
                    print(f"{YELLOW}      🎨 Heurística DSATUR...{RESET}")
                    result = dsatur_heuristic(graph, cost_matrix)

                elif algorithm == "rlf":
                    print(f"{YELLOW}      📋 Heurística Recursive Largest First...{RESET}")
                    result = recursive_largest_first_heuristic(graph, cost_matrix)

                elif algorithm == "simulated_annealing":
                    print(f"{YELLOW}      🔥 Simulated Annealing...{RESET}")
                    result = simulated_annealing(
                        graph, cost_matrix, max_iterations=10000
                    )
                
                elif algorithm == "adaptive_simmulated_annealing":
                    print(f"{RED}      🔥 Adaptive Simulated Annealing...{RESET}")
                    result = adaptive_simulated_annealing(graph, cost_matrix)

                elif algorithm == "trajectory_search":
                    print(f"{YELLOW}      🧭 Trajectory Search Heuristic...{RESET}")
                    result = trajectory_search_heuristic(
                        graph, cost_matrix, max_iterations=10000
                    )

                elif algorithm == "hybrid_metaheuristic":
                    print(f"{RED}      🧭 Hybrid Heuristic...{RESET}")
                    result = hybrid_metaheuristic(graph, cost_matrix)
                
                elif algorithm == "adaptive_metaheuristic":
                    print(f"{RED}      🧭 Adaptive Metaheuristic...{RESET}")
                    result = adaptive_metaheuristic(graph, cost_matrix)

                else:
                    print(f"{RED}      ⚠️ Algoritmo desconocido: {algorithm}{RESET}")
                    result = {
                        'cost': float('inf'),
                        'feasible': False,
                        'execution_time': 0,
                        'skipped': True,
                        'skip_reason': 'Algoritmo desconocido',
                        'algorithm': algorithm
                    }

                # Verificar que result no sea None
                if result is None:
                    print(f"{RED}      ❌ Algoritmo retornó None{RESET}")
                    result = {
                        'cost': float('inf'),
                        'feasible': False,
                        'execution_time': time.time() - algorithm_start,
                        'error': 'Algoritmo retornó None',
                        'algorithm': algorithm
                    }

                # Actualizar tiempo si no fue skipped
                if not result.get('skipped', False):
                    algorithm_time = time.time() - algorithm_start
                    if 'execution_time' not in result or result['execution_time'] == 0:
                        result['execution_time'] = algorithm_time
                
                # Asegurar campos básicos
                if 'feasible' not in result:
                    result['feasible'] = False
                if 'optimal' not in result:
                    result['optimal'] = False

                # Guardar resultado (incluido si fue skipped)
                if result.get('skipped', False):
                    # Algoritmo fue saltado
                    results[instance_name][algorithm] = {
                        "cost": float('inf'),
                        "execution_time": 0,
                        "feasible": False,
                        "optimal": False,
                        "skipped": True,
                        "skip_reason": result.get('skip_reason', 'No aplicable'),
                        "error": result.get('error', result.get('skip_reason', 'Algoritmo no aplicable'))
                    }
                    print(f"{YELLOW}      ⏭️ Skipped: {result.get('skip_reason', 'No aplicable')}{RESET}")
                    
                elif 'solution' in result and result.get('feasible', False):
                    # Solución válida encontrada
                    solution_file = os.path.join(
                        output_dir, f"{instance_name}_{algorithm}_solution.json"
                    )
                    save_solution(result["solution"], solution_file, result)

                    results[instance_name][algorithm] = {
                        "cost": result.get("cost", float('inf')),
                        "execution_time": result['execution_time'],
                        "feasible": result.get("feasible", False),
                        "optimal": result.get("optimal", False),
                        "solution_file": solution_file,
                        "operations": result.get("operations", result.get("iterations", 'N/A'))
                    }

                    print(
                        f"{GREEN}      ✓ Resultado → Costo: {result['cost']:.2f}, "
                        f"Tiempo: {result['execution_time']:.4f}s, "
                        f"Factible: {result.get('feasible', True)}{RESET}"
                    )
                else:
                    # Resultado sin solución válida
                    results[instance_name][algorithm] = {
                        "cost": result.get("cost", float('inf')),
                        "execution_time": result['execution_time'],
                        "feasible": False,
                        "optimal": False,
                        "error": result.get('error', 'Sin solución válida'),
                        "operations": result.get("operations", 'N/A')
                    }
                    print(f"{RED}      ❌ No se obtuvo solución válida para {algorithm}{RESET}")

            except Exception as e:
                print(f"{RED}      💥 Error ejecutando {algorithm} en {instance_name}: {e}{RESET}")
                import traceback
                traceback.print_exc()
                
                results[instance_name][algorithm] = {
                    "error": str(e),
                    "cost": float("inf"),
                    "execution_time": time.time() - algorithm_start,
                    "feasible": False,
                    "optimal": False,
                    "operations": 'N/A'
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
    sIncluye evaluación de:
    - Calidad de solución (Ratio R = ZAlg/Z*)
    - Tiempo de ejecución
    - Escalabilidad
    - Sensibilidad a densidad y distribución de costos
    
    Filtra valores NaN y evita warnings estadísticos.
    """
    comparison = {}
    algorithms = set()

    # Descubrir todos los algoritmos usados
    for instance_name in results:
        if instance_name.startswith("_"):
            continue
        instance_data = results[instance_name]
        
        # Si es estructura vieja con "runs"
        if isinstance(instance_data, dict) and "runs" in instance_data:
            algorithms.update(instance_data["runs"].keys())
        else:
            # Si es estructura nueva sin "runs"
            algorithms.update([k for k in instance_data.keys() if k not in ["metadata", "_metadata", "instance"]])

    # Si no hay algoritmos, retornar vacío
    if not algorithms:
        print(f"{YELLOW}⚠ No se encontraron algoritmos para comparar{RESET}")
        return {}

    for algorithm in algorithms:
        raw_costs = []
        raw_times = []
        quality_ratios = []  # R = ZAlg / Z*
        feasible, optimal = 0, 0
        count = 0
        skipped_count = 0
        
        # Datos para análisis de sensibilidad
        instance_data_list = []  # Lista de (densidad, cost_distribution, ratio, time, n_vertices)

        # Extraer valores sin filtrar
        for instance_name in results:
            if instance_name.startswith("_"):
                continue
            
            instance_dict = results[instance_name]
            
            # Extraer metadata de la instancia
            metadata = instance_dict.get("metadata", {})
            if not metadata and "_metadata" in instance_dict:
                metadata = instance_dict["_metadata"]
            
            # Determinar si es estructura antigua o nueva
            if "runs" in instance_dict:
                # Estructura antigua (con "runs")
                if algorithm not in instance_dict["runs"]:
                    continue
                data = instance_dict["runs"][algorithm]
            else:
                # Estructura nueva (sin "runs")
                if algorithm not in instance_dict:
                    continue
                data = instance_dict[algorithm]
            
            # Contar skipped
            if data.get("skipped", False):
                skipped_count += 1
                continue
            
            cost = data.get("cost", np.nan)
            execution_time = data.get("execution_time", np.nan)

            raw_costs.append(cost)
            raw_times.append(execution_time)

            feasible += 1 if data.get("feasible", False) else 0
            optimal += 1 if data.get("optimal", False) else 0
            count += 1
            
            # Calcular ratio de calidad (R = ZAlg / Z_opt)
            # Z_opt es el mínimo obtenido por cualquier algoritmo en esa instancia
            # Ignorar algoritmos que fueron skipped
            best_cost = float('inf')
            
            if "runs" in instance_dict:
                for algo in instance_dict["runs"]:
                    algo_data = instance_dict["runs"][algo]
                    if algo_data.get("skipped", False):  # Ignorar skipped
                        continue
                    algo_cost = algo_data.get("cost", float('inf'))
                    if isinstance(algo_cost, (int, float)) and algo_cost != float('inf'):
                        best_cost = min(best_cost, algo_cost)
            else:
                for algo in instance_dict:
                    if algo not in ["metadata", "_metadata", "instance"]:
                        algo_data = instance_dict[algo]
                        if algo_data.get("skipped", False):  # Ignorar skipped
                            continue
                        algo_cost = algo_data.get("cost", float('inf'))
                        if isinstance(algo_cost, (int, float)) and algo_cost != float('inf'):
                            best_cost = min(best_cost, algo_cost)
            
            if best_cost < float('inf') and isinstance(cost, (int, float)) and cost != float('inf'):
                ratio = cost / best_cost
                quality_ratios.append(ratio)
                
                # Extraer metadatos de la instancia
                density = metadata.get("density", 0.0)
                n_vertices = metadata.get("n_vertices", 0)
                cost_distribution = metadata.get("cost_distribution", "unknown")
                
                instance_data_list.append({
                    "density": density,
                    "cost_distribution": cost_distribution,
                    "ratio": ratio,
                    "time": execution_time if isinstance(execution_time, (int, float)) else 0.0,
                    "n_vertices": n_vertices,
                    "cost": cost
                })

        # Filtrar NaN y no numéricos
        costs = [c for c in raw_costs if isinstance(c, (int, float)) and not np.isnan(c) and c != float('inf')]
        times = [t for t in raw_times if isinstance(t, (int, float)) and not np.isnan(t) and t >= 0]
        
        # Análisis de sensibilidad a densidad
        density_sensitivity = _analyze_density_sensitivity(instance_data_list)
        
        # Análisis de sensibilidad a distribución de costos
        cost_distribution_sensitivity = _analyze_cost_distribution_sensitivity(instance_data_list)
        
        # Análisis de escalabilidad
        scalability = _analyze_scalability(instance_data_list)

        # Si no queda nada válido → almacenar info mínima
        if len(costs) == 0:
            comparison[algorithm] = {
                "costs": [],
                "times": times if times else [],
                "feasible_rate": 0.0,
                "optimal_rate": 0.0,
                "mean_cost": float('inf'),
                "std_cost": 0.0,
                "mean_time": np.mean(times) if times else 0.0,
                "median_time": np.median(times) if times else 0.0,
                "std_time": np.std(times) if len(times) > 1 else 0.0,
                "min_time": np.min(times) if times else 0.0,
                "max_time": np.max(times) if times else 0.0,
                "min_cost": float('inf'),
                "max_cost": float('inf'),
                "instances": count + skipped_count,
                "valid_instances": 0,
                "skipped_instances": skipped_count,
                "quality_ratios": [],
                "mean_quality_ratio": float('inf'),
                "median_quality_ratio": float('inf'),
                "std_quality_ratio": 0.0,
                "min_quality_ratio": float('inf'),
                "max_quality_ratio": float('inf'),
                "density_sensitivity": {},
                "cost_distribution_sensitivity": {},
                "scalability": {}
            }
            continue

        comparison[algorithm] = {
            "costs": costs,
            "times": times,
            "feasible_rate": feasible / count if count else 0,
            "optimal_rate": optimal / count if count else 0,
            "mean_cost": np.mean(costs),
            "std_cost": np.std(costs) if len(costs) > 1 else 0.0,
            "min_cost": np.min(costs),
            "max_cost": np.max(costs),
            "instances": count + skipped_count,
            "valid_instances": len(costs),
            "skipped_instances": skipped_count,
            # Estadísticas de tiempo
            "mean_time": np.mean(times) if times else 0.0,
            "median_time": np.median(times) if times else 0.0,
            "std_time": np.std(times) if len(times) > 1 else 0.0,
            "min_time": np.min(times) if times else 0.0,
            "max_time": np.max(times) if times else 0.0,
            # Calidad de solución (Ratio R = ZAlg / Z*)
            "quality_ratios": quality_ratios,
            "mean_quality_ratio": np.mean(quality_ratios) if quality_ratios else float('inf'),
            "median_quality_ratio": np.median(quality_ratios) if quality_ratios else float('inf'),
            "std_quality_ratio": np.std(quality_ratios) if len(quality_ratios) > 1 else 0.0,
            "min_quality_ratio": np.min(quality_ratios) if quality_ratios else float('inf'),
            "max_quality_ratio": np.max(quality_ratios) if quality_ratios else float('inf'),
            # Sensibilidad a densidad
            "density_sensitivity": density_sensitivity,
            # Sensibilidad a distribución de costos
            "cost_distribution_sensitivity": cost_distribution_sensitivity,
            # Escalabilidad
            "scalability": scalability
        }

    return comparison

def _analyze_density_sensitivity(instance_data: List[Dict]) -> Dict[str, Any]:
    """
    Analiza cómo varía el rendimiento (ratio, tiempo) con la densidad.
    """
    if not instance_data:
        return {}
    
    # Agrupar por rangos de densidad
    density_groups = {
        "sparse": [],      # d < 0.2
        "moderate": [],    # 0.2 <= d < 0.5
        "dense": [],       # d >= 0.5
    }
    
    for data in instance_data:
        density = data.get("density", 0.0)
        if density < 0.2:
            density_groups["sparse"].append(data)
        elif density < 0.5:
            density_groups["moderate"].append(data)
        else:
            density_groups["dense"].append(data)
    
    sensitivity = {}
    for category, group in density_groups.items():
        if group:
            ratios = [d["ratio"] for d in group]
            times = [d["time"] for d in group]
            sensitivity[category] = {
                "count": len(group),
                "avg_ratio": np.mean(ratios),
                "avg_time": np.mean(times),
                "std_ratio": np.std(ratios) if len(ratios) > 1 else 0.0,
                "std_time": np.std(times) if len(times) > 1 else 0.0
            }
    
    return sensitivity

def _analyze_cost_distribution_sensitivity(instance_data: List[Dict]) -> Dict[str, Any]:
    """
    Analiza cómo varía el rendimiento según la distribución de costos en la matriz.
    """
    if not instance_data:
        return {}
    
    # Agrupar por tipo de distribución de costos
    distribution_groups = {}
    
    for data in instance_data:
        dist = data.get("cost_distribution", "unknown")
        if dist not in distribution_groups:
            distribution_groups[dist] = []
        distribution_groups[dist].append(data)
    
    sensitivity = {}
    for dist, group in distribution_groups.items():
        if group:
            ratios = [d["ratio"] for d in group]
            times = [d["time"] for d in group]
            sensitivity[dist] = {
                "count": len(group),
                "avg_ratio": np.mean(ratios),
                "avg_time": np.mean(times),
                "std_ratio": np.std(ratios) if len(ratios) > 1 else 0.0,
                "std_time": np.std(times) if len(times) > 1 else 0.0
            }
    
    return sensitivity

def _analyze_scalability(instance_data: List[Dict]) -> Dict[str, Any]:
    """
    Analiza la escalabilidad: cómo varía tiempo y ratio con el tamaño (n_vertices).
    """
    if not instance_data:
        return {}
    
    # Agrupar por tamaño
    size_groups = {}
    
    for data in instance_data:
        n = data.get("n_vertices", 0)
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(data)
    
    # Ordenar por n
    scalability = {}
    for n in sorted(size_groups.keys()):
        group = size_groups[n]
        ratios = [d["ratio"] for d in group]
        times = [d["time"] for d in group]
        
        scalability[f"n_{n}"] = {
            "count": len(group),
            "avg_ratio": np.mean(ratios),
            "avg_time": np.mean(times),
            "std_ratio": np.std(ratios) if len(ratios) > 1 else 0.0,
            "std_time": np.std(times) if len(times) > 1 else 0.0,
            "min_time": np.min(times) if times else 0.0,
            "max_time": np.max(times) if times else 0.0
        }
    
    return scalability

def generate_benchmark_report(
    comparison: Dict[str, Any], output_file: str = "benchmark_report.md"
):
    """
    Genera un reporte exhaustivo en markdown a partir de las comparaciones de benchmarks.
    Incluye:
    - Ratio de calidad (R = ZAlg/Z*)
    - Análisis de escalabilidad
    - Sensibilidad a densidad
    - Sensibilidad a distribución de costos
    """
    with open(output_file, "w") as f:
        f.write("# Proyecto DAA - MCCPP Reporte de Evaluación Estadística\n\n")
        
        if not comparison:
            f.write("## ⚠️ Sin Resultados\n\n")
            f.write("No se obtuvieron resultados válidos en este experimento.\n")
            f.write("Posibles causas:\n")
            f.write("- Todos los algoritmos fallaron\n")
            f.write("- Las instancias no son compatibles con los algoritmos\n")
            f.write("- Errores en la ejecución\n\n")
            return
        
        # ==========================
        # 1. TABLA RESUMEN GENERAL
        # ==========================
        f.write("## 1. Resumen General de Algoritmos\n\n")
        f.write(
            "| Algoritmo | Ratio R* | Desv. R | Tiempo (s) | Desv.T | Tasa Opt | Inst. |\n"
        )
        f.write(
            "|-----------|----------|---------|-----------|--------|---------|-------|\n"
        )

        for algo, stats in sorted(comparison.items()):
            ratio_str = f"{stats['mean_quality_ratio']:.4f}" if stats['mean_quality_ratio'] != float('inf') else "∞"
            ratio_std = f"{stats['std_quality_ratio']:.4f}" if stats['std_quality_ratio'] != float('inf') else "N/A"
            time_str = f"{stats['mean_time']:.4f}"
            time_std = f"{stats['std_time']:.4f}"
            opt_rate = f"{stats['optimal_rate']:.2%}"
            
            f.write(
                f"| {algo} | {ratio_str} | {ratio_std} | {time_str} | {time_std} | {opt_rate} | {stats['valid_instances']}/{stats['instances']} |\n"
            )

        f.write("\n### Leyenda:\n")
        f.write("- **Ratio R*** = ZAlg/Z* (1.0 = óptimo, >1.0 = subóptima)\n")
        f.write("- **Desv. R** = Desviación estándar del ratio\n")
        f.write("- **Tiempo (s)** = Promedio de tiempo de ejecución\n")
        f.write("- **Desv.T** = Desviación estándar del tiempo\n")
        f.write("- **Tasa Opt** = Porcentaje de instancias donde fue óptimo\n")
        f.write("- **Inst.** = Instancias válidas/totales\n\n")

        # ==========================
        # 2. ESTADÍSTICAS DETALLADAS
        # ==========================
        f.write("## 2. Estadísticas Detalladas por Algoritmo\n\n")
        for algo, stats in sorted(comparison.items()):
            f.write(f"### {algo}\n\n")
            f.write(f"**Estadísticas Generales:**\n")
            f.write(f"- Instancias evaluadas: {stats['instances']}\n")
            f.write(f"- Instancias con solución válida: {stats['valid_instances']}\n")
            f.write(f"- Tasa de factibilidad: {stats['feasible_rate']:.2%}\n")
            f.write(f"- Tasa de optimalidad: {stats['optimal_rate']:.2%}\n\n")

            if stats['valid_instances'] > 0:
                # Costo
                f.write(f"**Costo de Solución:**\n")
                cost_min = f"{stats['min_cost']:.2f}" if stats['min_cost'] != float('inf') else "∞"
                cost_max = f"{stats['max_cost']:.2f}" if stats['max_cost'] != float('inf') else "∞"
                cost_mean = f"{stats['mean_cost']:.2f}" if stats['mean_cost'] != float('inf') else "∞"
                f.write(f"- Rango: {cost_min} - {cost_max}\n")
                f.write(f"- Promedio: {cost_mean} ± {stats['std_cost']:.2f}\n\n")

                # Tiempo
                f.write(f"**Tiempo de Ejecución:**\n")
                f.write(f"- Rango: {stats['min_time']:.6f}s - {stats['max_time']:.6f}s\n")
                f.write(f"- Promedio: {stats['mean_time']:.6f}s ± {stats['std_time']:.6f}s\n")
                f.write(f"- Mediana: {stats['median_time']:.6f}s\n\n")

                # Ratio de Calidad (R = ZAlg / Z*)
                f.write(f"**Calidad de Solución (Ratio R = ZAlg/Z*):**\n")
                if stats['mean_quality_ratio'] != float('inf'):
                    f.write(f"- Media: {stats['mean_quality_ratio']:.4f}\n")
                    f.write(f"- Mediana: {stats['median_quality_ratio']:.4f}\n")
                    f.write(f"- Rango: {stats['min_quality_ratio']:.4f} - {stats['max_quality_ratio']:.4f}\n")
                    std_ratio = stats.get('std_quality_ratio', 0)
                    if isinstance(std_ratio, (int, float)) and std_ratio != float('inf'):
                        f.write(f"- Desv. Estándar: {std_ratio:.4f}\n")
                    
                    if stats['mean_quality_ratio'] == 1.0:
                        f.write(f"- **Status**: ✓ ÓPTIMO (siempre encuentra solución óptima)\n\n")
                    elif stats['mean_quality_ratio'] < 1.05:
                        f.write(f"- **Status**: ✓ EXCELENTE (muy cercano al óptimo)\n\n")
                    elif stats['mean_quality_ratio'] < 1.20:
                        f.write(f"- **Status**: ✓ BUENO (generalmente cercano al óptimo)\n\n")
                    else:
                        f.write(f"- **Status**: ⚠ ACEPTABLE (solucion subóptima)\n\n")
                else:
                    f.write(f"- ∞ (No hay óptimo conocido para comparación)\n\n")

            else:
                f.write(f"- ⚠️ No se obtuvieron soluciones válidas para este algoritmo\n\n")

        # ==========================
        # 3. ANÁLISIS DE ESCALABILIDAD
        # ==========================
        f.write("## 3. Análisis de Escalabilidad\n\n")
        f.write("Comportamiento del algoritmo según el tamaño de la instancia (n):\n\n")
        
        for algo, stats in sorted(comparison.items()):
            scalability = stats.get('scalability', {})
            if scalability:
                f.write(f"### {algo}\n\n")
                f.write("| Tamaño | Instancias | Ratio Promedio | Tiempo Promedio | Tiempo Máx |\n")
                f.write("|--------|------------|----------------|-----------------|------------|\n")
                
                for size_key in sorted(scalability.keys(), key=lambda x: int(x.split('_')[1])):
                    size_data = scalability[size_key]
                    count = size_data.get('count', 0)
                    ratio = size_data.get('avg_ratio', 0)
                    time_avg = size_data.get('avg_time', 0)
                    time_max = size_data.get('max_time', 0)
                    
                    f.write(f"| {size_key.replace('n_', '')} | {count} | {ratio:.4f} | {time_avg:.6f}s | {time_max:.6f}s |\n")
                
                f.write("\n")

        # ==========================
        # 4. ANÁLISIS DE SENSIBILIDAD A DENSIDAD
        # ==========================
        f.write("## 4. Sensibilidad a la Densidad del Grafo\n\n")
        f.write("Cómo varía el rendimiento según la densidad de aristas:\n\n")
        
        for algo, stats in sorted(comparison.items()):
            density_sens = stats.get('density_sensitivity', {})
            if density_sens:
                f.write(f"### {algo}\n\n")
                f.write("| Densidad | Instancias | Ratio Promedio | Tiempo Promedio |\n")
                f.write("|----------|------------|----------------|----------------|\n")
                
                for density_cat in ['sparse', 'moderate', 'dense']:
                    if density_cat in density_sens:
                        data = density_sens[density_cat]
                        count = data.get('count', 0)
                        ratio = data.get('avg_ratio', 0)
                        time_avg = data.get('avg_time', 0)
                        
                        f.write(f"| {density_cat.capitalize()} | {count} | {ratio:.4f} | {time_avg:.6f}s |\n")
                
                f.write("\n**Interpretación:**\n")
                density_sens_list = list(density_sens.items())
                if len(density_sens_list) >= 2:
                    sparse_ratio = density_sens.get('sparse', {}).get('avg_ratio', 0) or density_sens.get('moderate', {}).get('avg_ratio', 0)
                    dense_ratio = density_sens.get('dense', {}).get('avg_ratio', 0)
                    
                    if sparse_ratio < dense_ratio * 0.95:
                        f.write(f"- **Mejor en grafos sparse**: Algoritmo es más eficiente con baja densidad\n\n")
                    elif dense_ratio < sparse_ratio * 0.95:
                        f.write(f"- **Mejor en grafos densos**: Algoritmo es más eficiente con alta densidad\n\n")
                    else:
                        f.write(f"- **Insensible a densidad**: Rendimiento estable independientemente de densidad\n\n")

        # ==========================
        # 5. ANÁLISIS DE DISTRIBUCIÓN DE COSTOS
        # ==========================
        f.write("## 5. Sensibilidad a Distribución de Costos\n\n")
        f.write("Cómo varía el rendimiento según la distribución de valores en la matriz de costos:\n\n")
        
        for algo, stats in sorted(comparison.items()):
            cost_dist_sens = stats.get('cost_distribution_sensitivity', {})
            if cost_dist_sens:
                f.write(f"### {algo}\n\n")
                f.write("| Distribución | Instancias | Ratio Promedio | Tiempo Promedio |\n")
                f.write("|--------------|------------|----------------|----------------|\n")
                
                for dist_type in sorted(cost_dist_sens.keys()):
                    data = cost_dist_sens[dist_type]
                    count = data.get('count', 0)
                    ratio = data.get('avg_ratio', 0)
                    time_avg = data.get('avg_time', 0)
                    
                    f.write(f"| {dist_type} | {count} | {ratio:.4f} | {time_avg:.6f}s |\n")
                
                f.write("\n")

        # ==========================
        # 6. CONCLUSIONES Y RECOMENDACIONES
        # ==========================
        f.write("## 6. Conclusiones y Recomendaciones\n\n")
        
        # Encontrar el mejor algoritmo por ratio
        best_algo = min(comparison.items(), key=lambda x: x[1].get('mean_quality_ratio', float('inf')))
        f.write(f"**Mejor en Calidad**: {best_algo[0]} (R = {best_algo[1]['mean_quality_ratio']:.4f})\n\n")
        
        # Encontrar el más rápido
        fastest_algo = min(comparison.items(), key=lambda x: x[1].get('mean_time', float('inf')))
        f.write(f"**Más Rápido**: {fastest_algo[0]} ({fastest_algo[1]['mean_time']:.6f}s promedio)\n\n")
        
        f.write("### Recomendaciones por Caso de Uso:\n\n")
        f.write("- **Para máxima calidad de solución**: Usar " + best_algo[0] + "\n")
        f.write("- **Para máxima velocidad**: Usar " + fastest_algo[0] + "\n")
        f.write("- **Balance calidad-velocidad**: Analizar curva de Pareto en gráficos\n\n")
        
        f.write("---\n")
        f.write(f"*Reporte generado el {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

# COMPARISON PLOTS

def create_algorithm_comparison_plot(comparison_data: Dict[str, Any], 
                                   output_file: str = "algorithm_comparison.png") -> None:
    """
    Plot de comparación basado en compare_algorithms() actualizado.
    Incluye gráfico de ratio de calidad.
    """
    algorithms = list(comparison_data.keys())

    import math

    # Filtrar valores NaN o None, reemplazarlos por 0 o un valor neutral seguro
    clean_costs = []
    clean_times = []
    clean_feasible = []
    clean_quality_ratios = []
    clean_algorithms = []

    
    avg_costs = [comparison_data[a]['mean_cost'] for a in algorithms]
    avg_times = [comparison_data[a]['mean_time'] for a in algorithms]
    feasible_rates = [comparison_data[a]['feasible_rate'] for a in algorithms]
    quality_ratios = [comparison_data[a]['mean_quality_ratio'] for a in algorithms]

    for algo, c, t, f, r in zip(algorithms, avg_costs, avg_times, feasible_rates, quality_ratios):
        if c is None or math.isnan(c):
            c = 0.0
        if t is None or math.isnan(t):
            t = 0.0
        if f is None or math.isnan(f):
            f = 0.0
        if r is None or math.isnan(r) or r == float('inf'):
            r = 0.0

        clean_algorithms.append(algo)
        clean_costs.append(c)
        clean_times.append(t)
        clean_feasible.append(f)
        clean_quality_ratios.append(r)

    algorithms = clean_algorithms
    avg_costs = clean_costs
    avg_times = clean_times
    feasible_rates = clean_feasible
    quality_ratios = clean_quality_ratios

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # ---------------------------------------------------
    # 1) COSTO PROMEDIO
    # ---------------------------------------------------
    bars1 = ax1.bar(range(len(algorithms)), avg_costs, color='steelblue')
    ax1.set_title("Costo Promedio de Solución", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Costo")

    # Fix ticks
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # ---------------------------------------------------
    # 2) RATIO DE CALIDAD (NUEVO)
    # ---------------------------------------------------
    bars2 = ax2.bar(range(len(algorithms)), quality_ratios, color='forestgreen')
    ax2.set_title("Ratio de Calidad (R = ZAlg/Z*)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Ratio (1.0 = Óptimo)")
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Óptimo')

    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha="right")
    ax2.legend()

    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom", fontsize=9)

    # ---------------------------------------------------
    # 3) TIEMPO PROMEDIO
    # ---------------------------------------------------
    bars3 = ax3.bar(range(len(algorithms)), avg_times, color='coral')
    ax3.set_title("Tiempo Promedio de Ejecución", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Tiempo (s)")

    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h, f"{h:.4f}s", ha="center", va="bottom", fontsize=9)

    # ---------------------------------------------------
    # 4) TASA FACTIBLE
    # ---------------------------------------------------
    bars4 = ax4.bar(range(len(algorithms)), feasible_rates, color='mediumpurple')
    ax4.set_title("Tasa de Factibilidad", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Tasa")
    ax4.set_ylim(0, 1)

    ax4.set_xticks(range(len(algorithms)))
    ax4.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars4:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2%}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_profile(instance_results: Dict[str, Any], 
                             output_file: str = "performance_profile.png") -> None:
    """
    Create performance profile plot for DAA Project - MCCPP
    
    Args:
        instance_results: results from benchmark suite
        output_file: output plot file path
    """
    # Extract algorithms and their costs per instance
    algorithms = set()
    instance_costs = {}
    
    for instance_name, instance_data in instance_results.items():
        if instance_name.startswith('_'):
            continue
        instance_costs[instance_name] = {}
        for algo, algo_data in instance_data.items():
            if algo == "metadata":
                continue
            algorithms.add(algo)
            instance_costs[instance_name][algo] = algo_data['cost']
    
    algorithms = list(algorithms)
    
    # Calculate performance ratios
    performance_ratios = {}
    for instance_name, costs in instance_costs.items():
        min_cost = min(costs.values())
        for algo in algorithms:
            if algo in costs:
                ratio = costs[algo] / min_cost
                if algo not in performance_ratios:
                    performance_ratios[algo] = []
                performance_ratios[algo].append(ratio)
    
    # Create performance profile
    plt.figure(figsize=(10, 6))
    tau_values = np.linspace(1, 2, 100)
    
    for algo in algorithms:
        if algo in performance_ratios and performance_ratios[algo]:
            ratios = sorted(performance_ratios[algo])
            profile = [np.mean([r <= tau for r in ratios]) for tau in tau_values]
            plt.plot(tau_values, profile, label=algo, linewidth=2)
    
    plt.xlabel('Performance Ratio τ')
    plt.ylabel('Fraction of Instances')
    plt.title('Performance Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_plots(comparison_group: Dict[str, Any],
                              benchmark_results: Dict[str, Any],
                              output_dir: str):
    """
    Genera múltiples gráficos de comparación incluyendo sensibilidad y escalabilidad.
    """
    os.makedirs(output_dir, exist_ok=True)

    create_algorithm_comparison_plot(
        comparison_group,
        os.path.join(output_dir, "algorithm_comparison.png")
    )

    create_performance_profile(
        benchmark_results,
        os.path.join(output_dir, "performance_profile.png")
    )
    
    # Gráficos de sensibilidad y escalabilidad
    create_scalability_plot(
        comparison_group,
        os.path.join(output_dir, "scalability_analysis.png")
    )
    
    create_density_sensitivity_plot(
        comparison_group,
        os.path.join(output_dir, "density_sensitivity.png")
    )
    
    create_quality_ratio_plot(
        comparison_group,
        os.path.join(output_dir, "quality_ratio_analysis.png")
    )

def create_scalability_plot(comparison_data: Dict[str, Any], 
                           output_file: str = "scalability_analysis.png") -> None:
    """
    Gráfico de escalabilidad: cómo varía tiempo y ratio con n.
    """
    algorithms = list(comparison_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Recolectar datos de escalabilidad
    for algo in algorithms:
        scalability = comparison_data[algo].get('scalability', {})
        if scalability:
            sizes = []
            times = []
            ratios = []
            
            for size_key in sorted(scalability.keys(), key=lambda x: int(x.split('_')[1])):
                n = int(size_key.split('_')[1])
                data = scalability[size_key]
                
                sizes.append(n)
                times.append(data.get('avg_time', 0))
                ratios.append(data.get('avg_ratio', 0))
            
            if sizes:
                # Tiempo vs Tamaño
                ax1.plot(sizes, times, marker='o', label=algo, linewidth=2)
                
                # Ratio vs Tamaño
                ax2.plot(sizes, ratios, marker='s', label=algo, linewidth=2)
    
    ax1.set_xlabel("Tamaño de Instancia (n vértices)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Tiempo Promedio (s)", fontsize=11, fontweight='bold')
    ax1.set_title("Escalabilidad: Tiempo vs Tamaño", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    ax2.set_xlabel("Tamaño de Instancia (n vértices)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Ratio de Calidad (ZAlg/Z*)", fontsize=11, fontweight='bold')
    ax2.set_title("Escalabilidad: Calidad vs Tamaño", fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Óptimo')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_density_sensitivity_plot(comparison_data: Dict[str, Any], 
                                   output_file: str = "density_sensitivity.png") -> None:
    """
    Gráfico de sensibilidad a densidad.
    """
    algorithms = list(comparison_data.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    density_categories = ['sparse', 'moderate', 'dense']
    x = np.arange(len(density_categories))
    width = 0.15
    
    ratios_by_algo = {}
    times_by_algo = {}
    
    for algo in algorithms:
        density_sens = comparison_data[algo].get('density_sensitivity', {})
        ratios = []
        times = []
        
        for cat in density_categories:
            if cat in density_sens:
                ratios.append(density_sens[cat].get('avg_ratio', 0))
                times.append(density_sens[cat].get('avg_time', 0))
            else:
                ratios.append(0)
                times.append(0)
        
        ratios_by_algo[algo] = ratios
        times_by_algo[algo] = times
    
    # Gráfico de ratios
    for i, algo in enumerate(algorithms):
        ax1.bar(x + i*width, ratios_by_algo[algo], width, label=algo)
    
    ax1.set_xlabel("Densidad del Grafo", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Ratio de Calidad Promedio", fontsize=11, fontweight='bold')
    ax1.set_title("Sensibilidad a Densidad: Calidad", fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax1.set_xticklabels([cat.capitalize() for cat in density_categories])
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Óptimo')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Gráfico de tiempos
    for i, algo in enumerate(algorithms):
        ax2.bar(x + i*width, times_by_algo[algo], width, label=algo)
    
    ax2.set_xlabel("Densidad del Grafo", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Tiempo Promedio (s)", fontsize=11, fontweight='bold')
    ax2.set_title("Sensibilidad a Densidad: Tiempo", fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax2.set_xticklabels([cat.capitalize() for cat in density_categories])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_quality_ratio_plot(comparison_data: Dict[str, Any], 
                             output_file: str = "quality_ratio_analysis.png") -> None:
    """
    Análisis detallado del ratio de calidad (ZAlg/Z*) por algoritmo.
    """
    algorithms = list(comparison_data.keys())
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1) Ratio promedio con desviación
    means = [comparison_data[a].get('mean_quality_ratio', 0) for a in algorithms]
    stds = [comparison_data[a].get('std_quality_ratio', 0) for a in algorithms]
    
    bars = ax1.bar(range(len(algorithms)), means, yerr=stds, capsize=5, 
                   color='forestgreen', alpha=0.7, error_kw={'elinewidth': 2})
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Óptimo')
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.set_ylabel("Ratio (1.0 = Óptimo)", fontsize=11, fontweight='bold')
    ax1.set_title("Ratio de Calidad Promedio ± Desv.Est.", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if mean < float('inf') and mean > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, mean, f"{mean:.4f}", 
                    ha="center", va="bottom", fontsize=9)
    
    # 2) Rango de ratios (min-max)
    mins = [comparison_data[a].get('min_quality_ratio', float('inf')) for a in algorithms]
    maxs = [comparison_data[a].get('max_quality_ratio', 0) for a in algorithms]
    
    x = np.arange(len(algorithms))
    ax2.scatter(x, mins, color='green', s=100, marker='_', linewidth=3, label='Mín')
    ax2.scatter(x, maxs, color='red', s=100, marker='_', linewidth=3, label='Máx')
    
    for i in range(len(algorithms)):
        if mins[i] < float('inf') and maxs[i] < float('inf'):
            ax2.plot([i, i], [mins[i], maxs[i]], 'k-', alpha=0.3)
    
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha='right')
    ax2.set_ylabel("Ratio", fontsize=11, fontweight='bold')
    ax2.set_title("Rango de Ratios (Mín-Máx)", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3) Tasa de optimalidad
    opt_rates = [comparison_data[a].get('optimal_rate', 0) * 100 for a in algorithms]
    bars3 = ax3.bar(range(len(algorithms)), opt_rates, color='steelblue', alpha=0.7)
    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha='right')
    ax3.set_ylabel("Porcentaje (%)", fontsize=11, fontweight='bold')
    ax3.set_title("Tasa de Optimalidad (% de instancias óptimas)", fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 110)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, rate in zip(bars3, opt_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, rate, f"{rate:.1f}%", 
                ha="center", va="bottom", fontsize=9)
    
    # 4) Box plot de ratios
    quality_ratio_lists = []
    valid_algos = []
    
    for algo in algorithms:
        ratios = comparison_data[algo].get('quality_ratios', [])
        if ratios and len(ratios) > 0:
            # Filtrar infinitos
            valid_ratios = [r for r in ratios if r < float('inf')]
            if valid_ratios:
                quality_ratio_lists.append(valid_ratios)
                valid_algos.append(algo)
    
    if quality_ratio_lists:
        bp = ax4.boxplot(quality_ratio_lists, labels=valid_algos, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Óptimo')
        ax4.set_ylabel("Ratio", fontsize=11, fontweight='bold')
        ax4.set_title("Distribución de Ratios (Box Plot)", fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
