"""
Ejecutor de experimentos para el Proyecto DAA - MCCPP

Modificado para:
 - separar instancias en: árboles; estructura+intervalos+special; y todas.
 - definir 3 grupos de algoritmos:
    * dp_tree_algorithms -> solo dynamic_programming_tree (para árboles)
    * structural_algorithms -> structural_bipartite, structural_interval
    * general_algorithms -> conjunto completo de algoritmos generales
 - ejecutar run_benchmark_suite 3 veces (una por experimento) y comparar cada
   experimento por separado (Paso 3a/3b/3c).
 - desde el Paso 4 en adelante queda igual.
"""

import os
import time
import json
from typing import Dict, List, Any
from src.instances_gen.generators import (
    generate_erdos_renyi_instances,
    generate_structured_instances,
    generate_tree_instances,
)
from src.instances_gen.special_cases import generate_special_case_instances
from src.instances_gen.interval_graphs import (
    generate_interval_graph_instances,
)
from src.evaluation.benchmarks import (
    run_benchmark_suite,
    compare_algorithms,
    generate_benchmark_report,
)
from src.evaluation.scalability_tests import (
    compare_scalability,
    plot_scalability_results,
    analyze_computational_complexity,
)

from code.src.evaluation.comparison_plots import generate_comparison_plots

def run_comprehensive_experiments(
    output_dir: str = "experiment_results", time_limit: float = 1000.0
) -> Dict[str, Any]:
    """
    Ejecuta experimentos comprehensivos para el Proyecto DAA - MCCPP

    Args:
        output_dir: directorio para guardar todos los resultados
        time_limit: límite de tiempo por algoritmo por instancia (segundos)

    Returns:
        diccionario con todos los resultados de los experimentos
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    # Paso 1: Generar instancias
    print("Paso 1: Generando instancias...")
    instances_dir = os.path.join(output_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    # Generar diferentes tipos de instancias
    er_instances = generate_erdos_renyi_instances(
        n_vertices_list=[5, 10, 15, 20],
        p=0.3,
        n_colors=[3, 5, 9, 14],
        n_instances=3,
        output_dir=instances_dir,
    )

    structured_instances = generate_structured_instances(
        n_vertices_list=[10, 20, 30, 50],
        n_colors=[6, 12, 18, 29],
        graph_types=["path", "cycle", "complete", "star"],
        n_instances=2,
        output_dir=instances_dir,
    )

    special_instances = generate_special_case_instances(output_dir=instances_dir)

    interval_instances = generate_interval_graph_instances(
        n_vertices_list=[10, 20, 30, 40],
        n_colors=[6, 12, 18, 29],
        n_instances=3,
        output_dir=instances_dir,
    )

    tree_instances = generate_tree_instances(
        n_vertices_list=[5, 10, 15, 20, 200],
        n_colors=3,
        seed=2332,
    )

    # --- Crear listas de archivos por categoría ---
    # A: sólo árboles
    tree_instance_files = [inst["filename"] for inst in tree_instances]

    # B: estructura + intervalos + special 
    structured_interval_special_instances = (
        structured_instances + interval_instances + special_instances 
    )
    structured_interval_special_files = [
        inst["filename"] for inst in structured_interval_special_instances
    ]

    # C: todas las instancias juntas (incluye ER, structured, special, interval, tree)
    all_instance_collections = (
        er_instances
        + structured_instances
        + special_instances
        + interval_instances
        + tree_instances
    )
    all_instance_files = [inst["filename"] for inst in all_instance_collections]

    # Informar conteos
    print(f"  -> Instancias (árboles): {len(tree_instance_files)}")
    print(
        f"  -> Instancias (estructura+interval+special): {len(structured_interval_special_files)}"
    )
    print(f"  -> Instancias (todas): {len(all_instance_files)}")

    # Paso 2: Definir conjuntos de algoritmos para cada experimento
    print("Paso 2: Preparando conjuntos de algoritmos para cada experimento...")

    # 1) Solo DP para árboles
    dp_tree_algorithms = ["dynamic_programming_tree"]

    # 2) Aproximaciones estructurales (bipartito + interval)
    structural_algorithms = ["structural_bipartite", "structural_interval"]

    # 3) Conjunto general (los demás algoritmos)
    general_algorithms = [
        "brute_force",
        "brute_force_backtracking",
        "brute_force_intelligent",
        "ilp_solver",
        "largest_first",
        "dsatur",
        "rlf",
        "weighted_set_cover",
        "improved_weighted_set_cover",
        "simulated_annealing",
        "adaptive_simmulated_annealing",
        "trajectory_search",
        "hybrid_metaheuristic",
        "adaptive_metaheuristic",
    ]

    # Paso 2: Ejecutar benchmarks - ahora 3 experimentos separados
    print("Paso 2: Ejecutando benchmarks por experimento...")

    experiment_results = {}

    # Experimento A: árboles con DP
    print("\n==== Experimento A: Instancias de árboles + DP (solo) ====")
    benchmarks_dir_a = os.path.join(output_dir, "benchmarks_trees")
    results_trees = run_benchmark_suite(
        instance_files=tree_instance_files,
        algorithms=dp_tree_algorithms,
        output_dir=benchmarks_dir_a,
        time_limit=time_limit,
    )
    experiment_results["trees"] = results_trees

    # Comparar resultados (Paso 3a)
    print("\nPaso 3a: Comparando algoritmos para Experimento A (árboles)...")
    comparison_trees = compare_algorithms(results_trees)
    generate_benchmark_report(
        comparison_trees, os.path.join(output_dir, "benchmark_report_trees.md")
    )

    generate_comparison_plots(
        comparison_group=comparison_trees,
        benchmark_results=results_trees,
        output_dir=os.path.join(output_dir, "plots_trees")
        )

    # Experimento B: estructura + interval + special con aproximaciones estructurales
    print("\n==== Experimento B: Estructura+Interval+Special + Aproximaciones Estructurales ====")
    benchmarks_dir_b = os.path.join(output_dir, "benchmarks_structural")
    results_structural = run_benchmark_suite(
        instance_files=structured_interval_special_files,
        algorithms=structural_algorithms,
        output_dir=benchmarks_dir_b,
        time_limit=time_limit,
    )
    experiment_results["structured_interval_special"] = results_structural

    # Comparar resultados (Paso 3b)
    print("\nPaso 3b: Comparando algoritmos para Experimento B (estructura/interval/special)...")
    comparison_structural = compare_algorithms(results_structural)
    generate_benchmark_report(
        comparison_structural,
        os.path.join(output_dir, "benchmark_report_structural_interval_special.md"),
    )

    generate_comparison_plots(
        comparison_group=comparison_structural,
        benchmark_results=results_structural,
        output_dir=os.path.join(output_dir, "plots_structural")
    )
    
    # Experimento C: todas las instancias con el conjunto general de algoritmos
    print("\n==== Experimento C: Todas las instancias + Algoritmos generales ====")
    benchmarks_dir_c = os.path.join(output_dir, "benchmarks_all")
    results_all = run_benchmark_suite(
        instance_files=all_instance_files,
        algorithms=general_algorithms,
        output_dir=benchmarks_dir_c,
        time_limit=time_limit,
    )
    experiment_results["all_instances"] = results_all

    # Comparar resultados (Paso 3c)
    print("\nPaso 3c: Comparando algoritmos para Experimento C (todas las instancias)...")
    comparison_all = compare_algorithms(results_all)
    generate_benchmark_report(
        comparison_all, os.path.join(output_dir, "benchmark_report_all.md")
    )

    generate_comparison_plots(
        comparison_group=comparison_all,
        benchmark_results=results_all,
        output_dir=os.path.join(output_dir, "plots_all")
    )

    # Paso 4: Pruebas de escalabilidad
    print("Paso 4: Ejecutando pruebas de escalabilidad...")

    # Importar algoritmos faltantes para escalabilidad
    from src.algorithms.heuristic.largest_first import largest_first_heuristic
    from src.algorithms.heuristic.dsatur import dsatur_heuristic
    from src.algorithms.heuristic.recursive_largest_first import (
        recursive_largest_first_heuristic,
    )
    from src.algorithms.approximation.weighted_set_cover import (
        weighted_set_cover_approximation,
        improved_weighted_set_cover,
    )
    from src.algorithms.metaheuristic.simulated_annealing import (
        simulated_annealing,
        adaptive_simulated_annealing,
    )
    from src.algorithms.metaheuristic.trajectory_search import (
        trajectory_search_heuristic,
    )
    from src.algorithms.metaheuristic.hybrid_metaherusitics import (
        hybrid_metaheuristic,
        adaptive_metaheuristic,
    )

    # Algoritmos escalables → todos los “globales” excepto fuerza bruta e ILP
    scalability_algorithms = {
        "largest_first": largest_first_heuristic,
        "dsatur": dsatur_heuristic,
        "rlf": recursive_largest_first_heuristic,
        "weighted_set_cover": weighted_set_cover_approximation,
        "improved_weighted_set_cover": improved_weighted_set_cover,
        "simulated_annealing": simulated_annealing,
        "adaptive_simulated_annealing": adaptive_simulated_annealing,
        "trajectory_search": trajectory_search_heuristic,
        "hybrid_metaheuristic": hybrid_metaheuristic,
        "adaptive_metaheuristic": adaptive_metaheuristic,
    }

    scalability_results = compare_scalability(
    algorithms=scalability_algorithms,
    n_vertices_range=[10, 20, 30, 50, 100, 200],
    n_colors_list=[7, 14, 21, 28, 43, 67],  
    graph_density=0.3,
    n_instances=3,
    time_limit=time_limit,
    )

    # Graficar resultados de escalabilidad
    plot_scalability_results(
        scalability_results,
        os.path.join(output_dir, "scalability_analysis.png"),
    )

    # Analizar complejidad computacional
    complexity_analysis = analyze_computational_complexity(
        scalability_results
    )

    print("Paso 5: Realizando análisis estadístico...")

    # Se generan 3 bloques de análisis correspondientes a:
    #  - Árboles (DP solo)
    #  - Estructurales
    #  - Globales
    statistical_groups = {
        "trees": comparison_trees,
        "structural": comparison_structural,
        "general": comparison_all,
    }

    from src.evaluation.statistical_analysis import (
        perform_statistical_analysis_grouped,
        perform_hypothesis_testing_grouped,
        generate_statistical_report_grouped,
    )

    # Ejecutar análisis estadístico agrupado
    grouped_stats = perform_statistical_analysis_grouped(statistical_groups)
    grouped_tests = perform_hypothesis_testing_grouped(statistical_groups)

    # Generar un único reporte estadístico por grupos
    generate_statistical_report_grouped(
        grouped_stats,
        grouped_tests,
        os.path.join(output_dir, "statistical_report.md"),
    )


    # PASO 6 — ANÁLISIS EMPÍRICO (CALIDAD, TIEMPOS, LÍMITES BF)

    print("\nPaso 6: Ejecutando análisis empírico avanzado...")

    from code.src.evaluation.empyric_analysis import (
        analyze_optimality_vs_bruteforce,
        analyze_time_vs_bruteforce,
        detect_bruteforce_limits,
        analyze_behavior_by_instance_type,
        calculate_solution_quality_metrics,
        generate_extended_empirical_report,
        empirical_initial_solution_analysis
    )

    empirical_reports = {}

    # 6A — ANÁLISIS EMPÍRICO POR GRUPO (uno por cada experimento)
    for group_name, benchmark_res in experiment_results.items():

        print(f"  → Analizando grupo empírico: {group_name}")

        if group_name == "trees":
            comparison_group = comparison_trees
        elif group_name == "structured_interval_special":
            comparison_group = comparison_structural
        elif group_name == "all_instances":
            comparison_group = comparison_all
        else:
            continue

        # A1 — Calidad de solución vs brute force
        optimality_info = analyze_optimality_vs_bruteforce(comparison_group)

        # A2 — Tiempos relativos vs brute force
        time_info = analyze_time_vs_bruteforce(comparison_group)

        # A3 — Comportamiento por tipo de instancia
        type_behavior = analyze_behavior_by_instance_type(benchmark_res)

        # A4 — Métricas de calidad y aproximación por algoritmo
        approximation_stats = {}

        for instance_name, algorithms_exec in benchmark_res.items():
            if instance_name.startswith("_"):
                continue

            meta = algorithms_exec.get("metadata", {})
            graph = meta.get("graph")
            cost_matrix = meta.get("cost_matrix")
            if graph is None or cost_matrix is None:
                continue

            # Determinar coste óptimo si existe brute force
            optimal_cost = None
            if "brute_force" in algorithms_exec:
                optimal_cost = algorithms_exec["brute_force"]["cost"]

            for algo, res in algorithms_exec.items():
                if algo == "metadata":
                    continue
                if "coloring" not in res:
                    continue

                metrics = calculate_solution_quality_metrics(
                    graph=graph,
                    coloring=res["coloring"],
                    cost_matrix=cost_matrix,
                    optimal_cost=optimal_cost,
                    algorithm=algo
                )

                approximation_stats.setdefault(algo, []).append(metrics)

        # A5 — Análisis empírico adicional: impacto de heurísticas iniciales
        initial_solution_analysis = empirical_initial_solution_analysis(
            instances=benchmark_res,
            output_dir=os.path.join(output_dir, f"initial_solutions_{group_name}")
        )

        # A6 — Generar reporte empírico completo
        report_path = os.path.join(output_dir, f"empirical_report_{group_name}.md")

        generate_extended_empirical_report(
            comparison_group=comparison_group,
            optimality_analysis=optimality_info,
            time_analysis=time_info,
            bf_limits=None,
            type_behavior=type_behavior,
            approximation_stats=approximation_stats,
            initial_solution_analysis=initial_solution_analysis,   # <── NEW
            output_file=report_path
        )

        empirical_reports[group_name] = report_path


    # 6B — LÍMITE COMPUTACIONAL GLOBAL DE FUERZA BRUTA
    print("\nPaso 6B: Detectando límite computacional de fuerza bruta...")

    bf_limit_info = detect_bruteforce_limits(results_all, time_threshold=60.0)
    bf_limit_report = os.path.join(output_dir, "empirical_bruteforce_limits.md")

    with open(bf_limit_report, "w") as f:
        f.write("# Brute Force Scalability Limit\n\n")
        f.write(f"- Time threshold: {bf_limit_info['threshold']}s\n")
        f.write(f"- Max size solvable under threshold: {bf_limit_info['max_size_under_limit']}\n")
        f.write(f"- First failing instance: {bf_limit_info['first_instance_over_limit']}\n\n")
        f.write("## Full Table\n")
        for entry in bf_limit_info["full_table"]:
            f.write(f"- {entry}\n")

    empirical_reports["bruteforce_limits"] = bf_limit_report


    # COMPILAR RESULTADOS FINALES
    all_results = {
        "experiment_trees": {
            "benchmarks": results_trees,
            "comparison": comparison_trees,
            "report_file": os.path.join(output_dir, "benchmark_report_trees.md"),
            "empirical_report": empirical_reports.get("trees")
        },
        "experiment_structural_interval_special": {
            "benchmarks": results_structural,
            "comparison": comparison_structural,
            "report_file": os.path.join(output_dir, "benchmark_report_structural_interval_special.md"),
            "empirical_report": empirical_reports.get("structured_interval_special")
        },
        "experiment_all": {
            "benchmarks": results_all,
            "comparison": comparison_all,
            "report_file": os.path.join(output_dir, "benchmark_report_all.md"),
            "empirical_report": empirical_reports.get("all_instances")
        },
        "scalability_results": scalability_results,
        "complexity_analysis": complexity_analysis,
        "grouped_statistical_analysis": grouped_stats,
        "grouped_hypothesis_tests": grouped_tests,
        "empirical_bruteforce_limits": bf_limit_info,
        "empirical_reports": empirical_reports
    }

    print(f"Experimentos completados. Resultados guardados en {output_dir}")
    return all_results
