import os
from typing import Dict, Any
from utils.instances_generator import (
    generate_erdos_renyi_instances, 
    generate_structured_instances, 
    generate_tree_instances, 
    generate_special_case_instances, 
    generate_interval_graph_instances
)
from utils.evaluation import (
    run_benchmark_suite, 
    compare_algorithms, 
    generate_benchmark_report,
    generate_comparison_plots
)

def run_comprehensive_experiments(output_dir: str = "experiment_results", time_limit: float = 1000.0) -> Dict[str, Any]:
    """
    Ejecuta experimentos comprehensivos para el Proyecto DAA - MCCPP
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("EXPERIMENTOS COMPREHENSIVOS - MCCPP")
    print("="*80 + "\n")

    # Paso 1: Generar instancias
    print("Paso 1: Generando instancias...")
    instances_dir = os.path.join(output_dir, "instances")
    os.makedirs(instances_dir, exist_ok=True)

    # Generar diferentes tipos de instancias
    try:
        er_instances = generate_erdos_renyi_instances(
            n_vertices_list=[5, 10, 15, 20],
            p=0.3,
            n_colors=[3, 5, 9, 14],
            n_instances=3,
            output_dir=instances_dir,
        )
        print(f"  ✓ Erdős-Rényi: {len(er_instances)} instancias")
    except Exception as e:
        print(f"  ✗ Error generando Erdős-Rényi: {e}")
        er_instances = []

    try:
        structured_instances = generate_structured_instances(
            n_vertices_list=[10, 20, 30, 50],
            n_colors=[6, 12, 18, 29],
            graph_types=["path", "cycle", "complete", "star"],
            n_instances=2,
            output_dir=instances_dir,
        )
        print(f"  ✓ Estructurados: {len(structured_instances)} instancias")
    except Exception as e:
        print(f"  ✗ Error generando estructurados: {e}")
        structured_instances = []

    try:
        special_instances = generate_special_case_instances(output_dir=instances_dir)
        print(f"  ✓ Especiales: {len(special_instances)} instancias")
    except Exception as e:
        print(f"  ✗ Error generando especiales: {e}")
        special_instances = []

    try:
        interval_instances = generate_interval_graph_instances(
            n_vertices_list=[10, 20, 30, 40],
            n_colors=[6, 12, 18, 29],
            n_instances=3,
            output_dir=instances_dir,
        )
        print(f"  ✓ Intervalos: {len(interval_instances)} instancias")
    except Exception as e:
        print(f"  ✗ Error generando intervalos: {e}")
        interval_instances = []

    try:
        tree_instances = generate_tree_instances(
            n_vertices_list=[5, 10, 15, 20, 200],
            n_colors=3,
            seed=2332,
            output_dir=instances_dir,
        )
        print(f"  ✓ Árboles: {len(tree_instances)} instancias")
    except Exception as e:
        print(f"  ✗ Error generando árboles: {e}")
        tree_instances = []

    # --- Crear listas de archivos por categoría ---
    tree_instance_files = [inst["filename"] for inst in tree_instances]
    
    structured_interval_special_instances = (
        structured_instances + interval_instances + special_instances 
    )
    structured_interval_special_files = [
        inst["filename"] for inst in structured_interval_special_instances
    ]
    
    all_instance_collections = (
        er_instances
        + structured_instances
        + special_instances
        + interval_instances
        + tree_instances
    )
    all_instance_files = [inst["filename"] for inst in all_instance_collections]

    # Informar conteos
    print(f"\n  → Instancias (árboles): {len(tree_instance_files)}")
    print(f"  → Instancias (estructura+interval+special): {len(structured_interval_special_files)}")
    print(f"  → Instancias (todas): {len(all_instance_files)}\n")

    # Definir conjuntos de algoritmos
    print("Paso 2: Preparando conjuntos de algoritmos...")

    dp_tree_algorithms = ["dynamic_programming_tree"]
    structural_algorithms = ["dp_interval"]
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

    experiment_results = {}

    # ==================================================================
    # Experimento A: Árboles con DP
    # ==================================================================
    if tree_instance_files:
        print("\n" + "="*80)
        print("==== Experimento A: Instancias de árboles + DP (solo) ====")
        print("="*80)
        
        benchmarks_dir_a = os.path.join(output_dir, "benchmarks_trees")
        
        try:
            results_trees = run_benchmark_suite(
                instance_files=tree_instance_files,
                algorithms=dp_tree_algorithms,
                output_dir=benchmarks_dir_a,
                time_limit=time_limit,
            )
            experiment_results["trees"] = results_trees
            
            # Comparar resultados
            print("\nPaso 3a: Comparando algoritmos para Experimento A (árboles)...")
            comparison_trees = compare_algorithms(results_trees)
            
            if comparison_trees:
                generate_benchmark_report(
                    comparison_trees, 
                    os.path.join(output_dir, "benchmark_report_trees.md")
                )
                
                try:
                    generate_comparison_plots(
                        comparison_group=comparison_trees,
                        benchmark_results=results_trees,
                        output_dir=os.path.join(output_dir, "plots_trees")
                    )
                except Exception as e:
                    print(f"  ⚠ Error generando gráficas para árboles: {e}")
            else:
                print("  ⚠ No hay resultados válidos para comparar en árboles")
                
        except Exception as e:
            print(f"  ✗ Error en Experimento A: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Saltando Experimento A: No hay instancias de árboles")

    # ==================================================================
    # Experimento B: Estructurales
    # ==================================================================
    if structured_interval_special_files:
        print("\n" + "="*80)
        print("==== Experimento B: Estructura+Interval+Special + Aproximaciones ====")
        print("="*80)
        
        benchmarks_dir_b = os.path.join(output_dir, "benchmarks_structural")
        
        try:
            results_structural = run_benchmark_suite(
                instance_files=structured_interval_special_files,
                algorithms=structural_algorithms,
                output_dir=benchmarks_dir_b,
                time_limit=time_limit,
            )
            experiment_results["structured_interval_special"] = results_structural
            
            print("\nPaso 3b: Comparando algoritmos para Experimento B...")
            comparison_structural = compare_algorithms(results_structural)
            
            if comparison_structural:
                generate_benchmark_report(
                    comparison_structural,
                    os.path.join(output_dir, "benchmark_report_structural.md"),
                )
                
                try:
                    generate_comparison_plots(
                        comparison_group=comparison_structural,
                        benchmark_results=results_structural,
                        output_dir=os.path.join(output_dir, "plots_structural")
                    )
                except Exception as e:
                    print(f"  ⚠ Error generando gráficas estructurales: {e}")
            else:
                print("  ⚠ No hay resultados válidos para comparar en estructurales")
                
        except Exception as e:
            print(f"  ✗ Error en Experimento B: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Saltando Experimento B: No hay instancias estructurales")

    # ==================================================================
    # Experimento C: General
    # ==================================================================
    if all_instance_files:
        print("\n" + "="*80)
        print("==== Experimento C: Todas las instancias + Algoritmos generales ====")
        print("="*80)
        
        benchmarks_dir_c = os.path.join(output_dir, "benchmarks_all")
        
        try:
            results_all = run_benchmark_suite(
                instance_files=all_instance_files,
                algorithms=general_algorithms,
                output_dir=benchmarks_dir_c,
                time_limit=time_limit,
            )
            experiment_results["all_instances"] = results_all
            
            print("\nPaso 3c: Comparando algoritmos para Experimento C...")
            comparison_all = compare_algorithms(results_all)
            
            if comparison_all:
                generate_benchmark_report(
                    comparison_all, 
                    os.path.join(output_dir, "benchmark_report_all.md")
                )
                
                try:
                    generate_comparison_plots(
                        comparison_group=comparison_all,
                        benchmark_results=results_all,
                        output_dir=os.path.join(output_dir, "plots_all")
                    )
                except Exception as e:
                    print(f"  ⚠ Error generando gráficas generales: {e}")
            else:
                print("  ⚠ No hay resultados válidos para comparar en general")
                
        except Exception as e:
            print(f"  ✗ Error en Experimento C: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n⚠ Saltando Experimento C: No hay instancias")

    print("\n" + "="*80)
    print("EXPERIMENTOS COMPLETADOS")
    print("="*80)
    
    return experiment_results