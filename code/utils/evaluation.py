import os
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt


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
