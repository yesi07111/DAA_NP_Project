"""
Comparison plots for DAA Project - MCCPP
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os

def create_algorithm_comparison_plot(comparison_data: Dict[str, Any], 
                                   output_file: str = "algorithm_comparison.png") -> None:
    """
    Plot de comparación basado en compare_algorithms() actualizado.
    """
    algorithms = list(comparison_data.keys())

    import math

    # Filtrar valores NaN o None, reemplazarlos por 0 o un valor neutral seguro
    clean_costs = []
    clean_times = []
    clean_feasible = []
    clean_algorithms = []

    
    avg_costs = [comparison_data[a]['mean_cost'] for a in algorithms]
    avg_times = [comparison_data[a]['mean_time'] for a in algorithms]
    feasible_rates = [comparison_data[a]['feasible_rate'] for a in algorithms]

    for algo, c, t, f in zip(algorithms, avg_costs, avg_times, feasible_rates):
        if c is None or math.isnan(c):
            c = 0.0
        if t is None or math.isnan(t):
            t = 0.0
        if f is None or math.isnan(f):
            f = 0.0

        clean_algorithms.append(algo)
        clean_costs.append(c)
        clean_times.append(t)
        clean_feasible.append(f)

    algorithms = clean_algorithms
    avg_costs = clean_costs
    avg_times = clean_times
    feasible_rates = clean_feasible

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # ---------------------------------------------------
    # 1) COSTO PROMEDIO
    # ---------------------------------------------------
    bars1 = ax1.bar(range(len(algorithms)), avg_costs)
    ax1.set_title("Average Cost")

    # Fix ticks
    ax1.set_xticks(range(len(algorithms)))
    ax1.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")

    # ---------------------------------------------------
    # 2) TIEMPO PROMEDIO
    # ---------------------------------------------------
    bars2 = ax2.bar(range(len(algorithms)), avg_times)
    ax2.set_title("Average Time (s)")

    ax2.set_xticks(range(len(algorithms)))
    ax2.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")

    # ---------------------------------------------------
    # 3) TASA FACTIBLE
    # ---------------------------------------------------
    bars3 = ax3.bar(range(len(algorithms)), feasible_rates)
    ax3.set_title("Feasibility Rate")
    ax3.set_ylim(0, 1)

    ax3.set_xticks(range(len(algorithms)))
    ax3.set_xticklabels(algorithms, rotation=45, ha="right")

    for bar in bars3:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")

    # ---------------------------------------------------
    # 4) COSTO vs TIEMPO
    # ---------------------------------------------------
    ax4.scatter(avg_times, avg_costs)
    for t, c, algo in zip(avg_times, avg_costs, algorithms):
        ax4.annotate(algo, (t, c), textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax4.set_title("Cost vs Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Cost")

    plt.tight_layout()
    plt.savefig(output_file)
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

def create_box_plot_comparison(algorithm_costs: Dict[str, List[float]], 
                              output_file: str = "cost_distribution.png") -> None:
    """
    Create box plot comparison of cost distributions for DAA Project - MCCPP
    
    Args:
        algorithm_costs: dictionary of algorithm -> list of costs
        output_file: output plot file path
    """
    plt.figure(figsize=(12, 6))
    
    # Prepare data for box plot
    data = [costs for costs in algorithm_costs.values()]
    labels = list(algorithm_costs.keys())
    
    plt.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
    plt.title('Distribution of Solution Costs by Algorithm')
    plt.ylabel('Cost')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def generate_comparison_plots(comparison_group: Dict[str, Any],
                              benchmark_results: Dict[str, Any],
                              output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    create_algorithm_comparison_plot(
        comparison_group,
        os.path.join(output_dir, "algorithm_comparison.png")
    )

    create_performance_profile(
        benchmark_results,
        os.path.join(output_dir, "performance_profile.png")
    )



    