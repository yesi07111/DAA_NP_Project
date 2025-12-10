"""
Scalability tests for DAA Project - MCCPP algorithms
"""
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Callable
import networkx as nx
import numpy as np
from src.instances_gen.generators import generate_erdos_renyi_instances
from src.utils.cost_utils import generate_cost_matrix

# def run_scalability_test(algorithm: Callable, algorithm_name: str,
#                         n_vertices_range: List[int], n_colors: int = 3,
#                         graph_density: float = 0.3, n_instances: int = 5,
#                         time_limit: float = 300.0) -> Dict[str, Any]:
#     """
#     Run scalability tests for an algorithm on increasing problem sizes for DAA Project - MCCPP
    
#     Args:
#         algorithm: algorithm function to test
#         algorithm_name: name of the algorithm
#         n_vertices_range: list of vertex counts to test
#         n_colors: number of colors to use
#         graph_density: density of generated graphs
#         n_instances: number of instances per vertex count
#         time_limit: time limit per instance (seconds)
    
#     Returns:
#         scalability test results
#     """
#     results = {
#         'algorithm': algorithm_name,
#         'n_vertices_range': n_vertices_range,
#         'n_colors': n_colors,
#         'graph_density': graph_density,
#         'n_instances_per_size': n_instances
#     }
    
#     times = []
#     costs = []
#     success_rates = []
    
#     for n_vertices in n_vertices_range:
#         print(f"Testing {algorithm_name} with n={n_vertices}")
#         size_times = []
#         size_costs = []
#         successful_runs = 0
        
#         for i in range(n_instances):
#             try:
#                 # Generate random instance
#                 graph = nx.erdos_renyi_graph(n_vertices, graph_density, seed=i)
#                 cost_matrix = generate_cost_matrix(n_vertices, n_colors, seed=i)
                
#                 # Run algorithm with time limit
#                 start_time = time.time()
#                 result = algorithm(graph, cost_matrix)
#                 end_time = time.time()
                
#                 execution_time = end_time - start_time
                
#                 if execution_time < time_limit:
#                     size_times.append(execution_time)
#                     size_costs.append(result['cost'])
#                     successful_runs += 1
#                 else:
#                     print(f"  Instance {i}: Time limit exceeded")
                    
#             except Exception as e:
#                 print(f"  Instance {i}: Error - {e}")
#                 continue
        
#         if size_times:
#             times.append({
#                 'n_vertices': n_vertices,
#                 'mean_time': np.mean(size_times),
#                 'std_time': np.std(size_times),
#                 'min_time': np.min(size_times),
#                 'max_time': np.max(size_times)
#             })
#             costs.append({
#                 'n_vertices': n_vertices,
#                 'mean_cost': np.mean(size_costs),
#                 'std_cost': np.std(size_costs)
#             })
#             success_rates.append(successful_runs / n_instances)
#         else:
#             times.append({
#                 'n_vertices': n_vertices,
#                 'mean_time': float('inf'),
#                 'std_time': 0,
#                 'min_time': float('inf'),
#                 'max_time': float('inf')
#             })
#             costs.append({
#                 'n_vertices': n_vertices,
#                 'mean_cost': float('inf'),
#                 'std_cost': 0
#             })
#             success_rates.append(0.0)
    
#     results['time_results'] = times
#     results['cost_results'] = costs
#     results['success_rates'] = success_rates
    
#     return results

def run_scalability_test(algorithm: Callable, algorithm_name: str,
                        n_vertices_range: List[int],
                        n_colors_list: List[int],
                        graph_density: float = 0.3,
                        n_instances: int = 5,
                        time_limit: float = 300.0,
                        **kwargs) -> Dict[str, Any]:
    """
    Run scalability tests where each graph size may have a different number of colors.
    """

    results = {
        'algorithm': algorithm_name,
        'n_vertices_range': n_vertices_range,
        'n_colors_list': n_colors_list,
        'graph_density': graph_density,
        'n_instances_per_size': n_instances,
    }

    times = []
    costs = []
    success_rates = []

    for idx, n_vertices in enumerate(n_vertices_range):
        n_colors = n_colors_list[idx]   # <<< ASIGNACIÓN DINÁMICA

        print(f"Testing {algorithm_name} with n={n_vertices}, colors={n_colors}")

        size_times = []
        size_costs = []
        successful_runs = 0

        for i in range(n_instances):
            try:
                graph = nx.erdos_renyi_graph(n_vertices, graph_density, seed=i)

                from src.utils.cost_utils import generate_cost_matrix
                cost_matrix = generate_cost_matrix(n_vertices, n_colors, seed=i)

                start_time = time.time()
                result = algorithm(graph, cost_matrix)
                end_time = time.time()

                execution_time = end_time - start_time

                if execution_time < time_limit:
                    size_times.append(execution_time)
                    size_costs.append(result['cost'])
                    successful_runs += 1
                else:
                    print(f"  Instance {i}: Time limit exceeded")

            except Exception as e:
                print(f"  Instance {i}: Error - {e}")
                continue

        if size_times:
            times.append({
                'n_vertices': n_vertices,
                'mean_time': np.mean(size_times),
                'std_time': np.std(size_times),
                'min_time': np.min(size_times),
                'max_time': np.max(size_times),
            })

            costs.append({
                'n_vertices': n_vertices,
                'mean_cost': np.mean(size_costs),
                'std_cost': np.std(size_costs),
            })

            success_rates.append(successful_runs / n_instances)

        else:
            times.append({
                'n_vertices': n_vertices,
                'mean_time': float('inf'),
                'std_time': 0,
                'min_time': float('inf'),
                'max_time': float('inf'),
            })

            costs.append({
                'n_vertices': n_vertices,
                'mean_cost': float('inf'),
                'std_cost': 0,
            })

            success_rates.append(0.0)

    results['time_results'] = times
    results['cost_results'] = costs
    results['success_rates'] = success_rates

    return results

def compare_scalability(algorithms: Dict[str, Callable], 
                        n_vertices_range: List[int],
                        n_colors_list: List[int] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Compare scalability of multiple algorithms for DAA Project - MCCPP

    Args:
        algorithms: dictionary of algorithm_name -> algorithm_function
        n_vertices_range: list of vertex counts to test
        n_colors_list: list of colors per vertex size (must match n_vertices_range length)
        **kwargs: additional arguments for run_scalability_test

    Returns:
        comparative scalability results
    """

    if n_colors_list is None:
        # si no lo envían, usar el n_colors único que venía en kwargs
        if "n_colors" in kwargs:
            n_colors_list = [kwargs["n_colors"]] * len(n_vertices_range)
        else:
            raise ValueError(
                "Debe especificar n_colors_list o un n_colors único en kwargs."
            )

    if len(n_colors_list) != len(n_vertices_range):
        raise ValueError(
            f"n_colors_list debe tener la misma longitud que n_vertices_range "
            f"({len(n_vertices_range)}). Recibido: {len(n_colors_list)}"
        )

    comparison = {}

    for algo_name, algo_func in algorithms.items():
        print(f"Running scalability test for {algo_name}")

        # Pasar lista de colores directamente al run_scalability_test
        results = run_scalability_test(
            algorithm=algo_func,
            algorithm_name=algo_name,
            n_vertices_range=n_vertices_range,
            n_colors_list=n_colors_list,   # <<< NUEVO
            **kwargs
        )

        comparison[algo_name] = results

    return comparison

def plot_scalability_results(comparison: Dict[str, Any], 
                           output_file: str = "scalability_analysis.png") -> None:
    """
    Plot scalability results for multiple algorithms for DAA Project - MCCPP
    
    Args:
        comparison: results from compare_scalability
        output_file: output plot file path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot time scalability
    for algo_name, results in comparison.items():
        n_vertices = [r['n_vertices'] for r in results['time_results']]
        mean_times = [r['mean_time'] for r in results['time_results']]
        std_times = [r['std_time'] for r in results['time_results']]
        
        ax1.plot(n_vertices, mean_times, 'o-', label=algo_name, linewidth=2, markersize=6)
        ax1.fill_between(n_vertices, 
                        [m - s for m, s in zip(mean_times, std_times)],
                        [m + s for m, s in zip(mean_times, std_times)],
                        alpha=0.2)
    
    ax1.set_xlabel('Number of Vertices')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Time Scalability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cost scalability
    for algo_name, results in comparison.items():
        n_vertices = [r['n_vertices'] for r in results['cost_results']]
        mean_costs = [r['mean_cost'] for r in results['cost_results']]
        
        ax2.plot(n_vertices, mean_costs, 'o-', label=algo_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Solution Cost')
    ax2.set_title('Cost Scalability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_computational_complexity(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze computational complexity from scalability results for DAA Project - MCCPP
    
    Args:
        comparison: results from compare_scalability
    
    Returns:
        complexity analysis results
    """
    complexity_analysis = {}
    
    for algo_name, results in comparison.items():
        n_vertices = [r['n_vertices'] for r in results['time_results']]
        mean_times = [r['mean_time'] for r in results['time_results']]
        
        # Fit different complexity models
        try:
            # Linear fit
            linear_fit = np.polyfit(n_vertices, mean_times, 1)
            linear_r2 = _calculate_r_squared(n_vertices, mean_times, np.polyval(linear_fit, n_vertices))
            
            # Quadratic fit
            quadratic_fit = np.polyfit(n_vertices, mean_times, 2)
            quadratic_r2 = _calculate_r_squared(n_vertices, mean_times, np.polyval(quadratic_fit, n_vertices))
            
            # Exponential fit (log space)
            log_times = np.log(mean_times)
            exp_fit = np.polyfit(n_vertices, log_times, 1)
            exp_r2 = _calculate_r_squared(n_vertices, log_times, np.polyval(exp_fit, n_vertices))
            
            complexity_analysis[algo_name] = {
                'linear_fit': {
                    'coefficients': linear_fit.tolist(),
                    'r_squared': linear_r2
                },
                'quadratic_fit': {
                    'coefficients': quadratic_fit.tolist(),
                    'r_squared': quadratic_r2
                },
                'exponential_fit': {
                    'coefficients': exp_fit.tolist(),
                    'r_squared': exp_r2
                },
                'best_fit': max([('linear', linear_r2), ('quadratic', quadratic_r2), ('exponential', exp_r2)], 
                               key=lambda x: x[1])
            }
            
        except Exception as e:
            print(f"Complexity analysis failed for {algo_name}: {e}")
            complexity_analysis[algo_name] = {'error': str(e)}
    
    return complexity_analysis

def _calculate_r_squared(x: List[float], y: List[float], y_pred: List[float]) -> float:
    """Calculate R-squared value for regression fit"""
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0