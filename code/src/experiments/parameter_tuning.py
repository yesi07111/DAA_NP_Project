"""
Parameter tuning for metaheuristics in DAA Project - MCCPP
"""
import time
import random
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np
from src.instances_gen.generators import generate_erdos_renyi_graph
from src.utils.cost_utils import generate_cost_matrix
from src.algorithms.metaheuristic.simulated_annealing import simulated_annealing
from src.algorithms.metaheuristic.trajectory_search import trajectory_search_heuristic

def tune_simulated_annealing_parameters(graph: nx.Graph, cost_matrix: np.ndarray,
                                       parameter_grid: Dict[str, List[Any]] = None,
                                       n_evaluations: int = 5,
                                       time_limit: float = 60.0) -> Dict[str, Any]:
    """
    Tune Simulated Annealing parameters for DAA Project - MCCPP
    
    Args:
        graph: graph instance
        cost_matrix: cost matrix
        parameter_grid: dictionary of parameter names -> list of values
        n_evaluations: number of evaluations per parameter combination
        time_limit: time limit per evaluation (seconds)
    
    Returns:
        tuning results with best parameters
    """
    # Default parameter grid if not provided
    if parameter_grid is None:
        parameter_grid = {
            'initial_temperature': [100, 500, 1000, 2000],
            'cooling_rate': [0.95, 0.97, 0.99, 0.995],
            'max_iterations': [5000, 10000, 20000],
            'penalty_weight': [100, 500, 1000]
        }
    
    best_cost = float('inf')
    best_params = None
    all_results = []
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_combinations = _generate_parameter_combinations(parameter_grid)
    
    print(f"Tuning Simulated Annealing with {len(param_combinations)} parameter combinations")
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        costs = []
        times = []
        
        print(f"  Testing combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        for eval_idx in range(n_evaluations):
            try:
                start_time = time.time()
                result = simulated_annealing(
                    graph=graph,
                    cost_matrix=cost_matrix,
                    **param_dict
                )
                end_time = time.time()
                
                if end_time - start_time < time_limit:
                    costs.append(result['cost'])
                    times.append(result['execution_time'])
                else:
                    print(f"    Evaluation {eval_idx}: Time limit exceeded")
                    
            except Exception as e:
                print(f"    Evaluation {eval_idx}: Error - {e}")
                continue
        
        if costs:
            avg_cost = np.mean(costs)
            avg_time = np.mean(times)
            
            all_results.append({
                'parameters': param_dict,
                'average_cost': avg_cost,
                'average_time': avg_time,
                'cost_std': np.std(costs),
                'n_successful': len(costs)
            })
            
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_params = param_dict
    
    return {
        'best_parameters': best_params,
        'best_cost': best_cost,
        'all_results': all_results,
        'parameter_grid': parameter_grid
    }

def tune_trajectory_search_parameters(graph: nx.Graph, cost_matrix: np.ndarray,
                                     parameter_grid: Dict[str, List[Any]] = None,
                                     n_evaluations: int = 3,
                                     time_limit: float = 120.0) -> Dict[str, Any]:
    """
    Tune Trajectory Search parameters for DAA Project - MCCPP
    
    Args:
        graph: graph instance
        cost_matrix: cost matrix
        parameter_grid: dictionary of parameter names -> list of values
        n_evaluations: number of evaluations per parameter combination
        time_limit: time limit per evaluation (seconds)
    
    Returns:
        tuning results with best parameters
    """
    # Default parameter grid if not provided
    if parameter_grid is None:
        parameter_grid = {
            'population_size': [5, 10, 20],
            'max_iterations': [500, 1000, 2000],
            'tabu_tenure': [20, 50, 100],
            'path_relinking_frequency': [5, 10, 20],
            'elite_pool_size': [3, 5, 10]
        }
    
    best_cost = float('inf')
    best_params = None
    all_results = []
    
    param_names = list(parameter_grid.keys())
    param_combinations = _generate_parameter_combinations(parameter_grid)
    
    print(f"Tuning Trajectory Search with {len(param_combinations)} parameter combinations")
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        costs = []
        times = []
        
        print(f"  Testing combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        for eval_idx in range(n_evaluations):
            try:
                start_time = time.time()
                result = trajectory_search_heuristic(
                    graph=graph,
                    cost_matrix=cost_matrix,
                    **param_dict
                )
                end_time = time.time()
                
                if end_time - start_time < time_limit:
                    costs.append(result['cost'])
                    times.append(result['execution_time'])
                else:
                    print(f"    Evaluation {eval_idx}: Time limit exceeded")
                    
            except Exception as e:
                print(f"    Evaluation {eval_idx}: Error - {e}")
                continue
        
        if costs:
            avg_cost = np.mean(costs)
            avg_time = np.mean(times)
            
            all_results.append({
                'parameters': param_dict,
                'average_cost': avg_cost,
                'average_time': avg_time,
                'cost_std': np.std(costs),
                'n_successful': len(costs)
            })
            
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_params = param_dict
    
    return {
        'best_parameters': best_params,
        'best_cost': best_cost,
        'all_results': all_results,
        'parameter_grid': parameter_grid
    }

def _generate_parameter_combinations(parameter_grid: Dict[str, List[Any]]) -> List[Tuple]:
    """
    Generate all combinations of parameters from a grid
    
    Args:
        parameter_grid: dictionary of parameter names -> list of values
    
    Returns:
        list of parameter combinations as tuples
    """
    from itertools import product
    param_names = sorted(parameter_grid.keys())
    param_values = [parameter_grid[name] for name in param_names]
    return list(product(*param_values))

def tune_metaheuristic_parameters(instance_sizes: List[int] = [20, 50],
                                  n_instances: int = 2,
                                  output_dir: str = "parameter_tuning_results") -> Dict[str, Any]:
    """
    Comprehensive parameter tuning across multiple instances for DAA Project - MCCPP
    
    Args:
        instance_sizes: list of instance sizes to tune on
        n_instances: number of instances per size
        output_dir: directory to save results
    
    Returns:
        comprehensive tuning results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    all_tuning_results = {}
    
    for n_vertices in instance_sizes:
        print(f"Tuning parameters for n_vertices = {n_vertices}")
        instance_results = {}
        
        for i in range(n_instances):
            # Generate random instance
            graph = generate_erdos_renyi_graph(n_vertices, 0.3, seed=i)
            cost_matrix = generate_cost_matrix(n_vertices, 3, seed=i)
            
            # Tune Simulated Annealing
            sa_tuning = tune_simulated_annealing_parameters(
                graph, cost_matrix, n_evaluations=2, time_limit=30.0
            )
            
            # Tune Trajectory Search
            ts_tuning = tune_trajectory_search_parameters(
                graph, cost_matrix, n_evaluations=2, time_limit=30.0
            )
            
            instance_results[f'instance_{i}'] = {
                'simulated_annealing': sa_tuning,
                'trajectory_search': ts_tuning
            }
        
        # Aggregate results across instances
        best_sa_params = _aggregate_parameters([r['simulated_annealing']['best_parameters'] for r in instance_results.values()])
        best_ts_params = _aggregate_parameters([r['trajectory_search']['best_parameters'] for r in instance_results.values()])
        
        all_tuning_results[n_vertices] = {
            'instance_results': instance_results,
            'aggregated_best_parameters': {
                'simulated_annealing': best_sa_params,
                'trajectory_search': best_ts_params
            }
        }
    
    # Save results
    import json
    with open(os.path.join(output_dir, "parameter_tuning_results.json"), 'w') as f:
        json.dump(all_tuning_results, f, indent=2)
    
    return all_tuning_results

def _aggregate_parameters(parameter_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate best parameters across multiple runs
    
    Args:
        parameter_list: list of parameter dictionaries
    
    Returns:
        aggregated parameters (for numerical parameters, take the mean)
    """
    if not parameter_list:
        return {}
    
    aggregated = {}
    for key in parameter_list[0].keys():
        values = [params[key] for params in parameter_list if key in params]
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = np.mean(values)
        else:
            # For non-numerical, take the most frequent
            from collections import Counter
            counter = Counter(values)
            aggregated[key] = counter.most_common(1)[0][0]
    
    return aggregated