"""
Main CLI interface for DAA Project - MCCPP
"""
import argparse
import os
import sys
import networkx as nx
from typing import Dict, Any

from src.utils.io_utils import load_instance, save_solution
from src.algorithms.exact.brute_force import brute_force_solver
from src.algorithms.exact.ilp_solver import ilp_solver
from src.algorithms.exact.dynamic_programming import dynamic_programming_interval
from src.algorithms.approximation.weighted_set_cover import weighted_set_cover_approximation
from src.algorithms.approximation.structural_approximation import structural_approximation_bipartite, structural_approximation_interval
from src.algorithms.heuristic.largest_first import largest_first_heuristic
from src.algorithms.heuristic.dsatur import dsatur_heuristic
from src.algorithms.heuristic.recursive_largest_first import recursive_largest_first_heuristic
from src.algorithms.metaheuristic.simulated_annealing import simulated_annealing
from src.algorithms.metaheuristic.trajectory_search import trajectory_search_heuristic

def main():
    parser = argparse.ArgumentParser(description='DAA Project - MCCPP Solver')
    parser.add_argument('algorithm', choices=[
        'brute_force', 'ilp_solver', 'dynamic_programming_interval',
        'weighted_set_cover', 'structural_bipartite', 'structural_interval',
        'largest_first', 'dsatur', 'rlf', 
        'simulated_annealing', 'trajectory_search'
    ], help='Algorithm to use')
    parser.add_argument('instance_file', help='Path to instance file')
    parser.add_argument('--output', '-o', help='Output solution file')
    parser.add_argument('--time-limit', '-t', type=float, default=300.0, 
                       help='Time limit in seconds (for exact algorithms)')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')
    
    args = parser.parse_args()
    
    # Load instance
    if not os.path.exists(args.instance_file):
        print(f"Error: Instance file {args.instance_file} not found")
        return 1
    
    graph, cost_matrix, metadata = load_instance(args.instance_file)
    print(f"Loaded instance with {graph.number_of_nodes()} vertices and {graph.number_of_edges()} edges")
    
    # Run algorithm
    algorithm_functions = {
        'brute_force': brute_force_solver,
        'ilp_solver': ilp_solver,
        'dynamic_programming_interval': dynamic_programming_interval,
        'weighted_set_cover': weighted_set_cover_approximation,
        'structural_bipartite': structural_approximation_bipartite,
        'structural_interval': structural_approximation_interval,
        'largest_first': largest_first_heuristic,
        'dsatur': dsatur_heuristic,
        'rlf': recursive_largest_first_heuristic,
        'simulated_annealing': simulated_annealing,
        'trajectory_search': trajectory_search_heuristic
    }
    
    algorithm_func = algorithm_functions[args.algorithm]
    
    # Prepare algorithm arguments
    algo_args = {}
    if args.algorithm in ['brute_force', 'ilp_solver']:
        algo_args['time_limit'] = args.time_limit
    if args.seed is not None:
        algo_args['seed'] = args.seed
    if args.algorithm == 'dynamic_programming_interval':
        algo_args['intervals'] = metadata.get('intervals')
    
    print(f"Running {args.algorithm}...")
    result = algorithm_func(graph, cost_matrix, **algo_args)
    
    # Print results
    print(f"Solution cost: {result['cost']}")
    print(f"Execution time: {result['execution_time']:.2f}s")
    if 'feasible' in result:
        print(f"Feasible: {result['feasible']}")
    if 'optimal' in result:
        print(f"Optimal: {result['optimal']}")
    
    # Save solution if output file provided
    if args.output:
        save_solution(result['solution'], args.output, result)
        print(f"Solution saved to {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())