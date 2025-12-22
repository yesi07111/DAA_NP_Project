"""
Input/Output utilities for DAA Project - MCCPP
"""
import pickle
import json
import numpy as np
import networkx as nx
from typing import Dict, Any, Tuple
import os

def save_instance(graph: nx.Graph, cost_matrix: np.ndarray, filename: str, 
                 metadata: Dict[str, Any] = None):
    """
    Save problem instance to file
    
    Args:
        graph: networkx Graph
        cost_matrix: cost matrix
        filename: output filename
        metadata: additional instance metadata
    """
    instance_data = {
        'graph_edges': list(graph.edges()),
        'graph_nodes': list(graph.nodes()),
        'cost_matrix': cost_matrix.tolist(),
        'n_vertices': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'n_colors': cost_matrix.shape[1],
        'metadata': metadata or {}
    }
    
    with open(filename, 'w') as f:
        json.dump(instance_data, f, indent=2)

def load_instance(filename: str) -> Tuple[nx.Graph, np.ndarray, Dict[str, Any]]:
    """
    Load problem instance from file
    
    Args:
        filename: input filename
    
    Returns:
        tuple of (graph, cost_matrix, metadata)
    """
    with open(filename, 'r') as f:
        instance_data = json.load(f)
    
    # Reconstruct graph
    graph = nx.Graph()
    graph.add_nodes_from(instance_data['graph_nodes'])
    graph.add_edges_from(instance_data['graph_edges'])
    
    # Reconstruct cost matrix
    cost_matrix = np.array(instance_data['cost_matrix'])
    
    return graph, cost_matrix, instance_data.get('metadata', {})

def save_solution(coloring: Dict[int, int], filename: str, 
                 metrics: Dict[str, Any] = None):
    """
    Save coloring solution to file
    
    Args:
        coloring: vertex to color mapping
        filename: output filename
        metrics: solution quality metrics
    """
    solution_data = {
        'coloring': coloring,
        'metrics': metrics or {}
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_data, f, indent=2)

def load_solution(filename: str) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """
    Load coloring solution from file
    
    Args:
        filename: input filename
    
    Returns:
        tuple of (coloring, metrics)
    """
    with open(filename, 'r') as f:
        solution_data = json.load(f)
    
    # Convert keys back to integers (JSON stores them as strings)
    coloring = {int(k): v for k, v in solution_data['coloring'].items()}
    
    return coloring, solution_data.get('metrics', {})

def export_results(results: Dict[str, Any], filename: str, format: str = 'json'):
    """
    Export experimental results
    
    Args:
        results: dictionary of results
        filename: output filename
        format: export format ('json', 'csv')
    """
    if format == 'json':
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    elif format == 'csv':
        # For tabular data, convert to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def ensure_directory(directory: str):
    """
    Ensure directory exists, create if not
    
    Args:
        directory: directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_instance_filename(instance_type: str, n_vertices: int, n_colors: int, 
                         density: float, seed: int, directory: str = "instances") -> str:
    """
    Generate standardized instance filename
    
    Args:
        instance_type: type of instance ('erdos_renyi', 'interval', etc.)
        n_vertices: number of vertices
        n_colors: number of colors
        density: graph density
        seed: random seed
        directory: output directory
    
    Returns:
        standardized filename
    """
    ensure_directory(directory)
    return os.path.join(directory, f"{instance_type}_n{n_vertices}_k{n_colors}_d{density:.2f}_s{seed}.json")

def convert_numpy_types(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte tipos de NumPy a tipos nativos de Python para serialización JSON
    
    Args:
        result_dict: diccionario con resultados que puede contener tipos de NumPy
    
    Returns:
        diccionario con tipos nativos de Python
    """
    converted = {}
    for key, value in result_dict.items():
        if isinstance(value, (np.integer, np.int32, np.int64)):
            converted[key] = int(value)
        elif isinstance(value, (np.floating, np.float32, np.float64)):
            converted[key] = float(value)
        elif isinstance(value, np.ndarray):
            converted[key] = value.tolist()
        elif isinstance(value, dict):
            # Convertir diccionarios recursivamente
            converted[key] = convert_numpy_types(value)
        elif isinstance(value, (list, tuple)):
            # Convertir listas/tuplas
            converted[key] = [convert_numpy_types(item) if isinstance(item, dict) 
                            else (int(item) if isinstance(item, (np.integer, np.int32, np.int64))
                            else (float(item) if isinstance(item, (np.floating, np.float32, np.float64))
                            else item)) for item in value]
        else:
            converted[key] = value
    return converted