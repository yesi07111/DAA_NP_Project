import os
import json
import numpy as np
import networkx as nx
from typing import Dict, Set, Tuple, Any


# COST UTILS

def generate_cost_matrix(n_vertices: int, n_colors: int, cost_range: Tuple[float, float] = (1, 100), 
                        seed: int = None) -> np.ndarray:
    """
    Generate random cost matrix
    
    Args:
        n_vertices: number of vertices
        n_colors: number of colors/frequencies
        cost_range: tuple of (min_cost, max_cost)
        seed: random seed
    
    Returns:
        n_vertices x n_colors cost matrix
    """
    if seed:
        np.random.seed(seed)
    
    min_cost, max_cost = cost_range
    return np.random.uniform(min_cost, max_cost, size=(n_vertices, n_colors))

def generate_structured_cost_matrix(n_vertices: int, n_colors: int, 
                                  cost_pattern: str = "uniform",
                                  seed: int = None) -> np.ndarray:
    """
    Generate cost matrix with specific patterns for testing
    
    Args:
        n_vertices: number of vertices
        n_colors: number of colors
        cost_pattern: type of cost structure
            - "uniform": random uniform costs
            - "preferential": some colors are generally cheaper
            - "vertex_specific": some vertices have generally lower costs
            - "binary": costs are either 0 or 1 (for MVC reduction tests)
        seed: random seed
    
    Returns:
        structured cost matrix
    """
    if seed:
        np.random.seed(seed)
    
    if cost_pattern == "uniform":
        return generate_cost_matrix(n_vertices, n_colors, (1, 100), seed)
    
    elif cost_pattern == "preferential":
        base_costs = np.linspace(1, 100, n_colors)
        noise = np.random.normal(0, 10, (n_vertices, n_colors))
        costs = base_costs + noise
        return np.clip(costs, 1, 100)
    
    elif cost_pattern == "vertex_specific":
        vertex_factors = np.random.uniform(0.5, 2.0, n_vertices)
        color_factors = np.random.uniform(0.5, 2.0, n_colors)
        base = np.random.uniform(10, 50, (n_vertices, n_colors))
        costs = base * vertex_factors.reshape(-1, 1) * color_factors.reshape(1, -1)
        return costs
    
    elif cost_pattern == "binary":
        # For MVC reduction: first color costs 0, others cost 1
        costs = np.ones((n_vertices, n_colors))
        costs[:, 0] = 0  # First color is "preferred" with zero cost
        return costs
    
    else:
        raise ValueError(f"Unknown cost pattern: {cost_pattern}")

def evaluate_solution(coloring: Dict[int, int], cost_matrix: np.ndarray) -> float:
    """
    Calculate total cost of a coloring solution
    
    Args:
        coloring: dictionary mapping vertex -> color index
        cost_matrix: n_vertices x n_colors cost matrix
    
    Returns:
        total cost of the coloring
    """
    total_cost = 0.0
    for vertex, color in coloring.items():
        total_cost += cost_matrix[vertex, color]
    return total_cost

# IO UTILS

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
    # Normalizar metadata y asegurar campos de factibilidad
    metadata = metadata.copy() if metadata is not None else {}
    n_colors = int(cost_matrix.shape[1])

    # Estimar número cromático cuando no esté presente
    chromatic_est = metadata.get('chromatic_number', None)
    try:
        if chromatic_est is None:
            if nx.is_tree(graph):
                chromatic_est = 2 if graph.number_of_nodes() > 1 else 1
            elif nx.is_bipartite(graph):
                chromatic_est = 2 if graph.number_of_nodes() > 1 else 1
            elif nx.is_empty(graph):
                chromatic_est = 0
            else:
                coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
                chromatic_est = len(set(coloring.values()))
    except Exception:
        chromatic_est = None

    if chromatic_est is not None:
        metadata['chromatic_number'] = int(chromatic_est)
        metadata['is_feasible'] = bool(chromatic_est <= n_colors)
    else:
        # Si no se pudo estimar, respetar lo que ya estaba o marcar infactible
        metadata.setdefault('is_feasible', False)

    instance_data = {
        'graph_edges': list(graph.edges()),
        'graph_nodes': list(graph.nodes()),
        'cost_matrix': cost_matrix.tolist(),
        'n_vertices': graph.number_of_nodes(),
        'n_edges': graph.number_of_edges(),
        'n_colors': n_colors,
        'metadata': metadata
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

def convert_numpy_types(obj: Any) -> Any:
    """
    Convierte tipos de NumPy a tipos nativos de Python para serialización JSON.
    Maneja recursivamente diccionarios, listas y tipos numéricos.
    """
    if isinstance(obj, dict):
        # Convierte tanto claves como valores
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# GRAPH UTILS

def generate_erdos_renyi_graph(n: int, p: float, seed: int = None) -> nx.Graph:
    """
    Generate Erdős-Rényi random graph G(n, p)

    Args:
        n: number of vertices
        p: probability of edge creation
        seed: random seed for reproducibility

    Returns:
        networkx Graph object
    """
    return nx.erdos_renyi_graph(n, p, seed=seed, directed=False)

def generate_interval_graph(
    n: int, max_length: float = 1.0, seed: int = None
) -> nx.Graph:
    """
    Generate interval graph for FISP-MDC problems

    Args:
        n: number of intervals
        max_length: maximum interval length
        seed: random seed

    Returns:
        interval graph and interval data
    """
    if seed:
        np.random.seed(seed)

    intervals = []
    for i in range(n):
        start = np.random.uniform(0, max_length * 0.8)
        end = start + np.random.uniform(0.1, max_length * 0.2)
        intervals.append((start, end))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Add edges for overlapping intervals
    for i in range(n):
        for j in range(i + 1, n):
            if (
                intervals[i][1] >= intervals[j][0]
                and intervals[j][1] >= intervals[i][0]
            ):
                G.add_edge(i, j)

    return G, intervals

def is_proper_coloring(graph: nx.Graph, coloring: Dict[int, int]) -> bool:
    """
    Verifica que una coloración sea válida (propia).
    
    Args:
        graph: Grafo de NetworkX
        coloring: Diccionario {vértice: color}
    
    Returns:
        True si la coloración es válida, False en caso contrario
    """
    if coloring is None or len(coloring) != graph.number_of_nodes():
        return False
    
    for u, v in graph.edges():
        if u not in coloring or v not in coloring:
            return False
        if coloring[u] == coloring[v]:
            return False
    
    return True

def calculate_degrees(graph: nx.Graph) -> Dict[int, int]:
    """
    Calculate degrees of all vertices

    Args:
        graph: networkx Graph

    Returns:
        dictionary of vertex -> degree
    """
    return dict(graph.degree())

def get_maximal_independent_set(
    graph: nx.Graph, available_vertices: Set[int] = None
) -> Set[int]:
    """
    Find a maximal independent set using greedy approach

    Args:
        graph: networkx Graph
        available_vertices: subset of vertices to consider

    Returns:
        set of vertices in the independent set
    """
    if available_vertices is None:
        available_vertices = set(graph.nodes())

    independent_set = set()
    remaining = available_vertices.copy()

    while remaining:
        # Select vertex with minimum degree in remaining graph
        node = min(remaining, key=lambda x: len(set(graph.neighbors(x)) & remaining))
        independent_set.add(node)

        # Remove selected node and its neighbors
        remaining.discard(node)
        remaining -= set(graph.neighbors(node))

    return independent_set

def graph_density(graph: nx.Graph) -> float:
    """
    Calculate graph density

    Args:
        graph: networkx Graph

    Returns:
        density value between 0 and 1
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0
    return (2 * graph.number_of_edges()) / (n * (n - 1))
