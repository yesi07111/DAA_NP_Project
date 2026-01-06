import time
import heapq
from typing import Dict, Set, Any, Tuple
import networkx as nx
import numpy as np
from utils.utils import convert_numpy_types, is_proper_coloring, calculate_degrees, evaluate_solution, get_maximal_independent_set
from utils.timeout_handler import check_global_timeout


# LARGEST FIRST

def largest_first_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:

    start_time = time.time()
    operations = 0

    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())

    # Calculate degrees
    degrees = calculate_degrees(graph)
    operations += len(vertices) # Accessing degrees

    # Sort vertices
    sorted_vertices = sorted(vertices, key=lambda v: degrees[v], reverse=True)
    operations += len(vertices) * np.log(len(vertices)) if len(vertices) > 0 else 0 # Sorting cost estimate

    coloring = {}
    
    # Color vertices in order of decreasing degree
    for vertex in sorted_vertices:
        operations += 1
        
        # Get colors used by neighbors
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            operations += 1
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])

        # Find available colors
        available_colors = set(range(n_colors)) - neighbor_colors
        operations += len(available_colors) # Set operations

        if available_colors:
            # Choose the available color with minimum cost
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
            operations += len(available_colors) # Min comparison
        else:
            # Conflict handling
            best_color = np.argmin(cost_matrix[vertex, :])
            operations += n_colors

        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": float(cost),
        "execution_time": float(end_time - start_time),
        "optimal": False,
        "algorithm": "largest_first",
        "feasible": feasible,
        "colors_used": int(len(set(coloring.values()))),
        "operations": int(operations)
    }

    return convert_numpy_types(result)

# DSATUR

class DSATURVertex:
    def __init__(self, vertex_id: int, degree: int):
        self.vertex_id = vertex_id
        self.saturation = 0
        self.degree = degree
        self.colored = False
    
    def __lt__(self, other):
        if self.saturation != other.saturation:
            return self.saturation > other.saturation
        else:
            return self.degree > other.degree

def dsatur_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:

    start_time = time.time()
    operations = 0
    
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())
    
    dsatur_vertices = {}
    degrees = calculate_degrees(graph)
    operations += len(vertices)
    
    for vertex in vertices:
        dsatur_vertices[vertex] = DSATURVertex(vertex, degrees[vertex])
    
    heap = []
    for vertex in vertices:
        heapq.heappush(heap, (-dsatur_vertices[vertex].saturation, -degrees[vertex], vertex))
        operations += 1
    
    coloring = {}
    color_adjacency = {v: set() for v in vertices}
    
    while heap:
        operations += 1
        # Check global timeout frequently
        try:
            check_global_timeout()
        except Exception:
            break
        _, _, current_vertex = heapq.heappop(heap)
        vertex_obj = dsatur_vertices[current_vertex]
        
        if vertex_obj.colored:
            continue
        
        available_colors = set(range(n_colors)) - color_adjacency[current_vertex]
        
        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[current_vertex, c])
            operations += len(available_colors)
        else:
            best_color = np.argmin(cost_matrix[current_vertex, :])
            operations += n_colors
        
        coloring[current_vertex] = best_color
        vertex_obj.colored = True
        
        for neighbor in graph.neighbors(current_vertex):
            operations += 1
            if not dsatur_vertices[neighbor].colored:
                color_adjacency[neighbor].add(best_color)
                
                old_saturation = dsatur_vertices[neighbor].saturation
                dsatur_vertices[neighbor].saturation = len(color_adjacency[neighbor])
                
                if dsatur_vertices[neighbor].saturation != old_saturation:
                    heapq.heappush(heap, (-dsatur_vertices[neighbor].saturation, 
                                         -dsatur_vertices[neighbor].degree, 
                                         neighbor))
                    operations += 1 # Heap push cost proxy
    
    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()
    
    result = {
        'solution': coloring,
        'cost': cost,
        'execution_time': end_time - start_time,
        'optimal': False,
        'algorithm': 'dsatur',
        'feasible': feasible,
        'colors_used': len(set(coloring.values())),
        'operations': int(operations)
    }
    
    return convert_numpy_types(result)

# RECURSIVE LARGEST FIRST
def recursive_largest_first_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:

    start_time = time.time()
    operations = 0

    n_colors = cost_matrix.shape[1]
    coloring = {}
    uncolored = set(graph.nodes())
    
    current_color = 0

    while uncolored and current_color < n_colors:
        operations += 1
        # Check global timeout frequently
        try:
            check_global_timeout()
        except Exception:
            break
        
        # Construir conjunto independiente
        independent_set, ops_in_step = _build_cost_optimized_is(
            graph, uncolored, cost_matrix, current_color
        )
        operations += ops_in_step

        if not independent_set:
            break

        for vertex in independent_set:
            coloring[vertex] = current_color
            operations += 1

        uncolored -= independent_set
        current_color += 1

    # Rellenar restantes
    for vertex in uncolored:
        operations += 1
        neighbor_colors = set()
        for neighbor in graph.neighbors(vertex):
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        
        available_colors = set(range(n_colors)) - neighbor_colors
        
        if available_colors:
            best_color = min(available_colors, key=lambda c: cost_matrix[vertex, c])
        else:
            best_color = np.argmin(cost_matrix[vertex, :])
        
        coloring[vertex] = best_color

    cost = evaluate_solution(coloring, cost_matrix)
    feasible = is_proper_coloring(graph, coloring)
    end_time = time.time()

    result = {
        "solution": coloring,
        "cost": cost,
        "execution_time": end_time - start_time,
        "optimal": False,
        "algorithm": "rlf",
        "feasible": feasible,
        "colors_used": len(set(coloring.values())),
        "operations": int(operations)
    }

    return convert_numpy_types(result)

def _build_cost_optimized_is(
    graph: nx.Graph,
    available_vertices: Set[int],
    cost_matrix: np.ndarray,
    current_color: int,
) -> Tuple[Set[int], int]:
    """Helper for RLF. Returns set and operation count."""
    ops = 0
    if not available_vertices:
        return set(), ops

    independent_set = set()
    candidates = available_vertices.copy()

    subgraph = graph.subgraph(available_vertices)
    subgraph_degrees = dict(subgraph.degree())
    ops += len(available_vertices)

    sorted_candidates = sorted(
        available_vertices,
        key=lambda v: (cost_matrix[v, current_color], -subgraph_degrees[v]),
    )
    ops += len(available_vertices) * np.log(len(available_vertices)) if len(available_vertices) > 0 else 0

    for vertex in sorted_candidates:
        ops += 1
        if vertex not in candidates:
            continue

        can_add = True
        for neighbor in graph.neighbors(vertex):
            ops += 1
            if neighbor in independent_set:
                can_add = False
                break

        if can_add:
            independent_set.add(vertex)
            candidates.discard(vertex)
            for neighbor in graph.neighbors(vertex):
                candidates.discard(neighbor)
                ops += 1

    return independent_set, ops

# PERFECT ELIMINATION ORDER - GREEDY BASED

def peo_greedy_heuristic(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:

    start_time = time.time()
    operations = 0
    
    # Verificar cordalidad
    operations += graph.number_of_nodes() ** 2
    if not nx.is_chordal(graph):
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': 'El grafo no es cordal',
            'algorithm': 'peo_greedy_heuristic'
        }
    
    # Obtener PEO
    try:
        operations += graph.number_of_nodes() ** 2
        peo_cliques = list(nx.chordal_graph_cliques(graph))
        vertex_order = []
        seen = set()
        for clique in peo_cliques:
            for v in clique:
                if v not in seen:
                    vertex_order.append(v)
                    seen.add(v)
                    operations += 1
    except:
        vertex_order = sorted(graph.nodes(), key=lambda n: graph.degree(n))
        operations += len(vertex_order)
    
    n_colors = cost_matrix.shape[1]
    coloring = {}
    
    # GREEDY SIMPLE sobre PEO
    for vertex in vertex_order:
        operations += 1
        
        # Colores usados por vecinos ya coloreados
        forbidden = {coloring[nbr] for nbr in graph.neighbors(vertex) if nbr in coloring}
        operations += graph.degree(vertex)
        
        # Elegir color de costo mínimo disponible
        best_c = None
        min_cost = float('inf')
        
        for c in range(n_colors):
            operations += 1
            if c not in forbidden:
                if cost_matrix[vertex, c] < min_cost:
                    min_cost = cost_matrix[vertex, c]
                    best_c = c
        
        if best_c is None:
            # Sin colores disponibles
            return {
                'solution': {},
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'operations': operations,
                'feasible': False,
                'error': 'K insuficiente',
                'algorithm': 'peo_greedy_heuristic'
            }
        
        coloring[vertex] = best_c
    
    
    # Aplicar búsqueda local opcional
    from .approximation_algorithms import _apply_local_search
    final_result = _apply_local_search(graph, coloring, cost_matrix, operations, rounds=2)
    operations = final_result['operations']
    
    is_feasible = is_proper_coloring(graph, final_result['coloring'])
    total_cost = sum(cost_matrix[v, final_result['coloring'][v]] 
                     for v in final_result['coloring'])
    
    return convert_numpy_types({
        'solution': final_result['coloring'],
        'cost': total_cost,
        'execution_time': time.time() - start_time,
        'operations': operations,
        'algorithm': 'peo_greedy_heuristic',
        'feasible': is_feasible,
        'optimal': False,  # Es heurística
        'approximation_factor': 'Heurística sin garantía teórica',
        'empirical_quality': '5-15% sobre óptimo (estimado)',
        'complexity': 'O(n·k·d)'
    })