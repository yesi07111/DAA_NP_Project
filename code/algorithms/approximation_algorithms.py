import time
import numpy as np
import networkx as nx
from typing import Dict, Any
from utils.utils import is_proper_coloring
from utils.utils import convert_numpy_types
from utils.timeout_handler import check_global_timeout

# ============================================================================
# UTILIDADES
# ============================================================================

def evaluate_cost(coloring: Dict[int, int], cost_matrix: np.ndarray) -> float:
    """Calcula el costo total de una coloración."""
    if not coloring:
        return float('inf')
    return sum(cost_matrix[v, c] for v, c in coloring.items())

def _apply_local_search(graph, coloring, cost_matrix, initial_ops, rounds=1):

    if not coloring: 
        return {'coloring': {}, 'operations': initial_ops}
    
    n_colors = cost_matrix.shape[1]
    operations = initial_ops
    
    for _ in range(rounds):
        improved = False
        nodes = list(graph.nodes())
        operations += 1
        
        for v in nodes:
            operations += 1
            if v not in coloring: 
                continue
            current_c = coloring[v]
            current_cost = cost_matrix[v, current_c]
            
            forbidden = {coloring[n] for n in graph.neighbors(v) if n in coloring}
            operations += graph.degree(v)
            
            for c in range(n_colors):
                operations += 1
                if c != current_c and c not in forbidden:
                    if cost_matrix[v, c] < current_cost:
                        coloring[v] = c
                        current_cost = cost_matrix[v, c]
                        improved = True
                        break 
        if not improved:
            break
            
    return {'coloring': coloring, 'operations': operations}

# ============================================================================
# 1. WEIGHTED SET COVER APPROXIMATION - O(ln n)
# ============================================================================

def weighted_set_cover_approximation(graph: nx.Graph, cost_matrix: np.ndarray, heuristic: str = "greedy_ratio") -> Dict[str, Any]:

    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    
    coloring = {}
    uncovered = set(graph.nodes())
    total_cost = 0.0
    operations = 0
    max_iterations = n_vertices * n_colors
    
    # Pre-cálculo para heurísticas
    if heuristic == 'max_degree':
        degrees = dict(graph.degree())
        operations += n_vertices
    
    while uncovered and operations < max_iterations:
        operations += 1
        # Check global timeout frequently
        try:
            check_global_timeout()
        except Exception:
            break
        
        best_ratio = float('inf')
        best_set = None
        best_color = -1
        best_set_cost = 0.0
        
        # Intentar encontrar el mejor conjunto para cada color
        for color in range(n_colors):
            operations += 1
            
            # Construir Conjunto Independiente Maximal (MIS)
            current_independent_set = set()
            candidates = []
            
            for v in uncovered:
                operations += 1
                is_valid_candidate = True
                for neighbor in graph.neighbors(v):
                    operations += 1
                    if coloring.get(neighbor) == color:
                        is_valid_candidate = False
                        break
                if is_valid_candidate:
                    candidates.append(v)
            
            if not candidates:
                continue

            # Ordenar candidatos según heurística
            if heuristic == 'max_degree':
                candidates.sort(key=lambda v: degrees[v], reverse=True)
            elif heuristic == 'min_cost':
                candidates.sort(key=lambda v: cost_matrix[v, color])
            else:  # 'greedy_ratio'
                candidates.sort(key=lambda v: cost_matrix[v, color])
            
            operations += len(candidates)

            # Construcción Greedy del MIS
            blocked_in_pass = set()
            for v in candidates:
                operations += 1
                if v not in blocked_in_pass:
                    current_independent_set.add(v)
                    blocked_in_pass.add(v)
                    blocked_in_pass.update(graph.neighbors(v))
                    operations += graph.degree(v)
            
            if not current_independent_set:
                continue

            # Evaluar el Ratio
            current_set_cost = sum(cost_matrix[v, color] for v in current_independent_set)
            operations += len(current_independent_set)
            vertices_covered_count = len(current_independent_set)
            
            ratio = current_set_cost / vertices_covered_count
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_set = current_independent_set
                best_color = color
                best_set_cost = current_set_cost
        
        if best_set is None:
            break
            
        # Aplicar solución parcial
        for v in best_set:
            coloring[v] = best_color
            uncovered.remove(v)
        
        total_cost += best_set_cost

    # Fase de limpieza
    for v in list(uncovered):
        operations += 1
        best_c = -1
        min_c_cost = float('inf')
        for c in range(n_colors):
            operations += 1
            if all(coloring.get(n) != c for n in graph.neighbors(v)):
                operations += graph.degree(v)
                if cost_matrix[v, c] < min_c_cost:
                    min_c_cost = cost_matrix[v, c]
                    best_c = c
        
        if best_c != -1:
            coloring[v] = best_c
            total_cost += min_c_cost
            uncovered.remove(v)

    execution_time = time.time() - start_time
    is_feasible = is_proper_coloring(graph, coloring) and len(coloring) == n_vertices
    
    return convert_numpy_types({
        'solution': coloring,
        'cost': total_cost if is_feasible else float('inf'),
        'execution_time': execution_time,
        'operations': operations,
        'optimal': False,
        'algorithm': f'wsc_greedy_{heuristic}',
        'heuristic_used': heuristic,
        'feasible': is_feasible,
        'approximation_factor': 'O(ln |V|)',
        'reference': 'Informe Sección 3.2.1'
    })

def improved_weighted_set_cover(graph: nx.Graph, cost_matrix: np.ndarray, heuristic: str = "greedy_ratio") -> Dict[str, Any]:

    # 1. Obtener solución inicial
    result = weighted_set_cover_approximation(graph, cost_matrix, heuristic)
    
    if not result['feasible']:
        return result
        
    start_time_ls = time.time()
    coloring = result['solution']
    current_total_cost = result['cost']
    n_colors = cost_matrix.shape[1]
    operations = result['operations']
    
    # 2. Búsqueda Local (Hill Climbing)
    improved = True
    ls_iterations = 0
    max_ls_iter = 50
    
    while improved and ls_iterations < max_ls_iter:
        improved = False
        ls_iterations += 1
        operations += 1
        
        for v in graph.nodes():
            operations += 1
            current_color = coloring[v]
            current_v_cost = cost_matrix[v, current_color]
            
            best_neighbor_color = current_color
            best_neighbor_cost = current_v_cost
            
            neighbor_colors = {coloring[n] for n in graph.neighbors(v) if n in coloring}
            operations += graph.degree(v)
            
            for c in range(n_colors):
                operations += 1
                if c == current_color: 
                    continue
                
                if cost_matrix[v, c] < best_neighbor_cost:
                    if c not in neighbor_colors:
                        best_neighbor_color = c
                        best_neighbor_cost = cost_matrix[v, c]
            
            if best_neighbor_color != current_color:
                coloring[v] = best_neighbor_color
                current_total_cost -= (current_v_cost - best_neighbor_cost)
                improved = True
    
    result['solution'] = coloring
    result['cost'] = current_total_cost
    result['execution_time'] += (time.time() - start_time_ls)
    result['operations'] = operations
    result['algorithm'] = f'wsc_greedy_{heuristic}_plus_local_search'
    result['local_search_iterations'] = ls_iterations
    
    return convert_numpy_types(result)

# ============================================================================
# 2. APROXIMACIÓN PARA GRAFOS DE INTERVALO (Y CORDALES)
# ============================================================================

def interval_graph_approximation(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:

    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    operations = 0

    # 1. Verificar cordalidad
    operations += n_vertices * n_vertices
    if not nx.is_chordal(graph):
         return {
            'solution': {}, 
            'cost': float('inf'), 
            'execution_time': time.time() - start_time,
            'operations': operations,
            'feasible': False,
            'error': 'El grafo no es cordal', 
            'algorithm': 'interval_peo_approximation'
        }
    
    # Obtener PEO (Perfect Elimination Ordering)
    try:
        operations += n_vertices * n_vertices
        peo_order = list(nx.chordal_graph_cliques(graph))
        vertex_order = []
        seen = set()
        for clique in peo_order:
            for v in clique:
                if v not in seen:
                    vertex_order.append(v)
                    seen.add(v)
    except:
        # Fallback: Ordenar por grado
        operations += n_vertices
        vertex_order = sorted(graph.nodes(), key=lambda n: graph.degree(n))

    # 2. Greedy Coloring sobre el orden PEO
    coloring = {}
    
    for vertex in vertex_order:
        operations += 1
        
        forbidden = {coloring[nbr] for nbr in graph.neighbors(vertex) if nbr in coloring}
        operations += graph.degree(vertex)
        
        best_c = -1
        min_cost = float('inf')
        
        for c in range(n_colors):
            operations += 1
            if c not in forbidden:
                if cost_matrix[vertex, c] < min_cost:
                    min_cost = cost_matrix[vertex, c]
                    best_c = c
        
        if best_c != -1:
            coloring[vertex] = best_c
        else:
            return {
                'solution': {}, 
                'cost': float('inf'), 
                'execution_time': time.time() - start_time,
                'operations': operations,
                'feasible': False, 
                'error': 'K insuficiente',
                'algorithm': 'interval_peo_approximation'
            }

    # 3. Optimización Local
    final_coloring = _apply_local_search(graph, coloring, cost_matrix, operations)
    operations = final_coloring['operations']
    
    is_feasible = is_proper_coloring(graph, final_coloring['coloring'])
    total_cost = evaluate_cost(final_coloring['coloring'], cost_matrix)

    return convert_numpy_types({
        'solution': final_coloring['coloring'],
        'cost': total_cost,
        'execution_time': time.time() - start_time,
        'operations': operations,
        'algorithm': 'interval_peo_approximation',
        'feasible': is_feasible,
        'approximation_factor': 'O(√|V|)',
        'reference': 'Informe Sección 3.2.2'
    })
