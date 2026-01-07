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

    # Fase de limpieza - MEJORADA para garantizar factibilidad
    for v in list(uncovered):
        operations += 1
        neighbor_colors = {coloring.get(n) for n in graph.neighbors(v) if n in coloring}
        operations += graph.degree(v)
        
        # Buscar el mejor color disponible (no usado por vecinos)
        best_c = -1
        min_c_cost = float('inf')
        for c in range(n_colors):
            operations += 1
            if c not in neighbor_colors:
                if cost_matrix[v, c] < min_c_cost:
                    min_c_cost = cost_matrix[v, c]
                    best_c = c
        
        # Si no hay color disponible, usar el de menor costo (aunque sea conflictivo)
        # Luego se arreglará en post-procesamiento
        if best_c == -1:
            best_c = np.argmin(cost_matrix[v, :])
            min_c_cost = cost_matrix[v, best_c]
            operations += n_colors
        
        coloring[v] = best_c
        total_cost += min_c_cost
        uncovered.remove(v)

    execution_time = time.time() - start_time
    is_feasible = is_proper_coloring(graph, coloring) and len(coloring) == n_vertices
    
    # Si no es factible, aplicar post-procesamiento para hacerlo factible
    if not is_feasible and len(coloring) == n_vertices:
        # Arreglar conflictos
        for _ in range(10):  # Máximo 10 iteraciones
            conflicts = []
            for u, v_node in graph.edges():
                if u in coloring and v_node in coloring and coloring[u] == coloring[v_node]:
                    conflicts.append((u, v_node))
                    operations += 1
            
            if not conflicts:
                is_feasible = True
                break
            
            for u, v_node in conflicts:
                operations += 1
                # Intentar mover v_node a un color mejor
                neighbor_colors = {coloring.get(n) for n in graph.neighbors(v_node) if n in coloring}
                for c in range(n_colors):
                    operations += 1
                    if c not in neighbor_colors:
                        old_cost = cost_matrix[v_node, coloring[v_node]]
                        new_cost = cost_matrix[v_node, c]
                        coloring[v_node] = c
                        total_cost = total_cost - old_cost + new_cost
                        break
    
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
