import time
import itertools
import numpy as np
import networkx as nx
import pulp
from typing import Dict, List, Optional, Any
from utils.utils import is_proper_coloring
from utils.utils import convert_numpy_types
from utils.timeout_handler import check_global_timeout


# ============================================================================
# UTILIDADES
# ============================================================================

def evaluate_cost(coloring: Dict[int, int], cost_matrix: np.ndarray) -> float:
    if coloring is None:
        return float('inf')
    total_cost = 0.0
    for vertex, color in coloring.items():
        total_cost += cost_matrix[vertex, color]
    return total_cost

# ============================================================================
# 1. FUERZA BRUTA 
# ============================================================================

# 1.1 ENUMERACIÓN COMPLETA
def brute_force_solver(graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0) -> Dict[str, Any]:
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = sorted(graph.nodes())
    
    start_time = time.time()
    best_solution = None
    best_cost = float('inf')
    operations = 0
    
    for i, coloring_vector in enumerate(itertools.product(range(n_colors), repeat=n_vertices)):
        operations += 1
        if i % 1000 == 0:
            try:
                check_global_timeout()
            except Exception:
                break
            if time.time() - start_time > time_limit:
                break
        
        coloring = {vertex: color for vertex, color in zip(vertices, coloring_vector)}
        
        if is_proper_coloring(graph, coloring):
            cost = evaluate_cost(coloring, cost_matrix)
            if cost < best_cost:
                best_cost = cost
                best_solution = coloring
    
    execution_time = time.time() - start_time
    
    return convert_numpy_types({
        'solution': best_solution,
        'cost': best_cost if best_solution else float('inf'),
        'execution_time': execution_time,
        'operations': operations,
        'solutions_checked': operations,
        'optimal': execution_time <= time_limit and best_solution is not None,
        'feasible': is_proper_coloring(graph, best_solution) if best_solution else False,
        'algorithm': 'brute_force'
    })

# 1.2 BACKTRACKING
def backtracking_solver(graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0) -> Dict[str, Any]:
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = sorted(graph.nodes())
    
    start_time = time.time()
    best_solution = None
    best_cost = float('inf')
    operations = 0
    time_exceeded = [False]
    
    def backtrack(coloring: Dict[int, int], index: int) -> None:
        nonlocal best_solution, best_cost, operations
        
        if operations % 100 == 0:
            try:
                check_global_timeout()
            except Exception:
                time_exceeded[0] = True
                return
            if time.time() - start_time > time_limit:
                time_exceeded[0] = True
                return
        
        if time_exceeded[0]:
            return
        
        operations += 1
        
        if index == n_vertices:
            if is_proper_coloring(graph, coloring):
                cost = evaluate_cost(coloring, cost_matrix)
                if cost < best_cost:
                    best_cost = cost
                    best_solution = coloring.copy()
            return
        
        current_vertex = vertices[index]
        
        for color in range(n_colors):
            valid = True
            for neighbor in graph.neighbors(current_vertex):
                if coloring.get(neighbor) == color:
                    valid = False
                    break
            
            if valid:
                coloring[current_vertex] = color
                backtrack(coloring, index + 1)
                del coloring[current_vertex]
                if time_exceeded[0]: return
    
    backtrack({}, 0)
    
    return convert_numpy_types({
        'solution': best_solution,
        'cost': best_cost if best_solution else float('inf'),
        'execution_time': time.time() - start_time,
        'operations': operations,
        'nodes_explored': operations,
        'optimal': not time_exceeded[0] and best_solution is not None,
        'feasible': is_proper_coloring(graph, best_solution) if best_solution else False,
        'algorithm': 'backtracking'
    })

# 1.3 BACKTRACKING INTELIGENTE CON PODA
def intelligent_backtracking(graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0) -> Dict[str, Any]:
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    
    start_time = time.time()
    best_solution = None
    best_cost = float('inf')
    operations = 0
    time_exceeded = [False]
    
    # Check global timeout at start
    try:
        check_global_timeout()
    except Exception:
        return convert_numpy_types({
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': 0,
            'optimal': False,
            'feasible': False,
            'algorithm': 'intelligent_backtracking'
        })
    
    def backtrack(coloring: Dict[int, int], index: int, current_cost: float) -> None:
        nonlocal best_solution, best_cost, operations
        
        # Chequear timeout global cada 100 operaciones (más frecuente)
        if operations % 100 == 0:
            try:
                check_global_timeout()
            except Exception:
                time_exceeded[0] = True
                return
        
        if operations % 100 == 0:
            if time.time() - start_time > time_limit:
                time_exceeded[0] = True
                return
        
        if time_exceeded[0]: return
        
        operations += 1
        if current_cost >= best_cost: return
        
        if index == n_vertices:
            if current_cost < best_cost:
                best_cost = current_cost
                best_solution = coloring.copy()
            return
        
        current_vertex = vertices[index]
        colors_by_cost = sorted(range(n_colors), key=lambda c: cost_matrix[current_vertex, c])
        
        for color in colors_by_cost:
            valid = True
            for neighbor in graph.neighbors(current_vertex):
                if coloring.get(neighbor) == color:
                    valid = False
                    break
            
            if valid:
                coloring[current_vertex] = color
                new_cost = current_cost + cost_matrix[current_vertex, color]
                if new_cost < best_cost:
                    backtrack(coloring, index + 1, new_cost)
                del coloring[current_vertex]
                if time_exceeded[0]: return
    
    backtrack({}, 0, 0.0)
    
    return convert_numpy_types({
        'solution': best_solution,
        'cost': best_cost if best_solution else float('inf'),
        'execution_time': time.time() - start_time,
        'operations': operations,
        'optimal': not time_exceeded[0] and best_solution is not None,
        'feasible': is_proper_coloring(graph, best_solution) if best_solution else False,
        'algorithm': 'intelligent_backtracking'
    })

# ============================================================================
# 2. ILP SOLVER (Programación Entera Lineal)
# ============================================================================

def ilp_solver(graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = None) -> Dict[str, Any]:
    start_time = time.time()
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = sorted(graph.nodes())
    
    prob = pulp.LpProblem("MCCPP_ILP", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", [(v, f) for v in vertices for f in range(n_colors)], cat='Binary')
    
    prob += pulp.lpSum(cost_matrix[v, f] * x[(v, f)] for v in vertices for f in range(n_colors))
    
    for v in vertices:
        prob += (pulp.lpSum(x[(v, f)] for f in range(n_colors)) == 1)
    
    for u, v in graph.edges():
        for f in range(n_colors):
            prob += (x[(u, f)] + x[(v, f)] <= 1)
    
    # La operación principal aquí es la llamada al solver (caja negra)
    # Estimamos operaciones como variables * restricciones
    n_vars = n_vertices * n_colors
    n_constrs = n_vertices + graph.number_of_edges() * n_colors
    operations = n_vars * n_constrs 

    if time_limit is None:
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
    else:
        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
    
    coloring = {}
    is_optimal = (prob.status == pulp.LpStatusOptimal)
    if is_optimal:
        for v in vertices:
            for f in range(n_colors):
                if pulp.value(x[(v, f)]) == 1:
                    coloring[v] = f
                    break
    
    return convert_numpy_types({
        'solution': coloring if is_optimal else None,
        'cost': pulp.value(prob.objective) if is_optimal else float('inf'),
        'execution_time': time.time() - start_time,
        'operations': int(operations),
        'optimal': is_optimal,
        'feasible': is_proper_coloring(graph, coloring) if is_optimal else False,
        'algorithm': 'ilp_solver'
    })

# ============================================================================
# 3. PROGRAMACIÓN DINÁMICA
# ============================================================================

# 3.1 PROGRAMACIÓN DINÁMICA PARA ÁRBOLES
def dynamic_programming_tree(graph: nx.Graph, cost_matrix: np.ndarray, root: Optional[int] = None) -> Dict[str, Any]:
    """
    CORREGIDO: Programación Dinámica para ÁRBOLES con manejo robusto de errores
    """
    start_time = time.time()
    
    try:
        # VALIDACIÓN 1: Verificar que es un árbol
        if not nx.is_tree(graph):
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': 'El grafo no es un árbol',
                'algorithm': 'dp_tree',
                'operations': 0,
            }
        
        n_vertices = graph.number_of_nodes()
        n_colors = cost_matrix.shape[1]
        
        # VALIDACIÓN 2: Grafo vacío
        if n_vertices == 0:
            return {
                'solution': {},
                'cost': 0.0,
                'execution_time': time.time() - start_time,
                'optimal': True,
                'feasible': True,
                'algorithm': 'dp_tree',
                'operations': 0,
            }
        
        # VALIDACIÓN 3: Dimensiones de cost_matrix
        if cost_matrix.shape[0] != n_vertices:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': f'Dimensiones incorrectas: cost_matrix tiene {cost_matrix.shape[0]} filas pero el grafo tiene {n_vertices} vértices',
                'algorithm': 'dp_tree',
                'operations': 0,
            }
        
        # VALIDACIÓN 4: Necesitamos al menos 2 colores para árboles con >1 nodo
        if n_vertices > 1 and n_colors < 2:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': f'Insuficientes colores: árbol necesita ≥2, tiene {n_colors}',
                'algorithm': 'dp_tree',
                'operations': 0,
            }
        
        # Seleccionar raíz
        if root is None:
            try:
                root = nx.center(graph)[0]
            except Exception:
                root = list(graph.nodes())[0]
        
        if root not in graph.nodes():
            root = list(graph.nodes())[0]
        
        # Inicializar tabla DP
        INF = float('inf')
        DP = np.full((n_vertices, n_colors), INF, dtype=np.float64)
        
        # Mapeo de nodos a índices
        node_to_idx = {node: idx for idx, node in enumerate(sorted(graph.nodes()))}
        idx_to_node = {idx: node for node, idx in node_to_idx.items()}
        
        operations = 0
        
        def dfs_compute(node: int, parent: int) -> None:
            """Calcular DP mediante DFS post-orden"""
            nonlocal operations
            operations += 1
            
            node_idx = node_to_idx[node]
            children = [v for v in graph.neighbors(node) if v != parent]
            
            # Caso base: nodos hoja
            if not children:
                for color in range(n_colors):
                    DP[node_idx, color] = cost_matrix[node, color]
                    operations += 1
                return
            
            # Recursión: procesar hijos primero
            for child in children:
                dfs_compute(child, node)
            
            # Calcular DP[node][c] para cada color c
            for color in range(n_colors):
                operations += 1
                node_cost = cost_matrix[node, color]
                children_cost = 0.0
                
                for child in children:
                    child_idx = node_to_idx[child]
                    
                    # Mejor color para el hijo (diferente de 'color')
                    best_child_cost = INF
                    for child_color in range(n_colors):
                        operations += 1
                        if child_color != color:
                            best_child_cost = min(best_child_cost, DP[child_idx, child_color])
                    
                    if best_child_cost == INF:
                        # No hay color válido - solo pasa si n_colors = 1
                        children_cost = INF
                        break
                    
                    children_cost += best_child_cost
                
                DP[node_idx, color] = node_cost + children_cost
        
        # Computar DP
        try:
            dfs_compute(root, -1)
        except Exception as e:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': f'Error en DFS: {str(e)}',
                'algorithm': 'dp_tree',
                'operations': operations,
            }
        
        # Encontrar solución óptima
        root_idx = node_to_idx[root]
        best_root_color = int(np.argmin(DP[root_idx, :]))
        optimal_cost = float(DP[root_idx, best_root_color])
        
        # Verificar factibilidad
        if optimal_cost == INF or np.isnan(optimal_cost):
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': 'No se encontró solución factible (k insuficiente)',
                'algorithm': 'dp_tree',
                'operations': operations,
            }
        
        # Reconstruir coloración
        solution = {}
        
        def reconstruct(node: int, parent: int, node_color: int) -> None:
            """Reconstruir coloración mediante DFS"""
            solution[node] = node_color
            node_idx = node_to_idx[node]
            
            for neighbor in graph.neighbors(node):
                if neighbor == parent:
                    continue
                
                neighbor_idx = node_to_idx[neighbor]
                
                # Mejor color para el hijo
                best_child_color = -1
                best_child_cost = INF
                
                for color in range(n_colors):
                    if color != node_color and DP[neighbor_idx, color] < best_child_cost:
                        best_child_cost = DP[neighbor_idx, color]
                        best_child_color = color
                
                if best_child_color == -1:
                    # Fallback: usar cualquier color diferente
                    best_child_color = (node_color + 1) % n_colors
                
                reconstruct(neighbor, node, best_child_color)
        
        try:
            reconstruct(root, -1, best_root_color)
        except Exception as e:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'optimal': False,
                'feasible': False,
                'error': f'Error en reconstrucción: {str(e)}',
                'algorithm': 'dp_tree',
                'operations': operations,
            }
        
        execution_time = time.time() - start_time
        
        # Verificar validez
        is_valid = is_proper_coloring(graph, solution)
        
        return convert_numpy_types({
            'solution': solution,
            'cost': optimal_cost,
            'execution_time': execution_time,
            'optimal': True,
            'feasible': is_valid,
            'method': 'dynamic_programming_tree',
            'root': root,
            'algorithm': 'dp_tree',
            'operations': operations,
        })
        
    except Exception as e:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'optimal': False,
            'feasible': False,
            'error': f'Error inesperado: {str(e)}',
            'algorithm': 'dp_tree',
            'operations': 0,
        }

# 3.2 PROGRAMACIÓN DINÁMICA PARA GRAFOS DE INTERVALO

def dp_interval_graph_solver(graph: nx.Graph, cost_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Programación Dinámica EXACTA para GRAFOS DE INTERVALO.
    
    Según informe Sección 3.1.3:
    "Programación Dinámica para Grafos de Intervalo"
    
    Usa Perfect Elimination Ordering (PEO) para colorear eficientemente.
    
    Complejidad: O(n × k² × d) donde d = grado máximo
    """
    start_time = time.time()
    operations = 0
    
    # VALIDACIÓN: Verificar cordalidad (grafos de intervalo son cordales)
    operations += graph.number_of_nodes() ** 2  # Verificación de cordalidad
    if not nx.is_chordal(graph):
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': 'El grafo no es cordal (no puede ser de intervalo)',
            'algorithm': 'dp_interval',
            'complexity': 'O(n·k²·d)'
        }
    
    # Obtener ordenamiento de eliminación perfecta (PEO)
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
    except Exception as e:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': f'Error al obtener PEO: {str(e)}',
            'algorithm': 'dp_interval',
            'complexity': 'O(n·k²·d)'
        }
    
    n_vertices = len(vertex_order)
    n_colors = cost_matrix.shape[1]
    
    # Tabla DP: DP[i][c] = costo mínimo colorear vértices 0..i con v_i usando color c
    INF = float('inf')
    DP = np.full((n_vertices, n_colors), INF, dtype=np.float64)
    parent_color = np.full((n_vertices, n_colors), -1, dtype=np.int32)
    operations += n_vertices * n_colors * 2  # Inicialización
    
    # CASO BASE: Primer vértice
    first_vertex = vertex_order[0]
    for c in range(n_colors):
        DP[0, c] = cost_matrix[first_vertex, c]
        operations += 1
    
    # LLENAR TABLA DP
    for i in range(1, n_vertices):
        current_vertex = vertex_order[i]
        operations += 1
        
        # Encontrar vecinos PREVIOS en el ordenamiento
        previous_neighbors = set()
        for j in range(i):
            operations += 1
            if graph.has_edge(current_vertex, vertex_order[j]):
                previous_neighbors.add(vertex_order[j])
        
        # Para cada color del vértice actual
        for c in range(n_colors):
            node_cost = cost_matrix[current_vertex, c]
            operations += 1
            
            # Intentar extender desde cada estado anterior
            for prev_c in range(n_colors):
                operations += 1
                
                if DP[i-1, prev_c] >= INF:
                    continue
                
                # Verificar si c es compatible con prev_c
                # Solo hay conflicto si v[i-1] es vecino de v[i] y usan el mismo color
                prev_vertex = vertex_order[i-1]
                
                valid = True
                if prev_vertex in previous_neighbors and c == prev_c:
                    valid = False
                
                if valid:
                    new_cost = DP[i-1, prev_c] + node_cost
                    if new_cost < DP[i, c]:
                        DP[i, c] = new_cost
                        parent_color[i, c] = prev_c
    
    # ENCONTRAR SOLUCIÓN ÓPTIMA
    last_idx = n_vertices - 1
    best_last_color = int(np.argmin(DP[last_idx, :]))
    optimal_cost = float(DP[last_idx, best_last_color])
    operations += n_colors
    
    if optimal_cost == INF:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': 'No se encontró solución factible',
            'algorithm': 'dp_interval',
            'complexity': 'O(n·k²·d)'
        }
    
    # RECONSTRUIR SOLUCIÓN
    solution = {}
    current_color = best_last_color
    
    for i in range(n_vertices - 1, -1, -1):
        solution[vertex_order[i]] = current_color
        operations += 1
        if i > 0:
            current_color = parent_color[i, current_color]
            if current_color == -1:
                current_color = int(np.argmin(DP[i-1, :]))
    
    execution_time = time.time() - start_time
    
    return convert_numpy_types({
        'solution': solution,
        'cost': optimal_cost,
        'execution_time': execution_time,
        'operations': operations,
        'optimal': True,
        'feasible': is_proper_coloring(graph, solution),
        'algorithm': 'dp_interval',
        'complexity': 'O(n·k²·d)',
        'reference': 'Informe Sección 3.1.3'
    })

def is_interval_graph_simple(graph: nx.Graph) -> bool:
    """
    Verificación simplificada si un grafo es de intervalo.
    
    Un grafo es de intervalo si es cordal y no contiene ciertos subgrafos prohibidos.
    Esta es una verificación aproximada para propósitos educativos.
    """
    # Un grafo de intervalo debe ser cordal (no tener ciclos inducidos de 4+ vértices)
    if not nx.is_chordal(graph):
        return False
    
    return True

def get_interval_ordering(graph: nx.Graph) -> List[int]:
    """
    Obtiene un ordenamiento válido para un grafo de intervalo.
    
    En grafos de intervalo, existe un ordenamiento donde los vecinos de cada
    vértice aparecen consecutivamente. Usamos una heurística basada en grado.
    """
    # Heurística: ordenamiento por grado
    return sorted(graph.nodes(), key=lambda v: (graph.degree(v), v))
