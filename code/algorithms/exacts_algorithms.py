import time
import itertools
import numpy as np
import networkx as nx
import pulp
from typing import Dict, Optional, Any, List, Tuple
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

def compute_interval_representation(graph: nx.Graph) -> Optional[Dict[int, Tuple[int, int]]]:
    """
    Intenta construir una representación de intervalos para el grafo.
    Retorna un diccionario vértice -> (inicio, fin) si es posible, None si no es de intervalo.
    
    Usa el algoritmo de reconocimiento de grafos de intervalo basado en PQ-trees.
    Para simplificar, usamos una aproximación: ordenamos por PEO y asignamos intervalos
    basados en la estructura de vecindad.
    """
    if not nx.is_chordal(graph):
        return None
    
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    
    # Obtener PEO mediante Maximum Cardinality Search
    vertices = list(graph.nodes())
    numbered = set()
    peo = []
    cardinality = {v: 0 for v in vertices}
    
    for _ in range(n):
        unnumbered = [v for v in vertices if v not in numbered]
        v = max(unnumbered, key=lambda x: cardinality[x])
        peo.append(v)
        numbered.add(v)
        for neighbor in graph.neighbors(v):
            if neighbor not in numbered:
                cardinality[neighbor] += 1
    
    peo = list(reversed(peo))
    
    # Asignar intervalos basados en el PEO
    intervals = {}
    position = {v: i for i, v in enumerate(peo)}
    
    for v in vertices:
        neighbors_pos = [position[u] for u in graph.neighbors(v)]
        if neighbors_pos:
            start = min(neighbors_pos + [position[v]])
            end = max(neighbors_pos + [position[v]])
        else:
            start = end = position[v]
        intervals[v] = (start, end)
    
    # Verificar si la representación es válida (esto es una heurística)
    for u in vertices:
        for v in vertices:
            if u >= v:
                continue
            u_start, u_end = intervals[u]
            v_start, v_end = intervals[v]
            
            # Los intervalos se solapan si hay intersección
            overlap = not (u_end < v_start or v_end < u_start)
            has_edge = graph.has_edge(u, v)
            
            if overlap != has_edge:
                # No es un grafo de intervalo válido
                return None
    
    return intervals

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
                    raise ValueError(
                        f"Reconstrucción imposible: no hay color válido para "
                        f"hijo {neighbor} con padre usando color {node_color}"
                    )
                
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
    Algoritmo de Programación Dinámica EXACTO para MCCPP en grafos de intervalo.
    
    Estrategia: Ordena los vértices por el extremo derecho de sus intervalos y procesa
    de izquierda a derecha. El estado mantiene la coloración de los intervalos "activos"
    (aquellos que aún se solapan con el punto actual de procesamiento).
    
    Complejidad: O(n · k^ω) donde ω es el tamaño de la clique máxima (número máximo
    de intervalos que se solapan simultáneamente).
    """
    start_time = time.time()
    operations = 0
    
    # Validar que sea cordal
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
            'algorithm': 'dp_interval_exact',
            'complexity': 'O(n·k^ω)'
        }
    
    n = graph.number_of_nodes()
    k = cost_matrix.shape[1]
    
    if n == 0:
        return {
            'solution': {},
            'cost': 0.0,
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': True,
            'feasible': True,
            'algorithm': 'dp_interval_exact'
        }
    
    # Intentar obtener representación de intervalos
    try:
        operations += n ** 2
        intervals = compute_interval_representation(graph)
        if intervals is None:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': time.time() - start_time,
                'operations': operations,
                'optimal': False,
                'feasible': False,
                'error': 'El grafo no es de intervalo (es cordal pero no satisface la propiedad de intervalos)',
                'algorithm': 'dp_interval_exact',
                'complexity': 'O(n·k^ω)'
            }
    except Exception as e:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': time.time() - start_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': f'Error al verificar propiedad de intervalo: {str(e)}',
            'algorithm': 'dp_interval_exact'
        }
    
    # Ordenar vértices por extremo derecho del intervalo (y por extremo izquierdo como desempate)
    sorted_vertices = sorted(intervals.keys(), key=lambda v: (intervals[v][1], intervals[v][0]))
    
    # Programación dinámica con timeout
    # DP[i] mapea estado -> (costo, coloración_completa)
    # estado = tupla con (vértice, color) de los intervalos activos hasta la posición i
    DP = [dict() for _ in range(n + 1)]
    DP[0][tuple()] = (0.0, {})
    
    timeout_limit = 180.0  # 180 segundos
    timeout_reached = False
    last_complete_iteration = 0
    
    for i in range(n):
        # Verificar timeout antes de cada iteración
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout_limit:
            timeout_reached = True
            last_complete_iteration = i
            break
        
        v = sorted_vertices[i]
        v_start, v_end = intervals[v]
        operations += 1
        
        # Para cada estado anterior
        for prev_state, (prev_cost, prev_coloring) in DP[i].items():
            # Verificar timeout periódicamente durante el procesamiento
            if time.time() - start_time >= timeout_limit:
                timeout_reached = True
                last_complete_iteration = i
                break
            
            operations += 1
            
            # Filtrar el estado anterior: eliminar intervalos que ya terminaron
            # Un intervalo u ha terminado si intervals[u][1] < v_start
            active_state = tuple((u, c) for u, c in prev_state if intervals[u][1] >= v_start)
            
            # Identificar colores prohibidos (vecinos activos)
            forbidden_colors = set()
            for u, c in active_state:
                if graph.has_edge(v, u):
                    forbidden_colors.add(c)
                operations += 1
            
            # Probar cada color disponible para v
            for color_v in range(k):
                operations += 1
                
                if color_v in forbidden_colors:
                    continue
                
                # Construir nuevo estado y coloración
                new_state = active_state + ((v, color_v),)
                new_coloring = prev_coloring.copy()
                new_coloring[v] = color_v
                new_cost = prev_cost + cost_matrix[v, color_v]
                
                # Actualizar tabla DP
                if new_state not in DP[i + 1] or new_cost < DP[i + 1][new_state][0]:
                    DP[i + 1][new_state] = (new_cost, new_coloring)
        
        if timeout_reached:
            break
    
    # Recuperar solución óptima
    execution_time = time.time() - start_time
    
    if timeout_reached:
        # Si se alcanzó el timeout, buscar la mejor solución parcial en la última iteración completa
        best_cost = float('inf')
        best_coloring = None
        best_iteration = 0
        
        # Buscar en todas las iteraciones completadas
        for iter_idx in range(last_complete_iteration, -1, -1):
            if len(DP[iter_idx]) > 0:
                for state, (cost, coloring) in DP[iter_idx].items():
                    if cost < best_cost:
                        best_cost = cost
                        best_coloring = coloring
                        best_iteration = iter_idx
        
        if best_coloring is None:
            return {
                'solution': None,
                'cost': float('inf'),
                'execution_time': execution_time,
                'operations': operations,
                'optimal': False,
                'feasible': False,
                'error': 'Timeout alcanzado sin solución factible',
                'algorithm': 'dp_interval_exact',
                'timeout': True,
                'timeout_limit': timeout_limit
            }
        
        # Completar la coloración con vértices faltantes usando una estrategia greedy
        colored_vertices = set(best_coloring.keys())
        remaining_vertices = [sorted_vertices[j] for j in range(best_iteration, n)]
        
        for v in remaining_vertices:
            if v not in colored_vertices:
                # Asignar el color de menor costo que no conflicte con vecinos
                forbidden_colors = set()
                for neighbor in graph.neighbors(v):
                    if neighbor in best_coloring:
                        forbidden_colors.add(best_coloring[neighbor])
                
                best_color = None
                best_color_cost = float('inf')
                for c in range(k):
                    if c not in forbidden_colors and cost_matrix[v, c] < best_color_cost:
                        best_color = c
                        best_color_cost = cost_matrix[v, c]
                
                if best_color is not None:
                    best_coloring[v] = best_color
                    best_cost += best_color_cost
        
        is_valid = is_proper_coloring(graph, best_coloring)
        
        return convert_numpy_types({
            'solution': best_coloring,
            'cost': best_cost,
            'execution_time': execution_time,
            'operations': operations,
            'optimal': False,  # No es óptimo debido al timeout
            'feasible': is_valid,
            'algorithm': 'dp_interval_exact',
            'complexity': 'O(n·k^ω)',
            'reference': 'DP exacto sobre representación de intervalos (interrumpido por timeout)',
            'note': f'Timeout alcanzado después de {execution_time:.2f}s. Solución completada con heurística greedy.',
            'timeout': True,
            'timeout_limit': timeout_limit,
            'vertices_processed': best_iteration,
            'total_vertices': n
        })
    
    # Caso normal: no se alcanzó timeout
    if len(DP[n]) == 0:
        return {
            'solution': None,
            'cost': float('inf'),
            'execution_time': execution_time,
            'operations': operations,
            'optimal': False,
            'feasible': False,
            'error': 'No se encontró solución factible',
            'algorithm': 'dp_interval_exact'
        }
    
    optimal_cost, optimal_coloring = min(DP[n].values(), key=lambda x: x[0])
    is_valid = is_proper_coloring(graph, optimal_coloring)
    
    return convert_numpy_types({
        'solution': optimal_coloring,
        'cost': optimal_cost,
        'execution_time': execution_time,
        'operations': operations,
        'optimal': True,
        'feasible': is_valid,
        'algorithm': 'dp_interval_exact',
        'complexity': 'O(n·k^ω)',
        'reference': 'DP exacto sobre representación de intervalos con intervalos activos',
        'note': 'Algoritmo exacto solo para grafos de intervalo, no para grafos cordales generales'
    })