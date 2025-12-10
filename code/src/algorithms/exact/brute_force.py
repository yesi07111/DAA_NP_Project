"""
Solucionador de fuerza bruta para el Proyecto DAA - MCCPP
"""

import itertools
import time
from typing import Dict
import networkx as nx
import numpy as np
from ...utils.graph_utils import is_proper_coloring
from ...utils.cost_utils import evaluate_solution
from ...utils.io_utils import convert_numpy_types


def brute_force_solver(
    graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0
) -> Dict[str, any]:
    """
    Solucionador de fuerza bruta que enumera todas las coloraciones posibles para el Proyecto DAA - MCCPP

    Args:
        graph: grafo de networkx
        cost_matrix: matriz de costos n_vertices x n_colors
        time_limit: tiempo máximo de ejecución en segundos

    Returns:
        diccionario con la solución y métricas
    """
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())

    start_time = time.time()
    best_solution = None
    best_cost = float("inf")
    solutions_checked = 0

    # Generar todas las coloraciones posibles (asignaciones de vértices a colores)
    for i, coloring_vector in enumerate(
        itertools.product(range(n_colors), repeat=n_vertices)
    ):
        # Verificar límite de tiempo cada 1000 iteraciones para no ralentizar demasiado
        if i % 1000 == 0 and time.time() - start_time > time_limit:
            return None

        # Crear diccionario de coloración
        coloring = {vertex: color for vertex, color in zip(vertices, coloring_vector)}

        # Verificar si la coloración es válida
        if is_proper_coloring(graph, coloring):
            cost = evaluate_solution(coloring, cost_matrix)
            solutions_checked += 1

            if cost < best_cost:
                best_cost = cost
                best_solution = coloring

    end_time = time.time()
    execution_time = end_time - start_time

    # Convertir tipos de NumPy a tipos nativos de Python para serialización JSON
    if best_solution is not None:
        best_solution = {int(k): int(v) for k, v in best_solution.items()}

    result = {
        "solution": best_solution,
        "cost": float(best_cost) if best_solution is not None else float("inf"),
        "execution_time": float(execution_time),
        "solutions_checked": int(solutions_checked),
        "optimal": (execution_time <= time_limit) and (best_solution is not None),
        "time_exceeded": execution_time > time_limit,
        "feasible": is_proper_coloring(graph, best_solution) if best_solution else False,
    }

    return result


def brute_force_with_backtracking(
    graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0
) -> Dict[str, any]:
    """
    Fuerza bruta con backtracking: más eficiente para grafos dispersos en el Proyecto DAA - MCCPP

    Args:
        graph: grafo de networkx
        cost_matrix: matriz de costos n_vertices x n_colors
        time_limit: tiempo máximo de ejecución en segundos

    Returns:
        diccionario con la solución y métricas
    """
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]
    vertices = list(graph.nodes())

    start_time = time.time()
    best_solution = None
    best_cost = float("inf")
    solutions_checked = 0
    time_exceeded = [False]  # Usar lista para poder modificar en scope anidado

    def backtrack(coloring: Dict[int, int], index: int, check_interval: int = 100):
        nonlocal best_solution, best_cost, solutions_checked

        # Verificar límite de tiempo cada 'check_interval' nodos del árbol de búsqueda
        if (
            solutions_checked % check_interval == 0
            and time.time() - start_time > time_limit
        ):
            time_exceeded[0] = True
            return

        if time_exceeded[0]:
            return

        # Si todos los vértices están coloreados, verificar solución
        if index == n_vertices:
            if is_proper_coloring(graph, coloring):
                cost = evaluate_solution(coloring, cost_matrix)
                solutions_checked += 1
                if cost < best_cost:
                    best_cost = cost
                    best_solution = coloring.copy()
            return

        current_vertex = vertices[index]

        # Probar todos los colores para el vértice actual en orden de costo ascendente
        colors_by_cost = sorted(
            range(n_colors), key=lambda c: cost_matrix[current_vertex, c]
        )

        for color in colors_by_cost:
            # Verificar si este color es válido para el vértice actual
            valid = True
            for neighbor in graph.neighbors(current_vertex):
                if coloring.get(neighbor) == color:
                    valid = False
                    break

            if valid:
                coloring[current_vertex] = color
                backtrack(coloring.copy(), index + 1, check_interval)
                del coloring[current_vertex]

                # Salir si se excedió el tiempo
                if time_exceeded[0]:
                    return

    # Iniciar backtracking con verificación más frecuente para instancias grandes
    check_interval = max(
        1, min(100, n_vertices // 10)
    )  # Ajustar frecuencia de verificación
    backtrack({}, 0, check_interval)

    if time_exceeded[0] or not best_solution:
        return None

    end_time = time.time()
    execution_time = end_time - start_time

    # Convertir tipos de NumPy a tipos nativos de Python para serialización JSON
    if best_solution is not None:
        best_solution = {int(k): int(v) for k, v in best_solution.items()}

    result = {
        "solution": best_solution,
        "cost": float(best_cost) if best_solution is not None else float("inf"),
        "execution_time": float(execution_time),
        "solutions_checked": int(solutions_checked),
        "optimal": (not time_exceeded[0]) and (best_solution is not None),
        "time_exceeded": time_exceeded[0],
        "algorithm": "backtracking",
        "feasible": is_proper_coloring(graph, best_solution) if best_solution else False,
    }

    return result


def intelligent_brute_force(
    graph: nx.Graph, cost_matrix: np.ndarray, time_limit: float = 300.0
) -> Dict[str, any]:
    """
    Fuerza bruta inteligente que usa poda por costo y ordenamiento de vértices para el Proyecto DAA - MCCPP

    Args:
        graph: grafo de networkx
        cost_matrix: matriz de costos n_vertices x n_colors
        time_limit: tiempo máximo de ejecución en segundos

    Returns:
        diccionario con la solución y métricas
    """
    n_vertices = graph.number_of_nodes()
    n_colors = cost_matrix.shape[1]

    start_time = time.time()
    best_solution = None
    best_cost = float("inf")
    solutions_checked = 0
    time_exceeded = [False]

    # Ordenar vértices por grado descendente (los más restrictivos primero)
    degrees = dict(graph.degree())
    vertices = sorted(graph.nodes(), key=lambda v: degrees[v], reverse=True)

    def backtrack(coloring: Dict[int, int], index: int, current_cost: float):
        nonlocal best_solution, best_cost, solutions_checked

        # Verificar límite de tiempo cada 10 soluciones
        if solutions_checked % 10 == 0 and time.time() - start_time > time_limit:
            time_exceeded[0] = True
            return

        if time_exceeded[0] or current_cost >= best_cost:
            return

        # Si todos los vértices están coloreados
        if index == n_vertices:
            if is_proper_coloring(graph, coloring):
                solutions_checked += 1
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = coloring.copy()
            return

        current_vertex = vertices[index]

        # Probar colores en orden de costo ascendente
        colors_by_cost = sorted(
            range(n_colors), key=lambda c: cost_matrix[current_vertex, c]
        )

        for color in colors_by_cost:
            # Verificar validez del color
            valid = True
            for neighbor in graph.neighbors(current_vertex):
                if coloring.get(neighbor) == color:
                    valid = False
                    break

            if valid:
                coloring[current_vertex] = color
                new_cost = current_cost + cost_matrix[current_vertex, color]
                backtrack(coloring.copy(), index + 1, new_cost)
                del coloring[current_vertex]

                if time_exceeded[0]:
                    return

    # Iniciar backtracking inteligente
    backtrack({}, 0, 0.0)

    end_time = time.time()
    execution_time = end_time - start_time

    # Convertir tipos de NumPy a tipos nativos de Python para serialización JSON
    if best_solution is not None:
        best_solution = {int(k): int(v) for k, v in best_solution.items()}

    print(best_solution)
    result = {
        "solution": best_solution,
        "cost": float(best_cost) if best_solution is not None else float("inf"),
        "execution_time": float(execution_time),
        "solutions_checked": int(solutions_checked),
        "optimal": (not time_exceeded[0]) and (best_solution is not None),
        "time_exceeded": time_exceeded[0],
        "algorithm": "intelligent_backtracking",
        "feasible": is_proper_coloring(graph, best_solution) if best_solution else False,
    }

    return convert_numpy_types(result)
