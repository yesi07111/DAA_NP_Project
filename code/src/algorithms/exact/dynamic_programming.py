# Solucionador DP (precomputación) para Grafos de Intervalo - FISP-MDC
# precomputa W_f(s,e) en O(k n^2) y DP bidimensional A[t][i].
# Retorna solución reconstruible (color por vértice), costo, factibilidad y tiempo.
import time
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any

from src.utils.io_utils import convert_numpy_types
from src.utils.cost_utils import evaluate_solution
from src.utils.graph_utils import is_proper_coloring


INF = float("inf")


def dynamic_programming_tree(
    graph: nx.Graph,
    cost_matrix: np.ndarray,
    debug: bool = False,
) -> Dict[str, any]:
    total_nodes = graph.number_of_nodes()
    total_colors = cost_matrix.shape[1]

    DP = np.full((total_nodes, total_colors), INF, dtype=np.float64)

    def dfs(node: int, parent: int) -> None:
        is_leaf = True
        for neighbor in graph.neighbors(node):
            if neighbor == parent:
                continue

            is_leaf = False

            dfs(neighbor, node)
        for color in range(total_colors):
            total_score = cost_matrix[node, color]
            for child_node in graph.neighbors(node):
                if child_node == parent:
                    continue

                best_child_score = INF

                for child_color in range(total_colors):
                    # print(">>>>>>>>", color, child_color)
                    if color == child_color:
                        continue

                    best_child_score = min(
                        best_child_score, DP[child_node, child_color]
                    )
                # print(">>>>", best_child_score, node, child_node)

                total_score += best_child_score

            DP[node, color] = min(DP[node, color], total_score)

        if is_leaf:
            for color in range(total_colors):
                DP[node, color] = cost_matrix[node, color]
            return

    dfs(0, -1)

    # Reconstruir solución desde DP
    root_min_cost = np.min(DP[0, :])
    root_color = np.argmin(DP[0, :])

    solution = {}

    def reconstruct(node: int, parent: int, node_color: int) -> None:
        solution[node] = node_color
        for neighbor in graph.neighbors(node):
            if neighbor == parent:
                continue
            child_color = -1
            child_score = INF
            for color in range(total_colors):
                if color == node_color:
                    continue

                if child_score > DP[neighbor, color]:
                    child_score = DP[neighbor, color]
                    child_color = color

            reconstruct(neighbor, node, child_color)

    reconstruct(0, -1, root_color)

    # print(convert_numpy_types(solution))
    # print(f"Costo: {evaluate_solution(convert_numpy_types(solution), cost_matrix)}")
    # print(f"Factible: {is_proper_coloring(graph, convert_numpy_types(solution))}")

    result = {
        "solution": solution,
        "cost": root_min_cost,
        "feasible": is_proper_coloring(graph, solution),
        "execution_time": 0,
        "method": "dp_precompute_Wf_Ati",
        "A_table": DP,
        "nodes_order": None,
    }

    return convert_numpy_types(result)
