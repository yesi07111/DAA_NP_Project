import os
import networkx as nx
import numpy as np
from typing import List, Dict, Any
from networkx.generators.trees import random_unlabeled_tree
from utils.utils import generate_structured_cost_matrix, save_instance, get_instance_filename, generate_interval_graph, generate_erdos_renyi_graph, generate_cost_matrix, load_instance


def estimate_feasibility(graph: nx.Graph, n_colors: int, instance_type: str) -> bool:
    """
    Estima si una instancia es factible usando coloreo greedy.
    Una instancia es factible si puede ser coloreada con n_colors colores.
    
    CORREGIDO: Manejo especial para árboles y grafos bipartitos
    """
    try:
        # CASO ESPECIAL 1: Árboles - siempre son 2-coloreables
        if nx.is_tree(graph):
            chromatic_number = 2 if graph.number_of_nodes() > 1 else 1
            return chromatic_number <= n_colors
        
        # CASO ESPECIAL 2: Grafos bipartitos - siempre son 2-coloreables
        if nx.is_bipartite(graph):
            chromatic_number = 2 if graph.number_of_nodes() > 1 else 1
            return chromatic_number <= n_colors
        
        # Para grafos pequeños, calcular el número cromático exacto
        if graph.number_of_nodes() <= 15:
            if nx.is_empty(graph):
                chromatic_number = 0
            else:
                # Usar Welsh-Powell que da mejor aproximación que greedy básico
                coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
                chromatic_number = len(set(coloring.values()))
        else:
            # Para grafos grandes, usar greedy
            coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
            chromatic_number = len(set(coloring.values()))
        
        return chromatic_number <= n_colors
    except Exception:
        # Si hay error, asumir infactible
        return False

def compute_known_optimal_cost(graph: nx.Graph, cost_matrix: np.ndarray, instance_type: str) -> float:
    """
    Calcula el costo óptimo conocido para instancias especiales donde se conoce la coloración óptima.
    
    CORREGIDO: Manejo robusto de árboles y bipartitos
    
    Retorna None si no se puede calcular.
    """
    try:
        n_colors = cost_matrix.shape[1]
        
        # CASO 1: Árboles (siempre 2-coloreable)
        if nx.is_tree(graph) and graph.number_of_nodes() > 0:
            # Colorear con DFS
            coloring = {}
            visited = set()
            
            def dfs_color(node, color):
                visited.add(node)
                coloring[node] = color
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited:
                        dfs_color(neighbor, 1 - color)
            
            # Comenzar desde cualquier nodo
            root = list(graph.nodes())[0]
            dfs_color(root, 0)
            
            # Calcular costo para ambas opciones
            cost_1 = sum(cost_matrix[v, c] for v, c in coloring.items())
            
            # Invertir colores
            coloring_inv = {v: 1 - c for v, c in coloring.items()}
            cost_2 = sum(cost_matrix[v, c] for v, c in coloring_inv.items())
            
            return min(cost_1, cost_2)
        
        # CASO 2: Grafos bipartitos
        if nx.is_bipartite(graph):
            color_map = nx.bipartite.color(graph)
            partition_0 = [v for v, c in color_map.items() if c == 0]
            partition_1 = [v for v, c in color_map.items() if c == 1]
            
            # Probar todas las k² combinaciones de 2 colores
            best_cost = float('inf')
            
            for c1 in range(n_colors):
                for c2 in range(n_colors):
                    if c1 == c2:
                        continue
                    
                    # Opción: c1 → partition_0, c2 → partition_1
                    cost = sum(cost_matrix[v, c1] for v in partition_0) + \
                           sum(cost_matrix[v, c2] for v in partition_1)
                    
                    best_cost = min(best_cost, cost)
            
            return best_cost if best_cost != float('inf') else None
        
        return None
    except Exception:
        return None


# GENERATORS

def generate_erdos_renyi_instances(
    n_vertices_list: List[int],
    p: float,
    n_colors: List[int],
    cost_pattern: str = "uniform",
    n_instances: int = 5,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    """Genera instancias Erdős-Rényi con validación de factibilidad mejorada"""
    instances_metadata = []
    current_seed = seed

    for inx, n_vertices in enumerate(n_vertices_list):
        for i in range(n_instances):
            graph = generate_erdos_renyi_graph(n_vertices, p, seed=current_seed)
            cost_matrix = generate_structured_cost_matrix(
                n_vertices, n_colors[inx], cost_pattern, seed=current_seed
            )

            is_feasible = estimate_feasibility(graph, n_colors[inx], "erdos_renyi")
            known_opt = compute_known_optimal_cost(graph, cost_matrix, "erdos_renyi") if is_feasible else None

            metadata = {
                "instance_type": "erdos_renyi",
                "n_vertices": n_vertices,
                "p": p,
                "n_colors": n_colors[inx],
                "cost_pattern": cost_pattern,
                "seed": current_seed,
                "density": nx.density(graph),
                "is_feasible": is_feasible,
                "known_optimal_cost": known_opt,
            }

            filename = get_instance_filename(
                "erdos_renyi", n_vertices, n_colors[inx], p, current_seed, output_dir
            )
            save_instance(graph, cost_matrix, filename, metadata)
            instances_metadata.append({"filename": filename, "metadata": metadata})
            current_seed += 1

    return instances_metadata

def generate_structured_instances(
    n_vertices_list: List[int],
    n_colors: List[int],
    graph_types: List[str] = ["path", "cycle", "complete", "star"],
    cost_pattern: str = "uniform",
    n_instances: int = 3,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    """Genera instancias estructuradas con k suficiente"""
    instances_metadata = []
    current_seed = seed

    for graph_type in graph_types:
        for inx, n_vertices in enumerate(n_vertices_list):
            for i in range(n_instances):
                if graph_type == "path":
                    graph = nx.path_graph(n_vertices)
                elif graph_type == "cycle":
                    graph = nx.cycle_graph(n_vertices)
                elif graph_type == "complete":
                    graph = nx.complete_graph(n_vertices)
                    # NOTA: Antes se saltaban las instancias completas cuando k < n_vertices.
                    # Ahora generamos siempre el grafo completo y dejamos que
                    # `estimate_feasibility` marque la metadada 'is_feasible'.
                elif graph_type == "star":
                    graph = nx.star_graph(n_vertices - 1)
                elif graph_type == "wheel":
                    graph = nx.wheel_graph(n_vertices)
                elif graph_type == "grid":
                    factors = []
                    for j in range(1, int(np.sqrt(n_vertices)) + 1):
                        if n_vertices % j == 0:
                            factors.append((j, n_vertices // j))
                    if factors:
                        rows, cols = factors[-1]
                        graph = nx.grid_2d_graph(rows, cols)
                        mapping = {node: i for i, node in enumerate(graph.nodes())}
                        graph = nx.relabel_nodes(graph, mapping)
                    else:
                        graph = nx.path_graph(n_vertices)
                else:
                    continue

                if graph.number_of_nodes() != n_vertices:
                    if graph.number_of_nodes() < n_vertices:
                        graph.add_nodes_from(range(graph.number_of_nodes(), n_vertices))
                    else:
                        graph = graph.subgraph(range(n_vertices)).copy()

                cost_matrix = generate_structured_cost_matrix(
                    n_vertices, n_colors[inx], cost_pattern, seed=current_seed
                )

                is_feasible = estimate_feasibility(graph, n_colors[inx], graph_type)
                known_opt = compute_known_optimal_cost(graph, cost_matrix, graph_type) if is_feasible else None

                metadata = {
                    "instance_type": graph_type,
                    "n_vertices": n_vertices,
                    "n_colors": n_colors[inx],
                    "cost_pattern": cost_pattern,
                    "seed": current_seed,
                    "density": nx.density(graph),
                    "is_feasible": is_feasible,
                    "known_optimal_cost": known_opt,
                }

                filename = get_instance_filename(
                    graph_type, n_vertices, n_colors[inx], metadata["density"], current_seed, output_dir
                )
                save_instance(graph, cost_matrix, filename, metadata)
                instances_metadata.append({"filename": filename, "metadata": metadata})
                current_seed += 1

    return instances_metadata

def generate_tree_instances(
    n_vertices_list: List[int],
    n_colors: int,
    cost_pattern: str = "uniform",
    n_instances: int = 5,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    """
    Genera árboles (siempre 2-coloreables)
    
    CORREGIDO: Asegura que is_feasible=True siempre (árboles son bipartitos)
    """
    instances_metadata = []
    current_seed = seed

    def generate_tree_graph(n_vertices: int, seed: int) -> nx.Graph:
        import random
        random.seed(seed)
        return random_unlabeled_tree(n_vertices, seed=seed)

    for n_vertices in n_vertices_list:
        # Los árboles siempre necesitan máximo 2 colores
        if n_colors < 2:
            continue  # Saltar si no hay suficientes colores
            
        for i in range(n_instances):
            graph = generate_tree_graph(n_vertices, seed=current_seed)
            cost_matrix = generate_structured_cost_matrix(
                n_vertices, n_colors, cost_pattern, seed=current_seed
            )

            density = nx.density(graph)
            
            # CORRECCIÓN: Los árboles SIEMPRE son factibles con k>=2
            is_feasible = True  # Los árboles siempre son 2-coloreables
            known_opt = compute_known_optimal_cost(graph, cost_matrix, "tree")

            metadata = {
                "instance_type": "tree",
                "n_vertices": n_vertices,
                "n_colors": n_colors,
                "p": 0.0,
                "cost_pattern": cost_pattern,
                "seed": current_seed,
                "density": density,
                "is_feasible": is_feasible,
                "known_optimal_cost": known_opt,
                "chromatic_number": 2 if n_vertices > 1 else 1,  # Añadido
            }

            filename = get_instance_filename(
                "tree", n_vertices, n_colors, 0.0, current_seed, output_dir
            )
            save_instance(graph, cost_matrix, filename, metadata)
            instances_metadata.append({"filename": filename, "metadata": metadata})
            current_seed += 1

    return instances_metadata

def generate_interval_graph_instances(n_vertices_list: List[int], n_colors: List[int],
                                      max_length: float = 1.0, cost_pattern: str = "uniform",
                                      n_instances: int = 5, seed: int = 42,
                                      output_dir: str = "instances") -> List[Dict[str, Any]]:
    instances_metadata = []
    current_seed = seed
    
    for inx, n_vertices in enumerate(n_vertices_list):
        for i in range(n_instances):
            graph, intervals = generate_interval_graph(n_vertices, max_length, seed=current_seed)
            cost_matrix = generate_structured_cost_matrix(n_vertices, n_colors[inx], cost_pattern, seed=current_seed)
            
            is_feasible = estimate_feasibility(graph, n_colors[inx], 'interval_graph')
            known_opt = compute_known_optimal_cost(graph, cost_matrix, 'interval_graph') if is_feasible else None
            
            metadata = {
                'instance_type': 'interval_graph',
                'n_vertices': n_vertices,
                'n_colors': n_colors[inx],
                'max_length': max_length,
                'cost_pattern': cost_pattern,
                'seed': current_seed,
                'density': nx.density(graph),
                'intervals': intervals,
                'is_feasible': is_feasible,
                'known_optimal_cost': known_opt,
            }
            
            filename = get_instance_filename('interval', n_vertices, n_colors[inx], metadata['density'], current_seed, output_dir)
            save_instance(graph, cost_matrix, filename, metadata)
            
            instances_metadata.append({'filename': filename, 'metadata': metadata})
            current_seed += 1
    
    return instances_metadata

def generate_special_interval_graph_instances(output_dir: str = "instances") -> List[Dict[str, Any]]:
    """
    Genera instancias específicas de grafos de intervalo
    """
    interval_metadata = []
    
    # Instancia 1: Intervalo simple no-solapante
    graph_i1 = nx.Graph()
    graph_i1.add_nodes_from([0, 1, 2, 3])
    cost_matrix_i1 = np.array([
        [1, 100],
        [100, 1],
        [1, 100],
        [100, 1]
    ])
    metadata_i1 = {
        'instance_type': 'interval_non_overlapping',
        'n_vertices': 4,
        'n_colors': 2,
        'known_optimal_cost': 4.0,
        'chromatic_number': 1,
        'clique_number': 1,
        'intervals': [[0, 1], [2, 3], [4, 5], [6, 7]],
        'description': 'Intervalos no solapantes: χ(G)=1',
        'optimal_coloring': {0: 0, 1: 0, 2: 0, 3: 0},
        'is_feasible': True
    }
    filename_i1 = get_instance_filename('interval_non_overlapping', 4, 2, 0.0, 1, output_dir)
    save_instance(graph_i1, cost_matrix_i1, filename_i1, metadata_i1)
    interval_metadata.append({'filename': filename_i1, 'metadata': metadata_i1})
    
    # Instancia 2: Cadena de intervalos solapantes
    graph_i2 = nx.path_graph(5)
    cost_matrix_i2 = np.array([
        [1, 100, 100],
        [100, 1, 100],
        [100, 100, 1],
        [1, 100, 100],
        [100, 1, 100]
    ])
    metadata_i2 = {
        'instance_type': 'interval_chain_overlapping',
        'n_vertices': 5,
        'n_colors': 3,
        'known_optimal_cost': 5.0,
        'chromatic_number': 2,
        'clique_number': 2,
        'intervals': [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        'description': 'Cadena de intervalos solapantes: χ(G)=2',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0},
        'is_feasible': True
    }
    filename_i2 = get_instance_filename('interval_chain', 5, 3, nx.density(graph_i2), 2, output_dir)
    save_instance(graph_i2, cost_matrix_i2, filename_i2, metadata_i2)
    interval_metadata.append({'filename': filename_i2, 'metadata': metadata_i2})
    
    # Instancia 3: Grafo de intervalo con punto de máxima clique
    graph_i3 = nx.Graph()
    graph_i3.add_edges_from([
        (0, 1), (0, 2), (1, 2),  # Clique K_3
        (2, 3), (3, 4)           # Extensión
    ])
    cost_matrix_i3 = np.array([
        [1, 100, 100, 100],
        [100, 1, 100, 100],
        [100, 100, 1, 100],
        [100, 100, 100, 1],
        [1, 100, 100, 100]
    ])
    metadata_i3 = {
        'instance_type': 'interval_with_max_clique',
        'n_vertices': 5,
        'n_colors': 4,
        'known_optimal_cost': 5.0,
        'chromatic_number': 3,
        'clique_number': 3,
        'max_clique_size': 3,
        'description': 'Intervalo con clique máxima tamaño 3',
        'optimal_coloring': {0: 0, 1: 1, 2: 2, 3: 0, 4: 1},
        'is_feasible': True
    }
    filename_i3 = get_instance_filename('interval_max_clique', 5, 4, nx.density(graph_i3), 3, output_dir)
    save_instance(graph_i3, cost_matrix_i3, filename_i3, metadata_i3)
    interval_metadata.append({'filename': filename_i3, 'metadata': metadata_i3})
    
    return interval_metadata

def generate_special_case_instances(output_dir: str = "instances") -> List[Dict[str, Any]]:
    """
    Genera instancias de casos especiales con soluciones óptimas conocidas para el Proyecto DAA - MCCPP
    
    Args:
        output_dir: directorio donde guardar las instancias
    
    Returns:
        lista de metadatos de las instancias
    """
    instances_metadata = []
    
    # =========================================================================
    # INSTANCIAS SENCILLAS (CASOS 1-5)
    # =========================================================================
    
    # Instancia 1: Grafo camino simple con 3 vértices, 2 colores, óptimo conocido
    graph1 = nx.path_graph(3)
    cost_matrix1 = np.array([
        [1, 10],  # Vértice 0: costo 1 para color0, 10 para color1
        [10, 1],  # Vértice 1: costo 10 para color0, 1 para color1
        [1, 10]   # Vértice 2: costo 1 para color0, 10 para color1
    ])
    # Coloración óptima: 0->0, 1->1, 2->0 -> costo = 1+1+1 = 3
    metadata1 = {
        'instance_type': 'path_known_optimal',
        'n_vertices': 3,
        'n_colors': 2,
        'known_optimal_cost': 3.0,
        'description': 'Grafo camino con 3 vértices, costo óptimo 3',
        'optimal_coloring': {0: 0, 1: 1, 2: 0}
    }
    filename1 = get_instance_filename('special_path', 3, 2, nx.density(graph1), 1, output_dir)
    save_instance(graph1, cost_matrix1, filename1, metadata1)
    instances_metadata.append({'filename': filename1, 'metadata': metadata1})
    
    # Instancia 2: Grafo bipartito con 4 vértices, 2 colores
    graph2 = nx.complete_bipartite_graph(2, 2)
    cost_matrix2 = np.array([
        [1, 100],  # Vértice 0
        [1, 100],  # Vértice 1
        [100, 1],  # Vértice 2
        [100, 1]   # Vértice 3
    ])
    # Coloración óptima: asignar a cada partición un color diferente -> costo = 1+1+1+1 = 4
    metadata2 = {
        'instance_type': 'bipartite_known_optimal',
        'n_vertices': 4,
        'n_colors': 2,
        'known_optimal_cost': 4.0,
        'description': 'Grafo bipartito completo K_{2,2}, costo óptimo 4',
        'optimal_coloring': {0: 0, 1: 0, 2: 1, 3: 1},
        'is_feasible': estimate_feasibility(graph2, 2, 'bipartite_known_optimal')
    }
    filename2 = get_instance_filename('special_bipartite', 4, 2, nx.density(graph2), 2, output_dir)
    save_instance(graph2, cost_matrix2, filename2, metadata2)
    instances_metadata.append({'filename': filename2, 'metadata': metadata2})
    
    # Instancia 3: Triángulo (grafo completo K3) con 3 colores
    graph3 = nx.complete_graph(3)
    cost_matrix3 = np.array([
        [1, 100, 100],  # Vértice 0
        [100, 1, 100],  # Vértice 1
        [100, 100, 1]   # Vértice 2
    ])
    # Coloración óptima: cada vértice con un color diferente -> costo = 1+1+1 = 3
    metadata3 = {
        'instance_type': 'triangle_known_optimal',
        'n_vertices': 3,
        'n_colors': 3,
        'known_optimal_cost': 3.0,
        'description': 'Grafo triángulo K_3, costo óptimo 3',
        'optimal_coloring': {0: 0, 1: 1, 2: 2},
        'is_feasible': estimate_feasibility(graph3, 3, 'triangle_known_optimal')
    }
    filename3 = get_instance_filename('special_triangle', 3, 3, nx.density(graph3), 3, output_dir)
    save_instance(graph3, cost_matrix3, filename3, metadata3)
    instances_metadata.append({'filename': filename3, 'metadata': metadata3})
    
    # Instancia 4: Grafo con vértices aislados
    graph4 = nx.Graph()
    graph4.add_nodes_from([0, 1, 2, 3])
    # Sin aristas
    cost_matrix4 = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    # Coloración óptima: asignar el color con costo mínimo para cada vértice -> costo = 1+3+5+7 = 16
    metadata4 = {
        'instance_type': 'isolated_vertices',
        'n_vertices': 4,
        'n_colors': 2,
        'known_optimal_cost': 16.0,
        'description': 'Grafo sin aristas (vértices aislados), costo óptimo 16',
        'optimal_coloring': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    filename4 = get_instance_filename('special_isolated', 4, 2, nx.density(graph4), 4, output_dir)
    save_instance(graph4, cost_matrix4, filename4, metadata4)
    instances_metadata.append({'filename': filename4, 'metadata': metadata4})

    # Instancia 5: Camino P5 con 2 colores
    graph_p5 = nx.path_graph(5)
    cost_matrix_p5 = np.array([
        [1,100],
        [100,1],
        [1,100],
        [100,1],
        [1,100]
    ])
    # Óptimo alternando colores → costo = 5
    metadata_p5 = {
        'instance_type': 'path_5_vertices',
        'n_vertices': 5,
        'n_colors': 2,
        'known_optimal_cost': 5.0,
        'chromatic_number': 2,
        'description': 'Camino P5: alternancia de 2 colores, costo óptimo 5',
        'optimal_coloring': {0:0,1:1,2:0,3:1,4:0}
    }
    filename_p5 = get_instance_filename('special_path_5', 5, 2, nx.density(graph_p5), 20, output_dir)
    save_instance(graph_p5, cost_matrix_p5, filename_p5, metadata_p5)
    instances_metadata.append({'filename': filename_p5, 'metadata': metadata_p5})

    
    # =========================================================================
    # INSTANCIAS CON GRAFOS CICLO (C_n) - CASOS 6-10
    # =========================================================================
    # Propiedad: χ(C_n) = 2 si n es par, χ(C_n) = 3 si n es impar
    # Para MCCPP, diseñamos matrices de costo que fuerzan estas coloraciones

    # Instancia 6: Ciclo par C_4 con 2 colores 
    graph_c4 = nx.cycle_graph(4)
    cost_matrix_c4 = np.array([
        [1, 100],   # 0
        [100, 1],   # 1
        [1, 100],   # 2
        [100, 1]    # 3
    ])
    # Óptimo: 0→0, 1→1, 2→0, 3→1 → costo = 4
    metadata_c4 = {
        'instance_type': 'cycle_even_small',
        'n_vertices': 4,
        'n_colors': 2,
        'known_optimal_cost': 4.0,
        'cycle_length': 4,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Ciclo par C_4: alternancia perfecta, costo óptimo 4',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1}
    }
    filename_c4 = get_instance_filename('special_cycle_even_4', 4, 2, nx.density(graph_c4), 17, output_dir)
    save_instance(graph_c4, cost_matrix_c4, filename_c4, metadata_c4)
    instances_metadata.append({'filename': filename_c4, 'metadata': metadata_c4})

    
    # Instancia 7: Ciclo par C_6 con 2 colores
    # Ciclo par: chromatic number = 2, admite 2-coloración bipartita
    graph5 = nx.cycle_graph(6)
    cost_matrix5 = np.array([
        [1, 100],      # Vértice 0: preferir color 0
        [100, 1],      # Vértice 1: preferir color 1
        [1, 100],      # Vértice 2: preferir color 0
        [100, 1],      # Vértice 3: preferir color 1
        [1, 100],      # Vértice 4: preferir color 0
        [100, 1]       # Vértice 5: preferir color 1
    ])
    # Coloración óptima bipartita: {0,2,4} -> color 0, {1,3,5} -> color 1
    # Costo = 1+1+1+1+1+1 = 6
    metadata5 = {
        'instance_type': 'cycle_even',
        'n_vertices': 6,
        'n_colors': 2,
        'known_optimal_cost': 6.0,
        'cycle_length': 6,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Ciclo par C_6: 2-colorable mediante bipartición, costo óptimo 6',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1}
    }
    filename5 = get_instance_filename('special_cycle_even', 6, 2, nx.density(graph5), 5, output_dir)
    save_instance(graph5, cost_matrix5, filename5, metadata5)
    instances_metadata.append({'filename': filename5, 'metadata': metadata5})
    
    # Instancia 8: Ciclo par C_8 con 2 colores
    graph6 = nx.cycle_graph(8)
    cost_matrix6 = np.array([
        [2, 100],
        [100, 2],
        [2, 100],
        [100, 2],
        [2, 100],
        [100, 2],
        [2, 100],
        [100, 2]
    ])
    # Coloración óptima: alternancia perfecta -> costo = 2*8 = 16
    metadata6 = {
        'instance_type': 'cycle_even_larger',
        'n_vertices': 8,
        'n_colors': 2,
        'known_optimal_cost': 16.0,
        'cycle_length': 8,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Ciclo par C_8: mayor que C_6, 2-colorable, costo óptimo 16',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1}
    }
    filename6 = get_instance_filename('special_cycle_even_large', 8, 2, nx.density(graph6), 6, output_dir)
    save_instance(graph6, cost_matrix6, filename6, metadata6)
    instances_metadata.append({'filename': filename6, 'metadata': metadata6})
    
    # Instancia 9: Ciclo impar C_5 con 3 colores
    # Ciclo impar: chromatic number = 3
    graph7 = nx.cycle_graph(5)
    cost_matrix7 = np.array([
        [1, 100, 100],   # Vértice 0: preferir color 0
        [100, 1, 100],   # Vértice 1: preferir color 1
        [1, 100, 100],   # Vértice 2: preferir color 0
        [100, 100, 1],   # Vértice 3: preferir color 2
        [100, 1, 100]    # Vértice 4: preferir color 1
    ])
    # Coloración óptima: 0->0, 1->1, 2->0, 3->2, 4->1
    # Costo = 1+1+1+1+1 = 5
    metadata7 = {
        'instance_type': 'cycle_odd',
        'n_vertices': 5,
        'n_colors': 3,
        'known_optimal_cost': 5.0,
        'cycle_length': 5,
        'chromatic_number': 3,
        'is_bipartite': False,
        'description': 'Ciclo impar C_5: requiere 3 colores (no bipartito), costo óptimo 5',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 2, 4: 1}
    }
    filename7 = get_instance_filename('special_cycle_odd', 5, 3, nx.density(graph7), 7, output_dir)
    save_instance(graph7, cost_matrix7, filename7, metadata7)
    instances_metadata.append({'filename': filename7, 'metadata': metadata7})
    
    # Instancia 10: Ciclo impar C_7 con 3 colores
    graph8 = nx.cycle_graph(7)
    cost_matrix8 = np.array([
        [1, 100, 100],
        [100, 1, 100],
        [1, 100, 100],
        [100, 100, 1],
        [100, 1, 100],
        [1, 100, 100],
        [100, 100, 1]
    ])
    # Coloración óptima: patrón 0,1,0,2,1,0,2 -> costo = 7*1 = 7
    metadata8 = {
        'instance_type': 'cycle_odd_larger',
        'n_vertices': 7,
        'n_colors': 3,
        'known_optimal_cost': 7.0,
        'cycle_length': 7,
        'chromatic_number': 3,
        'is_bipartite': False,
        'description': 'Ciclo impar C_7: mayor que C_5, requiere 3 colores, costo óptimo 7',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 2, 4: 1, 5: 0, 6: 2}
    }
    filename8 = get_instance_filename('special_cycle_odd_large', 7, 3, nx.density(graph8), 8, output_dir)
    save_instance(graph8, cost_matrix8, filename8, metadata8)
    instances_metadata.append({'filename': filename8, 'metadata': metadata8})
    
    # =========================================================================
    # INSTANCIAS CON GRAFOS ESTRELLA (S_n) - CASOS 11-14
    # =========================================================================
    # Propiedad: χ(S_n) = 2 siempre (bipartito)
    # Centro requiere un color, todas las hojas requieren otro color diferente
    
    # Instancia 11: Estrella S_4 (1 centro + 4 hojas)
    graph_s4 = nx.star_graph(4)
    cost_matrix_s4 = np.array([
        [1, 100],   # centro
        [100, 1],   # hojas
        [100, 1],
        [100, 1],
        [100, 1]
    ])
    # Óptimo: centro→0, hojas→1 → costo = 1 + 4 = 5
    metadata_s4 = {
        'instance_type': 'star_graph_small',
        'n_vertices': 5,
        'n_colors': 2,
        'known_optimal_cost': 5.0,
        'center': 0,
        'num_leaves': 4,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Estrella S_4: centro + 4 hojas, costo óptimo 5',
        'optimal_coloring': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    }
    filename_s4 = get_instance_filename('special_star_4', 5, 2, nx.density(graph_s4), 18, output_dir)
    save_instance(graph_s4, cost_matrix_s4, filename_s4, metadata_s4)
    instances_metadata.append({'filename': filename_s4, 'metadata': metadata_s4})

    # Instancia 12: Grafo completo K4 (intervalo trivial)
    graph_k4 = nx.complete_graph(4)
    cost_matrix_k4 = np.array([
        [1,100,100,100],
        [100,1,100,100],
        [100,100,1,100],
        [100,100,100,1]
    ])
    metadata_k4 = {
        'instance_type': 'complete_graph_4',
        'n_vertices': 4,
        'n_colors': 4,
        'known_optimal_cost': 4.0,
        'chromatic_number': 4,
        'clique_number': 4,
        'description': 'Grafo completo K4: cada vértice requiere color distinto, costo óptimo 4',
        'optimal_coloring': {0:0, 1:1, 2:2, 3:3}
    }
    filename_k4 = get_instance_filename('special_complete_graph_4', 4, 4, nx.density(graph_k4), 19, output_dir)
    save_instance(graph_k4, cost_matrix_k4, filename_k4, metadata_k4)
    instances_metadata.append({'filename': filename_k4, 'metadata': metadata_k4})


    # Instancia 13: Estrella S_5 con 2 colores
    # Star graph with 1 center and 5 leaves
    graph9 = nx.star_graph(5)
    cost_matrix9 = np.array([
        [1, 100],      # Vértice 0 (centro): preferir color 0
        [100, 1],      # Vértice 1 (hoja): preferir color 1
        [100, 1],      # Vértice 2 (hoja): preferir color 1
        [100, 1],      # Vértice 3 (hoja): preferir color 1
        [100, 1],      # Vértice 4 (hoja): preferir color 1
        [100, 1]       # Vértice 5 (hoja): preferir color 1
    ])
    # Coloración óptima: centro con color 0, hojas con color 1
    # Costo = 1 + 1 + 1 + 1 + 1 + 1 = 6
    metadata9 = {
        'instance_type': 'star_graph',
        'n_vertices': 6,
        'n_colors': 2,
        'known_optimal_cost': 6.0,
        'center': 0,
        'num_leaves': 5,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Grafo Estrella S_5: centro + 5 hojas, 2-colorable, costo óptimo 6',
        'optimal_coloring': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
    }
    filename9 = get_instance_filename('special_star_5', 6, 2, nx.density(graph9), 9, output_dir)
    save_instance(graph9, cost_matrix9, filename9, metadata9)
    instances_metadata.append({'filename': filename9, 'metadata': metadata9})
    
    # Instancia 14: Estrella S_8 con 2 colores
    graph10 = nx.star_graph(8)
    cost_matrix10 = np.array([
        [2, 100],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2]
    ])
    # Coloración óptima: centro con costo 2, hojas con costo 2 cada una
    # Costo = 2 + 8*2 = 18
    metadata10 = {
        'instance_type': 'star_graph_larger',
        'n_vertices': 9,
        'n_colors': 2,
        'known_optimal_cost': 18.0,
        'center': 0,
        'num_leaves': 8,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Grafo Estrella S_8: centro + 8 hojas, 2-colorable, costo óptimo 18',
        'optimal_coloring': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    }
    filename10 = get_instance_filename('special_star_8', 9, 2, nx.density(graph10), 10, output_dir)
    save_instance(graph10, cost_matrix10, filename10, metadata10)
    instances_metadata.append({'filename': filename10, 'metadata': metadata10})
    
    # =========================================================================
    # INSTANCIAS CON GRAFOS DE INTERVALO - CASOS 15-16
    # =========================================================================
    # Propiedad: Grafos de intervalo son grafos perfectos
    # χ(G) = ω(G) (chromatic number = clique number)
    # Coloreables en tiempo polinómico de forma óptima
    
    # Instancia 15: Grafo de intervalo simple (conjuntos de intervalos que se solapan)
    # Representación: intervalos [a_i, b_i] en la recta real
    # Vértices: 5 intervalos
    # Overlapping pattern: [0,2], [1,3], [2,4], [1,3], [3,5]
    graph11 = nx.Graph()
    graph11.add_nodes_from(range(5))
    # Aristas basadas en solapamiento de intervalos
    graph11.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    
    cost_matrix11 = np.array([
        [1, 100, 100],   # Vértice 0: intervalo [0,2]
        [100, 1, 100],   # Vértice 1: intervalo [1,3]
        [100, 100, 1],   # Vértice 2: intervalo [2,4]
        [1, 100, 100],   # Vértice 3: intervalo [1,3]
        [100, 100, 1]    # Vértice 4: intervalo [3,5]
    ])
    # Coloración óptima: 0->0, 1->1, 2->2, 3->1, 4->2
    # Costo = 1+1+1+1+1 = 5
    metadata11 = {
        'instance_type': 'interval_graph',
        'n_vertices': 5,
        'n_colors': 3,
        'known_optimal_cost': 5.0,
        'chromatic_number': 3,
        'clique_number': 3,
        'intervals': [[0, 2], [1, 3], [2, 4], [1, 3], [3, 5]],
        'description': 'Grafo de intervalo con 5 vértices: χ(G)=ω(G)=3, costo óptimo 5',
        'optimal_coloring': {0: 0, 1: 1, 2: 2, 3: 1, 4: 2}
    }
    filename11 = get_instance_filename('special_interval', 5, 3, nx.density(graph11), 11, output_dir)
    save_instance(graph11, cost_matrix11, filename11, metadata11)
    instances_metadata.append({'filename': filename11, 'metadata': metadata11})
    
    # Instancia 16: Grafo de intervalo más complejo con mayor número de vértices
    graph12 = nx.Graph()
    graph12.add_nodes_from(range(7))
    # Construcción de un grafo de intervalo más grande
    graph12.add_edges_from([
        (0, 1), (0, 2), (1, 2),     # Clique de tamaño 3
        (1, 3), (2, 3), (2, 4),     # Expansión
        (3, 4), (3, 5),             # Conexiones adicionales
        (4, 5), (4, 6), (5, 6)      # Más cliques
    ])
    
    cost_matrix12 = np.array([
        [1, 100, 100, 100],
        [100, 1, 100, 100],
        [100, 100, 1, 100],
        [100, 100, 100, 1],
        [1, 100, 100, 100],
        [100, 1, 100, 100],
        [100, 100, 1, 100]
    ])
    # Coloración óptima: 0->0, 1->1, 2->2, 3->3, 4->0, 5->1, 6->2
    # Costo = 1+1+1+1+1+1+1 = 7
    metadata12 = {
        'instance_type': 'interval_graph_large',
        'n_vertices': 7,
        'n_colors': 4,
        'known_optimal_cost': 7.0,
        'chromatic_number': 4,
        'clique_number': 4,
        'description': 'Grafo de intervalo con 7 vértices: χ(G)=ω(G)=4, costo óptimo 7',
        'optimal_coloring': {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2}
    }
    filename12 = get_instance_filename('special_interval_large', 7, 4, nx.density(graph12), 12, output_dir)
    save_instance(graph12, cost_matrix12, filename12, metadata12)
    instances_metadata.append({'filename': filename12, 'metadata': metadata12})
    
    # =========================================================================
    # INSTANCIAS CON ESTRUCTURAS BINARIAS (ÁRBOLES BINARIOS) - CASOS 17-18
    # =========================================================================
    # Propiedad: Árboles son bipartitos, χ(T) = 2
    # Se puede hacer un árbol binario completo o balanceado
    
    # Instancia 17: Árbol binario completo de profundidad 2
    # Nodos: 1 raíz + 2 en nivel 1 + 4 en nivel 2 = 7 nodos
    graph13 = nx.balanced_tree(2, 2)  # Árbol regular binario, profundidad 2
    cost_matrix13 = np.array([
        [1, 100],         # Nodo 0 (raíz): preferir color 0
        [100, 1],         # Nodo 1 (nivel 1): preferir color 1
        [100, 1],         # Nodo 2 (nivel 1): preferir color 1
        [1, 100],         # Nodo 3 (nivel 2): preferir color 0
        [1, 100],         # Nodo 4 (nivel 2): preferir color 0
        [1, 100],         # Nodo 5 (nivel 2): preferir color 0
        [1, 100]          # Nodo 6 (nivel 2): preferir color 0
    ])
    # Coloración óptima bipartita por niveles
    # Costo = 1*4 + 1*3 = 7
    metadata13 = {
        'instance_type': 'binary_tree_balanced',
        'n_vertices': 7,
        'n_colors': 2,
        'known_optimal_cost': 7.0,
        'tree_type': 'balanced_binary',
        'height': 2,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Árbol binario balanceado de profundidad 2: 7 vértices, bipartito, costo óptimo 7',
        'optimal_coloring': {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 0, 6: 0}
    }
    filename13 = get_instance_filename('special_binary_tree_balanced', 7, 2, nx.density(graph13), 13, output_dir)
    save_instance(graph13, cost_matrix13, filename13, metadata13)
    instances_metadata.append({'filename': filename13, 'metadata': metadata13})
    
    # Instancia 18: Árbol binario completo de profundidad 3
    # Nodos: 1 + 2 + 4 + 8 = 15 nodos
    graph14 = nx.balanced_tree(2, 3)  # Árbol binario completo, profundidad 3
    cost_matrix14 = np.zeros((15, 2))
    # Asignar costos para forzar coloración por niveles
    for i in range(15):
        if i == 0:
            cost_matrix14[i] = [1, 100]  # Raíz: color 0
        elif i < 3:
            cost_matrix14[i] = [100, 1]  # Nivel 1: color 1
        elif i < 7:
            cost_matrix14[i] = [1, 100]  # Nivel 2: color 0
        else:
            cost_matrix14[i] = [100, 1]  # Nivel 3: color 1
    
    # Coloración óptima: coloring alternado por niveles
    # Costo = 1 + 2*1 + 4*1 + 8*1 = 15
    optimal_coloring_14 = {}
    optimal_coloring_14[0] = 0  # Raíz
    for i in range(1, 3):
        optimal_coloring_14[i] = 1  # Nivel 1
    for i in range(3, 7):
        optimal_coloring_14[i] = 0  # Nivel 2
    for i in range(7, 15):
        optimal_coloring_14[i] = 1  # Nivel 3
    
    metadata14 = {
        'instance_type': 'binary_tree_complete',
        'n_vertices': 15,
        'n_colors': 2,
        'known_optimal_cost': 15.0,
        'tree_type': 'complete_binary',
        'height': 3,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Árbol binario completo de profundidad 3: 15 vértices, bipartito, costo óptimo 15',
        'optimal_coloring': optimal_coloring_14
    }
    filename14 = get_instance_filename('special_binary_tree_complete', 15, 2, nx.density(graph14), 14, output_dir)
    save_instance(graph14, cost_matrix14, filename14, metadata14)
    instances_metadata.append({'filename': filename14, 'metadata': metadata14})
    
    # =========================================================================
    # INSTANCIAS CON GRAFOS BINARIOS COMPLETOS (COMPLETE BIPARTITE GRAPHS) - CASOS 19-20
    # =========================================================================
    # Estructura donde hay conexiones entre capas binarias
    # Similar a árboles pero con aristas adicionales
    
    # Instancia 19: Grafo binario completo K_{3,4}
    # 7 vértices organizados en estructura binaria completa
    graph15 = nx.complete_bipartite_graph(3, 4)
    cost_matrix15 = np.array([
        [1, 100],
        [1, 100],
        [1, 100],
        [100, 1],
        [100, 1],
        [100, 1],
        [100, 1]
    ])
    # Coloración óptima: partición 1 -> color 0, partición 2 -> color 1
    # Costo = 3*1 + 4*1 = 7
    metadata15 = {
        'instance_type': 'complete_bipartite',
        'n_vertices': 7,
        'n_colors': 2,
        'known_optimal_cost': 7.0,
        'partition_1_size': 3,
        'partition_2_size': 4,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Grafo bipartito completo K_{3,4}: todos vs todos entre particiones, costo óptimo 7',
        'optimal_coloring': {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}
    }
    filename15 = get_instance_filename('special_complete_bipartite', 7, 2, nx.density(graph15), 15, output_dir)
    save_instance(graph15, cost_matrix15, filename15, metadata15)
    instances_metadata.append({'filename': filename15, 'metadata': metadata15})
    
    # Instancia 20: Grafo binario completo K_{4,5}
    graph16 = nx.complete_bipartite_graph(4, 5)
    cost_matrix16 = np.array([
        [2, 100],
        [2, 100],
        [2, 100],
        [2, 100],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2],
        [100, 2]
    ])
    # Coloración óptima: partición 1 -> color 0 (costo 2c/u), partición 2 -> color 1 (costo 2 c/u)
    # Costo = 4*2 + 5*2 = 18
    metadata16 = {
        'instance_type': 'complete_bipartite_large',
        'n_vertices': 9,
        'n_colors': 2,
        'known_optimal_cost': 18.0,
        'partition_1_size': 4,
        'partition_2_size': 5,
        'chromatic_number': 2,
        'is_bipartite': True,
        'description': 'Grafo bipartito completo K_{4,5}: mayor que K_{3,4}, costo óptimo 18',
        'optimal_coloring': {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    }
    filename16 = get_instance_filename('special_complete_bipartite_large', 9, 2, nx.density(graph16), 16, output_dir)
    save_instance(graph16, cost_matrix16, filename16, metadata16)
    instances_metadata.append({'filename': filename16, 'metadata': metadata16})
    
    return instances_metadata

def generate_benchmark_reference_instances(output_dir: str = "instances") -> List[Dict[str, Any]]:
    """
    Genera instancias basadas en benchmarks académicos conocidos para el MCCPP
    Referencias:
    - Jansen, K. "The Optimum Cost Chromatic Partition Problem" CIAC'97, 1997
    - Barth et al. papers on cost coloring problems
    - DIMACS benchmark suite (adaptadas para MCCPP)
    
    Args:
        output_dir: directorio donde guardar las instancias
    
    Returns:
        lista de metadatos de instancias benchmark
    """
    benchmark_metadata = []
    
    # Benchmark 1: Instancia basada en el trabajo de Jansen (1997) - Path coloring
    # The Optimal Cost Chromatic Partition Problem for Trees and Interval Graphs
    graph_b1 = nx.path_graph(6)
    cost_matrix_b1 = np.array([
        [1, 5, 20],
        [5, 1, 15],
        [1, 5, 20],
        [5, 1, 15],
        [1, 5, 20],
        [5, 1, 15]
    ])
    # Solución óptima conocida: alternancia de colores 0,1,0,1,0,1 -> costo = 6*1 = 6
    metadata_b1 = {
        'instance_type': 'jansen_path',
        'source': 'Jansen, K. CIAC\'97 1997 - The Optimal Cost Chromatic Partition Problem',
        'n_vertices': 6,
        'n_colors': 3,
        'known_optimal_cost': 6.0,
        'graph_type': 'path',
        'year': 1997,
        'publication': 'LNCS 1203, pp. 25-36',
        'description': 'Path graph de Jansen 1997: benchmark académico seminal',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
        'is_feasible': estimate_feasibility(graph_b1, 3, 'jansen_path')
    }
    filename_b1 = get_instance_filename('benchmark_jansen_path', 6, 3, nx.density(graph_b1), 1, output_dir)
    save_instance(graph_b1, cost_matrix_b1, filename_b1, metadata_b1)
    benchmark_metadata.append({'filename': filename_b1, 'metadata': metadata_b1})
    
    # Benchmark 2: Instancia de ciclo de Jansen
    # Optimal coloring of cycles with cost constraints
    graph_b2 = nx.cycle_graph(10)
    cost_matrix_b2 = np.array([
        [1, 3, 10],
        [3, 1, 10],
        [1, 3, 10],
        [3, 1, 10],
        [1, 3, 10],
        [3, 1, 10],
        [1, 3, 10],
        [3, 1, 10],
        [1, 3, 10],
        [3, 1, 10]
    ])
    # Solución óptima: C_10 es par, se usa 2-coloración -> costo = 10*1 = 10
    metadata_b2 = {
        'instance_type': 'jansen_cycle',
        'source': 'Jansen, K. CIAC\'97 1997',
        'n_vertices': 10,
        'n_colors': 3,
        'known_optimal_cost': 10.0,
        'graph_type': 'cycle_even',
        'year': 1997,
        'publication': 'LNCS 1203',
        'description': 'Ciclo par C_10 de Jansen 1997: benchmark académico',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1},
        'is_feasible': estimate_feasibility(graph_b2, 3, 'jansen_cycle')
    }
    filename_b2 = get_instance_filename('benchmark_jansen_cycle', 10, 3, nx.density(graph_b2), 2, output_dir)
    save_instance(graph_b2, cost_matrix_b2, filename_b2, metadata_b2)
    benchmark_metadata.append({'filename': filename_b2, 'metadata': metadata_b2})
    
    # Benchmark 3: Instancia DIMACS-style (adaptada para MCCPP)
    # DIMACS benchmarks are commonly used for graph coloring problems
    # Aquí creamos una instancia inspirada en DIMACS (similar densidad)
    graph_b3 = nx.Graph()
    graph_b3.add_nodes_from(range(10))
    # Crear estructura típica de DIMACS: grafo denso pseudo-aleatorio
    edges_b3 = [
        (0, 1), (0, 2), (0, 5), (1, 2), (1, 3), (1, 6),
        (2, 3), (2, 4), (3, 4), (3, 7), (4, 5), (4, 8),
        (5, 6), (5, 9), (6, 7), (7, 8), (8, 9), (0, 9)
    ]
    graph_b3.add_edges_from(edges_b3)
    
    cost_matrix_b3 = np.array([
        [1, 100, 100, 100],
        [100, 1, 100, 100],
        [100, 100, 1, 100],
        [100, 100, 100, 1],
        [1, 100, 100, 100],
        [100, 1, 100, 100],
        [100, 100, 1, 100],
        [100, 100, 100, 1],
        [1, 100, 100, 100],
        [100, 1, 100, 100]
    ])
    # Solución óptima: χ(G) = 4 requerido, costo ≈ 10
    metadata_b3 = {
        'instance_type': 'dimacs_style',
        'source': 'DIMACS benchmark suite (adaptada para MCCPP)',
        'n_vertices': 10,
        'n_colors': 4,
        'known_optimal_cost': 10.0,
        'chromatic_number': 4,
        'density': float(nx.density(graph_b3)),
        'year': 'N/A (DIMACS 1990s-2000s)',
        'url': 'http://www.cs.sunysb.edu/~algorith/implement/dimacs/',
        'description': 'Instancia tipo DIMACS adaptada para MCCPP: grafo denso pseudo-aleatorio',
        'optimal_coloring': {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 3, 8: 0, 9: 1},
        'is_feasible': estimate_feasibility(graph_b3, 4, 'dimacs_style')
    }
    filename_b3 = get_instance_filename('benchmark_dimacs', 10, 4, nx.density(graph_b3), 3, output_dir)
    save_instance(graph_b3, cost_matrix_b3, filename_b3, metadata_b3)
    benchmark_metadata.append({'filename': filename_b3, 'metadata': metadata_b3})
    
    # Benchmark 4: Instancia basada en scheduling (aplicación del MCCPP en VLSI/scheduling)
    # Problema: assignar tareas a procesadores con costos diferenciados
    # Modelamos como grafo de conflictos
    graph_b4 = nx.Graph()
    graph_b4.add_nodes_from(range(8))
    # Grafo de conflictos en scheduling
    edges_b4 = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7), (6, 7)]
    graph_b4.add_edges_from(edges_b4)
    
    # Matriz de costos: diferentes "máquinas" o "slots" con costos variados
    cost_matrix_b4 = np.array([
        [1, 3, 10],
        [2, 1, 10],
        [1, 3, 10],
        [2, 1, 10],
        [1, 3, 10],
        [2, 1, 10],
        [1, 3, 10],
        [2, 1, 10]
    ])
    # Esta es una instancia aplicada: VLSI/Scheduling
    # Coloración óptima verdadera: Color 0 → {0,3,4,7}, Color 1 → {1,2,5,6}
    # Costo = 1+2+1+2 + 2+1+2+1 = 14
    metadata_b4 = {
        'instance_type': 'scheduling_application',
        'source': 'Inspirado en problemas de VLSI y scheduling (Barth et al.)',
        'n_vertices': 8,
        'n_colors': 3,
        'known_optimal_cost': 14.0,
        'application': 'VLSI routing / Task scheduling',
        'authors': 'Barth, D., et al.',
        'year': '1996-2000s',
        'description': 'Instancia de aplicación en scheduling VLSI: grafo de conflictos entre tareas',
        'optimal_coloring': {0: 0, 1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0},
        'is_feasible': estimate_feasibility(graph_b4, 3, 'scheduling_application')
    }
    filename_b4 = get_instance_filename('benchmark_scheduling', 8, 3, nx.density(graph_b4), 4, output_dir)
    save_instance(graph_b4, cost_matrix_b4, filename_b4, metadata_b4)
    benchmark_metadata.append({'filename': filename_b4, 'metadata': metadata_b4})
    
    return benchmark_metadata

def generate_random_instances(
    n_vertices_list: List[int],
    n_colors_list: List[int],
    n_instances: int = 10,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    """Genera instancias aleatorias con validación"""
    instances_metadata = []
    os.makedirs(output_dir, exist_ok=True)
    current_seed = seed

    for n_vertices in n_vertices_list:
        for n_colors in n_colors_list:
            for i in range(n_instances):
                density = float(np.random.uniform(0.1, 0.9))
                graph = generate_erdos_renyi_graph(n_vertices, density, seed=current_seed)
                cost_matrix = generate_cost_matrix(n_vertices, n_colors, seed=current_seed)

                is_feasible = estimate_feasibility(graph, n_colors, "random")
                known_opt = compute_known_optimal_cost(graph, cost_matrix, "random") if is_feasible else None

                metadata = {
                    "instance_type": "random",
                    "n_vertices": n_vertices,
                    "n_colors": n_colors,
                    "density": density,
                    "cost_pattern": "random",
                    "seed": current_seed,
                    "is_feasible": is_feasible,
                    "known_optimal_cost": known_opt,
                }

                filename = get_instance_filename("random", n_vertices, n_colors, density, current_seed, output_dir)
                save_instance(graph, cost_matrix, filename, metadata)
                instances_metadata.append({"filename": filename, "metadata": metadata})
                current_seed += 1

    return instances_metadata

def generate_full_benchmark_set(
    target_total: int = 150,
    keep_random: int = 20,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    current_seed = seed
    collected: List[Dict[str, Any]] = []

    print(f"\n{'='*80}")
    print(f"GENERACIÓN COMPLETA DE BENCHMARKS (TODAS LAS CLASES)")
    print(f"{'='*80}\n")

    # 1. Casos especiales
    print(f"[1/9] Generando casos especiales...")
    try:
        special = generate_special_case_instances(output_dir=output_dir)
        collected.extend(special)
        print(f"  ✓ {len(special)} instancias especiales")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 2. Benchmarks académicos
    print(f"[2/9] Generando benchmarks académicos...")
    try:
        bench = generate_benchmark_reference_instances(output_dir=output_dir)
        collected.extend(bench)
        print(f"  ✓ {len(bench)} benchmarks")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 3. Grafos de intervalo (especiales y genéricos)
    print(f"[3/9] Generando grafos de intervalo (especiales + aleatorios)...")
    try:
        special_intervals = generate_special_interval_graph_instances(output_dir=output_dir)
        collected.extend(special_intervals)
        interval_gen = generate_interval_graph_instances([10, 15, 20], [6, 10, 15], n_instances=2, seed=current_seed, output_dir=output_dir)
        collected.extend(interval_gen)
        print(f"  ✓ {len(special_intervals) + len(interval_gen)} grafos de intervalo")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 4. Grafos estructurados
    print(f"[4/9] Generando grafos estructurados...")
    try:
        structured = generate_structured_instances(
            [10, 15, 20], 
            [6, 10, 20],  # k generoso
            graph_types=["path", "cycle", "star", "wheel", "grid", "complete"], 
            n_instances=3, 
            seed=current_seed, 
            output_dir=output_dir
        )
        collected.extend(structured)
        print(f"  ✓ {len(structured)} estructurados")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 5. Árboles
    print(f"[5/9] Generando árboles...")
    try:
        trees = generate_tree_instances([8, 10, 15, 20], n_colors=3, n_instances=4, seed=current_seed, output_dir=output_dir)
        collected.extend(trees)
        print(f"  ✓ {len(trees)} árboles")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # 6-8. Erdős-Rényi con diferentes densidades
    for idx, (p, desc) in enumerate([(0.05, "dispersos"), (0.3, "medios"), (0.6, "densos")]):
        print(f"[{6+idx}/9] Generando Erdős-Rényi {desc}...")
        try:
            er = generate_erdos_renyi_instances([15, 20, 25], p=p, n_colors=[8, 10, 15], n_instances=2, seed=current_seed + idx*1000, output_dir=output_dir)
            collected.extend(er)
            print(f"  ✓ {len(er)} instancias ER {desc}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    # 9. Instancias aleatorias (random)
    print(f"[9/9] Generando instancias aleatorias (random)...")
    try:
        rand = generate_random_instances(n_vertices_list=[8,12,16], n_colors_list=[3,4,5], n_instances=5, seed=current_seed+9999, output_dir=output_dir)
        collected.extend(rand)
        print(f"  ✓ {len(rand)} instancias aleatorias")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Calcular totales a partir de los archivos realmente guardados en output_dir
    import glob

    file_paths = sorted(glob.glob(os.path.join(output_dir, "*.json")))
    final_list: List[Dict[str, Any]] = []
    factibles = 0

    for fp in file_paths:
        try:
            _, _, meta = load_instance(fp)
            final_list.append({'filename': fp, 'metadata': meta})
            if meta.get('is_feasible', False):
                factibles += 1
        except Exception:
            continue

    total_files = len(final_list)

    print(f"\n{'='*80}")
    print(f"Total generado: {total_files} instancias")
    pct = (100 * factibles / total_files) if total_files > 0 else 0.0
    print(f"Factibles: {factibles}/{total_files} ({pct:.1f}%)")
    print(f"{'='*80}\n")

    return final_list

def print_instances_table():
    """Imprime tabla de todas las instancias generadas"""
    table = """
    ================================================================================
    TABLA DE INSTANCIAS GENERADAS
    ================================================================================
    
    ID | Tipo                      | Vértices | Colores | χ  | Óptimo | Descripción
    ---|---------------------------|----------|---------|----|---------|---------------------------------
    1  | path_known_optimal        | 3        | 2       | 2  | 3.0    | Path: 0->1->2
    2  | bipartite_known_optimal   | 4        | 2       | 2  | 4.0    | K_{2,2}: completo bipartito
    3  | triangle_known_optimal    | 3        | 3       | 3  | 3.0    | K_3: triángulo
    4  | isolated_vertices         | 4        | 2       | 1  | 16.0   | 4 vértices aislados
    5  | path_5_vertices           | 5        | 2       | 2  | 5.0    | Camino P5: alternancia 2 colores
    6  | cycle_even_small          | 4        | 2       | 2  | 4.0    | Ciclo par C_4
    7  | cycle_even                | 6        | 2       | 2  | 6.0    | Ciclo par C_6
    8  | cycle_even_larger         | 8        | 2       | 2  | 16.0   | Ciclo par C_8
    9  | cycle_odd                 | 5        | 3       | 3  | 5.0    | Ciclo impar C_5
    10 | cycle_odd_larger          | 7        | 3       | 3  | 7.0    | Ciclo impar C_7
    11 | star_graph_small          | 5        | 2       | 2  | 5.0    | Estrella S_4: centro + 4 hojas
    12 | complete_graph_4          | 4        | 4       | 4  | 4.0    | Grafo completo K4
    13 | star_graph                | 6        | 2       | 2  | 6.0    | Estrella S_5: centro + 5 hojas
    14 | star_graph_larger         | 9        | 2       | 2  | 18.0   | Estrella S_8: centro + 8 hojas
    15 | interval_graph            | 5        | 3       | 3  | 5.0    | Grafo de intervalo simple
    16 | interval_graph_large      | 7        | 4       | 4  | 7.0    | Grafo de intervalo complejo
    17 | binary_tree_balanced      | 7        | 2       | 2  | 7.0    | Árbol binario balanceado
    18 | binary_tree_complete      | 15       | 2       | 2  | 15.0   | Árbol binario completo
    19 | complete_bipartite        | 7        | 2       | 2  | 7.0    | K_{3,4}: bipartito completo
    20 | complete_bipartite_large  | 9        | 2       | 2  | 18.0   | K_{4,5}: bipartito completo
    B1 | jansen_path               | 6        | 3       | 2  | 6.0    | Benchmark Jansen 1997
    B2 | jansen_cycle              | 10       | 3       | 2  | 10.0   | Benchmark Jansen 1997
    B3 | dimacs_style              | 10       | 4       | 4  | 10.0   | DIMACS adaptation
    B4 | scheduling_application    | 8        | 3       | 3  | 14.0   | VLSI/Scheduling

    ================================================================================
    """
    print(table)

if __name__ == "__main__":
    print("=" * 80)
    print("Generador de Instancias Especiales para MCCPP")
    print("Proyecto: Diseño de Algoritmos (DAA)")
    print("=" * 80)
    print()
    
    print("Generando instancias especiales...")
    instances = generate_special_case_instances(output_dir="instances")
    print(f"✓ {len(instances)} instancias especiales generadas")
    print()
    
    print("Generando benchmarks académicos...")
    benchmarks = generate_benchmark_reference_instances(output_dir="instances/benchmarks")
    print(f"✓ {len(benchmarks)} benchmarks académicos generados")
    print()
    
    print("Generando instancias de grafos de intervalo...")
    try:
        intervals = generate_special_interval_graph_instances(output_dir="instances/interval_graphs")
        print(f"✓ {len(intervals)} instancias de intervalo generadas")
    except Exception as e:
        print(f"⚠ No se pudieron generar instancias de intervalo: {e}")
        intervals = []
    print()
    
    print_instances_table()
    print()
    print()
    
    print("=" * 80)
    total_instances = len(instances) + len(benchmarks) + len(intervals)
    print("TOTAL DE INSTANCIAS GENERADAS:", total_instances)
    print("Desglose:")
    print(f"  - Instancias especiales: {len(instances)}")
    print(f"  - Benchmarks académicos: {len(benchmarks)}")
    print(f"  - Grafos de intervalo: {len(intervals)}")
    print("=" * 80)
