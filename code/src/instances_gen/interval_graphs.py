"""
Instancias de grafos de intervalos para el Proyecto DAA - MCCPP
"""
import networkx as nx
import numpy as np
from typing import List, Dict, Any
from src.utils.graph_utils import generate_interval_graph
from src.utils.cost_utils import generate_structured_cost_matrix
from src.utils.io_utils import save_instance, get_instance_filename

def generate_interval_graph_instances(n_vertices_list: List[int], n_colors: List[int],
                                      max_length: float = 1.0, cost_pattern: str = "uniform",
                                      n_instances: int = 5, seed: int = 42,
                                      output_dir: str = "instances") -> List[Dict[str, Any]]:
    """
    Genera múltiples instancias de grafos de intervalos para FISP-MDC en el Proyecto DAA - MCCPP
    
    Args:
        n_vertices_list: lista del número de intervalos
        n_colors: número de colores (máquinas)
        max_length: longitud máxima del intervalo
        cost_pattern: tipo de estructura de costos
        n_instances: número de instancias por tamaño
        seed: semilla aleatoria
        output_dir: directorio donde guardar las instancias
    
    Returns:
        lista de metadatos de instancias que incluyen los intervalos
    """
    instances_metadata = []
    current_seed = seed
    
    for inx, n_vertices in enumerate(n_vertices_list):
        for i in range(n_instances):
            # Generar grafo de intervalos y los intervalos
            graph, intervals = generate_interval_graph(n_vertices, max_length, seed=current_seed)
            
            # Generar matriz de costos
            cost_matrix = generate_structured_cost_matrix(n_vertices, n_colors[inx], cost_pattern, seed=current_seed)
            
            # Preparar metadatos
            metadata = {
                'instance_type': 'interval_graph',
                'n_vertices': n_vertices,
                'n_colors': n_colors[inx],
                'max_length': max_length,
                'cost_pattern': cost_pattern,
                'seed': current_seed,
                'density': nx.density(graph),
                'intervals': intervals
            }
            
            # Guardar instancia (los intervalos se almacenan en los metadatos)
            filename = get_instance_filename('interval', n_vertices, n_colors[inx], metadata['density'], current_seed, output_dir)
            save_instance(graph, cost_matrix, filename, metadata)
            
            instances_metadata.append({
                'filename': filename,
                'metadata': metadata
            })
            
            current_seed += 1
    
    return instances_metadata

def generate_special_interval_graph_instances(output_dir: str = "instances/interval_graphs") -> List[Dict[str, Any]]:
    """
    Genera instancias específicas de grafos de intervalo
    Propiedades explotadas:
    - Interval graphs son grafos perfectos (χ(G) = ω(G))
    - Coloración óptima en tiempo polinómico (greedy sobre interval ordering)
    - Útiles como casos de prueba para validar optimality
    
    Args:
        output_dir: directorio para instancias de grafos de intervalo
    
    Returns:
        lista de metadatos
    """
    interval_metadata = []
    
    # Instancia 1: Intervalo simple no-solapante
    # Intervalos: [0,1], [2,3], [4,5], [6,7]
    graph_i1 = nx.Graph()
    graph_i1.add_nodes_from([0, 1, 2, 3])
    # Sin aristas (intervalos disjuntos)
    cost_matrix_i1 = np.array([
        [1, 100],
        [100, 1],
        [1, 100],
        [100, 1]
    ])
    # Óptimo: cada vértice puede usar su color mínimo -> costo = 4*1 = 4
    metadata_i1 = {
        'instance_type': 'interval_non_overlapping',
        'n_vertices': 4,
        'n_colors': 2,
        'known_optimal_cost': 4.0,
        'chromatic_number': 1,
        'clique_number': 1,
        'intervals': [[0, 1], [2, 3], [4, 5], [6, 7]],
        'description': 'Intervalos no solapantes: χ(G)=1, caso trivial para interval graphs',
        'optimal_coloring': {0: 0, 1: 0, 2: 0, 3: 0}
    }
    filename_i1 = get_instance_filename('interval_non_overlapping', 4, 2, 0.0, 1, output_dir)
    save_instance(graph_i1, cost_matrix_i1, filename_i1, metadata_i1)
    interval_metadata.append({'filename': filename_i1, 'metadata': metadata_i1})
    
    # Instancia 2: Cadena de intervalos solapantes
    # Patrón: cada intervalo se solapa con el siguiente
    # Intervalos: [0,2], [1,3], [2,4], [3,5], [4,6]
    graph_i2 = nx.path_graph(5)
    cost_matrix_i2 = np.array([
        [1, 100, 100],
        [100, 1, 100],
        [100, 100, 1],
        [1, 100, 100],
        [100, 1, 100]
    ])
    # Óptimo: coloración en 2 colores (alternancia) -> costo = 5*1 = 5
    metadata_i2 = {
        'instance_type': 'interval_chain_overlapping',
        'n_vertices': 5,
        'n_colors': 3,
        'known_optimal_cost': 5.0,
        'chromatic_number': 2,
        'clique_number': 2,
        'intervals': [[0, 2], [1, 3], [2, 4], [3, 5], [4, 6]],
        'description': 'Cadena de intervalos solapantes: χ(G)=2, 5 vértices',
        'optimal_coloring': {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    }
    filename_i2 = get_instance_filename('interval_chain', 5, 3, nx.density(graph_i2), 2, output_dir)
    save_instance(graph_i2, cost_matrix_i2, filename_i2, metadata_i2)
    interval_metadata.append({'filename': filename_i2, 'metadata': metadata_i2})
    
    # Instancia 3: Grafo de intervalo con punto de máxima clique
    # Simulamos: [0,3], [1,3], [2,3] (clique de tamaño 3) + [2,4], [3,4]
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
    # Óptimo: clique tamaño 3 requiere 3 colores, vértice 4 reutiliza color 0
    # Costo = 1 + 1 + 1 + 1 + 1 = 5
    metadata_i3 = {
        'instance_type': 'interval_with_max_clique',
        'n_vertices': 5,
        'n_colors': 4,
        'known_optimal_cost': 5.0,
        'chromatic_number': 3,
        'clique_number': 3,
        'max_clique_size': 3,
        'description': 'Intervalo con clique máxima tamaño 3: 5 vértices',
        'optimal_coloring': {0: 0, 1: 1, 2: 2, 3: 0, 4: 1}
    }
    filename_i3 = get_instance_filename('interval_max_clique', 5, 4, nx.density(graph_i3), 3, output_dir)
    save_instance(graph_i3, cost_matrix_i3, filename_i3, metadata_i3)
    interval_metadata.append({'filename': filename_i3, 'metadata': metadata_i3})
    
    return interval_metadata
