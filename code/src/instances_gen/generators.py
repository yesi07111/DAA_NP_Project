"""
Generadores de instancias para el Proyecto DAA - MCCPP
"""

import networkx as nx
from networkx.generators.trees import random_unlabeled_tree
import numpy as np
from typing import List, Dict, Any
from src.utils.graph_utils import generate_erdos_renyi_graph, generate_interval_graph
from src.utils.cost_utils import generate_cost_matrix, generate_structured_cost_matrix
from src.utils.io_utils import save_instance, get_instance_filename


def generate_erdos_renyi_instances(
    n_vertices_list: List[int],
    p: float,
    n_colors: List[int],
    cost_pattern: str = "uniform",
    n_instances: int = 5,
    seed: int = 42,
    output_dir: str = "instances",
) -> List[Dict[str, Any]]:
    """
    Genera múltiples instancias Erdős-Rényi para el Proyecto DAA - MCCPP.

    Instancias Erdős-Rényi (Grafos ER):
        Las instancias Erdős-Rényi corresponden a grafos generados bajo el modelo
        clásico de grafos aleatorios G(n, p). En este modelo:
            - n es el número de vértices.
            - p es la probabilidad de que un par de vértices forme una arista.
            - Cada arista potencial se incluye de manera independiente con probabilidad p.

    Propiedades clave de los grafos Erdős-Rényi:
        - Independencia de aristas:
            Cada arista se genera de forma independiente del resto, lo que convierte
            a este modelo en un estándar simple y ampliamente usado para experimentos
            con grafos aleatorios.

        - Densidad esperada:
            El número esperado de aristas es:
                E[|E|] = p * n*(n-1)/2
            Valores altos de p generan grafos densos; valores bajos generan grafos
            dispersos o esparsos.

        - Distribución de grados:
            En G(n, p), el grado de cada vértice sigue una distribución binomial:
                deg(v) ~ Bin(n-1, p)
            que se aproxima a una distribución de Poisson cuando n es grande y p es pequeño.

        - Umbral de conectividad:
            Los grafos ER presentan transiciones de fase. El umbral aproximado para
            que el grafo sea conexo es:
                p = (log n) / n

        - Ausencia de estructura:
            Los grafos ER no imponen estructura geométrica, jerárquica ni comunitaria.
            Son casos de prueba neutrales, adecuados como benchmark sin sesgos.

    Uso en MCCPP:
        Las instancias ER permiten evaluar algoritmos de manera controlada y reproducible.
        Facilitan pruebas sistemáticas variando tamaños y densidades, evitando depender
        de estructuras específicas de grafos reales.

    Args:
        n_vertices_list: lista de cantidades de vértices a generar.
        p: probabilidad de creación de aristas.
        n_colors: número de colores disponibles en la partición cromática.
        cost_pattern: tipo de estructura de costos aplicada.
        n_instances: número de instancias independientes por cada tamaño.
        seed: semilla aleatoria para reproducibilidad.
        output_dir: directorio donde se guardarán las instancias.

    Returns:
        lista con la metadata de las instancias generadas.
    """
    instances_metadata = []
    current_seed = seed

    for inx, n_vertices in enumerate(n_vertices_list):
        for i in range(n_instances):
            # Generar grafo
            graph = generate_erdos_renyi_graph(n_vertices, p, seed=current_seed)

            # Generar matriz de costos
            cost_matrix = generate_structured_cost_matrix(
                n_vertices, n_colors[inx], cost_pattern, seed=current_seed
            )

            # Preparar metadatos
            metadata = {
                "instance_type": "erdos_renyi",
                "n_vertices": n_vertices,
                "p": p,
                "n_colors": n_colors[inx],
                "cost_pattern": cost_pattern,
                "seed": current_seed,
                "density": nx.density(graph),
            }

            # Guardar instancia
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
    """
    Genera instancias con grafos estructurados para el Proyecto DAA - MCCPP

    Args:
        n_vertices_list: lista de cantidades de vértices
        n_colors: número de colores
        graph_types: tipos de grafos estructurados a generar
        cost_pattern: tipo de estructura de costos
        n_instances: número de instancias por tamaño y tipo
        seed: semilla aleatoria
        output_dir: directorio donde se guardarán las instancias

    Returns:
        lista con la metadata de las instancias
    """
    instances_metadata = []
    current_seed = seed

    for graph_type in graph_types:
        for inx, n_vertices in enumerate(n_vertices_list):
            for i in range(n_instances):
                # Generar grafo estructurado
                if graph_type == "path":
                    graph = nx.path_graph(n_vertices)
                elif graph_type == "cycle":
                    graph = nx.cycle_graph(n_vertices)
                elif graph_type == "complete":
                    graph = nx.complete_graph(n_vertices)
                elif graph_type == "star":
                    graph = nx.star_graph(
                        n_vertices - 1
                    )  # star_graph toma centro + n_hojas
                elif graph_type == "wheel":
                    graph = nx.wheel_graph(n_vertices)
                elif graph_type == "grid":
                    # Aproximar a rejilla: encontrar factores cercanos a sqrt(n_vertices)
                    factors = []
                    for j in range(1, int(np.sqrt(n_vertices)) + 1):
                        if n_vertices % j == 0:
                            factors.append((j, n_vertices // j))
                    if factors:
                        rows, cols = factors[-1]  # usar el factor más grande
                        graph = nx.grid_2d_graph(rows, cols)
                        # Renombrar nodos a 0..n_vertices-1
                        mapping = {node: i for i, node in enumerate(graph.nodes())}
                        graph = nx.relabel_nodes(graph, mapping)
                    else:
                        # Si no hay factores, usar un grafo camino
                        graph = nx.path_graph(n_vertices)
                else:
                    continue  # Saltar tipos de grafos desconocidos

                # Asegurar que el grafo tiene exactamente n_vertices (algunos generadores pueden crear tamaños diferentes)
                if graph.number_of_nodes() != n_vertices:
                    # Si el grafo es más pequeño, añadir vértices aislados
                    if graph.number_of_nodes() < n_vertices:
                        graph.add_nodes_from(range(graph.number_of_nodes(), n_vertices))
                    # Si es más grande, tomar subgrafo (no debería pasar con los anteriores)
                    else:
                        graph = graph.subgraph(range(n_vertices)).copy()

                # Generar matriz de costos
                cost_matrix = generate_structured_cost_matrix(
                    n_vertices, n_colors[inx], cost_pattern, seed=current_seed
                )

                # Preparar metadatos
                metadata = {
                    "instance_type": graph_type,
                    "n_vertices": n_vertices,
                    "n_colors": n_colors[inx],
                    "cost_pattern": cost_pattern,
                    "seed": current_seed,
                    "density": nx.density(graph),
                }

                # Guardar instancia
                filename = get_instance_filename(
                    graph_type,
                    n_vertices,
                    n_colors[inx],
                    metadata["density"],
                    current_seed,
                    output_dir,
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
    Genera múltiples instancias de Árboles Aleatorios para el Proyecto DAA - MCCPP.

    La estructura es siempre un Árbol (grafo conexo y acíclico).

    Args:
        n_vertices_list: lista de cantidades de vértices a generar.
        n_colors: número de colores disponibles en la partición cromática.
        cost_pattern: tipo de estructura de costos aplicada.
        n_instances: número de instancias independientes por cada tamaño.
        seed: semilla aleatoria inicial para reproducibilidad.
        output_dir: directorio donde se guardarán las instancias.

    Returns:
        lista con la metadata de las instancias generadas.
    """
    instances_metadata = []
    current_seed = seed

    def generate_tree_graph(n_vertices: int, seed: int) -> nx.Graph:
        """Genera un árbol aleatorio con n_vertices."""
        # Usamos la función NetworkX para generar un árbol de expansión aleatorio
        # dado un grafo completo, lo que garantiza un árbol.
        import random

        random.seed(seed)
        # Genera un árbol con un número fijo de aristas (n-1) de forma aleatoria.
        return random_unlabeled_tree(n_vertices, seed=seed)

    for n_vertices in n_vertices_list:
        for i in range(n_instances):
            # 1. Generar grafo (siempre un árbol)
            # Nota: La probabilidad p se ignora, ya que la estructura está fijada a ser un árbol.
            graph = generate_tree_graph(n_vertices, seed=current_seed)

            # 2. Generar matriz de costos
            cost_matrix = generate_structured_cost_matrix(
                n_vertices, n_colors, cost_pattern, seed=current_seed
            )

            # 3. Preparar metadatos
            # La densidad de un árbol es |E|/|V|*(|V|-1)/2. Como |E| = |V|-1:
            # density = 2 * (n_vertices - 1) / (n_vertices * (n_vertices - 1))
            density = nx.density(graph)

            metadata = {
                "instance_type": "tree",  # Tipo de instancia cambiado a 'tree'
                "n_vertices": n_vertices,
                "n_colors": n_colors,
                "p": 0.0,  # Se fija en 0.0 para mantener compatibilidad en metadata, pero no es relevante
                "cost_pattern": cost_pattern,
                "seed": current_seed,
                "density": density,
            }

            # 4. Guardar instancia
            # Usamos p=0.0 como placeholder en el nombre del archivo
            filename = get_instance_filename(
                "tree", n_vertices, n_colors, 0.0, current_seed, output_dir
            )
            save_instance(graph, cost_matrix, filename, metadata)

            instances_metadata.append({"filename": filename, "metadata": metadata})

            current_seed += 1

    return instances_metadata
