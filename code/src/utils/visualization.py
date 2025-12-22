"""
Visualization utilities for DAA Project - MCCPP
"""
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

def draw_graph_coloring(graph: nx.Graph, coloring: Dict[int, int], 
                       title: str = "Graph Coloring", 
                       save_path: Optional[str] = None,
                       show_labels: bool = True):
    """
    Draw graph with coloring
    
    Args:
        graph: networkx Graph
        coloring: vertex to color mapping
        title: plot title
        save_path: path to save image (optional)
        show_labels: whether to show vertex labels
    """
    plt.figure(figsize=(10, 8))
    
    # Get positions for all nodes
    pos = nx.spring_layout(graph, seed=42)
    
    # Get unique colors and create color map
    unique_colors = set(coloring.values())
    color_map = plt.cm.Set3(np.linspace(0, 1, len(unique_colors)))
    color_dict = {color: color_map[i] for i, color in enumerate(unique_colors)}
    
    # Draw nodes with colors
    node_colors = [color_dict[coloring[node]] for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                          node_size=500, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    
    # Draw labels if requested
    if show_labels:
        nx.draw_networkx_labels(graph, pos, font_size=10)
    
    # Create legend for colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color_dict[color], 
                                 markersize=10, label=f'Color {color}')
                      for color in unique_colors]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_cost_matrix(cost_matrix: np.ndarray, title: str = "Cost Matrix"):
    """
    Visualize cost matrix as heatmap
    
    Args:
        cost_matrix: n_vertices x n_colors cost matrix
        title: plot title
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cost_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
                cbar_kws={'label': 'Cost'})
    
    plt.xlabel("Colors/Frequencies")
    plt.ylabel("Vertices/Towers")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_algorithm_comparison(results: Dict[str, List[float]], 
                            metric: str = "cost",
                            title: str = "Algorithm Comparison"):
    """
    Plot comparison of multiple algorithms
    
    Args:
        results: dictionary of algorithm_name -> list of metric values
        metric: metric being compared ('cost', 'time', 'ratio')
        title: plot title
    """
    plt.figure(figsize=(12, 6))
    
    algorithms = list(results.keys())
    values = list(results.values())
    
    # Create box plot
    plt.boxplot(values, labels=algorithms)
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_scalability_analysis(sizes: List[int], times: List[float], 
                            costs: List[float], algorithm_name: str):
    """
    Plot scalability analysis of an algorithm
    
    Args:
        sizes: list of problem sizes (number of vertices)
        times: list of execution times
        costs: list of solution costs
        algorithm_name: name of the algorithm
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot time scalability
    ax1.plot(sizes, times, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Vertices')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title(f'{algorithm_name} - Time Scalability')
    ax1.grid(True, alpha=0.3)
    
    # Plot cost scalability
    ax2.plot(sizes, costs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Vertices')
    ax2.set_ylabel('Solution Cost')
    ax2.set_title(f'{algorithm_name} - Cost Scalability')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_convergence_curve(iterations: List[int], costs: List[float],
                          algorithm_name: str, best_known: float = None):
    """
    Plot convergence curve for metaheuristics
    
    Args:
        iterations: list of iteration numbers
        costs: list of best costs at each iteration
        algorithm_name: name of the algorithm
        best_known: known best cost (if available)
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(iterations, costs, 'b-', linewidth=2, label='Best Cost')
    
    if best_known is not None:
        plt.axhline(y=best_known, color='r', linestyle='--', 
                   label=f'Known Best: {best_known:.2f}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Solution Cost')
    plt.title(f'{algorithm_name} - Convergence Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Often useful for convergence plots
    
    plt.tight_layout()
    plt.show()

def save_multiple_plots(plots_data: List[Dict], filename_prefix: str):
    """
    Save multiple plots for report
    
    Args:
        plots_data: list of plot specifications
        filename_prefix: prefix for saved files
    """
    for i, plot_spec in enumerate(plots_data):
        plt.figure(figsize=plot_spec.get('figsize', (10, 6)))
        
        # Generate plot based on type
        plot_type = plot_spec['type']
        if plot_type == 'comparison':
            plot_algorithm_comparison(**plot_spec['data'])
        elif plot_type == 'scalability':
            plot_scalability_analysis(**plot_spec['data'])
        elif plot_type == 'convergence':
            plot_convergence_curve(**plot_spec['data'])
        
        # Save plot
        filename = f"{filename_prefix}_{i:02d}_{plot_type}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()