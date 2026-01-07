import sys
import os
import json
import numpy as np
import networkx as nx
from pathlib import Path
from utils.timeout_handler import set_global_timeout, reset_global_timeout
from utils.instances_generator import generate_full_benchmark_set
from utils.utils import load_instance, evaluate_solution
from utils.evaluation import compare_algorithms, generate_comparison_plots
from algorithms.exacts_algorithms import (
    brute_force_solver,
    backtracking_solver,
    intelligent_backtracking,
    ilp_solver,
    dynamic_programming_tree,
    dp_interval_graph_solver
)
from algorithms.approximation_algorithms import (
    weighted_set_cover_approximation,
    improved_weighted_set_cover
)
from algorithms.heuristic_algorithms import (
    largest_first_heuristic,
    dsatur_heuristic,
    recursive_largest_first_heuristic,
    peo_greedy_heuristic
)
from algorithms.metaheuristic_algorithms import (
    simulated_annealing,
    adaptive_simulated_annealing,
    trajectory_search_heuristic,
    hybrid_metaheuristic,
    adaptive_metaheuristic
)
from algorithms.exacts_algorithms import (
    brute_force_solver,
    backtracking_solver,
    intelligent_backtracking,
    ilp_solver,
    dynamic_programming_tree,
    dp_interval_graph_solver
)
from algorithms.approximation_algorithms import (
    weighted_set_cover_approximation,
    improved_weighted_set_cover
)
from algorithms.metaheuristic_algorithms import (
    simulated_annealing,
    adaptive_simulated_annealing,
    trajectory_search_heuristic,
    hybrid_metaheuristic,
    adaptive_metaheuristic
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI Colors
RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
BOLD = "\033[1m"

# Definición de algoritmos disponibles
ALGORITHMS = {
    'brute_force': {
        'func': brute_force_solver,
        'category': 'Exactos',
        'condition': lambda g, cm: True,  # SIEMPRE ejecutar (timeout global: 180s)
        'description': 'Fuerza bruta completa (SIEMPRE - con timeout 180s)'
    },
    'backtracking': {
        'func': backtracking_solver,
        'category': 'Exactos',
        'condition': lambda g, cm: True,  # SIEMPRE ejecutar (timeout global: 180s)
        'description': 'Backtracking básico (SIEMPRE - con timeout 180s)'
    },
    'intelligent_backtracking': {
        'func': intelligent_backtracking,
        'category': 'Exactos',
        'condition': lambda g, cm: True,  # SIEMPRE ejecutar (timeout global: 180s)
        'description': 'Backtracking con poda inteligente (SIEMPRE - con timeout 180s)'
    },
    'ilp_solver': {
        'func': ilp_solver,
        'category': 'Exactos',
        'condition': lambda g, cm: len(g.nodes) <= 50,
        'description': 'Programación lineal entera (ejecutar si n ≤ 50)'
    },
    'dp_tree': {
        'func': dynamic_programming_tree,
        'category': 'Exactos',
        'condition': lambda g, cm: nx.is_tree(g),
        'description': 'Programación dinámica para árboles (SOLO si es árbol)'
    },
    'dp_interval': {
        'func': dp_interval_graph_solver,
        'category': 'Exactos',
        'condition': lambda g, cm: nx.is_chordal(g),  # Aproximadamente interval graphs
        'description': 'Programación dinámica para grafos de intervalo (SOLO si es chordal/intervalo)'
    },
    'wsc': {
        'func': weighted_set_cover_approximation,
        'category': 'Aproximación',
        'condition': lambda g, cm: True,
        'description': 'Weighted Set Cover básico'
    },
    'wsc_improved': {
        'func': improved_weighted_set_cover,
        'category': 'Aproximación',
        'condition': lambda g, cm: True,
        'description': 'Weighted Set Cover mejorado'
    },
    'largest_first': {
        'func': largest_first_heuristic,
        'category': 'Heurísticas',
        'condition': lambda g, cm: True,
        'description': 'Largest First greedy'
    },
    'dsatur': {
        'func': dsatur_heuristic,
        'category': 'Heurísticas',
        'condition': lambda g, cm: True,
        'description': 'DSATUR heuristic'
    },
    'rlf': {
        'func': recursive_largest_first_heuristic,
        'category': 'Heurísticas',
        'condition': lambda g, cm: True,
        'description': 'Recursive Largest First'
    },
    'peo_greedy': {
        'func': peo_greedy_heuristic,
        'category': 'Heurísticas',
        'condition': lambda g, cm: nx.is_chordal(g),
        'description': 'Perfect Elimination Order - Greedy (solo grafos cordales)'
    },
    'simulated_annealing': {
        'func': simulated_annealing,
        'category': 'Metaheurísticas',
        'condition': lambda g, cm: len(g.nodes) <= 100,
        'description': 'Simulated Annealing'
    },
    'adaptive_sa': {
        'func': adaptive_simulated_annealing,
        'category': 'Metaheurísticas',
        'condition': lambda g, cm: len(g.nodes) <= 100,
        'description': 'Simulated Annealing adaptivo'
    },
    'trajectory_search': {
        'func': trajectory_search_heuristic,
        'category': 'Metaheurísticas',
        'condition': lambda g, cm: len(g.nodes) <= 100,
        'description': 'Búsqueda por trayectoria'
    },
    'hybrid': {
        'func': hybrid_metaheuristic,
        'category': 'Metaheurísticas',
        'condition': lambda g, cm: len(g.nodes) <= 100,
        'description': 'Metaheurística híbrida'
    },
    'adaptive': {
        'func': adaptive_metaheuristic,
        'category': 'Metaheurísticas',
        'condition': lambda g, cm: len(g.nodes) <= 100,
        'description': 'Metaheurística adaptiva'
    }
}

# MODO DE EJECUCIÓN GLOBAL
# Puede ser: "selective" (tamaño-basado) o "all_brute" (todos los brute forces siempre)
EXECUTION_MODE = "all_brute"  # Ahora por defecto modo "todos vs todos"

def get_algorithm_condition(algo_name: str, graph: nx.Graph, cost_matrix: np.ndarray) -> bool:
    """
    Determina si un algoritmo debe ejecutarse según el EXECUTION_MODE actual.
    
    En modo "all_brute": Los 3 brute forces SIEMPRE se ejecutan
    En modo "selective": Los algoritmos se eligen según el tamaño del grafo
    """
    n_vertices = graph.number_of_nodes()
    
    # Forzar ejecución de los 3 brute forces en modo "all_brute"
    if EXECUTION_MODE == "all_brute" and algo_name in ['brute_force', 'backtracking', 'intelligent_backtracking']:
        return True
    
    # Modo selectivo: aplicar condiciones basadas en tamaño
    if EXECUTION_MODE == "selective":
        if algo_name == 'brute_force':
            return n_vertices <= 7
        elif algo_name == 'backtracking':
            return 7 < n_vertices <= 12
        elif algo_name == 'intelligent_backtracking':
            return 12 < n_vertices <= 20
        elif algo_name == 'ilp_solver':
            return n_vertices <= 50
        elif algo_name == 'dp_tree':
            return nx.is_tree(graph)
        elif algo_name == 'dp_interval' or algo_name == 'peo_greedy':
            return nx.is_chordal(graph)
        elif algo_name in ['simulated_annealing', 'adaptive_sa', 'trajectory_search', 'hybrid', 'adaptive']:
            return n_vertices < 100
        else:
            # Aproximación y Heurísticas siempre
            return True
    
    # Fallback: usar condición del algoritmo
    return ALGORITHMS[algo_name]['condition'](graph, cost_matrix)

def print_banner():
    """Banner de bienvenida"""
    banner = f"""
{CYAN}{'='*80}
║              MCCPP - Minimum Cost Chromatic Partition Problem            ║
{'='*80}{RESET}

{BOLD}Autores:{RESET}
  • Yesenia Valdés Rodríguez (C411)
  • Laura Martir Beltrán (C411)
  • Adrián Hernández Castellanos (C412)

{CYAN}{'='*80}{RESET}
"""
    print(banner)

def print_menu():
    """Menú principal"""
    mode_desc = "TODOS vs TODOS (3 brute forces siempre)" if EXECUTION_MODE == "all_brute" else "SELECTIVO (por tamaño)"
    menu = f"""
{BOLD}{BLUE}┌─────────────────────────────── MENÚ PRINCIPAL ───────────────────────────────┐{RESET}
│  {YELLOW}Modo:{RESET} {mode_desc:<48}                      │
│                                                                              │
│  {CYAN}1.{RESET} {BOLD}Generar instancias de prueba{RESET}                                             │
│  {CYAN}2.{RESET} Listar instancias disponibles                                            │
│  {CYAN}3.{RESET} Listar algoritmos disponibles                                            │
│  {CYAN}4.{RESET} Ejecutar procesamiento completo (todos vs todos)                         │
│  {CYAN}5.{RESET} Ejecutar procesamiento personalizado                                     │
│  {CYAN}6.{RESET} Visualizar estadísticas guardadas                                        │
│  {CYAN}7.{RESET} Cambiar modo de ejecución                                                │
│  {CYAN}0.{RESET} Salir                                                                    │
│                                                                              │
{BOLD}{BLUE}└──────────────────────────────────────────────────────────────────────────────┘{RESET}
"""
    print(menu)

def select_instance():
    """Permite al usuario seleccionar una instancia"""
    print(f"\n{BOLD}Instancias disponibles:{RESET}")
    
    instances_dir = Path("instances")
    if not instances_dir.exists():
        print(f"{RED}✗ No se encontró el directorio 'instances/'{RESET}")
        return None
    
    instance_files = sorted(instances_dir.glob("*.json"))
    if not instance_files:
        print(f"{RED}✗ No hay instancias generadas{RESET}")
        return None
    
    # Mostrar todas las instancias
    for i, file in enumerate(instance_files, 1):
        print(f"  {CYAN}{i}.{RESET} {file.name}")
    
    print(f"\n  {CYAN}0.{RESET} Volver")
    
    try:
        choice = int(input(f"\n{BOLD}Selecciona una instancia (número): {RESET}"))
        if choice == 0:
            return None
        if 1 <= choice <= len(instance_files):
            return str(instance_files[choice - 1])
        else:
            print(f"{RED}✗ Opción inválida{RESET}")
            return None
    except ValueError:
        print(f"{RED}✗ Entrada inválida{RESET}")
        return None

def list_instances():
    """Lista todas las instancias disponibles con información de factibilidad"""
    print(f"\n{BOLD}Instancias disponibles:{RESET}")
    
    instances_dir = Path("instances")
    if not instances_dir.exists():
        print(f"{RED}✗ No se encontró el directorio 'instances/'{RESET}")
        return []
    
    instance_files = sorted(instances_dir.glob("*.json"))
    if not instance_files:
        print(f"{RED}✗ No hay instancias generadas{RESET}")
        return []
    
    factible_count = 0
    infactible_count = 0
    
    for i, file in enumerate(instance_files, 1):
        try:
            graph_loaded, cost_matrix, metadata = load_instance(str(file))

            # Preferir metadata si está presente, si no inferir
            n_vertices = metadata.get('n_vertices', graph_loaded.number_of_nodes())
            try:
                n_colors = metadata.get('n_colors', cost_matrix.shape[1])
            except Exception:
                n_colors = metadata.get('n_colors', '?')

            feasible_meta = metadata.get('is_feasible', None)

            # Inferir factibilidad cuando la metadata no la contiene
            feasible = False
            chromatic_est = None
            try:
                if feasible_meta is not None:
                    feasible = bool(feasible_meta)
                else:
                    if nx.is_tree(graph_loaded):
                        feasible = (n_vertices <= 1 and (n_colors == '?' or int(n_colors) >= 1)) or (
                            n_vertices > 1 and (n_colors == '?' or int(n_colors) >= 2)
                        )
                    elif nx.is_bipartite(graph_loaded):
                        feasible = (n_vertices <= 1 and (n_colors == '?' or int(n_colors) >= 1)) or (
                            n_vertices > 1 and (n_colors == '?' or int(n_colors) >= 2)
                        )
                    else:
                        test_coloring = nx.coloring.greedy_color(graph_loaded, strategy='largest_first')
                        chromatic_est = len(set(test_coloring.values()))
                        feasible = chromatic_est <= int(n_colors) if n_colors != '?' else False
            except Exception:
                feasible = False

            # Preparar texto de estimación cromática
            if chromatic_est is None:
                try:
                    test_coloring = nx.coloring.greedy_color(graph_loaded, strategy='largest_first')
                    chromatic_est = len(set(test_coloring.values()))
                    chi_info = f"χ≈{chromatic_est}"
                except Exception:
                    chi_info = "χ=?"
            else:
                chi_info = f"χ≈{chromatic_est}"

            if feasible:
                feasible_str = f"{GREEN}Sí{RESET}"
                factible_count += 1
            else:
                feasible_str = f"{RED}No{RESET}"
                infactible_count += 1

            print(f"  {CYAN}{i}.{RESET} {file.name}")
            print(f"     Factible: {feasible_str} | n={n_vertices}, k={n_colors}, {chi_info}")

        except Exception as e:
            print(f"  {CYAN}{i}.{RESET} {file.name} (Error: {str(e)[:30]})")
    
    print(f"\n{BOLD}Resumen:{RESET} {factible_count} factibles, {infactible_count} infactibles")
    
    if infactible_count > 0:
        print(f"{YELLOW}⚠ Las instancias infactibles necesitan más colores de los disponibles{RESET}")
        print(f"{YELLOW}  (χ > k). Los algoritmos no encontrarán soluciones válidas.{RESET}")
    
    return [str(f) for f in instance_files]

def list_algorithms():
    """Lista todos los algoritmos disponibles"""
    print(f"\n{BOLD}Algoritmos disponibles:{RESET}")
    
    categories = {}
    for name, info in ALGORITHMS.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info['description']))
    
    for cat, algos in categories.items():
        print(f"\n{BOLD}{MAGENTA}{cat}:{RESET}")
        for name, desc in algos:
            print(f"  {CYAN}{name}{RESET}: {desc}")

def toggle_execution_mode():
    """Cambia el modo de ejecución entre 'selective' y 'all_brute'"""
    global EXECUTION_MODE
    
    if EXECUTION_MODE == "selective":
        EXECUTION_MODE = "all_brute"
        print(f"\n{GREEN}✓ Modo de ejecución cambiado a:{RESET} {BOLD}TODOS vs TODOS{RESET}")
        print(f"  Ahora se ejecutarán los 3 algoritmos brute force (fuerza bruta, backtracking, backtracking inteligente)")
        print(f"  siempre en cada instancia, con timeout de 180 segundos cada uno.")
    else:
        EXECUTION_MODE = "selective"
        print(f"\n{GREEN}✓ Modo de ejecución cambiado a:{RESET} {BOLD}SELECTIVO (basado en tamaño){RESET}")
        print(f"  Ahora se ejecutarán los algoritmos de forma selectiva según el tamaño del grafo.")
    
    return

def select_multiple_instances():
    """Permite seleccionar múltiples instancias"""
    instances = list_instances()
    if not instances:
        return []
    
    print(f"\n{BOLD}Selección múltiple:{RESET}")
    print("  - Ingresa números separados por comas (ej: 1,3,5)")
    print("  - 'all' para seleccionar todas")
    print("  - '0' para cancelar")
    
    try:
        choice = input(f"\n{BOLD}Selecciona instancias: {RESET}").strip().lower()
        if choice == '0':
            return []
        if choice == 'all':
            return instances
        
        indices = [int(x.strip()) for x in choice.split(',')]
        selected = []
        for idx in indices:
            if 1 <= idx <= len(instances):
                selected.append(instances[idx - 1])
            else:
                print(f"{RED}✗ Índice {idx} inválido{RESET}")
        return selected
    except ValueError:
        print(f"{RED}✗ Entrada inválida{RESET}")
        return []

def select_multiple_algorithms():
    """Permite seleccionar múltiples algoritmos"""
    list_algorithms()
    
    print(f"\n{BOLD}Selección múltiple:{RESET}")
    print("  - Ingresa nombres separados por comas (ej: brute_force,dsatur,largest_first)")
    print("  - 'all' para seleccionar todos")
    print("  - 'category:name' para seleccionar categoría (ej: Exactos:all)")
    print("  - '0' para cancelar")
    
    try:
        choice = input(f"\n{BOLD}Selecciona algoritmos: {RESET}").strip()
        if choice == '0':
            return []
        if choice == 'all':
            return list(ALGORITHMS.keys())
        
        if ':' in choice:
            cat, subchoice = choice.split(':', 1)
            if cat in ['Exactos', 'Aproximación', 'Heurísticas', 'Metaheurísticas']:
                if subchoice == 'all':
                    return [name for name, info in ALGORITHMS.items() if info['category'] == cat]
                else:
                    return [subchoice] if subchoice in ALGORITHMS and ALGORITHMS[subchoice]['category'] == cat else []
        
        names = [x.strip() for x in choice.split(',')]
        selected = []
        for name in names:
            if name in ALGORITHMS:
                selected.append(name)
            else:
                print(f"{RED}✗ Algoritmo '{name}' no encontrado{RESET}")
        return selected
    except:
        print(f"{RED}✗ Entrada inválida{RESET}")
        return []

def format_result(result: dict, show_solution: bool = False) -> str:
    """Formatea un resultado para mostrar"""
    feasible_symbol = "✓" if result.get('feasible', False) else "✗"
    optimal_symbol = "★" if result.get('optimal', False) else "○"
    
    cost = result.get('cost', float('inf'))
    cost_str = f"{cost:.2f}" if cost != float('inf') else "∞"
    
    ops = result.get('operations', result.get('iterations', 'N/A'))
    # Formatear operaciones: si es número, agregar separador de miles; si no, dejar como está
    if isinstance(ops, (int, float)) and ops != 'N/A':
        ops_str = f"{int(ops):,}"
    else:
        ops_str = str(ops)
    
    time_str = f"{result.get('execution_time', 0):.4f}s"
    
    output = f"""
    {GREEN if result.get('feasible') else RED}{feasible_symbol} Factible{RESET}  {YELLOW}{optimal_symbol} Óptimo{RESET}
    Costo: {BOLD}{cost_str}{RESET}
    Operaciones: {CYAN}{ops_str}{RESET}
    Tiempo: {time_str}
    """
    
    if show_solution and result.get('solution'):
        output += f"  Colores usados: {len(set(result['solution'].values()))}\n"
    
    return output

def run_all_algorithms(instance_file: str):
    """Ejecuta todos los algoritmos en una instancia"""
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}EJECUTANDO TODOS LOS ALGORITMOS{RESET}")
    print(f"Modo: {BOLD}{'TODOS vs TODOS (3 brute forces siempre)' if EXECUTION_MODE == 'all_brute' else 'SELECTIVO (por tamaño)'}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    # Cargar instancia
    print(f"{BLUE}→ Cargando instancia:{RESET} {Path(instance_file).name}")
    graph, cost_matrix, metadata = load_instance(instance_file)
    
    n_vertices = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_colors = cost_matrix.shape[1]
    
    print(f"  Vértices: {n_vertices}, Aristas: {n_edges}, Colores: {n_colors}")
    print(f"  Densidad: {metadata.get('density', 'N/A'):.3f}")
    print(f"  Tipo: {metadata.get('instance_type', 'unknown')}\n")
    
    results = {}
    
    # Agrupar algoritmos por categoría
    categories = {}
    for name, info in ALGORITHMS.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)
    
    # Ejecutar algoritmos
    for category, algo_names in categories.items():
        print(f"\n{BOLD}{MAGENTA}▶ {category}{RESET}")
        print(f"{MAGENTA}{'─'*80}{RESET}")
        
        for algo_name in algo_names:
            # Determinar si se debe ejecutar usando la nueva función
            should_run = get_algorithm_condition(algo_name, graph, cost_matrix)
            
            if not should_run:
                print(f"  {YELLOW}⊗{RESET} {algo_name}: Saltado (instancia no cumple condiciones)")
                continue
            
            try:
                print(f"\n  {CYAN}◉{RESET} {BOLD}{algo_name}{RESET}")
                algo_func = ALGORITHMS[algo_name]['func']
                result = algo_func(graph, cost_matrix)
                results[algo_name] = result
                print(format_result(result))
                
            except Exception as e:
                print(f"  {RED}✗ ERROR: {e}{RESET}")
                results[algo_name] = {'error': str(e), 'cost': float('inf'), 'feasible': False}
    
    # Resumen comparativo
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}RESUMEN COMPARATIVO{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")
    
    # Ordenar por costo
    valid_results = [(name, r) for name, r in results.items() 
                     if r.get('feasible', False) and r.get('cost', float('inf')) < float('inf')]
    valid_results.sort(key=lambda x: x[1]['cost'])
    
    if valid_results:
        best_name, best_result = valid_results[0]
        print(f"{GREEN}★ MEJOR SOLUCIÓN:{RESET} {BOLD}{best_name}{RESET}")
        print(f"  Costo: {BOLD}{best_result['cost']:.2f}{RESET}")
        print(f"  Operaciones: {best_result.get('operations', best_result.get('iterations', 'N/A'))}")
        
        print(f"\n{BOLD}Top 5 Algoritmos por Costo:{RESET}")
        for i, (name, res) in enumerate(valid_results[:5], 1):
            ops = res.get('operations', res.get('iterations', 'N/A'))
            print(f"  {i}. {name:30s} | Costo: {res['cost']:10.2f} | Ops: {ops}")
    else:
        print(f"{RED}✗ Ningún algoritmo produjo solución factible{RESET}")

def generate_test_instances():
    """Genera instancias de prueba CON ADVERTENCIA sobre factibilidad"""
    print(f"\n{BOLD}GENERACIÓN DE INSTANCIAS{RESET}\n")
    
    output_dir = "instances"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{CYAN}→{RESET} Generando conjunto completo de benchmark (esto puede tardar)...")
    print(f"{YELLOW}ℹ Se garantiza que k ≥ χ(G) para todas las instancias{RESET}")
    generated = generate_full_benchmark_set(target_total=100, keep_random=20, seed=42, output_dir=output_dir)
    print(f"{GREEN}✓ {len(generated)} instancias generadas en '{output_dir}/'{RESET}")
    return

def run_processing(instance_files: list, algorithm_names: list, results_dir: str = "results"):

    from utils.utils import ensure_directory, is_proper_coloring
    ensure_directory(results_dir)

    import time, json

    for instance_file in instance_files:
        instance_name = Path(instance_file).stem
        
        try:
            graph, cost_matrix, metadata = load_instance(instance_file)
            feasible = metadata.get('is_feasible', False)
            n_colors = metadata.get('n_colors', cost_matrix.shape[1])
            
            # VALIDACIÓN: Verificar factibilidad real
            if not feasible:
                # Intentar colorear con greedy para confirmar
                try:
                    test_coloring = nx.coloring.greedy_color(graph, strategy='largest_first')
                    chromatic_number = len(set(test_coloring.values()))
                    
                    if chromatic_number > n_colors:
                        print(f"\n{YELLOW}⚠ {instance_name}: INFACTIBLE - necesita {chromatic_number} colores, solo tiene {n_colors}{RESET}")
                        print(f"  Saltando instancia infactible...")
                        continue
                except Exception:
                    pass
            
            feasible_str = "Sí" if feasible else "No"
            print(f"\n{BOLD}{BLUE}Procesando instancia:{RESET} {instance_name} (Factible: {feasible_str}, χ≤{n_colors})")
            
        except Exception as e:
            print(f"{RED}✗ Error cargando instancia {instance_file}: {e}{RESET}")
            continue

        per_instance_results = {"instance": instance_name, "metadata": metadata, "runs": {}}
        
        for algo_name in algorithm_names:
            if algo_name not in ALGORITHMS:
                print(f"  {YELLOW}⊗{RESET} {algo_name}: No encontrado, saltando")
                continue

            algo_info = ALGORITHMS[algo_name]
            
            # Determinar si se debe ejecutar según EXECUTION_MODE
            can_run = get_algorithm_condition(algo_name, graph, cost_matrix)

            if not can_run:
                print(f"  {YELLOW}⊗{RESET} {algo_name}: Condición no satisfecha, saltando")
                per_instance_results['runs'][algo_name] = {
                    'skipped': True,
                    'feasible': False,
                    'cost': float('inf'),
                    'execution_time': 0.0,
                    'operations': 0
                }
                continue

            print(f"  {CYAN}◉{RESET} Ejecutando {BOLD}{algo_name}{RESET}...", end=" ")
            func = algo_info['func']
            start = time.time()
            
            # Establecer timeout de 180 segundos (3 minutos) para este algoritmo
            set_global_timeout(180.0)
            
            try:
                result = func(graph, cost_matrix)
            except Exception as e:
                duration = time.time() - start
                print(f"{RED}ERROR: {str(e)[:50]}{RESET}")
                per_instance_results['runs'][algo_name] = {
                    'error': str(e),
                    'feasible': False,
                    'cost': float('inf'),
                    'execution_time': duration,
                    'operations': 0
                }
                continue
            finally:
                # Reiniciar el timeout global
                reset_global_timeout()

            duration = time.time() - start

            sol = result.get('solution') or result.get('coloring') or None
            feasible_result = result.get('feasible')
            
            if feasible_result is None and sol:
                feasible_result = bool(sol and is_proper_coloring(graph, sol))

            cost = result.get('cost')
            if cost is None and sol is not None and feasible_result:
                cost = evaluate_solution(sol, cost_matrix)

            operations = result.get('operations', result.get('iterations', None))

            per_instance_results['runs'][algo_name] = {
                'skipped': False,
                'feasible': bool(feasible_result),
                'cost': float(cost) if cost is not None else float('inf'),
                'execution_time': float(result.get('execution_time', duration)),
                'operations': int(operations) if isinstance(operations, (int, float)) else operations,
                'solution': sol
            }

            # Detect optimal: algorithm returns optimal=True OR cost matches known_optimal_cost
            is_optimal = result.get('optimal', False)
            known_opt = metadata.get('known_optimal_cost')
            
            if not is_optimal and known_opt is not None:
                try:
                    if abs(result.get('cost', float('inf')) - float(known_opt)) < 1e-6:
                        is_optimal = True
                except Exception:
                    pass
            
            per_instance_results['runs'][algo_name]['optimal'] = is_optimal

            ops_display = per_instance_results['runs'][algo_name]['operations'] if per_instance_results['runs'][algo_name]['operations'] is not None else 'N/A'
            cost_display = f"{per_instance_results['runs'][algo_name]['cost']:.2f}" if per_instance_results['runs'][algo_name]['cost'] != float('inf') else "∞"
            
            print(f"{GREEN}✓{RESET} Tiempo: {per_instance_results['runs'][algo_name]['execution_time']:.4f}s | Ops: {ops_display} | Costo: {cost_display} | Factible: {per_instance_results['runs'][algo_name]['feasible']}")

        out_file = os.path.join(results_dir, f"{instance_name}_results.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(per_instance_results, f, indent=2, default=str)

        # Calcular optimalidad global
        runs = per_instance_results['runs']
        feasible_runs_list = [r for r in runs.values() if not r.get('skipped') and r.get('feasible') and r.get('cost', float('inf')) < float('inf')]
        best_cost = None
        if feasible_runs_list:
            best_cost = min(r['cost'] for r in feasible_runs_list)

        # Detect optimal solutions by comparing with known_optimal_cost
        known_opt = metadata.get('known_optimal_cost')
        tol = 1e-6

        for name, r in runs.items():
            r_opt = False
            
            # Mark as optimal if:
            # 1. Algorithm explicitly returns optimal=True, OR
            # 2. Cost matches the known optimal cost
            if r.get('optimal', False):
                r_opt = True
            elif known_opt is not None and r.get('feasible', False):
                try:
                    if abs(r.get('cost', float('inf')) - float(known_opt)) < tol:
                        r_opt = True
                except Exception:
                    pass
            
            r['optimal'] = bool(r_opt)

        # Imprimir tabla ordenada
        header = f"\n    {'Algoritmo':<25} | {'Skip':^5} | {'Tiempo(s)':>10} | {'Ops':>8} | {'Costo':>12} | {'Factible':>8} | {'Óptimo':>6}"
        sep = '    ' + '-' * (len(header) - 4)

        def sort_key(item):
            _, r = item
            INF = float('inf')
            try:
                cost_raw = r.get('cost', INF)
                cost = float(cost_raw) if r.get('feasible') else INF
            except Exception:
                cost = INF

            try:
                timev = float(r.get('execution_time')) if (not r.get('skipped') and r.get('execution_time') is not None) else INF
            except Exception:
                timev = INF

            opt_flag = 0 if r.get('optimal') else 1

            if cost != INF:
                cost = round(cost, 6)

            return (cost, timev, opt_flag)

        ordered = sorted(runs.items(), key=sort_key)

        print(header)
        print(sep)
        for name, r in ordered:
            if r is None:
                line = f"    {name:<25} | {'NO':^5} | {'-':>10} | {'-':>8} | {'-':>12} | {'-':>8} | {'-':>6}"
            else:
                skipped = 'YES' if r.get('skipped') else 'NO'
                t = f"{r.get('execution_time', 0):.4f}" if not r.get('skipped') else '-'
                ops = str(r.get('operations')) if r.get('operations') is not None else 'N/A'
                costv = f"{r.get('cost', float('inf')):.2f}" if r.get('cost', None) is not None and r.get('cost') != float('inf') else '∞'
                feas = 'Yes' if r.get('feasible') else 'No'
                opt = 'Yes' if r.get('optimal') else 'No'
                line = f"    {name:<25} | {skipped:^5} | {t:>10} | {ops:>8} | {costv:>12} | {feas:>8} | {opt:>6}"
            print(line)

    # Generar plots al final
    try:
        from collections import defaultdict
        benchmark_results = {}
        instance_classes = {}
        
        for fpath in sorted(Path(results_dir).glob("*_results.json")):
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception:
                continue

            inst_name = data.get('instance', fpath.stem.replace('_results',''))
            runs = data.get('runs', {})
            benchmark_results[inst_name] = {}
            meta = data.get('metadata', {})
            cls = meta.get('instance_type', 'unknown')
            instance_classes[inst_name] = cls

            for algo, r in runs.items():
                benchmark_results[inst_name][algo] = {
                    'cost': r.get('cost', float('inf')),
                    'execution_time': r.get('execution_time', 0.0),
                    'feasible': r.get('feasible', False),
                    'optimal': r.get('optimal', False),
                    'operations': r.get('operations', r.get('iterations', 'N/A'))
                }

        class_groups = defaultdict(dict)
        for inst, cls in instance_classes.items():
            class_groups[cls][inst] = benchmark_results.get(inst, {})

        plots_base = os.path.join(results_dir, 'plots')
        for cls, cls_results in class_groups.items():
            if not cls_results:
                continue
            try:
                comparison = compare_algorithms(cls_results)
                generate_comparison_plots(comparison, cls_results, os.path.join(plots_base, cls))
                print(f"{GREEN}✓ Plots generados para clase '{cls}' en {os.path.join(plots_base, cls)}{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠ No se pudieron generar plots para '{cls}': {e}{RESET}")
    except Exception as e:
        print(f"{YELLOW}⚠ Error al agregar/graficar resultados: {e}{RESET}")

def view_statistics(results_dir: str = "results"):
    """Muestra estadísticas guardadas en `results/` (archivos *_results.json)."""
    from utils.utils import ensure_directory
    ensure_directory(results_dir)
    files = sorted(Path(results_dir).glob("*_results.json"))
    if not files:
        print(f"{YELLOW}⚠ No hay estadísticas guardadas en '{results_dir}'{RESET}")
        return

    print(f"\n{BOLD}Estadísticas disponibles:{RESET}")
    for i, f in enumerate(files, 1):
        print(f"  {CYAN}{i}.{RESET} {f.name}")

    try:
        choice = input(f"\n{BOLD}Ver (número) o 'all': {RESET}").strip().lower()
        def print_stats_file(path: Path):
            try:
                with open(path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception as e:
                print(f"{RED}✗ Error leyendo {path.name}: {e}{RESET}")
                return

            instance_name = data.get('instance', path.stem)
            metadata = data.get('metadata', {})
            runs = data.get('runs', {})

            print(f"\n{BOLD}{BLUE}Instancia:{RESET} {instance_name}")
            print(f"  Tipo: {metadata.get('instance_type', 'N/A')}, V={metadata.get('n_vertices', 'N/A')}, K={metadata.get('n_colors', 'N/A')}")

            # Ensure optimal flags exist
            feasible_runs = [r for r in runs.values() if not r.get('skipped') and r.get('feasible') and r.get('cost', float('inf')) < float('inf')]
            best_cost = None
            if feasible_runs:
                best_cost = min(r['cost'] for r in feasible_runs)

            # Detect optimal solutions by comparing with known_optimal_cost
            known_opt = metadata.get('known_optimal_cost')
            tol = 1e-6

            for name, r in runs.items():
                r_opt = False
                
                # Mark as optimal if:
                # 1. Algorithm explicitly returns optimal=True, OR
                # 2. Cost matches the known optimal cost
                if r.get('optimal', False):
                    r_opt = True
                elif known_opt is not None and r.get('feasible', False):
                    try:
                        if abs(r.get('cost', float('inf')) - float(known_opt)) < tol:
                            r_opt = True
                    except Exception:
                        pass
                
                r['optimal'] = bool(r_opt)

            # Print aligned table ordered by (costo, tiempo)
            header = f"\n    {'Algoritmo':<25} | {'Skip':^5} | {'Tiempo(s)':>10} | {'Ops':>8} | {'Costo':>12} | {'Factible':>8} | {'Óptimo':>6}"
            sep = '    ' + '-' * (len(header) - 4)

            def sort_key(item):
                _, r = item
                INF = float('inf')
                try:
                    cost_raw = r.get('cost', INF)
                    cost = float(cost_raw) if r.get('feasible') else INF
                except Exception:
                    cost = INF
                try:
                    timev = float(r.get('execution_time')) if (not r.get('skipped') and r.get('execution_time') is not None) else INF
                except Exception:
                    timev = INF
                opt_flag = 0 if r.get('optimal') else 1
                if cost != INF:
                    cost = round(cost, 6)
                return (cost, timev, opt_flag)

            ordered = sorted(runs.items(), key=sort_key)

            print(header)
            print(sep)
            for name, r in ordered:
                skipped = 'YES' if r.get('skipped') else 'NO'
                t = f"{r.get('execution_time', 0):.4f}" if not r.get('skipped') else '-'
                ops = str(r.get('operations')) if r.get('operations') is not None else 'N/A'
                costv = f"{r.get('cost', float('inf')):.2f}" if r.get('cost', None) is not None and r.get('cost') != float('inf') else '∞'
                feas = 'Yes' if r.get('feasible') else 'No'
                opt = 'Yes' if r.get('optimal') else 'No'
                line = f"    {name:<25} | {skipped:^5} | {t:>10} | {ops:>8} | {costv:>12} | {feas:>8} | {opt:>6}"
                print(line)

        if choice == 'all':
            for f in files:
                print_stats_file(f)
            return
        idx = int(choice)
        if 1 <= idx <= len(files):
            f = files[idx-1]
            print_stats_file(f)
        else:
            print(f"{RED}✗ Opción inválida{RESET}")
    except Exception as e:
        print(f"{RED}✗ Error: {e}{RESET}")

def main():
    """Función principal"""
    print_banner()
    
    while True:
        print_menu()
        
        try:
            choice = input(f"{BOLD}Selecciona una opción: {RESET}").strip()

            if choice == '0':
                print(f"\n{GREEN}👋 ¡Hasta luego!{RESET}")
                break

            elif choice == '1':
                generate_test_instances()

            elif choice == '2':
                list_instances()

            elif choice == '3':
                list_algorithms()

            elif choice == '4':
                # Procesamiento completo: todos los algoritmos vs todas las instancias
                instances = list_instances()
                if not instances:
                    continue
                algorithms = list(ALGORITHMS.keys())
                run_processing(instances, algorithms)

            elif choice == '5':
                instances = select_multiple_instances()
                if not instances:
                    continue
                algorithms = select_multiple_algorithms()
                if not algorithms:
                    continue
                run_processing(instances, algorithms)

            elif choice == '6':
                view_statistics()

            elif choice == '7':
                toggle_execution_mode()

            else:
                print(f"{YELLOW}⚠ Opción no implementada{RESET}")
        
        except KeyboardInterrupt:
            print(f"\n{YELLOW}⚠ Interrupción detectada{RESET}")
            break
        except Exception as e:
            print(f"{RED}✗ Error: {e}{RESET}")
        
        input(f"\n{CYAN}[Presiona ENTER para continuar]{RESET}")

if __name__ == "__main__":
    main()