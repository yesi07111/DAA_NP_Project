#!/usr/bin/env python3
"""
Benchmark para medir rendimiento de brute_force_solver con diferentes tamaños.
Encuentra el máximo n donde brute_force termina en ~180 segundos.
"""
import sys
import time
from pathlib import Path
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from utils.timeout_handler import set_global_timeout, reset_global_timeout
from algorithms.exacts_algorithms import brute_force_solver

def create_random_graph(n_vertices, density=0.3, seed=42):
    """Crear grafo aleatorio para testing"""
    np.random.seed(seed)
    graph = nx.erdos_renyi_graph(n_vertices, density, seed=seed)
    
    # Crear matriz de costos simple (costos uniformes para simplificar)
    n_colors = min(n_vertices, 10)  # Máximo 10 colores
    cost_matrix = np.random.uniform(1, 100, (n_vertices, n_colors))
    
    return graph, cost_matrix

def benchmark_brute_force():
    """Ejecuta benchmark de brute_force con tamaños crecientes"""
    
    print("=" * 90)
    print("BENCHMARK: brute_force_solver - Encontrar n_max seguro")
    print("=" * 90)
    print("\nProbando tamaños crecientes. Timeout global: 180 segundos\n")
    
    results = []
    timeout_limit = 180.0  # 3 minutos
    
    # Probar tamaños de 4 a 15
    for n in range(4, 16):
        density = max(0.2, 1.0 - (n - 4) * 0.08)  # Densidad decreciente para grafos más grandes
        
        print(f"Testing n={n:2d}...", end=" ", flush=True)
        
        graph, cost_matrix = create_random_graph(n, density=density, seed=42 + n)
        
        # Establecer timeout
        set_global_timeout(timeout_limit)
        start_time = time.time()
        
        try:
            result = brute_force_solver(graph, cost_matrix)
            elapsed = time.time() - start_time
            
            ops = result.get('operations', 0)
            cost = result.get('cost', float('inf'))
            feasible = result.get('feasible', False)
            
            status = "✓" if feasible else "✗"
            
            print(f"Time: {elapsed:7.4f}s | Ops: {ops:>12,} | Cost: {cost:>10.2f} | {status}")
            
            results.append({
                'n': n,
                'time': elapsed,
                'ops': ops,
                'cost': cost,
                'feasible': feasible,
                'density': density,
                'timeout': False
            })
            
            if elapsed > timeout_limit * 0.9:
                print(f"  ⚠ WARNING: Close to timeout limit! ({elapsed:.1f}s > {timeout_limit * 0.9:.1f}s)")
                
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Time: {elapsed:7.4f}s | ERROR: {str(e)[:40]}")
            
            results.append({
                'n': n,
                'time': elapsed,
                'ops': 0,
                'cost': float('inf'),
                'feasible': False,
                'density': density,
                'timeout': True,
                'error': str(e)
            })
        finally:
            reset_global_timeout()
    
    # Análisis de resultados
    print("\n" + "=" * 90)
    print("ANÁLISIS DE RESULTADOS")
    print("=" * 90)
    
    successful = [r for r in results if r['feasible']]
    
    if successful:
        max_successful = max(successful, key=lambda x: x['n'])
        print(f"\n✓ Máximo n exitoso: {max_successful['n']}")
        print(f"  Tiempo: {max_successful['time']:.4f}s")
        print(f"  Operaciones: {max_successful['ops']:,}")
        print(f"  Densidad: {max_successful['density']:.3f}")
        
        # Recomendar n_max seguro (90% del máximo exitoso)
        recommended_n_max = max_successful['n']
        if recommended_n_max > 4:
            # Ser conservador
            if max_successful['time'] > 90:  # Si toma más de 90s
                recommended_n_max -= 1
        
        print(f"\n📋 RECOMENDACIÓN:")
        print(f"  n_max para brute_force = {recommended_n_max}")
        print(f"  (Ejecutará siempre brute_force para n ≤ {recommended_n_max})")
        
        # Estimación de complejidad
        if len(successful) >= 2:
            times_by_n = [(r['n'], r['time']) for r in successful[-3:]]
            print(f"\n📊 Últimas 3 ejecuciones:")
            for n, t in times_by_n:
                print(f"  n={n}: {t:.4f}s")
    
    print("\n" + "=" * 90)
    
    # Tabla final
    print("\nTABLA COMPLETA:")
    print(f"{'n':>3} | {'Time(s)':>10} | {'Ops':>14} | {'Cost':>10} | {'Fact':>4} | {'Timeout':>7}")
    print("-" * 70)
    for r in results:
        time_str = f"{r['time']:.4f}"
        ops_str = f"{r['ops']:,}" if r['ops'] else "N/A"
        cost_str = f"{r['cost']:.2f}" if r['cost'] != float('inf') else "∞"
        fact_str = "✓" if r['feasible'] else "✗"
        timeout_str = "✓" if r['timeout'] else " "
        
        print(f"{r['n']:3d} | {time_str:>10} | {ops_str:>14} | {cost_str:>10} | {fact_str:>4} | {timeout_str:>7}")

if __name__ == "__main__":
    benchmark_brute_force()
