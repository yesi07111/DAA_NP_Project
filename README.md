# MCCPP - Minimum Cost Chromatic Partition Problem

Proyecto de Análisis y Diseño de Algoritmos (DAA) - Universidad de La Habana

**Autores:**
- Yesenia Valdés Rodríguez (C411)
- Laura Martir Beltrán (C411)
- Adrián Hernández Castellanos (C412)

## Descripción del Proyecto

Este proyecto implementa y analiza múltiples algoritmos para resolver el problema MCCPP (Minimum Cost Chromatic Partition Problem), que busca encontrar una coloración de grafo que minimice el costo total, respetando las restricciones de coloración válida.

## Estructura del Proyecto

```
.
├── algorithms/                          # Implementaciones de algoritmos
│   ├── exacts_algorithms.py            # Algoritmos exactos (fuerza bruta, DP, ILP, etc.)
│   ├── approximation_algorithms.py     # Algoritmos de aproximación (WSC, etc.)
│   ├── heuristic_algorithms.py         # Heurísticas (LF, DSATUR, RLF, etc.)
│   └── metaheuristic_algorithms.py     # Metaheurísticas (SA, TS+PR, etc.)
├── instances/                           # Instancias de prueba (generadas)
├── results/                             # Resultados de ejecuciones
├── utils/                               # Utilidades
│   ├── timeout_handler.py              # Manejo de timeouts globales
│   ├── utils.py                        # Funciones auxiliares
│   ├── evaluation.py                   # Evaluación y comparación
│   └── instances_generator.py          # Generación de instancias
├── main.py                             # 
├── requirements.txt                    # Dependencias
└── README.md                           # Este archivo
```

## Instalación

### Requisitos
- Python 3.8+
- pip

### Pasos de Instalación

1. **Clonar o descargar el proyecto:**
   ```bash
   cd [ruta_a_la_carpeta_raiz]
   ```

2. **Crear un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # En Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## Uso

### Ejecutar el Menú Principal

```bash
python main.py
```

Se abrirá un menú interactivo con las siguientes opciones:

1. **Generar instancias de prueba**
   - Genera automáticamente un conjunto de 100+ instancias de diferentes tipos
   - Garantiza que todas sean factibles (χ(G) ≤ k)
   - Tipos incluidos: Completos, Ciclos, Grillas, Erdős-Rényi, Árboles, etc.

2. **Listar instancias disponibles**
   - Muestra todas las instancias generadas
   - Indica si son factibles y estima χ(G)

3. **Listar algoritmos disponibles**
   - Muestra todos los algoritmos implementados por categoría

4. **Ejecutar procesamiento completo (todos vs todos)**
   - Ejecuta todos los algoritmos en instancias seleccionadas
   - Respeta el modo de ejecución actual (ver opción 7)

5. **Ejecutar procesamiento personalizado**
   - Permite seleccionar instancias y algoritmos específicos
   - Mayor control sobre qué se ejecuta

6. **Visualizar estadísticas guardadas**
   - Genera gráficos y reportes comparativos

7. **Cambiar modo de ejecución**
   - **Modo "TODOS vs TODOS"** (defecto): Los 3 algoritmos brute force siempre se ejecutan
   - **Modo "SELECTIVO"**: Los algoritmos se eligen según el tamaño del grafo

0. **Salir**

## Categorías de Algoritmos

### Exactos
- **brute_force**: Enumeración completa de coloraciones (exhaustivo)
- **backtracking**: Backtracking con pruning básico
- **intelligent_backtracking**: Backtracking con poda inteligente
- **ilp_solver**: Programación Lineal Entera (para n ≤ 50)
- **dp_tree**: Programación Dinámica para árboles
- **dp_interval**: Programación Dinámica para grafos de intervalo

### Aproximación
- **wsc**: Weighted Set Cover greedy
- **wsc_improved**: WSC + búsqueda local
- **interval_approx**: Aproximación específica para grafos de intervalo

### Heurísticas
- **largest_first**: Largest First (coloración por grado descendente)
- **dsatur**: DSATUR (Degree of Saturation)
- **rlf**: Recursive Largest First

### Metaheurísticas
- **simulated_annealing**: Recocido Simulado
- **adaptive_sa**: Recocido Simulado Adaptivo
- **trajectory_search**: Búsqueda por Trayectoria + Path Relinking
- **hybrid**: Metaheurística Híbrida (LS + SA)
- **adaptive**: Metaheurística Adaptiva

## Características Principales

### Timeout Global
- Todos los algoritmos tienen un límite de tiempo de **180 segundos** (3 minutos)
- El timeout se verifica frecuentemente en bucles largos
- Garantiza que el programa nunca se cuelgue indefinidamente

### Generación de Instancias
- Crea instancias de diferentes tipos de grafos
- Asegura factibilidad (k ≥ χ(G))
- Tamaños variados (6 a 30 vértices)

### Evaluación Completa
- Verifica factibilidad de soluciones
- Calcula costo total
- Cuenta operaciones ejecutadas
- Mide tiempo de ejecución
- Detecta soluciones óptimas cuando es posible

### Salida Estandarizada
Todos los algoritmos retornan un diccionario con:
```python
{
    'solution': dict,           # Coloración encontrada {vértice: color}
    'cost': float,              # Costo total de la solución
    'execution_time': float,    # Tiempo de ejecución en segundos
    'operations': int,          # Número de operaciones realizadas
    'feasible': bool,           # ¿Es una coloración válida?
    'optimal': bool,            # ¿Es óptima? (cuando es conocida)
    'algorithm': str            # Nombre del algoritmo
}
```

## Modos de Ejecución

### Modo "TODOS vs TODOS" (predeterminado)
Los 3 algoritmos brute force se ejecutan siempre:
- brute_force
- backtracking  
- intelligent_backtracking

Con timeout de 180 segundos cada uno.

**Ventaja:** Permite comparar con soluciones exactas en instancias pequeñas
**Desventaja:** Lento para grafos grandes (n > 20)

### Modo "SELECTIVO"
Los algoritmos se eligen según el tamaño:
- **n ≤ 7**: Ejecutar brute_force
- **7 < n ≤ 12**: Ejecutar backtracking
- **12 < n ≤ 20**: Ejecutar intelligent_backtracking
- **n > 20**: Ejecutar solo heurísticas y metaheurísticas

**Ventaja:** Más rápido, escalable a grafos grandes
**Desventaja:** Menos comparativas con exactos en grafos medianos

## Ejemplos de Uso

### Ejemplo 1: Ejecutar todos los algoritmos en una instancia
```bash
python main.py
# Selecciona opción 4 o 5
# Elige una o varias instancias
# (Los algoritmos se ejecutan automáticamente)
```

### Ejemplo 2: Cambiar a modo selectivo
```bash
python main.py
# Selecciona opción 7 para cambiar modo
# Ahora los algoritmos se eligen automáticamente por tamaño
```

### Ejemplo 3: Generar nuevas instancias
```bash
python main.py
# Selecciona opción 1
# Se generan automáticamente 100+ instancias
```

## Parámetros Ajustables

En `main.py` puedes ajustar:

```python
# Línea ~160
EXECUTION_MODE = "all_brute"  # Cambiar a "selective" para modo selectivo
```

## Estructura de Resultados

Los resultados se guardan en `results/` en formato JSON:
```
results/
├── instance_name_results.json
├── comparison_stats.json
└── ...
```

Cada archivo contiene:
- Metadata de la instancia
- Resultados detallados de cada algoritmo
- Comparativas y estadísticas

## Interpretación de Resultados

### Costo
- Número flotante representando el costo total
- Menores valores = mejores soluciones
- ∞ = algoritmo no encontró solución

### Feasible (Factible)
- ✓ = Coloración válida (respeta restricciones)
- ✗ = Coloración inválida (hay conflictos)

### Optimal (Óptimo)
- ★ = Solución óptima (cuando es verificada)
- ○ = No óptima (o no verificada)

### Operaciones
- Número de operaciones básicas ejecutadas
- Permite analizar complejidad práctica
- Independiente del tiempo de CPU

## Solución de Problemas

### "Error: Instancia infactible"
- La instancia necesita más colores de los disponibles
- Generar nuevas instancias (opción 1 del menú)
- O aumentar k en la generación

### "Timeout (180s excedido)"
- El algoritmo exacto es demasiado lento para esta instancia
- Cambiar a modo selectivo (opción 7)
- O usar instancias más pequeñas

### "No hay instancias generadas"
- Ejecutar "Generar instancias de prueba" (opción 1)
- Asegurarse de que el directorio `instances/` existe

## Referencias

- Informe del Proyecto DAA - Secciones 1-4
- Papers de algoritmos de coloración de grafos
- NetworkX Documentation: https://networkx.org/
- PuLP Documentation: https://coin-or.github.io/pulp/
