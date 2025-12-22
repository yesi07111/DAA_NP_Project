# Proyecto DAA - MCCPP (Minimum Cost Chromatic Partition Problem)

## Descripción del Proyecto

Este proyecto implementa soluciones para el **Problema de Partición Cromática de Costo Mínimo (MCCPP)**, un problema de optimización combinatoria que consiste en asignar colores a los vértices de un grafo de manera que vértices adyacentes tengan colores diferentes, minimizando el costo total de la coloración.

### Aplicaciones Prácticas
- **Diseño VLSI**: Asignación de frecuencias en circuitos integrados
- **Programación de Tareas**: Asignación de recursos con costos diferenciados
- **Asignación de Registros**: Optimización en compiladores
- **Redes de Comunicación**: Asignación de frecuencias sin interferencias

## 🚀 Instalación y Configuración

### Prerrequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/yesi07111/DAA_NP_Project
cd DAA_Project
```

2. **Crear entorno virtual (recomendado)**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instalar dependencias**:
```bash
pip install -r code/requirements.txt
```

### Dependencias Principales
- `networkx`: Manipulación y análisis de grafos
- `numpy`: Cálculos numéricos y matrices
- `matplotlib`: Visualización de resultados
- `scipy`: Análisis estadístico
- `pandas`: Procesamiento de datos

## 📁 Estructura del Proyecto

```
DAA_Project/
├── code/
│   ├── src/
│   │   ├── instances/
│   │   │   ├── generators.py          # Generadores de instancias ER y estructuradas
│   │   │   ├── special_cases.py       # Instancias especiales con óptimos conocidos
│   │   │   └── interval_graphs.py     # Instancias de grafos de intervalo
│   │   ├── algorithms/
│   │   │   ├── exact/                 # Algoritmos exactos
│   │   │   ├── heuristic/            # Algoritmos heurísticos
│   │   │   ├── approximation/        # Algoritmos de aproximación
│   │   │   └── metaheuristic/        # Algoritmos metaheurísticos
│   │   ├── evaluation/
│   │   │   ├── benchmarks.py         # Suite de benchmarks
│   │   │   ├── scalability_tests.py  # Pruebas de escalabilidad
│   │   │   └── statistical_analysis.py # Análisis estadístico
│   │   └── utils/
│   │       ├── graph_utils.py        # Utilidades para grafos
│   │       ├── cost_utils.py         # Generación de matrices de costo
│   │       └── io_utils.py           # Manejo de archivos
│   ├── instances/                    # Instancias generadas
│   │   ├── benchmarks/              # Benchmarks académicos
│   │   └── interval_graphs/         # Instancias de intervalo
│   ├── experiment_results/          # Resultados de experimentos
|   |
|   ├── run_experiments.py           # Punto de entrada para solo correr los experimentos
|   ├── main.py                      # Punto de entrada principal
│   └── requirements.txt
|
├── latex/
│   └── informe.tex                  # Código LaTeX del informe
└── informe.pdf                      # Informe final del proyecto
```

## 📊 Flujo de Trabajo

### Diagrama del Proceso

```
Generación de Instancias
         ↓
   Ejecución de Algoritmos
         ↓
  Evaluación de Resultados
         ↓
   Análisis Estadístico
         ↓
  Generación de Reportes
```

### Pasos Detallados

1. **Fase de Preparación**:
   - Generar instancias de prueba
   - Configurar parámetros experimentales
   - Preparar estructuras de datos

2. **Fase de Ejecución**:
   - Ejecutar algoritmos sobre instancias
   - Medir tiempos de ejecución
   - Registrar calidad de soluciones

3. **Fase de Análisis**:
   - Comparar algoritmos entre sí
   - Analizar escalabilidad
   - Realizar pruebas estadísticas

4. **Fase de Reporte**:
   - Generar tablas comparativas
   - Crear visualizaciones
   - Documentar hallazgos

## 📈 Tipos de Instancias Disponibles

### Instancias Especiales (20 tipos)
- **Caminos**: P3, P5
- **Ciclos**: C4, C6, C8 (pares), C5, C7 (impares)
- **Estrellas**: S4, S5, S8
- **Grafos Completos**: K3, K4
- **Bipartitos Completos**: K_{2,2}, K_{3,4}, K_{4,5}
- **Árboles Binarios**: Balanceado (7 vértices), Completo (15 vértices)
- **Grafos de Intervalo**: Simple (5 vértices), Complejo (7 vértices)

### Benchmarks Académicos (4 tipos)
- **Jansen Path** (1997): 6 vértices, 3 colores
- **Jansen Cycle** (1997): 10 vértices, 3 colores
- **DIMACS Style**: 10 vértices, 4 colores
- **Scheduling Application**: 8 vértices, 3 colores

## 🧮 Algoritmos Implementados

### Exactos
- `brute_force`: Búsqueda exhaustiva de todas las coloraciones válidas (para instancias pequeñas)
- `dp_interval_graphs`: Programación dinámica para grafos de intervalo (usando la estructura de intervalos)
- `ilp_solver`: Resolución mediante Programación Lineal Entera (usando PuLP)

### Heurísticas
- `largest_first`: Ordenamiento por grado descendente
- `dsatur`: Algoritmo DSATUR (Degree of SATURation)
- `rlf`: Algoritmo Recursive Largest First

### Algoritmos de Aproximación
- `weighted_set_cover`: Basado en cubiertas de conjuntos
- `structural_approximation`: Aproximaciones estructurales para:
  - **Grafos bipartitos**: Aprovecha la estructura 2-coloreable
  - **Grafos de intervalo**: Utiliza el ordenamiento temporal de intervalos
  - **Grafos generales**: Estrategia greedy mejorada con detección de propiedades

### Metaheurísticas
- `simulated_annealing`: Recocido simulado
- `trajectory_search`: Búsqueda por trayectorias

### Detalles de Algoritmos Especializados

**Aproximaciones Estructurales**:
- **Bipartitos**: Detecta particiones y asigna colores óptimos por conjunto
- **Intervalo**: Ordena por tiempo de finalización y asigna colores disponibles de menor costo
- **General**: Combina información de grado y varianza de costos

**ILP Solver**:
- Formula el problema como programa lineal entero
- Utiliza restricciones de adyacencia y asignación única
- Resuelve con solver CBC a través de PuLP

**Fuerza Bruta**:
- Genera todas las coloraciones posibles
- Filtra las válidas (vértices adyacentes con colores diferentes)
- Selecciona la de menor costo (garantiza optimalidad para instancias pequeñas)

**DP para Grafos de Intervalo**:
- Aprovecha la estructura lineal de los intervalos
- Algoritmo polinomial basado en ordenamiento temporal
- Garantiza optimalidad para esta clase de grafos
  
## 📋 Resultados y Reportes

El proyecto genera automáticamente:

1. **Reporte de Benchmarks**: Comparación de algoritmos en todas las instancias
2. **Análisis de Escalabilidad**: Comportamiento con instancias grandes
3. **Reporte Estadístico**: Pruebas de hipótesis y significancia
4. **Visualizaciones**: Gráficos de rendimiento y escalabilidad

### Ejemplo de Salida
```
========================================================================
EXPERIMENTOS COMPLETADOS - RESUMEN
========================================================================
Total de instancias: 24
Algoritmos evaluados: 8
Tiempo total de ejecución: 45 minutos

Mejores algoritmos por categoría:
- Instancias pequeñas: dsatur
- Instancias grandes: simulated_annealing
- Tiempo de ejecución: largest_first
```

---

**Nota**: Para más detalles sobre los algoritmos específicos o la teoría detrás del MCCPP, consultar el informe `informe.pdf` y la documentación en los archivos fuente.