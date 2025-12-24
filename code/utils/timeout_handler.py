import time
from typing import Callable, Dict, Any, Optional


class TimeoutException(Exception):
    """Excepción lanzada cuando se excede el tiempo máximo permitido."""
    pass

class AlgorithmTimeout:
    """
    Manejador de timeout para algoritmos.
    Ejecuta una función con un límite de tiempo máximo.
    """
    
    def __init__(self, timeout_seconds: float = 180.0):
        """
        Inicializa el manejador de timeout.
        
        Args:
            timeout_seconds: Tiempo máximo permitido en segundos (default: 180s = 3 minutos)
        """
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.result = None
        self.exception = None
    
    def start_timer(self):
        """Inicia el cronómetro."""
        self.start_time = time.time()
    
    def check_timeout(self):
        """
        Verifica si se ha excedido el tiempo límite.
        
        Returns:
            bool: True si el tiempo se ha excedido, False en caso contrario
            
        Raises:
            TimeoutException: Si se ha excedido el tiempo límite
        """
        if self.start_time is None:
            return False
        
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            raise TimeoutException(
                f"Se excedió el tiempo máximo de {self.timeout_seconds:.1f} segundos "
                f"(tiempo actual: {elapsed:.1f}s)"
            )
        return False
    
    def get_elapsed_time(self) -> float:
        """
        Retorna el tiempo transcurrido desde que se inició el timer.
        
        Returns:
            float: Tiempo transcurrido en segundos
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

def execute_with_timeout(
    func: Callable,
    graph: Any,
    cost_matrix: Any,
    timeout_seconds: float = 180.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Ejecuta una función de algoritmo con un límite de tiempo.
    
    Si la función no termina en el tiempo especificado, intenta recuperar
    el mejor resultado parcial si está disponible.
    
    Args:
        func: Función del algoritmo a ejecutar
        graph: Grafo de entrada
        cost_matrix: Matriz de costos
        timeout_seconds: Tiempo máximo permitido en segundos (default: 180s)
        **kwargs: Argumentos adicionales para la función
    
    Returns:
        Dict con los resultados de la ejecución, incluyendo información
        de timeout si aplica
    """
    timer = AlgorithmTimeout(timeout_seconds)
    timer.start_timer()
    
    try:
        # Ejecutar la función normalmente
        result = func(graph, cost_matrix, **kwargs)
        
        # Verificar que el resultado sea válido
        if not isinstance(result, dict):
            result = {'cost': float('inf'), 'feasible': False}
        
        # Agregar información de timeout
        result['timeout_exceeded'] = False
        result['max_timeout'] = timeout_seconds
        
        return result
        
    except TimeoutException as e:
        # El algoritmo excedió el tiempo límite
        return {
            'cost': float('inf'),
            'feasible': False,
            'solution': None,
            'coloring': None,
            'operations': 0,
            'execution_time': timer.get_elapsed_time(),
            'timeout_exceeded': True,
            'timeout_message': str(e),
            'max_timeout': timeout_seconds,
        }
    
    except Exception as e:
        # Cualquier otro error
        return {
            'cost': float('inf'),
            'feasible': False,
            'solution': None,
            'coloring': None,
            'operations': 0,
            'execution_time': timer.get_elapsed_time(),
            'error': str(e),
        }

# Global timeout handler para uso en algoritmos
_global_timeout: Optional[AlgorithmTimeout] = None
def set_global_timeout(timeout_seconds: float = 180.0):
    """Establece un timeout global para todos los algoritmos."""
    global _global_timeout
    _global_timeout = AlgorithmTimeout(timeout_seconds)
    _global_timeout.start_timer()

def get_global_timeout() -> Optional[AlgorithmTimeout]:
    """Retorna el timeout global actual."""
    return _global_timeout

def reset_global_timeout():
    """Reinicia el timeout global."""
    global _global_timeout
    _global_timeout = None

def check_global_timeout():
    """Verifica el timeout global. Levanta excepción si se excedió."""
    global _global_timeout
    if _global_timeout is not None:
        _global_timeout.check_timeout()
