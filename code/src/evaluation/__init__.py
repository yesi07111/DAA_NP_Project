"""
Evaluation module for DAA Project - MCCPP
"""
from .benchmarks import *
from .empyric_analysis import *
from .statistical_analysis import *
from .scalability_tests import *

__all__ = [
    'run_benchmark_suite',
    'calculate_approximation_ratio',
    'perform_statistical_analysis',
    'run_scalability_test',
    'generate_comprehensive_report'
]