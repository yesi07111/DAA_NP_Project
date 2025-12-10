"""
Statistical analysis for DAA Project - MCCPP results
"""
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple
import pandas as pd

def perform_statistical_analysis(algorithm_results: Dict[str, List[float]], 
                               confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Perform statistical analysis on algorithm results for DAA Project - MCCPP
    
    Args:
        algorithm_results: dictionary of algorithm_name -> list of costs
        confidence_level: confidence level for intervals (0.95 for 95%)
    
    Returns:
        statistical analysis results
    """
    analysis = {}
    
    for algo, costs in algorithm_results.items():
        if not costs:
            continue
            
        n = len(costs)
        mean = np.mean(costs)
        std = np.std(costs, ddof=1)  # sample standard deviation
        sem = std / np.sqrt(n)  # standard error of the mean
        
        # Confidence interval
        ci = stats.t.interval(confidence_level, n-1, loc=mean, scale=sem)
        
        # Normality test
        _, normality_p = stats.normaltest(costs)
        
        analysis[algo] = {
            'n': n,
            'mean': mean,
            'std': std,
            'sem': sem,
            'confidence_interval': ci,
            'confidence_level': confidence_level,
            'normality_p': normality_p,
            'is_normal': normality_p > 0.05,  # common threshold
            'min': np.min(costs),
            'max': np.max(costs),
            'median': np.median(costs),
            'q1': np.percentile(costs, 25),
            'q3': np.percentile(costs, 75)
        }
    
    return analysis

def perform_hypothesis_testing(algorithm_pairs: List[Tuple[str, str]], 
                              algorithm_results: Dict[str, List[float]],
                              test_type: str = 't_test') -> Dict[str, Any]:
    """
    Perform hypothesis testing between algorithm pairs for DAA Project - MCCPP
    
    Args:
        algorithm_pairs: list of (algo1, algo2) tuples to compare
        algorithm_results: dictionary of algorithm results
        test_type: type of statistical test ('t_test', 'mannwhitney')
    
    Returns:
        hypothesis test results
    """
    tests = {}
    
    for algo1, algo2 in algorithm_pairs:
        if algo1 not in algorithm_results or algo2 not in algorithm_results:
            continue
            
        costs1 = algorithm_results[algo1]
        costs2 = algorithm_results[algo2]
        
        if test_type == 't_test':
            # Student's t-test for independent samples
            t_stat, p_value = stats.ttest_ind(costs1, costs2, equal_var=False)
            test_name = "Welch's t-test"
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(costs1, costs2, alternative='two-sided')
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Effect size
        mean1, mean2 = np.mean(costs1), np.mean(costs2)
        std1, std2 = np.std(costs1, ddof=1), np.std(costs2, ddof=1)
        n1, n2 = len(costs1), len(costs2)
        
        # Cohen's d
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0
        
        tests[f"{algo1}_vs_{algo2}"] = {
            'test': test_name,
            'statistic': t_stat if test_type == 't_test' else u_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohens_d,
            'mean_difference': mean1 - mean2,
            'sample_sizes': (n1, n2)
        }
    
    return tests

def calculate_correlation_analysis(results_df: pd.DataFrame, 
                                 metrics: List[str]) -> Dict[str, Any]:
    """
    Calculate correlations between different metrics for DAA Project - MCCPP
    
    Args:
        results_df: DataFrame containing results
        metrics: list of metric columns to analyze
    
    Returns:
        correlation analysis results
    """
    correlation_matrix = results_df[metrics].corr(method='pearson')
    spearman_matrix = results_df[metrics].corr(method='spearman')
    
    # Significant correlations (p < 0.05)
    significant_correlations = []
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            if i < j:  # Avoid duplicates and self-correlations
                pearson_corr, pearson_p = stats.pearsonr(results_df[metric1], results_df[metric2])
                spearman_corr, spearman_p = stats.spearmanr(results_df[metric1], results_df[metric2])
                
                if pearson_p < 0.05 or spearman_p < 0.05:
                    significant_correlations.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'pearson_correlation': pearson_corr,
                        'pearson_p_value': pearson_p,
                        'spearman_correlation': spearman_corr,
                        'spearman_p_value': spearman_p
                    })
    
    return {
        'pearson_correlation_matrix': correlation_matrix.to_dict(),
        'spearman_correlation_matrix': spearman_matrix.to_dict(),
        'significant_correlations': significant_correlations
    }

def perform_statistical_analysis_grouped(groups: Dict[str, Dict[str, Any]]):
    """
    Recibe:
        groups = {
            "trees": comparison_trees,
            "structural": comparison_structural,
            "general": comparison_all
        }

    Devuelve estadísticas por grupo.
    """
    grouped_stats = {}

    for group_name, comparison in groups.items():
        grouped_stats[group_name] = {}

        for algo, stats in comparison.items():
            costs = stats["costs"]

            if len(costs) < 2:
                continue

            n = len(costs)
            mean = np.mean(costs)
            std = np.std(costs, ddof=1)
            sem = std / np.sqrt(n)
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=sem)

            grouped_stats[group_name][algo] = {
                "n": n,
                "mean": mean,
                "std": std,
                "sem": sem,
                "ci": ci,
                "min": np.min(costs),
                "max": np.max(costs),
                "median": np.median(costs),
            }

    return grouped_stats

def perform_hypothesis_testing_grouped(groups: Dict[str, Dict[str, Any]]):
    """
    Compara algoritmos ENTRE grupos:

    - DP(tree) vs estructurales
    - estructurales vs generales
    - tree vs generales
    """

    pairs = [
        ("trees", "structural"),
        ("structural", "general"),
        ("trees", "general"),
    ]

    tests = {}

    for g1, g2 in pairs:
        tests[f"{g1}_vs_{g2}"] = {}

        for algo1, stats1 in groups[g1].items():
            for algo2, stats2 in groups[g2].items():
                c1, c2 = stats1["costs"], stats2["costs"]

                if len(c1) < 2 or len(c2) < 2:
                    continue

                t_stat, p = stats.ttest_ind(c1, c2, equal_var=False)

                tests[f"{g1}_vs_{g2}"][f"{algo1}_vs_{algo2}"] = {
                    "t_stat": t_stat,
                    "p_value": p,
                    "significant": p < 0.05,
                    "mean_diff": np.mean(c1) - np.mean(c2),
                }

    return tests

def generate_statistical_report_grouped(stats, tests, output_file):
    """
    Produce un SOLO archivo MD con secciones separadas por grupo.
    """
    with open(output_file, "w") as f:
        f.write("# Statistical Analysis by Groups\n\n")

        # ---- ANALYSIS PER GROUP ----
        for group, data in stats.items():
            f.write(f"## Group: {group}\n\n")

            for algo, s in data.items():
                f.write(f"### {algo}\n")
                f.write(f"- n = {s['n']}\n")
                f.write(f"- mean = {s['mean']:.3f}\n")
                f.write(f"- std = {s['std']:.3f}\n")
                f.write(f"- 95% CI = ({s['ci'][0]:.3f}, {s['ci'][1]:.3f})\n")
                f.write(f"- median = {s['median']:.3f}\n")
                f.write(f"- min-max = {s['min']:.3f} – {s['max']:.3f}\n\n")

        # ---- HYPOTHESIS TESTING ----
        f.write("\n# Cross-Group Hypothesis Tests\n\n")

        for pair_name, group_tests in tests.items():
            f.write(f"## {pair_name}\n\n")

            for cmp_name, result in group_tests.items():
                f.write(f"### {cmp_name}\n")
                f.write(f"- p-value: {result['p_value']:.4f}\n")
                f.write(f"- significant: {'YES' if result['significant'] else 'NO'}\n")
                f.write(f"- mean difference: {result['mean_diff']:.3f}\n\n")

def generate_statistical_report(statistical_analysis: Dict[str, Any],
                              hypothesis_tests: Dict[str, Any],
                              output_file: str = "statistical_report.md") -> None:
    """
    Generate a statistical report from analysis results for DAA Project - MCCPP
    
    Args:
        statistical_analysis: results from perform_statistical_analysis
        hypothesis_tests: results from perform_hypothesis_testing
        output_file: output markdown file path
    """
    with open(output_file, 'w') as f:
        f.write("# DAA Project - MCCPP Statistical Analysis Report\n\n")
        
        f.write("## Descriptive Statistics\n\n")
        for algo, stats in statistical_analysis.items():
            f.write(f"### {algo}\n")
            f.write(f"- Sample size: {stats['n']}\n")
            f.write(f"- Mean cost: {stats['mean']:.2f}\n")
            f.write(f"- Standard deviation: {stats['std']:.2f}\n")
            f.write(f"- 95% CI: ({stats['confidence_interval'][0]:.2f}, {stats['confidence_interval'][1]:.2f})\n")
            f.write(f"- Range: {stats['min']:.2f} - {stats['max']:.2f}\n")
            f.write(f"- Median: {stats['median']:.2f}\n")
            f.write(f"- IQR: {stats['q1']:.2f} - {stats['q3']:.2f}\n")
            f.write(f"- Normality test p-value: {stats['normality_p']:.4f}\n")
            f.write(f"- Normal distribution: {'Yes' if stats['is_normal'] else 'No'}\n\n")
        
        f.write("## Hypothesis Tests\n\n")
        f.write("| Comparison | Test | p-value | Significant | Effect Size |\n")
        f.write("|------------|------|---------|-------------|-------------|\n")
        
        for comparison, test in hypothesis_tests.items():
            f.write(f"| {comparison} | {test['test']} | {test['p_value']:.4f} | "
                   f"{'Yes' if test['significant'] else 'No'} | {test['effect_size']:.2f} |\n")
        