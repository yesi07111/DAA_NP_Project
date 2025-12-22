"""
Script to run all experiments for DAA Project - MCCPP
"""
import os
import sys
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from code.src.experiments.experiment_runner import run_comprehensive_experiments
from code.src.evaluation.comparison_plots import generate_comparison_plots
from code.src.experiments.parameter_tuning import tune_metaheuristic_parameters

def main():
    parser = argparse.ArgumentParser(description='Run DAA Project - MCCPP Experiments')
    parser.add_argument('--output-dir', '-o', default='experiment_results', 
                       help='Output directory for results')
    parser.add_argument('--time-limit', '-t', type=float, default=300.0,
                       help='Time limit per algorithm per instance (seconds)')
    parser.add_argument('--run-tuning', action='store_true',
                       help='Run parameter tuning (time-consuming)')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    # Run comprehensive experiments
    print("Starting comprehensive experiments...")
    results = run_comprehensive_experiments(
        output_dir=args.output_dir,
        time_limit=args.time_limit
    )
    
    # Run parameter tuning if requested
    if args.run_tuning:
        print("Starting parameter tuning...")
        tuning_results = tune_metaheuristic_parameters(
            output_dir=os.path.join(args.output_dir, "parameter_tuning")
        )
        results['parameter_tuning'] = tuning_results
    
    # Generate plots if requested
    if args.generate_plots:
        print("Generating comparison plots...")
        generate_comparison_plots(
            results,
            output_dir=os.path.join(args.output_dir, "plots")
        )
    
    print("All experiments completed!")

if __name__ == '__main__':
    main()