"""
Master Execution Script
Runs the complete research pipeline from data collection to report generation
"""

import sys
import subprocess
import time
from datetime import datetime


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def run_script(script_path, description):
    """
    Run a Python script and handle errors
    
    Parameters:
    -----------
    script_path : str
        Path to the script to run
    description : str
        Description of what the script does
    """
    print_section(f"STEP: {description}")
    print(f"Running: {script_path}")
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} completed successfully")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error running {script_path}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def main():
    """
    Main execution pipeline
    """
    print("\n" + "="*80)
    print(" " * 15 + "SOLAR VEHICLE AI RESEARCH - COMPLETE PIPELINE")
    print(" " * 20 + "Tyrone Coles - 578013")
    print("="*80)
    print(f"\nPipeline started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis will run the complete research pipeline:")
    print("  1. Data Collection from NASA POWER")
    print("  2. Data Preprocessing")
    print("  3. Model Training & Evaluation")
    print("  4. SimPy Simulation")
    print("  5. Report Generation")
    print("\n" + "="*80)
    
    # Track overall progress
    total_start_time = time.time()
    steps_completed = 0
    total_steps = 5
    
    # Pipeline steps
    steps = [
        ("src/data_collection.py", "Data Collection from NASA POWER"),
        ("src/data_preprocessing.py", "Data Preprocessing & Feature Engineering"),
        ("src/model_training.py", "Model Training & Evaluation"),
        ("src/simulation.py", "SimPy Solar Vehicle Simulation"),
        ("src/generate_report.py", "Research Findings Report Generation")
    ]
    
    # Execute each step
    for script_path, description in steps:
        success = run_script(script_path, description)
        
        if success:
            steps_completed += 1
            print(f"\nProgress: {steps_completed}/{total_steps} steps completed")
        else:
            print(f"\n{'='*80}")
            print(" " * 25 + "PIPELINE FAILED")
            print("="*80)
            print(f"\nFailed at step {steps_completed + 1}: {description}")
            print(f"Please check the error messages above and fix the issue.")
            print(f"You can resume by running individual scripts starting from: {script_path}")
            sys.exit(1)
        
        # Small delay between steps
        time.sleep(1)
    
    # Pipeline completed successfully
    total_elapsed = time.time() - total_start_time
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    
    print("\n" + "="*80)
    print(" " * 25 + "PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nTotal execution time: {minutes} minutes {seconds} seconds")
    print(f"All {total_steps} steps completed successfully")
    
    print("\n" + "="*80)
    print("GENERATED FILES:")
    print("="*80)
    print("\nData Files:")
    print("  • data/raw/nasa_power_johannesburg_raw.csv")
    print("  • data/processed/train_data.csv")
    print("  • data/processed/val_data.csv")
    print("  • data/processed/test_data.csv")
    
    print("\nModels:")
    print("  • models/linear_regression_model.pkl")
    print("  • models/random_forest_model.pkl")
    print("  • models/feature_columns.json")
    
    print("\nResults & Metrics:")
    print("  • results/metrics/model_results.json")
    print("  • results/metrics/simulation_results.json")
    print("  • results/model_evaluation_summary.txt")
    print("  • results/simulation_summary.txt")
    print("  • results/research_findings_report.txt  ← MAIN REPORT FOR SUPERVISOR")
    
    print("\nVisualizations:")
    print("  • results/figures/data_visualization.png")
    print("  • results/figures/correlation_matrix.png")
    print("  • results/figures/predictions_timeseries.png")
    print("  • results/figures/scatter_plots.png")
    print("  • results/figures/model_comparison.png")
    print("  • results/figures/feature_importance.png")
    print("  • results/figures/simulation_battery_levels.png")
    print("  • results/figures/simulation_energy_flow.png")
    print("  • results/figures/simulation_scenario_comparison.png")
    print("  • results/figures/simulation_cumulative_energy.png")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("\n1. Review the main report: results/research_findings_report.txt")
    print("2. Examine all visualizations in: results/figures/")
    print("3. Check detailed metrics in: results/metrics/")
    print("4. Submit to supervisor by December 21st, 2024")
    
    print("\n" + "="*80)
    print("GIT COMMANDS TO COMMIT YOUR WORK:")
    print("="*80)
    print("\ngit add .")
    print('git commit -m "Complete research pipeline: data collection, model training, simulation, and results"')
    print("git push origin development")
    
    print("\n" + "="*80)
    print("✓ ALL TASKS COMPLETE - READY FOR SUBMISSION!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error in pipeline: {e}")
        sys.exit(1)