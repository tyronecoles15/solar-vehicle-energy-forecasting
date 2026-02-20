"""
Paired t-test to compare model performances.
Compares test metrics across different models: Linear Regression, Random Forest, LightGBM, and ANN.
"""

import json
import numpy as np
from scipy import stats
import pandas as pd

def load_model_results(filepath):
    """Load model results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_predictions(results):
    """Extract predictions from results."""
    y_true = np.array(results['test_predictions']['y_true'])
    lr_pred = np.array(results['test_predictions']['lr_pred'])
    rf_pred = np.array(results['test_predictions']['rf_pred'])
    lgb_pred = np.array(results['test_predictions']['lgb_median_pred'])
    ann_pred = None  # Will compute from ANN model if needed
    
    return y_true, lr_pred, rf_pred, lgb_pred

def calculate_errors(y_true, predictions):
    """Calculate error metrics."""
    mae = np.abs(y_true - predictions).mean()
    rmse = np.sqrt(((y_true - predictions) ** 2).mean())
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    r2 = 1 - (np.sum((y_true - predictions) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'abs_errors': np.abs(y_true - predictions),
        'squared_errors': (y_true - predictions) ** 2
    }

def paired_ttest_compare(model1_errors, model2_errors, metric_name='abs_errors', alpha=0.05):
    """
    Perform paired t-test to compare two models.
    
    H0: The mean difference is zero (models perform equally)
    H1: The mean difference is not zero (models differ in performance)
    """
    diff = model1_errors[metric_name] - model2_errors[metric_name]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(model1_errors[metric_name], 
                                       model2_errors[metric_name])
    
    # Calculate effect size (Cohen's d for paired samples)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # Confidence interval for mean difference
    n = len(diff)
    se_diff = std_diff / np.sqrt(n)
    ci_lower = mean_diff - stats.t.ppf(1 - alpha/2, n-1) * se_diff
    ci_upper = mean_diff + stats.t.ppf(1 - alpha/2, n-1) * se_diff
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < alpha,
        'n_samples': n
    }

def print_ttest_results(model1_name, model2_name, results, metric_name, alpha=0.05):
    """Print paired t-test results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"Paired t-test: {model1_name} vs {model2_name}")
    print(f"Metric: {metric_name}")
    print(f"{'='*70}")
    print(f"T-statistic:     {results['t_statistic']:.6f}")
    print(f"P-value:         {results['p_value']:.6e}")
    print(f"Significant:     {'Yes' if results['significant'] else 'No'} (α={alpha})")
    print(f"Mean Difference: {results['mean_difference']:.8f}")
    print(f"Std Difference:  {results['std_difference']:.8f}")
    print(f"Cohen's d:       {results['cohens_d']:.6f}")
    print(f"95% CI:          [{results['ci_lower']:.8f}, {results['ci_upper']:.8f}]")
    print(f"Sample Size:     {results['n_samples']}")
    
    if results['significant']:
        if results['mean_difference'] > 0:
            print(f"\n✓ {model2_name} performs significantly better than {model1_name}")
        else:
            print(f"\n✓ {model1_name} performs significantly better than {model2_name}")
    else:
        print(f"\n✓ No significant difference between {model1_name} and {model2_name}")

def main():
    # Load results
    results_file = 'results/metrics/model_results.json'
    results = load_model_results(results_file)
    
    # Extract predictions and actual values
    y_true, lr_pred, rf_pred, lgb_pred = extract_predictions(results)
    
    # Calculate error metrics for each model
    lr_errors = calculate_errors(y_true, lr_pred)
    rf_errors = calculate_errors(y_true, rf_pred)
    lgb_errors = calculate_errors(y_true, lgb_pred)
    
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON - PAIRED T-TESTS")
    print("="*70)
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS (Test Set):")
    print("-" * 70)
    print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'MAPE':<12}")
    print("-" * 70)
    print(f"{'Linear Regression':<20} {lr_errors['MAE']:<12.6f} {np.sqrt(np.mean(lr_errors['squared_errors'])):<12.6f} "
          f"{1 - (np.sum(lr_errors['squared_errors']) / np.sum((y_true - np.mean(y_true))**2)):<12.6f} "
          f"{np.mean(np.abs((y_true - lr_pred) / y_true)) * 100:<12.4f}")
    print(f"{'Random Forest':<20} {rf_errors['MAE']:<12.6f} {np.sqrt(np.mean(rf_errors['squared_errors'])):<12.6f} "
          f"{1 - (np.sum(rf_errors['squared_errors']) / np.sum((y_true - np.mean(y_true))**2)):<12.6f} "
          f"{np.mean(np.abs((y_true - rf_pred) / y_true)) * 100:<12.4f}")
    print(f"{'LightGBM':<20} {lgb_errors['MAE']:<12.6f} {np.sqrt(np.mean(lgb_errors['squared_errors'])):<12.6f} "
          f"{1 - (np.sum(lgb_errors['squared_errors']) / np.sum((y_true - np.mean(y_true))**2)):<12.6f} "
          f"{np.mean(np.abs((y_true - lgb_pred) / y_true)) * 100:<12.4f}")
    
    # Perform pairwise comparisons on absolute errors (MAE proxy)
    print("\n\nPAIRED T-TESTS ON ABSOLUTE ERRORS:")
    
    # LR vs RF
    result_lr_rf = paired_ttest_compare(lr_errors, rf_errors, 'abs_errors')
    print_ttest_results("Linear Regression", "Random Forest", result_lr_rf, "Absolute Errors")
    
    # LR vs LGB
    result_lr_lgb = paired_ttest_compare(lr_errors, lgb_errors, 'abs_errors')
    print_ttest_results("Linear Regression", "LightGBM", result_lr_lgb, "Absolute Errors")
    
    # RF vs LGB
    result_rf_lgb = paired_ttest_compare(rf_errors, lgb_errors, 'abs_errors')
    print_ttest_results("Random Forest", "LightGBM", result_rf_lgb, "Absolute Errors")
    
    # Perform pairwise comparisons on squared errors (RMSE proxy)
    print("\n\nPAIRED T-TESTS ON SQUARED ERRORS:")
    
    # LR vs RF
    result_lr_rf_sq = paired_ttest_compare(lr_errors, rf_errors, 'squared_errors')
    print_ttest_results("Linear Regression", "Random Forest", result_lr_rf_sq, "Squared Errors")
    
    # LR vs LGB
    result_lr_lgb_sq = paired_ttest_compare(lr_errors, lgb_errors, 'squared_errors')
    print_ttest_results("Linear Regression", "LightGBM", result_lr_lgb_sq, "Squared Errors")
    
    # RF vs LGB
    result_rf_lgb_sq = paired_ttest_compare(rf_errors, lgb_errors, 'squared_errors')
    print_ttest_results("Random Forest", "LightGBM", result_rf_lgb_sq, "Squared Errors")
    
    # Create summary dataframe
    print("\n\n" + "="*70)
    print("SUMMARY TABLE - ALL PAIRWISE COMPARISONS")
    print("="*70)
    
    summary_data = {
        'Comparison': [
            'LR vs RF (MAE)',
            'LR vs LGB (MAE)',
            'RF vs LGB (MAE)',
            'LR vs RF (RMSE)',
            'LR vs LGB (RMSE)',
            'RF vs LGB (RMSE)'
        ],
        'T-Statistic': [
            result_lr_rf['t_statistic'],
            result_lr_lgb['t_statistic'],
            result_rf_lgb['t_statistic'],
            result_lr_rf_sq['t_statistic'],
            result_lr_lgb_sq['t_statistic'],
            result_rf_lgb_sq['t_statistic']
        ],
        'P-Value': [
            result_lr_rf['p_value'],
            result_lr_lgb['p_value'],
            result_rf_lgb['p_value'],
            result_lr_rf_sq['p_value'],
            result_lr_lgb_sq['p_value'],
            result_rf_lgb_sq['p_value']
        ],
        'Significant': [
            'Yes' if result_lr_rf['significant'] else 'No',
            'Yes' if result_lr_lgb['significant'] else 'No',
            'Yes' if result_rf_lgb['significant'] else 'No',
            'Yes' if result_lr_rf_sq['significant'] else 'No',
            'Yes' if result_lr_lgb_sq['significant'] else 'No',
            'Yes' if result_rf_lgb_sq['significant'] else 'No'
        ],
        "Cohen's d": [
            result_lr_rf['cohens_d'],
            result_lr_lgb['cohens_d'],
            result_rf_lgb['cohens_d'],
            result_lr_rf_sq['cohens_d'],
            result_lr_lgb_sq['cohens_d'],
            result_rf_lgb_sq['cohens_d']
        ],
        'Mean Diff': [
            result_lr_rf['mean_difference'],
            result_lr_lgb['mean_difference'],
            result_rf_lgb['mean_difference'],
            result_lr_rf_sq['mean_difference'],
            result_lr_lgb_sq['mean_difference'],
            result_rf_lgb_sq['mean_difference']
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Save results to file
    output_file = 'results/paired_ttest_results.csv'
    df_summary.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to {output_file}")
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("• P-value < 0.05: Significant difference between models")
    print("• P-value ≥ 0.05: No significant difference between models")
    print("• Cohen's d: Effect size")
    print("  - |d| < 0.2: Small effect")
    print("  - 0.2 ≤ |d| < 0.5: Small to medium effect")
    print("  - 0.5 ≤ |d| < 0.8: Medium to large effect")
    print("  - |d| ≥ 0.8: Large effect")
    print("="*70)

if __name__ == '__main__':
    main()
