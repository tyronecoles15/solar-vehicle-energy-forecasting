#!/usr/bin/env python3
"""
Quick script to regenerate model_comparison figure with updated labels
"""
import json
import matplotlib.pyplot as plt
import os

# Load the results JSON
results_file = 'results/metrics/model_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Extract metrics
models = []
mae_list = []
rmse_list = []
r2_list = []
mape_list = []

# Linear Regression
if 'linear_regression' in results:
    models.append('Linear Regression')
    mae_list.append(results['linear_regression']['test_metrics']['MAE'])
    rmse_list.append(results['linear_regression']['test_metrics']['RMSE'])
    r2_list.append(results['linear_regression']['test_metrics']['R2'])
    mape_list.append(results['linear_regression']['test_metrics']['MAPE'])

# Random Forest
if 'random_forest' in results:
    models.append('Random Forest')
    mae_list.append(results['random_forest']['test_metrics']['MAE'])
    rmse_list.append(results['random_forest']['test_metrics']['RMSE'])
    r2_list.append(results['random_forest']['test_metrics']['R2'])
    mape_list.append(results['random_forest']['test_metrics']['MAPE'])

# LightGBM
if 'lightgbm_quantile' in results:
    models.append('LightGBM')
    mae_list.append(results['lightgbm_quantile']['test_metrics_median']['MAE'])
    rmse_list.append(results['lightgbm_quantile']['test_metrics_median']['RMSE'])
    r2_list.append(results['lightgbm_quantile']['test_metrics_median']['R2'])
    mape_list.append(results['lightgbm_quantile']['test_metrics_median']['MAPE'])

# ANN
if 'ann' in results:
    models.append('ANN')
    mae_list.append(results['ann']['test_metrics']['MAE'])
    rmse_list.append(results['ann']['test_metrics']['RMSE'])
    r2_list.append(results['ann']['test_metrics']['R2'])
    mape_list.append(results['ann']['test_metrics']['MAPE'])

metrics_data = {'MAE': mae_list, 'RMSE': rmse_list, 'R2': r2_list, 'MAPE': mape_list}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

colors = ['#3498db', '#2ecc71', '#8e44ad', '#e67e22', '#f39c12'][:len(models)]

# MAE
axes[0, 0].bar(models, metrics_data['MAE'], color=colors)
axes[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('MAE (MJ/m²/day)', fontsize=12)
axes[0, 0].axhline(y=0.12, color='r', linestyle='--', label='Target: 0.12')
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)

# RMSE
axes[0, 1].bar(models, metrics_data['RMSE'], color=colors)
axes[0, 1].set_title('Root Mean Square Error (RMSE)', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('RMSE (MJ/m²/day)', fontsize=12)
axes[0, 1].axhline(y=0.15, color='r', linestyle='--', label='Target: 0.15')
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)

# R²
axes[1, 0].bar(models, metrics_data['R2'], color=colors)
axes[1, 0].set_title('R² Score', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('R²', fontsize=12)
axes[1, 0].axhline(y=0.85, color='r', linestyle='--', label='Target: 0.85')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)

# MAPE
axes[1, 1].bar(models, metrics_data['MAPE'], color=colors)
axes[1, 1].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('%', fontsize=12)
axes[1, 1].axhline(y=12, color='r', linestyle='--', label='Target: 12%')
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Regenerated: model_comparison.png with corrected labels")
plt.close()
