#!/usr/bin/env python3
"""
Quick script to regenerate predictions_timeseries figure with error spike highlighting
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Load the results JSON
results_file = 'results/metrics/model_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Reconstruct test dates
test_data = pd.read_csv('data/processed/test_data.csv')
test_dates = pd.to_datetime(test_data['date'])
dates = range(len(test_dates))

# Extract predictions and actual values
y_true = np.array(results['test_predictions']['y_true'])
lr_pred = np.array(results['test_predictions']['linear_regression_pred'])
rf_pred = np.array(results['test_predictions']['random_forest_pred'])

# Get other model predictions if available
lgb_median = None
lgb_lower = None
lgb_upper = None
if 'lgb_quantile_preds' in results['test_predictions']:
    q_preds = results['test_predictions']['lgb_quantile_preds']
    if '0.5' in q_preds:
        lgb_median = np.array(q_preds['0.5'])
    if '0.1' in q_preds and '0.9' in q_preds:
        lgb_lower = np.array(q_preds['0.1'])
        lgb_upper = np.array(q_preds['0.9'])

ann_pred = None
if 'ann_pred' in results['test_predictions']:
    ann_pred = np.array(results['test_predictions']['ann_pred'])

# Figure 1: Predictions vs Actual (Time Series)
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Linear Regression
axes[0].plot(dates, y_true, label='Actual', color='black', linewidth=2, alpha=0.7)
axes[0].plot(dates, lr_pred, label='Linear Regression', color='blue', linewidth=1.5, alpha=0.8)
axes[0].set_title('Linear Regression: Predicted vs Actual GHI', fontsize=14, fontweight='bold')
axes[0].set_ylabel('GHI (MJ/m²/day)')
axes[0].legend(loc='upper right')
axes[0].grid(True, alpha=0.3)

# Random Forest - calculate errors and highlight spikes
rf_errors = np.abs(y_true - rf_pred)
error_threshold = np.mean(rf_errors) + 1.5 * np.std(rf_errors)
error_spikes = rf_errors > error_threshold

# Highlight error spike regions with vertical bands
for i in range(len(dates)):
    if error_spikes[i]:
        axes[1].axvspan(dates[i] - 0.5, dates[i] + 0.5, alpha=0.15, color='red')

axes[1].plot(dates, y_true, label='Actual', color='black', linewidth=2, alpha=0.7)
axes[1].plot(dates, rf_pred, label='Random Forest', color='green', linewidth=1.5, alpha=0.8)
if lgb_median is not None:
    axes[1].plot(dates, lgb_median, label='LightGBM (median)', color='#8e44ad', linewidth=1.5, alpha=0.9)
if ann_pred is not None:
    axes[1].plot(dates, ann_pred, label='ANN', color='#e67e22', linewidth=1.5, alpha=0.9)
if lgb_lower is not None and lgb_upper is not None:
    axes[1].fill_between(dates, lgb_lower, lgb_upper, color='#8e44ad', alpha=0.15, label='LightGBM 0.1-0.9 quantile')

# Add custom legend entry for error spikes
from matplotlib.patches import Patch
error_patch = Patch(facecolor='red', alpha=0.15, label=f'Error Spikes (>{error_threshold:.3f} MJ/m²/day)')
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles + [error_patch], labels + [error_patch.get_label()], loc='upper right')

axes[1].set_title('Random Forest, LightGBM & ANN: Predicted vs Actual GHI (with Error Spikes)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('GHI (MJ/m²/day)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/predictions_timeseries.png', dpi=300, bbox_inches='tight')
print("✓ Regenerated: predictions_timeseries.png with error spike highlighting")
plt.close()
