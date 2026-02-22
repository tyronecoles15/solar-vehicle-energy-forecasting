#!/usr/bin/env python3
"""
Quick script to regenerate LightGBM uncertainty figure with improved colors
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the results JSON
results_file = 'results/metrics/model_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Check if we have all three model predictions
if 'test_predictions' in results and 'lgb_quantile_preds' in results['test_predictions']:
    q_preds = results['test_predictions']['lgb_quantile_preds']
    if all(k in q_preds for k in ('0.1', '0.5', '0.9')):
        y_true = np.array(results['test_predictions']['y_true'])
        dates = range(len(y_true))
        lgb_lower = np.array(q_preds['0.1'])
        lgb_median = np.array(q_preds['0.5'])
        lgb_upper = np.array(q_preds['0.9'])

        plt.figure(figsize=(15, 5))
        # Plot shaded band first (background), then lines on top
        plt.fill_between(dates, lgb_lower, lgb_upper, color='#e74c3c', alpha=0.25, label='LightGBM 0.1-0.9 quantile (80% prediction interval)')
        plt.plot(dates, lgb_median, label='LightGBM (median)', color='#27ae60', linewidth=2.5)
        plt.plot(dates, y_true, label='Actual', color='#f39c12', linewidth=2, alpha=0.9)
        
        plt.title('LightGBM Probabilistic Forecast (0.1 - 0.9 Quantiles)', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('GHI (MJ/m²/day)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/lightgbm_uncertainty.png', dpi=300, bbox_inches='tight')
        print("✓ Regenerated: lightgbm_uncertainty.png with high-contrast color scheme")
        print("  - Uncertainty band: Red (#e74c3c, alpha=0.25)")
        print("  - Median forecast: Green (#27ae60, linewidth=2.5)")
        print("  - Actual values: Orange (#f39c12, linewidth=2)")
        plt.close()
    else:
        print("✗ Missing required quantiles (0.1, 0.5, 0.9)")
else:
    print("✗ No LightGBM quantile predictions found in results")
