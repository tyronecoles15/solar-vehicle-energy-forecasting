"""
Generate data summary table for report appendix
Shows the seven primary variables with mean, min, and max over the collection period
Uses RAW (pre-normalized) data to show actual meteorological values
"""

import pandas as pd
import json

# Load the RAW data (before normalization)
df = pd.read_csv('data/raw/nasa_power_johannesburg_raw.csv')

# Select the seven primary variables
primary_variables = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative_Humidity', 'Cloud_Coverage', 'Wind_Speed']

# Define units for each variable
units = {
    'GHI': 'kWh/m²/day',
    'DNI': 'kWh/m²/day',
    'DHI': 'kWh/m²/day',
    'Temperature': '°C',
    'Relative_Humidity': '%',
    'Cloud_Coverage': 'Oktas',
    'Wind_Speed': 'm/s'
}

# Create summary table
summary_data = []
for var in primary_variables:
    summary_data.append({
        'Variable': var,
        'Unit': units[var],
        'Mean': df[var].mean(),
        'Minimum': df[var].min(),
        'Maximum': df[var].max(),
        'Std Dev': df[var].std()
    })

df_summary = pd.DataFrame(summary_data)

# Print formatted table for report
print("\n" + "="*100)
print("APPENDIX A: DATA SUMMARY STATISTICS")
print("="*100)
print(f"\nCollection Period: {df['date'].min()} to {df['date'].max()}")
print(f"Total Records: {len(df):,}")
print(f"Location: Johannesburg, South Africa\n")

print(df_summary.to_string(index=False))

# Export to CSV for report inclusion
df_summary.to_csv('results/data_summary_table.csv', index=False)

# Also create a markdown-formatted version
print("\n\n" + "="*100)
print("MARKDOWN FORMAT FOR REPORT")
print("="*100)
print("\n| Variable | Unit | Mean | Minimum | Maximum | Std Dev |")
print("|----------|------|------|---------|---------|---------|")
for _, row in df_summary.iterrows():
    print(f"| {row['Variable']} | {row['Unit']} | {row['Mean']:.4f} | {row['Minimum']:.4f} | {row['Maximum']:.4f} | {row['Std Dev']:.4f} |")

# Also create LaTeX format for report
print("\n\n" + "="*100)
print("LATEX FORMAT FOR REPORT")
print("="*100)
print("\n\\begin{table}[h]")
print("\\centering")
print("\\caption{Data Summary Statistics for Solar Irradiance and Meteorological Variables}")
print("\\label{tab:data-summary}")
print("\\begin{tabular}{lcccccc}")
print("\\hline")
print("Variable & Unit & Mean & Minimum & Maximum & Std Dev \\\\")
print("\\hline")
for _, row in df_summary.iterrows():
    print(f"{row['Variable']} & {row['Unit']} & {row['Mean']:.4f} & {row['Minimum']:.4f} & {row['Maximum']:.4f} & {row['Std Dev']:.4f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n✓ Summary table saved to results/data_summary_table.csv")
