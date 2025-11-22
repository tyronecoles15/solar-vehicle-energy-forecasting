# Data Processing Documentation

## Overview
This document describes the data collection and preprocessing pipeline for the solar vehicle energy forecasting project.

## Data Source
- **Source**: NASA POWER Project
- **Location**: Johannesburg, South Africa (-26.2041, 28.0473)
- **Time Period**: 2014-2023 (10 years)
- **Frequency**: Daily measurements

## Variables Collected
1. **GHI** (Global Horizontal Irradiance) - Primary target variable
2. **DNI** (Direct Normal Irradiance)
3. **DHI** (Diffuse Horizontal Irradiance)
4. **Temperature** (2m above surface)
5. **Relative Humidity**
6. **Cloud Coverage**
7. **Wind Speed**

## Preprocessing Steps

### 1. Data Cleaning
- Missing value imputation using linear interpolation
- Outlier detection using z-scores (threshold: 3σ)
- Validation of solar irradiance values (removing negatives)

### 2. Feature Engineering
- **Temporal Features**: day, month, day_of_year, season
- **Lag Features**: Previous day's GHI, DNI, DHI
- **Moving Averages**: 3-day and 7-day for GHI and DNI

### 3. Normalization
- Min-Max scaling applied to all numeric features
- Range: [0, 1]

### 4. Data Splitting
- Training: 70% (2014-2020)
- Validation: 15% (2021-2022)
- Testing: 15% (2022-2023)

## Output Files
- `train_data.csv`: Training dataset
- `val_data.csv`: Validation dataset
- `test_data.csv`: Test dataset
- `processed_data_full.csv`: Complete processed dataset
- `preprocessing_report.json`: Detailed processing statistics

## Quality Metrics Achieved
- Missing values: < 0.1%
- Outliers handled: Yes
- Data integrity: Verified
- Total records: ~3,650 (10 years daily)