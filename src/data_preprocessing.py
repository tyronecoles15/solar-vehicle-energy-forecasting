"""
Data Preprocessing Pipeline
Implements the 4-step preprocessing methodology:
1. Data Cleaning
2. Feature Engineering
3. Normalization
4. Data Splitting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

class SolarDataPreprocessor:
    """
    Preprocesses solar irradiance and meteorological data
    """
    
    def __init__(self, input_file, output_dir='data/processed'):
        """
        Initialize preprocessor
        
        Parameters:
        -----------
        input_file : str
            Path to raw data CSV file
        output_dir : str
            Directory to save processed data
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.df = None
        self.df_original = None 
        self.scaler = MinMaxScaler()
        self.preprocessing_report = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self):
        """Load raw data from CSV"""
        print("="*60)
        print("STEP 1: LOADING DATA")
        print("="*60)
        
        self.df = pd.read_csv(self.input_file)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        print(f"✓ Loaded {len(self.df)} records")
        print(f"✓ Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"✓ Columns: {list(self.df.columns)}")
        
        self.preprocessing_report['original_records'] = len(self.df)
        self.preprocessing_report['original_columns'] = list(self.df.columns)
        
        return self.df
    
    def data_cleaning(self):
        """
        Step 1: Data Cleaning
        - Detect and handle missing values
        - Detect and handle outliers using z-scores
        - Perform data consistency checks
        """
        print("\n" + "="*60)
        print("STEP 2: DATA CLEANING")
        print("="*60)
        
        # Check for missing values
        print("\n1. Missing Values Analysis:")
        missing_before = self.df.isnull().sum()
        print(missing_before)
        
        self.preprocessing_report['missing_values_before'] = missing_before.to_dict()
        
        # Handle missing values using linear interpolation
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].interpolate(method='linear', limit_direction='both')
                print(f"  ✓ Interpolated {col}")
        
        missing_after = self.df.isnull().sum()
        print("\nMissing values after interpolation:")
        print(missing_after)
        
        # Detect outliers using z-scores
        print("\n2. Outlier Detection (z-score > 3):")
        outliers_detected = {}
        
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            outlier_indices = np.where(z_scores > 3)[0]
            outliers_detected[col] = len(outlier_indices)
            
            if len(outlier_indices) > 0:
                print(f"  {col}: {len(outlier_indices)} outliers detected")
                # Cap outliers at 3 standard deviations
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df.loc[self.df[col] > mean + 3*std, col] = mean + 3*std
                self.df.loc[self.df[col] < mean - 3*std, col] = mean - 3*std
        
        self.preprocessing_report['outliers_detected'] = outliers_detected
        
        # Check for invalid values (negative solar irradiance)
        print("\n3. Data Validation:")
        solar_cols = ['GHI', 'DNI', 'DHI']
        for col in solar_cols:
            negative_count = (self.df[col] < 0).sum()
            if negative_count > 0:
                print(f"  ✗ {col}: {negative_count} negative values found - setting to 0")
                self.df.loc[self.df[col] < 0, col] = 0
            else:
                print(f"  ✓ {col}: No invalid values")
        
        print(f"\n✓ Data cleaning complete")
        print(f"✓ Records retained: {len(self.df)}")
        
        self.preprocessing_report['records_after_cleaning'] = len(self.df)
        
    def feature_engineering(self):
        """
        Step 2: Feature Engineering
        - Temporal features (day, month, season)
        - Lag features (previous day's irradiance)
        - Moving averages (3-day and 7-day)
        """
        print("\n" + "="*60)
        print("STEP 3: FEATURE ENGINEERING")
        print("="*60)
        
        # Temporal features
        print("\n1. Creating Temporal Features:")
        self.df['day'] = self.df['date'].dt.day
        self.df['month'] = self.df['date'].dt.month
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        
        # Season (Southern Hemisphere)
        def get_season(month):
            if month in [12, 1, 2]:
                return 0  # Summer
            elif month in [3, 4, 5]:
                return 1  # Autumn
            elif month in [6, 7, 8]:
                return 2  # Winter
            else:
                return 3  # Spring
        
        self.df['season'] = self.df['month'].apply(get_season)
        print("  ✓ Day, Month, Day of Year, Season")
        
        # Lag features
        print("\n2. Creating Lag Features:")
        self.df['GHI_lag1'] = self.df['GHI'].shift(1)
        self.df['DNI_lag1'] = self.df['DNI'].shift(1)
        self.df['DHI_lag1'] = self.df['DHI'].shift(1)
        print("  ✓ 1-day lag for GHI, DNI, DHI")
        
        # Moving averages
        print("\n3. Creating Moving Averages:")
        self.df['GHI_ma3'] = self.df['GHI'].rolling(window=3, min_periods=1).mean()
        self.df['GHI_ma7'] = self.df['GHI'].rolling(window=7, min_periods=1).mean()
        self.df['DNI_ma3'] = self.df['DNI'].rolling(window=3, min_periods=1).mean()
        self.df['DNI_ma7'] = self.df['DNI'].rolling(window=7, min_periods=1).mean()
        print("  ✓ 3-day and 7-day moving averages for GHI and DNI")
        
        # Remove rows with NaN from lag features (first row)
        initial_length = len(self.df)
        self.df = self.df.dropna().reset_index(drop=True)
        print(f"\n✓ Feature engineering complete")
        print(f"✓ Records after removing NaN from lag features: {len(self.df)} (removed {initial_length - len(self.df)})")
        
        engineered_features = ['day', 'month', 'day_of_year', 'season', 
                               'GHI_lag1', 'DNI_lag1', 'DHI_lag1',
                               'GHI_ma3', 'GHI_ma7', 'DNI_ma3', 'DNI_ma7']
        self.preprocessing_report['engineered_features'] = engineered_features
        self.preprocessing_report['records_after_feature_engineering'] = len(self.df)
        
    def normalization(self):
        """
        Step 3: Normalization
        - Apply min-max scaling to prevent large-scale features from dominating
        """
        print("\n" + "="*60)
        print("STEP 4: NORMALIZATION")
        print("="*60)
        
        # Columns to normalize (all numeric except date-derived categorical)
        cols_to_normalize = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative_Humidity',
                             'Cloud_Coverage', 'Wind_Speed', 'GHI_lag1', 'DNI_lag1', 
                             'DHI_lag1', 'GHI_ma3', 'GHI_ma7', 'DNI_ma3', 'DNI_ma7']
        
        # Store original values for reference
        self.df_original = self.df.copy()
        
        # Apply min-max scaling
        print("\nApplying Min-Max Scaling (0-1 range):")
        for col in cols_to_normalize:
            if col in self.df.columns:
                original_min = self.df[col].min()
                original_max = self.df[col].max()
                self.df[col] = self.scaler.fit_transform(self.df[[col]])
                print(f"  ✓ {col}: [{original_min:.2f}, {original_max:.2f}] → [0.00, 1.00]")
        
        print("\n✓ Normalization complete")
        self.preprocessing_report['normalized_columns'] = cols_to_normalize
        
    def data_splitting(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """
        Step 4: Data Splitting
        - 70% training, 15% validation, 15% testing
        - Maintains temporal order (no shuffling for time series)
        """
        print("\n" + "="*60)
        print("STEP 5: DATA SPLITTING")
        print("="*60)
        
        total_records = len(self.df)
        
        # Calculate split indices
        train_end = int(total_records * train_ratio)
        val_end = int(total_records * (train_ratio + val_ratio))
        
        # Split data
        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()
        
        print(f"\nData Split Summary:")
        print(f"  Training Set:   {len(train_df)} records ({len(train_df)/total_records*100:.1f}%)")
        print(f"    Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"  Validation Set: {len(val_df)} records ({len(val_df)/total_records*100:.1f}%)")
        print(f"    Date range: {val_df['date'].min()} to {val_df['date'].max()}")
        print(f"  Test Set:       {len(test_df)} records ({len(test_df)/total_records*100:.1f}%)")
        print(f"    Date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        # Save splits
        train_df.to_csv(f"{self.output_dir}/train_data.csv", index=False)
        val_df.to_csv(f"{self.output_dir}/val_data.csv", index=False)
        test_df.to_csv(f"{self.output_dir}/test_data.csv", index=False)
        
        print(f"\n✓ Data splits saved to {self.output_dir}/")
        
        self.preprocessing_report['data_split'] = {
            'train_records': len(train_df),
            'val_records': len(val_df),
            'test_records': len(test_df),
            'train_dates': f"{train_df['date'].min()} to {train_df['date'].max()}",
            'val_dates': f"{val_df['date'].min()} to {val_df['date'].max()}",
            'test_dates': f"{test_df['date'].min()} to {test_df['date'].max()}"
        }
        
        return train_df, val_df, test_df
    
    def save_full_processed_data(self):
        """Save the complete processed dataset"""
        output_file = f"{self.output_dir}/processed_data_full.csv"
        self.df.to_csv(output_file, index=False)
        print(f"✓ Full processed dataset saved to: {output_file}")
        
    def generate_report(self):
        """Generate and save preprocessing report"""
        print("\n" + "="*60)
        print("GENERATING PREPROCESSING REPORT")
        print("="*60)
        
        report_file = f"{self.output_dir}/preprocessing_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.preprocessing_report, f, indent=4)
        
        print(f"✓ Preprocessing report saved to: {report_file}")
        
    def visualize_data(self):
        """Generate visualization of processed data"""
        print("\n" + "="*60)
        print("GENERATING DATA VISUALIZATIONS")
        print("="*60)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Solar Irradiance Data Overview', fontsize=16, fontweight='bold')
        
        # GHI time series
        axes[0, 0].plot(self.df['date'], self.df_original['GHI'], linewidth=0.5)
        axes[0, 0].set_title('Global Horizontal Irradiance (GHI)')
        axes[0, 0].set_ylabel('kWh/m²/day')
        axes[0, 0].grid(True, alpha=0.3)
        
        # DNI time series
        axes[0, 1].plot(self.df['date'], self.df_original['DNI'], linewidth=0.5, color='orange')
        axes[0, 1].set_title('Direct Normal Irradiance (DNI)')
        axes[0, 1].set_ylabel('kWh/m²/day')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Temperature time series
        axes[1, 0].plot(self.df['date'], self.df_original['Temperature'], linewidth=0.5, color='red')
        axes[1, 0].set_title('Temperature')
        axes[1, 0].set_ylabel('°C')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cloud coverage
        axes[1, 1].plot(self.df['date'], self.df_original['Cloud_Coverage'], linewidth=0.5, color='gray')
        axes[1, 1].set_title('Cloud Coverage')
        axes[1, 1].set_ylabel('%')
        axes[1, 1].grid(True, alpha=0.3)
        
        # GHI distribution
        axes[2, 0].hist(self.df_original['GHI'], bins=50, edgecolor='black', alpha=0.7)
        axes[2, 0].set_title('GHI Distribution')
        axes[2, 0].set_xlabel('kWh/m²/day')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Seasonal patterns
        seasonal_ghi = self.df_original.groupby(self.df['season'])['GHI'].mean()
        seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
        axes[2, 1].bar(seasons, seasonal_ghi.values, color=['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3'])
        axes[2, 1].set_title('Average GHI by Season')
        axes[2, 1].set_ylabel('kWh/m²/day')
        axes[2, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        viz_file = f"{self.output_dir}/data_visualization.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to: {viz_file}")
        plt.close()
        
        # Correlation matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr_cols = ['GHI', 'DNI', 'DHI', 'Temperature', 'Relative_Humidity', 
                     'Cloud_Coverage', 'Wind_Speed']
        correlation_matrix = self.df_original[corr_cols].corr()
        
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=1, ax=ax)
        ax.set_title('Correlation Matrix of Key Variables', fontsize=14, fontweight='bold')
        
        corr_file = f"{self.output_dir}/correlation_matrix.png"
        plt.savefig(corr_file, dpi=300, bbox_inches='tight')
        print(f"✓ Correlation matrix saved to: {corr_file}")
        plt.close()


def main():
    """
    Main execution function
    """
    print("\n" + "="*70)
    print(" "*15 + "SOLAR DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = SolarDataPreprocessor(
        input_file='data/raw/nasa_power_johannesburg_raw.csv',
        output_dir='data/processed'
    )
    
    # Execute preprocessing pipeline
    preprocessor.load_data()
    preprocessor.data_cleaning()
    preprocessor.feature_engineering()
    preprocessor.normalization()
    train_df, val_df, test_df = preprocessor.data_splitting()
    preprocessor.save_full_processed_data()
    preprocessor.generate_report()
    preprocessor.visualize_data()
    
    print("\n" + "="*70)
    print(" "*20 + "✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print("\nOutput files created:")
    print("  • data/processed/train_data.csv")
    print("  • data/processed/val_data.csv")
    print("  • data/processed/test_data.csv")
    print("  • data/processed/processed_data_full.csv")
    print("  • data/processed/preprocessing_report.json")
    print("  • data/processed/data_visualization.png")
    print("  • data/processed/correlation_matrix.png")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()