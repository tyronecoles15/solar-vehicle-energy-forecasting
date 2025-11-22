"""
NASA POWER Data Collection Script
Collects 10 years of solar irradiance and meteorological data for Johannesburg, SA
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import os

class NASAPowerDataCollector:
    """
    Collects meteorological and solar irradiance data from NASA POWER API
    """
    
    def __init__(self, latitude=-26.2041, longitude=28.0473):
        """
        Initialize data collector for Johannesburg, South Africa
        
        Parameters:
        -----------
        latitude : float
            Latitude of location (default: Johannesburg)
        longitude : float
            Longitude of location (default: Johannesburg)
        """
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        self.latitude = latitude
        self.longitude = longitude
        
    def collect_data(self, start_date, end_date, output_file):
        """
        Collect data from NASA POWER API
        
        Parameters:
        -----------
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        output_file : str
            Path to save the collected data
        """
        
        # Parameters to collect based on methodology document
        parameters = [
            'ALLSKY_SFC_SW_DWN',      # GHI - Global Horizontal Irradiance
            'ALLSKY_SFC_SW_DNI',       # DNI - Direct Normal Irradiance
            'ALLSKY_SFC_SW_DIFF',      # DHI - Diffuse Horizontal Irradiance
            'T2M',                     # Temperature at 2 meters
            'RH2M',                    # Relative Humidity at 2 meters
            'CLOUD_AMT',               # Cloud Amount
            'WS2M'                     # Wind Speed at 2 meters
        ]
        
        # Build API request URL
        params_str = ','.join(parameters)
        url = f"{self.base_url}?parameters={params_str}&community=RE&longitude={self.longitude}&latitude={self.latitude}&start={start_date}&end={end_date}&format=JSON"
        
        print(f"Requesting data from NASA POWER API...")
        print(f"Location: Johannesburg ({self.latitude}, {self.longitude})")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Parameters: {len(parameters)} variables")
        
        try:
            # Make API request
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Extract parameters data
            parameters_data = data['properties']['parameter']
            
            # Convert to DataFrame
            df = pd.DataFrame(parameters_data)
            
            # Reset index to make date a column
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            # Convert date format from YYYYMMDD to datetime
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # Rename columns to more readable names
            column_mapping = {
                'ALLSKY_SFC_SW_DWN': 'GHI',
                'ALLSKY_SFC_SW_DNI': 'DNI',
                'ALLSKY_SFC_SW_DIFF': 'DHI',
                'T2M': 'Temperature',
                'RH2M': 'Relative_Humidity',
                'CLOUD_AMT': 'Cloud_Coverage',
                'WS2M': 'Wind_Speed'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Display basic info
            print(f"\n✓ Data collected successfully!")
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"\nFirst few rows:")
            print(df.head())
            
            # Save to CSV
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df.to_csv(output_file, index=False)
            print(f"\n✓ Data saved to: {output_file}")
            
            # Save metadata
            metadata = {
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'location': 'Johannesburg, South Africa',
                'latitude': self.latitude,
                'longitude': self.longitude,
                'start_date': start_date,
                'end_date': end_date,
                'total_records': len(df),
                'parameters': list(column_mapping.values()),
                'source': 'NASA POWER Project',
                'api_url': self.base_url
            }
            
            metadata_file = output_file.replace('.csv', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"✓ Metadata saved to: {metadata_file}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Error collecting data: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None
    
    def get_data_summary(self, df):
        """
        Generate summary statistics for collected data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The collected data
        """
        print("\n" + "="*60)
        print("DATA SUMMARY STATISTICS")
        print("="*60)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            print(f"\n{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std Dev: {df[col].std():.2f}")
            print(f"  Min: {df[col].min():.2f}")
            print(f"  Max: {df[col].max():.2f}")
            print(f"  Missing values: {df[col].isna().sum()}")


def main():
    """
    Main execution function
    """
    # Initialize collector
    collector = NASAPowerDataCollector()
    
    # Define date range (10 years as per methodology)
    # Using 2014-2023 to have complete years
    start_date = '20140101'
    end_date = '20231231'
    
    # Output file path
    output_file = 'data/raw/nasa_power_johannesburg_raw.csv'
    
    # Collect data
    df = collector.collect_data(start_date, end_date, output_file)
    
    if df is not None:
        # Generate summary
        collector.get_data_summary(df)
        
        print("\n" + "="*60)
        print("✓ DATA COLLECTION COMPLETE")
        print("="*60)
        print(f"\nNext steps:")
        print("1. Review the data in: {output_file}")
        print("2. Check metadata in: {output_file.replace('.csv', '_metadata.json')}")
        print("3. Proceed to data preprocessing")
    else:
        print("\n✗ Data collection failed. Please check errors above.")


if __name__ == "__main__":
    main()