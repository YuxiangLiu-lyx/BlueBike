import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class BlueBikesDataCleaner:
    
    # Column mapping from old format to standardized format
    COLUMN_MAPPING = {
        # Old format columns
        'tripduration': 'duration',
        'starttime': 'start_time',
        'stoptime': 'end_time',
        'start station id': 'start_station_id',
        'start station name': 'start_station_name',
        'start station latitude': 'start_lat',
        'start station longitude': 'start_lng',
        'end station id': 'end_station_id',
        'end station name': 'end_station_name',
        'end station latitude': 'end_lat',
        'end station longitude': 'end_lng',
        'bikeid': 'bike_id',
        'usertype': 'user_type',   
        'birth year': 'birth_year',
        'gender': 'gender',
        'postal code': 'postal_code',
        
        # New format columns
        'ride_id': 'ride_id',
        'rideable_type': 'bike_type',
        'started_at': 'start_time',
        'ended_at': 'end_time',
        'start_station_name': 'start_station_name',
        'start_station_id': 'start_station_id',
        'end_station_name': 'end_station_name',
        'end_station_id': 'end_station_id',
        'start_lat': 'start_lat',
        'start_lng': 'start_lng',
        'end_lat': 'end_lat',
        'end_lng': 'end_lng',
        'member_casual': 'user_type',
    }
    
    # Standardize user types
    USER_TYPE_MAPPING = {
        'Subscriber': 'member',
        'subscriber': 'member',
        'Member': 'member',
        'member': 'member',
        'Customer': 'casual',
        'customer': 'casual',
        'Casual': 'casual',
        'casual': 'casual',
    }
    
    def __init__(self, raw_data_dir: str = None, processed_data_dir: str = None):
        project_root = get_project_root()
        
        if raw_data_dir is None:
            raw_data_dir = project_root / "data" / "raw" / "bluebike_raw"
        if processed_data_dir is None:
            processed_data_dir = project_root / "data" / "processed" / "bluebike_cleaned"
            
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single CSV file with proper encoding and type handling"""
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df['source_file'] = file_path.name
            return df
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
            return None
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to standardized names"""
        df_clean = df.copy()
        
        # Rename columns based on mapping
        for old_col, new_col in self.COLUMN_MAPPING.items():
            if old_col in df_clean.columns:
                df_clean.rename(columns={old_col: new_col}, inplace=True)
        
        return df_clean
    
    def parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time columns to datetime format"""
        df_clean = df.copy()
        
        if 'start_time' in df_clean.columns:
            df_clean['start_time'] = pd.to_datetime(df_clean['start_time'], errors='coerce')
        
        if 'end_time' in df_clean.columns:
            df_clean['end_time'] = pd.to_datetime(df_clean['end_time'], errors='coerce')
        
        return df_clean
    
    def calculate_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trip duration if not present"""
        df_clean = df.copy()
        
        if 'duration' not in df_clean.columns or df_clean['duration'].isna().all():
            if 'start_time' in df_clean.columns and 'end_time' in df_clean.columns:
                df_clean['duration'] = (df_clean['end_time'] - df_clean['start_time']).dt.total_seconds()
        else:
            # Ensure duration is numeric
            df_clean['duration'] = pd.to_numeric(df_clean['duration'], errors='coerce')
        
        return df_clean
    
    def standardize_user_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize user type values"""
        df_clean = df.copy()
        
        if 'user_type' in df_clean.columns:
            df_clean['user_type'] = df_clean['user_type'].map(
                lambda x: self.USER_TYPE_MAPPING.get(x, x) if pd.notna(x) else None
            )
        
        return df_clean
    
    def clean_station_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate station information"""
        df_clean = df.copy()
        
        # Convert station IDs to string to handle mixed types (numeric and alphanumeric)
        if 'start_station_id' in df_clean.columns:
            df_clean['start_station_id'] = df_clean['start_station_id'].astype(str)
        if 'end_station_id' in df_clean.columns:
            df_clean['end_station_id'] = df_clean['end_station_id'].astype(str)
        
        # Convert coordinates to numeric
        coord_columns = ['start_lat', 'start_lng', 'end_lat', 'end_lng']
        for col in coord_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove leading/trailing whitespace from station names
        if 'start_station_name' in df_clean.columns:
            df_clean['start_station_name'] = df_clean['start_station_name'].str.strip()
        if 'end_station_name' in df_clean.columns:
            df_clean['end_station_name'] = df_clean['end_station_name'].str.strip()
        
        return df_clean
    
    def remove_invalid_trips(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Remove invalid trips and return statistics"""
        df_clean = df.copy()
        original_count = len(df_clean)
        stats = {}
        
        # Remove trips with missing critical information
        df_clean = df_clean.dropna(subset=['start_time', 'end_time'])
        stats['missing_time'] = original_count - len(df_clean)
        
        # Remove trips with invalid duration (negative or too short/long)
        if 'duration' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['duration'] > 0) & 
                (df_clean['duration'] < 86400)  # Less than 24 hours
            ]
            stats['invalid_duration'] = original_count - stats['missing_time'] - len(df_clean)
        
        # Remove trips with missing or invalid coordinates
        coord_cols = ['start_lat', 'start_lng', 'end_lat', 'end_lng']
        if all(col in df_clean.columns for col in coord_cols):
            valid_coords = (
                (df_clean['start_lat'].between(42.2, 42.5)) &
                (df_clean['start_lng'].between(-71.2, -70.9)) &
                (df_clean['end_lat'].between(42.2, 42.5)) &
                (df_clean['end_lng'].between(-71.2, -70.9))
            )
            before_coord_filter = len(df_clean)
            df_clean = df_clean[valid_coords]
            stats['invalid_coordinates'] = before_coord_filter - len(df_clean)
        
        stats['total_removed'] = original_count - len(df_clean)
        stats['remaining'] = len(df_clean)
        stats['removal_rate'] = stats['total_removed'] / original_count * 100
        
        return df_clean, stats
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract useful time-based features"""
        df_clean = df.copy()
        
        if 'start_time' in df_clean.columns:
            df_clean['year'] = df_clean['start_time'].dt.year
            df_clean['month'] = df_clean['start_time'].dt.month
            df_clean['day'] = df_clean['start_time'].dt.day
            df_clean['hour'] = df_clean['start_time'].dt.hour
            df_clean['day_of_week'] = df_clean['start_time'].dt.dayofweek
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
        
        return df_clean
    
    def create_station_keys(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified station identifiers based on name and location
        This handles the case where station_id changed over time
        Coordinates rounded to 3 decimal places (approx 100m precision)
        """
        df_clean = df.copy()
        
        # Create station key by combining name and rounded coordinates
        if all(col in df_clean.columns for col in ['start_station_name', 'start_lat', 'start_lng']):
            df_clean['start_station_key'] = (
                df_clean['start_station_name'].fillna('unknown') + '_' + 
                df_clean['start_lat'].round(3).astype(str) + '_' + 
                df_clean['start_lng'].round(3).astype(str)
            )
        
        if all(col in df_clean.columns for col in ['end_station_name', 'end_lat', 'end_lng']):
            df_clean['end_station_key'] = (
                df_clean['end_station_name'].fillna('unknown') + '_' + 
                df_clean['end_lat'].round(3).astype(str) + '_' + 
                df_clean['end_lng'].round(3).astype(str)
            )
        
        return df_clean
    
    def process_single_file(self, file_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """Process a single CSV file through the entire cleaning pipeline"""
        print(f"Processing {file_path.name}...", end=" ")
        
        df = self.load_csv_file(file_path)
        if df is None:
            return None, {}
        
        original_rows = len(df)
        
        # Apply cleaning steps
        df = self.standardize_columns(df)
        df = self.parse_datetime(df)
        df = self.calculate_duration(df)
        df = self.standardize_user_type(df)
        df = self.clean_station_data(df)
        df, removal_stats = self.remove_invalid_trips(df)
        df = self.add_time_features(df)
        df = self.create_station_keys(df)
        
        stats = {
            'file': file_path.name,
            'original_rows': original_rows,
            'cleaned_rows': len(df),
            **removal_stats
        }
        
        print(f"Done ({len(df):,} rows, {removal_stats['removal_rate']:.1f}% removed)")
        
        return df, stats
    
    def clean_all_data(self, output_filename: str = "trips_cleaned.parquet") -> pd.DataFrame:
        """Process all CSV files in raw data directory and combine into one cleaned dataset"""
        print("BlueBikes Data Cleaning Pipeline")
        print("=" * 60)
        
        csv_files = sorted(self.raw_data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {self.raw_data_dir}")
        print("-" * 60)
        
        all_dataframes = []
        all_stats = []
        
        for csv_file in csv_files:
            df, stats = self.process_single_file(csv_file)
            if df is not None and len(df) > 0:
                all_dataframes.append(df)
                all_stats.append(stats)
        
        if not all_dataframes:
            print("No valid data found!")
            return None
        
        print("-" * 60)
        print("Combining all files...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by start time
        combined_df = combined_df.sort_values('start_time').reset_index(drop=True)
        
        # Fix data types for optional columns before saving
        if 'birth_year' in combined_df.columns:
            combined_df['birth_year'] = pd.to_numeric(combined_df['birth_year'], errors='coerce')
        if 'gender' in combined_df.columns:
            combined_df['gender'] = pd.to_numeric(combined_df['gender'], errors='coerce')
        if 'bike_id' in combined_df.columns:
            combined_df['bike_id'] = combined_df['bike_id'].astype(str)
        if 'postal_code' in combined_df.columns:
            combined_df['postal_code'] = combined_df['postal_code'].astype(str)
        
        # Save to processed directory
        output_path = self.processed_data_dir / output_filename
        combined_df.to_parquet(output_path, index=False)
        
        print(f"Saved cleaned data to: {output_path}")
        print("=" * 60)
        print(f"Total trips: {len(combined_df):,}")
        print(f"Date range: {combined_df['start_time'].min()} to {combined_df['start_time'].max()}")
        print(f"Total stations: {combined_df['start_station_id'].nunique()} (start), "
              f"{combined_df['end_station_id'].nunique()} (end)")
        
        # Print summary statistics
        stats_df = pd.DataFrame(all_stats)
        stats_summary = self.processed_data_dir / "cleaning_stats.csv"
        stats_df.to_csv(stats_summary, index=False)
        print(f"\nCleaning statistics saved to: {stats_summary}")
        
        return combined_df


def main():
    cleaner = BlueBikesDataCleaner()
    
    # Process all data
    cleaned_data = cleaner.clean_all_data(output_filename="trips_cleaned.parquet")
    
    if cleaned_data is not None:
        print("\nData cleaning completed")
        print(f"Cleaned dataset shape: {cleaned_data.shape}")
        print(f"\nColumns in cleaned data:")
        for col in cleaned_data.columns:
            print(f"  - {col}")


if __name__ == "__main__":
    main()

