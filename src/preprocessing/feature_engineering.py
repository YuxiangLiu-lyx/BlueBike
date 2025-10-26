import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class HourlyAggregator:
    
    def __init__(self, cleaned_data_path: str = None, output_dir: str = None):
        project_root = get_project_root()
        
        if cleaned_data_path is None:
            cleaned_data_path = project_root / "data" / "processed" / "bluebike_cleaned" / "trips_cleaned.parquet"
        if output_dir is None:
            output_dir = project_root / "data" / "processed" / "hourly"
        
        self.cleaned_data_path = Path(cleaned_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load cleaned trip data"""
        print(f"Loading data from {self.cleaned_data_path}...")
        self.df = pd.read_parquet(self.cleaned_data_path)
        print(f"Loaded {len(self.df):,} trips")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return self.df
    
    def aggregate_hourly_departures(self):
        """Aggregate departures by station and hour"""
        print("\nAggregating departures by station and hour...")
        
        # Extract date and hour from start_time
        print("Extracting time components...")
        self.df['date'] = self.df['start_time'].dt.date
        self.df['hour'] = self.df['start_time'].dt.hour
        
        # Group by station name, date, and hour
        print("Grouping and counting departures...")
        hourly = self.df.groupby([
            'start_station_name',
            'date',
            'hour'
        ], observed=True).size().reset_index(name='departure_count')
        
        print(f"Created {len(hourly):,} station-hour records")
        
        return hourly
    
    def add_time_features(self, hourly_df):
        """Add useful time-based features"""
        print("\nAdding time features...")
        
        # Convert date back to datetime for feature extraction
        hourly_df['datetime'] = pd.to_datetime(hourly_df['date'])
        
        # Extract time features
        hourly_df['year'] = hourly_df['datetime'].dt.year
        hourly_df['month'] = hourly_df['datetime'].dt.month
        hourly_df['day'] = hourly_df['datetime'].dt.day
        hourly_df['day_of_week'] = hourly_df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
        hourly_df['is_weekend'] = hourly_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add season
        hourly_df['season'] = hourly_df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Add time of day category
        def categorize_time(hour):
            if 6 <= hour < 10:
                return 'Morning_Rush'
            elif 10 <= hour < 16:
                return 'Midday'
            elif 16 <= hour < 20:
                return 'Evening_Rush'
            elif 20 <= hour < 24:
                return 'Evening'
            else:
                return 'Night'
        
        hourly_df['time_category'] = hourly_df['hour'].apply(categorize_time)
        
        # Drop the temporary datetime column
        hourly_df = hourly_df.drop('datetime', axis=1)
        
        # Rename for consistency
        hourly_df = hourly_df.rename(columns={
            'start_station_name': 'station_name'
        })
        
        # Reorder columns for better readability
        column_order = [
            'station_name',
            'date', 'year', 'month', 'day', 'day_of_week', 'is_weekend', 'season',
            'hour', 'time_category',
            'departure_count'
        ]
        
        # Reorder (only include columns that exist)
        existing_columns = [col for col in column_order if col in hourly_df.columns]
        hourly_df = hourly_df[existing_columns]
        
        return hourly_df
    
    def save_results(self, hourly_df):
        """Save aggregated data in multiple formats"""
        print("\nSaving results...")
        
        # Sort by station and time
        hourly_df = hourly_df.sort_values(['station_name', 'date', 'hour'])
        
        # Save full dataset as parquet
        parquet_path = self.output_dir / "hourly_departures.parquet"
        hourly_df.to_parquet(parquet_path, index=False)
        file_size = parquet_path.stat().st_size / 1024**2
        print(f"✓ Saved full dataset: {parquet_path}")
        print(f"  File size: {file_size:.1f} MB")
        
        # Save sample as CSV for inspection
        sample_size = min(1000, len(hourly_df))
        sample_path = self.output_dir / "hourly_departures_sample.csv"
        hourly_df.head(sample_size).to_csv(sample_path, index=False)
        print(f"✓ Saved sample ({sample_size} rows): {sample_path}")
        
        # Create and save summary statistics
        self.save_summary_stats(hourly_df)
    
    def save_summary_stats(self, hourly_df):
        """Generate and save summary statistics"""
        print("\nGenerating summary statistics...")
        
        summary = {
            'Total Records': len(hourly_df),
            'Unique Stations': hourly_df['station_name'].nunique(),
            'Date Range Start': hourly_df['date'].min(),
            'Date Range End': hourly_df['date'].max(),
            'Total Departures': hourly_df['departure_count'].sum(),
            'Avg Departures per Hour': hourly_df['departure_count'].mean(),
            'Median Departures per Hour': hourly_df['departure_count'].median(),
            'Max Departures in One Hour': hourly_df['departure_count'].max(),
            'Min Departures in One Hour': hourly_df['departure_count'].min(),
        }
        
        # Convert to DataFrame
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Value']
        
        # Save
        summary_path = self.output_dir / "hourly_departures_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"✓ Saved summary statistics: {summary_path}")
        
        # Print to console
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        print(summary_df.to_string())
        print("="*60)
    
    def run(self):
        """Execute the complete hourly aggregation pipeline"""
        print("="*60)
        print("Hourly Departure Aggregation")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Aggregate by hour
        hourly_df = self.aggregate_hourly_departures()
        
        # Add time features
        hourly_df = self.add_time_features(hourly_df)
        
        # Save results
        self.save_results(hourly_df)
        
        print("\n" + "="*60)
        print("Aggregation complete!")
        print("="*60)
        
        return hourly_df


class DailyAggregator:
    
    def __init__(self, cleaned_data_path: str = None, output_dir: str = None):
        project_root = get_project_root()
        
        if cleaned_data_path is None:
            cleaned_data_path = project_root / "data" / "processed" / "bluebike_cleaned" / "trips_cleaned.parquet"
        if output_dir is None:
            output_dir = project_root / "data" / "processed" / "daily"
        
        self.cleaned_data_path = Path(cleaned_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load cleaned trip data"""
        print(f"Loading data from {self.cleaned_data_path}...")
        self.df = pd.read_parquet(self.cleaned_data_path)
        print(f"Loaded {len(self.df):,} trips")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return self.df
    
    def aggregate_daily_departures(self):
        """Aggregate departures by station and day"""
        print("\nAggregating departures by station and day...")
        
        # Extract date from start_time
        print("Extracting date...")
        self.df['date'] = self.df['start_time'].dt.date
        
        # Group by station and date
        print("Grouping and counting departures...")
        daily = self.df.groupby([
            'start_station_key',
            'start_station_name',
            'date'
        ], observed=True).size().reset_index(name='departure_count')
        
        print(f"Created {len(daily):,} station-day records")
        
        return daily
    
    def add_time_features(self, daily_df):
        """Add useful time-based features"""
        print("\nAdding time features...")
        
        # Convert date to datetime for feature extraction
        daily_df['datetime'] = pd.to_datetime(daily_df['date'])
        
        # Extract time features
        daily_df['year'] = daily_df['datetime'].dt.year
        daily_df['month'] = daily_df['datetime'].dt.month
        daily_df['day'] = daily_df['datetime'].dt.day
        daily_df['day_of_week'] = daily_df['datetime'].dt.dayofweek
        daily_df['day_name'] = daily_df['datetime'].dt.day_name()
        daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add season
        daily_df['season'] = daily_df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Drop the temporary datetime column
        daily_df = daily_df.drop('datetime', axis=1)
        
        # Rename for consistency
        daily_df = daily_df.rename(columns={
            'start_station_name': 'station_name'
        })
        
        # Reorder columns for better readability
        column_order = [
            'station_name',
            'date', 'year', 'month', 'day', 'day_of_week', 'day_name', 'is_weekend', 'season',
            'departure_count'
        ]
        
        existing_columns = [col for col in column_order if col in daily_df.columns]
        daily_df = daily_df[existing_columns]
        
        return daily_df
    
    def save_results(self, daily_df):
        """Save aggregated data in multiple formats"""
        print("\nSaving results...")
        
        # Sort by station and date
        daily_df = daily_df.sort_values(['station_name', 'date'])
        
        # Save full dataset as parquet
        parquet_path = self.output_dir / "daily_departures.parquet"
        daily_df.to_parquet(parquet_path, index=False)
        file_size = parquet_path.stat().st_size / 1024**2
        print(f"✓ Saved full dataset: {parquet_path}")
        print(f"  File size: {file_size:.1f} MB")
        
        # Save sample as CSV for inspection
        sample_size = min(1000, len(daily_df))
        sample_path = self.output_dir / "daily_departures_sample.csv"
        daily_df.head(sample_size).to_csv(sample_path, index=False)
        print(f"✓ Saved sample ({sample_size} rows): {sample_path}")
        
        # Create and save summary statistics
        self.save_summary_stats(daily_df)
    
    def save_summary_stats(self, daily_df):
        """Generate and save summary statistics"""
        print("\nGenerating summary statistics...")
        
        summary = {
            'Total Records': len(daily_df),
            'Unique Stations': daily_df['station_name'].nunique(),
            'Date Range Start': daily_df['date'].min(),
            'Date Range End': daily_df['date'].max(),
            'Total Days': (daily_df['date'].max() - daily_df['date'].min()).days,
            'Total Departures': daily_df['departure_count'].sum(),
            'Avg Departures per Day': daily_df['departure_count'].mean(),
            'Median Departures per Day': daily_df['departure_count'].median(),
            'Max Departures in One Day': daily_df['departure_count'].max(),
            'Min Departures in One Day': daily_df['departure_count'].min(),
        }
        
        # Convert to DataFrame
        summary_df = pd.DataFrame([summary]).T
        summary_df.columns = ['Value']
        
        # Save
        summary_path = self.output_dir / "daily_departures_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"✓ Saved summary statistics: {summary_path}")
        
        # Print to console
        print("\n" + "="*60)
        print("Summary Statistics")
        print("="*60)
        print(summary_df.to_string())
        print("="*60)
    
    def run(self):
        """Execute the complete daily aggregation pipeline"""
        print("="*60)
        print("Daily Departure Aggregation")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Aggregate by day
        daily_df = self.aggregate_daily_departures()
        
        # Add time features
        daily_df = self.add_time_features(daily_df)
        
        # Save results
        self.save_results(daily_df)
        
        print("\n" + "="*60)
        print("Aggregation complete!")
        print("="*60)
        
        return daily_df


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'hourly':
        print("Running hourly aggregation...\n")
        aggregator = HourlyAggregator()
        aggregator.run()
        print("\nOutput files:")
        print("  - data/processed/hourly/hourly_departures.parquet")
        print("  - data/processed/hourly/hourly_departures_sample.csv")
        print("  - data/processed/hourly/hourly_departures_summary.csv")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'daily':
        print("Running daily aggregation...\n")
        aggregator = DailyAggregator()
        aggregator.run()
        print("\nOutput files:")
        print("  - data/processed/daily/daily_departures.parquet")
        print("  - data/processed/daily/daily_departures_sample.csv")
        print("  - data/processed/daily/daily_departures_summary.csv")
    
    else:
        print("Running both daily and hourly aggregations...\n")
        
        print("Step 1/2: Daily aggregation")
        print("-" * 60)
        daily_agg = DailyAggregator()
        daily_agg.run()
        
        print("\n\nStep 2/2: Hourly aggregation")
        print("-" * 60)
        hourly_agg = HourlyAggregator()
        hourly_agg.run()
        
        print("\n\nAll output files:")
        print("Daily:")
        print("  - data/processed/daily/daily_departures.parquet")
        print("  - data/processed/daily/daily_departures_sample.csv")
        print("  - data/processed/daily/daily_departures_summary.csv")
        print("\nHourly:")
        print("  - data/processed/hourly/hourly_departures.parquet")
        print("  - data/processed/hourly/hourly_departures_sample.csv")
        print("  - data/processed/hourly/hourly_departures_summary.csv")


if __name__ == "__main__":
    main()

