"""
Add additional features for XGBoost model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add US federal holiday indicator
    Holidays include: New Year's Day, Martin Luther King Jr. Day, Presidents' Day,
    Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day,
    Thanksgiving Day, Christmas Day
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Initialize is_holiday column
    df_copy['is_holiday'] = 0
    
    for year in df_copy['year'].unique():
        # Fixed date holidays
        fixed_holidays = [
            f'{year}-01-01',  # New Year's Day
            f'{year}-07-04',  # Independence Day
            f'{year}-12-25',  # Christmas Day
            f'{year}-11-11',  # Veterans Day
        ]
        
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = pd.date_range(start=f'{year}-01-01', end=f'{year}-01-31', freq='W-MON')[2]
        
        # Presidents' Day (3rd Monday in February)
        presidents_day = pd.date_range(start=f'{year}-02-01', end=f'{year}-02-28', freq='W-MON')[2]
        
        # Memorial Day (last Monday in May)
        memorial_day = pd.date_range(start=f'{year}-05-01', end=f'{year}-05-31', freq='W-MON')[-1]
        
        # Labor Day (1st Monday in September)
        labor_day = pd.date_range(start=f'{year}-09-01', end=f'{year}-09-30', freq='W-MON')[0]
        
        # Columbus Day (2nd Monday in October)
        columbus_day = pd.date_range(start=f'{year}-10-01', end=f'{year}-10-31', freq='W-MON')[1]
        
        # Thanksgiving Day (4th Thursday in November)
        thanksgiving = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30', freq='W-THU')[3]
        
        # Combine all holidays
        all_holidays = fixed_holidays + [
            mlk_day.strftime('%Y-%m-%d'),
            presidents_day.strftime('%Y-%m-%d'),
            memorial_day.strftime('%Y-%m-%d'),
            labor_day.strftime('%Y-%m-%d'),
            columbus_day.strftime('%Y-%m-%d'),
            thanksgiving.strftime('%Y-%m-%d')
        ]
        
        # Mark holidays
        df_copy.loc[df_copy['date'].isin(pd.to_datetime(all_holidays)), 'is_holiday'] = 1
    
    print(f"  Added holiday feature")
    print(f"  Total holidays marked: {df_copy['is_holiday'].sum():,}")
    
    return df_copy


def add_system_scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add system scale feature to capture growth trend:
    - active_stations_this_month: number of active stations in current month
    
    This reflects system expansion (new stations opening) without linear time extrapolation.
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Calculate monthly active station count
    df_copy['year_month'] = df_copy['date'].dt.to_period('M')
    
    monthly_stats = df_copy.groupby('year_month').agg({
        'station_name': 'nunique'  # Number of active stations
    }).reset_index()
    
    monthly_stats.columns = ['year_month', 'active_stations_this_month']
    
    # Merge back to main dataframe
    df_copy = df_copy.merge(
        monthly_stats,
        on='year_month',
        how='left'
    )
    
    # Drop temporary column
    df_copy = df_copy.drop('year_month', axis=1)
    
    print(f"  Added system scale feature")
    print(f"  Active stations range: {df_copy['active_stations_this_month'].min()} - {df_copy['active_stations_this_month'].max()}")
    
    return df_copy


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula"""
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def add_station_popularity_features(df: pd.DataFrame, train_end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    Add regional popularity feature based on historical data:
    - nearby_avg_popularity: average historical activity of nearby stations (within 1km)
    
    This represents the historical "hotness" of the surrounding area, not the station itself.
    Helps model understand location quality and can generalize to new stations.
    """
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    
    # Use only training data to calculate historical statistics (avoid data leakage)
    train_data = df_copy[df_copy['date'] <= train_end_date].copy()
    
    # Calculate station-level statistics
    station_stats = train_data.groupby('station_name').agg({
        'departure_count': 'mean',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    station_stats.columns = ['station_name', 'avg_departures', 'latitude', 'longitude']
    
    # Calculate nearby station features
    print("  Calculating nearby station densities...")
    nearby_counts = []
    nearby_avg_popularity = []
    
    for idx, row in station_stats.iterrows():
        lat1, lon1 = row['latitude'], row['longitude']
        
        # Calculate distances to all other stations
        distances = []
        popularities = []
        for _, other_row in station_stats.iterrows():
            if other_row['station_name'] != row['station_name']:
                lat2, lon2 = other_row['latitude'], other_row['longitude']
                dist = calculate_distance(lat1, lon1, lat2, lon2)
                if dist <= 1.0:  # Within 1km
                    distances.append(dist)
                    popularities.append(other_row['avg_departures'])
        
        nearby_counts.append(len(distances))
        nearby_avg_popularity.append(np.mean(popularities) if popularities else row['avg_departures'])
    
    station_stats['nearby_station_count'] = nearby_counts
    station_stats['nearby_avg_popularity'] = nearby_avg_popularity
    
    # Merge back to main dataframe (only nearby_avg_popularity)
    features_to_merge = station_stats[['station_name', 'nearby_avg_popularity']].copy()
    
    df_copy = df_copy.merge(features_to_merge, on='station_name', how='left')
    
    # For new stations (not in training data), use median of nearby popularity
    new_station_mask = df_copy['nearby_avg_popularity'].isna()
    if new_station_mask.sum() > 0:
        # Fill with median nearby popularity
        median_nearby_pop = station_stats['nearby_avg_popularity'].median()
        df_copy.loc[new_station_mask, 'nearby_avg_popularity'] = median_nearby_pop
        
        print(f"  New stations: {new_station_mask.sum():,} records")
        print(f"    → Filled with median nearby_avg_popularity: {median_nearby_pop:.1f}")
    
    print(f"  Added nearby area popularity feature")
    print(f"  Nearby popularity stats:")
    print(f"    Min: {df_copy['nearby_avg_popularity'].min():.1f}")
    print(f"    Mean: {df_copy['nearby_avg_popularity'].mean():.1f}")
    print(f"    Max: {df_copy['nearby_avg_popularity'].max():.1f}")
    
    return df_copy


def create_xgb_features():
    """
    Main function to add XGBoost features to daily station data
    """
    print("="*60)
    print("Adding XGBoost Features to Daily Data")
    print("="*60)
    
    project_root = get_project_root()
    
    # Check if weather data is available
    weather_file = project_root / "data" / "external" / "weather" / "station_daily_with_weather.parquet"
    daily_file = project_root / "data" / "processed" / "daily" / "station_daily_with_coords.parquet"
    
    if weather_file.exists():
        print("\n✓ Loading data with weather features...")
        df = pd.read_parquet(weather_file)
        has_weather = True
    else:
        print("\n⚠ Weather data not available yet, using base daily data...")
        print("  Run weather_api.py first to add weather features")
        df = pd.read_parquet(daily_file)
        has_weather = False
    
    print(f"  Loaded {len(df):,} records")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Add holiday feature
    print("\nAdding holiday feature...")
    df = add_holiday_feature(df)
    
    # Add nearby area popularity (regional feature, not station-specific)
    print("\nAdding nearby area popularity features...")
    print("  This represents the historical activity level of the surrounding area")
    print("  Helps model understand location quality without memorizing specific stations")
    df = add_station_popularity_features(df, train_end_date='2023-12-31')
    
    # Reorder columns for clarity
    base_columns = [
        'station_name',
        'latitude',
        'longitude',
        'date',
        'year',
        'month',
        'day',
        'day_of_week',
        'day_name',
        'is_weekend',
        'is_holiday',
        'season',
        'nearby_avg_popularity',
    ]
    
    # Add weather columns if available
    weather_columns = [
        'temperature_2m_max',
        'temperature_2m_min',
        'temperature_2m_mean',
        'precipitation_sum',
        'rain_sum',
        'snowfall_sum',
        'precipitation_hours',
        'windspeed_10m_max',
        'windgusts_10m_max'
    ] if has_weather else []
    
    # Target column
    target_column = ['departure_count']
    
    # Combine all columns
    all_columns = base_columns + weather_columns + target_column
    
    # Keep only existing columns
    existing_columns = [col for col in all_columns if col in df.columns]
    df_final = df[existing_columns].copy()
    
    # Save to parquet
    output_dir = project_root / "data" / "processed" / "daily"
    output_file = output_dir / "daily_with_xgb_features.parquet"
    df_final.to_parquet(output_file, index=False)
    
    # Save sample CSV
    sample_file = output_dir / "daily_with_xgb_features_sample.csv"
    df_final.head(10000).to_csv(sample_file, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("XGBoost Feature Engineering Complete")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  Parquet: {output_file}")
    print(f"  Sample CSV: {sample_file} (10,000 rows)")
    
    print(f"\nDataset summary:")
    print(f"  Total records: {len(df_final):,}")
    print(f"  Unique stations: {df_final['station_name'].nunique()}")
    print(f"  Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    
    print(f"\nFeature summary:")
    print(f"  Total features: {len(df_final.columns) - 1}")  # Exclude target
    print(f"  Time features: year, month, day, day_of_week, season, is_weekend, is_holiday")
    print(f"  Location features: latitude, longitude, nearby_avg_popularity")
    print(f"  Weather features: {'Yes (9 features)' if has_weather else 'No (run weather_api.py first)'}")
    print(f"\n  Holiday records: {df_final['is_holiday'].sum():,} ({df_final['is_holiday'].mean()*100:.2f}%)")
    print(f"  Weekend records: {df_final['is_weekend'].sum():,} ({df_final['is_weekend'].mean()*100:.2f}%)")
    print(f"  Avg nearby popularity: {df_final['nearby_avg_popularity'].mean():.1f} departures/day")
    
    print(f"\nAll columns ({len(df_final.columns)}):")
    print(f"  {list(df_final.columns)}")
    print(f"{'='*60}")
    
    return df_final


def main():
    create_xgb_features()


if __name__ == "__main__":
    main()

