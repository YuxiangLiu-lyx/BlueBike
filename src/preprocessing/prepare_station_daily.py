import pandas as pd
from pathlib import Path


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def prepare_station_daily_data():
    """
    Aggregate daily data with station coordinates
    Output: station_name, latitude, longitude, date, departure_count
    """
    print("="*60)
    print("Preparing Station Daily Data with Coordinates")
    print("="*60)
    
    project_root = get_project_root()
    
    # Load daily departure data
    print("\nLoading daily departure data...")
    daily_file = project_root / "data" / "processed" / "daily" / "daily_departures.parquet"
    df_daily = pd.read_parquet(daily_file)
    print(f"  Loaded {len(df_daily):,} daily records")
    print(f"  Stations: {df_daily['station_name'].nunique()}")
    print(f"  Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    
    # Load trips data to get coordinates
    print("\nLoading station coordinates...")
    trips_file = project_root / "data" / "processed" / "bluebike_cleaned" / "trips_cleaned.parquet"
    df_trips = pd.read_parquet(trips_file, columns=['start_station_name', 'start_lat', 'start_lng'])
    
    # Get median coordinates for each station
    station_coords = df_trips.groupby('start_station_name').agg({
        'start_lat': 'median',
        'start_lng': 'median'
    }).reset_index()
    
    station_coords.columns = ['station_name', 'latitude', 'longitude']
    print(f"  Extracted coordinates for {len(station_coords)} stations")
    
    # Merge daily data with coordinates
    print("\nMerging data...")
    df_merged = df_daily.merge(
        station_coords,
        on='station_name',
        how='left'
    )
    
    # Select and reorder columns
    columns_to_keep = [
        'station_name',
        'latitude',
        'longitude',
        'date',
        'departure_count',
        'year',
        'month',
        'day',
        'day_of_week',
        'day_name',
        'is_weekend',
        'season'
    ]
    
    df_final = df_merged[columns_to_keep].copy()
    
    # Remove records without coordinates
    records_before = len(df_final)
    df_final = df_final.dropna(subset=['latitude', 'longitude'])
    records_after = len(df_final)
    
    if records_before > records_after:
        print(f"  Removed {records_before - records_after:,} records without coordinates")
    
    # Sort by station and date
    df_final = df_final.sort_values(['station_name', 'date']).reset_index(drop=True)
    
    # Save parquet
    output_dir = project_root / "data" / "processed" / "daily"
    output_file = output_dir / "station_daily_with_coords.parquet"
    df_final.to_parquet(output_file, index=False)
    
    # Save sample CSV
    sample_file = output_dir / "station_daily_with_coords_sample.csv"
    df_final.head(10000).to_csv(sample_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Station daily data saved:")
    print(f"  Parquet: {output_file}")
    print(f"  Sample CSV: {sample_file} (10,000 rows)")
    print(f"\nTotal records: {len(df_final):,}")
    print(f"Unique stations: {df_final['station_name'].nunique()}")
    print(f"Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"\nColumns: {list(df_final.columns)}")
    print(f"{'='*60}")
    
    return df_final


def show_sample(n=10):
    """Display sample of prepared data"""
    project_root = get_project_root()
    file_path = project_root / "data" / "processed" / "daily" / "station_daily_with_coords.parquet"
    
    if not file_path.exists():
        print("File not found. Run prepare_station_daily_data() first.")
        return
    
    df = pd.read_parquet(file_path)
    print(f"\nSample data (first {n} rows):")
    print(df.head(n))
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nSummary statistics:")
    print(df[['latitude', 'longitude', 'departure_count']].describe())


def main():
    df = prepare_station_daily_data()
    show_sample()


if __name__ == "__main__":
    main()

