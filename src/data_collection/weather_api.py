import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import numpy as np


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class WeatherDataCollector:
    
    def __init__(self):
        """Initialize weather data collector"""
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        project_root = get_project_root()
        self.data_dir = project_root / "data" / "processed" / "daily"
        self.output_dir = project_root / "data" / "external" / "weather"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for caching individual station weather data
        self.cache_dir = self.output_dir / "station_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_station_daily_data(self):
        """Load prepared station daily data with coordinates"""
        print("Loading station daily data with coordinates...")
        
        input_file = self.data_dir / "station_daily_with_coords.parquet"
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        df = pd.read_parquet(input_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df):,} records")
        print(f"  Stations: {df['station_name'].nunique()}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def fetch_weather_for_date_range(self, latitude: float, longitude: float, 
                                      start_date: str, end_date: str,
                                      max_retries: int = 3) -> pd.DataFrame:
        """Fetch weather data for a date range with retry logic"""
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,windspeed_10m_max,windgusts_10m_max',
            'timezone': 'America/New_York'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'daily' not in data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(data['daily'])
                df['date'] = pd.to_datetime(df['time'])
                df.drop('time', axis=1, inplace=True)
                
                return df
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"    Rate limited (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    HTTP Error {e.response.status_code}: {e}")
                    time.sleep(10)
                    return pd.DataFrame()
            except requests.exceptions.RequestException as e:
                print(f"    Request Error: {e}")
                time.sleep(10)
                return pd.DataFrame()
        
        return pd.DataFrame()
    
    def get_station_cache_file(self, station_name: str) -> Path:
        """Get cache file path for a station"""
        # Create safe filename from station name
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in station_name)
        safe_name = safe_name.replace(' ', '_')[:100]
        return self.cache_dir / f"{safe_name}.parquet"
    
    def load_cached_weather(self, station_name: str) -> pd.DataFrame:
        """Load cached weather data for a station if exists"""
        cache_file = self.get_station_cache_file(station_name)
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return pd.DataFrame()
    
    def save_cached_weather(self, station_name: str, df_weather: pd.DataFrame):
        """Save weather data for a station to cache"""
        cache_file = self.get_station_cache_file(station_name)
        df_weather.to_parquet(cache_file, index=False)
    
    def add_weather_columns(self, test_mode: bool = False, test_stations: int = 5):
        """Add weather data as columns to station daily data with caching
        
        Args:
            test_mode: If True, only process first N stations for testing
            test_stations: Number of stations to process in test mode
        """
        print("="*60)
        if test_mode:
            print(f"Adding Weather Data (TEST MODE - {test_stations} stations)")
        else:
            print("Adding Weather Data to Station Daily Records")
        print("="*60)
        
        df = self.load_station_daily_data()
        
        # Group by station to get unique station-coordinate pairs
        print("\nGrouping stations by coordinates...")
        station_info = df.groupby('station_name').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        print(f"  Found {len(station_info)} unique stations")
        
        # Check cached stations
        cached_count = 0
        for _, row in station_info.iterrows():
            if self.get_station_cache_file(row['station_name']).exists():
                cached_count += 1
        
        to_process = len(station_info) - cached_count
        print(f"  Already cached: {cached_count}")
        print(f"  To process: {to_process}")
        
        # Estimate time (assuming ~5 requests per station, 6s per request)
        if to_process > 0:
            est_requests = to_process * 5
            est_minutes = (est_requests * 6) / 60
            print(f"  Estimated time: ~{est_minutes:.1f} minutes ({est_minutes/60:.1f} hours)")
            print(f"  Rate: ~10 requests/minute (API limit)")
        print()
        
        # Limit stations in test mode
        if test_mode:
            station_info = station_info.head(test_stations)
            # Recalculate for test mode
            test_to_process = 0
            for _, row in station_info.iterrows():
                if not self.get_station_cache_file(row['station_name']).exists():
                    test_to_process += 1
            print(f"Test mode: Processing only {len(station_info)} stations")
            if test_to_process > 0:
                test_est_requests = test_to_process * 5
                test_est_minutes = (test_est_requests * 6) / 60
                print(f"  Estimated test time: ~{test_est_minutes:.1f} minutes")
            print()
        
        # Process each station
        all_weather = {}
        
        for idx, row in station_info.iterrows():
            station = row['station_name']
            lat = row['latitude']
            lon = row['longitude']
            
            print(f"[{idx+1}/{len(station_info)}] {station[:50]}")
            
            # Check cache first
            cached_weather = self.load_cached_weather(station)
            if not cached_weather.empty:
                all_weather[station] = cached_weather
                print(f"  ✓ Loaded from cache ({len(cached_weather)} days)")
                continue
            
            # Get date range for this station
            station_dates = df[df['station_name'] == station]['date']
            min_date = station_dates.min().strftime('%Y-%m-%d')
            max_date = station_dates.max().strftime('%Y-%m-%d')
            
            # Fetch weather in 2-year chunks
            station_weather = []
            current_start = pd.to_datetime(min_date)
            final_end = pd.to_datetime(max_date)
            
            while current_start <= final_end:
                chunk_end = min(current_start + pd.DateOffset(years=2) - pd.DateOffset(days=1), final_end)
                
                chunk_start_str = current_start.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')
                
                df_chunk = self.fetch_weather_for_date_range(lat, lon, chunk_start_str, chunk_end_str)
                
                if not df_chunk.empty:
                    station_weather.append(df_chunk)
                
                current_start = chunk_end + pd.DateOffset(days=1)
                time.sleep(6.0)  # Wait 6s between requests (10 requests/min)
            
            if station_weather:
                df_weather = pd.concat(station_weather, ignore_index=True)
                all_weather[station] = df_weather
                # Save to cache
                self.save_cached_weather(station, df_weather)
                print(f"  ✓ Fetched and cached ({len(df_weather)} days)")
            else:
                print(f"  ✗ Failed to fetch weather data")
        
        # Merge weather data
        print("\nMerging weather data with daily records...")
        
        weather_list = []
        for station, weather_df in all_weather.items():
            weather_df['station_name'] = station
            weather_list.append(weather_df)
        
        if not weather_list:
            print("No weather data collected")
            return None
        
        df_weather_all = pd.concat(weather_list, ignore_index=True)
        
        # Merge with original data
        df_final = df.merge(
            df_weather_all,
            on=['station_name', 'date'],
            how='left'
        )
        
        # Save parquet
        output_file = self.output_dir / "station_daily_with_weather.parquet"
        df_final.to_parquet(output_file, index=False)
        
        # Save sample CSV (first 10,000 rows)
        sample_file = self.output_dir / "station_daily_with_weather_sample.csv"
        df_final.head(10000).to_csv(sample_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"Data with weather saved:")
        print(f"  Parquet: {output_file}")
        print(f"  Sample CSV: {sample_file} (10,000 rows)")
        print(f"\nTotal records: {len(df_final):,}")
        print(f"Records with weather: {df_final['temperature_2m_mean'].notna().sum():,}")
        print(f"Missing weather: {df_final['temperature_2m_mean'].isna().sum():,}")
        print(f"\nColumns: {list(df_final.columns)}")
        print(f"{'='*60}")
        
        return df_final


def main(test_mode: bool = False):
    """
    Run weather data collection
    
    Args:
        test_mode: If True, only process 5 stations for testing
    """
    collector = WeatherDataCollector()
    df = collector.add_weather_columns(test_mode=test_mode)


if __name__ == "__main__":
    import sys
    # Run in test mode if 'test' argument is provided
    test_mode = len(sys.argv) > 1 and sys.argv[1] == 'test'
    main(test_mode=test_mode)
