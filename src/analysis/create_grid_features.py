"""
Grid-based demand prediction feature preparation.

Divides Boston Metro Area into 500m x 500m grid cells and prepares
temporal and weather features for each cell. POI features are added separately.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class GridFeatureCreator:
    def __init__(self):
        self.project_root = get_project_root()
        self.grid_size_deg = 0.0045  # ~500m at Boston latitude
        
        # Output directory
        self.output_dir = self.project_root / "data" / "processed" / "grid"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reference date: 2024-07-15 (Monday, Summer peak)
        self.reference_date = datetime(2024, 7, 15)
        
        # Boston Metro Area boundaries (Boston + Cambridge + Somerville + Brookline)
        self.BOSTON_BOUNDARIES = {
            'lat_min': 42.227,   # South (Hyde Park, Mattapan)
            'lat_max': 42.418,   # North (Somerville, Medford border)
            'lon_min': -71.191,  # West (Brighton, Brookline)
            'lon_max': -70.994,  # East (East Boston, Logan Airport)
        }
        
    def get_boston_boundaries(self):
        """Define grid boundaries covering entire Boston Metro Area"""
        print("\n" + "="*80)
        print("Grid Boundaries Definition")
        print("="*80)
        
        # Use predefined Boston Metro Area boundaries
        lat_min = self.BOSTON_BOUNDARIES['lat_min']
        lat_max = self.BOSTON_BOUNDARIES['lat_max']
        lon_min = self.BOSTON_BOUNDARIES['lon_min']
        lon_max = self.BOSTON_BOUNDARIES['lon_max']
        
        print(f"\nCoverage: Boston + Cambridge + Somerville + Brookline")
        print(f"Latitude: {lat_min:.3f} to {lat_max:.3f}")
        print(f"Longitude: {lon_min:.3f} to {lon_max:.3f}")
        
        lat_km = (lat_max - lat_min) * 111
        lon_km = (lon_max - lon_min) * 111 * np.cos(np.radians((lat_min + lat_max) / 2))
        area_km2 = lat_km * lon_km
        
        print(f"Coverage: {lat_km:.1f} km (N-S) x {lon_km:.1f} km (E-W) = {area_km2:.0f} km²")
        
        n_lat = int(np.ceil((lat_max - lat_min) / self.grid_size_deg))
        n_lon = int(np.ceil((lon_max - lon_min) / self.grid_size_deg))
        total_cells = n_lat * n_lon
        
        print(f"\nGrid configuration: {n_lat} x {n_lon} = {total_cells} cells")
        print(f"Cell size: ~500m x 500m ({self.grid_size_deg:.4f}°)")
        
        estimated_time_min = total_cells * 6 / 60
        print(f"Estimated POI extraction time: {estimated_time_min:.0f} minutes ({estimated_time_min/60:.1f} hours)")
        
        data_file = self.project_root / "data" / "processed" / "daily" / "daily_with_xgb_features.parquet"
        if data_file.exists():
            df = pd.read_parquet(data_file)
            existing_lat_min = df['latitude'].min()
            existing_lat_max = df['latitude'].max()
            existing_lon_min = df['longitude'].min()
            existing_lon_max = df['longitude'].max()
            print(f"\nExisting stations: {existing_lat_min:.3f} to {existing_lat_max:.3f} (lat), "
                  f"{existing_lon_min:.3f} to {existing_lon_max:.3f} (lon)")
            print(f"Grid extends beyond current coverage")
        
        return lat_min, lat_max, lon_min, lon_max, n_lat, n_lon
    
    def create_grids(self, lat_min, lat_max, lon_min, lon_max, n_lat, n_lon):
        """Create grid cells covering the area"""
        print("\n" + "="*80)
        print("Grid Cell Generation")
        print("="*80)
        
        grids = []
        grid_id = 0
        
        for i in range(n_lat):
            for j in range(n_lon):
                # Grid boundaries
                grid_lat_min = lat_min + i * self.grid_size_deg
                grid_lat_max = lat_min + (i + 1) * self.grid_size_deg
                grid_lon_min = lon_min + j * self.grid_size_deg
                grid_lon_max = lon_min + (j + 1) * self.grid_size_deg
                
                # Grid center point (will be used as "virtual station")
                center_lat = (grid_lat_min + grid_lat_max) / 2
                center_lon = (grid_lon_min + grid_lon_max) / 2
                
                grids.append({
                    'grid_id': grid_id,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'lat_min': grid_lat_min,
                    'lat_max': grid_lat_max,
                    'lon_min': grid_lon_min,
                    'lon_max': grid_lon_max,
                    'grid_row': i,
                    'grid_col': j
                })
                
                grid_id += 1
        
        df_grids = pd.DataFrame(grids)
        print(f"\nGenerated {len(df_grids)} grid cells")
        
        return df_grids
    
    def add_temporal_features(self, df_grids):
        """Add temporal features for reference date"""
        print("\n" + "="*80)
        print("Temporal Feature Assignment")
        print("="*80)
        
        date = self.reference_date
        
        df_grids['date'] = date
        df_grids['year'] = date.year
        df_grids['month'] = date.month
        df_grids['day'] = date.day
        df_grids['day_of_week'] = date.weekday()  # 0=Monday
        df_grids['day_name'] = date.strftime('%A')
        
        # Derived temporal features
        df_grids['is_weekend'] = 0  # Monday is not weekend
        df_grids['is_holiday'] = 0  # July 15 is not a holiday
        df_grids['season'] = 'Summer'
        
        print(f"\nReference date: {date.strftime('%Y-%m-%d (%A)')}")
        print(f"Season: Summer, Weekday: Monday, Holiday: No")
        
        return df_grids
    
    def add_weather_features(self, df_grids):
        """Add weather features from actual 2024-07-15 data"""
        print("\n" + "="*80)
        print("Weather Feature Assignment")
        print("="*80)
        
        # Load weather data from existing stations
        weather_file = self.project_root / "data" / "external" / "weather" / "station_daily_with_weather.parquet"
        
        if not weather_file.exists():
            print("Weather file not found, using default summer values")
            # Use typical summer weather
            weather_features = {
                'temperature_2m_max': 28.0,
                'temperature_2m_min': 20.0,
                'temperature_2m_mean': 24.0,
                'precipitation_sum': 0.0,
                'rain_sum': 0.0,
                'snowfall_sum': 0.0,
                'precipitation_hours': 0.0,
                'windspeed_10m_max': 15.0,
                'windgusts_10m_max': 25.0
            }
        else:
            # Load actual weather from 2024-07-15
            df_weather = pd.read_parquet(weather_file)
            df_weather['date'] = pd.to_datetime(df_weather['date'])
            
            # Filter to reference date
            df_date = df_weather[df_weather['date'] == self.reference_date]
            
            if len(df_date) > 0:
                # Use average weather across all stations on that day
                weather_features = {
                    'temperature_2m_max': df_date['temperature_2m_max'].mean(),
                    'temperature_2m_min': df_date['temperature_2m_min'].mean(),
                    'temperature_2m_mean': df_date['temperature_2m_mean'].mean(),
                    'precipitation_sum': df_date['precipitation_sum'].mean(),
                    'rain_sum': df_date['rain_sum'].mean(),
                    'snowfall_sum': df_date['snowfall_sum'].mean(),
                    'precipitation_hours': df_date['precipitation_hours'].mean(),
                    'windspeed_10m_max': df_date['windspeed_10m_max'].mean(),
                    'windgusts_10m_max': df_date['windgusts_10m_max'].mean()
                }
                print(f"Loaded weather from 2024-07-15")
            else:
                print("No weather data for 2024-07-15, using typical summer values")
                weather_features = {
                    'temperature_2m_max': 28.0,
                    'temperature_2m_min': 20.0,
                    'temperature_2m_mean': 24.0,
                    'precipitation_sum': 0.0,
                    'rain_sum': 0.0,
                    'snowfall_sum': 0.0,
                    'precipitation_hours': 0.0,
                    'windspeed_10m_max': 15.0,
                    'windgusts_10m_max': 25.0
                }
        
        # Add weather features to all grids (same weather for all)
        for feature, value in weather_features.items():
            df_grids[feature] = value
        
        print(f"\nWeather (2024-07-15): "
              f"Temp {weather_features['temperature_2m_max']:.1f}°C / "
              f"{weather_features['temperature_2m_mean']:.1f}°C / "
              f"{weather_features['temperature_2m_min']:.1f}°C, "
              f"Precip {weather_features['precipitation_sum']:.1f}mm, "
              f"Wind {weather_features['windspeed_10m_max']:.1f}km/h")
        
        return df_grids
    
    def save_results(self, df_grids):
        """Save grid features"""
        print("\n" + "="*80)
        print("Output Generation")
        print("="*80)
        
        output_file = self.output_dir / "grid_base_features.parquet"
        df_grids.to_parquet(output_file, index=False)
        print(f"\nSaved: {output_file} ({len(df_grids)} records)")
        
        sample_file = self.output_dir / "grid_base_features_sample.csv"
        df_grids.head(100).to_csv(sample_file, index=False)
        print(f"Saved: {sample_file} ({min(100, len(df_grids))} records)")
        
        print(f"\nFeatures: {len(df_grids.columns)} columns")
        print(f"  Grid: grid_id, center_lat/lon, boundaries, row/col")
        print(f"  Temporal: date, year, month, day, day_of_week, season, is_weekend, is_holiday")
        print(f"  Weather: temperature, precipitation, wind")
        
        return output_file
    
    def run(self):
        """Execute grid creation pipeline"""
        print("\n" + "="*80)
        print("Grid Feature Creation Pipeline")
        print("="*80)
        
        lat_min, lat_max, lon_min, lon_max, n_lat, n_lon = self.get_boston_boundaries()
        df_grids = self.create_grids(lat_min, lat_max, lon_min, lon_max, n_lat, n_lon)
        df_grids = self.add_temporal_features(df_grids)
        df_grids = self.add_weather_features(df_grids)
        output_file = self.save_results(df_grids)
        
        print("\n" + "="*80)
        print("Pipeline Complete")
        print("="*80)
        
        return df_grids


def main():
    creator = GridFeatureCreator()
    df_grids = creator.run()


if __name__ == "__main__":
    main()

