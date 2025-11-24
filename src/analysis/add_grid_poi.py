"""
Add POI features to grid cells.

Extracts POI features for each grid cell center point using
OpenStreetMap data via OSMnx, with caching to avoid redundant API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
import json
import time


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class GridPOIExtractor:
    def __init__(self):
        self.project_root = get_project_root()
        self.input_file = self.project_root / "data" / "processed" / "grid" / "grid_base_features.parquet"
        self.output_file = self.project_root / "data" / "processed" / "grid" / "grid_with_poi_features.parquet"
        self.sample_file = self.project_root / "data" / "processed" / "grid" / "grid_with_poi_features_sample.csv"
        
        self.cache_dir = self.project_root / "data" / "external" / "grid_poi_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.radius = 500
        
        ox.settings.use_cache = True
        ox.settings.log_console = False
    
    def load_cached_poi(self, grid_id):
        """Load cached POI data"""
        cache_file = self.cache_dir / f"grid_{grid_id}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_cached_poi(self, grid_id, poi_data):
        """Save POI data to cache"""
        cache_file = self.cache_dir / f"grid_{grid_id}.json"
        with open(cache_file, 'w') as f:
            json.dump(poi_data, f)
    
    def extract_poi_for_grid(self, grid_id, lat, lon):
        """Extract POI features for grid center point"""
        
        cached = self.load_cached_poi(grid_id)
        if cached is not None:
            return cached
        
        poi_features = {}
        
        try:
            tags_dict = {
                'subway_stations': {'railway': 'station', 'station': 'subway'},
                'bus_stops': {'highway': 'bus_stop'},
                'restaurants': {'amenity': ['restaurant', 'cafe', 'fast_food']},
                'shops': {'shop': True},
                'offices': {'office': True},
                'parks': {'leisure': 'park'},
                'universities': {'amenity': 'university'},
                'schools': {'amenity': 'school'},
                'hospitals': {'amenity': 'hospital'},
                'banks': {'amenity': 'bank'}
            }
            
            for poi_type, tags in tags_dict.items():
                try:
                    pois = ox.features_from_point((lat, lon), tags=tags, dist=self.radius)
                    
                    if poi_type == 'parks':
                        if not pois.empty and 'geometry' in pois.columns:
                            pois_proj = pois.to_crs('EPSG:32619')
                            total_area = pois_proj['geometry'].area.sum()
                            poi_features[f'nearby_{poi_type}_area_sqm'] = total_area
                        else:
                            poi_features[f'nearby_{poi_type}_area_sqm'] = 0
                    else:
                        poi_features[f'nearby_{poi_type}_count'] = len(pois) if not pois.empty else 0
                    
                    time.sleep(0.5)
                    
                except Exception as e:
                    if poi_type == 'parks':
                        poi_features[f'nearby_{poi_type}_area_sqm'] = 0
                    else:
                        poi_features[f'nearby_{poi_type}_count'] = 0
            
            self.save_cached_poi(grid_id, poi_features)
            
            return poi_features
            
        except Exception as e:
            return {
                'nearby_subway_stations_count': 0,
                'nearby_bus_stops_count': 0,
                'nearby_restaurants_count': 0,
                'nearby_shops_count': 0,
                'nearby_offices_count': 0,
                'nearby_parks_area_sqm': 0,
                'nearby_universities_count': 0,
                'nearby_schools_count': 0,
                'nearby_hospitals_count': 0,
                'nearby_banks_count': 0
            }
    
    def add_poi_features(self):
        """Add POI features to grid dataset"""
        print("\n" + "="*80)
        print("POI Feature Extraction")
        print("="*80)
        
        print(f"\nLoading: {self.input_file}")
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Grid base features not found: {self.input_file}\n"
                "Please run: python src/analysis/create_grid_features.py"
            )
        
        df = pd.read_parquet(self.input_file)
        print(f"Loaded: {len(df)} grid cells")
        
        print(f"\nExtraction radius: {self.radius}m")
        print(f"Cache directory: {self.cache_dir}")
        
        cached_count = sum(1 for grid_id in df['grid_id'] if self.load_cached_poi(grid_id) is not None)
        print(f"Cached: {cached_count}/{len(df)} grids")
        
        if cached_count < len(df):
            remaining = len(df) - cached_count
            print(f"Remaining: {remaining} grids\n")
        
        grid_poi_map = {}
        
        for idx, row in df.iterrows():
            grid_id = row['grid_id']
            lat = row['center_lat']
            lon = row['center_lon']
            
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"[{idx + 1}/{len(df)}] Grid {grid_id} ({lat:.4f}, {lon:.4f})")
            
            poi_features = self.extract_poi_for_grid(grid_id, lat, lon)
            grid_poi_map[grid_id] = poi_features
        
        print(f"\n{'='*80}")
        print("Merging POI features")
        
        poi_columns = list(next(iter(grid_poi_map.values())).keys())
        for col in poi_columns:
            df[col] = df['grid_id'].map(lambda x: grid_poi_map.get(x, {}).get(col, 0))
        
        print(f"\nSaving: {self.output_file}")
        df.to_parquet(self.output_file, index=False)
        print(f"Saved: {len(df)} grids with {len(poi_columns)} POI features")
        
        print(f"\nSaving sample: {self.sample_file}")
        sample_df = df.head(100)
        sample_df.to_csv(self.sample_file, index=False)
        print(f"Saved: {len(sample_df)} grids")
        
        print(f"\n{'='*80}")
        print("POI Statistics")
        print("="*80)
        
        print(f"\nAverage per grid:")
        print(f"  Subway: {df['nearby_subway_stations_count'].mean():.1f}")
        print(f"  Bus stops: {df['nearby_bus_stops_count'].mean():.1f}")
        print(f"  Restaurants: {df['nearby_restaurants_count'].mean():.1f}")
        print(f"  Shops: {df['nearby_shops_count'].mean():.1f}")
        print(f"  Offices: {df['nearby_offices_count'].mean():.1f}")
        print(f"  Park area: {df['nearby_parks_area_sqm'].mean():.0f} m²")
        
        print(f"\nTop 10 grids by POI count:")
        df['total_poi_count'] = (
            df['nearby_subway_stations_count'] +
            df['nearby_bus_stops_count'] +
            df['nearby_restaurants_count'] +
            df['nearby_shops_count'] +
            df['nearby_offices_count'] +
            df['nearby_universities_count'] +
            df['nearby_schools_count'] +
            df['nearby_hospitals_count'] +
            df['nearby_banks_count']
        )
        top_grids = df.nlargest(10, 'total_poi_count')[['grid_id', 'center_lat', 'center_lon', 'total_poi_count']]
        for _, row in top_grids.iterrows():
            print(f"  Grid {int(row['grid_id']):4d} ({row['center_lat']:.4f}, {row['center_lon']:.4f}): {int(row['total_poi_count'])} POIs")
        
        print(f"\n{'='*80}")
        print("Extraction Complete")
        print("="*80)
        
        return df


def main():
    extractor = GridPOIExtractor()
    extractor.add_poi_features()


if __name__ == "__main__":
    main()

