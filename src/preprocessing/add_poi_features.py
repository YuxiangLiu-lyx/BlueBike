"""
Add Points of Interest (POI) features from OpenStreetMap.

Enriches station data with urban context features including transit access,
commercial activity, offices, parks, and educational institutions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import osmnx as ox
from shapely.geometry import Point
import time
import json

class POIFeatureExtractor:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.input_file = self.project_root / "data" / "processed" / "daily" / "daily_with_xgb_features.parquet"
        
        # Ensure output directory exists
        output_dir = self.project_root / "data" / "processed" / "daily"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_file = output_dir / "daily_with_poi_features.parquet"
        self.sample_file = output_dir / "daily_with_poi_features_sample.csv"
        
        # Cache directory for POI data by station
        self.cache_dir = self.project_root / "data" / "external" / "poi_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.radius = 500  # Search radius in meters
        
        ox.settings.use_cache = True
        ox.settings.log_console = False
    
    def get_unique_stations(self, df):
        """Get unique stations with their coordinates."""
        stations = df[['station_name', 'latitude', 'longitude']].drop_duplicates()
        print(f"\nFound {len(stations)} unique stations to process")
        return stations
    
    def load_cached_poi(self, station_name):
        """Load cached POI data for a station."""
        cache_file = self.cache_dir / f"{station_name.replace('/', '_')}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_cached_poi(self, station_name, poi_data):
        """Save POI data to cache."""
        cache_file = self.cache_dir / f"{station_name.replace('/', '_')}.json"
        with open(cache_file, 'w') as f:
            json.dump(poi_data, f)
    
    def extract_poi_for_station(self, lat, lon, station_name):
        """Extract POI features for a single station from OpenStreetMap."""
        
        cached = self.load_cached_poi(station_name)
        if cached is not None:
            return cached
        
        poi_features = {}
        
        try:
            point = Point(lon, lat)
            
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
            
            poi_features['total_amenities'] = sum([
                poi_features.get('nearby_restaurants_count', 0),
                poi_features.get('nearby_shops_count', 0),
                poi_features.get('nearby_banks_count', 0)
            ])
            
            poi_features['is_transit_hub'] = 1 if poi_features.get('nearby_subway_stations_count', 0) > 0 else 0
            poi_features['is_commercial_area'] = 1 if poi_features.get('nearby_shops_count', 0) > 10 else 0
            poi_features['is_near_university'] = 1 if poi_features.get('nearby_universities_count', 0) > 0 else 0
            
            self.save_cached_poi(station_name, poi_features)
            
            return poi_features
            
        except Exception as e:
            print(f"  Error processing {station_name}: {e}")
            # Return default values
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
                'nearby_banks_count': 0,
                'total_amenities': 0,
                'is_transit_hub': 0,
                'is_commercial_area': 0,
                'is_near_university': 0
            }
    
    def add_poi_features(self):
        """Add POI features to the dataset."""
        print("\n" + "="*80)
        print("Adding POI Features from OpenStreetMap")
        print("="*80)
        
        # Load existing data
        print(f"\nLoading data from: {self.input_file}")
        df = pd.read_parquet(self.input_file)
        print(f"Loaded {len(df)} records")
        
        stations = self.get_unique_stations(df)
        
        print(f"\nExtracting POI features (radius={self.radius}m)...\n")
        
        station_poi_map = {}
        
        for i, (idx, row) in enumerate(stations.iterrows(), 1):
            station_name = row['station_name']
            lat = row['latitude']
            lon = row['longitude']
            
            print(f"[{i}/{len(stations)}] {station_name}")
            
            poi_features = self.extract_poi_for_station(lat, lon, station_name)
            station_poi_map[station_name] = poi_features
        
        # Merge POI features back to main dataframe
        print("\nMerging POI features with main dataset...")
        
        poi_columns = list(next(iter(station_poi_map.values())).keys())
        for col in poi_columns:
            df[col] = df['station_name'].map(lambda x: station_poi_map.get(x, {}).get(col, 0))
        
        # Save results
        print(f"\nSaving enriched data to: {self.output_file}")
        df.to_parquet(self.output_file, index=False)
        print(f"Saved {len(df)} records with {len(poi_columns)} POI features")
        
        print(f"\nCreating sample CSV: {self.sample_file}")
        sample_df = df.head(10000)
        sample_df.to_csv(self.sample_file, index=False)
        print(f"Saved sample with {len(sample_df)} records")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("POI Feature Summary")
        print("="*80)
        
        print(f"\nStations by type:")
        print(f"  Transit hubs: {df['is_transit_hub'].sum()} stations")
        print(f"  Near universities: {df['is_near_university'].sum()} stations")
        print(f"  Commercial areas: {df['is_commercial_area'].sum()} stations")
        
        print(f"\nPOI statistics (per station):")
        print(f"  Avg subway stations nearby: {df['nearby_subway_stations_count'].mean():.1f}")
        print(f"  Avg restaurants nearby: {df['nearby_restaurants_count'].mean():.1f}")
        print(f"  Avg shops nearby: {df['nearby_shops_count'].mean():.1f}")
        print(f"  Avg park area: {df['nearby_parks_area_sqm'].mean():.0f} m²")
        
        print("\nPOI feature extraction complete")
        return df

def main():
    extractor = POIFeatureExtractor()
    extractor.add_poi_features()

if __name__ == "__main__":
    main()

