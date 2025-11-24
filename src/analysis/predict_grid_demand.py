"""
Predict bike demand for grid cells using XGBoost POI-only model.
Compare predicted high-demand areas with actual station distribution.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def get_project_root():
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class GridDemandPredictor:
    def __init__(self):
        self.project_root = get_project_root()
        
        # Input files
        self.grid_file = self.project_root / "data" / "processed" / "grid" / "grid_with_poi_features.parquet"
        self.model_file = self.project_root / "results" / "xgboost_poi_only" / "xgboost_model_poi_only.pkl"
        self.station_file = self.project_root / "data" / "processed" / "daily" / "daily_with_poi_features.parquet"
        
        # Output
        self.output_dir = self.project_root / "results" / "grid_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.output_dir / "grid_predictions.parquet"
        self.comparison_file = self.output_dir / "grid_comparison.csv"
    
    def load_model(self):
        """Load trained XGBoost model"""
        print("Loading XGBoost model...")
        with open(self.model_file, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded: {self.model_file.name}")
    
    def prepare_grid_features(self):
        """Prepare grid features for prediction"""
        print("\nLoading grid features...")
        df_grid = pd.read_parquet(self.grid_file)
        print(f"Loaded: {len(df_grid)} grid cells")
        
        # Features in exact order as xgboost_poi_only model training
        # Order: temporal -> season -> POI -> weather
        feature_order = [
            'month', 'day', 'day_of_week', 'season', 'is_weekend', 'is_holiday',
            'nearby_subway_stations_count', 'nearby_bus_stops_count',
            'nearby_restaurants_count', 'nearby_shops_count',
            'nearby_offices_count', 'nearby_universities_count',
            'nearby_schools_count', 'nearby_hospitals_count',
            'nearby_banks_count', 'nearby_parks_area_sqm',
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max'
        ]
        
        # Check available features
        available_features = []
        for feature in feature_order:
            if feature in df_grid.columns:
                available_features.append(feature)
            else:
                print(f"  Warning: {feature} not found")
        
        print(f"Available features: {len(available_features)}")
        
        # Prepare feature matrix
        X = df_grid[available_features].copy()
        
        # Fill missing values
        for col in X.columns:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median())
        
        # Encode categorical
        if 'season' in X.columns:
            season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
            X['season'] = X['season'].map(season_mapping)
        
        return df_grid, X, available_features
    
    def predict_demand(self, X):
        """Predict demand for all grid cells"""
        print("\nPredicting demand for grid cells...")
        predictions = self.model.predict(X)
        
        # Ensure no negative predictions
        predictions = np.maximum(predictions, 0)
        
        print(f"Predictions range: {predictions.min():.1f} - {predictions.max():.1f}")
        print(f"Mean predicted demand: {predictions.mean():.1f}")
        print(f"Median predicted demand: {np.median(predictions):.1f}")
        
        return predictions
    
    def calculate_actual_demand(self, df_grid):
        """Calculate actual demand for grids based on existing stations"""
        print("\nCalculating actual demand from stations...")
        
        # Load station data (2024 only for fair comparison)
        df_station = pd.read_parquet(self.station_file)
        df_station_2024 = df_station[df_station['year'] == 2024].copy()
        
        print(f"Loaded: {len(df_station_2024)} station-day records from 2024")
        
        # Calculate average daily demand per station
        station_avg = df_station_2024.groupby(['station_name', 'latitude', 'longitude'])['departure_count'].mean().reset_index()
        station_avg.columns = ['station_name', 'lat', 'lon', 'avg_daily_demand']
        
        print(f"Stations in 2024: {len(station_avg)}")
        
        # Assign each station to grid based on boundaries
        df_grid['actual_demand'] = 0.0
        df_grid['station_count'] = 0
        
        stations_assigned = 0
        stations_not_assigned = 0
        
        for _, station in station_avg.iterrows():
            # Find grid that contains this station
            mask = (
                (df_grid['lat_min'] <= station['lat']) & 
                (station['lat'] < df_grid['lat_max']) &
                (df_grid['lon_min'] <= station['lon']) & 
                (station['lon'] < df_grid['lon_max'])
            )
            
            matching_grids = df_grid[mask]
            
            if len(matching_grids) > 0:
                # Station is within grid boundaries
                grid_idx = matching_grids.index[0]
                df_grid.loc[grid_idx, 'actual_demand'] += station['avg_daily_demand']
                df_grid.loc[grid_idx, 'station_count'] += 1
                stations_assigned += 1
            else:
                # Station not in any grid (edge case)
                stations_not_assigned += 1
        
        grids_with_stations = (df_grid['station_count'] > 0).sum()
        print(f"Stations assigned to grids: {stations_assigned}/{len(station_avg)}")
        print(f"Stations not assigned: {stations_not_assigned}")
        print(f"Grids with stations: {grids_with_stations}/{len(df_grid)}")
        print(f"Total actual demand: {df_grid['actual_demand'].sum():.0f}")
        
        return df_grid
    
    def analyze_comparison(self, df_results):
        """Analyze predicted vs actual demand"""
        print("\n" + "="*80)
        print("Demand Analysis")
        print("="*80)
        
        # Overall statistics
        total_predicted = df_results['predicted_demand'].sum()
        total_actual = df_results['actual_demand'].sum()
        
        print(f"\nOverall:")
        print(f"  Total predicted demand: {total_predicted:.0f}")
        print(f"  Total actual demand: {total_actual:.0f}")
        print(f"  Ratio: {total_predicted/total_actual:.2f}x")
        
        # High demand grids
        predicted_high = df_results.nlargest(100, 'predicted_demand')
        actual_high = df_results.nlargest(100, 'actual_demand')
        
        print(f"\nTop 100 grids by predicted demand:")
        print(f"  Have stations: {(predicted_high['station_count'] > 0).sum()}/100")
        print(f"  Average stations per grid: {predicted_high['station_count'].mean():.2f}")
        print(f"  Actual demand captured: {predicted_high['actual_demand'].sum():.0f} ({predicted_high['actual_demand'].sum()/total_actual*100:.1f}%)")
        
        print(f"\nTop 100 grids by actual demand:")
        print(f"  Average predicted demand: {actual_high['predicted_demand'].mean():.1f}")
        print(f"  Average actual demand: {actual_high['actual_demand'].mean():.1f}")
        
        # Identify opportunities
        df_opportunity = df_results[
            (df_results['predicted_demand'] > df_results['predicted_demand'].quantile(0.75)) & 
            (df_results['station_count'] == 0)
        ].copy()
        
        print(f"\nHigh-demand grids without stations:")
        print(f"  Count: {len(df_opportunity)}")
        print(f"  Average predicted demand: {df_opportunity['predicted_demand'].mean():.1f}")
        
        # Top opportunities
        top_opportunities = df_opportunity.nlargest(20, 'predicted_demand')
        print(f"\nTop 20 expansion opportunities:")
        for idx, row in top_opportunities.head(20).iterrows():
            print(f"  Grid {int(row['grid_id']):4d}: Predicted {row['predicted_demand']:.0f} "
                  f"at ({row['center_lat']:.4f}, {row['center_lon']:.4f})")
        
        return df_opportunity
    
    def save_results(self, df_results, df_opportunity):
        """Save prediction results"""
        print("\n" + "="*80)
        print("Saving Results")
        print("="*80)
        
        # Save full predictions
        df_results.to_parquet(self.predictions_file, index=False)
        print(f"\nSaved: {self.predictions_file}")
        
        # Save comparison summary
        comparison_cols = [
            'grid_id', 'center_lat', 'center_lon',
            'predicted_demand', 'actual_demand', 'station_count',
            'nearby_subway_stations_count', 'nearby_bus_stops_count',
            'nearby_restaurants_count', 'nearby_shops_count',
            'nearby_offices_count', 'nearby_universities_count',
            'nearby_schools_count', 'nearby_hospitals_count',
            'nearby_banks_count', 'nearby_parks_area_sqm'
        ]
        
        available_cols = [col for col in comparison_cols if col in df_results.columns]
        df_comparison = df_results[available_cols].copy()
        df_comparison = df_comparison.sort_values('predicted_demand', ascending=False)
        
        comparison_csv = self.output_dir / "grid_demand_comparison.csv"
        df_comparison.to_csv(comparison_csv, index=False)
        print(f"Saved: {comparison_csv}")
        
        # Save opportunities
        if len(df_opportunity) > 0:
            opportunity_csv = self.output_dir / "expansion_opportunities.csv"
            df_opportunity[available_cols].to_csv(opportunity_csv, index=False)
            print(f"Saved: {opportunity_csv}")
    
    def run(self):
        """Execute prediction pipeline"""
        print("="*80)
        print("Grid Demand Prediction Pipeline")
        print("="*80)
        
        # Load model
        self.load_model()
        
        # Prepare features
        df_grid, X, features = self.prepare_grid_features()
        
        # Predict
        predictions = self.predict_demand(X)
        df_grid['predicted_demand'] = predictions
        
        # Calculate actual
        df_grid = self.calculate_actual_demand(df_grid)
        
        # Analyze
        df_opportunity = self.analyze_comparison(df_grid)
        
        # Save
        self.save_results(df_grid, df_opportunity)
        
        print("\n" + "="*80)
        print("Prediction Complete")
        print("="*80)
        
        return df_grid


def main():
    predictor = GridDemandPredictor()
    predictor.run()


if __name__ == "__main__":
    main()

