"""
XGBoost model with POI features for daily bike demand prediction
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class XGBoostPredictorWithPOI:
    def __init__(self):
        self.project_root = get_project_root()
        self.data_dir = self.project_root / "data" / "processed" / "daily"
        self.results_dir = self.project_root / "results" / "xgboost_with_poi"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def load_data(self):
        """Load processed data with POI features"""
        print("\n" + "="*60)
        print("Loading Data with POI Features")
        print("="*60)
        
        data_file = self.data_dir / "daily_with_poi_features.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                "Please run: python src/preprocessing/add_poi_features.py"
            )
        
        df = pd.read_parquet(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df):,} records")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def prepare_train_test_split(self, df):
        """Split data into train and test sets"""
        print("\n" + "="*60)
        print("Train/Test Split")
        print("="*60)
        
        # Train: 2015-2023 (excluding 2020), Test: 2024
        train_mask = (df['year'] >= 2015) & (df['year'] <= 2023) & (df['year'] != 2020)
        test_mask = (df['year'] == 2024)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"\nTraining data:")
        print(f"  Years: 2015-2023 (excluding 2020)")
        print(f"  Records: {len(train_df):,}")
        print(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        print(f"  Total departures: {train_df['departure_count'].sum():,}")
        
        print(f"\nTest data:")
        print(f"  Year: 2024")
        print(f"  Records: {len(test_df):,}")
        print(f"  Date range: {test_df['date'].min()} to {test_df['date'].max()}")
        print(f"  Total departures: {test_df['departure_count'].sum():,}")
        
        return train_df, test_df
    
    def prepare_features(self, train_df, test_df):
        """Prepare features for XGBoost including POI"""
        print("\n" + "="*60)
        print("Feature Engineering (with POI)")
        print("="*60)
        
        # Base features
        numeric_features = [
            'latitude', 'longitude',
            'month', 'day', 'day_of_week',
            'nearby_avg_popularity'
        ]
        
        categorical_features = [
            'season', 'is_weekend', 'is_holiday'
        ]
        
        # POI features
        poi_features = [
            'nearby_subway_stations_count',
            'nearby_bus_stops_count',
            'nearby_restaurants_count',
            'nearby_shops_count',
            'nearby_offices_count',
            'nearby_universities_count',
            'nearby_schools_count',
            'nearby_hospitals_count',
            'nearby_banks_count',
            'nearby_parks_area_sqm'
        ]
        
        # Weather features
        weather_features = [
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max'
        ]
        
        # Check available features
        available_poi = [col for col in poi_features if col in train_df.columns]
        available_weather = [col for col in weather_features if col in train_df.columns]
        
        all_features = numeric_features + categorical_features + available_poi + available_weather
        
        print(f"\nFeature counts:")
        print(f"  Base features: {len(numeric_features + categorical_features)}")
        print(f"  POI features: {len(available_poi)}")
        print(f"  Weather features: {len(available_weather)}")
        print(f"  Total features: {len(all_features)}")
        
        # Prepare X and y
        X_train = train_df[all_features].copy()
        y_train = train_df['departure_count'].values
        
        X_test = test_df[all_features].copy()
        y_test = test_df['departure_count'].values
        
        # Encode categorical features
        print("\nEncoding categorical features...")
        for col in categorical_features:
            if col == 'season':
                season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
                X_train[col] = X_train[col].map(season_map)
                X_test[col] = X_test[col].map(season_map)
        
        # Handle missing values
        if available_weather:
            for col in available_weather:
                train_missing = X_train[col].isna().sum()
                test_missing = X_test[col].isna().sum()
                if train_missing > 0 or test_missing > 0:
                    median_val = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_val)
                    X_test[col] = X_test[col].fillna(median_val)
        
        if available_poi:
            for col in available_poi:
                X_train[col] = X_train[col].fillna(0)
                X_test[col] = X_test[col].fillna(0)
        
        print(f"\nFeature matrix shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test, all_features
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model with POI features"""
        print("\n" + "="*60)
        print("Training XGBoost Model (with POI)")
        print("="*60)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 0.2,
            'reg_alpha': 1.0,
            'reg_lambda': 5.0,
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("\nHyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        self.model = xgb.XGBRegressor(**params)
        
        print("\nTraining model...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        print("Training complete")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, test_df):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        total_actual = y_test.sum()
        total_predicted = y_pred.sum()
        total_error = total_predicted - total_actual
        total_error_pct = (total_error / total_actual) * 100
        
        mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
        mpe = np.mean((y_pred - y_test) / np.maximum(y_test, 1)) * 100
        
        total_abs_error = np.abs(y_test - y_pred).sum()
        total_squared_error = ((y_test - y_pred) ** 2).sum()
        overall_accuracy = 100 - (total_abs_error / total_actual) * 100
        
        print(f"\nPerformance Metrics:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  MPE: {mpe:.2f}%")
        print(f"\nTotal Predictions:")
        print(f"  Actual Total: {total_actual:,.0f}")
        print(f"  Predicted Total: {total_predicted:,.0f}")
        print(f"  Total Error: {total_error:,.0f} ({total_error_pct:+.2f}%)")
        print(f"  Overall Accuracy: {overall_accuracy:.2f}%")
        
        # Create horizontal format metrics (matching original xgboost_model.py format)
        metrics_df = pd.DataFrame({
            'Total_Actual': [total_actual],
            'Total_Predicted': [total_predicted],
            'Total_Error': [total_error],
            'Total_Error_Pct': [total_error_pct],
            'Total_Absolute_Error': [total_abs_error],
            'Total_Squared_Error': [total_squared_error],
            'Overall_Accuracy': [overall_accuracy],
            'MAE': [mae],
            'RMSE': [rmse],
            'R2': [r2],
            'MAPE': [mape],
            'MPE': [mpe]
        })
        
        metrics_file = self.results_dir / "xgboost_metrics_with_poi.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nMetrics saved to: {metrics_file}")
        
        return y_pred, metrics_df
    
    def save_predictions(self, test_df, y_pred):
        """Save predictions to CSV"""
        print("\n" + "="*60)
        print("Saving Predictions")
        print("="*60)
        
        predictions_df = test_df[['date', 'station_name', 'departure_count']].copy()
        predictions_df['predicted_departure_count'] = y_pred
        predictions_df['error'] = predictions_df['departure_count'] - predictions_df['predicted_departure_count']
        predictions_df['absolute_error'] = predictions_df['error'].abs()
        predictions_df['percentage_error'] = (predictions_df['error'] / predictions_df['departure_count'].replace(0, np.nan)) * 100
        
        predictions_file = self.results_dir / "xgboost_predictions_2024_with_poi.csv"
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Predictions saved to: {predictions_file}")
        
        return predictions_df
    
    def save_feature_importance(self, feature_names):
        """Save and visualize feature importance"""
        print("\n" + "="*60)
        print("Feature Importance")
        print("="*60)
        
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 important features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:30s} : {row['importance']:.4f}")
        
        importance_file = self.results_dir / "feature_importance_with_poi.csv"
        importance_df.to_csv(importance_file, index=False)
        print(f"\nFeature importance saved to: {importance_file}")
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance (XGBoost with POI)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_plot = self.results_dir / "feature_importance_with_poi.png"
        plt.savefig(importance_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {importance_plot}")
    
    def save_model(self):
        """Save trained model"""
        model_file = self.results_dir / "xgboost_model_with_poi.pkl"
        joblib.dump(self.model, model_file)
        print(f"\nModel saved to: {model_file}")


def main():
    predictor = XGBoostPredictorWithPOI()
    
    # Load data
    df = predictor.load_data()
    
    # Split data
    train_df, test_df = predictor.prepare_train_test_split(df)
    
    # Prepare features
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_features(train_df, test_df)
    
    # Train model
    predictor.train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    y_pred, metrics_df = predictor.evaluate_model(X_test, y_test, test_df)
    
    # Save predictions
    predictor.save_predictions(test_df, y_pred)
    
    # Save feature importance
    predictor.save_feature_importance(feature_names)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "="*60)
    print("XGBoost with POI training complete")
    print("="*60)
    print(f"\nResults saved in: {predictor.results_dir}")


if __name__ == "__main__":
    main()

