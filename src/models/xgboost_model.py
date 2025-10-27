"""
XGBoost model for daily bike demand prediction
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


class XGBoostPredictor:
    def __init__(self):
        self.project_root = get_project_root()
        self.data_dir = self.project_root / "data" / "processed" / "daily"
        self.results_dir = self.project_root / "results" / "xgboost"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def load_data(self):
        """Load processed data with XGBoost features"""
        print("\n" + "="*60)
        print("Loading Data")
        print("="*60)
        
        data_file = self.data_dir / "daily_with_xgb_features.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                "Please run: python src/preprocessing/add_xgb_features.py"
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
        
        # Exclude 2020 (pandemic) and 2025 (incomplete)
        # Train: 2015-2023, Test: 2024
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
        """Prepare features for XGBoost"""
        print("\n" + "="*60)
        print("Feature Engineering")
        print("="*60)
        
        # Define feature columns
        numeric_features = [
            'latitude', 'longitude',
            'month', 'day', 'day_of_week',
            'nearby_avg_popularity'
        ]
        
        categorical_features = [
            'season', 'is_weekend', 'is_holiday'
        ]
        
        # Add weather features if available
        weather_features = [
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max'
        ]
        
        available_weather = [col for col in weather_features if col in train_df.columns]
        
        all_features = numeric_features + categorical_features + available_weather
        
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
            # is_weekend, is_holiday, station_popularity_tier are already numeric
        
        # Handle missing values in weather features
        if available_weather:
            print(f"\nWeather features available: {len(available_weather)}")
            for col in available_weather:
                train_missing = X_train[col].isna().sum()
                test_missing = X_test[col].isna().sum()
                if train_missing > 0 or test_missing > 0:
                    print(f"  {col}: {train_missing:,} train missing, {test_missing:,} test missing")
                    # Fill with median
                    median_val = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_val)
                    X_test[col] = X_test[col].fillna(median_val)
        else:
            print("\nNo weather features available")
        
        print(f"\nFinal feature count: {len(all_features)}")
        print(f"  Numeric: {len(numeric_features)}")
        print(f"  Categorical: {len(categorical_features)}")
        print(f"  Weather: {len(available_weather)}")
        print(f"\nFeature design philosophy:")
        print(f"  ✓ Time patterns: month, day_of_week, season, weekend, holiday")
        print(f"  ✓ Location: lat/lon + nearby area historical activity")
        print(f"  ✓ Weather: temperature, precipitation, wind")
        print(f"  ✗ No 'year' (prevents linear extrapolation)")
        print(f"  ✗ No prev_month stats (confounds seasonality)")
        
        # Store feature names
        self.feature_names = all_features
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.weather_features = available_weather
        
        return X_train, y_train, X_test, y_test, test_df
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        print("\n" + "="*60)
        print("Training XGBoost Model")
        print("="*60)
        
        # XGBoost parameters (tuned to prevent overfitting)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,                  # Reduced from 8 to prevent overfitting
            'learning_rate': 0.05,           # Reduced for more conservative learning
            'n_estimators': 500,             # Increased to compensate for lower learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,           # Increased from 3 for more regularization
            'gamma': 0.2,                    # Increased from 0.1 for more pruning
            'reg_alpha': 1.0,                # L1 regularization 
            'reg_lambda': 5.0,               # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': ['rmse', 'mae']
        }
        
        print("\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Train model with evaluation
        print("\nTraining...")
        self.model = xgb.XGBRegressor(**params)
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=50
        )
        
        print("\n✓ Training complete")
        
        return self.model
    
    def evaluate_predictions(self, y_true, y_pred, dataset_name="Test"):
        """Calculate evaluation metrics"""
        print("\n" + "="*60)
        print(f"{dataset_name} Set Evaluation")
        print("="*60)
        
        # Clip negative predictions to 0
        y_pred = np.clip(y_pred, 0, None)
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        mpe = np.mean((y_pred - y_true) / (y_true + 1e-10)) * 100
        median_ae = np.median(np.abs(y_true - y_pred))
        
        total_actual = y_true.sum()
        total_predicted = y_pred.sum()
        total_error = total_predicted - total_actual
        total_error_pct = total_error / total_actual * 100
        total_abs_error = np.abs(y_pred - y_true).sum()
        total_squared_error = ((y_pred - y_true) ** 2).sum()
        overall_accuracy = 1 - (total_abs_error / total_actual)
        
        print(f"\nOverall Metrics:")
        print(f"  Total Actual:                   {total_actual:,.0f} departures")
        print(f"  Total Predicted:                {total_predicted:,.0f} departures")
        print(f"  Total Error (signed):           {total_error:+,.0f} departures ({total_error_pct:+.2f}%)")
        print(f"  Total Absolute Error:           {total_abs_error:,.0f} departures")
        print(f"  Total Squared Error:            {total_squared_error:,.0f}")
        print(f"  Overall Accuracy:               {overall_accuracy*100:.2f}%")
        
        print(f"\nPer-Record Metrics:")
        print(f"  MAE (Mean Absolute Error):      {mae:.2f} departures/day")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} departures/day")
        print(f"  R² Score:                       {r2:.4f}")
        print(f"  MAPE (Mean Abs % Error):        {mape:.2f}%")
        print(f"  MPE (Mean % Error, signed):     {mpe:+.2f}% (+ = overestimate, - = underestimate)")
        print(f"  Median Absolute Error:          {median_ae:.2f} departures/day")
        
        print(f"\nActual statistics:")
        print(f"  Mean:   {y_true.mean():.2f}")
        print(f"  Median: {np.median(y_true):.2f}")
        print(f"  Std:    {y_true.std():.2f}")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean:   {y_pred.mean():.2f}")
        print(f"  Median: {np.median(y_pred):.2f}")
        print(f"  Std:    {y_pred.std():.2f}")
        
        print("="*60)
        
        return {
            'Total_Actual': total_actual,
            'Total_Predicted': total_predicted,
            'Total_Error': total_error,
            'Total_Error_Pct': total_error_pct,
            'Total_Absolute_Error': total_abs_error,
            'Total_Squared_Error': total_squared_error,
            'Overall_Accuracy': overall_accuracy * 100,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MPE': mpe,
            'Median_AE': median_ae,
            'Actual_Mean': y_true.mean(),
            'Predicted_Mean': y_pred.mean()
        }
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        print("\nGenerating feature importance plot...")
        
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        plt.figure(figsize=(12, 8))
        top_n = min(20, len(feature_importance))
        
        sns.barplot(
            data=feature_importance.head(top_n),
            x='importance',
            y='feature',
            palette='viridis'
        )
        
        plt.title('XGBoost Feature Importance (Top 20)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        output_file = self.results_dir / "feature_importance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_file}")
        
        # Save importance to CSV
        importance_csv = self.results_dir / "feature_importance.csv"
        feature_importance.to_csv(importance_csv, index=False)
        print(f"  Saved: {importance_csv}")
        
        # Print top 10
        print("\nTop 10 Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:.<40} {row['importance']:.4f}")
    
    def save_results(self, test_df, y_pred, metrics):
        """Save predictions and metrics"""
        print("\nSaving results...")
        
        # Save predictions
        results_df = test_df[['station_name', 'date', 'departure_count']].copy()
        results_df['predicted_departure_count'] = np.clip(y_pred, 0, None)
        results_df['error'] = results_df['predicted_departure_count'] - results_df['departure_count']
        results_df['absolute_error'] = np.abs(results_df['error'])
        results_df['percentage_error'] = (results_df['error'] / (results_df['departure_count'] + 1e-10)) * 100
        
        predictions_file = self.results_dir / "xgboost_predictions_2024.csv"
        results_df.to_csv(predictions_file, index=False)
        print(f"  Predictions: {predictions_file}")
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_file = self.results_dir / "xgboost_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"  Metrics: {metrics_file}")
        
        # Save model
        model_file = self.results_dir / "xgboost_model.pkl"
        joblib.dump(self.model, model_file)
        print(f"  Model: {model_file}")
    
    def run(self):
        """Run complete XGBoost training pipeline"""
        print("\n" + "="*60)
        print("XGBoost Daily Demand Prediction")
        print("="*60)
        
        # Load data
        df = self.load_data()
        
        # Split data
        train_df, test_df = self.prepare_train_test_split(df)
        
        # Prepare features
        X_train, y_train, X_test, y_test, test_df_full = self.prepare_features(train_df, test_df)
        
        # Train model
        self.train_model(X_train, y_train, X_test, y_test)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        metrics = self.evaluate_predictions(y_test, y_pred, "Test")
        
        # Plot feature importance
        self.plot_feature_importance()
        
        # Save results
        self.save_results(test_df_full, y_pred, metrics)
        
        print("\n" + "="*60)
        print("XGBoost Training Complete!")
        print("="*60)
        print(f"\nResults saved to: {self.results_dir}")


def main():
    predictor = XGBoostPredictor()
    predictor.run()


if __name__ == "__main__":
    main()

