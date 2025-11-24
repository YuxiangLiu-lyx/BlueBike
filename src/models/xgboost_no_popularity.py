"""
XGBoost model WITHOUT latitude, longitude, and nearby_avg_popularity
Testing if model can predict demand using only temporal and weather features
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class XGBoostNoPopularity:
    def __init__(self):
        self.project_root = get_project_root()
        self.data_dir = self.project_root / "data" / "processed" / "daily"
        self.results_dir = self.project_root / "results" / "xgboost_no_popularity"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        
    def load_data(self):
        print("\n" + "="*60)
        print("Loading Data (NO location features)")
        print("="*60)
        
        data_file = self.data_dir / "daily_with_xgb_features.parquet"
        df = pd.read_parquet(data_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"  Loaded {len(df):,} records")
        return df
    
    def prepare_train_test_split(self, df):
        train_mask = (df['year'] >= 2015) & (df['year'] <= 2023) & (df['year'] != 2020)
        test_mask = (df['year'] == 2024)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"\nTraining: {len(train_df):,} records")
        print(f"Test: {len(test_df):,} records")
        
        return train_df, test_df
    
    def prepare_features(self, train_df, test_df):
        print("\n" + "="*60)
        print("Feature Engineering (NO location features)")
        print("="*60)
        
        # Features WITHOUT latitude, longitude, nearby_avg_popularity
        numeric_features = [
            'month', 'day', 'day_of_week'
        ]
        
        categorical_features = [
            'season', 'is_weekend', 'is_holiday'
        ]
        
        weather_features = [
            'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
            'precipitation_sum', 'rain_sum', 'snowfall_sum',
            'precipitation_hours', 'windspeed_10m_max', 'windgusts_10m_max'
        ]
        
        available_weather = [col for col in weather_features if col in train_df.columns]
        all_features = numeric_features + categorical_features + available_weather
        
        print(f"\nTotal features: {len(all_features)}")
        print(f"  Location: 0 (excluded)")
        print(f"  Temporal: {len(numeric_features) + len(categorical_features)}")
        print(f"  Weather: {len(available_weather)}")
        
        X_train = train_df[all_features].copy()
        y_train = train_df['departure_count'].values
        X_test = test_df[all_features].copy()
        y_test = test_df['departure_count'].values
        
        # Encode categorical
        for col in categorical_features:
            if col == 'season':
                season_map = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
                X_train[col] = X_train[col].map(season_map)
                X_test[col] = X_test[col].map(season_map)
        
        # Handle missing values
        if available_weather:
            for col in available_weather:
                median_val = X_train[col].median()
                X_train[col] = X_train[col].fillna(median_val)
                X_test[col] = X_test[col].fillna(median_val)
        
        return X_train, X_test, y_train, y_test, all_features
    
    def train_model(self, X_train, y_train, X_test, y_test):
        print("\n" + "="*60)
        print("Training XGBoost (NO location features)")
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
        
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        print("Training complete")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, test_df):
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
        
        print(f"\nMetrics:")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Total Error: {total_error_pct:+.2f}%")
        
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
        
        metrics_file = self.results_dir / "xgboost_metrics_no_popularity.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nMetrics saved")
        
        return y_pred, metrics_df
    
    def save_results(self, test_df, y_pred, feature_names):
        predictions_df = test_df[['date', 'station_name', 'departure_count']].copy()
        predictions_df['predicted_departure_count'] = y_pred
        
        predictions_file = self.results_dir / "xgboost_predictions_2024_no_popularity.csv"
        predictions_df.to_csv(predictions_file, index=False)
        
        model_file = self.results_dir / "xgboost_model_no_popularity.pkl"
        joblib.dump(self.model, model_file)
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_file = self.results_dir / "feature_importance_no_popularity.csv"
        importance_df.to_csv(importance_file, index=False)
        
        print(f"Results saved to: {self.results_dir}")


def main():
    predictor = XGBoostNoPopularity()
    df = predictor.load_data()
    train_df, test_df = predictor.prepare_train_test_split(df)
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_features(train_df, test_df)
    predictor.train_model(X_train, y_train, X_test, y_test)
    y_pred, metrics_df = predictor.evaluate_model(X_test, y_test, test_df)
    predictor.save_results(test_df, y_pred, feature_names)
    
    print("\n" + "="*60)
    print("XGBoost (No Location Features) training complete")
    print("="*60)


if __name__ == "__main__":
    main()

