import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class BaselinePredictor:
    
    def __init__(self, daily_data_path: str = None):
        project_root = get_project_root()
        
        if daily_data_path is None:
            daily_data_path = project_root / "data" / "processed" / "daily" / "daily_departures.parquet"
        
        self.daily_data_path = Path(daily_data_path)
        self.results_dir = project_root / "results" / "models"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self):
        """Load daily data and split into train/test sets"""
        print("Loading data...")
        df = pd.read_parquet(self.daily_data_path)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        print("\nFiltering data...")
        print(f"  Excluding 2020 (COVID-19)")
        print(f"  Excluding 2025 (incomplete)")
        
        df_filtered = df[~df['year'].isin([2020, 2025])].copy()
        print(f"  Remaining: {len(df_filtered):,} records")
        
        train_df = df_filtered[df_filtered['year'] < 2024].copy()
        test_df = df_filtered[df_filtered['year'] == 2024].copy()
        
        print(f"\nTrain: {len(train_df):,} | Test: {len(test_df):,}")
        
        return train_df, test_df
    
    def estimate_2024_totals(self, train_df):
        """Estimate 2024 total using average growth rate"""
        print("\nEstimating 2024 total...")
        
        total_2015 = train_df[train_df['year'] == 2015]['departure_count'].sum()
        total_2023 = train_df[train_df['year'] == 2023]['departure_count'].sum()
        
        years_span = 8
        avg_growth_rate = (total_2023 - total_2015) / total_2015 / years_span
        estimated_2024_total = total_2023 * (1 + avg_growth_rate)
        
        print(f"  2015 total: {total_2015:,.0f}")
        print(f"  2023 total: {total_2023:,.0f}")
        print(f"  Average growth rate: {avg_growth_rate*100:.2f}%/year")
        print(f"  Estimated 2024 total: {estimated_2024_total:,.0f}")
        
        station_2024_estimates = {'__global__': estimated_2024_total}
        return station_2024_estimates
    
    def predict_baseline(self, train_df, test_df, station_2024_estimates):
        """Proportional allocation based on historical patterns"""
        print("\nGenerating predictions...")
        
        global_hist_total = train_df['departure_count'].sum()
        estimated_2024_total = station_2024_estimates['__global__']
        
        # Calculate historical proportion for each station-date
        hist_station_day = train_df.groupby(['station_name', 'month', 'day'])['departure_count'].sum().reset_index()
        hist_station_day.rename(columns={'departure_count': 'hist_sum'}, inplace=True)
        hist_station_day['global_proportion'] = hist_station_day['hist_sum'] / global_hist_total
        
        result_df = test_df.merge(
            hist_station_day[['station_name', 'month', 'day', 'global_proportion']], 
            on=['station_name', 'month', 'day'], 
            how='left'
        )
        
        result_df['prediction'] = estimated_2024_total * result_df['global_proportion']
        
        # Set missing values to 0 (new stations or unseen combinations)
        missing_mask = result_df['prediction'].isna()
        if missing_mask.sum() > 0:
            print(f"  Skipping {missing_mask.sum():,} records without historical data")
            result_df.loc[missing_mask, 'prediction'] = 0
        
        # Normalize to ensure total matches estimated 2024 total (only for non-zero predictions)
        non_zero_mask = result_df['prediction'] > 0
        if non_zero_mask.sum() > 0:
            raw_total = result_df.loc[non_zero_mask, 'prediction'].sum()
            normalization_factor = estimated_2024_total / raw_total
            result_df.loc[non_zero_mask, 'prediction'] = result_df.loc[non_zero_mask, 'prediction'] * normalization_factor
        
        predictions = result_df['prediction'].clip(lower=0).values
        actuals = result_df['departure_count'].values
        valid_mask = result_df['prediction'] > 0
        
        pred_total = predictions.sum()
        actual_total = actuals.sum()
        print(f"\n  Historical total: {global_hist_total:,.0f}")
        print(f"  Actual 2024: {actual_total:,.0f}")
        print(f"  Predicted 2024: {pred_total:,.0f}")
        print(f"  Estimation error: {(estimated_2024_total - actual_total) / actual_total * 100:+.2f}%")
        
        return predictions, actuals, valid_mask.values
    
    def evaluate_predictions(self, predictions, actuals, valid_mask):
        """Calculate evaluation metrics (only for records with historical data)"""
        print("\n" + "="*60)
        print("Baseline Model Evaluation Results")
        print("="*60)
        
        # Filter to only valid predictions
        valid_predictions = predictions[valid_mask]
        valid_actuals = actuals[valid_mask]
        
        print(f"\nEvaluating {len(valid_predictions):,} records with historical data")
        print(f"Skipped {(~valid_mask).sum():,} records without historical data")
        
        mae = mean_absolute_error(valid_actuals, valid_predictions)
        rmse = np.sqrt(mean_squared_error(valid_actuals, valid_predictions))
        r2 = r2_score(valid_actuals, valid_predictions)
        mape = np.mean(np.abs((valid_actuals - valid_predictions) / (valid_actuals + 1e-10))) * 100
        mpe = np.mean((valid_predictions - valid_actuals) / (valid_actuals + 1e-10)) * 100
        median_ae = np.median(np.abs(valid_actuals - valid_predictions))
        
        total_actual = valid_actuals.sum()
        total_predicted = valid_predictions.sum()
        total_error = total_predicted - total_actual
        total_error_pct = total_error / total_actual * 100
        total_abs_error = np.abs(valid_predictions - valid_actuals).sum()
        total_squared_error = ((valid_predictions - valid_actuals) ** 2).sum()
        avg_accuracy = 1 - (total_abs_error / total_actual)
        
        print(f"\nOverall Metrics:")
        print(f"  Total Actual:                   {total_actual:,.0f} departures")
        print(f"  Total Predicted:                {total_predicted:,.0f} departures")
        print(f"  Total Error (signed):           {total_error:+,.0f} departures ({total_error_pct:+.2f}%)")
        print(f"  Total Absolute Error:           {total_abs_error:,.0f} departures")
        print(f"  Total Squared Error:            {total_squared_error:,.0f}")
        print(f"  Overall Accuracy:               {avg_accuracy*100:.2f}%")
        
        print(f"\nPer-Record Metrics:")
        print(f"  MAE (Mean Absolute Error):      {mae:.2f} departures/day")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.2f} departures/day")
        print(f"  R² Score:                       {r2:.4f}")
        print(f"  MAPE (Mean Abs % Error):        {mape:.2f}%")
        print(f"  MPE (Mean % Error, signed):     {mpe:+.2f}% (+ = overestimate, - = underestimate)")
        print(f"  Median Absolute Error:          {median_ae:.2f} departures/day")
        
        print(f"\nActual statistics:")
        print(f"  Mean:   {valid_actuals.mean():.2f}")
        print(f"  Median: {np.median(valid_actuals):.2f}")
        print(f"  Std:    {valid_actuals.std():.2f}")
        
        print(f"\nPrediction statistics:")
        print(f"  Mean:   {valid_predictions.mean():.2f}")
        print(f"  Median: {np.median(valid_predictions):.2f}")
        print(f"  Std:    {valid_predictions.std():.2f}")
        
        print("="*60)
        
        return {
            'Valid_Records': len(valid_predictions),
            'Skipped_Records': (~valid_mask).sum(),
            'Total_Actual': total_actual,
            'Total_Predicted': total_predicted,
            'Total_Error': total_error,
            'Total_Error_Pct': total_error_pct,
            'Total_Absolute_Error': total_abs_error,
            'Total_Squared_Error': total_squared_error,
            'Overall_Accuracy': avg_accuracy * 100,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MPE': mpe,
            'Median_AE': median_ae,
            'Actual_Mean': valid_actuals.mean(),
            'Predicted_Mean': valid_predictions.mean()
        }
    
    def save_results(self, test_df, predictions, actuals, metrics, valid_mask):
        """Save predictions and metrics to CSV"""
        print("\nSaving results...")
        
        results_df = test_df.copy()
        results_df['predicted_departure_count'] = predictions
        results_df['actual_departure_count'] = actuals
        
        # Calculate errors
        results_df['error'] = predictions - actuals
        results_df['absolute_error'] = np.abs(predictions - actuals)
        results_df['percentage_error'] = (predictions - actuals) / (actuals + 1e-10) * 100
        
        # Set error columns to NaN for skipped records (will show as empty in CSV)
        results_df.loc[~valid_mask, ['error', 'absolute_error', 'percentage_error']] = np.nan
        
        output_path = self.results_dir / "baseline_predictions_2024.csv"
        results_df.to_csv(output_path, index=False)
        print(f"  Predictions: {output_path}")
        
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.results_dir / "baseline_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"  Metrics: {metrics_path}")
    
    def run(self):
        """Execute baseline prediction pipeline"""
        print("="*60)
        print("Baseline Daily Prediction Model")
        print("="*60)
        
        train_df, test_df = self.load_and_prepare_data()
        station_2024_estimates = self.estimate_2024_totals(train_df)
        predictions, actuals, valid_mask = self.predict_baseline(train_df, test_df, station_2024_estimates)
        metrics = self.evaluate_predictions(predictions, actuals, valid_mask)
        self.save_results(test_df, predictions, actuals, metrics, valid_mask)
        
        print("\n" + "="*60)
        print("Complete")
        print("="*60)
        
        return predictions, actuals, metrics


def main():
    predictor = BaselinePredictor()
    predictions, actuals, metrics = predictor.run()


if __name__ == "__main__":
    main()

