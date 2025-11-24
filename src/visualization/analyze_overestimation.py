"""
Analyze days where XGBoost significantly overestimated demand
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import holidays


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def analyze_overestimation():
    """Analyze days with highest prediction errors and plot hourly patterns"""
    project_root = get_project_root()
    
    # Load XGBoost predictions
    predictions_file = project_root / "results" / "xgboost" / "xgboost_predictions_2024.csv"
    df_pred = pd.read_csv(predictions_file)
    df_pred['date'] = pd.to_datetime(df_pred['date'])
    
    # Calculate daily totals
    daily_totals = df_pred.groupby('date').agg({
        'departure_count': 'sum',
        'predicted_departure_count': 'sum'
    }).reset_index()
    
    daily_totals.rename(columns={
        'departure_count': 'actual',
        'predicted_departure_count': 'predicted'
    }, inplace=True)
    
    daily_totals['error'] = daily_totals['predicted'] - daily_totals['actual']
    daily_totals['error_pct'] = (daily_totals['error'] / daily_totals['actual']) * 100
    
    # Sort by error (highest overestimation first)
    daily_totals = daily_totals.sort_values('error', ascending=False)
    
    # Get top 6 overestimated days
    top_overestimated = daily_totals.head(6)
    
    print(f"\n{'='*80}")
    print("Top 5 Days with Highest Overestimation")
    print(f"{'='*80}")
    
    # Get US holidays for 2024
    us_holidays = holidays.US(years=2024)
    
    # Add day info
    top_overestimated['day_of_week'] = top_overestimated['date'].dt.day_name()
    top_overestimated['is_holiday'] = top_overestimated['date'].apply(lambda x: x in us_holidays)
    top_overestimated['holiday_name'] = top_overestimated['date'].apply(
        lambda x: us_holidays.get(x, 'No') if x in us_holidays else 'No'
    )
    
    for idx, row in top_overestimated.iterrows():
        print(f"\n{row['date'].strftime('%Y-%m-%d')} ({row['day_of_week']})")
        print(f"  Holiday: {row['holiday_name']}")
        print(f"  Actual: {row['actual']:,.0f} departures")
        print(f"  Predicted: {row['predicted']:,.0f} departures")
        print(f"  Error: +{row['error']:,.0f} ({row['error_pct']:+.1f}%)")
    
    print(f"\n{'='*80}")
    
    # Save summary to text file
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "overestimation_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Top 5 Days with Highest Overestimation\n")
        f.write("="*80 + "\n\n")
        
        for idx, row in top_overestimated.iterrows():
            f.write(f"{row['date'].strftime('%Y-%m-%d')} ({row['day_of_week']})\n")
            f.write(f"  Holiday: {row['holiday_name']}\n")
            f.write(f"  Actual: {row['actual']:,.0f} departures\n")
            f.write(f"  Predicted: {row['predicted']:,.0f} departures\n")
            f.write(f"  Error: +{row['error']:,.0f} ({row['error_pct']:+.1f}%)\n\n")
    
    # Load hourly data to check capacity
    hourly_file = project_root / "data" / "processed" / "hourly" / "hourly_departures.parquet"
    df_hourly = pd.read_parquet(hourly_file)
    df_hourly['date'] = pd.to_datetime(df_hourly['date'])
    # hour is already an integer (0-23), no need to convert
    
    # Calculate average hourly pattern for 2024 (for comparison)
    df_hourly_2024 = df_hourly[df_hourly['date'].dt.year == 2024].copy()
    
    # Get weekday/weekend averages
    df_hourly_2024['is_weekend'] = df_hourly_2024['date'].dt.dayofweek >= 5
    
    weekday_avg = df_hourly_2024[~df_hourly_2024['is_weekend']].groupby('hour')['departure_count'].sum().reset_index()
    weekday_avg['departure_count'] = weekday_avg['departure_count'] / df_hourly_2024[~df_hourly_2024['is_weekend']]['date'].nunique()
    
    weekend_avg = df_hourly_2024[df_hourly_2024['is_weekend']].groupby('hour')['departure_count'].sum().reset_index()
    weekend_avg['departure_count'] = weekend_avg['departure_count'] / df_hourly_2024[df_hourly_2024['is_weekend']]['date'].nunique()
    
    # Create visualization - 6 subplots (3x2)
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Hourly Departure Pattern on Days with Highest Overestimation', 
                 fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(top_overestimated.iterrows()):
        if i >= 6:
            break
            
        target_date = row['date']
        
        # Get hourly data for this date (actual departures)
        day_data = df_hourly[df_hourly['date'] == target_date].copy()
        
        if day_data.empty:
            print(f"\nWarning: No hourly data found for {target_date}")
            axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=20)
            axes[i].set_title(f"{target_date.strftime('%Y-%m-%d')}", fontsize=12)
            continue
        
        # Aggregate by hour (sum across all stations)
        hourly_agg = day_data.groupby('hour')['departure_count'].sum().reset_index()
        hourly_agg = hourly_agg.sort_values('hour')
        
        # Determine if it's a weekend
        is_target_weekend = target_date.dayofweek >= 5
        comparison_avg = weekend_avg if is_target_weekend else weekday_avg
        
        # Plot bar chart
        ax = axes[i]
        bars = ax.bar(hourly_agg['hour'], hourly_agg['departure_count'], 
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5, 
                     label='Actual (this day)')
        
        # Plot comparison line (average pattern)
        ax.plot(comparison_avg['hour'], comparison_avg['departure_count'], 
               color='green', linestyle='--', linewidth=2.5, marker='o', markersize=4,
               label=f'2024 {"Weekend" if is_target_weekend else "Weekday"} Avg', alpha=0.7)
        
        # Highlight peak hours
        max_hour = hourly_agg.loc[hourly_agg['departure_count'].idxmax(), 'hour']
        bars[int(max_hour)].set_color('orangered')
        bars[int(max_hour)].set_alpha(0.9)
        
        # Check for capacity constraint signs
        capacity_warning = False
        warning_text = []
        
        # Check 1: If actual is significantly below average during peak hours (8-9, 17-18)
        peak_hours = [8, 9, 17, 18]
        for h in peak_hours:
            if h in hourly_agg['hour'].values and h in comparison_avg['hour'].values:
                actual_val = hourly_agg[hourly_agg['hour'] == h]['departure_count'].iloc[0]
                avg_val = comparison_avg[comparison_avg['hour'] == h]['departure_count'].iloc[0]
                if actual_val < avg_val * 0.7:  # 30% below average
                    capacity_warning = True
                    warning_text.append(f'Hour {h}: {actual_val/avg_val*100:.0f}% of avg')
        
        # Check 2: Flat peak (ceiling effect) - if 3+ consecutive hours near the same high value
        if len(hourly_agg) >= 3:
            sorted_vals = hourly_agg.sort_values('departure_count', ascending=False)
            top_3 = sorted_vals.head(3)['departure_count'].values
            if top_3.std() / top_3.mean() < 0.05:  # Very similar values
                capacity_warning = True
                warning_text.append('Flat peak detected')
        
        # Add title with comprehensive info
        title = f"{target_date.strftime('%Y-%m-%d')} ({row['day_of_week']})"
        if row['is_holiday']:
            title += f" - {row['holiday_name']}"
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('Total Departures', fontsize=11)
        ax.set_xticks(range(0, 24))
        ax.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=9)
        
        # Add text annotation
        textstr = f'Daily Total: {row["actual"]:,.0f}\n'
        textstr += f'Predicted: {row["predicted"]:,.0f}\n'
        textstr += f'Overestimated by: {row["error"]:,.0f} (+{row["error_pct"]:.1f}%)'
        
        if capacity_warning:
            textstr += '\n\nPossible Capacity Issue:\n'
            textstr += '\n'.join(warning_text)
        
        box_color = 'lightsalmon' if capacity_warning else 'wheat'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))
        
        # Add peak hour marker
        peak_val = hourly_agg['departure_count'].max()
        ax.text(max_hour, peak_val, f'Peak\n{int(peak_val):,}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "overestimation_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nVisualization saved to: {output_file}")
    
    # Additional analysis: check if low actual departures indicate capacity issues
    print(f"\n{'='*80}")
    print("Capacity Analysis")
    print(f"{'='*80}")
    
    # Load daily data with weather to see if weather was good
    daily_weather_file = project_root / "data" / "processed" / "daily" / "daily_with_xgb_features.parquet"
    df_daily = pd.read_parquet(daily_weather_file)
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    
    # Append capacity analysis to file
    with open(summary_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("Capacity Analysis\n")
        f.write("="*80 + "\n\n")
    
    for idx, row in top_overestimated.iterrows():
        target_date = row['date']
        
        # Get weather for this date
        day_weather = df_daily[df_daily['date'] == target_date]
        
        if day_weather.empty:
            continue
        
        avg_temp = day_weather['temperature_2m_mean'].mean()
        total_precip = day_weather['precipitation_sum'].mean()
        
        print(f"\n{target_date.strftime('%Y-%m-%d')}:")
        print(f"  Avg Temperature: {avg_temp:.1f}°C")
        print(f"  Precipitation: {total_precip:.1f}mm")
        
        # Compare with average 2024 day
        avg_2024_departures = daily_totals['actual'].mean()
        print(f"  Actual departures: {row['actual']:,.0f} (2024 avg: {avg_2024_departures:,.0f})")
        
        # Write to file
        with open(summary_file, 'a') as f:
            f.write(f"{target_date.strftime('%Y-%m-%d')}:\n")
            f.write(f"  Avg Temperature: {avg_temp:.1f}°C\n")
            f.write(f"  Precipitation: {total_precip:.1f}mm\n")
            f.write(f"  Actual departures: {row['actual']:,.0f} (2024 avg: {avg_2024_departures:,.0f})\n")
        
        if row['actual'] < avg_2024_departures * 0.8:
            print(f"  → Unusually low activity (below 80% of average)")
            print(f"  → Possible capacity constraint or external factors")
            with open(summary_file, 'a') as f:
                f.write(f"  → Unusually low activity (below 80% of average)\n")
                f.write(f"  → Possible capacity constraint or external factors\n")
        
        with open(summary_file, 'a') as f:
            f.write("\n")
    
    print(f"\n{'='*80}")
    print(f"\nSummary saved to: {summary_file}")


def main():
    analyze_overestimation()


if __name__ == "__main__":
    main()

