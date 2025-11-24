"""
Compare three models: Baseline, XGBoost No Popularity, XGBoost POI Only
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    """Get project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def load_metrics():
    """Load metrics from all three models"""
    project_root = get_project_root()
    
    # Load baseline metrics
    baseline_file = project_root / "results" / "models" / "baseline_metrics.csv"
    baseline = pd.read_csv(baseline_file)
    
    # Load XGBoost no popularity metrics
    no_pop_file = project_root / "results" / "xgboost_no_popularity" / "xgboost_metrics_no_popularity.csv"
    no_pop = pd.read_csv(no_pop_file)
    
    # Load XGBoost POI only metrics
    poi_only_file = project_root / "results" / "xgboost_poi_only" / "xgboost_metrics_poi_only.csv"
    poi_only = pd.read_csv(poi_only_file)
    
    return baseline, no_pop, poi_only


def create_comparison_plot():
    """Create comprehensive comparison visualization"""
    baseline, no_pop, poi_only = load_metrics()
    
    # Define metrics to compare
    metrics = [
        ('MAE', 'Mean Absolute Error'),
        ('RMSE', 'Root Mean Squared Error'),
        ('R2', 'R² Score'),
        ('MAPE', 'Mean Absolute Percentage Error (%)'),
        ('Total_Error_Pct', 'Total Error (%)'),
        ('Overall_Accuracy', 'Overall Accuracy (%)'),
        ('MPE', 'Mean Percentage Error (%)')
    ]
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Model names and colors
    model_names = ['Baseline', 'XGB\nNo Popularity', 'XGB\nPOI Only']
    colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, Orange, Green
    
    # Plot each metric
    for idx, (metric_key, metric_name) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Get values
        values = []
        for df in [baseline, no_pop, poi_only]:
            if metric_key in df.columns:
                values.append(df[metric_key].values[0])
            else:
                values.append(np.nan)
        
        # Create bar chart
        bars = ax.bar(model_names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                height = bar.get_height()
                # Handle negative values
                y_pos = height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02 if height >= 0 else height - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{val:.2f}',
                       ha='center', va=va, fontsize=9, fontweight='bold')
        
        # Calculate improvements
        baseline_val = values[0]
        no_pop_val = values[1]
        poi_only_val = values[2]
        
        # For R² and Overall Accuracy, higher is better
        # For others, lower is better
        if metric_key in ['R2', 'Overall_Accuracy']:
            improve_no_pop = ((no_pop_val - baseline_val) / abs(baseline_val)) * 100 if not np.isnan(no_pop_val) else 0
            improve_poi = ((poi_only_val - baseline_val) / abs(baseline_val)) * 100 if not np.isnan(poi_only_val) else 0
            improve_poi_vs_no_pop = ((poi_only_val - no_pop_val) / abs(no_pop_val)) * 100 if not np.isnan(poi_only_val) else 0
        else:
            improve_no_pop = ((baseline_val - no_pop_val) / abs(baseline_val)) * 100 if not np.isnan(no_pop_val) else 0
            improve_poi = ((baseline_val - poi_only_val) / abs(baseline_val)) * 100 if not np.isnan(poi_only_val) else 0
            improve_poi_vs_no_pop = ((no_pop_val - poi_only_val) / abs(no_pop_val)) * 100 if not np.isnan(poi_only_val) else 0
        
        # Add improvement annotations
        if not np.isnan(no_pop_val):
            color_no_pop = 'green' if improve_no_pop > 0 else 'red'
            sign_no_pop = '+' if improve_no_pop > 0 else ''
            ax.text(1, ax.get_ylim()[1] * 0.95, f'{sign_no_pop}{improve_no_pop:.1f}%',
                   ha='center', fontsize=8, color=color_no_pop, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_no_pop, alpha=0.8))
        
        if not np.isnan(poi_only_val):
            color_poi = 'green' if improve_poi > 0 else 'red'
            sign_poi = '+' if improve_poi > 0 else ''
            ax.text(2, ax.get_ylim()[1] * 0.95, f'{sign_poi}{improve_poi:.1f}%',
                   ha='center', fontsize=8, color=color_poi, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color_poi, alpha=0.8))
            
            # Add POI vs No Pop comparison
            color_poi_vs = 'green' if improve_poi_vs_no_pop > 0 else 'red'
            sign_poi_vs = '+' if improve_poi_vs_no_pop > 0 else ''
            ax.text(2, ax.get_ylim()[1] * 0.85, f'({sign_poi_vs}{improve_poi_vs_no_pop:.1f}% vs No Pop)',
                   ha='center', fontsize=7, color=color_poi_vs, style='italic')
        
        ax.set_title(metric_name, fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        
        # Adjust y-axis to accommodate negative values if present
        if any(v < 0 for v in values if not np.isnan(v)):
            y_min, y_max = ax.get_ylim()
            margin = (y_max - y_min) * 0.15
            ax.set_ylim(y_min - margin, y_max + margin)
    
    # Add summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['Metric', 'Baseline', 'XGB No Pop', 'Change', 'XGB POI Only', 'Change vs Baseline', 'Change vs No Pop'])
    
    for metric_key, metric_name in metrics:
        baseline_val = baseline[metric_key].values[0] if metric_key in baseline.columns else np.nan
        no_pop_val = no_pop[metric_key].values[0] if metric_key in no_pop.columns else np.nan
        poi_only_val = poi_only[metric_key].values[0] if metric_key in poi_only.columns else np.nan
        
        # Calculate changes
        if metric_key in ['R2', 'Overall_Accuracy']:
            change_no_pop = ((no_pop_val - baseline_val) / abs(baseline_val)) * 100 if not np.isnan(no_pop_val) else np.nan
            change_poi = ((poi_only_val - baseline_val) / abs(baseline_val)) * 100 if not np.isnan(poi_only_val) else np.nan
            change_poi_vs_no_pop = ((poi_only_val - no_pop_val) / abs(no_pop_val)) * 100 if not np.isnan(poi_only_val) else np.nan
        else:
            change_no_pop = ((baseline_val - no_pop_val) / abs(baseline_val)) * 100 if not np.isnan(no_pop_val) else np.nan
            change_poi = ((baseline_val - poi_only_val) / abs(baseline_val)) * 100 if not np.isnan(poi_only_val) else np.nan
            change_poi_vs_no_pop = ((no_pop_val - poi_only_val) / abs(no_pop_val)) * 100 if not np.isnan(poi_only_val) else np.nan
        
        row = [
            metric_key,
            f'{baseline_val:.2f}' if not np.isnan(baseline_val) else 'N/A',
            f'{no_pop_val:.2f}' if not np.isnan(no_pop_val) else 'N/A',
            f'{change_no_pop:+.1f}%' if not np.isnan(change_no_pop) else 'N/A',
            f'{poi_only_val:.2f}' if not np.isnan(poi_only_val) else 'N/A',
            f'{change_poi:+.1f}%' if not np.isnan(change_poi) else 'N/A',
            f'{change_poi_vs_no_pop:+.1f}%' if not np.isnan(change_poi_vs_no_pop) else 'N/A'
        ]
        table_data.append(row)
    
    # Create table
    table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.18, 0.12, 0.12, 0.12, 0.12, 0.17, 0.17])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data rows
    for i in range(1, len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')
            
            # Color code change columns
            if j in [3, 5, 6] and i > 0:
                text = cell.get_text().get_text()
                if text != 'N/A' and '+' in text:
                    cell.set_text_props(color='green', weight='bold')
                elif text != 'N/A' and '-' in text:
                    cell.set_text_props(color='red', weight='bold')
    
    # Add title
    fig.suptitle('Model Comparison: Baseline vs XGBoost (No Popularity) vs XGBoost (POI Only)',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add subtitle
    fig.text(0.5, 0.95, 
            'Comparing the impact of removing nearby_avg_popularity and adding POI features',
            ha='center', fontsize=11, style='italic', color='gray')
    
    # Save figure
    project_root = get_project_root()
    output_dir = project_root / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "three_model_comparison.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Three Model Comparison Created")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    
    print(f"\nModel Performance Summary:")
    print(f"  Baseline:")
    print(f"    R²: {baseline['R2'].values[0]:.4f}")
    print(f"    MAE: {baseline['MAE'].values[0]:.2f}")
    print(f"    Total Error: {baseline['Total_Error_Pct'].values[0]:.2f}%")
    
    print(f"\n  XGBoost No Popularity:")
    print(f"    R²: {no_pop['R2'].values[0]:.4f} ({(no_pop['R2'].values[0] - baseline['R2'].values[0]) / baseline['R2'].values[0] * 100:+.1f}%)")
    print(f"    MAE: {no_pop['MAE'].values[0]:.2f} ({(baseline['MAE'].values[0] - no_pop['MAE'].values[0]) / baseline['MAE'].values[0] * 100:+.1f}%)")
    print(f"    Total Error: {no_pop['Total_Error_Pct'].values[0]:.2f}% ({(baseline['Total_Error_Pct'].values[0] - no_pop['Total_Error_Pct'].values[0]) / baseline['Total_Error_Pct'].values[0] * 100:+.1f}%)")
    
    print(f"\n  XGBoost POI Only:")
    print(f"    R²: {poi_only['R2'].values[0]:.4f} ({(poi_only['R2'].values[0] - baseline['R2'].values[0]) / baseline['R2'].values[0] * 100:+.1f}% vs baseline, {(poi_only['R2'].values[0] - no_pop['R2'].values[0]) / no_pop['R2'].values[0] * 100:+.1f}% vs no pop)")
    print(f"    MAE: {poi_only['MAE'].values[0]:.2f} ({(baseline['MAE'].values[0] - poi_only['MAE'].values[0]) / baseline['MAE'].values[0] * 100:+.1f}% vs baseline, {(no_pop['MAE'].values[0] - poi_only['MAE'].values[0]) / no_pop['MAE'].values[0] * 100:+.1f}% vs no pop)")
    print(f"    Total Error: {poi_only['Total_Error_Pct'].values[0]:.2f}% ({(baseline['Total_Error_Pct'].values[0] - poi_only['Total_Error_Pct'].values[0]) / baseline['Total_Error_Pct'].values[0] * 100:+.1f}% vs baseline, {(no_pop['Total_Error_Pct'].values[0] - poi_only['Total_Error_Pct'].values[0]) / no_pop['Total_Error_Pct'].values[0] * 100:+.1f}% vs no pop)")
    
    print(f"{'='*60}")


def main():
    create_comparison_plot()


if __name__ == "__main__":
    main()

