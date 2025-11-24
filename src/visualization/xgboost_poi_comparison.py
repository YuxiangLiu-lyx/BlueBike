"""
Compare XGBoost models with and without POI features
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


def create_xgboost_poi_comparison_plot():
    """Create comprehensive comparison between XGBoost models with/without POI"""
    project_root = get_project_root()
    
    # Load metrics
    xgb_no_poi_file = project_root / "results" / "xgboost" / "xgboost_metrics.csv"
    xgb_with_poi_file = project_root / "results" / "xgboost_with_poi" / "xgboost_metrics_with_poi.csv"
    
    if not xgb_no_poi_file.exists():
        raise FileNotFoundError(f"XGBoost (without POI) metrics not found: {xgb_no_poi_file}")
    if not xgb_with_poi_file.exists():
        raise FileNotFoundError(f"XGBoost (with POI) metrics not found: {xgb_with_poi_file}")
    
    xgb_no_poi = pd.read_csv(xgb_no_poi_file)
    xgb_with_poi = pd.read_csv(xgb_with_poi_file)
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    color_no_poi = '#3498db'    # Blue
    color_with_poi = '#2ecc71'  # Green
    
    # Extract metrics (data is in first row)
    actual_total = float(xgb_no_poi['Total_Actual'].iloc[0])
    
    metrics = {
        'R² Score': (float(xgb_no_poi['R2'].iloc[0]), float(xgb_with_poi['R2'].iloc[0])),
        'MAE': (float(xgb_no_poi['MAE'].iloc[0]), float(xgb_with_poi['MAE'].iloc[0])),
        'RMSE': (float(xgb_no_poi['RMSE'].iloc[0]), float(xgb_with_poi['RMSE'].iloc[0])),
        'Total Error (%)': (float(xgb_no_poi['Total_Error_Pct'].iloc[0]), float(xgb_with_poi['Total_Error_Pct'].iloc[0])),
        'MPE (%)': (float(xgb_no_poi['MPE'].iloc[0]), float(xgb_with_poi['MPE'].iloc[0])),
        'Overall Accuracy (%)': (float(xgb_no_poi['Overall_Accuracy'].iloc[0]), float(xgb_with_poi['Overall_Accuracy'].iloc[0])),
        'Total Predicted': (float(xgb_no_poi['Total_Predicted'].iloc[0]), float(xgb_with_poi['Total_Predicted'].iloc[0])),
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('XGBoost: Without POI vs With POI Features', fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes
    axes = axes.flatten()
    
    # Plot each metric
    metric_names = list(metrics.keys())
    x = np.arange(2)
    width = 0.6
    
    for idx, metric_name in enumerate(metric_names[:7]):
        ax = axes[idx]
        values = metrics[metric_name]
        
        # Create bars
        bars = ax.bar(x, values, width, color=[color_no_poi, color_with_poi], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Set labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Without POI', 'With POI'], fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height
            else:
                va = 'top'
                y_pos = height
            
            # Format value
            if 'Total Predicted' in metric_name:
                label = f'{val:,.0f}'
            elif '%' in metric_name:
                label = f'{val:+.2f}%'
            elif metric_name in ['MAE', 'RMSE']:
                label = f'{val:.2f}'
            else:
                label = f'{val:.4f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   label, ha='center', va=va, fontsize=10, fontweight='bold')
        
        # Add zero line for metrics that can be negative
        if any(v < 0 for v in values):
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Calculate improvement based on metric type
        if 'MPE' in metric_name or 'Total Error' in metric_name:
            improvement = ((abs(values[0]) - abs(values[1])) / abs(values[0])) * 100
        elif 'R²' in metric_name or 'Overall Accuracy' in metric_name:
            improvement = ((values[1] - values[0]) / abs(values[0])) * 100
        else:
            improvement = ((values[0] - values[1]) / abs(values[0])) * 100
        
        color_bg = 'lightgreen' if improvement > 0 else 'lightcoral'
        ax.text(0.5, ax.get_ylim()[1] * 0.85, 
               f'Change: {improvement:+.1f}%',
               ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor=color_bg, alpha=0.5))
        
        ax.grid(axis='y', alpha=0.3)
    
    # Summary table
    axes[7].axis('off')
    axes[8].axis('off')
    
    ax_summary = fig.add_subplot(3, 3, (8, 9))
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    table_data = [
        ['Metric', 'Without POI', 'With POI', 'Change'],
    ]
    
    for metric_name, (no_poi_val, with_poi_val) in metrics.items():
        if 'MPE' in metric_name or 'Total Error' in metric_name:
            change = ((abs(no_poi_val) - abs(with_poi_val)) / abs(no_poi_val)) * 100
        elif 'R²' in metric_name or 'Overall Accuracy' in metric_name:
            change = ((with_poi_val - no_poi_val) / abs(no_poi_val)) * 100
        else:
            change = ((no_poi_val - with_poi_val) / abs(no_poi_val)) * 100
        
        if 'Total Predicted' in metric_name:
            no_poi_str = f'{no_poi_val:,.0f}'
            with_poi_str = f'{with_poi_val:,.0f}'
        elif '%' in metric_name:
            no_poi_str = f'{no_poi_val:+.2f}%'
            with_poi_str = f'{with_poi_val:+.2f}%'
        elif metric_name in ['MAE', 'RMSE']:
            no_poi_str = f'{no_poi_val:.2f}'
            with_poi_str = f'{with_poi_val:.2f}'
        else:
            no_poi_str = f'{no_poi_val:.4f}'
            with_poi_str = f'{with_poi_val:.4f}'
        
        table_data.append([metric_name, no_poi_str, with_poi_str, f'{change:+.1f}%'])
    
    table_data.append(['Actual Total', f'{actual_total:,}', f'{actual_total:,}', '—'])
    
    table = ax_summary.table(cellText=table_data, cellLoc='center', loc='center',
                           colWidths=[0.35, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            if j == 3:  # Change column
                try:
                    change_val = float(table_data[i][3].rstrip('%'))
                    if change_val > 0:
                        table[(i, j)].set_facecolor('#d5f4e6')
                    elif change_val < 0:
                        table[(i, j)].set_facecolor('#fadbd8')
                except:
                    pass
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "xgboost_poi_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'='*60}")
    print("XGBoost POI Comparison Visualization Created")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    print(f"\nMetrics Comparison:")
    r2_no_poi = float(xgb_no_poi['R2'].iloc[0])
    r2_with_poi = float(xgb_with_poi['R2'].iloc[0])
    mae_no_poi = float(xgb_no_poi['MAE'].iloc[0])
    mae_with_poi = float(xgb_with_poi['MAE'].iloc[0])
    rmse_no_poi = float(xgb_no_poi['RMSE'].iloc[0])
    rmse_with_poi = float(xgb_with_poi['RMSE'].iloc[0])
    error_no_poi = float(xgb_no_poi['Total_Error_Pct'].iloc[0])
    error_with_poi = float(xgb_with_poi['Total_Error_Pct'].iloc[0])
    
    print(f"  • R²: {r2_no_poi:.4f} → {r2_with_poi:.4f}")
    print(f"  • MAE: {mae_no_poi:.2f} → {mae_with_poi:.2f}")
    print(f"  • RMSE: {rmse_no_poi:.2f} → {rmse_with_poi:.2f}")
    print(f"  • Total Error: {error_no_poi:+.2f}% → {error_with_poi:+.2f}%")
    
    r2_improvement = ((r2_with_poi - r2_no_poi) / abs(r2_no_poi)) * 100
    print(f"\n  POI Features Impact:")
    print(f"    R² improvement: {r2_improvement:+.2f}%")
    print(f"{'='*60}")


def main():
    create_xgboost_poi_comparison_plot()


if __name__ == "__main__":
    main()

