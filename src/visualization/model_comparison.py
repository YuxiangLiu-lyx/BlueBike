"""
Create visualization comparing Baseline and XGBoost models
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


def create_model_comparison_plot():
    """Create comprehensive model comparison visualization"""
    project_root = get_project_root()
    
    # Load metrics
    baseline = pd.read_csv(project_root / "results" / "models" / "baseline_metrics.csv")
    xgb = pd.read_csv(project_root / "results" / "xgboost" / "xgboost_metrics.csv")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    color_baseline = '#3498db'  # Blue
    color_xgb = '#e74c3c'       # Red
    
    # Extract metrics
    actual_total = 4724254
    
    metrics = {
        'R² Score': (baseline['R2'].iloc[0], xgb['R2'].iloc[0]),
        'MAE': (baseline['MAE'].iloc[0], xgb['MAE'].iloc[0]),
        'RMSE': (baseline['RMSE'].iloc[0], xgb['RMSE'].iloc[0]),
        'Total Error (%)': (baseline['Total_Error_Pct'].iloc[0], xgb['Total_Error_Pct'].iloc[0]),
        'MPE (%)': (baseline['MPE'].iloc[0], xgb['MPE'].iloc[0]),
        'Overall Accuracy (%)': (baseline['Overall_Accuracy'].iloc[0], xgb['Overall_Accuracy'].iloc[0]),
        'Total Predicted': (baseline['Total_Predicted'].iloc[0], xgb['Total_Predicted'].iloc[0]),
    }
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Baseline vs XGBoost Model Comparison', fontsize=20, fontweight='bold', y=0.98)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Plot each metric
    metric_names = list(metrics.keys())
    x = np.arange(2)
    width = 0.6
    
    for idx, metric_name in enumerate(metric_names[:7]):
        ax = axes[idx]
        values = metrics[metric_name]
        
        # Create bars side by side
        bars = ax.bar(x, values, width, color=[color_baseline, color_xgb], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Set labels
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline', 'XGBoost'], fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            # Position label above or below based on sign
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
        
        # Add improvement percentage
        improvement = ((values[1] - values[0]) / abs(values[0])) * 100
        ax.text(0.5, ax.get_ylim()[1] * 0.85, 
               f'Change: {improvement:+.1f}%',
               ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='yellow' if improvement > 0 else 'lightcoral', alpha=0.5))
        
        ax.grid(axis='y', alpha=0.3)
    
    # Summary table in the last two subplots
    axes[7].axis('off')
    axes[8].axis('off')
    
    # Combine last two subplots for summary
    ax_summary = fig.add_subplot(3, 3, (8, 9))
    ax_summary.axis('tight')
    ax_summary.axis('off')
    
    table_data = [
        ['Metric', 'Baseline', 'XGBoost', 'Change'],
    ]
    
    for metric_name, (base_val, xgb_val) in metrics.items():
        change = ((xgb_val - base_val) / abs(base_val)) * 100
        
        if 'Total Predicted' in metric_name:
            base_str = f'{base_val:,.0f}'
            xgb_str = f'{xgb_val:,.0f}'
        elif '%' in metric_name:
            base_str = f'{base_val:+.2f}%'
            xgb_str = f'{xgb_val:+.2f}%'
        elif metric_name in ['MAE', 'RMSE']:
            base_str = f'{base_val:.2f}'
            xgb_str = f'{xgb_val:.2f}'
        else:
            base_str = f'{base_val:.4f}'
            xgb_str = f'{xgb_val:.4f}'
        
        table_data.append([metric_name, base_str, xgb_str, f'{change:+.1f}%'])
    
    # Add actual total as reference
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
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Model Comparison Visualization Created")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    print(f"\nMetrics Displayed:")
    print(f"  • R² Score: {baseline['R2'].iloc[0]:.4f} → {xgb['R2'].iloc[0]:.4f}")
    print(f"  • MAE: {baseline['MAE'].iloc[0]:.2f} → {xgb['MAE'].iloc[0]:.2f}")
    print(f"  • RMSE: {baseline['RMSE'].iloc[0]:.2f} → {xgb['RMSE'].iloc[0]:.2f}")
    print(f"  • Total Error: {baseline['Total_Error_Pct'].iloc[0]:+.2f}% → {xgb['Total_Error_Pct'].iloc[0]:+.2f}%")
    print(f"  • MPE: {baseline['MPE'].iloc[0]:+.2f}% → {xgb['MPE'].iloc[0]:+.2f}%")
    print(f"{'='*60}")


def main():
    create_model_comparison_plot()


if __name__ == "__main__":
    main()

