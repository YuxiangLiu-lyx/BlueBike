"""
Visualize feature importance for XGBoost No Location model
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    return current.parent.parent.parent


def create_feature_importance_plot():
    project_root = get_project_root()
    
    # Load feature importance
    importance_file = project_root / "results" / "xgboost_no_popularity" / "feature_importance_no_popularity.csv"
    
    if not importance_file.exists():
        raise FileNotFoundError(
            f"Feature importance file not found: {importance_file}\n"
            "Please run: python src/models/xgboost_no_popularity.py"
        )
    
    df = pd.read_csv(importance_file)
    
    # Categorize features
    df['category'] = 'Other'
    df.loc[df['feature'].str.contains('temperature|precipitation|rain|snow|wind'), 'category'] = 'Weather'
    df.loc[df['feature'].isin(['month', 'day', 'day_of_week', 'season', 'is_weekend', 'is_holiday']), 'category'] = 'Temporal'
    
    # Get top 15 features
    top_15 = df.head(15)
    
    # Color mapping
    colors = {
        'Weather': '#3498db',   # Blue
        'Temporal': '#e74c3c',  # Red
        'Other': '#95a5a6'      # Gray
    }
    
    bar_colors = [colors[cat] for cat in top_15['category']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot feature importance
    ax.barh(range(len(top_15)), top_15['importance'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_15)))
    ax.set_yticklabels(top_15['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Feature Importance\n(XGBoost - No Location Features)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat, alpha=0.8) 
                      for cat in ['Weather', 'Temporal', 'Other']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "results" / "xgboost_no_popularity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "feature_importance_no_popularity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'='*60}")
    print("Feature Importance Visualization Created")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    
    print(f"\nTop 15 Feature Importance:")
    for idx, row in top_15.iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    print(f"{'='*60}")


def main():
    create_feature_importance_plot()


if __name__ == "__main__":
    main()

