"""
Analyze and visualize POI feature importance in XGBoost model
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


def create_poi_feature_importance_plot():
    """Create feature importance visualization highlighting POI features"""
    project_root = get_project_root()
    
    # Load feature importance
    importance_file = project_root / "results" / "xgboost_with_poi" / "feature_importance_with_poi.csv"
    
    if not importance_file.exists():
        raise FileNotFoundError(
            f"Feature importance file not found: {importance_file}\n"
            "Please run: python src/models/xgboost_model_with_poi.py"
        )
    
    df = pd.read_csv(importance_file)
    
    # Define POI features
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
    
    # Categorize features
    df['category'] = 'Other'
    df.loc[df['feature'].isin(poi_features), 'category'] = 'POI'
    df.loc[df['feature'].str.contains('temperature|precipitation|rain|snow|wind'), 'category'] = 'Weather'
    df.loc[df['feature'].isin(['month', 'day', 'day_of_week', 'season', 'is_weekend', 'is_holiday']), 'category'] = 'Temporal'
    df.loc[df['feature'].isin(['latitude', 'longitude', 'nearby_avg_popularity']), 'category'] = 'Location'
    
    # Get top 20 features
    top_20 = df.head(20)
    
    # Color mapping
    colors = {
        'POI': '#2ecc71',       # Green
        'Weather': '#3498db',   # Blue
        'Temporal': '#e74c3c',  # Red
        'Location': '#f39c12',  # Orange
        'Other': '#95a5a6'      # Gray
    }
    
    bar_colors = [colors[cat] for cat in top_20['category']]
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot: Top 20 feature importance
    ax.barh(range(len(top_20)), top_20['importance'], color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Feature Importance (XGBoost with POI)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[cat], label=cat, alpha=0.8) 
                      for cat in ['POI', 'Weather', 'Temporal', 'Location', 'Other']]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = project_root / "results" / "xgboost_with_poi"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "poi_feature_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n{'='*60}")
    print("POI Feature Analysis Created")
    print(f"{'='*60}")
    print(f"Saved to: {output_file}")
    
    print(f"\nTop 20 Feature Importance:")
    for idx, row in top_20.iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    print(f"{'='*60}")


def main():
    create_poi_feature_importance_plot()


if __name__ == "__main__":
    main()

