import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class StationAnalyzer:
    
    def __init__(self, daily_data_path: str = None):
        project_root = get_project_root()
        
        if daily_data_path is None:
            daily_data_path = project_root / "data" / "processed" / "daily" / "daily_departures.parquet"
        
        self.daily_data_path = Path(daily_data_path)
        self.output_dir = project_root / "results" / "figures" / "station"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    def load_data(self):
        """Load daily departure data"""
        print(f"Loading data from {self.daily_data_path}...")
        self.df = pd.read_parquet(self.daily_data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"Loaded {len(self.df):,} records")
        return self.df
    
    def plot_station_analysis(self):
        """Generate station analysis with 3 subplots"""
        print("\nGenerating station analysis...")
        
        # Calculate station total departures by station name
        station_totals = self.df.groupby('station_name')['departure_count'].sum().sort_values(ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Station Activity Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Subplot 1: TOP 20 stations
        top20 = station_totals.head(20)
        # Truncate long names
        names = [name[:35] + '...' if len(name) > 35 else name for name in top20.index]
        y_pos = np.arange(len(names))
        axes[0].barh(y_pos, top20.values / 1000, color=self.colors[1], alpha=0.8)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(names, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Total Departures (Thousands)', fontweight='bold', fontsize=12)
        axes[0].set_title('TOP 20 Busiest Stations', fontweight='bold', pad=15, fontsize=13)
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Subplot 2: Station activity distribution
        # Create bins for station total departures (dynamically adjust based on max value)
        max_val = station_totals.max()
        
        if max_val > 1000000:
            bins = [0, 10000, 50000, 100000, 200000, 500000, 1000000, max_val + 1]
            labels = ['0-10K', '10K-50K', '50K-100K', '100K-200K', '200K-500K', '500K-1M', '1M+']
        elif max_val > 500000:
            bins = [0, 10000, 50000, 100000, 200000, 500000, max_val + 1]
            labels = ['0-10K', '10K-50K', '50K-100K', '100K-200K', '200K-500K', '500K+']
        elif max_val > 200000:
            bins = [0, 10000, 50000, 100000, 200000, max_val + 1]
            labels = ['0-10K', '10K-50K', '50K-100K', '100K-200K', '200K+']
        else:
            bins = [0, 10000, 50000, 100000, max_val + 1]
            labels = ['0-10K', '10K-50K', '50K-100K', '100K+']
        
        # Create a temporary dataframe for binning
        station_ranges = pd.cut(station_totals, 
                               bins=bins, 
                               labels=labels,
                               include_lowest=True)
        range_counts = station_ranges.value_counts().sort_index()
        
        bars = axes[1].bar(range(len(range_counts)), range_counts.values, 
                          color=self.colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(len(range_counts)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right')
        axes[1].set_xlabel('Total Departure Range', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Number of Stations', fontweight='bold', fontsize=12)
        axes[1].set_title('Station Distribution by Activity Level', fontweight='bold', pad=15, fontsize=13)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 3: Station count over time
        station_count_by_year = self.df.groupby('year')['station_name'].nunique()
        axes[2].plot(station_count_by_year.index, station_count_by_year.values, 
                    marker='o', linewidth=2.5, markersize=8, color=self.colors[3])
        axes[2].set_xlabel('Year', fontweight='bold', fontsize=12)
        axes[2].set_ylabel('Number of Active Stations', fontweight='bold', fontsize=12)
        axes[2].set_title('Station Network Growth Over Time', fontweight='bold', pad=15, fontsize=13)
        axes[2].grid(True, alpha=0.3)
        for x, y in zip(station_count_by_year.index, station_count_by_year.values):
            axes[2].text(x, y + 5, str(y), ha='center', fontsize=9, fontweight='bold')
        
        # Hide the 4th subplot
        axes[3].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / "station_activity.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
        # Store for summary
        self.station_totals = station_totals
    
    def generate_summary_text(self):
        """Generate key statistics summary"""
        print("\nGenerating summary statistics...")
        
        top20_departures = self.station_totals.head(20).sum()
        total_departures = self.station_totals.sum()
        top20_pct = top20_departures / total_departures * 100
        
        top10_pct = self.station_totals.head(10).sum() / total_departures * 100
        
        # Calculate 80% threshold
        cumsum = self.station_totals.cumsum()
        threshold_80 = (cumsum >= total_departures * 0.8).idxmax()
        stations_for_80 = self.station_totals.index.get_loc(threshold_80) + 1
        pct_stations_for_80 = stations_for_80 / len(self.station_totals) * 100
        
        # Station growth statistics
        station_count_by_year = self.df.groupby('year')['station_name'].nunique()
        first_year_count = station_count_by_year.iloc[0]
        last_year_count = station_count_by_year.iloc[-1]
        growth_pct = (last_year_count - first_year_count) / first_year_count * 100
        
        summary_lines = [
            "="*60,
            "BlueBikes Station Activity Analysis - Key Statistics",
            "="*60,
            "",
            f"Station Overview:",
            f"  Total unique stations: {len(self.station_totals):,}",
            f"  Total departures: {total_departures:,}",
            "",
            f"Top Stations:",
            f"  Busiest station: {self.station_totals.index[0]}",
            f"    Total departures: {self.station_totals.iloc[0]:,}",
            f"  Top 10 stations contribute: {top10_pct:.1f}% of all trips",
            f"  Top 20 stations contribute: {top20_pct:.1f}% of all trips",
            "",
            f"Activity Distribution:",
            f"  Average departures per station: {self.station_totals.mean():,.0f}",
            f"  Median departures per station: {self.station_totals.median():,.0f}",
            f"  Most active station: {self.station_totals.max():,} departures",
            f"  Least active station: {self.station_totals.min():,} departures",
            "",
            f"Concentration Analysis:",
            f"  {pct_stations_for_80:.1f}% of stations generate 80% of trips",
            f"  This represents {stations_for_80} out of {len(self.station_totals)} stations",
            "",
            f"Station Network Growth:",
            f"  First year ({station_count_by_year.index[0]}): {first_year_count} stations",
            f"  Latest year ({station_count_by_year.index[-1]}): {last_year_count} stations",
            f"  Total growth: {growth_pct:+.1f}% ({last_year_count - first_year_count:+d} stations)",
            "",
            "="*60,
        ]
        
        output_path = self.output_dir / "station_analysis_summary.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"✓ Saved: {output_path}")
        print('\n'.join(summary_lines))
    
    def run_all(self):
        """Generate all visualizations"""
        print("="*60)
        print("Station Activity Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Generate plots
        self.plot_station_analysis()
        self.generate_summary_text()
        
        print("\n" + "="*60)
        print("Station analysis complete!")
        print("="*60)
        print("\nOutput files:")
        print(f"  - {self.output_dir}/station_activity.png")
        print(f"  - {self.output_dir}/station_analysis_summary.txt")


def main():
    analyzer = StationAnalyzer()
    analyzer.run_all()


if __name__ == "__main__":
    main()

