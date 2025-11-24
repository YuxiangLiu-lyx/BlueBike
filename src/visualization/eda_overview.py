import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class DataOverview:
    
    def __init__(self, data_path: str = None):
        project_root = get_project_root()
        
        if data_path is None:
            data_path = project_root / "data" / "processed" / "bluebike_cleaned" / "trips_cleaned.parquet"
        
        self.data_path = Path(data_path)
        self.output_dir = project_root / "results" / "figures" / "overview"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats_dir = project_root / "results" / "statistics"
        self.stats_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load cleaned trip data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        print(f"Loaded {len(self.df):,} trips")
        print(f"Date range: {self.df['start_time'].min()} to {self.df['start_time'].max()}")
        return self.df
    
    def yearly_statistics(self):
        """Calculate yearly departure statistics"""
        print("\nCalculating yearly statistics...")
        
        # Group by year and count
        yearly = self.df.groupby('year').agg({
            'start_time': 'count',  # Total departures
            'start_station_name': 'nunique'  # Unique stations
        }).reset_index()
        
        yearly.columns = ['Year', 'Total_Departures', 'Active_Stations']
        
        # Calculate growth rate
        yearly['Growth_Rate'] = yearly['Total_Departures'].pct_change() * 100
        
        # Calculate average departures per station
        yearly['Avg_Departures_Per_Station'] = (
            yearly['Total_Departures'] / yearly['Active_Stations']
        ).round(0).astype(int)
        
        # Format growth rate
        yearly['Growth_Rate'] = yearly['Growth_Rate'].apply(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
        )
        
        return yearly
    
    def save_statistics(self, yearly_stats):
        """Save statistics to CSV"""
        output_file = self.stats_dir / "yearly_statistics.csv"
        yearly_stats.to_csv(output_file, index=False)
        print(f"\nStatistics saved to: {output_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("Yearly Departure Statistics")
        print("="*80)
        print(yearly_stats.to_string(index=False))
        print("="*80)
    
    def plot_yearly_trend(self, yearly_stats):
        """Create visualization of yearly trend"""
        print("\nGenerating visualization...")
        
        # Convert Growth_Rate back to numeric for internal use
        yearly_numeric = yearly_stats.copy()
        yearly_numeric['Growth_Rate_Numeric'] = yearly_stats['Growth_Rate'].apply(
            lambda x: float(x.strip('%+')) if x != "-" else 0
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot total departures
        ax.plot(yearly_numeric['Year'], 
                yearly_numeric['Total_Departures'] / 1_000_000,
                marker='o', 
                linewidth=2, 
                markersize=8,
                color='#2E86AB',
                label='Total Departures')
        
        # Add value labels on points
        for idx, row in yearly_numeric.iterrows():
            ax.text(row['Year'], 
                   row['Total_Departures'] / 1_000_000 + 0.1,
                   f"{row['Total_Departures']/1_000_000:.2f}M",
                   ha='center',
                   fontsize=9)
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Departures (Millions)', fontsize=12, fontweight='bold')
        ax.set_title('BlueBikes Annual Departure Trend (2015-2025)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / "yearly_departure_trend.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_file}")
        
        plt.close()
    
    def run_analysis(self):
        """Run complete yearly analysis"""
        print("="*80)
        print("BlueBikes Yearly Overview Analysis")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Calculate statistics
        yearly_stats = self.yearly_statistics()
        
        # Save results
        self.save_statistics(yearly_stats)
        
        # Create visualization
        self.plot_yearly_trend(yearly_stats)
        
        print("\n" + "="*80)
        print("Analysis complete")
        print("="*80)


def main():
    analyzer = DataOverview()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

