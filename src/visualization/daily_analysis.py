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


class DailyBehaviorAnalyzer:
    
    def __init__(self, daily_data_path: str = None):
        project_root = get_project_root()
        
        if daily_data_path is None:
            daily_data_path = project_root / "data" / "processed" / "daily" / "daily_departures.parquet"
        
        self.daily_data_path = Path(daily_data_path)
        self.output_dir = project_root / "results" / "figures" / "daily"
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
    
    def plot_daily_patterns(self):
        """Generate daily behavior pattern analysis with 2 subplots"""
        print("\nGenerating daily behavior patterns...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Daily Departure Patterns', fontsize=16, fontweight='bold', y=1.02)
        
        # Subplot 1: Average by day of week
        ax1 = axes[0]
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_avg = self.df.groupby('day_of_week')['departure_count'].mean()
        colors_dow = [self.colors[2] if i < 5 else self.colors[3] for i in range(7)]
        bars = ax1.bar(day_names, dow_avg.values, color=colors_dow, alpha=0.8)
        ax1.set_xlabel('Day of Week', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Avg Departures per Station', fontweight='bold', fontsize=12)
        ax1.set_title('Average Departures by Day of Week', fontweight='bold', pad=15, fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 2: Weekend vs Weekday by month
        ax2 = axes[1]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        weekday_monthly = self.df[self.df['is_weekend'] == 0].groupby('month')['departure_count'].mean()
        weekend_monthly = self.df[self.df['is_weekend'] == 1].groupby('month')['departure_count'].mean()
        x = np.arange(len(months))
        width = 0.35
        ax2.bar(x - width/2, weekday_monthly.values, width, label='Weekday', 
               color=self.colors[2], alpha=0.8)
        ax2.bar(x + width/2, weekend_monthly.values, width, label='Weekend', 
               color=self.colors[3], alpha=0.8)
        ax2.set_xlabel('Month', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Avg Departures per Station', fontweight='bold', fontsize=12)
        ax2.set_title('Weekday vs Weekend by Month', fontweight='bold', pad=15, fontsize=13)
        ax2.set_xticks(x)
        ax2.set_xticklabels(months, rotation=45, ha='right')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / "daily_patterns.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    def generate_summary_text(self):
        """Generate key statistics summary"""
        print("\nGenerating summary statistics...")
        
        weekday_avg = self.df[self.df['is_weekend'] == 0]['departure_count'].mean()
        weekend_avg = self.df[self.df['is_weekend'] == 1]['departure_count'].mean()
        
        dow_avg = self.df.groupby('day_of_week')['departure_count'].mean()
        busiest_dow = dow_avg.idxmax()
        quietest_dow = dow_avg.idxmin()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        summary_lines = [
            "="*60,
            "BlueBikes Daily Behavior Analysis - Key Statistics",
            "="*60,
            "",
            f"Dataset Overview:",
            f"  Total records: {len(self.df):,}",
            f"  Date range: {self.df['date'].min().date()} to {self.df['date'].max().date()}",
            f"  Total departures: {self.df['departure_count'].sum():,}",
            "",
            f"Daily Patterns:",
            f"  Overall average: {self.df['departure_count'].mean():.1f} departures/station/day",
            f"  Weekday average: {weekday_avg:.1f} departures/station/day",
            f"  Weekend average: {weekend_avg:.1f} departures/station/day",
            f"  Weekday/Weekend ratio: {weekday_avg/weekend_avg:.2f}x",
            "",
            f"Day of Week Analysis:",
            f"  Busiest day: {day_names[busiest_dow]} ({dow_avg[busiest_dow]:.1f} departures/station)",
            f"  Quietest day: {day_names[quietest_dow]} ({dow_avg[quietest_dow]:.1f} departures/station)",
            "",
            f"Monthly Patterns:",
            f"  Busiest month (weekday): {self.df[self.df['is_weekend']==0].groupby('month')['departure_count'].mean().idxmax()}",
            f"  Busiest month (weekend): {self.df[self.df['is_weekend']==1].groupby('month')['departure_count'].mean().idxmax()}",
            "",
            "="*60,
        ]
        
        output_path = self.output_dir / "daily_analysis_summary.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        print(f"✓ Saved: {output_path}")
        print('\n'.join(summary_lines))
    
    def run_all(self):
        """Generate all visualizations"""
        print("="*60)
        print("Daily Behavior Pattern Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Generate plots
        self.plot_daily_patterns()
        self.generate_summary_text()
        
        print("\n" + "="*60)
        print("Daily analysis complete!")
        print("="*60)
        print("\nOutput files:")
        print(f"  - {self.output_dir}/daily_patterns.png")
        print(f"  - {self.output_dir}/daily_analysis_summary.txt")


def main():
    analyzer = DailyBehaviorAnalyzer()
    analyzer.run_all()


if __name__ == "__main__":
    main()
