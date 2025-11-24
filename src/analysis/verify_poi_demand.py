"""
Verify POI-Demand relationship and investigate discrepancies.
"""

import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
input_file = project_root / "data" / "processed" / "daily" / "daily_with_poi_features.parquet"

print("Loading data...")
df = pd.read_parquet(input_file)

# Aggregate by station
station_data = df.groupby('station_name').agg({
    'departure_count': ['mean', 'sum', 'count'],
    'nearby_subway_stations_count': 'first',
    'nearby_restaurants_count': 'first',
    'nearby_shops_count': 'first',
    'nearby_offices_count': 'first',
    'nearby_universities_count': 'first',
    'date': ['min', 'max']
}).reset_index()

station_data.columns = ['station_name', 'avg_departures', 'total_departures', 'days_count',
                        'subway', 'restaurants', 'shops', 'offices', 'universities',
                        'first_date', 'last_date']

# Calculate total POI
station_data['total_poi'] = (station_data['subway'] + station_data['restaurants'] + 
                             station_data['shops'] + station_data['offices'] + 
                             station_data['universities'])

# Calculate operational years
station_data['years_active'] = (station_data['last_date'] - station_data['first_date']).dt.days / 365

print("\n" + "="*80)
print("MIT Station Analysis")
print("="*80)
mit_station = station_data[station_data['station_name'].str.contains('MIT', case=False, na=False)]
if not mit_station.empty:
    for _, row in mit_station.iterrows():
        print(f"\nStation: {row['station_name']}")
        print(f"  Avg Daily Departures: {row['avg_departures']:.1f}")
        print(f"  Total Departures: {row['total_departures']:.0f}")
        print(f"  Days of Data: {row['days_count']}")
        print(f"  Years Active: {row['years_active']:.1f}")
        print(f"  First Date: {row['first_date'].date()}")
        print(f"  Last Date: {row['last_date'].date()}")
        print(f"\n  POI Breakdown:")
        print(f"    Subway: {row['subway']}")
        print(f"    Restaurants: {row['restaurants']}")
        print(f"    Shops: {row['shops']}")
        print(f"    Offices: {row['offices']}")
        print(f"    Universities: {row['universities']}")
        print(f"    Total POI: {row['total_poi']}")

print("\n" + "="*80)
print("Top 10 by Average Daily Departures")
print("="*80)
top_10_avg = station_data.nlargest(10, 'avg_departures')
for rank, (_, row) in enumerate(top_10_avg.iterrows(), 1):
    print(f"\n{rank}. {row['station_name']}")
    print(f"   Avg: {row['avg_departures']:.1f}/day | Total POI: {row['total_poi']} | "
          f"Years: {row['years_active']:.1f}")
    print(f"   Subway:{row['subway']} Restaurants:{row['restaurants']} "
          f"Shops:{row['shops']} Universities:{row['universities']}")

print("\n" + "="*80)
print("Top 10 by Total Departures (All Time)")
print("="*80)
top_10_total = station_data.nlargest(10, 'total_departures')
for rank, (_, row) in enumerate(top_10_total.iterrows(), 1):
    print(f"\n{rank}. {row['station_name']}")
    print(f"   Total: {row['total_departures']:.0f} | Avg: {row['avg_departures']:.1f}/day | "
          f"POI: {row['total_poi']} | Years: {row['years_active']:.1f}")

print("\n" + "="*80)
print("Top 10 by Total POI Count")
print("="*80)
top_10_poi = station_data.nlargest(10, 'total_poi')
for rank, (_, row) in enumerate(top_10_poi.iterrows(), 1):
    print(f"\n{rank}. {row['station_name']}")
    print(f"   Total POI: {row['total_poi']} | Avg Departures: {row['avg_departures']:.1f}/day")
    print(f"   Subway:{row['subway']} Restaurants:{row['restaurants']} "
          f"Shops:{row['shops']} Universities:{row['universities']}")

# Correlation analysis
print("\n" + "="*80)
print("Correlation Analysis")
print("="*80)
print(f"Correlation between Total POI and Avg Departures: "
      f"{station_data['total_poi'].corr(station_data['avg_departures']):.3f}")
print(f"Correlation between Subway Count and Avg Departures: "
      f"{station_data['subway'].corr(station_data['avg_departures']):.3f}")
print(f"Correlation between Restaurants and Avg Departures: "
      f"{station_data['restaurants'].corr(station_data['avg_departures']):.3f}")
print(f"Correlation between Universities and Avg Departures: "
      f"{station_data['universities'].corr(station_data['avg_departures']):.3f}")

# Check if old stations have lower averages
print("\n" + "="*80)
print("Old vs New Stations")
print("="*80)
old_stations = station_data[station_data['years_active'] > 7]
new_stations = station_data[station_data['years_active'] < 3]
print(f"Old stations (>7 years): Avg departures = {old_stations['avg_departures'].mean():.1f}")
print(f"New stations (<3 years): Avg departures = {new_stations['avg_departures'].mean():.1f}")

