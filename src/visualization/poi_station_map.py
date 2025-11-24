"""
Interactive map comparing different POI types with actual ridership demand.

Allows toggling between different POI categories to analyze which urban features
correlate most strongly with bike demand by comparing Top 10 lists.
"""

import pandas as pd
import folium
from folium import plugins
from pathlib import Path
import numpy as np

class POIStationMapper:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.input_file = self.project_root / "data" / "processed" / "daily" / "daily_with_poi_features.parquet"
        self.output_dir = self.project_root / "results" / "poi_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "station_type_map.html"
    
    def classify_stations(self, df):
        """Classify stations based on POI features."""
        
        # Get base columns
        base_cols = ['station_name', 'latitude', 'longitude']
        
        # Add POI columns that exist
        poi_cols = ['nearby_subway_stations_count', 'nearby_bus_stops_count',
                   'nearby_restaurants_count', 'nearby_shops_count',
                   'nearby_offices_count', 'nearby_universities_count',
                   'nearby_schools_count', 'nearby_hospitals_count',
                   'nearby_banks_count', 'nearby_parks_area_sqm',
                   'is_transit_hub', 'is_commercial_area', 'is_near_university']
        
        available_cols = base_cols + [col for col in poi_cols if col in df.columns]
        stations = df[available_cols].drop_duplicates()
        
        # Fill missing POI columns with 0
        for col in poi_cols:
            if col not in stations.columns:
                stations[col] = 0
        
        def get_station_type(row):
            """Determine primary station type based on POI features."""
            scores = {
                'Transit Hub': 0,
                'Commercial': 0,
                'University': 0,
                'Park/Recreation': 0,
                'Office': 0,
                'Residential': 0
            }
            
            # Transit hub scoring
            if row['is_transit_hub'] == 1:
                scores['Transit Hub'] += 3
            scores['Transit Hub'] += row['nearby_subway_stations_count'] * 2
            
            # Commercial scoring
            if row['is_commercial_area'] == 1:
                scores['Commercial'] += 3
            scores['Commercial'] += (row['nearby_shops_count'] + row['nearby_restaurants_count']) / 10
            
            # University scoring
            if row['is_near_university'] == 1:
                scores['University'] += 5
            
            # Park/Recreation scoring
            scores['Park/Recreation'] += row['nearby_parks_area_sqm'] / 10000  # per hectare
            
            # Office scoring
            scores['Office'] += row['nearby_offices_count'] / 5
            
            # Residential is default if all other scores are low
            if max(scores.values()) < 2:
                scores['Residential'] = 3
            
            return max(scores, key=scores.get)
        
        stations['station_type'] = stations.apply(get_station_type, axis=1)
        
        return stations
    
    def create_map_visualization(self):
        """Create interactive map visualization of station types."""
        print("\n" + "="*80)
        print("POI Station Map Visualization")
        print("="*80)
        
        print(f"\nLoading data from: {self.input_file}")
        df = pd.read_parquet(self.input_file)
        print(f"Loaded {len(df)} records")
        
        print("\nClassifying stations...")
        stations = self.classify_stations(df)
        print(f"Classified {len(stations)} unique stations")
        
        type_counts = stations['station_type'].value_counts()
        print("\nStation distribution:")
        for station_type, count in type_counts.items():
            print(f"  {station_type}: {count}")
        
        # Get average demand per station
        df_with_demand = df.groupby('station_name').agg({
            'departure_count': 'mean',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        stations = stations.merge(df_with_demand, on='station_name', suffixes=('', '_demand'))
        
        # Create base map centered on Boston
        center_lat = stations['latitude'].mean()
        center_lon = stations['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap',
            control_scale=True
        )
        
        # Color scheme
        colors = {
            'Transit Hub': '#e74c3c',
            'Commercial': '#f39c12',
            'University': '#9b59b6',
            'Park/Recreation': '#2ecc71',
            'Office': '#3498db',
            'Residential': '#95a5a6'
        }
        
        # Icon shapes
        icons = {
            'Transit Hub': 'train',
            'Commercial': 'shopping-cart',
            'University': 'graduation-cap',
            'Park/Recreation': 'tree',
            'Office': 'briefcase',
            'Residential': 'home'
        }
        
        # Normalize demand for circle size (5-20 range)
        min_demand = stations['departure_count'].min()
        max_demand = stations['departure_count'].max()
        stations['demand_circle_size'] = 5 + 15 * (stations['departure_count'] - min_demand) / (max_demand - min_demand)
        
        # Define POI categories to analyze with colors
        poi_categories = {
            'Subway Stations': {
                'column': 'nearby_subway_stations_count',
                'color': '#e74c3c'  # Red
            },
            'Bus Stops': {
                'column': 'nearby_bus_stops_count',
                'color': '#f39c12'  # Orange
            },
            'Restaurants': {
                'column': 'nearby_restaurants_count',
                'color': '#e67e22'  # Dark orange
            },
            'Shops': {
                'column': 'nearby_shops_count',
                'color': '#9b59b6'  # Purple
            },
            'Offices': {
                'column': 'nearby_offices_count',
                'color': '#34495e'  # Dark blue-gray
            },
            'Universities': {
                'column': 'nearby_universities_count',
                'color': '#8e44ad'  # Dark purple
            },
            'Schools': {
                'column': 'nearby_schools_count',
                'color': '#16a085'  # Teal
            },
            'Hospitals': {
                'column': 'nearby_hospitals_count',
                'color': '#c0392b'  # Dark red
            },
            'Banks': {
                'column': 'nearby_banks_count',
                'color': '#27ae60'  # Green
            },
            'Parks': {
                'column': 'nearby_parks_area_sqm',
                'color': '#2ecc71'  # Light green
            }
        }
        
        # Create base layer: All stations by ridership
        base_fg = folium.FeatureGroup(name='All Stations (by Ridership)', show=True)
        
        # Create feature groups for each POI category
        poi_feature_groups = {}
        poi_top10_groups = {}
        
        for category_name, category_info in poi_categories.items():
            col_name = category_info['column']
            if col_name in stations.columns:
                poi_feature_groups[category_name] = folium.FeatureGroup(
                    name=f'[POI] {category_name}', 
                    show=False
                )
                poi_top10_groups[category_name] = folium.FeatureGroup(
                    name=f'  └─ Top 10 {category_name}', 
                    show=False
                )
        
        # Create Top 10 ridership group
        top_demand_fg = folium.FeatureGroup(name='Top 10 by Ridership', show=True)
        
        # Add base layer markers (all stations by ridership)
        for _, row in stations.iterrows():
            station_type = row['station_type']
            
            # Create popup content
            poi_items = []
            for category_name, category_info in poi_categories.items():
                col_name = category_info['column']
                if col_name in row.index:
                    if col_name == 'nearby_parks_area_sqm':
                        poi_items.append(f"• {category_name}: {int(row[col_name])} m²")
                    else:
                        poi_items.append(f"• {category_name}: {int(row[col_name])}")
            
            poi_details = "<br>".join(poi_items) if poi_items else "No POI data"
            
            popup_html = f"""
            <div style="font-family: Arial; width: 300px;">
                <h4 style="margin-bottom: 10px;">{row['station_name']}</h4>
                <hr style="margin: 5px 0;">
                <b>Type:</b> {station_type}<br>
                <b>Avg Daily Departures:</b> {row['departure_count']:.0f}<br>
                <hr style="margin: 5px 0;">
                <b>Nearby POI:</b><br>
                {poi_details}
            </div>
            """
            
            # Base marker by ridership - uniform blue color
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['demand_circle_size'],
                popup=folium.Popup(popup_html, max_width=320),
                color='black',
                fillColor='#3498db',  # Uniform blue for all stations
                fillOpacity=0.6,
                weight=1,
                tooltip=f"{row['station_name']} ({row['departure_count']:.0f}/day)"
            ).add_to(base_fg)
        
        # Add POI-specific layers
        for category_name, category_info in poi_categories.items():
            col_name = category_info['column']
            color = category_info['color']
            
            if col_name not in stations.columns:
                continue
            
            # Normalize this POI type for circle size
            min_poi = stations[col_name].min()
            max_poi = stations[col_name].max()
            
            if max_poi > min_poi:
                # For stations with POI=0, use minimum size (3), others scale from 5-20
                stations[f'{col_name}_size'] = stations[col_name].apply(
                    lambda x: 3 if x == 0 else 5 + 15 * (x - min_poi) / (max_poi - min_poi)
                )
            else:
                stations[f'{col_name}_size'] = 10
            
            # Add markers for this POI category (show all stations, even with 0 POI)
            for _, row in stations.iterrows():
                
                station_type = row['station_type']
                poi_value = row[col_name]
                
                unit = " m²" if col_name == 'nearby_parks_area_sqm' else ""
                
                popup_html = f"""
                <div style="font-family: Arial; width: 280px;">
                    <h4 style="margin-bottom: 10px;">{row['station_name']}</h4>
                    <hr style="margin: 5px 0;">
                    <b>{category_name}:</b> {int(poi_value)}{unit}<br>
                    <b>Daily Departures:</b> {row['departure_count']:.0f}<br>
                    <b>Type:</b> {station_type}
                </div>
                """
                
                # Lower opacity for stations with POI=0
                opacity = 0.2 if poi_value == 0 else 0.6
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=row[f'{col_name}_size'],
                    popup=folium.Popup(popup_html, max_width=300),
                    color='black',
                    fillColor=color,
                    fillOpacity=opacity,
                    weight=1,
                    tooltip=f"{row['station_name']} ({int(poi_value)}{unit})"
                ).add_to(poi_feature_groups[category_name])
        
        # Add base layer
        base_fg.add_to(m)
        
        # Add Top 10 by ridership
        top_10_demand = stations.nlargest(10, 'departure_count')
        
        for rank, (_, row) in enumerate(top_10_demand.iterrows(), 1):
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(
                    f"<b>#{rank} Busiest Station</b><br>"
                    f"{row['station_name']}<br>"
                    f"Daily Departures: <b>{row['departure_count']:.0f}</b>",
                    max_width=250
                ),
                tooltip=f"#{rank} - {row['station_name']}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(top_demand_fg)
        
        top_demand_fg.add_to(m)
        
        # Add POI-specific layers and their Top 10s
        for category_name, category_info in poi_categories.items():
            col_name = category_info['column']
            
            if col_name not in stations.columns:
                continue
            
            # Add POI layer
            poi_feature_groups[category_name].add_to(m)
            
            # Add Top 10 for this POI category
            top_10_poi = stations.nlargest(10, col_name)
            
            unit = " m²" if col_name == 'nearby_parks_area_sqm' else ""
            
            for rank, (_, row) in enumerate(top_10_poi.iterrows(), 1):
                
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=folium.Popup(
                        f"<b>#{rank} in {category_name}</b><br>"
                        f"{row['station_name']}<br>"
                        f"{category_name}: <b>{int(row[col_name])}{unit}</b><br>"
                        f"Daily Departures: {row['departure_count']:.0f}",
                        max_width=280
                    ),
                    tooltip=f"#{rank} - {row['station_name']}",
                    icon=folium.Icon(color='blue', icon='info-sign', prefix='glyphicon')
                ).add_to(poi_top10_groups[category_name])
            
            poi_top10_groups[category_name].add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 280px; max-height: 600px; overflow-y: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:10px; padding: 8px; border-radius: 5px;">
            <h4 style="margin: 0 0 8px 0; font-size: 13px;">Color Legend</h4>
            
            <p style="margin: 2px 0; font-weight: bold;">Base Layer:</p>
            <p style="margin: 2px 0 6px 10px;"><span style="color: #3498db;">⬤</span> Ridership (Blue)</p>
            
            <p style="margin: 2px 0; font-weight: bold;">POI Layers:</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #e74c3c;">⬤</span> Subway Stations</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #f39c12;">⬤</span> Bus Stops</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #e67e22;">⬤</span> Restaurants</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #9b59b6;">⬤</span> Shops</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #34495e;">⬤</span> Offices</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #8e44ad;">⬤</span> Universities</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #16a085;">⬤</span> Schools</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #c0392b;">⬤</span> Hospitals</p>
            <p style="margin: 2px 0 2px 10px;"><span style="color: #27ae60;">⬤</span> Banks</p>
            <p style="margin: 2px 0 6px 10px;"><span style="color: #2ecc71;">⬤</span> Parks</p>
            
            <hr style="margin: 6px 0;">
            <p style="margin: 3px 0;"><b>⭐</b> Top 10 by Ridership</p>
            <p style="margin: 3px 0;"><b>ℹ️</b> Top 10 by each POI</p>
            <p style="margin: 3px 0;">Circle size = Value</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add title
        title_html = '''
        <div style="position: fixed; 
                    top: 10px; left: 50px; width: 580px; height: 110px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 5px;">
            <h3 style="margin: 0; padding: 0;">POI Category Analysis: Which Urban Features Drive Bike Demand?</h3>
            <p style="margin: 5px 0; font-size: 12px; color: #666;">
                Compare different POI categories with actual ridership to identify key demand drivers
            </p>
            <p style="margin: 5px 0; font-size: 11px; color: #999; font-style: italic;">
                Toggle individual POI layers and compare their Top 10 with the ridership Top 10. High overlap = strong correlation.
            </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        print(f"\nSaving interactive map to: {self.output_file}")
        m.save(str(self.output_file))
        print(f"Map saved: {self.output_file}")
        
        # Save station classification data
        classification_file = self.output_dir / "station_classifications.csv"
        stations_summary = stations[['station_name', 'latitude', 'longitude', 'station_type']].copy()
        
        # Add POI counts for reference
        stations_summary = stations_summary.merge(
            stations[['station_name', 'nearby_subway_stations_count', 'nearby_shops_count',
                     'nearby_restaurants_count', 'nearby_universities_count', 'nearby_offices_count']],
            on='station_name'
        )
        
        stations_summary.to_csv(classification_file, index=False)
        print(f"\nStation classifications saved to: {classification_file}")
        
        # Print some interesting findings
        print("\n" + "="*80)
        print("Key Insights")
        print("="*80)
        
        transit_hubs = stations[stations['station_type'] == 'Transit Hub']
        if len(transit_hubs) > 0:
            print(f"\nTransit Hubs ({len(transit_hubs)} stations):")
            top_transit = transit_hubs.nlargest(3, 'nearby_subway_stations_count')
            for _, row in top_transit.iterrows():
                print(f"  - {row['station_name']}: {int(row['nearby_subway_stations_count'])} subway stations nearby")
        
        commercial = stations[stations['station_type'] == 'Commercial']
        if len(commercial) > 0:
            print(f"\nCommercial Areas ({len(commercial)} stations):")
            commercial['total_commercial'] = commercial['nearby_shops_count'] + commercial['nearby_restaurants_count']
            top_commercial = commercial.nlargest(3, 'total_commercial')
            for _, row in top_commercial.iterrows():
                print(f"  - {row['station_name']}: {int(row['nearby_shops_count'])} shops, {int(row['nearby_restaurants_count'])} restaurants")
        
        university = stations[stations['station_type'] == 'University']
        if len(university) > 0:
            print(f"\nUniversity Areas ({len(university)} stations):")
            for _, row in university.head(5).iterrows():
                print(f"  - {row['station_name']}")
        
        print(f"\n" + "="*80)
        print("Top 10 Stations by Ridership")
        print("="*80)
        for rank, (_, row) in enumerate(top_10_demand.iterrows(), 1):
            print(f"  {rank}. {row['station_name']}: {row['departure_count']:.0f} daily departures")
        
        # Analyze overlap between each POI category and demand
        print(f"\n" + "="*80)
        print("POI Category Correlation Analysis (Top 10 Overlap)")
        print("="*80)
        
        demand_top10_set = set(top_10_demand['station_name'].values)
        
        overlap_results = []
        for category_name, category_info in poi_categories.items():
            col_name = category_info['column']
            
            if col_name not in stations.columns:
                continue
            
            top_10_poi = stations.nlargest(10, col_name)
            poi_top10_set = set(top_10_poi['station_name'].values)
            
            overlap = demand_top10_set & poi_top10_set
            overlap_pct = len(overlap) / 10 * 100
            
            overlap_results.append((category_name, len(overlap), overlap_pct, overlap))
        
        # Sort by overlap percentage
        overlap_results.sort(key=lambda x: x[1], reverse=True)
        
        for category_name, count, pct, overlap_stations in overlap_results:
            print(f"\n{category_name}:")
            print(f"  Overlap: {count}/10 stations ({pct:.0f}%)")
            if overlap_stations:
                print(f"  Common stations: {', '.join(sorted(overlap_stations))}")
        
        print("\n" + "="*80)
        print("Interpretation:")
        print("="*80)
        print("  High overlap (70-100%) = Strong correlation with ridership")
        print("  Medium overlap (40-60%) = Moderate correlation")
        print("  Low overlap (0-30%) = Weak correlation")
        
        print("\nVisualization complete")
        print("\nMap features:")
        print("  • Base layer: All stations sized by ridership")
        print("  • [POI] layers: Toggle to see individual POI distributions")
        print("  • ⭐ Red stars: Top 10 by ridership (always visible)")
        print("  • ℹ️ Blue icons: Top 10 for each POI category")
        print("  • Compare overlaps to identify key demand drivers!")

def main():
    mapper = POIStationMapper()
    mapper.create_map_visualization()

if __name__ == "__main__":
    main()

