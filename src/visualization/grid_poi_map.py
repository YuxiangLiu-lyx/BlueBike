"""
Visualize grid POI distribution by category on interactive map.
"""

import pandas as pd
import folium
from pathlib import Path


def get_project_root():
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class GridPOIMapper:
    def __init__(self):
        self.project_root = get_project_root()
        self.input_file = self.project_root / "data" / "processed" / "grid" / "grid_with_poi_features.parquet"
        self.output_dir = self.project_root / "results" / "grid_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "grid_poi_map.html"
        
        self.poi_categories = {
            'Subway Stations': {'column': 'nearby_subway_stations_count', 'color': '#e74c3c'},
            'Bus Stops': {'column': 'nearby_bus_stops_count', 'color': '#f39c12'},
            'Restaurants': {'column': 'nearby_restaurants_count', 'color': '#e67e22'},
            'Shops': {'column': 'nearby_shops_count', 'color': '#3498db'},
            'Offices': {'column': 'nearby_offices_count', 'color': '#9b59b6'},
            'Universities': {'column': 'nearby_universities_count', 'color': '#1abc9c'},
            'Schools': {'column': 'nearby_schools_count', 'color': '#16a085'},
            'Hospitals': {'column': 'nearby_hospitals_count', 'color': '#c0392b'},
            'Banks': {'column': 'nearby_banks_count', 'color': '#34495e'},
            'Parks': {'column': 'nearby_parks_area_sqm', 'color': '#27ae60', 'is_area': True}
        }
    
    def create_map(self):
        print("Loading grid POI data...")
        df = pd.read_parquet(self.input_file)
        print(f"Loaded: {len(df)} grid cells")
        
        # Calculate total POI count
        df['total_poi'] = (
            df['nearby_subway_stations_count'] +
            df['nearby_bus_stops_count'] +
            df['nearby_restaurants_count'] +
            df['nearby_shops_count'] +
            df['nearby_offices_count'] +
            df['nearby_universities_count'] +
            df['nearby_schools_count'] +
            df['nearby_hospitals_count'] +
            df['nearby_banks_count']
        )
        
        print(f"\nPOI statistics:")
        print(f"  Total grids: {len(df)}")
        print(f"  Grids with POI: {(df['total_poi'] > 0).sum()}")
        print(f"  Average POI per grid: {df['total_poi'].mean():.1f}")
        
        # Create base map
        center_lat = df['center_lat'].mean()
        center_lon = df['center_lon'].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        print("\nCreating map layers...")
        
        # Base layer: Total POI density (uniform color, size by total count)
        base_fg = folium.FeatureGroup(name='Total POI Density', show=True)
        df_with_poi = df[df['total_poi'] > 0].copy()
        
        if len(df_with_poi) > 0:
            max_total = df_with_poi['total_poi'].max()
            df_with_poi['circle_size'] = 5 + (df_with_poi['total_poi'] / max_total * 25)
            
            for _, row in df_with_poi.iterrows():
                popup_html = f"""
                <div style="font-family: Arial; min-width: 200px;">
                    <h4>Grid {int(row['grid_id'])}</h4>
                    <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                    <b>Total POI:</b> {int(row['total_poi'])}<br>
                    <hr style="margin: 5px 0;">
                    <b>Breakdown:</b><br>
                    &nbsp;&nbsp;Subway: {int(row['nearby_subway_stations_count'])}<br>
                    &nbsp;&nbsp;Bus: {int(row['nearby_bus_stops_count'])}<br>
                    &nbsp;&nbsp;Restaurants: {int(row['nearby_restaurants_count'])}<br>
                    &nbsp;&nbsp;Shops: {int(row['nearby_shops_count'])}<br>
                    &nbsp;&nbsp;Offices: {int(row['nearby_offices_count'])}<br>
                    &nbsp;&nbsp;Universities: {int(row['nearby_universities_count'])}<br>
                    &nbsp;&nbsp;Schools: {int(row['nearby_schools_count'])}<br>
                    &nbsp;&nbsp;Hospitals: {int(row['nearby_hospitals_count'])}<br>
                    &nbsp;&nbsp;Banks: {int(row['nearby_banks_count'])}<br>
                    &nbsp;&nbsp;Park area: {int(row['nearby_parks_area_sqm'])} m²<br>
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['center_lat'], row['center_lon']],
                    radius=row['circle_size'],
                    popup=folium.Popup(popup_html, max_width=300),
                    color='#2c3e50',
                    fill=True,
                    fillColor='#2c3e50',
                    fillOpacity=0.5,
                    weight=1
                ).add_to(base_fg)
        
        base_fg.add_to(m)
        
        # Individual POI category layers
        for poi_name, poi_info in self.poi_categories.items():
            poi_col = poi_info['column']
            poi_color = poi_info['color']
            is_area = poi_info.get('is_area', False)
            
            poi_fg = folium.FeatureGroup(name=f'[POI] {poi_name}', show=False)
            
            df_poi = df[df[poi_col] > 0].copy()
            
            if len(df_poi) > 0:
                max_poi = df_poi[poi_col].max()
                
                if is_area:
                    df_poi['poi_circle_size'] = 5 + (df_poi[poi_col] / max_poi * 25)
                else:
                    df_poi['poi_circle_size'] = 5 + (df_poi[poi_col] / max_poi * 25)
                
                for _, row in df_poi.iterrows():
                    if is_area:
                        value_str = f"{int(row[poi_col])} m²"
                    else:
                        value_str = f"{int(row[poi_col])}"
                    
                    popup_html = f"""
                    <div style="font-family: Arial; min-width: 180px;">
                        <h4>{poi_name}</h4>
                        <b>Grid {int(row['grid_id'])}</b><br>
                        <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                        <hr style="margin: 5px 0;">
                        <b>{poi_name}:</b> {value_str}<br>
                        <b>Total POI:</b> {int(row['total_poi'])}<br>
                    </div>
                    """
                    
                    folium.CircleMarker(
                        location=[row['center_lat'], row['center_lon']],
                        radius=row['poi_circle_size'],
                        popup=folium.Popup(popup_html, max_width=250),
                        color=poi_color,
                        fill=True,
                        fillColor=poi_color,
                        fillOpacity=0.6,
                        weight=1
                    ).add_to(poi_fg)
            
            poi_fg.add_to(m)
            print(f"  {poi_name}: {len(df_poi)} grids")
        
        # Add Top 10 markers for total POI
        top10_fg = folium.FeatureGroup(name='Top 10 Grids', show=True)
        top10 = df.nlargest(10, 'total_poi')
        
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            folium.Marker(
                location=[row['center_lat'], row['center_lon']],
                icon=folium.Icon(color='red', icon='star'),
                popup=f"<b>Rank #{rank}</b><br>Grid {int(row['grid_id'])}<br>Total POI: {int(row['total_poi'])}"
            ).add_to(top10_fg)
        
        top10_fg.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 10px; width: 280px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:13px; padding: 12px">
        <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 14px;">Grid POI Distribution</p>
        <p style="margin: 5px 0; font-size: 12px;">
            <b>Total Grids:</b> {len(df)}<br>
            <b>Grids with POI:</b> {(df['total_poi'] > 0).sum()}<br>
            <b>Coverage:</b> Boston + Cambridge + Somerville<br>
            <b>Grid size:</b> 500m × 500m
        </p>
        <hr style="margin: 8px 0;">
        <p style="margin: 5px 0 3px 0; font-weight: bold;">Layers:</p>
        <p style="margin: 2px 0; font-size: 12px;">
            <span style="color: #2c3e50; font-size: 16px;">●</span> Total POI Density<br>
        '''
        
        for poi_name, poi_info in self.poi_categories.items():
            legend_html += f'<span style="color: {poi_info["color"]}; font-size: 16px;">●</span> {poi_name}<br>\n'
        
        legend_html += '''
            <span style="color: red; font-size: 16px;">★</span> Top 10 Grids
        </p>
        <hr style="margin: 8px 0;">
        <p style="margin: 5px 0 0 0; font-size: 11px; color: #666;">
            Circle size indicates count/area<br>
            Click circles for details
        </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(str(self.output_file))
        print(f"\nMap saved: {self.output_file}")
        
        # Print top 10
        print(f"\nTop 10 grids by total POI:")
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"  #{rank:2d} Grid {int(row['grid_id']):4d}: {int(row['total_poi']):3d} POIs "
                  f"at ({row['center_lat']:.4f}, {row['center_lon']:.4f})")
        
        return m


def main():
    mapper = GridPOIMapper()
    mapper.create_map()
    print("\nVisualization complete")


if __name__ == "__main__":
    main()
