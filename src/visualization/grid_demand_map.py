"""
Visualize grid demand predictions vs actual station distribution.
"""

import pandas as pd
import folium
from pathlib import Path


def get_project_root():
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class GridDemandMapper:
    def __init__(self):
        self.project_root = get_project_root()
        self.predictions_file = self.project_root / "results" / "grid_analysis" / "grid_predictions.parquet"
        self.output_dir = self.project_root / "results" / "grid_analysis"
        self.output_file = self.output_dir / "grid_demand_map.html"
    
    def create_map(self):
        print("Loading grid predictions...")
        df = pd.read_parquet(self.predictions_file)
        print(f"Loaded: {len(df)} grid cells")
        
        # Calculate statistics
        print(f"\nDemand statistics:")
        print(f"  Predicted demand range: {df['predicted_demand'].min():.1f} - {df['predicted_demand'].max():.1f}")
        print(f"  Actual demand range: {df['actual_demand'].min():.1f} - {df['actual_demand'].max():.1f}")
        print(f"  Grids with stations: {(df['station_count'] > 0).sum()}")
        
        # Get Top 20s and Top 100s
        top20_predicted = df.nlargest(20, 'predicted_demand')
        top20_stations = df.nlargest(20, 'station_count')
        top100_predicted = df.nlargest(100, 'predicted_demand')
        top100_stations = df.nlargest(100, 'station_count')
        
        print(f"\nTop 20 by predicted demand:")
        print(f"  Average predicted: {top20_predicted['predicted_demand'].mean():.1f}")
        print(f"  Have stations: {(top20_predicted['station_count'] > 0).sum()}/20")
        
        print(f"\nTop 20 by station count:")
        print(f"  Average stations: {top20_stations['station_count'].mean():.1f}")
        print(f"  Average predicted: {top20_stations['predicted_demand'].mean():.1f}")
        print(f"  Average actual: {top20_stations['actual_demand'].mean():.1f}")
        
        print(f"\nTop 100 by predicted demand:")
        print(f"  Average predicted: {top100_predicted['predicted_demand'].mean():.1f}")
        print(f"  Have stations: {(top100_predicted['station_count'] > 0).sum()}/100")
        print(f"  Average actual: {top100_predicted['actual_demand'].mean():.1f}")
        
        print(f"\nTop 100 by station count:")
        print(f"  Average stations: {top100_stations['station_count'].mean():.1f}")
        print(f"  Average predicted: {top100_stations['predicted_demand'].mean():.1f}")
        print(f"  Average actual: {top100_stations['actual_demand'].mean():.1f}")
        
        # Create base map
        center_lat = df['center_lat'].mean()
        center_lon = df['center_lon'].mean()
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        print("\nCreating map layers...")
        
        # Layer 1: All predicted demand (background heatmap style)
        demand_fg = folium.FeatureGroup(name='Predicted Demand (All Grids)', show=True)
        
        df_with_demand = df[df['predicted_demand'] > 0].copy()
        max_demand = df_with_demand['predicted_demand'].max()
        df_with_demand['circle_size'] = 5 + (df_with_demand['predicted_demand'] / max_demand * 20)
        
        def get_demand_color(demand):
            if demand > 60:
                return '#d73027'
            elif demand > 40:
                return '#fc8d59'
            elif demand > 25:
                return '#fee090'
            elif demand > 15:
                return '#91bfdb'
            else:
                return '#4575b4'
        
        for _, row in df_with_demand.iterrows():
            folium.CircleMarker(
                location=[row['center_lat'], row['center_lon']],
                radius=row['circle_size'],
                popup=f"Grid {int(row['grid_id'])}<br>"
                      f"Predicted: {row['predicted_demand']:.1f}<br>"
                      f"Actual: {row['actual_demand']:.1f}<br>"
                      f"Stations: {int(row['station_count'])}",
                color=get_demand_color(row['predicted_demand']),
                fill=True,
                fillColor=get_demand_color(row['predicted_demand']),
                fillOpacity=0.5,
                weight=1
            ).add_to(demand_fg)
        
        demand_fg.add_to(m)
        
        # Find overlapping grids for both Top 20 and Top 100
        top_pred_ids = set(top20_predicted['grid_id'])
        top_station_ids = set(top20_stations['grid_id'])
        overlap_ids = top_pred_ids & top_station_ids
        
        top100_pred_ids = set(top100_predicted['grid_id'])
        top100_station_ids = set(top100_stations['grid_id'])
        overlap100_ids = top100_pred_ids & top100_station_ids
        
        # Layer 2: Top 100 Overlapping grids (orange stars)
        overlap100_fg = folium.FeatureGroup(name='⭐ Both Top 100s (32 grids)', show=True)
        
        overlap100_df = df[df['grid_id'].isin(overlap100_ids)].copy()
        overlap100_df = overlap100_df.sort_values('predicted_demand', ascending=False)
        
        for _, row in overlap100_df.iterrows():
            popup_html = f"""
            <div style="font-family: Arial; min-width: 240px;">
                <h4 style="color: #ff9800; margin: 0 0 8px 0;">
                    ⭐ In Both Top 100s
                </h4>
                <b>Grid {int(row['grid_id'])}</b><br>
                <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                <hr style="margin: 8px 0;">
                <b>Predicted Demand:</b> {row['predicted_demand']:.1f}/day<br>
                <b>Actual Demand:</b> {row['actual_demand']:.1f}/day<br>
                <b>Stations:</b> {int(row['station_count'])}<br>
                <b>Ratio:</b> Actual/Predicted = {row['actual_demand']/row['predicted_demand']:.2f}x<br>
            </div>
            """
            
            folium.Marker(
                location=[row['center_lat'], row['center_lon']],
                icon=folium.Icon(color='orange', icon='star', prefix='fa'),
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(overlap100_fg)
        
        overlap100_fg.add_to(m)
        
        # Layer 3: Top 20 Overlapping grids (purple stars)
        overlap_fg = folium.FeatureGroup(name='⭐ Both Top 20s (3 grids)', show=True)
        
        for rank_p, (_, row) in enumerate(top20_predicted.iterrows(), 1):
            if row['grid_id'] in overlap_ids:
                # Find rank in station list
                rank_s = top20_stations[top20_stations['grid_id'] == row['grid_id']].index[0]
                rank_s = list(top20_stations.index).index(rank_s) + 1
                
                popup_html = f"""
                <div style="font-family: Arial; min-width: 240px;">
                    <h4 style="color: #9b59b6; margin: 0 0 8px 0;">
                        ⭐ In Both Top 20s
                    </h4>
                    <b>Grid {int(row['grid_id'])}</b><br>
                    <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                    <hr style="margin: 8px 0;">
                    <b>Rank in Predicted:</b> #{rank_p}<br>
                    <b>Rank in Stations:</b> #{rank_s}<br>
                    <hr style="margin: 8px 0;">
                    <b>Predicted Demand:</b> {row['predicted_demand']:.1f}/day<br>
                    <b>Actual Demand:</b> {row['actual_demand']:.1f}/day<br>
                    <b>Stations:</b> {int(row['station_count'])}<br>
                    <b>Ratio:</b> Actual/Predicted = {row['actual_demand']/row['predicted_demand']:.2f}x<br>
                </div>
                """
                
                folium.Marker(
                    location=[row['center_lat'], row['center_lon']],
                    icon=folium.Icon(color='purple', icon='star', prefix='fa'),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(overlap_fg)
        
        overlap_fg.add_to(m)
        
        # Layer 4: Top 20 by predicted demand (excluding overlaps)
        top_pred_fg = folium.FeatureGroup(name='Top 20 Predicted Demand', show=True)
        
        for rank, (_, row) in enumerate(top20_predicted.iterrows(), 1):
            if row['grid_id'] not in overlap_ids:  # Skip overlaps
                popup_html = f"""
                <div style="font-family: Arial; min-width: 220px;">
                    <h4 style="color: #d73027; margin: 0 0 8px 0;">
                        #{rank} Predicted Demand
                    </h4>
                    <b>Grid {int(row['grid_id'])}</b><br>
                    <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                    <hr style="margin: 8px 0;">
                    <b>Predicted Demand:</b> {row['predicted_demand']:.1f}/day<br>
                    <b>Actual Demand:</b> {row['actual_demand']:.1f}/day<br>
                    <b>Stations:</b> {int(row['station_count'])}<br>
                    <hr style="margin: 8px 0;">
                    <b>POI Summary:</b><br>
                    &nbsp;&nbsp;Subway: {int(row['nearby_subway_stations_count'])}<br>
                    &nbsp;&nbsp;Bus: {int(row['nearby_bus_stops_count'])}<br>
                    &nbsp;&nbsp;Restaurants: {int(row['nearby_restaurants_count'])}<br>
                    &nbsp;&nbsp;Shops: {int(row['nearby_shops_count'])}<br>
                </div>
                """
                
                folium.Marker(
                    location=[row['center_lat'], row['center_lon']],
                    icon=folium.Icon(color='red', icon='arrow-up', prefix='fa'),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(top_pred_fg)
        
        top_pred_fg.add_to(m)
        
        # Layer 5: Top 20 by station count (excluding overlaps)
        top_station_fg = folium.FeatureGroup(name='Top 20 Station Count', show=True)
        
        for rank, (_, row) in enumerate(top20_stations.iterrows(), 1):
            if row['grid_id'] not in overlap_ids:  # Skip overlaps
                popup_html = f"""
                <div style="font-family: Arial; min-width: 220px;">
                    <h4 style="color: #2ecc71; margin: 0 0 8px 0;">
                        #{rank} Station Count
                    </h4>
                    <b>Grid {int(row['grid_id'])}</b><br>
                    <b>Location:</b> ({row['center_lat']:.4f}, {row['center_lon']:.4f})<br>
                    <hr style="margin: 8px 0;">
                    <b>Stations:</b> {int(row['station_count'])}<br>
                    <b>Actual Demand:</b> {row['actual_demand']:.1f}/day<br>
                    <b>Predicted Demand:</b> {row['predicted_demand']:.1f}/day<br>
                    <hr style="margin: 8px 0;">
                    <b>Demand Ratio:</b> Actual/Predicted = {row['actual_demand']/row['predicted_demand']:.2f}x<br>
                </div>
                """
                
                folium.Marker(
                    location=[row['center_lat'], row['center_lon']],
                    icon=folium.Icon(color='green', icon='home', prefix='fa'),
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(top_station_fg)
        
        top_station_fg.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add legend
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 10px; width: 300px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:13px; padding: 12px">
        <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 15px;">
            Grid Demand Comparison
        </p>
        
        <p style="margin: 5px 0; font-size: 12px;">
            <b>Total Grids:</b> {len(df)}<br>
            <b>Grids with Stations:</b> {(df['station_count'] > 0).sum()}<br>
            <b>Date:</b> 2024-07-15 (Monday)
        </p>
        
        <hr style="margin: 8px 0;">
        
        <p style="margin: 5px 0 3px 0; font-weight: bold;">Predicted Demand:</p>
        <p style="margin: 2px 0; font-size: 12px;">
            <span style="color: #d73027; font-size: 16px;">●</span> Very High (>60)<br>
            <span style="color: #fc8d59; font-size: 16px;">●</span> High (40-60)<br>
            <span style="color: #fee090; font-size: 16px;">●</span> Medium (25-40)<br>
            <span style="color: #91bfdb; font-size: 16px;">●</span> Low (15-25)<br>
            <span style="color: #4575b4; font-size: 16px;">●</span> Very Low (<15)
        </p>
        
        <hr style="margin: 8px 0;">
        
        <p style="margin: 5px 0 3px 0; font-weight: bold;">Markers:</p>
        <p style="margin: 2px 0; font-size: 12px;">
            <span style="color: #ff9800; font-size: 18px;">★</span> Both Top 100s ({len(overlap100_ids)})<br>
            <span style="color: #9b59b6; font-size: 18px;">★</span> Both Top 20s ({len(overlap_ids)})<br>
            <span style="color: #d73027; font-size: 16px;">↑</span> Top 20 Predicted Only<br>
            <span style="color: #2ecc71; font-size: 16px;">⌂</span> Top 20 Stations Only
        </p>
        
        <hr style="margin: 8px 0;">
        
        <p style="margin: 5px 0 0 0; font-size: 11px; color: #666;">
            Circle size = demand magnitude<br>
            Toggle layers in top-right corner
        </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save map
        m.save(str(self.output_file))
        print(f"\nMap saved: {self.output_file}")
        
        # Print comparison analysis
        print("\n" + "="*80)
        print("Top 20 Comparison Analysis")
        print("="*80)
        
        # Overlap analysis for Top 20
        overlap = top_pred_ids & top_station_ids
        
        print(f"\nOverlap: {len(overlap)}/20 grids appear in both Top 20 lists")
        
        if len(overlap) > 0:
            print(f"\nGrids in both Top 20s:")
            overlap_df = df[df['grid_id'].isin(overlap)].sort_values('predicted_demand', ascending=False)
            for _, row in overlap_df.iterrows():
                print(f"  Grid {int(row['grid_id']):4d}: "
                      f"Predicted {row['predicted_demand']:5.1f}, "
                      f"Actual {row['actual_demand']:5.1f}, "
                      f"{int(row['station_count'])} stations")
        
        print(f"\nPredicted Top 20 only (no stations or few stations):")
        pred_only = top_pred_ids - top_station_ids
        pred_only_df = df[df['grid_id'].isin(pred_only)].sort_values('predicted_demand', ascending=False).head(10)
        for _, row in pred_only_df.iterrows():
            print(f"  Grid {int(row['grid_id']):4d}: "
                  f"Predicted {row['predicted_demand']:5.1f}, "
                  f"Actual {row['actual_demand']:5.1f}, "
                  f"{int(row['station_count'])} stations "
                  f"at ({row['center_lat']:.4f}, {row['center_lon']:.4f})")
        
        # Top 100 comparison analysis
        print("\n" + "="*80)
        print("Top 100 Comparison Analysis")
        print("="*80)
        
        top100_pred_ids = set(top100_predicted['grid_id'])
        top100_station_ids = set(top100_stations['grid_id'])
        overlap100 = top100_pred_ids & top100_station_ids
        
        print(f"\nOverlap: {len(overlap100)}/100 grids appear in both Top 100 lists")
        print(f"Overlap rate: {len(overlap100)/100*100:.1f}%")
        
        print(f"\nTop 100 Predicted only (not in Top 100 stations):")
        pred_only100 = top100_pred_ids - top100_station_ids
        print(f"  Count: {len(pred_only100)}")
        pred_only100_df = df[df['grid_id'].isin(pred_only100)]
        print(f"  Average predicted: {pred_only100_df['predicted_demand'].mean():.1f}")
        print(f"  Have any stations: {(pred_only100_df['station_count'] > 0).sum()}/{len(pred_only100)}")
        
        print(f"\nTop 100 Stations only (not in Top 100 predicted):")
        station_only100 = top100_station_ids - top100_pred_ids
        print(f"  Count: {len(station_only100)}")
        station_only100_df = df[df['grid_id'].isin(station_only100)]
        print(f"  Average stations: {station_only100_df['station_count'].mean():.1f}")
        print(f"  Average predicted: {station_only100_df['predicted_demand'].mean():.1f}")
        print(f"  Average actual: {station_only100_df['actual_demand'].mean():.1f}")
        print(f"  Model underestimation: {station_only100_df['actual_demand'].mean() / station_only100_df['predicted_demand'].mean():.2f}x")
        
        return m


def main():
    mapper = GridDemandMapper()
    mapper.create_map()
    print("\nVisualization complete")


if __name__ == "__main__":
    main()

