import folium
from folium import plugins
import webbrowser
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from geopandas import GeoDataFrame
from rtree import index


def read_parquet_file(file_path: str) -> pd.DataFrame:
    """Read parquet file using dask"""
    df = dd.read_parquet(file_path)
    return df.compute()

def remove_fuzzy_points(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fuzzy points from the dataframe"""
    return df[df['fuzzed_point'] == False]

def organize_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Group the dataframe by the journey_id column"""
    sorted_df = df.sort_values(by=["journey_id", "capture_time"])
    return sorted_df.groupby("journey_id")

def read_crash_csv(file_path: str) -> pd.DataFrame:
    """Read crash csv file"""
    return pd.read_csv(file_path)

def remove_invalid_locations(df: pd.DataFrame) -> pd.DataFrame:
    """Remove invalid locations from the dataframe"""
    return df[
        df['LATDECDG'].notna() & 
        df['LONDECDG'].notna()
    ]

def plot_trips(trips: pd.DataFrame):
    """Plot the trips"""
    for name, group in trips:
        plt.plot(group["longitude"], group["latitude"])
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trips")
    plt.show()

def plot_crashes(crash_df: pd.DataFrame):
    """Plot the crashes"""
    plt.scatter(crash_df["LONDECDG"], crash_df["LATDECDG"], s=1, c="red")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Crash Data")
    plt.show()

def plot_trips_and_crashes(trips: pd.DataFrame, crash_df: pd.DataFrame):
    """Plot the trips and crashes"""
    for name, group in trips:
        plt.plot(group["longitude"], group["latitude"])
    plt.scatter(crash_df["LONDECDG"], crash_df["LATDECDG"], s=1, c="red")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Trips and Crashes")
    plt.show()

def plot_trips_and_crashes_on_map(trips: pd.DataFrame, crash_df: pd.DataFrame):
    """Create interactive map with trips and crashes"""
    # Create base map centered on data
    center_lat = crash_df['LATDECDG'].mean()
    center_lon = crash_df['LONDECDG'].mean()
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=12,
                  tiles='OpenStreetMap')
    
    # Add crash points
    crashes = folium.FeatureGroup(name='Crashes')
    for _, crash in crash_df.iterrows():
        folium.CircleMarker(
            location=[crash['LATDECDG'], crash['LONDECDG']],
            radius=3,
            color='red',
            fill=True,
            popup=f"Crash ID: {crash['DOCTNMBR']}",
        ).add_to(crashes)
    crashes.add_to(m)
    
    # Add trips as lines
    trip_lines = folium.FeatureGroup(name='Trips')
    for name, group in trips:
        points = [[row['latitude'], row['longitude']] 
                 for _, row in group.iterrows()]
        folium.PolyLine(
            points,
            weight=2,
            color='blue',
            opacity=0.8
        ).add_to(trip_lines)
    trip_lines.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save and open in browser
    output_file = 'map.html'
    m.save(output_file)
    webbrowser.open('file://' + str(Path.cwd() / output_file))

def plot_location_data(df: pd.DataFrame):
    # data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-6a5f581a-69fa-4ee0-ba1b-c1cfa650228b-c000.snappy.parquet"
    data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-69f6702f-a092-42c8-8b86-0858c1d89a8c-c000.snappy.parquet"

    # Load the data
    data_dir = Path("data")
    # data_file = data_dir / "sample.parquet"
    df = pd.read_parquet(data_file)
    # df = df.compute()

    fuzzy = df[df['fuzzed_point'] == True]
    unfuzzy = df[df['fuzzed_point'] == False]
    
    # Plot the data
    plt.scatter(fuzzy["longitude"], fuzzy["latitude"], s=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Location Data")
    plt.show()

    # Plot the data
    plt.scatter(unfuzzy["longitude"], unfuzzy["latitude"], s=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Location Data")
    plt.show()

    # group by the journey_id column to get trips, then show a histogram for the number of points in each trip, in both fuzzy and unfuzzy sets
    fuzzy_trips = fuzzy.groupby("journey_id").size()
    unfuzzy_trips = unfuzzy.groupby("journey_id").size()

    plt.hist(fuzzy_trips, bins=range(1, 100, 1), alpha=0.5, label="Fuzzy")
    plt.hist(unfuzzy_trips, bins=range(1, 100, 1), alpha=0.5, label="Unfuzzy")
    plt.xlabel("Number of points in trip")
    plt.ylabel("Number of trips")
    plt.title("Trip Lengths")
    plt.legend()
    plt.show()

    # create another historgram for the length of trips, but only count unique coordinates. Repeated coordinates should not be counted.
    fuzzy_unique_trips = fuzzy.groupby("journey_id")[["latitude", "longitude"]].nunique()
    unfuzzy_unique_trips = unfuzzy.groupby("journey_id")[["latitude", "longitude"]].nunique()

    plt.hist(fuzzy_unique_trips["latitude"], bins=range(1, 100, 1), alpha=0.5, label="Fuzzy")
    plt.hist(unfuzzy_unique_trips["latitude"], bins=range(1, 100, 1), alpha=0.5, label="Unfuzzy")
    plt.xlabel("Number of unique points in trip")
    plt.ylabel("Number of trips")
    plt.title("Unique Trip Lengths")
    plt.legend()
    plt.show()

def find_overlapping_trips_and_crashes(trips: pd.DataFrame, crash_df: pd.DataFrame, buffer_meters: float = 50):
    """Find trips that overlap with crash locations within buffer distance"""
    
    def create_trip_lines():
        """Convert trips to LineString geometries"""
        lines = []
        trip_ids = []
        for name, group in trips:
            coords = list(zip(group['longitude'], group['latitude']))
            if len(coords) >= 2:  # Need at least 2 points for a line
                lines.append(LineString(coords))
                trip_ids.append(name)
        return GeoDataFrame({'geometry': lines, 'trip_id': trip_ids})

    def create_crash_points():
        """Convert crashes to Point geometries"""
        points = [Point(xy) for xy in zip(crash_df['LONDECDG'], crash_df['LATDECDG'])]
        return GeoDataFrame({'geometry': points}, index=crash_df.index)

    # Create spatial dataframes
    trip_gdf = create_trip_lines()
    crash_gdf = create_crash_points()

    # Create spatial index for trips
    spatial_index = index.Index()
    for idx, trip in trip_gdf.iterrows():
        spatial_index.insert(idx, trip.geometry.bounds)

    # Find overlaps
    overlaps = []
    for idx, crash in crash_gdf.iterrows():
        # Get candidate trips from spatial index
        crash_buffer = crash.geometry.buffer(buffer_meters / 111000)  # Convert meters to degrees
        candidates = list(spatial_index.intersection(crash_buffer.bounds))
        
        # Check actual distance
        for cand_idx in candidates:
            trip_line = trip_gdf.iloc[cand_idx].geometry
            if trip_line.distance(crash.geometry) < (buffer_meters / 111000):
                overlaps.append({
                    'crash_id': crash_df.iloc[idx]['DOCTNMBR'],
                    'trip_id': trip_gdf.iloc[cand_idx]['trip_id'],
                    'distance_meters': trip_line.distance(crash.geometry) * 111000
                })

    return pd.DataFrame(overlaps)

def visualize_overlaps(trips: pd.DataFrame, crash_df: pd.DataFrame, x: pd.DataFrame):
    """Create interactive map highlighting overlapping trips and crashes"""
    m = folium.Map(location=[crash_df['LATDECDG'].mean(), crash_df['LONDECDG'].mean()], 
                  zoom_start=12)
    
    # Add all trips in light gray
    for name, group in trips:
        print("Overlaps columns:", overlaps.columns)
        print("Overlaps head:", overlaps.head())
        color = 'red' if name in overlaps['trip_id'].values else 'gray'
        opacity = 0.8 if name in overlaps['trip_id'].values else 0.3
        folium.PolyLine(
            locations=list(zip(group['latitude'], group['longitude'])),
            color=color,
            weight=2,
            opacity=opacity
        ).add_to(m)
    
    # Add crashes with popups showing matching trip IDs
    for _, crash in crash_df.iterrows():
        matching_trips = overlaps[overlaps['crash_id'] == crash['DOCTNMBR']]['trip_id'].tolist()
        popup_text = f"Crash ID: {crash['DOCTNMBR']}<br>Matching trips: {matching_trips}"
        folium.CircleMarker(
            location=[crash['LATDECDG'], crash['LONDECDG']],
            radius=5,
            color='blue' if matching_trips else 'gray',
            popup=popup_text
        ).add_to(m)
    
    m.save('overlaps.html')

# Update main:
if __name__ == "__main__":
    # data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-6a5f581a-69fa-4ee0-ba1b-c1cfa650228b-c000.snappy.parquet"
    data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-69f6702f-a092-42c8-8b86-0858c1d89a8c-c000.snappy.parquet"
    df = read_parquet_file(data_file)
    df = remove_fuzzy_points(df)
    trips = organize_trips(df)
    plot_trips(trips)

    crash_file = "crash-data-download_big.csv"
    # crash_file = "crash-data-download_fatal.csv"
    crash_df = read_crash_csv(crash_file)
    crash_df = remove_invalid_locations(crash_df)
    plot_crashes(crash_df)
    
    # Add interactive map
    # plot_trips_and_crashes_on_map(trips, crash_df)

    overlaps = find_overlapping_trips_and_crashes(trips, crash_df)
    print(f"Found {len(overlaps)} overlapping trips and crashes")
    visualize_overlaps(trips, crash_df, overlaps)