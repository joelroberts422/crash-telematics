import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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

if __name__ == "__main__":
    # data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-6a5f581a-69fa-4ee0-ba1b-c1cfa650228b-c000.snappy.parquet"
    data_file = "data/journeysplus_batch_delivery/region=usa/year=2024/month=09/day=15/hour=00/part-00000-69f6702f-a092-42c8-8b86-0858c1d89a8c-c000.snappy.parquet"
    df = read_parquet_file(data_file)
    df = remove_fuzzy_points(df)
    trips = organize_trips(df)
    plot_trips(trips)

    crash_file = "crash-data-download.csv"
    crash_df = read_crash_csv(crash_file)
    plot_crashes(crash_df)

    plot_trips_and_crashes(trips, crash_df)