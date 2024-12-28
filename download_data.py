import matplotlib.pyplot as plt
import requests
import datetime
import pandas as pd
from google.cloud import storage
from pathlib import Path
import os
from tqdm import tqdm
import dask.dataframe as dd
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import time


def explore_bucket_contents():
    bucket_name = "stl-willow-production-toclient-sample-us-jp-1w"
    key_file = "stl-willow-production-toclient-f2270639c0db.json"
    
    try:
        # Initialize storage client
        print(f"Initializing client with credentials from: {key_file}")
        storage_client = storage.Client.from_service_account_json(key_file)
        
        # Get specific bucket
        print(f"Accessing bucket: {bucket_name}")
        bucket = storage_client.bucket(bucket_name)

        # Summarize the files in the bucket, including how many files there are, how big the files are generally, how much data is in the bucket in total, and what the content types are.
        print(f"Summarizing bucket contents:")
        total_size = 0
        content_types = set()
        num_files = 0
        for blob in bucket.list_blobs():
            total_size += blob.size
            content_types.add(blob.content_type)
            num_files += 1
        print(f"Number of files: {num_files}")
        print(f"Total size: {total_size//1024} KB")
        print(f"Content types: {content_types}")

        # # List all files
        # print("\nListing bucket contents:")
        # blobs = bucket.list_blobs()
        # for blob in blobs:
        #     print(f"\nFile: {blob.name}")
        #     print(f"Size: {blob.size//1024} KB")
        #     print(f"Content type: {blob.content_type}")
                
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"Details: {str(e)}")
    
def download_parquet_file(bucket, blob_name: str, output_path: str, max_retries: int = 3) -> bool:
    """Download single parquet file with retries"""
    for attempt in range(max_retries):
        try:
            blob = bucket.blob(blob_name)
            blob.download_to_filename(output_path)
            return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download {blob_name} after {max_retries} attempts: {str(e)}")
                return False
            time.sleep(1 * (attempt + 1))  # Exponential backoff

def process_parquet_files(bucket_name: str, key_file: str, output_dir: str = "data"):
    """Process parquet files with retry logic"""
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        storage_client = storage.Client.from_service_account_json(key_file)
        bucket = storage_client.bucket(bucket_name)
        
        # Get list of parquet files
        parquet_files = [blob.name for blob in bucket.list_blobs() 
                        if blob.name.endswith('.parquet')]
        
        # Download files with progress bar
        successful_downloads = []
        for file in tqdm(parquet_files, desc="Downloading files"):
            output_path = os.path.join(output_dir, file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if download_parquet_file(bucket, file, output_path):
                successful_downloads.append(output_path)
        
        print(f"\nSuccessfully downloaded {len(successful_downloads)} of {len(parquet_files)} files")
        return successful_downloads
        
    except Exception as e:
        print(f"Error: {type(e).__name__}")
        print(f"Details: {str(e)}")
        return []


if __name__ == '__main__':
    bucket_name = "stl-willow-production-toclient-sample-us-jp-1w"
    key_file = "stl-willow-production-toclient-f2270639c0db.json"

    # explore_bucket_contents()
    process_parquet_files(bucket_name, key_file)
