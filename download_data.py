#!/usr/bin/env python3
"""
Download preprocessed data from Hugging Face Hub
Run this script before training models to get the required data files.
"""
from pathlib import Path
import sys

def download_file(repo_id: str, filename: str, local_path: Path):
    """Download a single file from Hugging Face Hub"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download
    
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {filename}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir="."
    )
    print(f"  → Saved to {local_path}")

def main():
    # Hugging Face dataset repository
    REPO_ID = "matrix1900/bluebike-data"
    
    print("=" * 60)
    print("Downloading BlueBike preprocessed data from Hugging Face")
    print("=" * 60)
    
    files = [
        # Optional: Uncomment to download trip-level data for EDA visualizations (1GB)
        # ("data/processed/bluebike_cleaned/trips_cleaned.parquet", 
        #  Path("data/processed/bluebike_cleaned/trips_cleaned.parquet")),
        
        ("data/processed/daily/daily_departures.parquet", 
         Path("data/processed/daily/daily_departures.parquet")),
        
        ("data/processed/daily/daily_with_xgb_features.parquet", 
         Path("data/processed/daily/daily_with_xgb_features.parquet")),
        
        ("data/processed/daily/daily_with_poi_features.parquet", 
         Path("data/processed/daily/daily_with_poi_features.parquet")),
        
        ("data/processed/grid/grid_with_poi_features.parquet", 
         Path("data/processed/grid/grid_with_poi_features.parquet")),
    ]
    
    for filename, local_path in files:
        try:
            download_file(REPO_ID, filename, local_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("All data files downloaded successfully!")
    print("=" * 60)
    print("\nYou can now run the models:")
    print("  python src/models/baseline.py")
    print("  python src/models/xgboost_model.py")
    print("  python src/models/xgboost_no_popularity.py")
    print("  python src/models/xgboost_poi_only.py")

if __name__ == "__main__":
    main()

