import os
import requests
import zipfile
from pathlib import Path
from typing import List, Optional
import time


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    # Navigate up from src/data_collection/bluebikes_download.py to project root
    return current.parent.parent.parent


class BlueBikesDownloader:
    
    BASE_URL = "https://s3.amazonaws.com/hubway-data"
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Use absolute path from project root
            data_dir = get_project_root() / "data" / "raw"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_file_urls(self, start_year: int, start_month: int, 
                          end_year: int, end_month: int) -> List[tuple]:
        """Generate download URLs for all months in the specified range"""
        urls = []
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            year_month = f"{current_year}{current_month:02d}"
            
            # BlueBikes has used different naming conventions over time
            possible_filenames = [
                f"{year_month}-bluebikes-tripdata.zip",
                f"{year_month}-hubway-tripdata.zip",
                f"{year_month}_hubway_tripdata.zip",
            ]
            
            for filename in possible_filenames:
                url = f"{self.BASE_URL}/{filename}"
                urls.append((current_year, current_month, url, filename))
            
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
                
        return urls
    
    def download_file(self, url: str, filename: str, max_retries: int = 3) -> Optional[Path]:
        """Download a single file with progress indication"""
        output_path = self.data_dir / filename
        
        if output_path.exists():
            print(f"Already downloaded: {filename}")
            return output_path
        
        for attempt in range(max_retries):
            try:
                print(f"Downloading {filename}...", end=" ", flush=True)
                response = requests.get(url, stream=True, timeout=30)
                
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(output_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                    
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"Done ({size_mb:.1f} MB)")
                    return output_path
                    
                elif response.status_code == 404:
                    print(f"Not found")
                    return None
                else:
                    print(f"Failed (status {response.status_code})")
                    
            except Exception as e:
                print(f"Error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def extract_zip(self, zip_path: Path, remove_after: bool = True) -> List[Path]:
        """Extract CSV files from zip archive"""
        extracted_files = []
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                for csv_file in csv_files:
                    extract_path = self.data_dir / csv_file
                    
                    if not extract_path.exists():
                        zip_ref.extract(csv_file, self.data_dir)
                        print(f"  Extracted: {csv_file}")
                    else:
                        print(f"  Already exists: {csv_file}")
                    
                    extracted_files.append(extract_path)
            
            if remove_after and extracted_files:
                os.remove(zip_path)
                print(f"  Removed: {zip_path.name}")
                        
        except zipfile.BadZipFile:
            print(f"  Invalid zip file: {zip_path.name}")
            
        return extracted_files
    
    def download_all(self, start_year: int = 2015, start_month: int = 1,
                     end_year: int = 2025, end_month: int = 9,
                     extract: bool = True, remove_zip: bool = True) -> List[Path]:
        """
        Download all BlueBikes trip data for the specified date range
        
        Default range covers the entire available history (2015-2025/09)
        """
        print(f"BlueBikes Data Downloader")
        print(f"Date range: {start_year}/{start_month:02d} to {end_year}/{end_month:02d}")
        print(f"Output directory: {self.data_dir.absolute()}")
        print("-" * 60)
        
        urls = self.generate_file_urls(start_year, start_month, end_year, end_month)
        downloaded_files = []
        attempted_months = set()
        
        for year, month, url, filename in urls:
            month_key = (year, month)
            
            if month_key in attempted_months:
                continue
            
            downloaded_path = self.download_file(url, filename)
            
            if downloaded_path:
                attempted_months.add(month_key)
                
                if extract and filename.endswith('.zip'):
                    extracted = self.extract_zip(downloaded_path, remove_after=remove_zip)
                    downloaded_files.extend(extracted)
                else:
                    downloaded_files.append(downloaded_path)
        
        print("-" * 60)
        print(f"Download complete. Total files: {len(downloaded_files)}")
        return downloaded_files


def main():
    downloader = BlueBikesDownloader()
    
    # Download all available BlueBikes data (2015-2025/09)
    files = downloader.download_all(
        start_year=2015,
        start_month=1,
        end_year=2025,
        end_month=9,
        extract=True,
        remove_zip=True
    )
    
    if files:
        print("\nDownloaded CSV files:")
        for file_path in sorted(files):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file_path.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

