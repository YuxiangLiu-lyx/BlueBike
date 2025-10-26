import pandas as pd
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')


def get_project_root() -> Path:
    """Get the project root directory"""
    current = Path(__file__).resolve()
    return current.parent.parent.parent


class DataSorter:
    
    def __init__(self, raw_data_dir: str = None, sorted_data_dir: str = None):
        project_root = get_project_root()
        
        if raw_data_dir is None:
            raw_data_dir = project_root / "data" / "raw" / "bluebike_raw"
        if sorted_data_dir is None:
            sorted_data_dir = project_root / "data" / "processed" / "data_sorted"
            
        self.raw_data_dir = Path(raw_data_dir)
        self.sorted_data_dir = Path(sorted_data_dir)
        self.sorted_data_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_station_column(self, df: pd.DataFrame) -> str:
        """Detect the start station name column in different data formats"""
        possible_names = [
            'start_station_name',
            'start station name',
            'start Station name',
        ]
        
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        
        raise ValueError(f"Could not find start station column. Available columns: {df.columns.tolist()}")
    
    def sort_file(self, file_path: Path) -> None:
        """Sort a single CSV file by start station name"""
        print(f"Processing {file_path.name}...", end=" ", flush=True)
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            original_rows = len(df)
            
            # Detect the correct column name
            station_col = self.detect_station_column(df)
            
            # Sort by start station name
            df_sorted = df.sort_values(by=station_col, na_position='last')
            
            # Save to sorted directory with same filename
            output_path = self.sorted_data_dir / file_path.name
            df_sorted.to_csv(output_path, index=False)
            
            print(f"Done ({original_rows:,} rows sorted)")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    def sort_all_files(self) -> List[Path]:
        """Sort all CSV files in raw data directory"""
        print("Data Sorting by Start Station Name")
        print("=" * 60)
        
        csv_files = sorted(self.raw_data_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {self.raw_data_dir}")
        print(f"Output directory: {self.sorted_data_dir}")
        print("-" * 60)
        
        sorted_files = []
        
        for csv_file in csv_files:
            self.sort_file(csv_file)
            sorted_files.append(self.sorted_data_dir / csv_file.name)
        
        print("-" * 60)
        print(f"Sorting complete. {len(sorted_files)} files sorted and saved.")
        
        return sorted_files


def main():
    sorter = DataSorter()
    
    # Sort all files by start station name
    sorted_files = sorter.sort_all_files()
    
    print("\nSorted files saved in data/processed/data_sorted/")


if __name__ == "__main__":
    main()

