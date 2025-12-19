import pandas as pd
from typing import Generator, List, Dict, Any
from pathlib import Path

class CSVProcessor:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.required_columns = ["product_id", "product_name"]

    def load_csv(self) -> pd.DataFrame:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        
        # Try reading with default (comma)
        try:
            # Check for "Tabelle 1" or similar metadata lines
            with open(self.input_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            
            skip_rows = 0
            if "Tabelle" in first_line:
                skip_rows = 1
            
            # Detect separator by reading a few lines
            df = pd.read_csv(self.input_path, skiprows=skip_rows, sep=None, engine='python')
            
        except Exception as e:
             raise ValueError(f"Could not read CSV file: {e}")
        
        # Normalize columns from n8n style if present (Non-destructive)
        if "supply_product_name" in df.columns and "product_name" not in df.columns:
            df["product_name"] = df["supply_product_name"]
            
        # Ensure we have a valid ID for every row
        # Some rows (services) have empty product_id but have service_id
        if "service_id" in df.columns and "product_id" in df.columns:
            # Fill NaN product_id with service_id
            df["product_id"] = df["product_id"].fillna(df["service_id"])
            # Ensure IDs are strings and remove decimal .0 if present (common in pandas float reading)
            df["product_id"] = df["product_id"].astype(str).str.replace(r'\.0$', '', regex=True)
            
        return df

    def create_batches(self, df: pd.DataFrame, batch_size: int = 10) -> Generator[List[Dict[str, Any]], None, None]:
        records = df.to_dict('records')
        for i in range(0, len(records), batch_size):
            yield records[i:i + batch_size]

    def save_results(self, results: List[Dict[str, Any]]):
        df = pd.DataFrame(results)
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
