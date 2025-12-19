import pytest
from src.processor import CSVProcessor
import pandas as pd

def test_load_csv_validation(tmp_path):
    # Create dummy csv
    d = tmp_path / "data"
    d.mkdir()
    p = d / "input.csv"
    # Missing product_name
    p.write_text("product_id\n1")
    
    processor = CSVProcessor(str(p), str(d / "output.csv"))
    
    with pytest.raises(ValueError, match="Missing required columns"):
        processor.load_csv()

def test_create_batches():
    # Create 25 items
    df = pd.DataFrame([{'product_id': i, 'product_name': f'P{i}'} for i in range(25)])
    processor = CSVProcessor("dummy", "dummy")
    
    batches = list(processor.create_batches(df, batch_size=10))
    
    assert len(batches) == 3
    assert len(batches[0]) == 10
    assert len(batches[1]) == 10
    assert len(batches[2]) == 5
    assert batches[0][0]['product_id'] == 0
