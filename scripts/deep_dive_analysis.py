import pandas as pd

CSV_FILE = "data/subset_1k.csv"
IDS_TO_CHECK = [
    "FhSIEwIpgAA", # Hybrid-Wechselrichter
    "FFxnocp1EAA", # Installation
    "FZEfJQGEMAA", # Lieferung
    "GDHUaHgoAAA", # Teil einer pv anlage
    "GJly4dpyEAA", # Produkt aus PV branche
    "F_yaj-86QAA", # Netzersatzanlage
    "FtXWlnywEAA"  # Könnte PV sein
]

def main():
    print("🔍 Deep Dive: Analyzing Ambiguous Products...")
    try:
        df = pd.read_csv(CSV_FILE)
        df['product_id'] = df['product_id'].astype(str)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Filter
    subset = df[df['product_id'].isin(IDS_TO_CHECK)]
    
    print(f"Found {len(subset)} / {len(IDS_TO_CHECK)} products.")
    print("-" * 80)
    
    for _, row in subset.iterrows():
        print(f"ID: {row['product_id']}")
        print(f"Analysis: {row.get('product_text', 'No text found')}")
        print("-" * 80)

if __name__ == "__main__":
    main()
