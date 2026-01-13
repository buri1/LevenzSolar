import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = "user_files/TEST_DATA_PV_ohne.xlsx"
OUTPUT_DIR = Path("data")

def main():
    print(f"📖 Reading {INPUT_FILE}...")
    df = pd.read_excel(INPUT_FILE)
    print(f"✅ Loaded {len(df)} rows. Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # 1. Ensure product_id exists
    if "product_id" not in df.columns:
        if "service_id" in df.columns:
            print("   ℹ️ Mapping 'service_id' to 'product_id'...")
            df["product_id"] = df["service_id"]
        else:
            print("   ⚠️ No ID column found. Generating 'product_id' from index...")
            df["product_id"] = df.index.astype(str)
            df["product_id"] = "GEN_" + df["product_id"]

    # 2. Fix quantity column if missing or NaN
    # The pipeline expects "quantity". If "position_item_quantity" exists, use it.
    if "quantity" not in df.columns and "position_item_quantity" in df.columns:
         print("   ℹ️ Mapping 'position_item_quantity' to 'quantity'...")
         df["quantity"] = df["position_item_quantity"]
    
    # 3. Create 'product_text' for LLM
    # Combine relevant columns to ensure LLM sees all info
    text_cols = [
        "supply_product_name", 
        "drafts_supply_product_name", 
        "supply_service_name", 
        "drafts_supply_service_name",
        "drafts_description",
        "supply_product_description"
    ]
    
    print("   ℹ️ Combining text columns for 'product_text'...")
    def combine_text(row):
        parts = []
        for col in text_cols:
            if col in row and pd.notnull(row[col]):
                parts.append(str(row[col]))
        return " | ".join(parts)

    df["product_text"] = df.apply(combine_text, axis=1)
    
    # 4. Save Full Dataset
    full_path = OUTPUT_DIR / "full_dataset.csv"
    df.to_csv(full_path, index=False)
    print(f"💾 Saved full dataset: {full_path}")

    # 4. Create 1k Subset (Stratified if possible, or Random)
    # We want to ensure we have some positives.
    positives = df[df["is_pv_module"] == 1]
    negatives = df[df["is_pv_module"] == 0]
    
    print(f"   Distribution: {len(positives)} Positives, {len(negatives)} Negatives")

    # Subset 1k
    sample_size = 1000
    if len(df) > sample_size:
        subset_1k = df.sample(n=sample_size, random_state=42)
        subset_1k.to_csv(OUTPUT_DIR / "subset_1k.csv", index=False)
        print(f"💾 Saved 1k subset: {OUTPUT_DIR / 'subset_1k.csv'} (Positives: {subset_1k['is_pv_module'].sum()})")
    
    # Subset 10k
    sample_size = 10000
    if len(df) > sample_size:
        subset_10k = df.sample(n=sample_size, random_state=42)
        subset_10k.to_csv(OUTPUT_DIR / "subset_10k.csv", index=False)
        print(f"💾 Saved 10k subset: {OUTPUT_DIR / 'subset_10k.csv'} (Positives: {subset_10k['is_pv_module'].sum()})")

    # 5. Create a specific validation set (First 100 rows strict) for debugging
    subset_val = df.head(100)
    subset_val.to_csv(OUTPUT_DIR / "subset_validation.csv", index=False)
    print(f"💾 Saved validation subset: {OUTPUT_DIR / 'subset_validation.csv'}")

if __name__ == "__main__":
    main()
