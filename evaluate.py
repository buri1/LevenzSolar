import pandas as pd
from pathlib import Path

def evaluate():
    pred_file = "data/output.csv"
    truth_file = "data/Testdaten Mit Loesung CSV.csv"
    
    print(f"🔍 Comparing:\n  Pred:  {pred_file}\n  Truth: {truth_file}\n")
    
    # Load Predictions (output.csv is saved with semicolon sep in main.py)
    try:
        df_pred = pd.read_csv(pred_file, sep=';', dtype={'product_id': str})
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return

    # Load Ground Truth
    # Need to handle potential "Tabelle 1" line again just in case, though pandas might handle it if we skip
    try:
        with open(truth_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        skip = 1 if "Tabelle" in first_line else 0
        
        # Detect sep (likely semicolon)
        df_truth = pd.read_csv(truth_file, sep=None, engine='python', skiprows=skip, dtype={'product_id': str})
        
        # Apply ID Logic to Truth
        if "service_id" in df_truth.columns and "product_id" in df_truth.columns:
             df_truth["product_id"] = df_truth["product_id"].fillna(df_truth["service_id"])
             # Be careful with mixed types if service_id was read as float
             df_truth["product_id"] = df_truth["product_id"].astype(str).str.replace(r'\.0$', '', regex=True)

    except Exception as e:
        print(f"❌ Error loading truth: {e}")
        return

    # Prepare for merge
    # Prediction column is 'is_pv_module' (1/0)
    # Truth column is 'is_pv_module' (often 1/0 or '1'/ '0')
    
    # Rename truth column to avoid collision
    df_truth_clean = df_truth[['product_id', 'is_pv_module']].rename(columns={'is_pv_module': 'ground_truth'})
    
    # Merge
    merged = pd.merge(df_pred, df_truth_clean, on='product_id', how='inner')
    
    if merged.empty:
        print("⚠️ No matching Product IDs found!")
        return
        
    # Analyze
    # Ensure numeric comparison
    merged['pred_val'] = pd.to_numeric(merged['is_pv_module'], errors='coerce').fillna(0).astype(int)
    merged['truth_val'] = pd.to_numeric(merged['ground_truth'], errors='coerce').fillna(0).astype(int)
    
    # Differences
    diffs = merged[merged['pred_val'] != merged['truth_val']]
    
    total = len(merged)
    errors = len(diffs)
    correct = total - errors
    accuracy = (correct / total) * 100
    
    print(f"📊 Results:")
    print(f"  Total Evaluated: {total}")
    print(f"  Correct:         {correct}")
    print(f"  Errors:          {errors}")
    print(f"  Accuracy:        {accuracy:.2f}%")
    
    print("-" * 60)
    print("📈 Class Distribution:")
    print(f"  Predicted 'Solar Module' (1): {merged['pred_val'].sum()}")
    print(f"  Actual 'Solar Module' (1):    {merged['truth_val'].sum()}")
    print("-" * 60)
    
    if errors > 0:
        print(f"\n⚠️  Found {errors} Discrepancies:\n")
        for i, row in diffs.iterrows():
            print(f"🛑 ID: {row['product_id']}")
            print(f"   Name:     {row.get('product_name', 'N/A')}")
            print(f"   Pred:     {'PV Module' if row['pred_val'] == 1 else 'Not PV Module'} ({row['pred_val']})")
            print(f"   Truth:    {'PV Module' if row['truth_val'] == 1 else 'Not PV Module'} ({row['truth_val']})")
            print(f"   Reasoning: {row.get('Reasoning', 'N/A')}")
            print("-" * 40)
            
        # Save diffs to file
        diff_file = "data/evaluation_errors.csv"
        diffs.to_csv(diff_file, index=False, sep=';')
        print(f"\n💾 Errors saved to {diff_file}")
    else:
        print("\n✅ Perfect Match! No errors found.")

if __name__ == "__main__":
    evaluate()
