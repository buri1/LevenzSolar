import pandas as pd

PRED_FILE = "data/output_eval_1k_v3.csv"
TRUTH_FILE = "data/subset_1k.csv"

def main():
    print("🔍 Analyzing False Positives...")
    
    try:
        pred_df = pd.read_csv(PRED_FILE, sep=';')
        truth_df = pd.read_csv(TRUTH_FILE)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Ensure IDs are strings
    pred_df['product_id'] = pred_df['product_id'].astype(str)
    truth_df['product_id'] = truth_df['product_id'].astype(str)
    
    # Rename ground truth column
    if 'is_pv_module' in truth_df.columns:
        truth_subset = truth_df.rename(columns={'is_pv_module': 'ground_truth'})
    
    # Merge
    merged = pd.merge(pred_df, truth_subset, on='product_id', how='inner')
    
    # FPs: Predicted=1 (True), Truth=0 (False)
    # Check column names after merge (likely is_pv_module_x and ground_truth)
    pred_col = 'is_pv_module_x' if 'is_pv_module_x' in merged.columns else 'is_pv_module'
    
    fps = merged[(merged[pred_col] == True) & (merged['ground_truth'] == 0)]
    
    print(f"\nFound {len(fps)} False Positives:")
    print("-" * 80)
    
    for i, row in fps.iterrows():
        print(f"ID: {row['product_id']}")
        print(f"Product Text: {row.get('product_text', 'N/A')}")
        print(f"LLM Reasoning: {row.get('Reasoning', 'N/A')}")
        print("-" * 80)

if __name__ == "__main__":
    main()
