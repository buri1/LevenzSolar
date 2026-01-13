import pandas as pd

PRED_FILE = "data/output_eval_1k_v3.csv"
TRUTH_FILE = "data/subset_1k.csv"

def main():
    print("📊 Calculating Metrics (Pandas)...")
    
    try:
        pred_df = pd.read_csv(PRED_FILE, sep=';')
        truth_df = pd.read_csv(TRUTH_FILE)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    pred_df['product_id'] = pred_df['product_id'].astype(str)
    truth_df['product_id'] = truth_df['product_id'].astype(str)
    
    truth_subset = truth_df[['product_id', 'is_pv_module']].rename(columns={'is_pv_module': 'ground_truth'})
    merged = pd.merge(pred_df, truth_subset, on='product_id', how='inner')
    merged = merged.dropna(subset=['is_pv_module', 'ground_truth'])
    
    y_pred = merged['is_pv_module'].astype(int)
    y_true = merged['ground_truth'].astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    total = len(merged)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"""
    RESULTS ({total} samples):
    --------------------------------
    Accuracy:  {accuracy*100:.2f}%
    Precision: {precision*100:.2f}%
    Recall:    {recall*100:.2f}%
    F1-Score:  {f1*100:.2f}%
    
    Confusion Matrix:
    TP: {tp}, FP: {fp}
    TN: {tn}, FN: {fn}
    """)

if __name__ == "__main__":
    main()
