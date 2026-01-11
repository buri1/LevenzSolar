import pandas as pd
from pathlib import Path
import argparse


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics"""
    total = len(y_true)
    if total == 0:
        return {}
    
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "total_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate PV module classification results')
    parser.add_argument('--pred', type=str, default='data/output.csv', help='Predictions CSV file')
    parser.add_argument('--truth', type=str, default='data/Testdaten Mit Loesung CSV.csv', help='Ground truth CSV file')
    parser.add_argument('--save-errors', type=str, default='data/evaluation_errors.csv', help='Save errors to file')
    args = parser.parse_args()
    
    pred_file = args.pred
    truth_file = args.truth
    
    print(f"🔍 Comparing:\n  Pred:  {pred_file}\n  Truth: {truth_file}\n")
    
    # Load Predictions
    try:
        df_pred = pd.read_csv(pred_file, sep=';', dtype={'product_id': str})
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return

    # Load Ground Truth
    try:
        with open(truth_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
        skip = 1 if "Tabelle" in first_line else 0
        
        df_truth = pd.read_csv(truth_file, sep=None, engine='python', skiprows=skip, dtype={'product_id': str})
        
        # Apply ID Logic
        if "service_id" in df_truth.columns and "product_id" in df_truth.columns:
            df_truth["product_id"] = df_truth["product_id"].fillna(df_truth["service_id"])
            df_truth["product_id"] = df_truth["product_id"].astype(str).str.replace(r'\.0$', '', regex=True)

    except Exception as e:
        print(f"❌ Error loading truth: {e}")
        return

    # Prepare for merge
    df_truth_clean = df_truth[['product_id', 'is_pv_module']].rename(columns={'is_pv_module': 'ground_truth'})
    
    # Merge
    merged = pd.merge(df_pred, df_truth_clean, on='product_id', how='inner')
    
    if merged.empty:
        print("⚠️ No matching Product IDs found!")
        return
        
    # Ensure numeric comparison
    merged['pred_val'] = pd.to_numeric(merged['is_pv_module'], errors='coerce').fillna(0).astype(int)
    merged['truth_val'] = pd.to_numeric(merged['ground_truth'], errors='coerce').fillna(0).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(merged['truth_val'].tolist(), merged['pred_val'].tolist())
    
    # Print results
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EVALUATION RESULTS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total Evaluated:    {metrics['total_samples']:<39} ║
╠══════════════════════════════════════════════════════════════╣
║  CLASSIFICATION METRICS                                      ║
║  ├─ Accuracy:        {metrics['accuracy']*100:.2f}%{'':<34} ║
║  ├─ Precision:       {metrics['precision']*100:.2f}%{'':<34} ║
║  ├─ Recall:          {metrics['recall']*100:.2f}%{'':<34} ║
║  └─ F1 Score:        {metrics['f1']*100:.2f}%{'':<34} ║
╠══════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                            ║
║  ├─ True Positives:  {metrics['true_positives']:<39} ║
║  ├─ False Positives: {metrics['false_positives']:<39} ║
║  ├─ False Negatives: {metrics['false_negatives']:<39} ║
║  └─ True Negatives:  {metrics['true_negatives']:<39} ║
╠══════════════════════════════════════════════════════════════╣
║  CLASS DISTRIBUTION                                          ║
║  ├─ Predicted PV:    {merged['pred_val'].sum():<39} ║
║  └─ Actual PV:       {merged['truth_val'].sum():<39} ║
╚══════════════════════════════════════════════════════════════╝""")
    
    # Show errors
    diffs = merged[merged['pred_val'] != merged['truth_val']]
    
    if len(diffs) > 0:
        print(f"\n⚠️  Found {len(diffs)} Discrepancies:\n")
        for i, row in diffs.iterrows():
            print(f"🛑 ID: {row['product_id']}")
            print(f"   Name:     {row.get('product_name', row.get('supply_product_name', 'N/A'))[:60]}...")
            print(f"   Pred:     {'PV Module' if row['pred_val'] == 1 else 'Not PV Module'} ({row['pred_val']})")
            print(f"   Truth:    {'PV Module' if row['truth_val'] == 1 else 'Not PV Module'} ({row['truth_val']})")
            if 'Reasoning' in row:
                print(f"   Reasoning: {row['Reasoning']}")
            print("-" * 40)
            
        # Save diffs to file
        diffs.to_csv(args.save_errors, index=False, sep=';')
        print(f"\n💾 Errors saved to {args.save_errors}")
    else:
        print("\n✅ Perfect Match! No errors found.")
    
    # Power extraction analysis (if available)
    if 'power_watts' in merged.columns and 'total_power_watts' in merged.columns:
        pv_correct = merged[(merged['pred_val'] == 1) & (merged['truth_val'] == 1)]
        power_extracted = pv_correct['power_watts'].notna().sum()
        total_power = pv_correct['total_power_watts'].sum()
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  POWER EXTRACTION ANALYSIS                   ║
╠══════════════════════════════════════════════════════════════╣
║  Correctly classified PV modules:  {len(pv_correct):<24} ║
║  With power extracted:             {power_extracted:<24} ║
║  Power extraction rate:            {power_extracted/len(pv_correct)*100 if len(pv_correct) > 0 else 0:.1f}%{'':<22} ║
║  Total extracted power:            {total_power/1000:.2f} kWp{'':<19} ║
╚══════════════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    evaluate()
