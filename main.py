import os
import sys
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
from src.processor import CSVProcessor
from src.llm_client import LLMClient
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Load env vars
load_dotenv()


def process_single_batch(client: LLMClient, batch: list, batch_num: int, total: int, lock: Lock) -> tuple:
    """Process a single batch and return results with batch number"""
    try:
        results = client.classify_batch(batch)
        with lock:
            print(f"   ✓ Batch {batch_num}/{total} fertig ({len(results)} Ergebnisse)")
        return batch_num, results, None
    except Exception as e:
        with lock:
            print(f"   ❌ Fehler in Batch {batch_num}: {e}")
        return batch_num, [], str(e)


def main():
    parser = argparse.ArgumentParser(description='LLM-basierte PV-Modul Klassifizierung')
    parser.add_argument('--batch-size', type=int, default=10, help='Anzahl Produkte pro Batch (default: 10)')
    parser.add_argument('--parallel', type=int, default=1, help='Anzahl paralleler Workers (default: 1, max empfohlen: 5)')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'zhipuai'], help='LLM Provider (openai oder zhipuai)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model Name (z.B. gpt-4o-mini oder glm-4-plus)')
    parser.add_argument('--input', type=str, default='data/Testdaten ohne Loesung - mit head spalte.csv', help='Input CSV Datei')
    parser.add_argument('--output', type=str, default='data/output.csv', help='Output CSV Datei')
    parser.add_argument('--limit', type=int, default=None, help='Max. Anzahl Zeilen zum Verarbeiten (für Tests)')
    parser.add_argument('--test-file', type=str, default=None, help='Ground Truth CSV Datei (Standard: Input Datei wenn is_pv_module vorhanden)')
    args = parser.parse_args()

    print("🚀 Starte Solar-Modul Klassifizierung mit Leistungsextraktion")
    print(f"   Model: {args.model}")
    print(f"   Batch-Size: {args.batch_size}")
    print(f"   Parallel Workers: {args.parallel}")
    
    # User provided files
    input_file = args.input
    output_file = args.output
    # User provided files
    input_file = args.input
    output_file = args.output
    
    # Determined test file (Ground Truth)
    # If a specific test file is not provided, we will check if the input file has ground truth later
    test_file = args.test_file if args.test_file else None
    
    # Check API Key
    if args.provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in .env. Please set it.")
        return
    elif args.provider == 'zhipuai' and not (os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")):
        print("❌ ZHIPUAI_API_KEY (or ZAI_API_KEY) not found in .env. Please set it.")
        return

    # 1. Initialize Processor
    processor = CSVProcessor(input_file, output_file)
    try:
        df_input = processor.load_csv()
        if args.limit:
            df_input = df_input.head(args.limit)
            print(f"⚠️  Limitiert auf {args.limit} Zeilen (Test-Modus)")
        print(f"✅ {len(df_input)} Produkte geladen aus {input_file}")
    except Exception as e:
        print(f"❌ Fehler beim Laden der CSV: {e}")
        return

    # 2. Initialize Client
    try:
        client = LLMClient(provider=args.provider, model=args.model)
    except Exception as e:
        print(f"❌ Fehler beim Initialisieren des LLM Clients: {e}")
        return

    # 3. Process Batches
    all_results = []
    batches = list(processor.create_batches(df_input, batch_size=args.batch_size))
    total_batches = len(batches)
    
    if total_batches == 0:
        print("⚠️ Keine Batches zu verarbeiten.")
        return

    print(f"📦 Starte Verarbeitung von {total_batches} Batches...")
    start_time = time.time()

    if args.parallel > 1:
        # Parallel processing
        print(f"   ⚡ Parallele Verarbeitung mit {args.parallel} Workers")
        lock = Lock()
        results_dict = {}
        
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(process_single_batch, client, batch, i+1, total_batches, lock): i
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(futures):
                batch_num, results, error = future.result()
                if results:
                    results_dict[batch_num] = results
        
        # Combine results in order
        for batch_num in sorted(results_dict.keys()):
            for res in results_dict[batch_num]:
                all_results.append(res.model_dump(by_alias=True))
    else:
        # Sequential processing
        for i, batch in enumerate(batches, 1):
            print(f"   Batch {i}/{total_batches}: {len(batch)} Produkte...")
            try:
                results = client.classify_batch(batch)
                for res in results:
                    all_results.append(res.model_dump(by_alias=True))
                print(f"   ✓ Batch {i} fertig")
            except Exception as e:
                print(f"   ❌ Fehler in Batch {i}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Verarbeitung abgeschlossen in {elapsed_time:.1f}s")
            
    # 4. Merge & Save Output
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Ensure ID types match for merging
        df_input['product_id'] = df_input['product_id'].astype(str)
        results_df['product_id'] = results_df['product_id'].astype(str)
        
        # Columns to drop from input before merge (to avoid _x _y suffixes)
        cols_to_drop = ['is_pv_module', 'Reasoning', 'Confidence', 'power_watts', 'quantity', 'total_power_watts', 'power_source']
        df_input_clean = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])
        
        # Deduplicate results by product_id
        result_cols = ['product_id', 'is_pv_module', 'Reasoning', 'Confidence', 'power_watts', 'quantity', 'total_power_watts', 'power_source']
        available_cols = [c for c in result_cols if c in results_df.columns]
        results_subset = results_df[available_cols].drop_duplicates(subset=['product_id'])
        
        final_df = pd.merge(df_input_clean, results_subset, on='product_id', how='left')
        
        # Convert boolean to 1/0 to match test data format
        final_df['is_pv_module'] = final_df['is_pv_module'].apply(lambda x: 1 if x is True else 0 if x is False else None)

        final_df.to_csv(output_file, index=False, sep=';')
        print(f"💾 Ergebnisse gespeichert: {output_file}")
        
        # Power extraction summary
        if 'power_watts' in final_df.columns and 'total_power_watts' in final_df.columns:
            pv_modules = final_df[final_df['is_pv_module'] == 1]
            total_power = pv_modules['total_power_watts'].sum()
            modules_with_power = pv_modules['power_watts'].notna().sum()
            
            print(f"\n⚡ LEISTUNGSEXTRAKTION:")
            print(f"   PV-Module klassifiziert:  {len(pv_modules)}")
            print(f"   Davon mit Leistung:       {modules_with_power}")
            print(f"   Gesamt-Leistung:          {total_power/1000:.2f} kWp")
    else:
        print("⚠️ Keine Ergebnisse zum Speichern.")

    # 5. Print API Usage Report
    print(client.get_usage_report())

    # 6. Evaluation
    test_path = Path(test_file) if test_file else Path(input_file)
    
    # If using input file, check if it has the label column
    if not test_file:
         # Check if input df has ground truth
         if 'is_pv_module' in df_input.columns:
             print("\n📊 Nutze Input-Datei als Ground Truth für Evaluation...")
         else:
             print("\n⚠️ Keine Test-Datei angegeben und Input hat keine 'is_pv_module' Spalte. Skipping Evaluation.")
             return

    if test_path.exists() and all_results:
        print("\n📊 Starte Evaluation gegen Testdaten...")
        try:
            # Load test file
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            skip_rows = 1 if "Tabelle" in first_line else 0
            
            test_df = pd.read_csv(test_path, skiprows=skip_rows, sep=None, engine='python')
            
            # Apply same ID logic to Test Data
            if "service_id" in test_df.columns and "product_id" in test_df.columns:
                test_df["product_id"] = test_df["product_id"].fillna(test_df["service_id"])
                test_df["product_id"] = test_df["product_id"].astype(str).str.replace(r'\.0$', '', regex=True)
            
            # Ensure proper string types
            test_df['product_id'] = test_df['product_id'].astype(str)
            final_df['product_id'] = final_df['product_id'].astype(str)
            
            # Rename truth column
            test_subset = test_df[['product_id', 'is_pv_module']].rename(columns={'is_pv_module': 'ground_truth'})
            
            merged_eval = pd.merge(final_df, test_subset, on='product_id', how='inner')
            
            if merged_eval.empty:
                print("⚠️ Keine übereinstimmenden IDs zwischen Output und Testdaten gefunden.")
            else:
                # Drop NaN
                merged_eval = merged_eval.dropna(subset=['is_pv_module', 'ground_truth'])
                
                # Convert to int for comparison
                merged_eval['pred'] = merged_eval['is_pv_module'].astype(int)
                merged_eval['truth'] = merged_eval['ground_truth'].astype(int)
                
                # Calculate metrics
                total = len(merged_eval)
                correct = (merged_eval['pred'] == merged_eval['truth']).sum()
                
                tp = ((merged_eval['pred'] == 1) & (merged_eval['truth'] == 1)).sum()
                fp = ((merged_eval['pred'] == 1) & (merged_eval['truth'] == 0)).sum()
                fn = ((merged_eval['pred'] == 0) & (merged_eval['truth'] == 1)).sum()
                tn = ((merged_eval['pred'] == 0) & (merged_eval['truth'] == 0)).sum()
                
                accuracy = correct / total * 100 if total > 0 else 0
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EVALUATION RESULTS                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total Samples:      {total:<39} ║
║  Correct:            {correct:<39} ║
╠══════════════════════════════════════════════════════════════╣
║  METRICS                                                     ║
║  ├─ Accuracy:        {accuracy:.2f}%{'':<34} ║
║  ├─ Precision:       {precision:.2f}%{'':<34} ║
║  ├─ Recall:          {recall:.2f}%{'':<34} ║
║  └─ F1 Score:        {f1:.2f}%{'':<34} ║
╠══════════════════════════════════════════════════════════════╣
║  CONFUSION MATRIX                                            ║
║  ├─ True Positives:  {tp:<39} ║
║  ├─ False Positives: {fp:<39} ║
║  ├─ False Negatives: {fn:<39} ║
║  └─ True Negatives:  {tn:<39} ║
╚══════════════════════════════════════════════════════════════╝""")
                
                # Show errors
                errors = merged_eval[merged_eval['pred'] != merged_eval['truth']]
                if not errors.empty:
                    print(f"\n⚠️  {len(errors)} Fehler:")
                    for _, row in errors.head(10).iterrows():  # Show max 10 errors
                        print(f"  - ID {row['product_id']}: Pred={row['pred']}, True={row['truth']}")
                        if 'Reasoning' in row:
                            print(f"    Reasoning: {row['Reasoning']}")
                        
        except Exception as e:
            print(f"❌ Fehler bei der Evaluation: {e}")
    
    print("\n✅ Fertig!")


if __name__ == "__main__":
    main()
