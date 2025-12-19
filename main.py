import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from src.processor import CSVProcessor
from src.llm_client import LLMClient
import pandas as pd

# Load env vars
load_dotenv()

def main():
    print("🚀 Starte Solar-Modul Klassifizierung")
    
    # User provided files
    input_file = "data/Testdaten ohne Loesung - mit head spalte.csv"
    output_file = "data/output.csv"
    test_file = "data/Testdaten Mit Loesung CSV.csv" # Ground Truth
    
    # Check API Key
    if not os.getenv("OPENAI_API_KEY"):
         print("❌ OPENAI_API_KEY not found in .env. Please set it.")
         return

    # 1. Initialize Processor
    processor = CSVProcessor(input_file, output_file)
    try:
        df_input = processor.load_csv()
        print(f"✅ {len(df_input)} Produkte geladen aus {input_file}")
    except Exception as e:
        print(f"❌ Fehler beim Laden der CSV: {e}")
        return

    # 2. Initialize Client
    try:
        client = LLMClient()
    except Exception as e:
        print(f"❌ Fehler beim Initialisieren des LLM Clients: {e}")
        return

    # 3. Process Batches
    all_results = []
    # Use copy for processing so we don't mess up original df yet
    batches = list(processor.create_batches(df_input))
    total_batches = len(batches)
    
    if total_batches == 0:
        print("⚠️ Keine Batches zu verarbeiten.")
        return

    print(f"📦 Starte Verarbeitung von {total_batches} Batches...")

    for i, batch in enumerate(batches, 1):
        print(f"   Batch {i}/{total_batches}: {len(batch)} Produkte...")
        try:
            results = client.classify_batch(batch)
            # Add to list
            for res in results:
                all_results.append(res.model_dump(by_alias=True))
            print(f"   ✓ Batch {i} fertig")
        except Exception as e:
            print(f"   ❌ Fehler in Batch {i}: {e}")
            
    # 4. Merge & Save Output
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Ensure ID types match for merging
        df_input['product_id'] = df_input['product_id'].astype(str)
        results_df['product_id'] = results_df['product_id'].astype(str)
        
        # Merge results into original DF
        # We want to fill the columns is_pv_module, reasoning, confidence in the original input
        # If they exist and are empty, we update them. Or just merge on product_id.
        # Since input file has empty cols, let's drop them from input before merge to avoid _x _y suffixes
        cols_to_drop = ['is_pv_module', 'Reasoning', 'Confidence']
        df_input_clean = df_input.drop(columns=[c for c in cols_to_drop if c in df_input.columns])
        
        # Select only relevant columns from results to merge back
        # We need product_id to join, and the new fields
        # CRITICAL FIX: Deduplicate results by product_id to avoid N*N merge explosion
        # We take the first classification for each ID (assuming consistency)
        results_subset = results_df[['product_id', 'is_pv_module', 'Reasoning', 'Confidence']].drop_duplicates(subset=['product_id'])
        
        final_df = pd.merge(df_input_clean, results_subset, on='product_id', how='left')
        
        # Convert boolean to whatever user prefers? User file had '1' for True. 
        # But for 'output.csv' let's keep it clean or match 1/0?
        # Let's write True/False for clarity, unless user strictly asked for 1/0.
        # "die letzten 3 spalten ... müssen vom LLM ausgefüllt werden" -> implied matching format.
        # Let's map True -> 1, False -> 0 to match the 'Testdaten Mit Loesung' format strictly.
        final_df['is_pv_module'] = final_df['is_pv_module'].apply(lambda x: 1 if x is True else 0 if x is False else None)

        final_df.to_csv(output_file, index=False, sep=';') # Use ; to match input style
        print(f"💾 Ergebnisse gespeichert: {output_file}")
    else:
        print("⚠️ Keine Ergebnisse zum Speichern.")

    # 5. Evaluation
    test_path = Path(test_file)
    if test_path.exists() and all_results:
        print("\n📊 Starte Evaluation gegen Testdaten...")
        try:
            # We need to load test file similarly (skip Metadata line)
            with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
            skip_rows = 1 if "Tabelle" in first_line else 0
            
            test_df = pd.read_csv(test_path, skiprows=skip_rows, sep=None, engine='python')
            
            # Apply same ID logic to Test Data
            if "service_id" in test_df.columns and "product_id" in test_df.columns:
                 test_df["product_id"] = test_df["product_id"].fillna(test_df["service_id"])
                 test_df["product_id"] = test_df["product_id"].astype(str).str.replace(r'\.0$', '', regex=True)
            
            # Ensure proper string types for merging
            test_df['product_id'] = test_df['product_id'].astype(str)
            final_df['product_id'] = final_df['product_id'].astype(str)
            
            # In Test file, 'is_pv_module' is the ground truth.
            # Let's rename it to 'ground_truth' to avoid confusion during merge
            test_subset = test_df[['product_id', 'is_pv_module']].rename(columns={'is_pv_module': 'ground_truth'})
            
            merged_eval = pd.merge(final_df, test_subset, on='product_id', how='inner')
            
            if merged_eval.empty:
                print("⚠️ Keine übereinstimmenden IDs zwischen Output und Testdaten gefunden.")
            else:
                # Calculate accuracy
                # Pred is 'is_pv_module' (1/0), Truth is 'ground_truth' (1/0)
                
                # Drop rows where prediction might be NaN (failed batch)
                merged_eval = merged_eval.dropna(subset=['is_pv_module', 'ground_truth'])
                
                correct = (merged_eval['is_pv_module'] == merged_eval['ground_truth']).sum()
                total = len(merged_eval)
                
                if total > 0:
                    accuracy = (correct / total) * 100
                    print(f"   Korrekt: {correct}/{total}")
                    print(f"   Accuracy: {accuracy:.2f}%")
                    
                    # Show errors
                    errors = merged_eval[merged_eval['is_pv_module'] != merged_eval['ground_truth']]
                    if not errors.empty:
                        print(f"⚠️  {len(errors)} Fehler:")
                        for _, row in errors.iterrows():
                            print(f"  - ID {row['product_id']}: Pred={row['is_pv_module']}, True={row['ground_truth']}")
                            print(f"    Reasoning: {row['Reasoning']}")
                else:
                    print("⚠️ Keine gültigen Einträge für die Evaluation.")
                
        except Exception as e:
            print(f"❌ Fehler bei der Evaluation: {e}")
    
    print("\n✅ Fertig!")

if __name__ == "__main__":
    main()
