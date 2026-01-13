import subprocess
import time
import pandas as pd
import re
import os
import sys

# Define test parameters
TEST_FILE = "data/subset_1k.csv"
LIMIT = 1000
PARALLEL = 20
PROVIDERS = [
    {"provider": "zhipuai", "model": "glm-4.5-flash"},
    {"provider": "zhipuai", "model": "glm-4-plus"}
]
BATCH_SIZES = [50, 100, 200, 500]

RESULTS = []

def run_test(provider, model, batch_size):
    print(f"\n🔥 STRESS TEST: {model} | Batch {batch_size} | 20 Workers")
    
    output_file = f"data/stress_{model}_b{batch_size}.csv"
    
    cmd = [
        sys.executable, "main.py",
        "--provider", provider,
        "--model", model,
        "--batch-size", str(batch_size),
        "--parallel", str(PARALLEL),
        "--input", TEST_FILE,
        "--output", output_file,
        "--limit", str(LIMIT)
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Failed (Return Code {result.returncode})")
            # print(result.stderr[:500]) # Print first 500 chars of error
            return None

        output = result.stdout
        
        # Regex extraction
        metrics = {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "cost": 0.0, 
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        }
        
        # Parse output for metrics
        m_prec = re.search(r"Precision:\s+([\d\.]+)", output)
        m_rec = re.search(r"Recall:\s+([\d\.]+)", output)
        m_cost = re.search(r"Total \(USD\):\s+\$([\d\.]+)", output)
        
        if m_prec: metrics["precision"] = float(m_prec.group(1))
        if m_rec: metrics["recall"] = float(m_rec.group(1))
        if m_cost: metrics["cost"] = float(m_cost.group(1))

        # Check for missing items / validation errors (often happens with huge batches)
        # We can detect this if Recall drops significantly or if there are "Validation error" logs
        val_errors = output.count("Validation error")
        json_errors = output.count("JSON Error")
        
        print(f"   ⏱️ Time: {duration:.2f}s | 💰 Cost: ${metrics['cost']:.4f}")
        print(f"   🎯 Prec: {metrics['precision']}% | Rec: {metrics['recall']}%")
        print(f"   ⚠️ Errors: {val_errors} Validation | {json_errors} JSON")
        
        return {
            "model": model,
            "batch_size": batch_size,
            "duration": round(duration, 2),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "cost": metrics["cost"],
            "validation_errors": val_errors,
            "json_errors": json_errors
        }

    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print(f"🚀 Starting Batch Size Stress Test")
    print(f"File: {TEST_FILE} ({LIMIT} rows)")
    
    for prov in PROVIDERS:
        for b_size in BATCH_SIZES:
            res = run_test(prov["provider"], prov["model"], b_size)
            if res:
                RESULTS.append(res)
            # Cool down to ensure log clean-up
            time.sleep(2)

    # Save Results
    df = pd.DataFrame(RESULTS)
    df.to_csv("data/stress_test_results.csv", index=False)
    
    # Print Summary
    print("\n\n🏆 STRESS TEST SUMMARY 🏆")
    print(f"{'Model':<15} | {'Batch':<5} | {'Prec':<6} | {'Rec':<6} | {'Cost':<6} | {'ValErr':<6}")
    print("-" * 60)
    for r in RESULTS:
        print(f"{r['model']:<15} | {r['batch_size']:<5} | {r['precision']:<6} | {r['recall']:<6} | {r['cost']:<6} | {r['validation_errors']:<6}")

if __name__ == "__main__":
    main()
