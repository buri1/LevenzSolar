import subprocess
import time
import pandas as pd
import re
import os

TEST_FILE = "data/subset_1k.csv"
LIMIT = 1000 # Run on full 1k subset
PARALLEL = 20 # User requested high parallelism for ZAI

# Configurations to test
CONFIGS = [
    # Baseline
    {"provider": "openai", "model": "gpt-4o-mini", "batch_size": 10},
    {"provider": "openai", "model": "gpt-4o-mini", "batch_size": 50},
    
    # ZhipuAI variants
    {"provider": "zhipuai", "model": "glm-4-plus", "batch_size": 10},
    {"provider": "zhipuai", "model": "glm-4-plus", "batch_size": 20},
    {"provider": "zhipuai", "model": "glm-4.5-preview", "batch_size": 10}, # Assuming preview name
    {"provider": "zhipuai", "model": "glm-4.5-air", "batch_size": 10},
    {"provider": "zhipuai", "model": "glm-4.5-air", "batch_size": 50}, # Test larger batch with cheaper model
    {"provider": "zhipuai", "model": "glm-4.5-flash", "batch_size": 50}, # Test speed/free model
]

RESULTS = []

def run_test(config):
    print(f"\n🧪 Testing: {config['provider']} / {config['model']} / Batch {config['batch_size']}")
    
    output_file = f"data/bench_{config['provider']}_{config['model']}_b{config['batch_size']}.csv"
    
    cmd = [
        ".\\venv\\Scripts\\python", "main.py",
        "--provider", config['provider'],
        "--model", config['model'],
        "--batch-size", str(config['batch_size']),
        "--parallel", str(PARALLEL),
        "--input", TEST_FILE,
        "--output", output_file,
        "--limit", str(LIMIT)
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    start_time = time.time()
    try:
        # Run process and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Failed: {result.stderr}")
            return None
        
        # Parse stdout for metrics
        output = result.stdout
        
        # Extract metrics using regex from the printed table/summary
        # Looking for "Precision: ... 100.00%"
        precision_match = re.search(r"Precision:\s+([\d\.]+)", output)
        recall_match = re.search(r"Recall:\s+([\d\.]+)", output)
        f1_match = re.search(r"F1 Score:\s+([\d\.]+)", output)
        cost_match = re.search(r"Total \(USD\):\s+\$([\d\.]+)", output)
        
        prec = float(precision_match.group(1)) if precision_match else 0
        rec = float(recall_match.group(1)) if recall_match else 0
        f1 = float(f1_match.group(1)) if f1_match else 0
        cost = float(cost_match.group(1)) if cost_match else 0
        
        return {
            "model": config['model'],
            "batch_size": config['batch_size'],
            "duration_sec": round(duration, 2),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "cost_usd": cost,
            "items_per_min": round((LIMIT / duration) * 60, 0)
        }
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print(f"🚀 Starting Benchmark Suite on {TEST_FILE} ({LIMIT} rows)")
    print(f"Parallel Workers: {PARALLEL}")
    
    for config in CONFIGS:
        res = run_test(config)
        if res:
            RESULTS.append(res)
            print(f"   ✅ Result: {res}")
        else:
            print("   ⚠️ No result captured.")
        
        # Cooldown to avoid instant rate limit hits between runs
        time.sleep(5)

    # Print Summary Table
    print("\n\n🏆 BENCHMARK RESULTS 🏆")
    print(f"{'Model':<20} | {'Batch':<5} | {'Time(s)':<8} | {'Speed(IPM)':<10} | {'Prec%':<6} | {'Rec%':<6} | {'Cost($)':<8}")
    print("-" * 80)
    for r in RESULTS:
        print(f"{r['model']:<20} | {r['batch_size']:<5} | {r['duration_sec']:<8} | {r['items_per_min']:<10} | {r['precision']:<6} | {r['recall']:<6} | {r['cost_usd']:<8}")

    # Save to CSV
    df = pd.DataFrame(RESULTS)
    df.to_csv("data/benchmark_results.csv", index=False)
    print("\n📄 Results saved to data/benchmark_results.csv")

if __name__ == "__main__":
    main()
