"""
GPT-5-Mini vs GPT-4o-Mini Verification Benchmark
=================================================
Tests the hypothesis that GPT-5-mini can match GPT-4o-mini performance
with the new max_completion_tokens fix.

Run: python scripts/benchmark_gpt5mini_vs_4omini.py
"""
import subprocess
import time
import re
import os
import sys

TEST_FILE = "data/subset_1k.csv"
LIMIT = 100  # Small test to save costs
PARALLEL = 2

def run_test(model, batch_size, test_name):
    """Run a single test and return metrics"""
    print(f"\n{'='*60}")
    print(f"🧪 TEST: {test_name}")
    print(f"   Model: {model} | Batch: {batch_size} | Workers: {PARALLEL}")
    print(f"{'='*60}")
    
    output_file = f"data/verify_{model.replace('.', '_')}_b{batch_size}.csv"
    
    cmd = [
        sys.executable, "main.py",
        "--provider", "openai",
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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', env=env, timeout=600)
        duration = time.time() - start_time
        
        output = result.stdout
        stderr = result.stderr
        
        # Parse metrics
        m_prec = re.search(r"Precision:\s+([\d\.]+)", output)
        m_rec = re.search(r"Recall:\s+([\d\.]+)", output)
        m_cost = re.search(r"Total \(USD\):\s+\$([\d\.]+)", output)
        m_model = re.search(r"Model:\s+(\S+)", output)
        
        precision = float(m_prec.group(1)) if m_prec else 0
        recall = float(m_rec.group(1)) if m_rec else 0
        cost = float(m_cost.group(1)) if m_cost else 0
        logged_model = m_model.group(1) if m_model else "unknown"
        
        # Check for errors
        val_errors = output.count("Validation error")
        
        print(f"\n📊 RESULTS:")
        print(f"   ⏱️  Duration:     {duration:.1f}s")
        print(f"   🎯 Precision:    {precision}%")
        print(f"   📈 Recall:       {recall}%")
        print(f"   💰 Cost:         ${cost:.4f}")
        print(f"   📝 Logged Model: {logged_model}")
        print(f"   ⚠️  Val Errors:  {val_errors}")
        
        if result.returncode != 0:
            print(f"   ❌ Exit Code: {result.returncode}")
            if stderr:
                print(f"   STDERR: {stderr[:500]}")
        else:
            print(f"   ✅ SUCCESS")
        
        return {
            "test_name": test_name,
            "model": model,
            "logged_model": logged_model,
            "batch_size": batch_size,
            "duration_sec": round(duration, 1),
            "precision": precision,
            "recall": recall,
            "cost_usd": cost,
            "val_errors": val_errors,
            "success": result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        print(f"   ❌ TIMEOUT after 600s!")
        return {"test_name": test_name, "model": model, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return {"test_name": test_name, "model": model, "success": False, "error": str(e)}

def main():
    print("🚀 GPT-5-Mini vs GPT-4o-Mini Verification")
    print("=" * 60)
    print("Testing with the new max_completion_tokens fix")
    print("=" * 60)
    
    results = []
    
    # Test 1: GPT-4o-mini (Baseline)
    results.append(run_test("gpt-4o-mini", 50, "Baseline: GPT-4o-mini"))
    time.sleep(3)
    
    # Test 2: GPT-5-mini with same config
    results.append(run_test("gpt-5-mini", 50, "Fixed: GPT-5-mini"))
    time.sleep(3)
    
    # Test 3: GPT-5-mini Batch 100 (stress test)
    if results[-1].get("success"):
        results.append(run_test("gpt-5-mini", 100, "Stress: GPT-5-mini Batch 100"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    for r in results:
        if r.get("success"):
            print(f"✅ {r['test_name']}: Prec={r['precision']}% | Rec={r['recall']}% | {r['duration_sec']}s | ${r['cost_usd']:.4f}")
            print(f"   Logged Model: {r.get('logged_model', 'N/A')}")
        else:
            print(f"❌ {r['test_name']}: FAILED ({r.get('error', 'unknown')})")
    
    # Comparison
    baseline = next((r for r in results if "Baseline" in r.get("test_name", "")), None)
    fixed = next((r for r in results if "Fixed" in r.get("test_name", "")), None)
    
    if baseline and fixed and baseline.get("success") and fixed.get("success"):
        print("\n📈 COMPARISON:")
        prec_diff = fixed["precision"] - baseline["precision"]
        rec_diff = fixed["recall"] - baseline["recall"]
        time_ratio = fixed["duration_sec"] / baseline["duration_sec"] if baseline["duration_sec"] > 0 else 0
        
        print(f"   Precision Δ: {prec_diff:+.1f}%")
        print(f"   Recall Δ:    {rec_diff:+.1f}%")
        print(f"   Time Ratio:  {time_ratio:.2f}x")
        
        if fixed["precision"] >= 95 and fixed["recall"] >= 95:
            print("\n🎉 GPT-5-MINI FIX VERIFIED! Performance matches expectations.")
        else:
            print("\n⚠️  GPT-5-mini still underperforming. Investigate further.")

if __name__ == "__main__":
    main()
