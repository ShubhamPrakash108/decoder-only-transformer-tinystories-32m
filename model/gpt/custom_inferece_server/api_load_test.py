import time
import requests

URL = "http://localhost:8000/v1/completions"
MODEL = "gpt-custom"
MAX_TOKENS = 20
NUM_REQUESTS = 4

# A shared prompt will trigger cache hits
SHARED_PROMPT = "The history of artificial intelligence began in the 1950s when researchers started to build intelligent machines. " * 20

# Unique prompts will trigger cache misses (cold starts)
UNIQUE_PROMPTS = [
    "Once upon a time, there was a little dog who loved to play fetch in the beautiful park. " * 20,
    "In the deep ocean, scientists recently discovered a brand new species of glowing jellyfish swimming around. " * 20,
    "Alan Turing published his famous paper on computing machinery during the early days of modern computer science. " * 20,
    "The recipe for a perfect chocolate cake requires exactly two cups of flour and fresh cocoa powder. " * 20
]

def send_request(prompt):
    start = time.perf_counter()
    try:
        resp = requests.post(URL, json={"model": MODEL, "prompt": prompt, "max_tokens": MAX_TOKENS})
        latency = time.perf_counter() - start
        
        if resp.status_code == 200:
            return latency, True
        else:
            print(f"      [Error {resp.status_code}]: {resp.text}")
            return latency, False
    except requests.exceptions.ConnectionError:
        print("      [Error] Could not connect. Is the server running on localhost:8000?")
        return 0, False

print("="*60)
print("  API Load Test: KV Caching Speedup")
print("="*60)
print("Note: Since batching is not built yet (Stage 4), requests are sent sequentially.\n")

# TEST 1: Cold Starts (No Cache / Cache Misses)
print(f"[1/2] Sending {NUM_REQUESTS} requests with UNIQUE prompts (Normal / Cache Miss)...")
cold_latencies = []

for i, prompt in enumerate(UNIQUE_PROMPTS):
    latency, success = send_request(prompt)
    if success:
        cold_latencies.append(latency)
        print(f"      Request {i+1} took {latency:.3f}s")

avg_cold = sum(cold_latencies) / len(cold_latencies) if cold_latencies else 0

# TEST 2: Warm Starts (Shared Prefix Cache Hits)
print(f"\n[2/2] Sending {NUM_REQUESTS} requests with IDENTICAL prompts (KV Cache Hit)...")

# 1. Send an initial request just to prefill the cache for this prompt
print("      (Pre-filling cache for shared prompt...)")
send_request(SHARED_PROMPT)

warm_latencies = []
for i in range(NUM_REQUESTS):
    latency, success = send_request(SHARED_PROMPT)
    if success:
        warm_latencies.append(latency)
        print(f"      Request {i+1} took {latency:.3f}s")

avg_warm = sum(warm_latencies) / len(warm_latencies) if warm_latencies else 0

# RESULTS
print("\n" + "="*60)
print("  RESULTS")
print("="*60)

if avg_warm > 0 and avg_cold > 0:
    speedup = avg_cold / avg_warm
    print(f"  Avg latency NORMAL (Cold Start):   {avg_cold:.3f}s")
    print(f"  Avg latency KV CACHE (Warm Start): {avg_warm:.3f}s")
    print(f"\n  API Speedup: {speedup:.2f}x faster using KV Cache!")
else:
    print("  Test failed to complete. Check server logs.")
print("="*60)
