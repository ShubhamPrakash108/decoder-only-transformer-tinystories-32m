import requests
import threading
import time

URL = "http://localhost:8000/v1/completions"
PROMPTS = [
    "Once upon a time",
    "The scientist discovered",
    "In the deep ocean",
    "The little puppy ran",
]

results = {}

def send(i, prompt):
    start_time = time.time()
    resp = requests.post(URL, json={"model": "gpt-custom", "prompt": prompt, "max_tokens": 128})
    latency = time.time() - start_time
    results[i] = (prompt, resp.json()["choices"][0]["text"], latency)

print("Firing 4 requests simultaneously...\n")

threads = [threading.Thread(target=send, args=(i, p)) for i, p in enumerate(PROMPTS)]
for t in threads: t.start()
for t in threads: t.join()

print("RESULTS (Notice how the latency stacks up!):")
for i, (prompt, text, latency) in results.items():
    print(f"[{i+1}] Latency: {latency:.2f}s | Prompt: {prompt}")
    print(f"    Response: {text}\n")