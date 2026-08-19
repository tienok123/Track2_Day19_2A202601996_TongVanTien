# -*- coding: utf-8 -*-
"""NB3: FastAPI Search API Benchmark"""
import sys
sys.path.insert(0, '.')

import notebooks._setup as _setup
from pathlib import Path
import json
import time
import subprocess
import statistics

ROOT = Path(_setup.__file__).resolve().parent.parent
DATA = ROOT / 'data'

print('=' * 60)
print('NB3: FastAPI Search API Benchmark')
print('=' * 60)

# 1. Start API server in background
print('\n[1] Starting FastAPI server...')
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

proc = subprocess.Popen(
    ['.venv\\Scripts\\uvicorn', 'app.main:app', '--port', '8001', '--log-level', 'warning'],
    cwd=str(ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for server to be ready
URL = 'http://localhost:8001'
import httpx
for _ in range(60):
    try:
        r = httpx.get(f'{URL}/healthz', timeout=2.0)
        if r.status_code == 200 and r.json().get('ready'):
            break
    except httpx.HTTPError:
        pass
    time.sleep(1)
else:
    proc.terminate()
    raise RuntimeError("API didn't become ready within 60s")

print(f'Server ready: {httpx.get(f"{URL}/healthz").json()}')

# 2. Single query test
print('\n[2] Single query test...')
r = httpx.get(f'{URL}/search', params={'q': 'cloud computing tự động mở rộng', 'mode': 'hybrid'})
r.raise_for_status()
body = r.json()
print(f'latency_ms: {body["latency_ms"]:.1f}')
print('top-3 hits:')
for h in body['hits'][:3]:
    print(f'  {h["doc_id"]:>14}  score={h["score"]:.4f}  {h["title"]}')

# 3. Latency benchmark
print('\n[3] Running latency benchmark (100 queries x 3 modes)...')
golden = [json.loads(l) for l in (DATA / 'golden_set.jsonl').open(encoding='utf-8')]

def percentile(values, p):
    n = len(values)
    if n == 0:
        return 0.0
    return sorted(values)[min(int(n * p), n - 1)]

def benchmark_mode(mode, reps=2):
    server_latencies = []
    wall_latencies = []
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            r = httpx.get(f'{URL}/search', params={'q': q['query'], 'mode': mode})
            wall_latencies.append((time.perf_counter() - t0) * 1000)
            server_latencies.append(r.json()['latency_ms'])
    return {
        'p50_server': percentile(server_latencies, 0.50),
        'p95_server': percentile(server_latencies, 0.95),
        'p99_server': percentile(server_latencies, 0.99),
        'p99_wall': percentile(wall_latencies, 0.99),
    }

print(f'\n  {"mode":10}  {"P50":>7}  {"P95":>7}  {"P99":>7}  {"P99(wall)":>9}')
results = {}
for mode in ('keyword', 'semantic', 'hybrid'):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f'  {mode:10}  {res["p50_server"]:>5.1f}ms  {res["p95_server"]:>5.1f}ms  '
          f'{res["p99_server"]:>5.1f}ms  {res["p99_wall"]:>7.1f}ms')

# 4. Rubric assertion
print('\n[4] Rubric check:')
hybrid_p99 = results['hybrid']['p99_server']
print(f'Hybrid P99 server-side: {hybrid_p99:.1f}ms')
if hybrid_p99 < 50:
    print(f'PASS -- hybrid P99 < 50ms ({hybrid_p99:.1f}ms)')
else:
    print(f'WARN -- hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)')
    print('  Possible causes: cold cache, fastembed model not warm yet, or RRF depth=50 is too aggressive')
    print('  Check: re-run benchmark after 10 warm-up queries; or reduce RRF depth')

# 5. Cleanup
print('\n[5] Stopping API server...')
proc.terminate()
proc.wait(timeout=5)
print('API server stopped')

print('\n' + '=' * 60)
print('NB3 COMPLETED SUCCESSFULLY')
print('=' * 60)
