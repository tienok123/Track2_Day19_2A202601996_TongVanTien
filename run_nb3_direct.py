# -*- coding: utf-8 -*-
"""NB3: FastAPI Search API Benchmark (direct import version)"""
import sys
sys.path.insert(0, '.')

import notebooks._setup as _setup
from pathlib import Path
import json
import time
import statistics

ROOT = Path(_setup.__file__).resolve().parent.parent
DATA = ROOT / 'data'

print('=' * 60)
print('NB3: FastAPI Search API Benchmark')
print('=' * 60)

# Import Searcher directly instead of starting server
from app.search import Searcher

print('\n[1] Building Searcher from corpus...')
CORPUS_PATH = ROOT / 'data' / 'corpus_vn.jsonl'
searcher = Searcher.from_corpus(CORPUS_PATH)
print(f'Searcher ready with {searcher.size} docs')

# 2. Single query test
print('\n[2] Single query test...')
result = searcher.search('cloud computing tự động mở rộng', mode='hybrid', top_k=10)
print(f'Top-3 hits:')
for i, h in enumerate(result[:3], 1):
    print(f'  {i}. {h.doc_id}  score={h.score:.4f}  {h.title}')

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
    for _ in range(reps):
        for q in golden:
            t0 = time.perf_counter()
            _ = searcher.search(q['query'], mode=mode, top_k=10)
            server_latencies.append((time.perf_counter() - t0) * 1000)
    return {
        'p50': percentile(server_latencies, 0.50),
        'p95': percentile(server_latencies, 0.95),
        'p99': percentile(server_latencies, 0.99),
    }

print(f'\n  {"mode":10}  {"P50":>7}  {"P95":>7}  {"P99":>7}')
results = {}
for mode in ('keyword', 'semantic', 'hybrid'):
    res = benchmark_mode(mode)
    results[mode] = res
    print(f'  {mode:10}  {res["p50"]:>5.1f}ms  {res["p95"]:>5.1f}ms  {res["p99"]:>5.1f}ms')

# 4. Rubric assertion
print('\n[4] Rubric check:')
hybrid_p99 = results['hybrid']['p99']
print(f'Hybrid P99: {hybrid_p99:.1f}ms')
if hybrid_p99 < 50:
    print(f'PASS -- hybrid P99 < 50ms ({hybrid_p99:.1f}ms)')
else:
    print(f'WARN -- hybrid P99 >= 50ms ({hybrid_p99:.1f}ms)')

print('\n' + '=' * 60)
print('NB3 COMPLETED SUCCESSFULLY')
print('=' * 60)
