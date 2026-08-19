# -*- coding: utf-8 -*-
"""NB4: Feast Feature Store"""
import sys
sys.path.insert(0, '.')

import notebooks._setup as _setup
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone

ROOT = Path(_setup.__file__).resolve().parent.parent
FEAST_DIR = ROOT / 'app' / 'feast_repo'
FEAST_DATA = FEAST_DIR / 'data'
FEAST_DATA.mkdir(exist_ok=True)

print('=' * 60)
print('NB4: Feast Feature Store')
print('=' * 60)

import polars as pl

NOW = datetime.now(timezone.utc).replace(microsecond=0)

# 1. Generate offline data
print('\n[1] Generating offline data...')

def make_user_profile(n_users=100):
    return pl.DataFrame({
        'user_id': [f'u_{i:03d}' for i in range(n_users)],
        'reading_speed_wpm': [180 + (i * 7) % 200 for i in range(n_users)],
        'preferred_language': ['vi' if i % 3 != 0 else 'en' for i in range(n_users)],
        'topic_affinity': [
            ['ai_ml', 'cloud', 'security', 'database', 'devops'][i % 5]
            for i in range(n_users)
        ],
        'event_timestamp': [NOW - timedelta(hours=i % 48) for i in range(n_users)],
    })

def make_item_popularity(n_items=1000):
    return pl.DataFrame({
        'doc_id': [f'item_{i:04d}' for i in range(n_items)],
        'click_count_24h': [(i * 13) % 500 for i in range(n_items)],
        'ctr_7d': [round(((i * 7) % 100) / 100.0, 3) for i in range(n_items)],
        'avg_dwell_seconds': [10.0 + (i * 0.7) % 90 for i in range(n_items)],
        'event_timestamp': [NOW - timedelta(minutes=i % 720) for i in range(n_items)],
    })

def make_query_velocity(n_users=100):
    return pl.DataFrame({
        'user_id': [f'u_{i:03d}' for i in range(n_users)],
        'queries_last_hour': [(i * 11) % 50 for i in range(n_users)],
        'distinct_topics_24h': [1 + (i * 3) % 10 for i in range(n_users)],
        'event_timestamp': [NOW - timedelta(minutes=i % 30) for i in range(n_users)],
    })

make_user_profile().write_parquet(FEAST_DATA / 'user_profile.parquet')
make_item_popularity().write_parquet(FEAST_DATA / 'item_popularity.parquet')
make_query_velocity().write_parquet(FEAST_DATA / 'query_velocity.parquet')
print(f'Wrote 3 Parquet sources to {FEAST_DATA}')
for p in sorted(FEAST_DATA.glob('*.parquet')):
    print(f'  {p.name}  {p.stat().st_size/1024:.1f} KB')

# 2. feast apply
print('\n[2] Running feast apply...')
res = subprocess.run(
    ['.venv\\Scripts\\feast', 'apply'],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print('STDOUT:')
print(res.stdout[-2000:])
if res.stderr:
    print('STDERR:')
    print(res.stderr[-500:])
if res.returncode != 0:
    print(f'feast apply FAILED with code {res.returncode}')

# 3. feast materialize-incremental
print('\n[3] Running feast materialize-incremental...')
end_dt = NOW.strftime('%Y-%m-%dT%H:%M:%S')
res = subprocess.run(
    ['.venv\\Scripts\\feast', 'materialize-incremental', end_dt],
    cwd=str(FEAST_DIR),
    capture_output=True, text=True, check=False,
)
print(res.stdout[-1500:])
if res.stderr:
    print('STDERR (tail):')
    print(res.stderr[-500:])
if res.returncode != 0:
    print(f'materialize FAILED with code {res.returncode}')

# 4. Online lookup
print('\n[4] Online lookup...')
import time
from feast import FeatureStore

fs = FeatureStore(repo_path=str(FEAST_DIR))

REQUEST_FEATURES = [
    'user_profile_features:reading_speed_wpm',
    'user_profile_features:preferred_language',
    'user_profile_features:topic_affinity',
    'query_velocity_features:queries_last_hour',
    'query_velocity_features:distinct_topics_24h',
]

t0 = time.perf_counter()
features = fs.get_online_features(
    features=REQUEST_FEATURES,
    entity_rows=[{'user_id': 'u_001'}],
).to_dict()
single_latency_ms = (time.perf_counter() - t0) * 1000
print(f'Single lookup: {single_latency_ms:.2f}ms')
print({k: v[0] for k, v in features.items()})

# 5. Batch latency benchmark
print('\n[5] Batch latency benchmark (100 lookups)...')
latencies = []
for i in range(100):
    user_id = f'u_{i:03d}'
    t0 = time.perf_counter()
    fs.get_online_features(
        features=REQUEST_FEATURES,
        entity_rows=[{'user_id': user_id}],
    ).to_dict()
    latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
p50 = latencies[50]
p95 = latencies[95]
p99 = latencies[99]
print(f'Online lookup latency over 100 calls:')
print(f'  P50 = {p50:.2f}ms')
print(f'  P95 = {p95:.2f}ms')
print(f'  P99 = {p99:.2f}ms')

if p99 < 10:
    print(f'PASS -- online lookup P99 < 10ms ({p99:.2f}ms)')
else:
    print(f'WARN -- P99 = {p99:.2f}ms')

# 6. PIT join
print('\n[6] PIT join (offline)...')
import pandas as pd
entity_df = pd.DataFrame({
    'user_id': ['u_001', 'u_002', 'u_003'],
    'event_timestamp': [NOW - timedelta(hours=2), NOW - timedelta(hours=1), NOW],
})

historical = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        'user_profile_features:reading_speed_wpm',
        'user_profile_features:topic_affinity',
    ],
).to_df()
print(historical.to_string())

print('\n' + '=' * 60)
print('NB4 COMPLETED SUCCESSFULLY')
print('=' * 60)
