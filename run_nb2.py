# -*- coding: utf-8 -*-
"""NB2: Hybrid Search với RRF"""
import sys
sys.path.insert(0, '.')

import notebooks._setup as _setup
from pathlib import Path
import json
import statistics

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

DATA = Path(_setup.__file__).resolve().parent.parent / 'data'

print('=' * 60)
print('NB2: Hybrid Search với RRF')
print('=' * 60)

# 1. Load corpus + build both indices
print('\n[1] Loading corpus...')
docs = [json.loads(line) for line in (DATA / 'corpus_vn.jsonl').open(encoding='utf-8')]
print(f'Loaded {len(docs)} docs')

# BM25
print('Building BM25 index...')
tokenized = [(d['title'] + ' ' + d['text']).lower().split() for d in docs]
bm25 = BM25Okapi(tokenized)

# Vector
print('Building vector index...')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
client = QdrantClient(':memory:')
client.create_collection(
    collection_name='lab19',
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
BATCH = 64
points = []
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d['title'] + ' ' + d['text'] for d in batch]
    vectors = list(embedder.embed(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i, vector=v.tolist(),
            payload={'doc_id': d['doc_id'], 'topic': d['topic']},
        ))
client.upsert(collection_name='lab19', points=points)
print(f'BM25 + vector indices ready ({len(docs)} docs)')

# 2. Per-mode search functions
print('\n[2] Defining search functions...')
TOP_K = 10
RRF_K = 60

def search_keyword(query: str, top_k: int = TOP_K) -> list:
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [docs[i]['doc_id'] for i in ranked]

def search_semantic(query: str, top_k: int = TOP_K) -> list:
    q_vec = next(embedder.embed([query])).tolist()
    res = client.query_points(collection_name='lab19', query=q_vec, limit=top_k)
    return [p.payload['doc_id'] for p in res.points]

def search_hybrid(query: str, top_k: int = TOP_K, rrf_k: int = RRF_K) -> list:
    depth = max(top_k * 5, 50)
    kw_ids = search_keyword(query, depth)
    sem_ids = search_semantic(query, depth)

    # RRF fusion
    rrf = {}
    for rank, doc_id in enumerate(kw_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, doc_id in enumerate(sem_ids, start=1):
        rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    return [doc_id for doc_id, _ in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]]

# Quick sanity
test_q = 'co giãn linh hoạt theo nhu cầu sử dụng'
print(f'\nTest query: {test_q}')
print(f'  keyword top-3:  {search_keyword(test_q)[:3]}')
print(f'  semantic top-3: {search_semantic(test_q)[:3]}')
print(f'  hybrid top-3:   {search_hybrid(test_q)[:3]}')

# 3. Evaluate on golden set
print('\n[3] Evaluating on golden set (50 queries)...')
golden = [json.loads(line) for line in (DATA / 'golden_set.jsonl').open(encoding='utf-8')]
doc_topic = {d['doc_id']: d['topic'] for d in docs}

def precision_at_10(retrieved_ids: list, target_topic: str) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for d in retrieved_ids if doc_topic.get(d) == target_topic) / len(retrieved_ids)

p_kw, p_sem, p_hyb = [], [], []
for q in golden:
    p_kw.append(precision_at_10(search_keyword(q['query']), q['topic']))
    p_sem.append(precision_at_10(search_semantic(q['query']), q['topic']))
    p_hyb.append(precision_at_10(search_hybrid(q['query']), q['topic']))

print(f'\nPrecision@10 (avg over {len(golden)} queries):')
print(f'  Keyword (BM25)   : {statistics.mean(p_kw):.1%}')
print(f'  Semantic (vector): {statistics.mean(p_sem):.1%}')
print(f'  Hybrid  (RRF=60) : {statistics.mean(p_hyb):.1%}')

# 4. Slice by query type
print('\n[4] Precision@10 by query type:')
from collections import defaultdict

by_type = defaultdict(lambda: {'kw': [], 'sem': [], 'hyb': []})
for q, kw, sem, hyb in zip(golden, p_kw, p_sem, p_hyb):
    by_type[q['mode_hint']]['kw'].append(kw)
    by_type[q['mode_hint']]['sem'].append(sem)
    by_type[q['mode_hint']]['hyb'].append(hyb)

print(f"  {'type':12} {'n':>3}  {'kw':>7} {'sem':>7} {'hyb':>7}")
for t in ('exact', 'paraphrase', 'mixed'):
    m = by_type[t]
    n = len(m['kw'])
    kw_mean = statistics.mean(m['kw']) if m['kw'] else 0
    sem_mean = statistics.mean(m['sem']) if m['sem'] else 0
    hyb_mean = statistics.mean(m['hyb']) if m['hyb'] else 0
    print(f'  {t:12} {n:>3}  {kw_mean:6.1%} {sem_mean:6.1%} {hyb_mean:6.1%}')

print('\n' + '=' * 60)
print('NB2 COMPLETED SUCCESSFULLY')
print('=' * 60)
