# -*- coding: utf-8 -*-
"""NB1: Embeddings & Vector Indexing"""
import sys
sys.path.insert(0, '.')

import notebooks._setup as _setup
from pathlib import Path
import json

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

DATA = Path(_setup.__file__).resolve().parent.parent / 'data'

print('=' * 60)
print('NB1: Embeddings & Vector Indexing')
print('=' * 60)

# 1. Load corpus
docs = []
with (DATA / 'corpus_vn.jsonl').open(encoding='utf-8') as f:
    for line in f:
        docs.append(json.loads(line))

print(f'\n[1] Corpus size: {len(docs)} docs')
print('First doc:')
print(json.dumps(docs[0], ensure_ascii=False, indent=2))

# 2. Embedding model
print('\n[2] Loading embedding model...')
embedder = TextEmbedding(model_name='BAAI/bge-small-en-v1.5')
sample = list(embedder.embed(['cloud computing tiếng Việt']))[0]
print(f'Vector dim: {len(sample)}')
print(f'First 8 values: {sample[:8].tolist()}')

# 3. Qdrant in-memory
print('\n[3] Creating Qdrant in-memory collection...')
client = QdrantClient(':memory:')
client.create_collection(
    collection_name='lab19',
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
print('Collection created: lab19')

# 4. Embed + upsert
print('\n[4] Embedding and indexing corpus...')
BATCH = 64
points = []
for start in range(0, len(docs), BATCH):
    batch = docs[start:start + BATCH]
    texts = [d['title'] + ' ' + d['text'] for d in batch]
    vectors = list(embedder.embed(texts))
    for i, (d, v) in enumerate(zip(batch, vectors)):
        points.append(PointStruct(
            id=start + i,
            vector=v.tolist(),
            payload={'doc_id': d['doc_id'], 'topic': d['topic'], 'title': d['title']},
        ))

client.upsert(collection_name='lab19', points=points)
n_indexed = client.count(collection_name='lab19').count
print(f'Indexed: {n_indexed} vectors')
assert n_indexed == 1000, f'expected 1000 indexed, got {n_indexed}'
print('ASSERTION PASSED: 1000 vectors indexed')

# 5. First search
print('\n[5] First similarity search:')
query = 'cloud computing và tự động mở rộng'
q_vec = next(embedder.embed([query])).tolist()
hits = client.query_points(collection_name='lab19', query=q_vec, limit=5).points

print(f'Query: {query!r}')
print('Top-5:')
for i, h in enumerate(hits, 1):
    topic = h.payload['topic']
    title = h.payload['title']
    print(f'  {i}. [{topic:>9}] score={h.score:.3f}  {title}')

# 6. Paraphrase search
print('\n[6] Paraphrase query search:')
query2 = 'phương pháp tự động mở rộng hạ tầng theo lưu lượng người dùng'
q_vec2 = next(embedder.embed([query2])).tolist()
hits2 = client.query_points(collection_name='lab19', query=q_vec2, limit=5).points

print(f'Query (paraphrase): {query2!r}')
for h in hits2:
    topic = h.payload['topic']
    title = h.payload['title']
    print(f'  [{topic:>9}] score={h.score:.3f}  {title}')

print('\n' + '=' * 60)
print('NB1 COMPLETED SUCCESSFULLY')
print('=' * 60)
