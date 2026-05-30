"""
02_features.py
SBERT 임베딩 + 노드 피처 벡터 생성
입력: data/processed/df_sampled.parquet
출력: data/processed/sbert_embeddings.pt  [N, 384]
      data/processed/node_features.pt     [N, feat_dim]
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE  = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

print("="*60)
print("[1] 데이터 로드")
print("="*60)
df = pd.read_parquet(PROC / "df_sampled.parquet")
print(f"shape: {df.shape}")
print(f"컬럼: {df.columns.tolist()}")

# ── SBERT 임베딩 ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[2] SBERT 임베딩 (all-MiniLM-L6-v2, 384d)")
print("="*60)

model = SentenceTransformer('all-MiniLM-L6-v2')
texts = df['text'].fillna("").tolist()

print(f"임베딩 대상: {len(texts):,}건")
embeddings = model.encode(
    texts,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,  # L2 정규화 — 코사인 유사도 일관성
)
emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
torch.save(emb_tensor, PROC / "sbert_embeddings.pt")
print(f"저장: sbert_embeddings.pt shape={emb_tensor.shape}")

# ── 추가 피처 생성 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[3] 추가 노드 피처 생성")
print("="*60)

# rating 정규화 [1,5] → [0,1]
rating_feat = torch.tensor(
    ((df['rating'].fillna(3.0).values - 1.0) / 4.0),
    dtype=torch.float32
).unsqueeze(1)  # [N, 1]

# timestamp 정규화 (min-max → [0,1])
ts = df['timestamp'].values.astype(np.float64)
ts_min, ts_max = ts.min(), ts.max()
ts_norm = (ts - ts_min) / (ts_max - ts_min + 1e-8)
ts_feat = torch.tensor(ts_norm, dtype=torch.float32).unsqueeze(1)  # [N, 1]

# tag 컬럼 제외: tag='fake'/'real'이 label과 100% 동일 → 정답 누수
# 포함 시 PR-AUC=1.000이 되므로 반드시 제외
print("tag 컬럼 제외 (label과 완전 동일, 정답 누수)")

# 최종 노드 피처: SBERT(384) + rating(1) + timestamp(1) = 386차원
node_features = torch.cat([emb_tensor, rating_feat, ts_feat], dim=1)
print(f"\n노드 피처 최종 shape: {node_features.shape}")
print(f"  - SBERT     : 384")
print(f"  - rating    : 1")
print(f"  - timestamp : 1")
print(f"  - 합계       : {node_features.shape[1]}")

torch.save(node_features, PROC / "node_features.pt")
print(f"\n저장: node_features.pt shape={node_features.shape}")

# 라벨 & 마스크 저장
labels = torch.tensor(df['label'].values, dtype=torch.long)
train_mask = torch.tensor(df['split'].values == 'train', dtype=torch.bool)
test_mask  = torch.tensor(df['split'].values == 'test',  dtype=torch.bool)
timestamps = torch.tensor(df['timestamp'].values, dtype=torch.long)

torch.save(labels,     PROC / "labels.pt")
torch.save(train_mask, PROC / "train_mask.pt")
torch.save(test_mask,  PROC / "test_mask.pt")
torch.save(timestamps, PROC / "timestamps.pt")
print(f"저장: labels.pt, train_mask.pt, test_mask.pt, timestamps.pt")

print("\n" + "="*60)
print("✅ 피처 생성 완료")
print(f"   train: {train_mask.sum().item():,}  test: {test_mask.sum().item():,}")
print(f"   spam in train: {labels[train_mask].sum().item():,}  ({labels[train_mask].float().mean().item():.3f})")
print(f"   spam in test : {labels[test_mask].sum().item():,}  ({labels[test_mask].float().mean().item():.3f})")
print(f"   → 다음 단계: 03_graph_build.py (헤테로 그래프 구축)")
print("="*60)
