"""
03_graph_build.py
4종 엣지 헤테로 그래프 구축
입력: df_sampled.parquet, node_features.pt, labels.pt, timestamps.pt, masks
출력: data/graphs/hetero_graph.pt (HeteroData)
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch_geometric.data import HeteroData
from scipy.spatial import cKDTree
from collections import defaultdict

BASE  = Path(__file__).resolve().parent.parent
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
GRAPH.mkdir(parents=True, exist_ok=True)

print("="*60)
print("[1] 데이터 로드")
print("="*60)
df = pd.read_parquet(PROC / "df_sampled.parquet")
node_features = torch.load(PROC / "node_features.pt", weights_only=True)
labels        = torch.load(PROC / "labels.pt",        weights_only=True)
train_mask    = torch.load(PROC / "train_mask.pt",    weights_only=True)
test_mask     = torch.load(PROC / "test_mask.pt",     weights_only=True)
timestamps    = torch.load(PROC / "timestamps.pt",    weights_only=True)

N = len(df)
print(f"노드 수: {N:,}")
print(f"피처 dim: {node_features.shape[1]}")
assert len(df) == node_features.shape[0], "노드 수 불일치"

# ── 엣지 1: R-T-R (동일 prod_id + 동일 연-월) ──────────────────────────────
print("\n" + "="*60)
print("[2] 엣지 구축")
print("="*60)
print("\n[R-T-R] 동일 prod_id + 동일 연-월")

df['year_month'] = df['date'].dt.to_period('M').astype(str)
rtr_src, rtr_dst = [], []
for (prod, ym), group in df.groupby(['prod_id', 'year_month']):
    nodes = group['node_id'].tolist()
    if len(nodes) < 2:
        continue
    # 양방향 엣지 — 그룹 내 모든 쌍 (상한: 그룹당 최대 500쌍)
    if len(nodes) > 32:  # 32개 초과 시 샘플링하여 엣지 폭발 방지
        np.random.seed(42)
        nodes_sampled = np.random.choice(nodes, 32, replace=False).tolist()
    else:
        nodes_sampled = nodes
    for i in range(len(nodes_sampled)):
        for j in range(len(nodes_sampled)):
            if i != j:
                rtr_src.append(nodes_sampled[i])
                rtr_dst.append(nodes_sampled[j])

rtr_edge_index = torch.tensor([rtr_src, rtr_dst], dtype=torch.long)
print(f"  R-T-R 엣지 수: {rtr_edge_index.shape[1]:,}")

# ── 엣지 2: R-S-R (동일 prod_id + 동일 rating) ───────────────────────────────
print("\n[R-S-R] 동일 prod_id + 동일 rating")
df['rating_int'] = df['rating'].fillna(3.0).astype(int)
rsr_src, rsr_dst = [], []
for (prod, rat), group in df.groupby(['prod_id', 'rating_int']):
    nodes = group['node_id'].tolist()
    if len(nodes) < 2:
        continue
    if len(nodes) > 32:
        np.random.seed(42)
        nodes_sampled = np.random.choice(nodes, 32, replace=False).tolist()
    else:
        nodes_sampled = nodes
    for i in range(len(nodes_sampled)):
        for j in range(len(nodes_sampled)):
            if i != j:
                rsr_src.append(nodes_sampled[i])
                rsr_dst.append(nodes_sampled[j])

rsr_edge_index = torch.tensor([rsr_src, rsr_dst], dtype=torch.long)
print(f"  R-S-R 엣지 수: {rsr_edge_index.shape[1]:,}")

# ── 엣지 3: R-Burst-R (동일 prod_id + 72h 이내, Δt 엣지 피처 포함) ───────────
print("\n[R-Burst-R] 동일 prod_id + 72h 이내 (cKDTree, Δt 피처)")
BURST_WINDOW_SEC = 72 * 3600  # 72시간 (초 단위)
MAX_EDGES_PER_PROD = 2000      # 식당당 최대 엣지 수 (O(n²) 방지)

burst_src, burst_dst, burst_delta_t = [], [], []
for prod, group in df.groupby('prod_id'):
    nodes = group['node_id'].values
    ts_vals = group['timestamp'].values.astype(np.float64).reshape(-1, 1)
    if len(nodes) < 2:
        continue
    tree = cKDTree(ts_vals)
    pairs = list(tree.query_pairs(r=BURST_WINDOW_SEC))
    if len(pairs) > MAX_EDGES_PER_PROD:
        # 시간 간격 작은 순으로 우선 — 가장 긴밀한 burst 선택
        pairs = sorted(pairs, key=lambda p: abs(ts_vals[p[0], 0] - ts_vals[p[1], 0]))
        pairs = pairs[:MAX_EDGES_PER_PROD]
    for (i, j) in pairs:
        ni, nj = int(nodes[i]), int(nodes[j])
        dt = float(abs(ts_vals[i, 0] - ts_vals[j, 0])) / 3600.0  # 시간 단위
        # 양방향
        burst_src.extend([ni, nj])
        burst_dst.extend([nj, ni])
        burst_delta_t.extend([dt, dt])

burst_edge_index = torch.tensor([burst_src, burst_dst], dtype=torch.long)
burst_edge_attr  = torch.tensor(burst_delta_t, dtype=torch.float32).unsqueeze(1)  # [E, 1]
print(f"  R-Burst-R 엣지 수: {burst_edge_index.shape[1]:,}")
print(f"  Δt 범위: {burst_edge_attr.min().item():.1f}h ~ {burst_edge_attr.max().item():.1f}h")

# ── 엣지 4: R-U-R (동일 user_id) ──────────────────────────────────────────────
print("\n[R-U-R] 동일 user_id")
MAX_EDGES_PER_USER = 20  # 헤비 유저 엣지 폭발 방지
rur_src, rur_dst = [], []
for user, group in df.groupby('user_id'):
    nodes = group['node_id'].tolist()
    if len(nodes) < 2:
        continue
    if len(nodes) > 10:
        # 시간순으로 연속된 쌍만 (슬라이딩 윈도우 w=3)
        sorted_nodes = group.sort_values('date')['node_id'].tolist()
        for k in range(len(sorted_nodes)):
            for l in range(k+1, min(k+4, len(sorted_nodes))):
                rur_src.extend([sorted_nodes[k], sorted_nodes[l]])
                rur_dst.extend([sorted_nodes[l], sorted_nodes[k]])
                if len(rur_src) > MAX_EDGES_PER_USER * 2:
                    break
            else:
                continue
            break
    else:
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j:
                    rur_src.append(nodes[i])
                    rur_dst.append(nodes[j])

rur_edge_index = torch.tensor([rur_src, rur_dst], dtype=torch.long)
print(f"  R-U-R 엣지 수: {rur_edge_index.shape[1]:,}")

# ── HeteroData 구축 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[3] HeteroData 구축")
print("="*60)

data = HeteroData()

# 노드
data['review'].x         = node_features    # [N, feat_dim]
data['review'].y         = labels            # [N]
data['review'].timestamp = timestamps        # [N]
data['review'].train_mask = train_mask       # [N]
data['review'].test_mask  = test_mask        # [N]
data['review'].node_id    = torch.arange(N)  # 원래 인덱스

# 엣지
data['review', 'rtr',   'review'].edge_index = rtr_edge_index
data['review', 'rsr',   'review'].edge_index = rsr_edge_index
data['review', 'burst', 'review'].edge_index = burst_edge_index
data['review', 'burst', 'review'].edge_attr  = burst_edge_attr   # Δt [E, 1]
data['review', 'rur',   'review'].edge_index = rur_edge_index

print(f"\n노드 수  : {data['review'].x.shape[0]:,}")
print(f"피처 dim : {data['review'].x.shape[1]}")
print(f"R-T-R   : {data['review', 'rtr',   'review'].edge_index.shape[1]:,}")
print(f"R-S-R   : {data['review', 'rsr',   'review'].edge_index.shape[1]:,}")
print(f"R-Burst-R: {data['review', 'burst', 'review'].edge_index.shape[1]:,}")
print(f"R-U-R   : {data['review', 'rur',   'review'].edge_index.shape[1]:,}")
total_edges = sum([
    data['review', 'rtr',   'review'].edge_index.shape[1],
    data['review', 'rsr',   'review'].edge_index.shape[1],
    data['review', 'burst', 'review'].edge_index.shape[1],
    data['review', 'rur',   'review'].edge_index.shape[1],
])
print(f"총 엣지  : {total_edges:,}")

torch.save(data, GRAPH / "hetero_graph.pt")
print(f"\n저장: {GRAPH / 'hetero_graph.pt'}")

# 엣지 통계 저장
import json
stats = {
    "nodes": N,
    "feat_dim": int(node_features.shape[1]),
    "edges_rtr": int(rtr_edge_index.shape[1]),
    "edges_rsr": int(rsr_edge_index.shape[1]),
    "edges_burst": int(burst_edge_index.shape[1]),
    "edges_rur": int(rur_edge_index.shape[1]),
    "total_edges": total_edges,
    "train_nodes": int(train_mask.sum().item()),
    "test_nodes":  int(test_mask.sum().item()),
    "spam_ratio": round(float(labels.float().mean().item()), 4),
}
with open(RES / "graph_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"저장: {RES / 'graph_stats.json'}")

print("\n" + "="*60)
print("✅ 그래프 구축 완료")
print("   → 다음 단계: 04_train_baseline.py (GCN/GAT/BWGNN)")
print("="*60)
