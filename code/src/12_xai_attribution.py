"""
12_xai_attribution.py
XAI — 사기 판단 근거 설명 (Edge Attribution + Feature Importance)

목적: "왜 이 리뷰가 사기로 탐지됐는가"를 수치로 설명
방법:
  1. 엣지 타입 기여도: 각 엣지 타입을 하나씩 제거했을 때 fraud_prob 변화 측정
  2. Burst Δt 분포: 사기 노드 주변 burst 엣지의 Δt가 정상보다 짧은지 확인
  3. 특징 기여도: 사기 노드의 SBERT 임베딩 이상도 (정상 평균과의 거리)

출력: results/xai_edge_attribution.csv
      results/xai_case_studies.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import copy
from pathlib import Path
from sklearn.metrics import average_precision_score
from torch_geometric.nn import HeteroConv, SAGEConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"
PROC  = BASE / "data" / "processed"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 모델 정의 (HeteroBWGNN_boost — 5종 엣지) ─────────────────────────────────
EDGE_TYPES_BOOST = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
    ("review", "sim",   "review"),
]

class DualFreqConv(MessagePassing):
    def __init__(self, a, b):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(a * 2, b)
    def forward(self, x, ei):
        low = self.propagate(ei, x=x)
        return self.lin(torch.cat([low, x - low], -1))
    def message(self, x_j):
        return x_j

class HeteroBWGNN_Boost(nn.Module):
    def __init__(self, d, h=128, dr=0.3):
        super().__init__()
        self.proj  = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: DualFreqConv(h, h) for et in EDGE_TYPES_BOOST}, aggr="sum")
        self.conv2 = HeteroConv({et: DualFreqConv(h, h) for et in EDGE_TYPES_BOOST}, aggr="sum")
        self.bn1   = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h)
        self.drop  = nn.Dropout(dr)
        self.cls   = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        d = {"review": x}
        d = self.conv1(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
        d = self.conv2(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

def get_probs(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        return torch.sigmoid(logits[mask]).cpu().numpy()

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
data  = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False).to(DEVICE)
FEAT  = data["review"].x.shape[1]
test_mask = data["review"].test_mask
labels    = data["review"].y.numpy()

model = HeteroBWGNN_Boost(FEAT).to(DEVICE)
model.load_state_dict(torch.load(MOD / "HeteroBWGNN_boost_best.pt", weights_only=True))

# ─────────────────────────────────────────────────────────────────────────────
# [분석 1] 엣지 타입별 기여도 (Edge Ablation)
# 각 엣지 타입을 하나씩 제거 → PR-AUC 변화 = 해당 엣지의 기여도
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[분석 1] 엣지 타입별 기여도 (Ablation)")
print("=" * 60)

# 기준 PR-AUC (전체 엣지)
base_probs  = get_probs(model, data, test_mask)
base_pr_auc = average_precision_score(labels[test_mask.numpy()], base_probs)
print(f"  기준 PR-AUC (전체 엣지): {base_pr_auc:.4f}")

attribution_rows = []
for et in EDGE_TYPES_BOOST:
    # 해당 엣지 타입 제거한 데이터 복사
    data_ablated = copy.deepcopy(data)
    data_ablated[et].edge_index = torch.zeros(2, 0, dtype=torch.long, device=DEVICE)
    if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
        data_ablated[et].edge_attr = torch.zeros(0, 1, device=DEVICE)

    ablated_probs  = get_probs(model, data_ablated, test_mask)
    ablated_pr_auc = average_precision_score(labels[test_mask.numpy()], ablated_probs)
    delta = base_pr_auc - ablated_pr_auc

    edge_name = et[1].upper()
    n_edges   = data[et].edge_index.shape[1]
    print(f"  {edge_name:8s}: 제거 후 PR-AUC={ablated_pr_auc:.4f}  Δ={delta:+.4f}  (엣지 {n_edges:,}개)")

    attribution_rows.append({
        "edge_type":       edge_name,
        "n_edges":         n_edges,
        "base_pr_auc":     round(base_pr_auc, 4),
        "ablated_pr_auc":  round(ablated_pr_auc, 4),
        "delta_pr_auc":    round(delta, 4),
        "contribution_pct": round(delta / base_pr_auc * 100, 2),
    })

df_attr = pd.DataFrame(attribution_rows).sort_values("delta_pr_auc", ascending=False)
df_attr.to_csv(RES / "xai_edge_attribution.csv", index=False)
print(f"\n저장: results/xai_edge_attribution.csv")
print(df_attr[["edge_type","n_edges","delta_pr_auc","contribution_pct"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# [분석 2] Burst Δt 분포: 스팸 vs 정상 비교
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[분석 2] Burst Δt 분포 — 스팸 vs 정상")
print("=" * 60)

burst_ei   = data["review", "burst", "review"].edge_index.cpu().numpy()
burst_attr = data["review", "burst", "review"].edge_attr.squeeze().cpu().numpy()
y_np       = data["review"].y.cpu().numpy()

# 소스 노드 기준 스팸/정상 분리
spam_mask_burst  = y_np[burst_ei[0]] == 1
legit_mask_burst = y_np[burst_ei[0]] == 0

spam_dt  = burst_attr[spam_mask_burst]
legit_dt = burst_attr[legit_mask_burst]

print(f"  스팸 burst Δt: 평균={spam_dt.mean():.1f}h  중앙값={np.median(spam_dt):.1f}h  std={spam_dt.std():.1f}h")
print(f"  정상 burst Δt: 평균={legit_dt.mean():.1f}h  중앙값={np.median(legit_dt):.1f}h  std={legit_dt.std():.1f}h")

# 구간별 비율
for threshold in [6, 12, 24, 48]:
    spam_ratio  = (spam_dt  < threshold).mean()
    legit_ratio = (legit_dt < threshold).mean()
    print(f"  Δt < {threshold:2d}h: 스팸 {spam_ratio:.1%}  정상 {legit_ratio:.1%}  "
          f"(스팸이 {spam_ratio/legit_ratio:.2f}x 더 집중됨)" if legit_ratio > 0 else "")

# ─────────────────────────────────────────────────────────────────────────────
# [분석 3] 케이스 스터디 — 상위 10개 스팸 노드 설명
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[분석 3] 고확률 스팸 노드 케이스 스터디")
print("=" * 60)

model.eval()
with torch.no_grad():
    all_probs = torch.sigmoid(model(data)).cpu().numpy()

# Test set에서 실제 스팸이면서 확률 높은 상위 노드
test_idx   = np.where(test_mask.numpy())[0]
spam_test  = test_idx[y_np[test_idx] == 1]
fraud_probs = all_probs[spam_test]
top10_idx  = spam_test[np.argsort(-fraud_probs)[:10]]

df_sample = pd.read_parquet(PROC / "df_sampled.parquet")

case_studies = []
for node_id in top10_idx:
    node_prob = float(all_probs[node_id])

    # 연결된 엣지 통계
    burst_connected = (burst_ei[0] == node_id).sum() + (burst_ei[1] == node_id).sum()
    sim_ei    = data["review", "sim",  "review"].edge_index.cpu().numpy()
    sim_connected  = (sim_ei[0] == node_id).sum()   + (sim_ei[1] == node_id).sum()

    # Burst Δt (해당 노드)
    node_burst_mask = (burst_ei[0] == node_id) | (burst_ei[1] == node_id)
    node_burst_dt   = burst_attr[node_burst_mask]
    avg_dt = float(node_burst_dt.mean()) if len(node_burst_dt) > 0 else -1.0

    row = df_sample.iloc[node_id] if node_id < len(df_sample) else None
    case_studies.append({
        "node_id":          int(node_id),
        "fraud_prob":       round(node_prob, 4),
        "burst_edges":      int(burst_connected),
        "sim_edges":        int(sim_connected),
        "avg_burst_dt_h":   round(avg_dt, 1),
        "rating":           float(row["rating"]) if row is not None else None,
        "text_preview":     str(row["text"])[:80] + "..." if row is not None else "",
    })
    print(f"  노드 {node_id:5d}  P={node_prob:.3f}  burst={burst_connected}개  "
          f"sim={sim_connected}개  avgΔt={avg_dt:.1f}h")

with open(RES / "xai_case_studies.json", "w", encoding="utf-8") as f:
    json.dump(case_studies, f, ensure_ascii=False, indent=2)

print(f"\n저장: results/xai_case_studies.json")

# ─────────────────────────────────────────────────────────────────────────────
# [분석 4] 엣지 타입별 기여도 시각화 (텍스트 바 차트)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[분석 4] 엣지 기여도 요약 (ΔPR-AUC 기준)")
print("=" * 60)
for _, row in df_attr.iterrows():
    bar = "█" * max(0, int(abs(row["delta_pr_auc"]) * 200))
    sign = "+" if row["delta_pr_auc"] >= 0 else "-"
    print(f"  {row['edge_type']:8s} {sign}{abs(row['delta_pr_auc']):.4f}  {bar}")

print("\n✅ XAI 분석 완료")
print("  → 발표 슬라이드: '왜 사기인가' 근거를 엣지 기여도로 설명 가능")
print("  → 케이스 스터디: 실제 사기 리뷰 10개의 탐지 근거 저장됨")
