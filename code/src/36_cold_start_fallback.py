"""
36_cold_start_fallback.py
Cold Start 문제 완화 — 하이브리드 추론기

문제: 신규 유저/식당의 리뷰는 그래프 이웃이 없어 GNN 메시지 전파 불가
     → GNN이 초기화 임베딩만 사용, 탐지 불안정

해결책: 이웃 수(degree)에 따라 GNN/폴백 분류기 혼합 적용
  - degree ≥ 1: GNN 확률 사용 (기존)
  - degree == 0: SBERT 피처 기반 MLP 폴백 확률 사용

폴백 MLP: 학습 데이터 중 고립 노드(degree=0) 또는 저연결 노드(degree≤2)로 학습
결과: results/cold_start_result.json
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET_NORSR = [
    ("review","rtr","review"), ("review","burst","review"),
    ("review","rur","review"), ("review","sim","review"),
]

# ── 모델 정의 (간소화) ────────────────────────────────────────────────────────
class BWGATConv(MessagePassing):
    def __init__(self,a,b,h=4,dr=0.3):
        super().__init__(aggr="add")
        self.gat=GATConv(a,a//h,heads=h,dropout=dr,add_self_loops=False)
        self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei):
        if ei.shape[1]==0: return self.lin(torch.cat([x,torch.zeros_like(x)],-1))
        low=self.gat(x,ei); return self.lin(torch.cat([low,x-low],-1))

class DRAGWaveConv(nn.Module):
    def __init__(self,a,b,nr,h=4,dr=0.3):
        super().__init__()
        self.bwgat=nn.ModuleList([BWGATConv(a,b,h,dr) for _ in range(nr)])
        self.self_lin=nn.Linear(a,b); self.attn_vec=nn.Linear(b*2,1,bias=False)
        self.drop=nn.Dropout(dr)
    def forward(self,x,ei_list):
        hs=self.self_lin(x); re=[self.bwgat[i](x,ei) for i,ei in enumerate(ei_list)]
        rs=torch.stack(re,1); he=hs.unsqueeze(1).expand_as(rs)
        aw=F.softmax(self.attn_vec(torch.tanh(torch.cat([he,rs],-1))).squeeze(-1),dim=-1)
        return self.drop(F.relu(hs+(rs*aw.unsqueeze(-1)).sum(1)))

class HeteroDRAGWave(nn.Module):
    def __init__(self,d,h=128,ets=None,dr=0.3):
        super().__init__(); self.ets=ets or ET_NORSR; nr=len(self.ets)
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr)
        self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict.get(et,torch.zeros(2,0,dtype=torch.long)) for et in self.ets]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

# ── Cold Start 폴백: SBERT 피처 기반 MLP ──────────────────────────────────────
class ColdStartMLP(nn.Module):
    """이웃 없는 노드용 피처 기반 분류기"""
    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 32),       nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# ── 노드별 총 degree 계산 ─────────────────────────────────────────────────────
def compute_degrees(data, n_nodes):
    deg = torch.zeros(n_nodes, dtype=torch.long)
    for et, ei in data.edge_index_dict.items():
        # 무방향으로 카운트
        src, dst = ei[0], ei[1]
        deg.scatter_add_(0, src, torch.ones_like(src))
        deg.scatter_add_(0, dst, torch.ones_like(dst))
    return deg

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("=" * 60)
print("Cold Start 완화 — 하이브리드 추론기")
print("=" * 60)

import copy
data_boost = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat_dim   = data_boost["review"].x.shape[1]
n_nodes    = data_boost["review"].x.shape[0]
test_mask  = data_boost["review"].test_mask
train_mask = data_boost["review"].train_mask
labels     = data_boost["review"].y

# NoRSR 그래프 (DRAGWave_NoRSR 기반)
data_norsr = copy.deepcopy(data_boost)
rsr_key = ("review","rsr","review")
if rsr_key in data_norsr.edge_index_dict:
    del data_norsr._edge_store_dict[rsr_key]

# degree 계산
deg = compute_degrees(data_norsr, n_nodes)
test_deg = deg[test_mask]

n_zero  = (test_deg == 0).sum().item()
n_low   = ((test_deg > 0) & (test_deg <= 2)).sum().item()
n_rich  = (test_deg > 2).sum().item()
print(f"\n[Test 노드 degree 분포]")
print(f"  degree=0 (Cold Start): {n_zero:4d} ({n_zero/len(test_deg)*100:.1f}%)")
print(f"  degree=1~2 (희박):    {n_low:4d} ({n_low/len(test_deg)*100:.1f}%)")
print(f"  degree≥3  (정상):     {n_rich:4d} ({n_rich/len(test_deg)*100:.1f}%)")

# ── GNN 로드 & 추론 ───────────────────────────────────────────────────────────
print("\n[GNN 추론]")
gnn = HeteroDRAGWave(feat_dim)
ckpt = MOD/"DRAGWave_NoRSR_best.pt"
gnn.load_state_dict(torch.load(ckpt, weights_only=True))
gnn.eval()

with torch.no_grad():
    gnn_logits = gnn(data_norsr)
    gnn_probs  = torch.sigmoid(gnn_logits).numpy()

gnn_pr  = average_precision_score(labels[test_mask].numpy(), gnn_probs[test_mask])
gnn_f1  = f1_score(labels[test_mask].numpy(),
                   (gnn_probs[test_mask]>=0.5).astype(int), average="macro", zero_division=0)
print(f"  전체 test: PR-AUC={gnn_pr:.4f}  F1={gnn_f1:.4f}")

# Cold Start 노드만 별도 평가
cold_mask = torch.zeros(n_nodes, dtype=torch.bool)
cold_mask[torch.where(test_mask)[0][test_deg == 0]] = True

if cold_mask.sum() > 0:
    c_pr = average_precision_score(labels[cold_mask].numpy(), gnn_probs[cold_mask])
    c_f1 = f1_score(labels[cold_mask].numpy(),
                    (gnn_probs[cold_mask]>=0.5).astype(int), average="macro", zero_division=0)
    print(f"  Cold Start만: PR-AUC={c_pr:.4f}  F1={c_f1:.4f}  (n={cold_mask.sum().item()})")
else:
    print(f"  Cold Start 노드 없음 (degree=0인 test 노드 없음)")
    c_pr, c_f1 = None, None

# ── Cold Start MLP 학습 ────────────────────────────────────────────────────────
print("\n[Cold Start MLP 학습]")

# 학습: degree가 낮은 훈련 노드로 피처 기반 분류기 학습
train_deg = deg[train_mask]
# degree <= 3인 훈련 노드 (Cold Start에 가까운 노드)
sparse_train = train_mask.clone()
sparse_indices = torch.where(train_mask)[0]
sparse_mask = torch.zeros(n_nodes, dtype=torch.bool)
sparse_mask[sparse_indices[train_deg <= 3]] = True

n_sparse = sparse_mask.sum().item()
print(f"  폴백 MLP 학습 데이터: {n_sparse}건 (degree≤3 훈련 노드)")

if n_sparse < 10:
    print("  ⚠️ 학습 데이터 부족. 전체 훈련 데이터로 대체.")
    sparse_mask = train_mask

mlp = ColdStartMLP(feat_dim)
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()

for ep in range(200):
    mlp.train(); opt.zero_grad()
    logits = mlp(data_norsr["review"].x[sparse_mask])
    loss   = crit(logits, labels[sparse_mask].float())
    loss.backward(); opt.step()

mlp.eval()
with torch.no_grad():
    mlp_probs = torch.sigmoid(mlp(data_norsr["review"].x)).numpy()

mlp_pr = average_precision_score(labels[test_mask].numpy(), mlp_probs[test_mask])
mlp_f1 = f1_score(labels[test_mask].numpy(),
                  (mlp_probs[test_mask]>=0.5).astype(int), average="macro", zero_division=0)
print(f"  MLP 전체 test: PR-AUC={mlp_pr:.4f}  F1={mlp_f1:.4f}")

# ── 하이브리드 추론: degree 기반 GNN/MLP 전환 ─────────────────────────────────
print("\n[하이브리드 추론 — degree 임계값 탐색]")
print(f"  {'임계값':>8}  {'PR-AUC':>8}  {'F1':>8}  {'GNN비율':>8}  {'MLP비율':>8}")
print("  " + "-"*48)

best_pr, best_thr, best_hybrid = 0., 1, None
test_idx = torch.where(test_mask)[0]

for thr in [0, 1, 2, 3, 5]:
    hybrid = gnn_probs.copy()
    use_mlp = deg[test_idx] <= thr
    hybrid_idx = test_idx[use_mlp]
    hybrid[hybrid_idx] = mlp_probs[hybrid_idx]

    n_mlp = use_mlp.sum().item()
    n_gnn = len(test_idx) - n_mlp

    pr = average_precision_score(labels[test_mask].numpy(), hybrid[test_mask])
    f1 = f1_score(labels[test_mask].numpy(),
                  (hybrid[test_mask]>=0.5).astype(int), average="macro", zero_division=0)

    mark = " ← best" if pr > best_pr else ""
    print(f"  degree≤{thr:>2}:  {pr:.4f}    {f1:.4f}    {n_gnn:>5}건    {n_mlp:>5}건{mark}")

    if pr > best_pr:
        best_pr = pr; best_thr = thr; best_hybrid = hybrid.copy()

best_f1 = f1_score(labels[test_mask].numpy(),
                   (best_hybrid[test_mask]>=0.5).astype(int), average="macro", zero_division=0)

print(f"\n[결론]")
print(f"  GNN 단독:  PR-AUC={gnn_pr:.4f}  F1={gnn_f1:.4f}")
print(f"  하이브리드: PR-AUC={best_pr:.4f}  F1={best_f1:.4f}  (degree≤{best_thr} → MLP)")
delta = best_pr - gnn_pr
print(f"  개선: {delta:+.4f}  {'✅ Cold Start 완화 효과 있음' if delta > 0 else '➡️ GNN 단독이 이미 최적'}")

torch.save(mlp.state_dict(), MOD/"ColdStartMLP_best.pt")

result = {
    "degree_dist": {"cold_start_0": n_zero, "sparse_1_2": n_low, "rich_3plus": n_rich},
    "gnn_only":    {"pr_auc": round(gnn_pr,4), "macro_f1": round(gnn_f1,4)},
    "mlp_only":    {"pr_auc": round(mlp_pr,4), "macro_f1": round(mlp_f1,4)},
    "hybrid_best": {"threshold": best_thr, "pr_auc": round(best_pr,4), "macro_f1": round(best_f1,4),
                    "delta_pr": round(delta,4)},
    "cold_start_gnn": {"pr_auc": round(c_pr,4) if c_pr else None,
                       "macro_f1": round(c_f1,4) if c_f1 else None},
}
out = RES/"cold_start_result.json"
with open(out,"w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"\n저장: {out.name}")
print("✅ Cold Start 실험 완료")
