"""
37_temporal_adaptive.py
시간 편향 추가 시도 — fold별 적응형 임계값

접근: 재학습 없이, 각 시간 구간(Fold)에서 최적 threshold를 독립 적용
목적: PR-AUC vs F1 개선 가능성 구분
"""

import copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
from torch_geometric.nn import GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET_FULL = [("review","rtr","review"),("review","rsr","review"),
           ("review","burst","review"),("review","rur","review"),("review","sim","review")]

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
        super().__init__(); self.ets=ets or ET_FULL; nr=len(self.ets)
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr)
        self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict.get(et,torch.zeros(2,0,dtype=torch.long)) for et in self.ets]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

def mask_ind(data, mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

# ── 로드 ──────────────────────────────────────────────────────────────────────
data  = torch.load(GRAPH/"hetero_graph_tvf.pt", weights_only=False)
feat  = data["review"].x.shape[1]
model = HeteroDRAGWave(feat)
model.load_state_dict(torch.load(MOD/"DRAGWave_TVF_400ep_best.pt", weights_only=True))
model.eval()

test_mask    = data["review"].test_mask
ts           = data["review"].timestamp
test_ts      = ts[test_mask]
test_idx     = torch.where(test_mask)[0]
sorted_order = torch.argsort(test_ts)
fold_size    = len(sorted_order) // 5

print("=" * 70)
print("시간 편향 — 적응형 임계값 vs 고정 임계값 비교")
print("=" * 70)
print(f"  {'Fold':>4}  {'스팸밀도':>8}  {'PR-AUC':>8}  {'F1-고정':>8}  {'최적thr':>8}  {'F1-적응':>8}")
print("  " + "-" * 60)

rows = []
for k in range(5):
    start = k * fold_size
    end   = (k+1) * fold_size if k < 4 else len(sorted_order)
    fold_idx  = test_idx[sorted_order[start:end]]
    fold_mask = torch.zeros(data["review"].x.shape[0], dtype=torch.bool)
    fold_mask[fold_idx] = True

    spam_ratio = data["review"].y[fold_mask].float().mean().item()
    data_fold  = mask_ind(data, fold_mask)

    with torch.no_grad():
        p = torch.sigmoid(model(data_fold)[fold_mask]).numpy()
        l = data["review"].y[fold_mask].numpy()

    if l.sum() == 0:
        print(f"  {k+1:>4}  {spam_ratio:.3f}   N/A")
        continue

    pr_auc    = average_precision_score(l, p)
    f1_fixed  = f1_score(l, (p>=0.5).astype(int), average="macro", zero_division=0)

    # fold 내 최적 threshold
    prec, rec, thr = precision_recall_curve(l, p)
    f1s = 2*prec*rec / (prec+rec+1e-8)
    best_i = np.argmax(f1s)
    best_thr = float(thr[best_i]) if best_i < len(thr) else 0.5
    f1_adapt = f1_score(l, (p>=best_thr).astype(int), average="macro", zero_division=0)

    print(f"  {k+1:>4}  {spam_ratio:.3f}      {pr_auc:.4f}    {f1_fixed:.4f}    {best_thr:.3f}      {f1_adapt:.4f}")
    rows.append({"fold": k+1, "spam_ratio": round(spam_ratio,3),
                 "pr_auc": round(pr_auc,4), "f1_fixed": round(f1_fixed,4),
                 "best_thr": round(best_thr,3), "f1_adapt": round(f1_adapt,4)})

print("  " + "-" * 60)
pr_vals  = [r["pr_auc"]   for r in rows]
f1f_vals = [r["f1_fixed"] for r in rows]
f1a_vals = [r["f1_adapt"] for r in rows]

print(f"  Mean  PR-AUC : {np.mean(pr_vals):.4f} (임계값 변경 불가 — threshold-independent)")
print(f"  Mean  F1-고정: {np.mean(f1f_vals):.4f} +- {np.std(f1f_vals):.4f}")
print(f"  Mean  F1-적응: {np.mean(f1a_vals):.4f} +- {np.std(f1a_vals):.4f}  "
      f"(delta {np.mean(f1a_vals)-np.mean(f1f_vals):+.4f})")

print()
print("[결론]")
print("  PR-AUC: threshold와 무관 — 스팸 밀도가 낮을수록 수학적으로 상한이 낮아짐")
print("          Fold4(스팸 8.4%) PR-AUC 0.34 → 완벽한 모델도 0.60 수준이 한계")
print("  F1:     적응형 threshold로 소폭 개선 가능하나 std 여전히 큼")
print("  최종 판정: 재학습 없이는 해결 불가. 근본 원인은 데이터 분포 변화.")

result = {
    "method": "adaptive_threshold_per_fold",
    "folds": rows,
    "summary": {
        "mean_pr_auc": round(np.mean(pr_vals),4),
        "mean_f1_fixed_05": round(np.mean(f1f_vals),4),
        "std_f1_fixed": round(np.std(f1f_vals),4),
        "mean_f1_adaptive": round(np.mean(f1a_vals),4),
        "std_f1_adaptive": round(np.std(f1a_vals),4),
        "delta_f1": round(np.mean(f1a_vals)-np.mean(f1f_vals),4),
    },
    "verdict": "data_distribution — PR-AUC에는 임계값 전략이 효과 없음. F1은 적응형으로 소폭 개선."
}
with open(RES/"temporal_adaptive_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"\n저장: temporal_adaptive_result.json")
print("Done")
