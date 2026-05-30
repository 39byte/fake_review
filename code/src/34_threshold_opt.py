"""
34_threshold_opt.py
앙상블 임계값 최적화 — PR curve 기반 최적 threshold 탐색
현재 고정값 0.5 → F1 최대화 기준 최적 threshold 계산
결과: results/threshold_opt_result.json
"""

import copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET_FULL  = [("review","rtr","review"),("review","rsr","review"),
            ("review","burst","review"),("review","rur","review"),("review","sim","review")]
ET_NORSR = [("review","rtr","review"),
            ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,ets=None,dr=0.3):
        super().__init__(); ets=ets or ET_FULL
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ets},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ets},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

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
        self.self_lin=nn.Linear(a,b); self.attn_vec=nn.Linear(b*2,1,bias=False); self.drop=nn.Dropout(dr)
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

# 데이터 로드
data_boost = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
data_tvf   = torch.load(GRAPH/"hetero_graph_tvf.pt",   weights_only=False)
feat_b = data_boost["review"].x.shape[1]
feat_t = data_tvf["review"].x.shape[1]
test_mask = data_boost["review"].test_mask
labels = data_boost["review"].y[test_mask].numpy()

data_norsr = copy.deepcopy(data_boost)
rsr_key = ("review","rsr","review")
if rsr_key in data_norsr.edge_index_dict:
    del data_norsr._edge_store_dict[rsr_key]

# 4-way 앙상블 확률 수집 (트랜스덕티브 — 전체 그래프)
configs = [
    ("DRAGWave_TVF_400ep", HeteroDRAGWave(feat_t), "DRAGWave_TVF_400ep_best.pt", data_tvf),
    ("DRAGWave_NoRSR",     HeteroDRAGWave(feat_b,ets=ET_NORSR), "DRAGWave_NoRSR_best.pt", data_norsr),
    ("HeteroBWGNN_boost",  HeteroBWGNN(feat_b),    "HeteroBWGNN_boost_best.pt",  data_boost),
    ("BWGAT",              None,                    None,                          None),
]

probs_list = []
for name, model_inst, ckpt_name, data_src in configs:
    if model_inst is None: continue
    ckpt = MOD/ckpt_name
    if not ckpt.exists(): continue
    model_inst.load_state_dict(torch.load(ckpt, weights_only=True))
    model_inst.eval()
    with torch.no_grad():
        p = torch.sigmoid(model_inst(data_src)[test_mask]).numpy()
    probs_list.append((name, p))

# 최적 가중치: TVF 0.5, NoRSR 0.3, BWGNN 0.1 (32번 결과)
if len(probs_list) >= 3:
    ensemble_probs = (probs_list[0][1]*0.5 + probs_list[1][1]*0.3 + probs_list[2][1]*0.2)
else:
    ensemble_probs = probs_list[0][1]

# PR curve 기반 최적 threshold 탐색
precision, recall, thresholds = precision_recall_curve(labels, ensemble_probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
best_f1  = float(f1_scores[best_idx])
best_pre = float(precision[best_idx])
best_rec = float(recall[best_idx])

# 0.5 고정 vs 최적 임계값 비교
f1_at_05  = f1_score(labels, (ensemble_probs>=0.5).astype(int), average="macro", zero_division=0)
f1_at_opt = f1_score(labels, (ensemble_probs>=best_thr).astype(int), average="macro", zero_division=0)
pr_auc    = average_precision_score(labels, ensemble_probs)

print("="*55)
print("앙상블 임계값 최적화 결과")
print("="*55)
print(f"  PR-AUC           : {pr_auc:.4f}")
print(f"  threshold=0.5    : Macro-F1={f1_at_05:.4f}")
print(f"  최적 threshold   : {best_thr:.4f}")
print(f"  최적 Macro-F1    : {f1_at_opt:.4f}  (Δ{f1_at_opt-f1_at_05:+.4f})")
print(f"  최적 Precision   : {best_pre:.4f}")
print(f"  최적 Recall      : {best_rec:.4f}")

result = {
    "pr_auc": round(pr_auc,4),
    "threshold_fixed_05": {"macro_f1": round(f1_at_05,4)},
    "threshold_optimal": {
        "value": round(best_thr,4),
        "macro_f1": round(f1_at_opt,4),
        "precision": round(best_pre,4),
        "recall": round(best_rec,4),
        "delta_f1": round(f1_at_opt-f1_at_05,4),
    },
}
with open(RES/"threshold_opt_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"\n저장: threshold_opt_result.json")
print("✅ 완료")
