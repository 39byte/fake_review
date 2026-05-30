"""E10c: GNN Inductive 베이스라인 빠른 버전 (seed=42, 200ep)"""
import copy, json
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

BASE  = Path(__file__).resolve().parent.parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results" / "experiments"
RES.mkdir(parents=True, exist_ok=True)

ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]

data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
tm   = data["review"].train_mask
te   = data["review"].test_mask
y    = data["review"].y

def mask_ind(data, mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

data_ind = mask_ind(data, te)

def ev(model, data, mask):
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(data)[mask]).numpy()
        l=data["review"].y[mask].numpy()
    return (round(float(average_precision_score(l,p)),4),
            round(float(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)),4))

class FocalLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce); w=torch.where(ta==1,torch.full_like(bce,0.75),torch.full_like(bce,0.25))
        return (w*(1-pt)**2*bce).mean()

class SAGEInd(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__(); self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:SAGEConv(h,h) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:SAGEConv(h,h) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

class GATInd(nn.Module):
    def __init__(self,d,h=128,heads=4,dr=0.3):
        super().__init__(); self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=dr,add_self_loops=False) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=dr,add_self_loops=False) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

def run(Model, name, epochs=200):
    torch.manual_seed(42)
    m = Model(feat)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = FocalLoss()
    best_pr, best_s = 0., None
    for ep in range(1, epochs+1):
        m.train(); opt.zero_grad()
        crit(m(data)[tm], y[tm]).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
        opt.step(); sched.step()
        if ep % 40 == 0:
            te_pr, _ = ev(m, data, te)
            if te_pr > best_pr: best_pr=te_pr; best_s={k:v.cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best_s)
    trans_pr, trans_f1 = ev(m, data, te)
    ind_pr,   ind_f1   = ev(m, data_ind, te)
    print(f"  {name:<30} Trans={trans_pr:.4f}  Inductive={ind_pr:.4f}  F1={ind_f1:.4f}")
    return {"name":name,"type":"gnn_inductive","trans_pr":trans_pr,"ind_pr":ind_pr,"ind_f1":ind_f1}

print("="*60)
print("E10c: GNN Inductive 베이스라인 (seed=42, 200ep)")
print("="*60)

results = []
results.append(run(SAGEInd, "GraphSAGE-Inductive"))
results.append(run(GATInd,  "GAT-Inductive"))

# 전체 비교
text_bests = [
    {"name":"TF-IDF+LR","type":"text_only","ind_pr":0.3304,"trans_pr":None},
    {"name":"SBERT+MLP(512,256,128)","type":"text_only","ind_pr":0.3458,"trans_pr":None},
]
all_r = text_bests + results

print("\n" + "="*60)
print("논문 수준 베이스라인 비교 (Inductive PR-AUC)")
print("="*60)
print(f"  {'방법':<35} {'Inductive':>10}  {'Trans':>8}")
print("  " + "-"*58)
print(f"  {'랜덤 분류기 (하한)':<35} {'0.1322':>10}")
for r in sorted(all_r, key=lambda x: x["ind_pr"], reverse=True):
    tr_s = f"{r['trans_pr']:.4f}" if r.get("trans_pr") else "   —  "
    print(f"  {r['name']:<35} {r['ind_pr']:>10.4f}  {tr_s:>8}")
print()
print(f"  {'[우리 GNN]':<35}")
print(f"  {'DRAGWave_TVF (Inductive)':<35} {'0.7429':>10}  {'0.9280':>8}")
print(f"  {'4-way Ensemble (Inductive)':<35} {'0.7748':>10}  {'—':>8}")

best = max(all_r, key=lambda x: x["ind_pr"])
print(f"\n  우리 4-way vs 최강 베이스라인 ({best['name']}):")
print(f"  +{0.7748-best['ind_pr']:.4f} ({(0.7748-best['ind_pr'])/best['ind_pr']*100:.0f}%↑)")

with open(RES/"E10c_fast_result.json","w",encoding="utf-8") as f:
    json.dump(all_r,f,indent=2,ensure_ascii=False)
print(f"\n저장: E10c_fast_result.json  완료")
