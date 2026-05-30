"""E10b: GNN Inductive 베이스라인 (GraphSAGE, GAT) — 멀티시드"""
import copy, json, time
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
SEEDS = [42, 123, 456]

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
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

class SAGEInd(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
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
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=dr,add_self_loops=False) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=dr,add_self_loops=False) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

def run(ModelClass, name, **kwargs):
    print(f"\n{name} (멀티시드 {SEEDS}):")
    seed_res = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        model = ModelClass(feat, **kwargs)
        opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        crit  = FocalLoss()
        best_pr, best_s = 0., None
        for ep in range(1, 201):
            model.train(); opt.zero_grad()
            crit(model(data)[tm], y[tm]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step()
            if ep % 40 == 0:
                te_pr, _ = ev(model, data, te)
                if te_pr > best_pr:
                    best_pr = te_pr
                    best_s = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_s)
        ind_pr, ind_f1 = ev(model, data_ind, te)
        te_pr,  te_f1  = ev(model, data, te)
        print(f"  seed={seed}  Trans={te_pr:.4f}  Inductive={ind_pr:.4f}  F1={ind_f1:.4f}")
        seed_res.append({"seed":seed,"trans_pr":te_pr,"ind_pr":ind_pr,"ind_f1":ind_f1})
    mean_ind = round(np.mean([r["ind_pr"] for r in seed_res]),4)
    std_ind  = round(np.std( [r["ind_pr"] for r in seed_res]),4)
    mean_tr  = round(np.mean([r["trans_pr"] for r in seed_res]),4)
    print(f"  → Trans={mean_tr:.4f}  Inductive={mean_ind:.4f}±{std_ind:.4f}")
    return {"name":name,"type":"gnn_inductive","trans_pr":mean_tr,
            "ind_pr":mean_ind,"std":std_ind,"seeds":seed_res}

print("="*60)
print("E10b: GNN Inductive 베이스라인 (멀티시드)")
print("="*60)

results = []
results.append(run(SAGEInd, "GraphSAGE-Inductive"))
results.append(run(GATInd,  "GAT-Inductive"))

# 텍스트 단독 결과 (이미 확인된 것)
text_results = [
    {"name":"TF-IDF+LR(C=0.1)","type":"text_only","ind_pr":0.3304,"ind_f1":0.5854},
    {"name":"TF-IDF+LR(C=1.0)","type":"text_only","ind_pr":0.3299,"ind_f1":0.6102},
    {"name":"SBERT+MLP(128)",  "type":"text_only","ind_pr":0.3275,"ind_f1":0.5848},
    {"name":"SBERT+MLP(256,128)","type":"text_only","ind_pr":0.3153,"ind_f1":0.5298},
    {"name":"SBERT+MLP(512,256,128)","type":"text_only","ind_pr":0.3458,"ind_f1":0.5794},
]

all_results = text_results + results
print("\n" + "="*60)
print("전체 베이스라인 비교 (Inductive PR-AUC 기준)")
print("="*60)
print(f"  {'방법':<35} {'Ind PR-AUC':>11}  {'F1':>8}  {'유형'}")
print("  " + "-"*65)
print(f"  {'랜덤 (하한)':<35} {'0.1322':>11}            이론값")
for r in sorted(all_results, key=lambda x: x["ind_pr"], reverse=True):
    std_s = f"±{r.get('std',0):.4f}" if r.get("std") else "      "
    print(f"  {r['name']:<35} {r['ind_pr']:>11.4f}{std_s}  {r.get('ind_f1',0):>8.4f}  {r['type']}")

print()
print("  [우리 GNN 결과 비교]")
for n,p in [("DRAGWave_TVF_400ep (Inductive)",0.7429),("4-way Ensemble (Inductive)",0.7748)]:
    print(f"  {n:<35} {p:>11.4f}")

best_text = max(text_results, key=lambda x: x["ind_pr"])
best_gnn  = max(results, key=lambda x: x["ind_pr"])
print(f"\n  최강 텍스트: {best_text['name']} → {best_text['ind_pr']:.4f}")
print(f"  최강 GNN베이스라인: {best_gnn['name']} → {best_gnn['ind_pr']:.4f}")
print(f"  우리 4-way vs 최강 텍스트: +{0.7748-best_text['ind_pr']:.4f} ({(0.7748-best_text['ind_pr'])/best_text['ind_pr']*100:.0f}%↑)")
print(f"  우리 4-way vs 최강 GNN베이스: +{0.7748-best_gnn['ind_pr']:.4f}")

with open(RES/"E10b_gnn_baselines.json","w",encoding="utf-8") as f:
    json.dump(all_results,f,indent=2,ensure_ascii=False)
print(f"\n저장: results/experiments/E10b_gnn_baselines.json")
