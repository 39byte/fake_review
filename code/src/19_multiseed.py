"""
19_multiseed.py
단일 시드 문제 해결 — 3개 시드로 DRAGWave 300ep 실험
목표: "PR-AUC 0.934 ± σ" 형태로 신뢰구간 제시
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, time, json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH=BASE/"data/graphs"; MOD=BASE/"models"; RES=BASE/"results"
DEVICE=torch.device("cpu")
EDGE_TYPES=[("review","rtr","review"),("review","rsr","review"),
            ("review","burst","review"),("review","rur","review"),("review","sim","review")]
N_REL=len(EDGE_TYPES)

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce); w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

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
        self.self_lin=nn.Linear(a,b); self.attn=nn.Linear(b*2,1,bias=False); self.drop=nn.Dropout(dr)
    def forward(self,x,ei_list):
        hs=self.self_lin(x); re=[self.bwgat[i](x,ei) for i,ei in enumerate(ei_list)]
        rs=torch.stack(re,1); he=hs.unsqueeze(1).expand_as(rs)
        aw=F.softmax(self.attn(torch.tanh(torch.cat([he,rs],-1))).squeeze(-1),dim=-1)
        return self.drop(F.relu(hs+(rs*aw.unsqueeze(-1)).sum(1)))

class DRAGWave(nn.Module):
    def __init__(self,d,h=128,nr=N_REL,heads=4,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h); self.l1=DRAGWaveConv(h,h,nr,heads,dr); self.l2=DRAGWaveConv(h,h,nr,heads,dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict[et] for et in EDGE_TYPES]
        h1=self.bn1(self.l1(x,ei)); h2=self.bn2(self.l2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

def run(seed, data, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    feat=data["review"].x.shape[1]; model=DRAGWave(feat).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    crit=FocalLoss(); tm=data["review"].train_mask; lb=data["review"].y
    best_pr,best_st,no_imp=0.,None,0; t0=time.time()
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        loss=crit(model(data)[tm],lb[tm]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
        if ep%30==0 or ep==1:
            model.eval()
            with torch.no_grad():
                p=torch.sigmoid(model(data)[data["review"].test_mask]).numpy()
                l=data["review"].y[data["review"].test_mask].numpy()
            pr=round(average_precision_score(l,p),4)
            f1=round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
            print(f"  [seed={seed}] ep={ep:3d}  PR-AUC={pr}  F1={f1}  ({time.time()-t0:.0f}s)")
            if pr>best_pr: best_pr=pr; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp>=5: print(f"  [seed={seed}] Early stop ep={ep}"); break
    model.load_state_dict(best_st); model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(data)[data["review"].test_mask]).numpy()
        l=data["review"].y[data["review"].test_mask].numpy()
    return round(average_precision_score(l,p),4), round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)

data=torch.load(GRAPH/"hetero_graph_boost.pt",weights_only=False)
seeds=[42,123,456]; results=[]
print("=== DRAGWave 멀티 시드 실험 (300 epoch × 3 seed) ===")
for s in seeds:
    print(f"\n[Seed {s}]")
    pr,f1=run(s,data,300)
    print(f"  → FINAL PR-AUC={pr}  F1={f1}")
    results.append({"seed":s,"pr_auc":pr,"macro_f1":f1})

prs=[r["pr_auc"] for r in results]; f1s=[r["macro_f1"] for r in results]
print(f"\n=== 멀티 시드 결과 ===")
for r in results: print(f"  seed={r['seed']}: PR-AUC={r['pr_auc']}  F1={r['macro_f1']}")
print(f"\n  PR-AUC: {np.mean(prs):.4f} ± {np.std(prs):.4f}")
print(f"  Macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

summary={"seeds":seeds,"results":results,
          "pr_auc_mean":round(np.mean(prs),4),"pr_auc_std":round(np.std(prs),4),
          "macro_f1_mean":round(np.mean(f1s),4),"macro_f1_std":round(np.std(f1s),4)}
with open(RES/"multiseed_result.json","w") as f: json.dump(summary,f,indent=2)
print(f"\n저장: results/multiseed_result.json")
