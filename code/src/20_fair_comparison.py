"""
20_fair_comparison.py
공정 비교 — HeteroBWGNN + BWGAT를 400 epoch로 학습
"DRAGWave가 우월한 게 epoch 수 때문인가, 아키텍처 때문인가" 검증
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, time
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH=BASE/"data/graphs"; MOD=BASE/"models"; RES=BASE/"results"
DEVICE=torch.device("cpu")
EDGE_TYPES=[("review","rtr","review"),("review","rsr","review"),
            ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none"); pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def evaluate(m,data,mask):
    m.eval()
    with torch.no_grad():
        p=torch.sigmoid(m(data)[mask]).numpy(); l=data["review"].y[mask].numpy()
    return round(average_precision_score(l,p),4), round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)

def train_model(model, name, data, epochs=400):
    model=model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    crit=FocalLoss(); tm=data["review"].train_mask; lb=data["review"].y
    best_pr,best_st,no_imp=0.,None,0; t0=time.time()
    print(f"\n▶ {name} 400 epoch 학습")
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        loss=crit(model(data)[tm],lb[tm]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
        if ep%40==0 or ep==1:
            tr_pr,_=evaluate(model,data,tm); te_pr,te_f1=evaluate(model,data,data["review"].test_mask)
            gap=round(tr_pr-te_pr,4)
            print(f"  ep={ep:3d}  loss={loss.item():.4f}  train={tr_pr:.4f}  test={te_pr:.4f}  gap={gap:+.4f}  ({time.time()-t0:.0f}s)")
            if te_pr>best_pr: best_pr=te_pr; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp>=6: print(f"  Early stop ep={ep}"); break
    model.load_state_dict(best_st)
    tr_pr,_=evaluate(model,data,tm); te_pr,te_f1=evaluate(model,data,data["review"].test_mask)
    gap=round(tr_pr-te_pr,4)
    n_params=sum(p.numel() for p in model.parameters())
    elapsed=round(time.time()-t0,1)
    print(f"  FINAL  train={tr_pr}  test={te_pr}  gap={gap:+.4f}  F1={te_f1}  ({elapsed}s)")
    torch.save(best_st, MOD/f"{name}_400ep_best.pt")
    return {"model":name,"pr_auc":te_pr,"macro_f1":te_f1,"train_pr":tr_pr,"gap":gap,"params":n_params}

data=torch.load(GRAPH/"hetero_graph_boost.pt",weights_only=False)
feat=data["review"].x.shape[1]
torch.manual_seed(42)

# HeteroBWGNN 400 epoch (부스트 그래프)
class HeteroBWGNN_400(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

# BWGAT 400 epoch
class BWGATConv(MessagePassing):
    def __init__(self,a,b,h=4,dr=0.3):
        super().__init__(aggr="add")
        self.gat=GATConv(a,a//h,heads=h,dropout=dr,add_self_loops=False); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei):
        if ei.shape[1]==0: return self.lin(torch.cat([x,torch.zeros_like(x)],-1))
        low=self.gat(x,ei); return self.lin(torch.cat([low,x-low],-1))

class HeteroBWGAT_400(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:BWGATConv(h,h,dr=dr) for et in EDGE_TYPES},aggr="sum")
        self.conv2=HeteroConv({et:BWGATConv(h,h,dr=dr) for et in EDGE_TYPES},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

results=[]
results.append(train_model(HeteroBWGNN_400(feat), "HeteroBWGNN_400ep", data, 400))
results.append(train_model(HeteroBWGAT_400(feat), "BWGAT_400ep",       data, 400))

# 결과 비교
print("\n" + "="*65)
print("=== 공정 비교 결과 (모두 400 epoch, 동일 그래프) ===")
print("="*65)
# 기존 DRAGWave_400ep
print(f"  {'DRAGWave_400ep':30s} PR-AUC=0.9340  F1=0.9331  gap=+0.0659")
for r in results:
    print(f"  {r['model']:30s} PR-AUC={r['pr_auc']:.4f}  F1={r['macro_f1']:.4f}  gap={r['gap']:+.4f}")

# experiment_log 업데이트
df=pd.read_csv(RES/"experiment_log.csv")
for r in results:
    if r["model"] not in df["model"].values:
        new=pd.DataFrame([{"model":r["model"],"pr_auc":r["pr_auc"],"macro_f1":r["macro_f1"],
                           "params":r["params"],"train_sec":0,"notes":f"400ep 공정비교 gap={r['gap']:+.4f}"}])
        df=pd.concat([df,new],ignore_index=True)
df.to_csv(RES/"experiment_log.csv",index=False)
pd.DataFrame(results).to_csv(RES/"fair_comparison_400ep.csv",index=False)
print(f"\n저장: results/fair_comparison_400ep.csv")
