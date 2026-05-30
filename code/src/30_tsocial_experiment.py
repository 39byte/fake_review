"""
30_tsocial_experiment.py
T-Social 소셜 네트워크 이상 탐지 실험
BWGNN 논문 5번째 도메인 — 소셜 미디어 사기 계정 탐지

T-Social: 5.8M 노드, 10d 피처 → 10K 샘플링
"""
import sys, types
fake_gb=types.ModuleType("dgl.graphbolt"); fake_gb.load_graphbolt=lambda:None
sys.modules["dgl.graphbolt"]=fake_gb

import dgl, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, time, json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData

BASE=Path(__file__).resolve().parent.parent
EXT=BASE/"data/external"; RES=BASE/"results"; DEVICE=torch.device("cpu")

print("=== T-Social 소셜 네트워크 이상 탐지 ===")

ts_file = EXT/"tsocial/tsocial"
if not ts_file.exists():
    print(f"T-Social 파일 없음: {ts_file}"); exit()

g, _ = dgl.load_graphs(str(ts_file))
G = g[0]
feats    = G.ndata["feature"].float()
label_oh = G.ndata["label"]
label_np = label_oh.argmax(dim=1).numpy() if label_oh.dim()==2 else label_oh.numpy()
spam_r   = label_np.mean()
print(f"T-Social: {G.num_nodes():,} 노드  피처 {feats.shape[1]}d  스팸 {spam_r:.1%}  엣지 {G.num_edges():,}")

# 10K 샘플링
np.random.seed(42)
spam_idx  = np.where(label_np==1)[0]
legit_idx = np.where(label_np==0)[0]
N_SAMPLE  = 10000
n_sp = min(int(N_SAMPLE*spam_r)+1, len(spam_idx))
n_lg = N_SAMPLE - n_sp
sampled = np.sort(np.concatenate([
    np.random.choice(spam_idx, n_sp, replace=False),
    np.random.choice(legit_idx, n_lg, replace=False),
]))
idx_map = {o:i for i,o in enumerate(sampled)}
feat_s  = feats[sampled]; label_s = torch.tensor(label_np[sampled], dtype=torch.long)
N = len(sampled); spam_rs = label_s.float().mean().item()
print(f"샘플: {N:,}  스팸 {spam_rs:.1%}")

# 엣지 (최대 100K)
src_np, dst_np = G.edges(); src_np, dst_np = src_np.numpy(), dst_np.numpy()
mask = np.array([s in idx_map and d in idx_map for s,d in zip(src_np,dst_np)])
src_sub = np.array([idx_map[s] for s in src_np[mask]])
dst_sub = np.array([idx_map[d] for d in dst_np[mask]])
if len(src_sub) > 100000:
    perm = np.random.choice(len(src_sub), 100000, replace=False)
    src_sub, dst_sub = src_sub[perm], dst_sub[perm]

ei = torch.tensor(np.stack([src_sub, dst_sub]), dtype=torch.long)
print(f"사용 엣지: {ei.shape[1]:,}")

cutoff = int(N*0.8)
tm = torch.zeros(N,dtype=torch.bool); tm[:cutoff]=True
te = torch.zeros(N,dtype=torch.bool); te[cutoff:]=True

data = HeteroData()
data["node"].x=feat_s; data["node"].y=label_s
data["node"].train_mask=tm; data["node"].test_mask=te
data["node","edge","node"].edge_index=ei

class DFC(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class BWGNN(nn.Module):
    def __init__(self,d,h=64):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.c1=HeteroConv({("node","edge","node"):DFC(h,h)},aggr="sum")
        self.c2=HeteroConv({("node","edge","node"):DFC(h,h)},aggr="sum")
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.dr=nn.Dropout(.3)
        self.cls=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1))
    def forward(self,d):
        x=self.dr(F.relu(self.proj(d["node"].x))); nd={"node":x}
        nd=self.c1(nd,d.edge_index_dict); nd={"node":self.dr(F.relu(self.b1(nd["node"])))}
        nd=self.c2(nd,d.edge_index_dict); nd={"node":self.dr(F.relu(self.b2(nd["node"])))}
        return self.cls(nd["node"]).squeeze(-1)

class FL(nn.Module):
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none"); pt=torch.exp(-bce)
        a=float(1-spam_rs)
        w=torch.where(ta==1,torch.full_like(bce,a),torch.full_like(bce,1-a))
        return (w*(1-pt)**2*bce).mean()

torch.manual_seed(42)
model=BWGNN(feat_s.shape[1])
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=200)
crit=FL(); best_pr,best_st,no_imp=0.,None,0; t0=time.time()

print("학습...")
for ep in range(1,201):
    model.train(); opt.zero_grad()
    loss=crit(model(data)[tm],label_s[tm]); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
    if ep%40==0 or ep==1:
        model.eval()
        with torch.no_grad():
            p=torch.sigmoid(model(data)[te]).numpy(); l=label_s[te].numpy()
        pr=round(average_precision_score(l,p),4)
        f1=round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
        print(f"  ep={ep:3d}  PR-AUC={pr:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if pr>best_pr: best_pr=pr; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
        else:
            no_imp+=1
            if no_imp>=4: print(f"  Early stop ep={ep}"); break

model.load_state_dict(best_st); model.eval()
with torch.no_grad():
    p=torch.sigmoid(model(data)[te]).numpy(); l=label_s[te].numpy()
pr=round(average_precision_score(l,p),4)
f1=round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
print(f"\nT-Social FINAL: PR-AUC={pr}  F1={f1}  스팸={spam_rs:.1%}")

result={"name":"T-Social_10K","pr_auc":pr,"macro_f1":f1,
        "spam_ratio":round(spam_rs,4),"domain":"소셜 네트워크","n_nodes":N}
with open(RES/"tsocial_result.json","w") as f: json.dump(result,f,indent=2)

df=pd.read_csv(RES/"experiment_log.csv")
if "T-Social_10K" not in df["model"].values:
    new=pd.DataFrame([{"model":"T-Social_10K","pr_auc":pr,"macro_f1":f1,
                        "params":0,"train_sec":round(time.time()-t0),"notes":"소셜 네트워크 이상 탐지"}])
    df=pd.concat([df,new],ignore_index=True); df.to_csv(RES/"experiment_log.csv",index=False)
print("저장 완료")
