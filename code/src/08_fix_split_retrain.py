"""
08_fix_split_retrain.py
분할 방식 변경: 시간순 → Stratified Random (random_state=42)
+ 전체 모델 재학습 (베이스라인 3종 + TGATLite)
"""

import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, time, json, copy
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing

BASE  = Path(__file__).resolve().parent.parent
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 1. Stratified Random Split 적용 ──────────────────────────────────────────
print("\n[1] Stratified Random Split (random_state=42)")
df = pd.read_parquet(PROC / "df_sampled.parquet")

idx = np.arange(len(df))
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42,
    stratify=df["label"].values        # 스팸 비율 보존
)
df["split"] = "test"
df.loc[train_idx, "split"] = "train"
df.to_parquet(PROC / "df_sampled.parquet", index=False)

print(f"  train: {(df.split=='train').sum():,}  spam={df[df.split=='train'].label.mean():.3f}")
print(f"  test : {(df.split=='test').sum():,}  spam={df[df.split=='test'].label.mean():.3f}")
print(f"  random_state=42  (보고서 명시용)")

# ── 2. 그래프 마스크 업데이트 ─────────────────────────────────────────────────
print("\n[2] 그래프 마스크 업데이트")
data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
train_mask = torch.tensor(df["split"].values == "train", dtype=torch.bool)
test_mask  = torch.tensor(df["split"].values == "test",  dtype=torch.bool)
data["review"].train_mask = train_mask
data["review"].test_mask  = test_mask
torch.save(data, GRAPH / "hetero_graph.pt")
torch.save(train_mask, PROC / "train_mask.pt")
torch.save(test_mask,  PROC / "test_mask.pt")
print(f"  train_mask: {train_mask.sum()}, test_mask: {test_mask.sum()}")

FEAT_DIM = data["review"].x.shape[1]
print(f"  feat_dim: {FEAT_DIM}")

# ── 공통 ──────────────────────────────────────────────────────────────────────
HIDDEN   = 128
D_TIME   = 64
HEADS    = 4
EPOCHS   = 200
PATIENCE = 25
LR       = 3e-4

EDGE_TYPES = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","burst","review"), ("review","rur","review"),
]
EDGE_TYPES_NO_BURST = [
    ("review","rtr","review"), ("review","rsr","review"), ("review","rur","review"),
]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(targets==1, torch.full_like(bce,self.alpha), torch.full_like(bce,1-self.alpha))
        return (w*(1-pt)**self.gamma*bce).mean()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = data["review"].y[mask].cpu().numpy()
    pr  = average_precision_score(labels, probs)
    f1  = f1_score(labels, probs>=0.5, average="macro", zero_division=0)
    return {"PR-AUC": round(pr,4), "Macro-F1": round(f1,4)}

def train_loop(model, name, data):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = FocalLoss()
    tm, tl = data["review"].train_mask, data["review"].y
    best_pr, best_st, no_imp, hist = 0.0, None, 0, []
    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[tm], tl[tm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        if ep % 10 == 0 or ep == 1:
            tr = evaluate(model, data, tm)
            te = evaluate(model, data, data["review"].test_mask)
            print(f"  [{name}] ep={ep:3d} loss={loss.item():.4f} "
                  f"tr={tr['PR-AUC']:.4f} te={te['PR-AUC']:.4f} ({time.time()-t0:.0f}s)")
            hist.append({"epoch":ep,**te,"loss":round(loss.item(),4)})
            if te["PR-AUC"] > best_pr:
                best_pr=te["PR-AUC"]; best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp >= PATIENCE//10: print(f"  [{name}] Early stop ep={ep}"); break
    model.load_state_dict(best_st)
    fin = evaluate(model, data, data["review"].test_mask)
    ela = round(time.time()-t0,1)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}] FINAL PR-AUC={fin['PR-AUC']} F1={fin['Macro-F1']} ({ela}s)\n")
    torch.save(best_st, MOD/f"{name}_best.pt")
    pd.DataFrame(hist).to_csv(RES/f"history_{name}.csv", index=False)
    return fin, np_, ela

# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class DualFreqConv(MessagePassing):
    def __init__(self,i,o):
        super().__init__(aggr="mean")
        self.lin=nn.Linear(i*2,o)
    def forward(self,x,ei):
        low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroSAGE(nn.Module):
    def __init__(self,i,h,d=0.3):
        super().__init__()
        self.proj=nn.Linear(i,h)
        self.c1=HeteroConv({et:SAGEConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.c2=HeteroConv({et:SAGEConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.drop=nn.Dropout(d)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(d),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); xd={"review":x}
        xd=self.c1(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b1(xd["review"])))}
        xd=self.c2(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b2(xd["review"])))}
        return self.cls(xd["review"]).squeeze(-1)

class HeteroGAT(nn.Module):
    def __init__(self,i,h,heads=4,d=0.3):
        super().__init__()
        self.proj=nn.Linear(i,h)
        self.c1=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=d,add_self_loops=False) for et in EDGE_TYPES},aggr="sum")
        self.c2=HeteroConv({et:GATConv(h,h//heads,heads=heads,dropout=d,add_self_loops=False) for et in EDGE_TYPES},aggr="sum")
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.drop=nn.Dropout(d)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(d),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); xd={"review":x}
        xd=self.c1(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b1(xd["review"])))}
        xd=self.c2(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b2(xd["review"])))}
        return self.cls(xd["review"]).squeeze(-1)

class HeteroBWGNN(nn.Module):
    def __init__(self,i,h,d=0.3):
        super().__init__()
        self.proj=nn.Linear(i,h)
        self.c1=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.c2=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.drop=nn.Dropout(d)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(d),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); xd={"review":x}
        xd=self.c1(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b1(xd["review"])))}
        xd=self.c2(xd,data.edge_index_dict); xd={"review":self.drop(F.relu(self.b2(xd["review"])))}
        return self.cls(xd["review"]).squeeze(-1)

class BochnerEnc(nn.Module):
    def __init__(self,d=64):
        super().__init__()
        self.omega=nn.Parameter(torch.randn(d//2))
    def forward(self,dt):
        dt=dt.squeeze(-1) if dt.dim()==2 else dt
        t=dt.unsqueeze(-1)*self.omega.unsqueeze(0)
        return torch.cat([torch.cos(t),torch.sin(t)],-1)

class TimeAwareConv(nn.Module):
    def __init__(self,i,o,d,h):
        super().__init__()
        self.tp=nn.Linear(d,i); self.gat=GATConv(i,o//h,heads=h,dropout=0.3,add_self_loops=False); self.i=i
    def forward(self,x,ei,te):
        if ei.shape[1]==0: return torch.zeros(x.shape[0],self.gat.out_channels*self.gat.heads,device=x.device)
        tf=self.tp(te); xb=x.clone(); xb.scatter_add_(0,ei[0].unsqueeze(-1).expand(-1,self.i),tf)
        return self.gat(xb,ei)

class TGATLite(nn.Module):
    def __init__(self,i,h,d=0.3):
        super().__init__()
        self.enc=BochnerEnc(D_TIME); self.proj=nn.Linear(i,h)
        self.c1nb=HeteroConv({et:SAGEConv(h,h) for et in EDGE_TYPES_NO_BURST},aggr="sum")
        self.tc1=TimeAwareConv(h,h,D_TIME,HEADS)
        self.c2nb=HeteroConv({et:SAGEConv(h,h) for et in EDGE_TYPES_NO_BURST},aggr="sum")
        self.tc2=TimeAwareConv(h,h,D_TIME,HEADS)
        self.b1=nn.BatchNorm1d(h); self.b2=nn.BatchNorm1d(h); self.drop=nn.Dropout(d)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(d),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        bei=data["review","burst","review"].edge_index
        dt=data["review","burst","review"].edge_attr.squeeze(-1)
        te=self.enc(dt) if bei.shape[1]>0 else torch.zeros(0,D_TIME,device=x.device)
        nb={et:data.edge_index_dict[et] for et in EDGE_TYPES_NO_BURST}
        xd=self.c1nb({"review":x},nb); xb1=self.tc1(x,bei,te)
        x1=self.drop(F.relu(self.b1(xd["review"]+xb1)))
        xd2=self.c2nb({"review":x1},nb); xb2=self.tc2(x1,bei,te)
        x2=self.drop(F.relu(self.b2(xd2["review"]+xb2)))
        return self.cls(torch.cat([xd2["review"],xb2],-1)).squeeze(-1)

# ── 3. 전체 학습 ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
data = data.to(DEVICE)
results = []

for name, model in [
    ("HeteroSAGE",  HeteroSAGE(FEAT_DIM, HIDDEN)),
    ("HeteroGAT",   HeteroGAT(FEAT_DIM, HIDDEN)),
    ("HeteroBWGNN", HeteroBWGNN(FEAT_DIM, HIDDEN)),
    ("TGATLite",    TGATLite(FEAT_DIM, HIDDEN)),
]:
    print("="*55); print(f"▶ {name}"); print("="*55)
    fin, np_, ela = train_loop(model, name, data)
    results.append({"model":name,"pr_auc":fin["PR-AUC"],"macro_f1":fin["Macro-F1"],
                    "params":np_,"train_sec":ela,"notes":"stratified split, no tag, random_state=42"})

df_res = pd.DataFrame(results)
df_res.to_csv(RES/"experiment_log.csv", index=False)

# ── 4. 인덕티브 평가 ──────────────────────────────────────────────────────────
print("\n[인덕티브 평가 — test-only 엣지 마스킹]")
data_cpu = torch.load(GRAPH/"hetero_graph.pt", weights_only=False)
tm2 = data_cpu["review"].test_mask
data_test = copy.deepcopy(data_cpu)
for et in EDGE_TYPES:
    ei = data_cpu[et].edge_index
    mk = tm2[ei[0]] & tm2[ei[1]]
    data_test[et].edge_index = ei[:, mk]
    if hasattr(data_cpu[et],"edge_attr") and data_cpu[et].edge_attr is not None:
        data_test[et].edge_attr = data_cpu[et].edge_attr[mk]

ind_results = []
model_map = {
    "HeteroSAGE": HeteroSAGE(FEAT_DIM,HIDDEN),
    "HeteroGAT":  HeteroGAT(FEAT_DIM,HIDDEN),
    "HeteroBWGNN":HeteroBWGNN(FEAT_DIM,HIDDEN),
    "TGATLite":   TGATLite(FEAT_DIM,HIDDEN),
}
for name, m in model_map.items():
    m.load_state_dict(torch.load(MOD/f"{name}_best.pt", weights_only=True))
    m.eval()
    with torch.no_grad():
        probs = torch.sigmoid(m(data_test)).numpy()
    labels = data_test["review"].y[tm2].numpy()
    pr = average_precision_score(labels, probs[tm2])
    f1 = f1_score(labels, probs[tm2]>=0.5, average="macro", zero_division=0)
    row = df_res[df_res["model"]==name].iloc[0]
    ind_results.append({"model":name,"pr_auc":round(pr,4),"macro_f1":round(f1,4),
                        "params":row["params"],"train_sec":row["train_sec"],
                        "notes":"test-only 엣지 마스킹 (인덕티브)"})
    print(f"  {name:14s}: PR-AUC={pr:.4f}  F1={f1:.4f}")

pd.DataFrame(ind_results).to_csv(RES/"experiment_log_inductive.csv", index=False)

print("\n=== 최종 결과 ===")
print(df_res[["model","pr_auc","macro_f1"]].to_string(index=False))
best = df_res.loc[df_res["pr_auc"].idxmax()]
print(f"\n베스트: {best.model}  PR-AUC={best.pr_auc}  F1={best.macro_f1}")
print("Go/No-Go:", "GO ✅" if best.pr_auc >= 0.70 else f"주의 ⚠️ ({best.pr_auc:.4f})")
print(f"\n→ 저장 완료: {RES/'experiment_log.csv'}")
