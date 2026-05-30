"""
E01_30k_val_split.py
실험 1: 30K + Validation Set 추가 (70 / 15 / 15)

가설: Val 기반 Early stopping으로 Test leakage 제거 → Gap 감소
비교 기준: DRAGWave_400ep (Trans 0.934, Gap 0.066)

변경사항:
  - 기존 80/20 → 70/15/15 (Train/Val/Test)
  - Early stopping: val_mask 기준
  - 최종 평가: test_mask 기준
"""
import copy, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results" / "experiments"
RES.mkdir(parents=True, exist_ok=True)

ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def mask_inductive(data, mask):
    import copy
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(data)[mask]).numpy()
        l=data["review"].y[mask].numpy()
    pr=average_precision_score(l,p)
    f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
    return round(pr,4), round(f1,4)

# ── 데이터 로드 & 70/15/15 재분할 ────────────────────────────────────────────
print("="*65)
print("E01: 30K + Val (70/15/15) — Test Leakage 제거 실험")
print("="*65)

torch.manual_seed(42)
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
n = data["review"].x.shape[0]
feat = data["review"].x.shape[1]
ts = data["review"].timestamp
labels = data["review"].y

# 시간순 정렬 후 70/15/15 분할
sorted_idx = torch.argsort(ts)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)

train_mask[sorted_idx[:n_train]] = True
val_mask[sorted_idx[n_train:n_train+n_val]] = True
test_mask[sorted_idx[n_train+n_val:]] = True

print(f"\n[분할 결과]")
print(f"  Train: {train_mask.sum():,}  스팸={labels[train_mask].float().mean():.3f}")
print(f"  Val:   {val_mask.sum():,}    스팸={labels[val_mask].float().mean():.3f}")
print(f"  Test:  {test_mask.sum():,}   스팸={labels[test_mask].float().mean():.3f}")
print(f"  파라미터/노드 비율: 321537/{train_mask.sum().item()} = {321537/train_mask.sum().item():.1f}")

# ── 학습 ─────────────────────────────────────────────────────────────────────
model = HeteroBWGNN(feat)
opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
crit  = FocalLoss()

best_val_pr, best_state, no_imp = 0., None, 0
history, t0 = [], time.time()

print(f"\n[학습] 400 epoch, Early stopping 기준: Val PR-AUC")
print(f"  {'ep':>4}  {'train_pr':>9}  {'val_pr':>8}  {'test_pr':>8}  {'gap(tr-te)':>10}")
print("  " + "-"*50)

for ep in range(1, 401):
    model.train(); opt.zero_grad()
    loss = crit(model(data)[train_mask], labels[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 40 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            logits = model(data)
            p_tr = torch.sigmoid(logits[train_mask]).numpy()
            l_tr = labels[train_mask].numpy()
        train_pr = round(float(average_precision_score(l_tr, p_tr)), 4)
        val_pr,  _ = evaluate(model, data, val_mask)
        test_pr, test_f1 = evaluate(model, data, test_mask)
        gap = round(train_pr - test_pr, 4)
        print(f"  {ep:4d}  {train_pr:9.4f}  {val_pr:8.4f}  {test_pr:8.4f}  {gap:+10.4f}")
        history.append({"epoch":ep,"train_pr":train_pr,"val_pr":val_pr,
                        "test_pr":test_pr,"gap":gap})

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state  = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 5:
                print(f"  Early stop at ep={ep} (val patience=5)")
                break

# ── 최종 평가 ─────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(model(data)[train_mask]).numpy()
train_pr_f = round(float(average_precision_score(labels[train_mask].numpy(), p_tr)), 4)
test_pr_f, test_f1_f = evaluate(model, data, test_mask)
val_pr_f,  val_f1_f  = evaluate(model, data, val_mask)
gap_f = round(train_pr_f - test_pr_f, 4)

# 인덕티브 평가
data_ind = mask_inductive(data, test_mask)
ind_pr_f, ind_f1_f = evaluate(model, data_ind, test_mask)

print(f"\n{'='*65}")
print(f"[E01 최종 결과] 30K + Val (70/15/15)")
print(f"{'='*65}")
print(f"  Train PR-AUC : {train_pr_f:.4f}")
print(f"  Val   PR-AUC : {val_pr_f:.4f}")
print(f"  Test  PR-AUC : {test_pr_f:.4f}  F1={test_f1_f:.4f}")
print(f"  Gap (tr-te)  : {gap_f:+.4f}")
print(f"  Inductive    : {ind_pr_f:.4f}")
print()
print(f"  [기준 대비]")
print(f"  Test PR-AUC : 0.9242 → {test_pr_f:.4f}  ({test_pr_f-0.9242:+.4f})")
print(f"  Gap         : +0.0710 → {gap_f:+.4f}  ({gap_f-0.071:+.4f})")
print(f"  Inductive   : 0.6526 → {ind_pr_f:.4f}  ({ind_pr_f-0.6526:+.4f})")

torch.save(best_state, MOD/"E01_BWGNN_30k_val_best.pt")
result = {
    "experiment": "E01_30k_val_split",
    "config": {"nodes":30000, "split":"70/15/15", "early_stop":"val"},
    "train_pr": train_pr_f, "val_pr": val_pr_f,
    "test_pr": test_pr_f, "test_f1": test_f1_f,
    "gap": gap_f, "inductive_pr": ind_pr_f, "inductive_f1": ind_f1_f,
    "history": history,
}
with open(RES/"E01_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"\n저장: results/experiments/E01_result.json")
