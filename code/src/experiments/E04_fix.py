"""
E04_fix.py
E04 재실험: 50K + 60/20/20 올바른 분할

버그 수정: n_tr=40K(80%)로 계산해 test가 0개였던 문제 수정
           60/20/20으로 Train=30K / Val=10K / Test=10K
"""
import copy, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing

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

def mask_ind(data, mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

def evaluate(model, data, mask):
    if mask.sum() == 0: return 0.0, 0.0
    model.eval()
    with torch.no_grad():
        p=torch.sigmoid(model(data)[mask]).numpy()
        l=data["review"].y[mask].numpy()
    if l.sum()==0: return 0.0, 0.0
    pr=average_precision_score(l,p)
    f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
    return round(pr,4), round(f1,4)

# ── 50K 그래프 로드 ───────────────────────────────────────────────────────────
print("="*65)
print("E04 Fix: 50K + 60/20/20 (올바른 분할)")
print("="*65)

torch.manual_seed(42)
data50 = torch.load(GRAPH/"hetero_graph_50k.pt", weights_only=False)
n50    = data50["review"].x.shape[0]
feat50 = data50["review"].x.shape[1]
ts50   = data50["review"].timestamp
idx50  = torch.argsort(ts50)
y50    = data50["review"].y

# ── 올바른 60/20/20 분할 ──────────────────────────────────────────────────────
n_tr = int(n50 * 0.60)  # 30,000
n_va = int(n50 * 0.20)  # 10,000
n_te = n50 - n_tr - n_va  # 10,000

tm50 = torch.zeros(n50, dtype=torch.bool); tm50[idx50[:n_tr]] = True
vm50 = torch.zeros(n50, dtype=torch.bool); vm50[idx50[n_tr:n_tr+n_va]] = True
te50 = torch.zeros(n50, dtype=torch.bool); te50[idx50[n_tr+n_va:]] = True

print(f"\n[분할 확인]")
print(f"  Train: {tm50.sum():,}  스팸={y50[tm50].float().mean():.3f}")
print(f"  Val:   {vm50.sum():,}  스팸={y50[vm50].float().mean():.3f}")
print(f"  Test:  {te50.sum():,}  스팸={y50[te50].float().mean():.3f}")

n_params = 387329
print(f"\n  파라미터/Train 비율: {n_params}/{n_tr} = {n_params/n_tr:.1f} (기존 16.1 → 12.9 개선)")

# ── 학습 ─────────────────────────────────────────────────────────────────────
model = HeteroBWGNN(feat50)
opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
crit  = FocalLoss()

best_val, best_state, no_imp = 0., None, 0
history = []; t0 = time.time()

print(f"\n  {'ep':>4}  {'tr_pr':>8}  {'val_pr':>8}  {'te_pr':>8}  {'gap':>8}")
print("  " + "-"*45)

for ep in range(1, 401):
    model.train(); opt.zero_grad()
    loss = crit(model(data50)[tm50], y50[tm50])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 50 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(model(data50)[tm50]).numpy()
        tr_pr = round(float(average_precision_score(y50[tm50].numpy(), p_tr)), 4)
        val_pr, _ = evaluate(model, data50, vm50)
        te_pr,  _ = evaluate(model, data50, te50)
        gap = round(tr_pr - te_pr, 4)
        print(f"  {ep:4d}  {tr_pr:8.4f}  {val_pr:8.4f}  {te_pr:8.4f}  {gap:+8.4f}")
        history.append({"ep":ep,"tr_pr":tr_pr,"val_pr":val_pr,"te_pr":te_pr,"gap":gap})

        if val_pr > best_val:
            best_val = val_pr
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 5:
                print(f"  Early stop ep={ep}")
                break

# 최종 평가
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(model(data50)[tm50]).numpy()
tr_f = round(float(average_precision_score(y50[tm50].numpy(), p_tr)), 4)
te_f, te_f1f = evaluate(model, data50, te50)
val_f, _ = evaluate(model, data50, vm50)
gap_f = round(tr_f - te_f, 4)

# 인덕티브 평가
data_ind = mask_ind(data50, te50)
ind_f, ind_f1f = evaluate(model, data_ind, te50)

print(f"\n{'='*65}")
print(f"[E04 최종] Train={tr_f:.4f}  Val={val_f:.4f}  Test={te_f:.4f}  Gap={gap_f:+.4f}  Inductive={ind_f:.4f}")
print(f"[기준 대비] Test: 0.9242→{te_f:.4f} ({te_f-0.9242:+.4f})  Gap: +0.071→{gap_f:+.4f}  Ind: 0.6526→{ind_f:.4f} ({ind_f-0.6526:+.4f})")

torch.save(best_state, MOD/"E04fix_BWGNN_50k_val_best.pt")
result = {
    "experiment":"E04_50k_60_20_20",
    "config":{"nodes":50000,"split":"60/20/20","early_stop":"val"},
    "n_params":n_params,"n_train":n_tr,"param_ratio":round(n_params/n_tr,1),
    "train_pr":tr_f,"val_pr":val_f,"test_pr":te_f,"test_f1":te_f1f,
    "gap":gap_f,"inductive_pr":ind_f,"inductive_f1":ind_f1f,
    "history":history,
}
with open(RES/"E04_fix_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"저장: results/experiments/E04_fix_result.json")
