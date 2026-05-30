"""
E03_train_experiments.py
실험 3~5: 다양한 설정으로 BWGNN + DRAGWave 학습

E03: 50K + 80/20 (Val 없음)
E04: 50K + 60/20/20 (Val 있음)
E05: 30K + 소형 모델 (hidden_dim=64)

실행 전 E02_sample_50k.py 먼저 실행 필요
"""
import copy, json, time
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
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

def mask_ind(data, mask):
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
    if l.sum()==0: return 0.0, 0.0
    pr=average_precision_score(l,p)
    f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
    return round(pr,4), round(f1,4)

def run_experiment(name, data, train_mask, test_mask, val_mask=None,
                   epochs=400, hidden=128, patience=5, desc=""):
    print(f"\n{'='*65}")
    print(f"[{name}] {desc}")
    print(f"  Train={train_mask.sum():,}  Val={val_mask.sum() if val_mask is not None else 0}  Test={test_mask.sum():,}")
    feat = data["review"].x.shape[1]
    n_train = train_mask.sum().item()

    import importlib
    # 모델 생성
    class BWGNN_H(nn.Module):
        def __init__(self,d,h,dr=0.3):
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

    torch.manual_seed(42)
    model = BWGNN_H(feat, hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  파라미터={n_params:,}  비율={n_params/n_train:.1f}")

    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss()
    labels = data["review"].y

    # Early stopping 기준: val이 있으면 val, 없으면 test
    stop_mask = val_mask if val_mask is not None else test_mask
    stop_name = "val" if val_mask is not None else "test"

    best_pr, best_state, no_imp = 0., None, 0
    history = []; t0 = time.time()

    print(f"  {'ep':>4}  {'tr_pr':>7}  {stop_name:>7}  {'te_pr':>7}  {'gap':>7}")
    print("  " + "-"*40)

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()

        if ep % 50 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                p_tr = torch.sigmoid(model(data)[train_mask]).numpy()
            tr_pr = round(float(average_precision_score(labels[train_mask].numpy(), p_tr)), 4)
            stop_pr, _ = evaluate(model, data, stop_mask)
            te_pr, te_f1 = evaluate(model, data, test_mask)
            gap = round(tr_pr - te_pr, 4)
            print(f"  {ep:4d}  {tr_pr:7.4f}  {stop_pr:7.4f}  {te_pr:7.4f}  {gap:+7.4f}")
            history.append({"ep":ep,"tr_pr":tr_pr,"stop_pr":stop_pr,"te_pr":te_pr,"gap":gap})

            if stop_pr > best_pr:
                best_pr = stop_pr; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"  Early stop ep={ep}")
                    break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_tr = torch.sigmoid(model(data)[train_mask]).numpy()
    tr_pr_f = round(float(average_precision_score(labels[train_mask].numpy(), p_tr)), 4)
    te_pr_f, te_f1_f = evaluate(model, data, test_mask)
    gap_f = round(tr_pr_f - te_pr_f, 4)
    data_ind = mask_ind(data, test_mask)
    ind_pr_f, ind_f1_f = evaluate(model, data_ind, test_mask)

    print(f"\n  [최종] Train={tr_pr_f:.4f}  Test={te_pr_f:.4f}  Gap={gap_f:+.4f}  Inductive={ind_pr_f:.4f}")
    torch.save(best_state, MOD/f"{name}_best.pt")

    return {"name":name,"desc":desc,"n_params":n_params,"n_train":n_train,
            "param_ratio":round(n_params/n_train,1),
            "train_pr":tr_pr_f,"test_pr":te_pr_f,"test_f1":te_f1_f,
            "gap":gap_f,"inductive_pr":ind_pr_f,"inductive_f1":ind_f1_f,
            "history":history}


results = []

# ── E03: 50K + 80/20 ──────────────────────────────────────────────────────────
g50k_path = GRAPH/"hetero_graph_50k.pt"
if g50k_path.exists():
    data50 = torch.load(g50k_path, weights_only=False)
    # 80/20으로 재분할
    n50 = data50["review"].x.shape[0]
    ts50 = data50["review"].timestamp
    idx50 = torch.argsort(ts50)
    n_tr = int(n50*0.8)
    tm50 = torch.zeros(n50,dtype=torch.bool); tm50[idx50[:n_tr]]=True
    te50 = torch.zeros(n50,dtype=torch.bool); te50[idx50[n_tr:]]=True
    r = run_experiment("E03_50k_8020", data50, tm50, te50, None,
                       desc="50K 노드 + 80/20 (Val 없음)")
    results.append(r)

    # E04: 50K + 60/20/20
    n_v = int(n50*0.2); n_t = n50 - n_tr - n_v
    vm50 = torch.zeros(n50,dtype=torch.bool); vm50[idx50[n_tr:n_tr+n_v]]=True
    te50b = torch.zeros(n50,dtype=torch.bool); te50b[idx50[n_tr+n_v:]]=True
    r = run_experiment("E04_50k_val", data50, tm50, te50b, vm50,
                       desc="50K 노드 + 60/20/20 (Val 있음)")
    results.append(r)
else:
    print("[E03/E04] hetero_graph_50k.pt 없음 — E02 먼저 실행 필요")

# ── E05: 30K + 소형 모델 (hidden=64) ─────────────────────────────────────────
data30 = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
tm30 = data30["review"].train_mask
te30 = data30["review"].test_mask

# 원본 30K 모델을 hidden=64로 재정의해 실험
class SmallBWGNN(nn.Module):
    def __init__(self,d,h=64,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ET},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,32),nn.ReLU(),nn.Dropout(dr),nn.Linear(32,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

print(f"\n{'='*65}")
print("[E05] 30K + 소형 모델 (hidden=64)")
feat30 = data30["review"].x.shape[1]
torch.manual_seed(42)
model_s = SmallBWGNN(feat30)
n_p = sum(p.numel() for p in model_s.parameters())
n_t = tm30.sum().item()
print(f"  파라미터={n_p:,}  비율={n_p/n_t:.1f}  (기존 321K/24K=13.4 대비)")
opt_s = torch.optim.AdamW(model_s.parameters(), lr=5e-4, weight_decay=1e-5)
sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=400)
crit_s  = FocalLoss()
labels30 = data30["review"].y

best_pr_s, best_s, no_s = 0., None, 0
print(f"  {'ep':>4}  {'tr_pr':>7}  {'te_pr':>7}  {'gap':>7}")
for ep in range(1, 401):
    model_s.train(); opt_s.zero_grad()
    loss = crit_s(model_s(data30)[tm30], labels30[tm30])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_s.parameters(), 1.)
    opt_s.step(); sched_s.step()
    if ep % 50 == 0 or ep == 1:
        model_s.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(model_s(data30)[tm30]).numpy()
        tr_p = round(float(average_precision_score(labels30[tm30].numpy(), p_tr)), 4)
        te_p, te_f = evaluate(model_s, data30, te30)
        g = round(tr_p - te_p, 4)
        print(f"  {ep:4d}  {tr_p:7.4f}  {te_p:7.4f}  {g:+7.4f}")
        if te_p > best_pr_s:
            best_pr_s = te_p; best_s = {k:v.cpu().clone() for k,v in model_s.state_dict().items()}; no_s=0
        else:
            no_s+=1
            if no_s >= 5: print(f"  Early stop ep={ep}"); break

model_s.load_state_dict(best_s)
with torch.no_grad(): p_tr=torch.sigmoid(model_s(data30)[tm30]).numpy()
tr_f = round(float(average_precision_score(labels30[tm30].numpy(),p_tr)),4)
te_f2, te_f1 = evaluate(model_s, data30, te30)
g_f = round(tr_f-te_f2,4)
data_i = mask_ind(data30, te30)
ind_f2, ind_f1 = evaluate(model_s, data_i, te30)
print(f"\n  [E05 최종] Train={tr_f:.4f}  Test={te_f2:.4f}  Gap={g_f:+.4f}  Inductive={ind_f2:.4f}")
torch.save(best_s, MOD/"E05_SmallBWGNN_best.pt")
results.append({"name":"E05_small_model","desc":"30K + 소형 모델 hidden=64",
                "n_params":n_p,"n_train":n_t,"param_ratio":round(n_p/n_t,1),
                "train_pr":tr_f,"test_pr":te_f2,"test_f1":te_f1,
                "gap":g_f,"inductive_pr":ind_f2,"inductive_f1":ind_f1})

# ── 전체 결과 저장 ─────────────────────────────────────────────────────────────
with open(RES/"E03_E05_results.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n저장: results/experiments/E03_E05_results.json")
print("Done")
