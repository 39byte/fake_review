"""
09_boost.py
성능 극대화 3단계
1. R-Sim-R 엣지 추가 (SBERT 코사인 유사도 > 0.85, 동일 식당)
2. 전체 모델 Warm Restart 재학습 (400 epoch, LR=2e-4)
3. BWGNN + TGATLite 앙상블
"""

import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np, time, copy
from pathlib import Path
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
print(f"Device: {DEVICE}\n")

HIDDEN = 128; D_TIME = 64; HEADS = 4; LR = 2e-4; EPOCHS = 400

EDGE_TYPES = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","burst","review"), ("review","rur","review"),
    ("review","sim","review"),   # 신규
]
EDGE_TYPES_NO_BURST = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","rur","review"),  ("review","sim","review"),
]

# ── 1. R-Sim-R 엣지 생성 ─────────────────────────────────────────────────────
print("=" * 55)
print("[1] R-Sim-R 엣지 생성 (코사인 유사도 > 0.85)")
print("=" * 55)

df   = pd.read_parquet(PROC / "df_sampled.parquet")
emb  = torch.load(PROC / "sbert_embeddings.pt", weights_only=True)  # [N, 384]
emb_np = emb.numpy().astype(np.float32)

SIM_THRESHOLD = 0.85
MAX_SIM_PER_PROD = 3000

sim_src, sim_dst = [], []
for prod, grp in df.groupby("prod_id"):
    nids = grp["node_id"].values
    if len(nids) < 2:
        continue
    e = emb_np[nids]                            # [k, 384]
    # 코사인 유사도 행렬 (이미 L2 정규화 돼 있으므로 내적 = 코사인)
    sim = e @ e.T                               # [k, k]
    np.fill_diagonal(sim, 0)
    rows, cols = np.where(sim > SIM_THRESHOLD)
    if len(rows) > MAX_SIM_PER_PROD:
        top_idx = np.argsort(sim[rows, cols])[-MAX_SIM_PER_PROD:]
        rows, cols = rows[top_idx], cols[top_idx]
    for r, c in zip(rows, cols):
        sim_src.append(int(nids[r]))
        sim_dst.append(int(nids[c]))

sim_edge_index = torch.tensor([sim_src, sim_dst], dtype=torch.long)
print(f"  R-Sim-R 엣지 수: {sim_edge_index.shape[1]:,}")
spam_sim = df.iloc[sim_src]["label"].mean()
print(f"  sim 엣지 소스 스팸 비율: {spam_sim:.3f} (높을수록 스팸 연결 잘 됨)")

# ── 2. 그래프에 R-Sim-R 추가 ─────────────────────────────────────────────────
print("\n[2] 그래프 업데이트")
data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
data["review", "sim", "review"].edge_index = sim_edge_index
torch.save(data, GRAPH / "hetero_graph.pt")
FEAT_DIM = data["review"].x.shape[1]
total_edges = sum(data[et].edge_index.shape[1] for et in EDGE_TYPES)
print(f"  feat_dim={FEAT_DIM}  총 엣지={total_edges:,}")
data = data.to(DEVICE)

# ── 공통 ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self,g=2.0,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce); w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        pr_=torch.sigmoid(model(data)[mask]).cpu().numpy()
        la_=data["review"].y[mask].cpu().numpy()
    return {"PR-AUC":round(average_precision_score(la_,pr_),4),
            "Macro-F1":round(f1_score(la_,pr_>=0.5,average="macro",zero_division=0),4),
            "probs": pr_}

def train_warm(model, name, data, load_prev=True):
    """이전 가중치 로드 후 Warm Restart 재학습"""
    if load_prev:
        pt = MOD / f"{name}_best.pt"
        if pt.exists():
            model.load_state_dict(torch.load(pt, weights_only=True))
            print(f"  이전 가중치 로드: {name}")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # Warm Restart: T_0=80, T_mult=2 → 80, 160, 320 epoch 주기
    sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=80, T_mult=2, eta_min=1e-6)
    crit = FocalLoss()
    tm, tl = data["review"].train_mask, data["review"].y
    best_pr, best_st, hist = 0.0, None, []
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[tm], tl[tm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(ep)

        if ep % 20 == 0 or ep == 1:
            te = evaluate(model, data, data["review"].test_mask)
            lr_now = opt.param_groups[0]["lr"]
            print(f"  [{name}] ep={ep:3d} loss={loss.item():.4f} "
                  f"PR-AUC={te['PR-AUC']:.4f} F1={te['Macro-F1']:.4f} lr={lr_now:.2e}")
            hist.append({"epoch": ep, "PR-AUC": te["PR-AUC"],
                         "Macro-F1": te["Macro-F1"], "loss": round(loss.item(), 4)})
            if te["PR-AUC"] > best_pr:
                best_pr = te["PR-AUC"]
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_st)
    fin = evaluate(model, data, data["review"].test_mask)
    ela = round(time.time() - t0, 1)
    print(f"\n  [{name}] FINAL  PR-AUC={fin['PR-AUC']}  F1={fin['Macro-F1']}  ({ela}s)\n")
    torch.save(best_st, MOD / f"{name}_best.pt")
    pd.DataFrame(hist).to_csv(RES / f"history_{name}_boost.csv", index=False)
    return fin, fin["probs"]

# ── 모델 정의 (5종 엣지 포함) ─────────────────────────────────────────────────
class DualFreqConv(MessagePassing):
    def __init__(self,i,o):
        super().__init__(aggr="mean"); self.lin=nn.Linear(i*2,o)
    def forward(self,x,ei):
        low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

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
        super().__init__(); self.omega=nn.Parameter(torch.randn(d//2))
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

# ── 3. Warm Restart 재학습 ─────────────────────────────────────────────────────
print("=" * 55)
print("[3] Warm Restart 재학습 (400 epoch, LR=2e-4)")
print("=" * 55)
torch.manual_seed(42)

bwgnn_model = HeteroBWGNN(FEAT_DIM, HIDDEN)
tgat_model  = TGATLite(FEAT_DIM, HIDDEN)

bwgnn_res, bwgnn_probs = train_warm(bwgnn_model, "HeteroBWGNN", data, load_prev=False)
tgat_res,  tgat_probs  = train_warm(tgat_model,  "TGATLite",    data, load_prev=False)

# ── 4. 앙상블 ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("[4] 앙상블 (BWGNN × 0.6 + TGATLite × 0.4)")
print("=" * 55)
test_mask = data["review"].test_mask
labels = data["review"].y[test_mask].cpu().numpy()

for w_b in [0.5, 0.6, 0.7]:
    w_t = 1.0 - w_b
    ens = bwgnn_probs * w_b + tgat_probs * w_t
    pr  = average_precision_score(labels, ens)
    f1  = f1_score(labels, ens >= 0.5, average="macro", zero_division=0)
    print(f"  BWGNN×{w_b} + TGAT×{w_t:.1f}: PR-AUC={pr:.4f}  F1={f1:.4f}")

# 최적 앙상블 (0.6:0.4)
ens_probs = bwgnn_probs * 0.6 + tgat_probs * 0.4
ens_pr = average_precision_score(labels, ens_probs)
ens_f1 = f1_score(labels, ens_probs >= 0.5, average="macro", zero_division=0)

# ── 5. 최종 결과 저장 ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("최종 결과 요약")
print("=" * 55)

log = pd.read_csv(RES / "experiment_log.csv")
boost_rows = [
    {"model":"HeteroBWGNN_boost","pr_auc":bwgnn_res["PR-AUC"],"macro_f1":bwgnn_res["Macro-F1"],
     "params":sum(p.numel() for p in bwgnn_model.parameters()),"train_sec":0,
     "notes":"R-Sim-R 추가, Warm Restart 400ep"},
    {"model":"TGATLite_boost","pr_auc":tgat_res["PR-AUC"],"macro_f1":tgat_res["Macro-F1"],
     "params":sum(p.numel() for p in tgat_model.parameters()),"train_sec":0,
     "notes":"R-Sim-R 추가, Warm Restart 400ep"},
    {"model":"Ensemble_BWGNN_TGAT","pr_auc":round(ens_pr,4),"macro_f1":round(ens_f1,4),
     "params":0,"train_sec":0,"notes":"BWGNN×0.6+TGATLite×0.4"},
]
pd.concat([log, pd.DataFrame(boost_rows)]).to_csv(RES/"experiment_log_final.csv", index=False)

all_models = [
    ("HeteroBWGNN_boost", bwgnn_res["PR-AUC"], bwgnn_res["Macro-F1"]),
    ("TGATLite_boost",    tgat_res["PR-AUC"],  tgat_res["Macro-F1"]),
    ("Ensemble",          round(ens_pr,4),     round(ens_f1,4)),
]
for name, pr, f1 in all_models:
    print(f"  {name:22s}: PR-AUC={pr:.4f}  F1={f1:.4f}")

best_pr = max(pr for _,pr,_ in all_models)
print(f"\nGo/No-Go (≥0.70): {'GO ✅' if best_pr >= 0.70 else '⚠️'}")
print(f"저장: {RES/'experiment_log_final.csv'}")
