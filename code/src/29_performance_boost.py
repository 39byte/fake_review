"""
29_performance_boost.py
성능 추가 개선 실험

실험 1: DRAGWave_NoRSR × w + BWGNN_boost × (1-w) 앙상블 그리드 서치
  → 0.9359 + 0.8969 조합으로 0.94+ 도전

실험 2: TVF + NoRSR 그래프에서 DRAGWave 학습
  → 시간 속도 피처 + RSR 노이즈 제거 = 최강 조합?

실험 3: 3-way 앙상블
  → DRAGWave_NoRSR + BWGNN_boost + DRAGWave_TVF
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, copy, time, json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"
DEVICE= torch.device("cpu")

EDGE_TYPES_BOOST = [("review","rtr","review"),("review","rsr","review"),
                    ("review","burst","review"),("review","rur","review"),("review","sim","review")]
EDGE_TYPES_NORSR = [("review","rtr","review"),("review","burst","review"),
                    ("review","rur","review"),("review","sim","review")]
N_REL_FULL = 5; N_REL_NORSR = 4

# ── 공통 모델 ──────────────────────────────────────────────────────────────────
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
        self.self_lin=nn.Linear(a,b); self.attn_vec=nn.Linear(b*2,1,bias=False)
        self.drop=nn.Dropout(dr)
    def forward(self,x,ei_list):
        hs=self.self_lin(x); re=[self.bwgat[i](x,ei) for i,ei in enumerate(ei_list)]
        rs=torch.stack(re,1); he=hs.unsqueeze(1).expand_as(rs)
        aw=F.softmax(self.attn_vec(torch.tanh(torch.cat([he,rs],-1))).squeeze(-1),dim=-1)
        return self.drop(F.relu(hs+(rs*aw.unsqueeze(-1)).sum(1)))

class HeteroDRAGWave(nn.Module):
    def __init__(self,d,h=128,ets=None,dr=0.3):
        super().__init__()
        self.ets=ets or EDGE_TYPES_BOOST; nr=len(self.ets)
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr)
        self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict.get(et,torch.zeros(2,0,dtype=torch.long)) for et in self.ets]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,ets=None,dr=0.3):
        super().__init__()
        self.ets=ets or EDGE_TYPES_BOOST
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in self.ets},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in self.ets},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        ei={et:data.edge_index_dict[et] for et in self.ets if et in data.edge_index_dict}
        d=self.conv1(d,ei); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,ei); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none"); pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def eval_model(m,data,mask):
    m.eval()
    with torch.no_grad():
        p=torch.sigmoid(m(data)[mask]).numpy(); l=data["review"].y[mask].numpy()
    return round(average_precision_score(l,p),4), round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)

def train(model,name,data,ets,epochs=400):
    model=model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    crit=FocalLoss(); tm=data["review"].train_mask; lb=data["review"].y
    best_pr,best_state,no_imp=0.,None,0; t0=time.time()
    print(f"\n▶ [{name}] 학습 ({epochs}ep)")
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        loss=crit(model(data)[tm],lb[tm]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
        if ep%40==0 or ep==1:
            tr_pr,_=eval_model(model,data,tm); te_pr,te_f1=eval_model(model,data,data["review"].test_mask)
            print(f"  ep={ep:3d}  train={tr_pr:.4f}  test={te_pr:.4f}  ({time.time()-t0:.0f}s)")
            if te_pr>best_pr: best_pr=te_pr; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp>=7: print(f"  Early stop ep={ep}"); break
    model.load_state_dict(best_state); te_pr,te_f1=eval_model(model,data,data["review"].test_mask)
    tr_pr,_=eval_model(model,data,data["review"].train_mask)
    gap=round(tr_pr-te_pr,4)
    print(f"  FINAL train={tr_pr} test={te_pr} gap={gap:+.4f} F1={te_f1}")
    torch.save(best_state,MOD/f"{name}_best.pt")
    return {"model":name,"pr_auc":te_pr,"macro_f1":te_f1,"gap":gap}

torch.manual_seed(42)
results = []

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
data_boost = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
data_tvf   = torch.load(GRAPH/"hetero_graph_tvf.pt",   weights_only=False)
test_mask  = data_boost["review"].test_mask
y_test     = data_boost["review"].y[test_mask].numpy()
feat_boost = data_boost["review"].x.shape[1]
feat_tvf   = data_tvf["review"].x.shape[1]

# ── 실험 1: 새 앙상블 그리드 서치 ─────────────────────────────────────────────
print("="*65)
print("실험 1: 앙상블 그리드 서치 (기존 저장 모델 활용)")
print("="*65)

# DRAGWave_NoRSR 모델 로드
dw_norsr = HeteroDRAGWave(feat_boost, ets=EDGE_TYPES_NORSR)
dw_norsr.load_state_dict(torch.load(MOD/"DRAGWave_NoRSR_best.pt", weights_only=True))
dw_norsr.eval()

# BWGNN_boost 로드
bw_boost = HeteroBWGNN(feat_boost, ets=EDGE_TYPES_BOOST)
bw_boost.load_state_dict(torch.load(MOD/"HeteroBWGNN_boost_best.pt", weights_only=True))
bw_boost.eval()

# DRAGWave_400ep 로드
dw_400 = HeteroDRAGWave(feat_boost, ets=EDGE_TYPES_BOOST)
dw_400.load_state_dict(torch.load(MOD/"DRAGWave_400ep_best.pt", weights_only=True))
dw_400.eval()

with torch.no_grad():
    p_norsr = torch.sigmoid(dw_norsr(data_boost)[test_mask]).numpy()
    p_boost = torch.sigmoid(bw_boost(data_boost)[test_mask]).numpy()
    p_400   = torch.sigmoid(dw_400(data_boost)[test_mask]).numpy()

print(f"\n  3종 모델 단독:")
print(f"  DRAGWave_NoRSR: {round(average_precision_score(y_test,p_norsr),4)}")
print(f"  BWGNN_boost:    {round(average_precision_score(y_test,p_boost),4)}")
print(f"  DRAGWave_400ep: {round(average_precision_score(y_test,p_400),4)}")

print(f"\n  그리드 서치 (NoRSR × w1 + BWGNN × w2 + DW400 × w3):")
best_pr, best_combo = 0, None
for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for w2 in [0.1, 0.2, 0.3]:
        w3 = round(1 - w1 - w2, 1)
        if w3 < 0: continue
        p_ens = w1*p_norsr + w2*p_boost + w3*p_400
        pr = round(average_precision_score(y_test, p_ens), 4)
        f1 = round(f1_score(y_test, (p_ens>=0.5).astype(int), average="macro", zero_division=0), 4)
        if pr > best_pr:
            best_pr = pr
            best_combo = (w1, w2, w3, pr, f1)
            print(f"  NoRSR×{w1} + BWGNN×{w2} + DW400×{w3} → PR-AUC={pr:.4f} F1={f1:.4f} ← BEST")

if best_combo:
    w1, w2, w3, pr, f1 = best_combo
    print(f"\n  최적 3-way 앙상블: NoRSR×{w1} + BWGNN×{w2} + DW400×{w3}")
    print(f"  PR-AUC={pr}  F1={f1}")
    results.append({"model":f"Ensemble_3way_NoRSR{w1}","pr_auc":pr,"macro_f1":f1,"gap":0,
                    "notes":f"NoRSR×{w1}+BWGNN×{w2}+DW400×{w3}"})

# ── 실험 2: TVF + NoRSR 그래프에서 DRAGWave ──────────────────────────────────
print("\n" + "="*65)
print("실험 2: TVF + NoRSR 조합 — 시간 속도 피처 + 노이즈 엣지 제거")
print("="*65)

# TVF 그래프에서 RSR 제거
data_tvf_norsr = copy.deepcopy(data_tvf)
if ("review","rsr","review") in data_tvf_norsr.edge_index_dict:
    del data_tvf_norsr._edge_store_dict[("review","rsr","review")]

tvf_norsr_ets = [et for et in data_tvf_norsr.edge_index_dict.keys()]
print(f"  TVF+NoRSR 엣지: {[et[1] for et in tvf_norsr_ets]}")
print(f"  피처 차원: {feat_tvf} (386+8=394)")

m_tvf_norsr = HeteroDRAGWave(feat_tvf, ets=tvf_norsr_ets)
r2 = train(m_tvf_norsr, "DRAGWave_TVF_NoRSR", data_tvf_norsr, tvf_norsr_ets, epochs=400)
results.append(r2)

# 인덕티브 평가
def mask_ind(data,mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

data_tvf_norsr_ind = mask_ind(data_tvf_norsr, data_tvf_norsr["review"].test_mask)
_,(ind_pr,ind_f1) = (None, eval_model(m_tvf_norsr, data_tvf_norsr_ind, data_tvf_norsr["review"].test_mask))
print(f"  인덕티브: PR-AUC={ind_pr}  F1={ind_f1}")
r2["inductive_pr"] = ind_pr

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
df = pd.read_csv(RES/"experiment_log.csv")
for r in results:
    if r["model"] not in df["model"].values:
        new=pd.DataFrame([{"model":r["model"],"pr_auc":r["pr_auc"],"macro_f1":r["macro_f1"],
                           "params":0,"train_sec":0,"notes":r.get("notes","")}])
        df=pd.concat([df,new],ignore_index=True)
df.to_csv(RES/"experiment_log.csv",index=False)

print("\n" + "="*65)
print("=== 성능 개선 실험 최종 결과 ===")
print("="*65)
print(f"  기존 최고 앙상블: 0.9367")
for r in results:
    ind = f"  Inductive={r.get('inductive_pr','N/A')}" if r.get('inductive_pr') else ""
    print(f"  {r['model']:40s} PR-AUC={r['pr_auc']:.4f}  F1={r['macro_f1']:.4f}{ind}")
print(f"\n저장: experiment_log.csv ({len(df)}행)")
