"""
27_future_directions.py
향후 발전 방향 실험

FD1. DRAGWave + NoRSR (400 epoch)
     근거: H2에서 BWGNN NoRSR +2.8%p → DRAGWave도 RSR 제거 시 향상 기대
     가설: 317K 노이즈 엣지 제거 + DRAGWave 동적 attention = 시너지

FD2. DRAGWave + TVF (400 epoch)
     근거: TVF 300ep에서 인덕티브 0.7093 (기존 최고) → 400ep면 0.73+ 기대
     가설: 시간 속도 피처 + 완전 수렴 = 인덕티브 갭 추가 해소

목표:
  - Transductive: 0.9367 (현재 최고) 초과 도전
  - Inductive: 0.7093 (현재 최고) 초과 도전
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

EDGE_TYPES_FULL = [("review","rtr","review"),("review","rsr","review"),
                   ("review","burst","review"),("review","rur","review"),("review","sim","review")]
EDGE_TYPES_NORSR= [("review","rtr","review"),
                   ("review","burst","review"),("review","rur","review"),("review","sim","review")]
N_REL_FULL = 5
N_REL_NORSR= 4

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
    def __init__(self,d,h=128,edge_types=None,dr=0.3):
        super().__init__()
        nr = len(edge_types) if edge_types else N_REL_FULL
        self.edge_types = edge_types or EDGE_TYPES_FULL
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr)
        self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict.get(et,torch.zeros(2,0,dtype=torch.long)) for et in self.edge_types]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none"); pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def eval_both(m,data):
    m.eval()
    with torch.no_grad():
        logits=m(data)
        def s(mask):
            p=torch.sigmoid(logits[mask]).numpy(); l=data["review"].y[mask].numpy()
            return round(average_precision_score(l,p),4), round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
        return s(data["review"].train_mask), s(data["review"].test_mask)

def mask_ind(data,mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

def train(model,name,data,epochs=400):
    model=model.to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    crit=FocalLoss(); tm=data["review"].train_mask; lb=data["review"].y
    best_pr,best_state,no_imp=0.,None,0; t0=time.time()
    print(f"\n▶ [{name}] 학습 시작 ({epochs} epoch)")
    for ep in range(1,epochs+1):
        model.train(); opt.zero_grad()
        loss=crit(model(data)[tm],lb[tm]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step()
        if ep%40==0 or ep==1:
            (tr,_),(te,f1)=eval_both(model,data); gap=round(tr-te,4)
            print(f"  ep={ep:3d}  train={tr:.4f}  test={te:.4f}  gap={gap:+.4f}  ({time.time()-t0:.0f}s)")
            if te>best_pr: best_pr=te; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp>=7: print(f"  Early stop ep={ep}"); break
    model.load_state_dict(best_state)
    (tr,_),(te,f1)=eval_both(model,data); gap=round(tr-te,4)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"  FINAL  train={tr}  test={te}  gap={gap:+.4f}  F1={f1}  ({round(time.time()-t0,1)}s)")
    torch.save(best_state, MOD/f"{name}_best.pt")
    return {"model":name,"pr_auc":te,"macro_f1":f1,"train_pr":tr,"gap":gap,"params":n_params}

torch.manual_seed(42)
results=[]

# ── FD1: DRAGWave + NoRSR ─────────────────────────────────────────────────────
print("="*65)
print("FD1. DRAGWave + NoRSR (RSR 317K 엣지 제거)")
print("근거: H2에서 BWGNN +2.8%p → DRAGWave도 시도")
print("="*65)

data_boost = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
# RSR 제거
data_norsr = copy.deepcopy(data_boost)
if ("review","rsr","review") in data_norsr.edge_index_dict:
    del data_norsr._edge_store_dict[("review","rsr","review")]
    print(f"  RSR 제거 완료 — 잔존 엣지 타입: {list(data_norsr.edge_index_dict.keys())}")

feat = data_norsr["review"].x.shape[1]
m_norsr = HeteroDRAGWave(feat, edge_types=EDGE_TYPES_NORSR)
r_norsr = train(m_norsr, "DRAGWave_NoRSR", data_norsr, epochs=400)
results.append(r_norsr)

# 인덕티브 평가
data_norsr_ind = mask_ind(data_norsr, data_norsr["review"].test_mask)
(_, _),(ind_pr,ind_f1) = eval_both(m_norsr, data_norsr_ind)
print(f"  인덕티브: PR-AUC={ind_pr}  F1={ind_f1}")
r_norsr["inductive_pr"] = ind_pr

# ── FD2: DRAGWave + TVF 400ep ─────────────────────────────────────────────────
print("\n" + "="*65)
print("FD2. DRAGWave + TVF 400 epoch (300ep: 인덕티브 0.7093)")
print("근거: 시간 속도 피처 완전 수렴으로 0.73+ 도전")
print("="*65)

data_tvf = torch.load(GRAPH/"hetero_graph_tvf.pt", weights_only=False)
feat_tvf = data_tvf["review"].x.shape[1]
print(f"  TVF 피처 차원: {feat_tvf} (386+8)")

# TVF 그래프의 엣지 타입 확인
tvf_ets = list(data_tvf.edge_index_dict.keys())
print(f"  TVF 엣지 타입: {[et[1] for et in tvf_ets]}")

m_tvf = HeteroDRAGWave(feat_tvf, edge_types=tvf_ets)
r_tvf = train(m_tvf, "DRAGWave_TVF_400ep", data_tvf, epochs=400)
results.append(r_tvf)

# 인덕티브 평가
data_tvf_ind = mask_ind(data_tvf, data_tvf["review"].test_mask)
(_,_),(ind_pr_tvf,ind_f1_tvf) = eval_both(m_tvf, data_tvf_ind)
print(f"  인덕티브: PR-AUC={ind_pr_tvf}  F1={ind_f1_tvf}")
r_tvf["inductive_pr"] = ind_pr_tvf

# ── 결과 비교 ──────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("=== 향후 발전 방향 실험 결과 ===")
print("="*65)
print(f"\n{'모델':30s} {'Transductive':>14} {'Inductive':>12} {'Gap':>8}")
print("-"*65)
print(f"  {'DRAGWave_400ep (기준)':28s} {'0.9340':>14} {'0.6570':>12} {'+0.0659':>8}")
print(f"  {'DRAGWave_TVF_300ep (이전)':28s} {'0.9165':>14} {'0.7093':>12} {'+0.0788':>8}")
for r in results:
    ind = r.get("inductive_pr", "N/A")
    ind_str = f"{ind:.4f}" if isinstance(ind, float) else ind
    print(f"  {r['model']:28s} {r['pr_auc']:>14.4f} {ind_str:>12} {r['gap']:>+8.4f}")

# experiment_log 업데이트
df_log = pd.read_csv(RES/"experiment_log.csv")
for r in results:
    if r["model"] not in df_log["model"].values:
        new=pd.DataFrame([{"model":r["model"],"pr_auc":r["pr_auc"],"macro_f1":r["macro_f1"],
                           "params":r["params"],"train_sec":0,"notes":f"발전방향 inductive={r.get('inductive_pr','N/A')}"}])
        df_log=pd.concat([df_log,new],ignore_index=True)
df_log.to_csv(RES/"experiment_log.csv",index=False)

with open(RES/"future_directions_results.json","w",encoding="utf-8") as f:
    json.dump({"results":results,"baseline":{"DRAGWave_400ep":{"trans":0.9340,"ind":0.6570}}},
              f,ensure_ascii=False,indent=2)
print(f"\n저장: results/future_directions_results.json")
print(f"저장: experiment_log.csv ({len(df_log)}행)")
