"""
38_temporal_retrain.py
시간 편향 완화 — 처음부터 time-decay로 재학습

파인튜닝(33번)과의 차이:
  33번: 이미 수렴된 체크포인트에서 100ep 추가 미세조정 → 기존 local min에 고착
  38번: 무작위 초기화에서 400ep 전체 학습 → 다른 local min 탐색 가능

모델: HeteroBWGNN (빠름 ~540s, 결과로 방향성 판단)
그래프: hetero_graph_boost.pt (5종 엣지)
학습: time-decay Focal Loss (λ=5, 훈련 초기=저가중치 / 후기=고가중치)
평가: K-fold 인덕티브 (5 fold)
"""

import copy, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET_FULL = [("review","rtr","review"),("review","rsr","review"),
           ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj =nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in ET_FULL},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in ET_FULL},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

def mask_ind(data,mask):
    d2=copy.deepcopy(data)
    for et,ei in data.edge_index_dict.items():
        m=mask[ei[0]]&mask[ei[1]]; d2[et].edge_index=ei[:,m]
        if hasattr(data[et],"edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr=data[et].edge_attr[m]
    return d2

def kfold_eval(model, data, test_mask, K=5):
    ts=data["review"].timestamp; test_ts=ts[test_mask]
    test_idx=torch.where(test_mask)[0]
    so=torch.argsort(test_ts); fs=len(so)//K
    results=[]
    for k in range(K):
        s=k*fs; e=(k+1)*fs if k<K-1 else len(so)
        fm=torch.zeros(data["review"].x.shape[0],dtype=torch.bool)
        fm[test_idx[so[s:e]]]=True
        spam_ratio=data["review"].y[fm].float().mean().item()
        df=mask_ind(data,fm)
        model.eval()
        with torch.no_grad():
            p=torch.sigmoid(model(df)[fm]).numpy(); l=data["review"].y[fm].numpy()
        if l.sum()==0: results.append((k+1,spam_ratio,None,None)); continue
        pr=average_precision_score(l,p)
        f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
        results.append((k+1,spam_ratio,round(pr,4),round(f1,4)))
    return results

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
print("="*65)
print("재학습 실험 — Time-Decay Focal Loss from scratch")
print("="*65)

data=torch.load(GRAPH/"hetero_graph_boost.pt",weights_only=False)
feat=data["review"].x.shape[1]
train_mask=data["review"].train_mask
test_mask =data["review"].test_mask
labels    =data["review"].y

# ── 기준: 기존 체크포인트 K-fold ───────────────────────────────────────────────
print("\n[기준: HeteroBWGNN_boost (기존 체크포인트)]")
base_model=HeteroBWGNN(feat)
base_model.load_state_dict(torch.load(MOD/"HeteroBWGNN_boost_best.pt",weights_only=True))
base_results=kfold_eval(base_model,data,test_mask)
print(f"  {'Fold':>4}  {'스팸밀도':>8}  {'PR-AUC':>8}  {'F1':>8}")
for k,sr,pr,f1 in base_results:
    pr_s=f"{pr:.4f}" if pr else "  N/A "
    f1_s=f"{f1:.4f}" if f1 else "  N/A "
    print(f"  {k:>4}  {sr:.3f}      {pr_s}    {f1_s}")
base_prs=[r[2] for r in base_results if r[2]]
print(f"  Mean PR-AUC: {np.mean(base_prs):.4f} +- {np.std(base_prs):.4f}")

# ── Time-Decay 가중치 ──────────────────────────────────────────────────────────
ts_train=(data["review"].timestamp[train_mask])
ts_norm=(ts_train-ts_train.min())/(ts_train.max()-ts_train.min()+1e-8)
LAMBDA=5.0
decay_w=torch.exp(torch.tensor(LAMBDA)*ts_norm)
decay_w=decay_w/decay_w.mean()

print(f"\n[재학습 설정]  lambda={LAMBDA}  epochs=400  lr=5e-4 (from scratch)")
print(f"  훈련 초기 20% 가중치: {decay_w[ts_norm<0.2].mean().item():.3f}x")
print(f"  훈련 후기 20% 가중치: {decay_w[ts_norm>0.8].mean().item():.3f}x")

# ── 재학습 ────────────────────────────────────────────────────────────────────
class TimeFocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,logits,targets,w):
        bce=F.binary_cross_entropy_with_logits(logits,targets.float(),reduction="none")
        pt=torch.exp(-bce)
        cw=torch.where(targets==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (cw*(1-pt)**self.g*bce*w).mean()

torch.manual_seed(42)
model_rt=HeteroBWGNN(feat)
opt  =torch.optim.AdamW(model_rt.parameters(),lr=5e-4,weight_decay=1e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=400)
crit=TimeFocalLoss()

best_pr,best_state=0.,None
t0=time.time()

for ep in range(1,401):
    model_rt.train(); opt.zero_grad()
    loss=crit(model_rt(data)[train_mask],labels[train_mask],decay_w)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_rt.parameters(),1.)
    opt.step(); sched.step()

    if ep%50==0 or ep==1:
        model_rt.eval()
        with torch.no_grad():
            p=torch.sigmoid(model_rt(data)[test_mask]).numpy()
            l=labels[test_mask].numpy()
        pr=average_precision_score(l,p)
        f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
        print(f"  ep={ep:3d}  loss={loss.item():.4f}  PR-AUC={pr:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if pr>best_pr: best_pr=pr; best_state={k:v.cpu().clone() for k,v in model_rt.state_dict().items()}

model_rt.load_state_dict(best_state)
torch.save(best_state, MOD/"BWGNN_TimedecayRetrain_best.pt")

# ── K-fold 인덕티브 평가 ───────────────────────────────────────────────────────
print(f"\n[재학습 K-fold 인덕티브 평가]")
rt_results=kfold_eval(model_rt,data,test_mask)
print(f"  {'Fold':>4}  {'스팸밀도':>8}  {'기준PR':>8}  {'재학습PR':>8}  {'Delta':>8}")
for (k,sr,pr_b,_),(k2,sr2,pr_rt,f1_rt) in zip(base_results,rt_results):
    if pr_b is None or pr_rt is None: continue
    d=pr_rt-pr_b
    mark=" *" if d>0.01 else ""
    print(f"  {k:>4}  {sr:.3f}      {pr_b:.4f}    {pr_rt:.4f}    {d:+.4f}{mark}")

rt_prs=[r[2] for r in rt_results if r[2]]
base_prs=[r[2] for r in base_results if r[2]]
delta_mean=np.mean(rt_prs)-np.mean(base_prs)

print(f"\n  기준   Mean PR-AUC: {np.mean(base_prs):.4f} +- {np.std(base_prs):.4f}")
print(f"  재학습 Mean PR-AUC: {np.mean(rt_prs):.4f} +- {np.std(rt_prs):.4f}  (delta {delta_mean:+.4f})")

if delta_mean > 0.02:
    verdict="effective — 재학습으로 시간 편향 완화 가능"
elif delta_mean > 0.005:
    verdict="marginal — 소폭 개선, 근본 해결 미달"
else:
    verdict="ineffective — 재학습도 효과 없음. 데이터 분포 문제로 최종 결론"
print(f"\n  판정: {verdict}")

result={
    "baseline_kfold": [{"fold":k,"spam_ratio":sr,"pr_auc":pr} for k,sr,pr,_ in base_results if pr],
    "retrain_kfold":  [{"fold":k,"spam_ratio":sr,"pr_auc":pr} for k,sr,pr,_ in rt_results if pr],
    "summary":{
        "base_mean":   round(float(np.mean(base_prs)),4),
        "retrain_mean":round(float(np.mean(rt_prs)),4),
        "delta":       round(float(delta_mean),4),
        "base_std":    round(float(np.std(base_prs)),4),
        "retrain_std": round(float(np.std(rt_prs)),4),
    },
    "verdict": verdict,
}
with open(RES/"temporal_retrain_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"\n저장: temporal_retrain_result.json")
print("Done")
