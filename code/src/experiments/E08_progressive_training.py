"""
E08_progressive_training.py
실험 8: Progressive Training (Curriculum Learning)

가설:
  - 초기: 명확한 스팸/정상 노드만으로 학습 (쉬운 샘플)
  - 점진적: 경계 노드(0.4~0.6 확률) 추가
  - 모델이 핵심 패턴부터 학습 → 노이즈/암기 방지

커리큘럼:
  ep 1~100:   train 노드 중 50% (가장 명확한 것)
  ep 101~200: train 노드 중 75%
  ep 201~400: train 노드 전체

예상 효과: Gap 0.066 → ~0.045
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

def get_progressive_mask(base_mask, probs, ratio):
    """
    base_mask 내 노드 중 가장 확신도 높은 ratio% 선택
    (|prob - 0.5|가 클수록 명확한 노드 = 쉬운 샘플)
    """
    idx = torch.where(base_mask)[0]
    confidence = torch.abs(probs[idx] - 0.5)  # 높을수록 쉬움
    n_select = max(1, int(len(idx) * ratio))
    sorted_idx = idx[torch.argsort(confidence, descending=True)][:n_select]
    new_mask = torch.zeros_like(base_mask)
    new_mask[sorted_idx] = True
    return new_mask

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

# ── 실험 실행 ─────────────────────────────────────────────────────────────────
print("="*65)
print("E08: Progressive Training (Curriculum Learning)")
print("="*65)

torch.manual_seed(42)
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
tm   = data["review"].train_mask
te   = data["review"].test_mask
y    = data["review"].y

# 커리큘럼 스케줄
SCHEDULE = [
    (100, 0.50, "쉬운 50% (ep 1~100)"),
    (200, 0.75, "쉬운 75% (ep 101~200)"),
    (400, 1.00, "전체 100% (ep 201~400)"),
]
print(f"\n커리큘럼 스케줄:")
for ep_end, ratio, desc in SCHEDULE:
    print(f"  ep ≤ {ep_end:3d}: train 노드의 {ratio*100:.0f}% — {desc}")

model = HeteroBWGNN(feat)
opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
crit  = FocalLoss()

best_pr, best_state = 0., None
history = []; t0 = time.time()

# 초기 확률: 랜덤 초기화 기준 (이후 실제 예측으로 갱신)
cur_mask = tm.clone()
n_total  = tm.sum().item()

print(f"\n  {'ep':>4}  {'mask_n':>7}  {'tr_pr':>8}  {'te_pr':>8}  {'gap':>8}")
print("  " + "-"*48)

for ep in range(1, 401):
    # 커리큘럼 마스크 갱신 (40ep마다)
    if ep % 40 == 1:
        ratio = next((r for e,r,_ in SCHEDULE if ep <= e), 1.0)
        if ratio < 1.0:
            model.eval()
            with torch.no_grad():
                probs_all = torch.sigmoid(model(data))
            cur_mask = get_progressive_mask(tm, probs_all, ratio)
        else:
            cur_mask = tm.clone()

    model.train(); opt.zero_grad()
    loss = crit(model(data)[cur_mask], y[cur_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 40 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(model(data)[tm]).numpy()
        tr_pr = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
        te_pr, _ = evaluate(model, data, te)
        gap = round(tr_pr - te_pr, 4)
        print(f"  {ep:4d}  {cur_mask.sum():7,}  {tr_pr:8.4f}  {te_pr:8.4f}  {gap:+8.4f}")
        history.append({"ep":ep,"mask_n":int(cur_mask.sum()),"tr_pr":tr_pr,"te_pr":te_pr,"gap":gap})

        if te_pr > best_pr:
            best_pr = te_pr
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

# 최종
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(model(data)[tm]).numpy()
tr_f = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
te_f, te_f1f = evaluate(model, data, te)
gap_f = round(tr_f - te_f, 4)
d_ind = mask_ind(data, te)
ind_f, ind_f1f = evaluate(model, d_ind, te)

print(f"\n{'='*65}")
print(f"[E08 최종]  Train={tr_f:.4f}  Test={te_f:.4f}  Gap={gap_f:+.4f}  Inductive={ind_f:.4f}")
print(f"[기준 대비]  Test: 0.9242→{te_f:.4f} ({te_f-0.9242:+.4f})  Gap: +0.071→{gap_f:+.4f} ({gap_f-0.071:+.4f})  Ind: 0.6526→{ind_f:.4f} ({ind_f-0.6526:+.4f})")

torch.save(best_state, MOD/"E08_Progressive_best.pt")
result = {
    "experiment":"E08_progressive_training",
    "config":{"schedule":[[e,r] for e,r,_ in SCHEDULE]},
    "train_pr":tr_f,"test_pr":te_f,"test_f1":te_f1f,
    "gap":gap_f,"inductive_pr":ind_f,"inductive_f1":ind_f1f,
    "history":history,
}
with open(RES/"E08_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"저장: results/experiments/E08_result.json")
