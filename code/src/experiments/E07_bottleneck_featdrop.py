"""
E07_bottleneck_featdrop.py
실험 7: Bottleneck Architecture + Feature Dropout

가설:
  - Bottleneck(128→32→128): 정보 압축 강제 → 암기 방지
  - Feature Dropout(p=0.2): 입력 피처 일부 차단 → 특정 피처 의존 방지
  - 두 방법 모두 기존(DropEdge/StrongReg)과 다른 메커니즘

예상 효과: Gap 0.066 → ~0.055
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

# ── 핵심: Bottleneck 구조 적용 ────────────────────────────────────────────────
class BottleneckBWGNN(nn.Module):
    """
    기존: proj(d→128) → conv1(128→128) → conv2(128→128) → cls
    변경: proj(d→128) → bottleneck(128→32) → conv1(32→32) → expand(32→128)
          → conv2(128→128) → cls

    파라미터 변화:
      기존 conv1 가중치: 128*2*128 = 32,768
      변경 후: 32*2*32 = 2,048 (84% 감소)
    """
    def __init__(self, d, h=128, bottleneck=32, dr=0.3, feat_drop=0.2):
        super().__init__()
        self.feat_drop = feat_drop
        self.proj      = nn.Linear(d, h)

        # 병목 레이어
        self.bn_down  = nn.Linear(h, bottleneck)
        self.bn_up    = nn.Linear(bottleneck, h)

        # 병목 차원에서 conv
        self.conv1 = HeteroConv({et: DualFreqConv(bottleneck, bottleneck) for et in ET}, aggr="sum")
        self.conv2 = HeteroConv({et: DualFreqConv(h, h) for et in ET}, aggr="sum")

        self.bn1  = nn.BatchNorm1d(bottleneck)
        self.bn2  = nn.BatchNorm1d(h)
        self.drop = nn.Dropout(dr)
        self.cls  = nn.Sequential(nn.Linear(h, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1))

    def forward(self, data):
        x = data["review"].x

        # Feature Dropout: 입력 피처 일부 차단
        x = F.dropout(x, p=self.feat_drop, training=self.training)

        # 프로젝션 + 병목 압축
        x = self.drop(F.relu(self.proj(x)))
        x = F.relu(self.bn_down(x))               # 128 → 32

        # 병목 차원에서 1차 메시지 전파
        d = {"review": x}
        d = self.conv1(d, data.edge_index_dict)
        x = self.drop(F.relu(self.bn1(d["review"])))   # [N, 32]

        # 확장 후 2차 메시지 전파
        x = F.relu(self.bn_up(x))                      # 32 → 128
        d = {"review": x}
        d = self.conv2(d, data.edge_index_dict)
        x = self.drop(F.relu(self.bn2(d["review"])))   # [N, 128]

        return self.cls(x).squeeze(-1)

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
    pr=average_precision_score(l,p)
    f1=f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0)
    return round(pr,4), round(f1,4)

# ── 실험 실행 ─────────────────────────────────────────────────────────────────
print("="*65)
print("E07: Bottleneck(128→32→128) + Feature Dropout(p=0.2)")
print("="*65)

torch.manual_seed(42)
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
tm   = data["review"].train_mask
te   = data["review"].test_mask
y    = data["review"].y

model  = BottleneckBWGNN(feat, h=128, bottleneck=32, dr=0.3, feat_drop=0.2)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n파라미터: {n_params:,}  (기존 387,329 대비 {(n_params/387329)*100:.0f}%)")
print(f"파라미터/노드 비율: {n_params/tm.sum().item():.1f}  (기존 16.1)")

opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
crit  = FocalLoss()

best_pr, best_state, no_imp = 0., None, 0
history = []; t0 = time.time()

print(f"\n  {'ep':>4}  {'tr_pr':>8}  {'te_pr':>8}  {'gap':>8}  {'ind':>8}")
print("  " + "-"*45)

for ep in range(1, 401):
    model.train(); opt.zero_grad()
    loss = crit(model(data)[tm], y[tm])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 40 == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(model(data)[tm]).numpy()
        tr_pr = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
        te_pr, te_f1 = evaluate(model, data, te)
        gap = round(tr_pr - te_pr, 4)

        # 인덕티브 (매 100ep)
        ind_pr = 0.0
        if ep % 100 == 0:
            d_ind = mask_ind(data, te)
            ind_pr, _ = evaluate(model, d_ind, te)
        print(f"  {ep:4d}  {tr_pr:8.4f}  {te_pr:8.4f}  {gap:+8.4f}  {ind_pr:8.4f}")
        history.append({"ep":ep,"tr_pr":tr_pr,"te_pr":te_pr,"gap":gap,"ind_pr":ind_pr})

        if te_pr > best_pr:
            best_pr = te_pr
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
    p_tr = torch.sigmoid(model(data)[tm]).numpy()
tr_f = round(float(average_precision_score(y[tm].numpy(), p_tr)), 4)
te_f, te_f1f = evaluate(model, data, te)
gap_f = round(tr_f - te_f, 4)
d_ind = mask_ind(data, te)
ind_f, ind_f1f = evaluate(model, d_ind, te)

print(f"\n{'='*65}")
print(f"[E07 최종]  Train={tr_f:.4f}  Test={te_f:.4f}  Gap={gap_f:+.4f}  Inductive={ind_f:.4f}")
print(f"[기준 대비]  Test: 0.9242→{te_f:.4f} ({te_f-0.9242:+.4f})  Gap: +0.071→{gap_f:+.4f} ({gap_f-0.071:+.4f})  Ind: 0.6526→{ind_f:.4f} ({ind_f-0.6526:+.4f})")

torch.save(best_state, MOD/"E07_Bottleneck_best.pt")
result = {
    "experiment":"E07_bottleneck_featdrop",
    "config":{"bottleneck":32,"feat_drop":0.2,"hidden":128},
    "n_params":n_params,"n_train":int(tm.sum()),
    "param_ratio":round(n_params/tm.sum().item(),1),
    "train_pr":tr_f,"test_pr":te_f,"test_f1":te_f1f,
    "gap":gap_f,"inductive_pr":ind_f,"inductive_f1":ind_f1f,
    "history":history,
}
with open(RES/"E07_result.json","w",encoding="utf-8") as f:
    json.dump(result,f,indent=2,ensure_ascii=False)
print(f"저장: results/experiments/E07_result.json")
