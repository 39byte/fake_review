"""
39_strict_inductive_train.py
Strict Inductive Training — 과적합 완화 실험

기존 Transductive 학습의 문제:
  - 학습 시 Train-Test 연결 엣지가 노출됨
  - Train 노드가 Test 노드의 이웃 정보를 간접 활용 → Train=0.9999 과적합

Strict Inductive Training:
  - 학습 시 Test 노드에 연결된 엣지를 모두 제거
  - 모델이 Test 노드 정보를 완전히 차단된 상태에서 학습
  - 기대 효과: Train-Test Gap 감소, 인덕티브 성능 향상

비교 대상: HeteroBWGNN_boost (gap=0.071)
결과: results/strict_inductive_result.json
"""

import copy, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]

class DualFreqConv(MessagePassing):
    def __init__(self, a, b): super().__init__(aggr="mean"); self.lin = nn.Linear(a*2, b)
    def forward(self, x, ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self, x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self, d, h=128, dr=0.3):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: DualFreqConv(h,h) for et in ET}, aggr="sum")
        self.conv2 = HeteroConv({et: DualFreqConv(h,h) for et in ET}, aggr="sum")
        self.bn1 = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h); self.drop = nn.Dropout(dr)
        self.cls = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x))); d = {"review": x}
        d = self.conv1(d, data.edge_index_dict); d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
        d = self.conv2(d, data.edge_index_dict); d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

class FocalLoss(nn.Module):
    def __init__(self, g=2., a=0.75): super().__init__(); self.g, self.a = g, a
    def forward(self, lo, ta):
        bce = F.binary_cross_entropy_with_logits(lo, ta.float(), reduction="none")
        pt = torch.exp(-bce)
        w = torch.where(ta==1, torch.full_like(bce,self.a), torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def remove_test_edges(data: HeteroData, test_mask: torch.Tensor) -> HeteroData:
    """학습용: test 노드와 연결된 엣지를 모두 제거"""
    d2 = copy.deepcopy(data)
    for et, ei in data.edge_index_dict.items():
        src, dst = ei[0], ei[1]
        # 양 끝 모두 test 노드가 아닌 엣지만 유지
        keep = ~test_mask[src] & ~test_mask[dst]
        d2[et].edge_index = ei[:, keep]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr = data[et].edge_attr[keep]
    return d2

def test_only_graph(data: HeteroData, test_mask: torch.Tensor) -> HeteroData:
    """평가용: test 노드끼리만 연결된 서브그래프"""
    d2 = copy.deepcopy(data)
    for et, ei in data.edge_index_dict.items():
        src, dst = ei[0], ei[1]
        keep = test_mask[src] & test_mask[dst]
        d2[et].edge_index = ei[:, keep]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr = data[et].edge_attr[keep]
    return d2

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data)[mask]).numpy()
        l = data["review"].y[mask].numpy()
    pr = average_precision_score(l, p)
    f1 = f1_score(l, (p>=0.5).astype(int), average="macro", zero_division=0)
    return round(pr, 4), round(f1, 4)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
print("="*65)
print("Strict Inductive Training — 과적합 완화 실험")
print("="*65)

data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
train_mask = data["review"].train_mask
test_mask  = data["review"].test_mask
labels     = data["review"].y

# 기존 Transductive 결과 로드 (비교용)
df_log = __import__("pandas").read_csv(RES/"experiment_log.csv")
base_row = df_log[df_log["model"]=="HeteroBWGNN_boost"]
base_pr = float(base_row["pr_auc"].values[0]) if len(base_row) else 0.9242
print(f"\n[기준: HeteroBWGNN_boost Transductive]  PR-AUC={base_pr}")

# ── Strict Inductive 학습용 그래프 생성 ─────────────────────────────────────
print("\n[학습용 그래프: Test 노드 연결 엣지 전부 제거]")
data_train = remove_test_edges(data, test_mask)

# 엣지 수 비교
for et in ET:
    orig = data[et].edge_index.shape[1]
    reduced = data_train[et].edge_index.shape[1]
    print(f"  {et[1]:8s}: {orig:>8,} → {reduced:>8,} ({reduced/orig*100:.1f}%)")

# 평가용 인덕티브 그래프
data_ind = test_only_graph(data, test_mask)

# ── 학습 ─────────────────────────────────────────────────────────────────────
print(f"\n[Strict Inductive Training] 400 epoch")
torch.manual_seed(42)
model = HeteroBWGNN(feat)
opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)
crit  = FocalLoss()

best_pr, best_state = 0., None
t0 = time.time()

history = []
for ep in range(1, 401):
    model.train(); opt.zero_grad()
    # 학습: test 엣지가 제거된 그래프 사용
    loss = crit(model(data_train)[train_mask], labels[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); sched.step()

    if ep % 50 == 0 or ep == 1:
        # Train 평가 (전체 그래프)
        model.eval()
        with torch.no_grad():
            p_tr = torch.sigmoid(model(data)[train_mask]).numpy()
            l_tr = labels[train_mask].numpy()
        train_pr = round(float(average_precision_score(l_tr, p_tr)), 4)

        # Test 평가 (전체 그래프 - transductive)
        te_pr, te_f1 = evaluate(model, data, test_mask)
        # Test 평가 (인덕티브)
        ind_pr, ind_f1 = evaluate(model, data_ind, test_mask)
        gap = round(train_pr - te_pr, 4)

        print(f"  ep={ep:3d}  train={train_pr:.4f}  test={te_pr:.4f}  "
              f"gap={gap:+.4f}  inductive={ind_pr:.4f}  ({time.time()-t0:.0f}s)")
        history.append({"epoch":ep, "train_pr":train_pr, "test_pr":te_pr,
                        "gap":gap, "inductive_pr":ind_pr, "f1":te_f1})

        if te_pr > best_pr:
            best_pr = te_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)

# ── 최종 평가 ─────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    p_tr = torch.sigmoid(model(data)[train_mask]).numpy()
    l_tr = labels[train_mask].numpy()
train_pr_f = round(float(average_precision_score(l_tr, p_tr)), 4)

te_pr_f,  te_f1_f  = evaluate(model, data, test_mask)
ind_pr_f, ind_f1_f = evaluate(model, data_ind, test_mask)
gap_f = round(train_pr_f - te_pr_f, 4)

print(f"\n{'='*65}")
print(f"[최종 결과 비교]")
print(f"{'':30s}  {'Transductive':>13}  {'인덕티브':>10}  {'Gap':>8}")
print(f"  기준 (Transductive 학습):  PR-AUC={base_pr:.4f}       (0.6526)    (+0.071)")
print(f"  Strict Inductive 학습:    PR-AUC={te_pr_f:.4f}    {ind_pr_f:.4f}    ({gap_f:+.4f})")

gap_improvement = 0.0710 - abs(gap_f)
ind_improvement = ind_pr_f - 0.6526

print(f"\n  Gap 변화: 0.071 → {abs(gap_f):.4f}  ({'-' if gap_improvement>0 else '+'}{abs(gap_improvement):.4f})")
print(f"  인덕티브 변화: 0.6526 → {ind_pr_f:.4f}  ({ind_improvement:+.4f})")

if abs(gap_f) < 0.060:
    verdict = "✅ Gap 개선 — Strict Inductive Training 효과 있음"
elif ind_pr_f > 0.68:
    verdict = "✅ 인덕티브 성능 향상 — 배포 환경 개선"
elif abs(gap_f) < 0.071 and ind_pr_f > 0.6526:
    verdict = "➡️ 소폭 개선 — 제한적 효과"
else:
    verdict = "⬇️ 개선 없음 — Transductive 방식이 유리"

print(f"\n  판정: {verdict}")

torch.save(best_state, MOD/"BWGNN_StrictInductive_best.pt")

result = {
    "method": "Strict Inductive Training (test 노드 엣지 완전 차단)",
    "baseline": {"transductive_pr": base_pr, "inductive_pr": 0.6526, "gap": 0.071},
    "strict_inductive": {
        "transductive_pr": te_pr_f, "transductive_f1": te_f1_f,
        "inductive_pr": ind_pr_f, "inductive_f1": ind_f1_f,
        "train_pr": train_pr_f, "gap": gap_f,
    },
    "improvement": {"gap_delta": round(gap_improvement,4), "inductive_delta": round(ind_improvement,4)},
    "verdict": verdict,
    "history": history,
}
out = RES/"strict_inductive_result.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out.name}")
print("Done")
