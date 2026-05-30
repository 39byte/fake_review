"""
07_ablation.py
Ablation: TGATLite에서 Bochner 시간 인코딩 제거 → burst 엣지를 일반 SAGEConv로 처리
"""

import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, time
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_DIM = 388
HIDDEN   = 128
D_TIME   = 64
HEADS    = 4

EDGE_TYPES = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
]

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(targets==1, torch.full_like(bce, self.alpha), torch.full_like(bce, 1-self.alpha))
        return (w * (1-pt)**self.gamma * bce).mean()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = data["review"].y[mask].cpu().numpy()
    return {
        "PR-AUC":   round(average_precision_score(labels, probs), 4),
        "Macro-F1": round(f1_score(labels, probs>=0.5, average="macro", zero_division=0), 4),
    }

# ── TGATLite-NoTime: 시간 인코딩 없이 burst 엣지를 SAGEConv로만 처리 ──────────
class TGATLiteNoTime(nn.Module):
    """Ablation: Bochner 시간 인코딩 제거, burst 엣지도 SAGEConv 처리"""
    def __init__(self, in_ch, hidden, dropout=0.3):
        super().__init__()
        self.proj  = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(dropout)
        self.cls   = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        x_dict = {"review": x}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn1(x_dict["review"])))}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn2(x_dict["review"])))}
        return self.cls(x_dict["review"]).squeeze(-1)

# ── 학습 ──────────────────────────────────────────────────────────────────────
data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False).to(DEVICE)
torch.manual_seed(42)

model = TGATLiteNoTime(FEAT_DIM, HIDDEN).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
criterion = FocalLoss()
train_mask = data["review"].train_mask
labels     = data["review"].y

best_pr, best_state, no_improve = 0.0, None, 0
history = []
t0 = time.time()

print("▶ TGATLite-NoTime (Ablation) 학습")
for epoch in range(1, 151):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(data)[train_mask], labels[train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step(); scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        te = evaluate(model, data, data["review"].test_mask)
        print(f"  ep={epoch:3d}  loss={loss.item():.4f}  PR-AUC={te['PR-AUC']:.4f}  F1={te['Macro-F1']:.4f}")
        history.append({"epoch": epoch, **te, "loss": round(loss.item(), 4)})
        if te["PR-AUC"] > best_pr:
            best_pr, best_state, no_improve = te["PR-AUC"], {k:v.cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= 2: print(f"  Early stop ep={epoch}"); break

model.load_state_dict(best_state)
final   = evaluate(model, data, data["review"].test_mask)
elapsed = round(time.time()-t0, 1)
torch.save(best_state, MOD / "TGATLite_NoTime_best.pt")
pd.DataFrame(history).to_csv(RES / "history_TGATLite_NoTime.csv", index=False)

# inductive 평가도 동시 실행
import copy
test_mask = data["review"].test_mask
data_test = copy.deepcopy(data)
for et in EDGE_TYPES:
    ei = data[et].edge_index
    mask = test_mask[ei[0]] & test_mask[ei[1]]
    data_test[et].edge_index = ei[:, mask]
    if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
        data_test[et].edge_attr = data[et].edge_attr[mask]
final_ind = evaluate(model, data_test, test_mask)

# 기존 inductive 로그에 추가
log_path = RES / "experiment_log_inductive.csv"
df = pd.read_csv(log_path)
new = pd.DataFrame([{
    "model": "TGATLite_NoTime",
    "pr_auc": final_ind["PR-AUC"],
    "macro_f1": final_ind["Macro-F1"],
    "params": sum(p.numel() for p in model.parameters()),
    "train_sec": elapsed,
    "notes": "Ablation: Bochner 시간 인코딩 제거",
}])
pd.concat([df, new]).to_csv(log_path, index=False)

print(f"\n=== Ablation 결과 ===")
print(f"TGATLite_NoTime  표준: PR-AUC={final['PR-AUC']}  F1={final['Macro-F1']}")
print(f"TGATLite_NoTime  인덕: PR-AUC={final_ind['PR-AUC']}  F1={final_ind['Macro-F1']}")

df_all = pd.read_csv(log_path)
tgat = df_all[df_all["model"]=="TGATLite"]
notime = df_all[df_all["model"]=="TGATLite_NoTime"]
if len(tgat) and len(notime):
    delta = round(tgat["macro_f1"].values[0] - notime["macro_f1"].values[0], 4)
    print(f"\n시간 인코딩 Macro-F1 기여: +{delta}")
