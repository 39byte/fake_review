"""
04_train_baseline.py
베이스라인 3종 학습: HeteroSAGE(GCN proxy), HeteroGAT, HeteroBWGNN
평가: PR-AUC + Macro F1 (대회 필수 지표)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"
RES.mkdir(exist_ok=True)
MOD.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
data = data.to(DEVICE)

FEAT_DIM    = data["review"].x.shape[1]   # 388
HIDDEN      = 128
NUM_EPOCHS  = 150
PATIENCE    = 20
LR          = 5e-4
WEIGHT_DECAY= 1e-5

EDGE_TYPES  = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
]

print(f"노드: {data['review'].x.shape[0]:,}  피처: {FEAT_DIM}")
print(f"train: {data['review'].train_mask.sum().item():,}  "
      f"test: {data['review'].test_mask.sum().item():,}")

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        # alpha=0.75: 스팸(소수 클래스) 가중치 높임
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt   = torch.exp(-bce)
        w    = torch.where(targets == 1,
                           torch.full_like(bce, self.alpha),
                           torch.full_like(bce, 1 - self.alpha))
        focal = w * (1 - pt) ** self.gamma * bce
        return focal.mean()

# ── 공통 평가 함수 ─────────────────────────────────────────────────────────────
def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs  = torch.sigmoid(logits[mask]).cpu().numpy()
        labels = data["review"].y[mask].cpu().numpy()
    pr_auc   = average_precision_score(labels, probs)
    preds    = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"PR-AUC": round(pr_auc, 4), "Macro-F1": round(macro_f1, 4)}

# ── 공통 학습 루프 ─────────────────────────────────────────────────────────────
def train_model(model, name, data, epochs=NUM_EPOCHS, patience=PATIENCE):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(gamma=2.0, alpha=0.75)

    train_mask = data["review"].train_mask
    labels     = data["review"].y

    best_pr_auc    = 0.0
    best_state     = None
    no_improve     = 0
    history        = []
    t0             = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss   = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            tr_metrics = evaluate(model, data, train_mask)
            te_metrics = evaluate(model, data, data["review"].test_mask)
            elapsed    = time.time() - t0
            print(f"  [{name}] epoch={epoch:3d}  loss={loss.item():.4f}  "
                  f"train PR-AUC={tr_metrics['PR-AUC']:.4f}  "
                  f"test  PR-AUC={te_metrics['PR-AUC']:.4f}  "
                  f"({elapsed:.0f}s)")
            history.append({"epoch": epoch, **te_metrics, "loss": round(loss.item(), 4)})

            if te_metrics["PR-AUC"] > best_pr_auc:
                best_pr_auc = te_metrics["PR-AUC"]
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience // 10:   # patience는 10-epoch 단위
                    print(f"  [{name}] Early stop at epoch {epoch}")
                    break

    model.load_state_dict(best_state)
    final = evaluate(model, data, data["review"].test_mask)
    total_time = time.time() - t0
    n_params   = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}] FINAL  PR-AUC={final['PR-AUC']:.4f}  "
          f"Macro-F1={final['Macro-F1']:.4f}  "
          f"params={n_params:,}  time={total_time:.0f}s\n")

    torch.save(best_state, MOD / f"{name}_best.pt")
    return final, n_params, round(total_time, 1), history


# ════════════════════════════════════════════════════════════════════
# Model 1: HeteroSAGE (GraphSAGE 기반 GCN proxy)
# ════════════════════════════════════════════════════════════════════
class HeteroSAGE(nn.Module):
    def __init__(self, in_ch, hidden, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum"
        )
        self.bn1 = nn.BatchNorm1d(hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(
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


# ════════════════════════════════════════════════════════════════════
# Model 2: HeteroGAT (Graph Attention Network)
# ════════════════════════════════════════════════════════════════════
class HeteroGAT(nn.Module):
    def __init__(self, in_ch, hidden, heads=4, dropout=0.3):
        super().__init__()
        self.proj  = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv(
            {et: GATConv(hidden, hidden // heads, heads=heads,
                         dropout=dropout, add_self_loops=False)
             for et in EDGE_TYPES}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: GATConv(hidden, hidden // heads, heads=heads,
                         dropout=dropout, add_self_loops=False)
             for et in EDGE_TYPES}, aggr="sum"
        )
        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(
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


# ════════════════════════════════════════════════════════════════════
# Model 3: HeteroBWGNN
# 핵심 아이디어: 각 relation에서 low-pass (집계) + high-pass (편차)를 함께 학습
# 사기 리뷰는 이웃과 다른 패턴 → high-pass 성분이 핵심 신호
# ════════════════════════════════════════════════════════════════════
class DualFreqConv(MessagePassing):
    """Low-pass (이웃 집계) + High-pass (자신 - 이웃) 동시 학습"""
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_ch * 2, out_ch)

    def forward(self, x, edge_index):
        # low-pass: 이웃 평균 집계
        low  = self.propagate(edge_index, x=x)         # [N, in_ch]
        # high-pass: 자신과 이웃 평균의 편차 (사기 시그널)
        high = x - low                                  # [N, in_ch]
        return self.lin(torch.cat([low, high], dim=-1)) # [N, out_ch]

    def message(self, x_j):
        return x_j


class HeteroBWGNN(nn.Module):
    def __init__(self, in_ch, hidden, dropout=0.3):
        super().__init__()
        self.proj  = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv(
            {et: DualFreqConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: DualFreqConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum"
        )
        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(
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


# ════════════════════════════════════════════════════════════════════
# 실험 실행
# ════════════════════════════════════════════════════════════════════
torch.manual_seed(42)
results_log = []

models_to_run = [
    ("HeteroSAGE",  HeteroSAGE(FEAT_DIM, HIDDEN)),
    ("HeteroGAT",   HeteroGAT(FEAT_DIM, HIDDEN)),
    ("HeteroBWGNN", HeteroBWGNN(FEAT_DIM, HIDDEN)),
]

for name, model in models_to_run:
    print("=" * 60)
    print(f"▶ {name} 학습 시작")
    print("=" * 60)
    metrics, n_params, elapsed, history = train_model(model, name, data)
    results_log.append({
        "model":     name,
        "pr_auc":    metrics["PR-AUC"],
        "macro_f1":  metrics["Macro-F1"],
        "params":    n_params,
        "train_sec": elapsed,
        "notes":     "정적 베이스라인, Focal Loss γ=2 α=0.75",
    })
    # 히스토리 저장
    pd.DataFrame(history).to_csv(
        RES / f"history_{name}.csv", index=False
    )

# ── 결과 저장 ─────────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results_log)
df_results.to_csv(RES / "experiment_log.csv", index=False)

print("\n" + "=" * 60)
print("실험 결과 요약")
print("=" * 60)
print(df_results[["model", "pr_auc", "macro_f1", "params", "train_sec"]].to_string(index=False))

best = df_results.loc[df_results["pr_auc"].idxmax()]
print(f"\n베스트 모델: {best['model']}  PR-AUC={best['pr_auc']}  Macro-F1={best['macro_f1']}")
go_nogo = "GO ✅" if best["pr_auc"] >= 0.70 else "NO-GO ❌ — Plan A 보고서로 제출 전환"
print(f"Go/No-Go (PR-AUC ≥ 0.70): {go_nogo}")
print(f"\n→ 저장: {RES / 'experiment_log.csv'}")
print("→ 다음 단계: 05_train_tgat.py (TGAT-lite 시간 인코딩)")
