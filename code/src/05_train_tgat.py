"""
05_train_tgat.py
TGAT-lite: Bochner 시간 인코딩 + HeteroGAT
핵심 가설: burst 엣지의 Δt 자체가 사기 시그널 → 연속 시간 인코딩으로 학습
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
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
data = data.to(DEVICE)

FEAT_DIM = data["review"].x.shape[1]   # 388
HIDDEN   = 128
D_TIME   = 64    # Bochner 시간 인코딩 차원
HEADS    = 4
NUM_EPOCHS = 150
PATIENCE   = 20
LR         = 5e-4

EDGE_TYPES_NO_BURST = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "rur",   "review"),
]

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce  = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt   = torch.exp(-bce)
        w    = torch.where(targets == 1,
                           torch.full_like(bce, self.alpha),
                           torch.full_like(bce, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * bce).mean()

# ── 평가 함수 ─────────────────────────────────────────────────────────────────
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

# ════════════════════════════════════════════════════════════════════
# Bochner 시간 인코더
# φ(Δt) = [cos(ω₁·Δt), sin(ω₁·Δt), ..., cos(ωₖ·Δt), sin(ωₖ·Δt)]
# ω는 학습 가능한 주파수 파라미터
# ════════════════════════════════════════════════════════════════════
class BochnerTimeEncoder(nn.Module):
    def __init__(self, d_time: int = 64):
        super().__init__()
        self.d_time = d_time
        # 학습 가능한 주파수 ω (초기: 표준정규)
        self.omega = nn.Parameter(torch.randn(d_time // 2))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t: [E] (단위: 시간)
        delta_t = delta_t.squeeze(-1) if delta_t.dim() == 2 else delta_t
        t = delta_t.unsqueeze(-1) * self.omega.unsqueeze(0)  # [E, d/2]
        return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)  # [E, d_time]


# ════════════════════════════════════════════════════════════════════
# TGAT-lite
# burst 엣지: GAT attention score에 Δt 인코딩을 소스 노드 피처에 합산
# 나머지 엣지: 표준 SAGEConv
# ════════════════════════════════════════════════════════════════════
class TimeAwareConv(nn.Module):
    """burst 엣지 전용: 소스 노드 피처에 시간 임베딩을 더한 뒤 GAT 적용"""
    def __init__(self, in_ch, out_ch, d_time, heads):
        super().__init__()
        self.time_proj = nn.Linear(d_time, in_ch)   # 시간 임베딩 → 노드 피처 공간
        self.gat = GATConv(in_ch, out_ch // heads, heads=heads,
                           dropout=0.3, add_self_loops=False)
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.heads  = heads

    def forward(self, x, edge_index, time_emb):
        # time_emb: [E_burst, d_time] → [E_burst, in_ch]
        time_feat = self.time_proj(time_emb)  # [E_burst, in_ch]
        # 소스 노드에 시간 피처 scatter-add (각 엣지 → 소스 노드 보정)
        src = edge_index[0]  # [E_burst]
        x_boosted = x.clone()
        x_boosted.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.in_ch), time_feat)
        return self.gat(x_boosted, edge_index)  # [N, out_ch]


class TGATLite(nn.Module):
    def __init__(self, in_ch, hidden, d_time=D_TIME, heads=HEADS, dropout=0.3):
        super().__init__()
        self.time_encoder = BochnerTimeEncoder(d_time)
        self.proj = nn.Linear(in_ch, hidden)

        # Layer 1
        self.conv1_nonburst = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum"
        )
        self.time_conv1 = TimeAwareConv(hidden, hidden, d_time, heads)

        # Layer 2
        self.conv2_nonburst = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum"
        )
        self.time_conv2 = TimeAwareConv(hidden, hidden, d_time, heads)

        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(
            nn.Linear(hidden * 2, 64),   # burst(hidden) + non-burst(hidden)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))

        burst_ei   = data["review", "burst", "review"].edge_index
        delta_t    = data["review", "burst", "review"].edge_attr.squeeze(-1)
        time_emb   = self.time_encoder(delta_t)  # [E_burst, d_time]

        # Layer 1
        x_dict_nb  = self.conv1_nonburst({"review": x}, {et: data.edge_index_dict[et]
                                          for et in EDGE_TYPES_NO_BURST})
        x_burst1   = self.time_conv1(x, burst_ei, time_emb)  # [N, hidden]
        x1 = self.drop(F.relu(self.bn1(x_dict_nb["review"] + x_burst1)))

        # Layer 2
        x_dict_nb2 = self.conv2_nonburst({"review": x1}, {et: data.edge_index_dict[et]
                                          for et in EDGE_TYPES_NO_BURST})
        x_burst2   = self.time_conv2(x1, burst_ei, time_emb)  # [N, hidden]
        x2 = self.drop(F.relu(self.bn2(x_dict_nb2["review"] + x_burst2)))

        # 최종: non-burst + burst 경로 concat
        out = torch.cat([x_dict_nb2["review"], x_burst2], dim=-1)  # [N, hidden*2]
        return self.cls(out).squeeze(-1)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train_model(model, name, data):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = FocalLoss()

    train_mask = data["review"].train_mask
    labels     = data["review"].y
    best_pr_auc, best_state, no_improve = 0.0, None, 0
    history = []
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss   = criterion(logits[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            tr = evaluate(model, data, train_mask)
            te = evaluate(model, data, data["review"].test_mask)
            print(f"  [{name}] ep={epoch:3d}  loss={loss.item():.4f}  "
                  f"train={tr['PR-AUC']:.4f}  test={te['PR-AUC']:.4f}  "
                  f"({time.time()-t0:.0f}s)")
            history.append({"epoch": epoch, **te, "loss": round(loss.item(), 4)})
            if te["PR-AUC"] > best_pr_auc:
                best_pr_auc = te["PR-AUC"]
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE // 10:
                    print(f"  [{name}] Early stop ep={epoch}")
                    break

    model.load_state_dict(best_state)
    final   = evaluate(model, data, data["review"].test_mask)
    elapsed = round(time.time() - t0, 1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}] FINAL  PR-AUC={final['PR-AUC']:.4f}  "
          f"Macro-F1={final['Macro-F1']:.4f}  params={n_params:,}  time={elapsed}s\n")
    torch.save(best_state, MOD / f"{name}_best.pt")
    pd.DataFrame(history).to_csv(RES / f"history_{name}.csv", index=False)
    return final, n_params, elapsed


# ── 실행 ──────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
print("=" * 60)
print("▶ TGAT-lite 학습 시작 (Bochner 시간 인코딩)")
print("=" * 60)

model = TGATLite(FEAT_DIM, HIDDEN)
metrics, n_params, elapsed = train_model(model, "TGATLite", data)

# 기존 실험 로그에 추가
log_path = RES / "experiment_log.csv"
if log_path.exists():
    df_log = pd.read_csv(log_path)
else:
    df_log = pd.DataFrame()

new_row = pd.DataFrame([{
    "model":     "TGATLite",
    "pr_auc":    metrics["PR-AUC"],
    "macro_f1":  metrics["Macro-F1"],
    "params":    n_params,
    "train_sec": elapsed,
    "notes":     "TGAT-lite: Bochner 시간 인코딩, burst Δt edge feature",
}])
df_log = pd.concat([df_log, new_row], ignore_index=True)
df_log.to_csv(log_path, index=False)

print("\n=== 전체 실험 결과 ===")
print(df_log[["model", "pr_auc", "macro_f1", "params", "train_sec"]].to_string(index=False))

# Go/No-Go
best = df_log.loc[df_log["pr_auc"].idxmax()]
bwgnn_row = df_log[df_log["model"] == "HeteroBWGNN"]
bwgnn_auc = bwgnn_row["pr_auc"].values[0] if len(bwgnn_row) else 0.0
tgat_auc  = metrics["PR-AUC"]

print(f"\n[Go/No-Go] TGAT-lite({tgat_auc:.4f}) vs BWGNN({bwgnn_auc:.4f}): "
      + ("GO ✅ TGAT 제출" if tgat_auc >= bwgnn_auc else "NO-GO ❌ BWGNN 결과로 제출"))
print("→ 저장:", log_path)
print("→ 다음 단계: 보고서 작성 (/itda-report)")
