"""
05b_train_tgat_v2.py
Tri-Path Dual Memory GNN (TGATLiteV2)

핵심 기여 (vs Lee et al. AAAI 2024 TGN):
  Lee et al.: 금융 거래 도메인, user 단일 메모리 m_u(t), 단일 릴레이션
  TGATLiteV2: 리뷰 도메인, dual memory [m_u + m_p] + burst temporal 경로 분리

기여 1 — Log-Bochner 시간 인코더
  Standard: φ(Δt) = [cos(ω·Δt), sin(ω·Δt)]
  Log-Bochner: φ(Δt) = [cos(ω·log(1+Δt)), sin(ω·log(1+Δt))]
  근거: burst 탐지는 0~72h 짧은 구간 정밀도가 핵심 → log-scale이 단기 간격 구분력 강화

기여 2 — Tri-Path 분리 (User / Product / Burst)
  노드=리뷰인 제약 하에서 TGN의 단일 엔티티 메모리를 이중 엔티티로 일반화
  Path 1 (User Memory)    : rur edges     → 사용자 행동 이력 집계 (m_u 근사)
  Path 2 (Product Memory) : rtr + rsr     → 상품 리뷰 클러스터 집계 (m_p 근사)
  Path 3 (Burst Temporal) : burst + ΔT    → 단기 공모 신호 (시간 인코딩 특화)
  → concat([user, product, burst]) 로 세 의미론적 신호를 명시적으로 분리

Ablation 모델:
  TGATLite-LogBochner : 기존 아키텍처 + Log-Bochner만 교체 → 로그 인코딩 단독 효과
  TGATLiteV2          : 경로 분리 + Log-Bochner → 통합 기여
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
data = data.to(DEVICE)

FEAT_DIM   = data["review"].x.shape[1]
HIDDEN     = 128
D_TIME     = 64
HEADS      = 4
NUM_EPOCHS = 400
PATIENCE   = 30
LR         = 5e-4

EDGE_TYPES_USER    = [("review", "rur", "review")]
EDGE_TYPES_PRODUCT = [("review", "rtr", "review"), ("review", "rsr", "review")]
EDGE_TYPES_NO_BURST = EDGE_TYPES_USER + EDGE_TYPES_PRODUCT


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(targets == 1,
                          torch.full_like(bce, self.alpha),
                          torch.full_like(bce, 1 - self.alpha))
        return (w * (1 - pt) ** self.gamma * bce).mean()


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
# 기여 1: Log-Bochner 시간 인코더
# φ(Δt) = [cos(ω·log(1+Δt)), sin(ω·log(1+Δt))]
# log-scale: burst 0~72h 구간에서 단기 간격 구분력 향상
# ════════════════════════════════════════════════════════════════════
class LogBochnerTimeEncoder(nn.Module):
    def __init__(self, d_time: int = 64):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(d_time // 2))

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        delta_t = delta_t.squeeze(-1) if delta_t.dim() == 2 else delta_t
        # log(1+Δt): Δt=0 → 0, Δt→∞ → log scale 압축
        log_dt = torch.log1p(delta_t)
        t = log_dt.unsqueeze(-1) * self.omega.unsqueeze(0)  # [E, d/2]
        return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)  # [E, d_time]


# ════════════════════════════════════════════════════════════════════
# TimeAwareConv: burst 엣지 전용 (원본과 동일, 재사용)
# ════════════════════════════════════════════════════════════════════
class TimeAwareConv(nn.Module):
    def __init__(self, in_ch, out_ch, d_time, heads):
        super().__init__()
        self.time_proj = nn.Linear(d_time, in_ch)
        self.gat = GATConv(in_ch, out_ch // heads, heads=heads,
                           dropout=0.3, add_self_loops=False)
        self.in_ch = in_ch

    def forward(self, x, edge_index, time_emb):
        time_feat = self.time_proj(time_emb)
        src = edge_index[0]
        x_boosted = x.clone()
        x_boosted.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.in_ch), time_feat)
        return self.gat(x_boosted, edge_index)


# ════════════════════════════════════════════════════════════════════
# Ablation: TGATLite + Log-Bochner만 교체 (경로 분리 없음)
# 비교 목적: 로그 인코딩 단독 기여 측정
# ════════════════════════════════════════════════════════════════════
class TGATLiteLogBochner(nn.Module):
    def __init__(self, in_ch, hidden, d_time=D_TIME, heads=HEADS, dropout=0.3):
        super().__init__()
        self.time_encoder = LogBochnerTimeEncoder(d_time)  # Log-Bochner
        self.proj = nn.Linear(in_ch, hidden)

        self.conv1_nonburst = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum"
        )
        self.time_conv1 = TimeAwareConv(hidden, hidden, d_time, heads)
        self.conv2_nonburst = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum"
        )
        self.time_conv2 = TimeAwareConv(hidden, hidden, d_time, heads)

        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls  = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))

        burst_ei = data["review", "burst", "review"].edge_index
        delta_t  = data["review", "burst", "review"].edge_attr.squeeze(-1)
        time_emb = self.time_encoder(delta_t)

        x_nb1   = self.conv1_nonburst({"review": x},
                      {et: data.edge_index_dict[et] for et in EDGE_TYPES_NO_BURST})
        x_burst1 = self.time_conv1(x, burst_ei, time_emb)
        x1 = self.drop(F.relu(self.bn1(x_nb1["review"] + x_burst1)))

        x_nb2   = self.conv2_nonburst({"review": x1},
                      {et: data.edge_index_dict[et] for et in EDGE_TYPES_NO_BURST})
        x_burst2 = self.time_conv2(x1, burst_ei, time_emb)

        out = torch.cat([x_nb2["review"], x_burst2], dim=-1)
        return self.cls(out).squeeze(-1)


# ════════════════════════════════════════════════════════════════════
# 기여 2: TGATLiteV2 — Tri-Path Dual Memory GNN
#
# Lee et al. (AAAI 2024) TGN과의 차이:
#   TGN  : user 단일 메모리 m_u(t), 노드=유저/계정
#   V2   : dual memory [m_u + m_p], 노드=리뷰 (대회 제약)
#          + burst temporal 경로를 제3의 명시적 신호로 분리
#
# 경로별 의미:
#   User Memory    (rur)       → 동일 사용자 리뷰들 간 행동 패턴 집계 (m_u 근사)
#   Product Memory (rtr+rsr)   → 동일 상품 리뷰들 간 클러스터 패턴 집계 (m_p 근사)
#   Burst Temporal (burst+ΔT)  → 단기 공모 신호 (시간 인코딩 특화)
# ════════════════════════════════════════════════════════════════════
class TGATLiteV2(nn.Module):
    def __init__(self, in_ch, hidden, d_time=D_TIME, heads=HEADS, dropout=0.3):
        super().__init__()
        self.time_encoder = LogBochnerTimeEncoder(d_time)  # Log-Bochner
        self.proj = nn.Linear(in_ch, hidden)

        # Path 1: User Memory (rur 전용 SAGEConv)
        self.user_conv1 = SAGEConv(hidden, hidden)
        self.user_conv2 = SAGEConv(hidden, hidden)

        # Path 2: Product Memory (rtr + rsr HeteroConv)
        self.product_conv1 = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_PRODUCT}, aggr="mean"
        )
        self.product_conv2 = HeteroConv(
            {et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_PRODUCT}, aggr="mean"
        )

        # Path 3: Burst Temporal (Log-Bochner + TimeAwareConv)
        self.time_conv1 = TimeAwareConv(hidden, hidden, d_time, heads)
        self.time_conv2 = TimeAwareConv(hidden, hidden, d_time, heads)

        self.bn1  = nn.BatchNorm1d(hidden)
        self.bn2  = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)

        # concat(user, product, burst) = hidden*3
        self.cls = nn.Sequential(
            nn.Linear(hidden * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))

        burst_ei = data["review", "burst", "review"].edge_index
        delta_t  = data["review", "burst", "review"].edge_attr.squeeze(-1)
        time_emb = self.time_encoder(delta_t)
        rur_ei   = data["review", "rur", "review"].edge_index

        # ── Layer 1: 세 경로 병렬 처리 ───────────────────────────────
        x_user1    = self.user_conv1(x, rur_ei)                          # [N, H]
        x_product1 = self.product_conv1(
            {"review": x},
            {et: data.edge_index_dict[et] for et in EDGE_TYPES_PRODUCT}
        )["review"]                                                       # [N, H]
        x_burst1   = self.time_conv1(x, burst_ei, time_emb)             # [N, H]

        # Layer 2 입력: 세 경로 평균 (정보 보존)
        x1 = self.drop(F.relu(self.bn1((x_user1 + x_product1 + x_burst1) / 3.0)))

        # ── Layer 2: 세 경로 병렬 처리 ───────────────────────────────
        x_user2    = self.user_conv2(x1, rur_ei)                         # [N, H]
        x_product2 = self.product_conv2(
            {"review": x1},
            {et: data.edge_index_dict[et] for et in EDGE_TYPES_PRODUCT}
        )["review"]                                                       # [N, H]
        x_burst2   = self.time_conv2(x1, burst_ei, time_emb)            # [N, H]

        # ── 최종: 세 의미론적 경로 명시적 concat ─────────────────────
        out = torch.cat([x_user2, x_product2, x_burst2], dim=-1)        # [N, H*3]
        return self.cls(out).squeeze(-1)


# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train_model(model, name, data, notes=""):
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
          f"Macro-F1={final['Macro-F1']:.4f}  params={n_params:,}  time={elapsed}s")
    torch.save(best_state, MOD / f"{name}_best.pt")
    pd.DataFrame(history).to_csv(RES / f"history_{name}.csv", index=False)
    return final, n_params, elapsed, notes


# ── 실행 ──────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
results = []

print("=" * 60)
print("▶ Ablation: TGATLite-LogBochner (경로 분리 없음, 로그 인코딩만)")
print("=" * 60)
m_log = TGATLiteLogBochner(FEAT_DIM, HIDDEN)
metrics, n_params, elapsed, notes = train_model(
    m_log, "TGATLiteV2_LogBochner_400ep", data,
    notes="Log-Bochner 단독 ablation: log(1+Δt) 인코딩, 경로 미분리"
)
results.append({"model": "TGATLiteV2_LogBochner_400ep",
                "pr_auc": metrics["PR-AUC"], "macro_f1": metrics["Macro-F1"],
                "params": n_params, "train_sec": elapsed, "notes": notes})

print("\n" + "=" * 60)
print("▶ TGATLiteV2: Tri-Path Dual Memory GNN (제안 모델)")
print("=" * 60)
m_v2 = TGATLiteV2(FEAT_DIM, HIDDEN)
metrics, n_params, elapsed, notes = train_model(
    m_v2, "TGATLiteV2_400ep", data,
    notes="Tri-Path(User+Product+Burst) + Log-Bochner: Dual Memory 일반화"
)
results.append({"model": "TGATLiteV2_400ep",
                "pr_auc": metrics["PR-AUC"], "macro_f1": metrics["Macro-F1"],
                "params": n_params, "train_sec": elapsed, "notes": notes})

# ── 결과 통합 및 비교 ──────────────────────────────────────────────────────────
log_path = RES / "experiment_log.csv"
df_log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
df_new = pd.DataFrame(results)
df_log = pd.concat([df_log, df_new], ignore_index=True)
df_log.to_csv(log_path, index=False)

print("\n" + "=" * 60)
print("=== Ablation 비교 (모델별 기여) ===")
print("=" * 60)
ablation_models = ["TGATLite", "TGATLiteV2_LogBochner_400ep", "TGATLiteV2_400ep"]
cols = ["model", "pr_auc", "macro_f1", "params"]
subset = df_log[df_log["model"].isin(ablation_models)][cols]
print(subset.to_string(index=False))

print("""
── Ablation 해석 ──────────────────────────────────────────────
TGATLite          : Bochner,    경로 통합   (기준 모델)
TGATLite_LogBochner: Log-Bochner, 경로 통합  (기여 1만 적용)
TGATLiteV2        : Log-Bochner, 경로 분리   (기여 1+2 통합)

ΔF1(LogBochner - TGATLite)  = 로그 인코딩 단독 기여
ΔF1(V2 - LogBochner)        = Dual Memory 경로 분리 기여
ΔF1(V2 - TGATLite)          = 전체 기여
──────────────────────────────────────────────────────────────
""")

print("→ 저장:", log_path)
print("→ 모델:", MOD / "TGATLiteV2_best.pt")
print("→ 다음 단계: /itda-report 로 보고서 반영")
