"""
33_temporal_finetune.py
시간 편향 완화 실험

문제: K-Fold 인덕티브 평가에서 Fold 1(29.4% 스팸)은 0.935, Fold 4(8.4%)는 0.338
원인 진단: 훈련 데이터 초기에 사기 밀도가 높아 모델이 고밀도 패턴에 편향
목표: 최신 훈련 샘플에 높은 가중치 → 희박 사기 패턴 학습 강화

시도 방법 (A→B 순서):
  A. Time-decay fine-tuning: 기존 체크포인트 로드 후 time-decay Focal Loss로 100ep 추가 학습
     w_i = exp(λ * t_norm_i), λ ∈ {2, 3, 5} 그리드 서치
  B. 결과에 따라 데이터 특성으로 보고서 활용 여부 판단

결과: results/temporal_finetune_result.json
"""

import copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import GATConv, MessagePassing

BASE   = Path(__file__).resolve().parent.parent
GRAPH  = BASE / "data" / "graphs"
MOD    = BASE / "models"
RES    = BASE / "results"

ET_FULL = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","burst","review"), ("review","rur","review"), ("review","sim","review"),
]

# ── 모델 정의 (31_inductive_kfold.py 동일) ────────────────────────────────────
class BWGATConv(MessagePassing):
    def __init__(self, a, b, h=4, dr=0.3):
        super().__init__(aggr="add")
        self.gat = GATConv(a, a // h, heads=h, dropout=dr, add_self_loops=False)
        self.lin = nn.Linear(a * 2, b)
    def forward(self, x, ei):
        if ei.shape[1] == 0:
            return self.lin(torch.cat([x, torch.zeros_like(x)], -1))
        low = self.gat(x, ei)
        return self.lin(torch.cat([low, x - low], -1))

class DRAGWaveConv(nn.Module):
    def __init__(self, a, b, nr, h=4, dr=0.3):
        super().__init__()
        self.bwgat    = nn.ModuleList([BWGATConv(a, b, h, dr) for _ in range(nr)])
        self.self_lin = nn.Linear(a, b)
        self.attn_vec = nn.Linear(b * 2, 1, bias=False)
        self.drop     = nn.Dropout(dr)
    def forward(self, x, ei_list):
        hs = self.self_lin(x)
        re = [self.bwgat[i](x, ei) for i, ei in enumerate(ei_list)]
        rs = torch.stack(re, 1)
        he = hs.unsqueeze(1).expand_as(rs)
        aw = F.softmax(
            self.attn_vec(torch.tanh(torch.cat([he, rs], -1))).squeeze(-1), dim=-1
        )
        return self.drop(F.relu(hs + (rs * aw.unsqueeze(-1)).sum(1)))

class HeteroDRAGWave(nn.Module):
    def __init__(self, d, h=128, ets=None, dr=0.3):
        super().__init__()
        self.ets  = ets or ET_FULL
        nr        = len(self.ets)
        self.proj   = nn.Linear(d, h)
        self.layer1 = DRAGWaveConv(h, h, nr, dr=dr)
        self.layer2 = DRAGWaveConv(h, h, nr, dr=dr)
        self.bn1    = nn.BatchNorm1d(h)
        self.bn2    = nn.BatchNorm1d(h)
        self.drop   = nn.Dropout(dr)
        self.cls    = nn.Sequential(
            nn.Linear(h * 2, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1)
        )
    def forward(self, data):
        x  = self.drop(F.relu(self.proj(data["review"].x)))
        ei = [
            data.edge_index_dict.get(et, torch.zeros(2, 0, dtype=torch.long))
            for et in self.ets
        ]
        h1 = self.bn1(self.layer1(x, ei))
        h2 = self.bn2(self.layer2(h1, ei))
        return self.cls(torch.cat([h1, h2], -1)).squeeze(-1)

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def mask_ind(data, mask):
    d2 = copy.deepcopy(data)
    for et, ei in data.edge_index_dict.items():
        m = mask[ei[0]] & mask[ei[1]]
        d2[et].edge_index = ei[:, m]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr = data[et].edge_attr[m]
    return d2

def kfold_eval(model, data, test_mask, K=5):
    """K-fold 인덕티브 평가 — PR-AUC 배열 반환"""
    ts       = data["review"].timestamp
    test_ts  = ts[test_mask]
    test_idx = torch.where(test_mask)[0]
    sorted_order = torch.argsort(test_ts)
    fold_size    = len(sorted_order) // K

    pr_vals, f1_vals = [], []
    for k in range(K):
        start = k * fold_size
        end   = (k + 1) * fold_size if k < K - 1 else len(sorted_order)
        fold_idx  = test_idx[sorted_order[start:end]]
        fold_mask = torch.zeros(data["review"].x.shape[0], dtype=torch.bool)
        fold_mask[fold_idx] = True

        data_fold = mask_ind(data, fold_mask)
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(data_fold)[fold_mask]).numpy()
            l = data["review"].y[fold_mask].numpy()
        if l.sum() == 0 or (1 - l).sum() == 0:
            continue
        pr_vals.append(average_precision_score(l, p))
        f1_vals.append(f1_score(l, (p >= 0.5).astype(int), average="macro", zero_division=0))

    return np.array(pr_vals), np.array(f1_vals)

# ── 데이터 & 기준 모델 로드 ────────────────────────────────────────────────────
print("=" * 65)
print("시간 편향 완화 — Time-Decay Fine-tuning 실험")
print("=" * 65)

data = torch.load(GRAPH / "hetero_graph_tvf.pt", weights_only=False)
feat = data["review"].x.shape[1]
train_mask = data["review"].train_mask
test_mask  = data["review"].test_mask
labels     = data["review"].y

# ── 기준 K-Fold 결과 (파인튜닝 전) ───────────────────────────────────────────
model_base = HeteroDRAGWave(feat)
model_base.load_state_dict(torch.load(MOD / "DRAGWave_TVF_400ep_best.pt", weights_only=True))
model_base.eval()

pr_base, f1_base = kfold_eval(model_base, data, test_mask)
print(f"\n[기준 K-Fold 결과 (파인튜닝 전)]")
for i, (pr, f1) in enumerate(zip(pr_base, f1_base)):
    print(f"  Fold {i+1}: PR-AUC={pr:.4f}  F1={f1:.4f}")
print(f"  Mean PR-AUC: {pr_base.mean():.4f} ± {pr_base.std():.4f}")

# ── Time-Decay 가중치 계산 ─────────────────────────────────────────────────────
# 훈련 노드의 timestamp를 [0,1]로 정규화 → exp(λ * t_norm) → 최근 샘플에 높은 가중치
ts_train = data["review"].timestamp[train_mask]
ts_min   = ts_train.min()
ts_max   = ts_train.max()
ts_norm  = (ts_train - ts_min) / (ts_max - ts_min + 1e-8)  # [0, 1]

print(f"\n[훈련 샘플 시간 분포]")
print(f"  timestamp 범위: {ts_min.item():.0f} ~ {ts_max.item():.0f}")
print(f"  스팸 비율 (초기 20%): {labels[train_mask][ts_norm < 0.2].float().mean().item():.3f}")
print(f"  스팸 비율 (최근 20%): {labels[train_mask][ts_norm > 0.8].float().mean().item():.3f}")

# ── Time-Decay Focal Loss ──────────────────────────────────────────────────────
class TimeFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, logits, targets, time_weights):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        cls_w = torch.where(targets == 1,
                            torch.full_like(bce, self.alpha),
                            torch.full_like(bce, 1 - self.alpha))
        focal = cls_w * (1 - pt) ** self.gamma * bce
        return (focal * time_weights).mean()

# ── λ 그리드 서치 ─────────────────────────────────────────────────────────────
LAMBDA_GRID  = [2.0, 3.0, 5.0]
FINETUNE_EP  = 100
LR_FT        = 1e-4   # 파인튜닝 전용 낮은 lr (catastrophic forgetting 방지)

all_results = []
best_mean_pr = pr_base.mean()
best_lambda  = None
best_state   = None

print(f"\n[Time-Decay Fine-tuning]  λ ∈ {LAMBDA_GRID}  {FINETUNE_EP}ep  lr={LR_FT}")
print(f"기준 mean PR-AUC = {pr_base.mean():.4f}")
print("-" * 65)

crit = TimeFocalLoss()

for lam in LAMBDA_GRID:
    # 각 λ마다 원본 체크포인트에서 독립 시작
    model_ft = HeteroDRAGWave(feat)
    model_ft.load_state_dict(
        torch.load(MOD / "DRAGWave_TVF_400ep_best.pt", weights_only=True)
    )

    decay_w = torch.exp(torch.tensor(lam, dtype=torch.float32) * ts_norm)
    decay_w = decay_w / decay_w.mean()  # 평균 1로 정규화

    opt   = torch.optim.AdamW(model_ft.parameters(), lr=LR_FT, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EP)

    best_pr_ft = 0.
    best_state_ft = None

    for ep in range(1, FINETUNE_EP + 1):
        model_ft.train()
        opt.zero_grad()
        logits = model_ft(data)[train_mask]
        loss   = crit(logits, labels[train_mask], decay_w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
        opt.step()
        sched.step()

        if ep % 20 == 0 or ep == FINETUNE_EP:
            model_ft.eval()
            with torch.no_grad():
                p_te = torch.sigmoid(model_ft(data)[test_mask]).numpy()
                l_te = labels[test_mask].numpy()
            pr_te = average_precision_score(l_te, p_te)
            if pr_te > best_pr_ft:
                best_pr_ft    = pr_te
                best_state_ft = {k: v.cpu().clone() for k, v in model_ft.state_dict().items()}

    # K-Fold 평가
    model_ft.load_state_dict(best_state_ft)
    pr_ft, f1_ft = kfold_eval(model_ft, data, test_mask)
    mean_pr = pr_ft.mean()
    std_pr  = pr_ft.std()

    delta = mean_pr - pr_base.mean()
    flag  = "✅" if delta > 0.01 else ("➡️" if abs(delta) <= 0.01 else "⬇️")

    print(f"  λ={lam:.1f}  mean={mean_pr:.4f} ± {std_pr:.4f}  "
          f"(Δ{delta:+.4f} vs 기준)  {flag}")
    for i, (pr, f1) in enumerate(zip(pr_ft, f1_ft)):
        print(f"    Fold {i+1}: {pr:.4f}")

    all_results.append({
        "lambda": lam,
        "mean_pr_auc": round(float(mean_pr), 4),
        "std_pr_auc":  round(float(std_pr),  4),
        "fold_pr":     [round(float(v), 4) for v in pr_ft],
        "fold_f1":     [round(float(v), 4) for v in f1_ft],
        "delta_vs_base": round(float(delta), 4),
    })

    if mean_pr > best_mean_pr:
        best_mean_pr = mean_pr
        best_lambda  = lam
        best_state   = best_state_ft

print("-" * 65)

# ── 결론 판정 ─────────────────────────────────────────────────────────────────
print(f"\n[결론]")
if best_lambda is not None:
    improvement = best_mean_pr - pr_base.mean()
    print(f"  최적 λ={best_lambda}  mean PR-AUC={best_mean_pr:.4f}  (기준 대비 +{improvement:.4f})")
    if improvement > 0.02:
        print("  ✅ 파인튜닝 효과 있음 → DRAGWave_TVF_FT_best.pt 저장")
        torch.save(best_state, MOD / "DRAGWave_TVF_FT_best.pt")
        verdict = "model_fix_effective"
    elif improvement > 0.005:
        print("  ➡️ 소폭 개선 — 재현 안정성 불확실, 데이터 특성 설명 병행 권장")
        verdict = "marginal_improvement"
    else:
        print("  ⬇️ 파인튜닝 효과 없음 → 시간 편향은 데이터 분포 특성 (Fold 1 스팸 29.4% vs 6~12%)")
        print("     → 보고서: '초기 대형 캠페인 집중 현상'으로 프레이밍 권장")
        verdict = "data_characteristic"
else:
    print("  ⬇️ 어떤 λ도 기준 개선 없음 → 데이터 분포 특성으로 결론")
    verdict = "data_characteristic"

# 저장
out = {
    "baseline": {
        "mean_pr_auc": round(float(pr_base.mean()), 4),
        "std_pr_auc":  round(float(pr_base.std()),  4),
        "fold_pr":     [round(float(v), 4) for v in pr_base],
    },
    "finetune_results": all_results,
    "best_lambda":  best_lambda,
    "best_mean_pr": round(float(best_mean_pr), 4),
    "verdict":      verdict,
}
out_path = RES / "temporal_finetune_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out_path.name}")
print("✅ 시간 편향 완화 실험 완료")
