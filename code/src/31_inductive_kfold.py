"""
31_inductive_kfold.py
DRAGWave_TVF_400ep — 시간 K-Fold 인덕티브 평가

test 기간을 K=5 시간 구간으로 분할, 각 fold에서 test-only 엣지 마스킹 후 평가.
목적: 인덕티브 성능 분포(mean ± std) 측정 → 특정 시간대에서만 강한지 확인.
결과: results/inductive_kfold_result.json
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
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
    ("review", "sim",   "review"),
]

# ── 모델 정의 (27_future_directions.py 동일 구조) ────────────────────────────
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
    """mask 내 노드끼리만 연결된 서브그래프 생성"""
    d2 = copy.deepcopy(data)
    for et, ei in data.edge_index_dict.items():
        m = mask[ei[0]] & mask[ei[1]]
        d2[et].edge_index = ei[:, m]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr = data[et].edge_attr[m]
    return d2

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data)[mask]).numpy()
        l = data["review"].y[mask].numpy()
    if l.sum() == 0 or (1 - l).sum() == 0:
        return {"pr_auc": None, "macro_f1": None,
                "n_spam": int(l.sum()), "n_total": int(len(l))}
    return {
        "pr_auc":    round(float(average_precision_score(l, p)), 4),
        "macro_f1":  round(float(f1_score(l, (p >= 0.5).astype(int),
                                          average="macro", zero_division=0)), 4),
        "n_spam":    int(l.sum()),
        "n_total":   int(len(l)),
    }

# ── 데이터 & 모델 로드 ────────────────────────────────────────────────────────
print("=" * 65)
print("DRAGWave_TVF_400ep — 시간 K-Fold 인덕티브 평가")
print("=" * 65)

tvf_path  = GRAPH / "hetero_graph_tvf.pt"
ckpt_path = MOD   / "DRAGWave_TVF_400ep_best.pt"

for p, label in [(tvf_path, "hetero_graph_tvf.pt"), (ckpt_path, "DRAGWave_TVF_400ep_best.pt")]:
    if not p.exists():
        print(f"[ERROR] {label} 없음. 27_future_directions.py 먼저 실행하세요.")
        raise SystemExit(1)

data = torch.load(tvf_path, weights_only=False)
feat = data["review"].x.shape[1]
print(f"  TVF 그래프: 노드 {data['review'].x.shape[0]:,}  피처 {feat}d")

model = HeteroDRAGWave(feat)
model.load_state_dict(torch.load(ckpt_path, weights_only=True))
model.eval()
print(f"  체크포인트: {ckpt_path.name}")

# ── 전체 인덕티브 베이스라인 ─────────────────────────────────────────────────
test_mask    = data["review"].test_mask
data_ind_all = mask_ind(data, test_mask)
baseline     = evaluate(model, data_ind_all, test_mask)

print(f"\n[전체 인덕티브 베이스라인]")
print(f"  PR-AUC={baseline['pr_auc']}  F1={baseline['macro_f1']}  "
      f"스팸={baseline['n_spam']}/{baseline['n_total']}")

# ── 시간 K-Fold ──────────────────────────────────────────────────────────────
K        = 5
ts       = data["review"].timestamp
test_ts  = ts[test_mask]
test_idx = torch.where(test_mask)[0]

sorted_order = torch.argsort(test_ts)
fold_size    = len(sorted_order) // K

print(f"\n[K={K} 시간 Fold 인덕티브 평가]")
print(f"  {'Fold':>4}  {'시간 구간':>20}  {'PR-AUC':>8}  {'F1':>8}  {'스팸/전체':>10}")
print("-" * 60)

fold_results = []
for k in range(K):
    start = k * fold_size
    end   = (k + 1) * fold_size if k < K - 1 else len(sorted_order)

    fold_node_idx = test_idx[sorted_order[start:end]]
    fold_mask     = torch.zeros(data["review"].x.shape[0], dtype=torch.bool)
    fold_mask[fold_node_idx] = True

    ts_lo = float(test_ts[sorted_order[start]])
    ts_hi = float(test_ts[sorted_order[end - 1]])

    data_fold = mask_ind(data, fold_mask)
    res       = evaluate(model, data_fold, fold_mask)

    fold_results.append({"fold": k + 1, "ts_lo": round(ts_lo, 4),
                          "ts_hi": round(ts_hi, 4), **res})

    pr_s = f"{res['pr_auc']:.4f}" if res["pr_auc"] is not None else "  N/A "
    f1_s = f"{res['macro_f1']:.4f}" if res["macro_f1"] is not None else "  N/A "
    print(f"  {k+1:>4}   [{ts_lo:.3f}~{ts_hi:.3f}]   {pr_s:>8}  {f1_s:>8}  "
          f"{res['n_spam']:>4}/{res['n_total']:>5}")

# 요약
valid   = [r for r in fold_results if r["pr_auc"] is not None]
pr_vals = [r["pr_auc"]   for r in valid]
f1_vals = [r["macro_f1"] for r in valid]

print("-" * 60)
if valid:
    print(f"  Mean  PR-AUC={np.mean(pr_vals):.4f} ± {np.std(pr_vals):.4f}  "
          f"F1={np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}")
    print(f"  Min={min(pr_vals):.4f}  Max={max(pr_vals):.4f}  "
          f"Range={max(pr_vals)-min(pr_vals):.4f}")

    stable = np.std(pr_vals) < 0.05
    print(f"\n  안정성 판정: {'✅ 안정적 (std < 0.05)' if stable else '⚠️ 불안정 (std ≥ 0.05)'}")

# 저장
out = {
    "model":               "DRAGWave_TVF_400ep",
    "baseline_inductive":  baseline,
    "kfold":               fold_results,
    "summary": {
        "k":           K,
        "mean_pr_auc": round(np.mean(pr_vals), 4) if valid else None,
        "std_pr_auc":  round(np.std(pr_vals),  4) if valid else None,
        "mean_f1":     round(np.mean(f1_vals),  4) if valid else None,
        "std_f1":      round(np.std(f1_vals),   4) if valid else None,
    },
}

out_path = RES / "inductive_kfold_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out_path.name}")
print("✅ K-Fold 인덕티브 평가 완료")
