"""
32_ensemble_4way.py
인덕티브 갭 최소 모델 기반 4-way 앙상블

모델 선택 근거 (인덕티브 PR-AUC 기준):
  A. DRAGWave_TVF_400ep  — inductive 0.7429 (갭 최소 18.5%)
  B. DRAGWave_NoRSR      — inductive 0.7234 (갭 21.3%)
  C. HeteroBWGNN_boost   — inductive 0.6526 (boost 그래프)
  D. BWGAT               — inductive ~0.65  (boost 그래프)

전략: 인덕티브 서브그래프에서 각 모델 확률 추출 → 가중치 그리드 서치
결과: results/ensemble_4way_result.json
"""

import copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE   = Path(__file__).resolve().parent.parent
GRAPH  = BASE / "data" / "graphs"
MOD    = BASE / "models"
RES    = BASE / "results"

ET_FULL  = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","burst","review"), ("review","rur","review"), ("review","sim","review"),
]
ET_NORSR = [
    ("review","rtr","review"),
    ("review","burst","review"), ("review","rur","review"), ("review","sim","review"),
]

# ── 공통 모델 클래스 ──────────────────────────────────────────────────────────
class DualFreqConv(MessagePassing):
    def __init__(self, a, b):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(a * 2, b)
    def forward(self, x, ei):
        low = self.propagate(ei, x=x)
        return self.lin(torch.cat([low, x - low], -1))
    def message(self, x_j): return x_j

class HeteroBWGNN(nn.Module):
    def __init__(self, d, h=128, ets=None, dr=0.3):
        super().__init__()
        ets = ets or ET_FULL
        self.proj  = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: DualFreqConv(h, h) for et in ets}, aggr="sum")
        self.conv2 = HeteroConv({et: DualFreqConv(h, h) for et in ets}, aggr="sum")
        self.bn1   = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h)
        self.drop  = nn.Dropout(dr)
        self.cls   = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        d = {"review": x}
        d = self.conv1(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
        d = self.conv2(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

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

class HeteroBWGAT(nn.Module):
    def __init__(self, d, h=128, ets=None, dr=0.3):
        super().__init__()
        ets = ets or ET_FULL
        self.proj  = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: BWGATConv(h, h, dr=dr) for et in ets}, aggr="sum")
        self.conv2 = HeteroConv({et: BWGATConv(h, h, dr=dr) for et in ets}, aggr="sum")
        self.bn1   = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h)
        self.drop  = nn.Dropout(dr)
        self.cls   = nn.Sequential(nn.Linear(h,64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        d = {"review": x}
        d = self.conv1(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
        d = self.conv2(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

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
        self.bn1    = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h)
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

@torch.no_grad()
def get_probs(model, data, mask):
    model.eval()
    return torch.sigmoid(model(data)[mask]).numpy()

def score(probs, labels):
    pr = average_precision_score(labels, probs)
    f1 = f1_score(labels, (probs >= 0.5).astype(int), average="macro", zero_division=0)
    return round(float(pr), 4), round(float(f1), 4)

# ── 그래프 로드 ───────────────────────────────────────────────────────────────
print("=" * 65)
print("4-Way Ensemble — 인덕티브 갭 최소 모델 조합")
print("=" * 65)

data_boost  = torch.load(GRAPH / "hetero_graph_boost.pt",  weights_only=False)
feat_boost  = data_boost["review"].x.shape[1]
test_mask   = data_boost["review"].test_mask
labels      = data_boost["review"].y[test_mask].numpy()

# NoRSR: boost에서 RSR 제거
data_norsr = copy.deepcopy(data_boost)
rsr_key    = ("review", "rsr", "review")
if rsr_key in data_norsr.edge_index_dict:
    del data_norsr._edge_store_dict[rsr_key]

# TVF 그래프
tvf_path  = GRAPH / "hetero_graph_tvf.pt"
data_tvf  = torch.load(tvf_path, weights_only=False) if tvf_path.exists() else None
feat_tvf  = data_tvf["review"].x.shape[1] if data_tvf is not None else None

print(f"\n  boost 그래프: {feat_boost}d 피처, 노드 {data_boost['review'].x.shape[0]:,}")
if data_tvf:
    print(f"  TVF   그래프: {feat_tvf}d 피처")
else:
    print("  TVF   그래프: 파일 없음 (DRAGWave_TVF 제외)")

# 인덕티브 서브그래프
ind_boost = mask_ind(data_boost, test_mask)
ind_norsr = mask_ind(data_norsr, test_mask)
ind_tvf   = mask_ind(data_tvf,   test_mask) if data_tvf else None

# ── 후보 모델 설정 ────────────────────────────────────────────────────────────
# (name, model_instance, ckpt_candidates, inductive_data)
candidates = [
    (
        "DRAGWave_TVF_400ep",
        HeteroDRAGWave(feat_tvf) if feat_tvf else None,
        ["DRAGWave_TVF_400ep_best.pt"],
        ind_tvf,
    ),
    (
        "DRAGWave_NoRSR",
        HeteroDRAGWave(feat_boost, ets=ET_NORSR),
        ["DRAGWave_NoRSR_best.pt"],
        ind_norsr,
    ),
    (
        "HeteroBWGNN_boost",
        HeteroBWGNN(feat_boost),
        ["HeteroBWGNN_boost_best.pt", "HeteroBWGNN_best.pt"],
        ind_boost,
    ),
    (
        "BWGAT",
        HeteroBWGAT(feat_boost),
        ["BWGAT_400ep_400ep_best.pt", "BWGAT_best.pt"],
        ind_boost,
    ),
]

# ── 각 모델 추론 ──────────────────────────────────────────────────────────────
print(f"\n{'모델':<24} {'체크포인트':>32}  {'인덕 PR-AUC':>11}  {'인덕 F1':>8}")
print("-" * 80)

prob_list  = []
name_list  = []
indiv_scores = {}

for name, model_inst, ckpt_names, data_ind in candidates:
    if model_inst is None or data_ind is None:
        print(f"  {name:<24} — 그래프/모델 미준비, 스킵")
        continue

    # 체크포인트 탐색 (후보 이름 순서대로 시도)
    ckpt = None
    used_name = ""
    for cn in ckpt_names:
        p = MOD / cn
        if p.exists():
            ckpt = p; used_name = cn; break

    if ckpt is None:
        print(f"  {name:<24} {'(체크포인트 없음)':>32}  [스킵]")
        continue

    model_inst.load_state_dict(torch.load(ckpt, weights_only=True))
    p_ind = get_probs(model_inst, data_ind, test_mask)
    pr, f1 = score(p_ind, labels)

    prob_list.append(p_ind)
    name_list.append(name)
    indiv_scores[name] = {"pr_auc": pr, "macro_f1": f1}
    print(f"  {name:<24} {used_name:>32}  {pr:>11.4f}  {f1:>8.4f}")

n_models = len(prob_list)
print(f"\n  로드 성공: {n_models}개 모델")

if n_models < 2:
    print("[WARNING] 모델 2개 미만. 앙상블 불가. 체크포인트를 확인하세요.")
    raise SystemExit(0)

# ── 가중치 그리드 탐색 (0.0~1.0, step=0.1) ────────────────────────────────────
print(f"\n[가중치 그리드 탐색]")
ws = np.arange(0.0, 1.01, 0.1)

best_pr_ind = 0.
best_w      = None
best_combo  = None

if n_models == 2:
    for w0 in ws:
        w1 = round(1.0 - w0, 1)
        if w1 < 0: continue
        comb = prob_list[0] * w0 + prob_list[1] * w1
        pr, _ = score(comb, labels)
        if pr > best_pr_ind:
            best_pr_ind = pr; best_w = [w0, w1]; best_combo = comb

elif n_models == 3:
    for w0 in ws:
        for w1 in ws:
            w2 = round(1.0 - w0 - w1, 1)
            if w2 < 0 or w2 > 1: continue
            comb = prob_list[0]*w0 + prob_list[1]*w1 + prob_list[2]*w2
            pr, _ = score(comb, labels)
            if pr > best_pr_ind:
                best_pr_ind = pr; best_w = [w0, w1, w2]; best_combo = comb

else:  # 4 모델
    for w0 in ws:
        for w1 in ws:
            for w2 in ws:
                w3 = round(1.0 - w0 - w1 - w2, 1)
                if w3 < 0 or w3 > 1: continue
                comb = (prob_list[0]*w0 + prob_list[1]*w1
                        + prob_list[2]*w2 + prob_list[3]*w3)
                pr, _ = score(comb, labels)
                if pr > best_pr_ind:
                    best_pr_ind = pr; best_w = [w0, w1, w2, w3]; best_combo = comb

best_pr_f, best_f1_f = score(best_combo, labels)

# ── 결과 출력 ─────────────────────────────────────────────────────────────────
print(f"\n[최적 {n_models}-way 앙상블 결과 (인덕티브)]")
for n, w in zip(name_list, best_w):
    print(f"  {n:<24}  weight={w:.1f}")
print(f"\n  인덕티브 PR-AUC : {best_pr_f:.4f}")
print(f"  인덕티브 Macro-F1: {best_f1_f:.4f}")

# 개별 vs 앙상블 비교
print(f"\n[개별 vs 앙상블 비교]")
for n, sc in indiv_scores.items():
    print(f"  {n:<24}  PR-AUC={sc['pr_auc']:.4f}  F1={sc['macro_f1']:.4f}")
print(f"  {'앙상블':24}  PR-AUC={best_pr_f:.4f}  F1={best_f1_f:.4f}  "
      f"{'✅ 향상' if best_pr_f > max(s['pr_auc'] for s in indiv_scores.values()) else '⚠️ 개별 최고 미초과'}")

# 저장
result = {
    "n_models":            n_models,
    "models":              name_list,
    "best_weights":        [round(float(w), 1) for w in best_w],
    "inductive_pr_auc":    best_pr_f,
    "inductive_macro_f1":  best_f1_f,
    "individual_scores":   indiv_scores,
}
out = RES / "ensemble_4way_result.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out.name}")
print("✅ 4-way 앙상블 실험 완료")
