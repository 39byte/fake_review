"""
06_eval_inductive.py
방법 A: test 추론 시 train→test 엣지 차단
test 노드끼리만 연결된 엣지로 subgraph 구성 후 재평가
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import copy
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"
MOD   = BASE / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 모델 정의 (04, 05 와 동일) ────────────────────────────────────────────────
EDGE_TYPES = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
]
EDGE_TYPES_NO_BURST = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "rur",   "review"),
]
FEAT_DIM = 388
HIDDEN   = 128
D_TIME   = 64
HEADS    = 4

class DualFreqConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_ch * 2, out_ch)
    def forward(self, x, edge_index):
        low  = self.propagate(edge_index, x=x)
        high = x - low
        return self.lin(torch.cat([low, high], dim=-1))
    def message(self, x_j):
        return x_j

class HeteroSAGE(nn.Module):
    def __init__(self, in_ch, hidden, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        x_dict = {"review": x}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn1(x_dict["review"])))}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn2(x_dict["review"])))}
        return self.cls(x_dict["review"]).squeeze(-1)

class HeteroGAT(nn.Module):
    def __init__(self, in_ch, hidden, heads=4, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv({et: GATConv(hidden, hidden//heads, heads=heads, dropout=dropout, add_self_loops=False) for et in EDGE_TYPES}, aggr="sum")
        self.conv2 = HeteroConv({et: GATConv(hidden, hidden//heads, heads=heads, dropout=dropout, add_self_loops=False) for et in EDGE_TYPES}, aggr="sum")
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        x_dict = {"review": x}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn1(x_dict["review"])))}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn2(x_dict["review"])))}
        return self.cls(x_dict["review"]).squeeze(-1)

class HeteroBWGNN(nn.Module):
    def __init__(self, in_ch, hidden, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv({et: DualFreqConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.conv2 = HeteroConv({et: DualFreqConv(hidden, hidden) for et in EDGE_TYPES}, aggr="sum")
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        x_dict = {"review": x}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn1(x_dict["review"])))}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {"review": self.drop(F.relu(self.bn2(x_dict["review"])))}
        return self.cls(x_dict["review"]).squeeze(-1)

class BochnerTimeEncoder(nn.Module):
    def __init__(self, d_time=64):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(d_time // 2))
    def forward(self, delta_t):
        delta_t = delta_t.squeeze(-1) if delta_t.dim() == 2 else delta_t
        t = delta_t.unsqueeze(-1) * self.omega.unsqueeze(0)
        return torch.cat([torch.cos(t), torch.sin(t)], dim=-1)

class TimeAwareConv(nn.Module):
    def __init__(self, in_ch, out_ch, d_time, heads):
        super().__init__()
        self.time_proj = nn.Linear(d_time, in_ch)
        self.gat = GATConv(in_ch, out_ch // heads, heads=heads, dropout=0.3, add_self_loops=False)
        self.in_ch = in_ch
    def forward(self, x, edge_index, time_emb):
        if edge_index.shape[1] == 0:
            return torch.zeros(x.shape[0], self.gat.out_channels * self.gat.heads, device=x.device)
        time_feat = self.time_proj(time_emb)
        src = edge_index[0]
        x_boosted = x.clone()
        x_boosted.scatter_add_(0, src.unsqueeze(-1).expand(-1, self.in_ch), time_feat)
        return self.gat(x_boosted, edge_index)

class TGATLite(nn.Module):
    def __init__(self, in_ch, hidden, d_time=D_TIME, heads=HEADS, dropout=0.3):
        super().__init__()
        self.time_encoder = BochnerTimeEncoder(d_time)
        self.proj = nn.Linear(in_ch, hidden)
        self.conv1_nonburst = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum")
        self.time_conv1 = TimeAwareConv(hidden, hidden, d_time, heads)
        self.conv2_nonburst = HeteroConv({et: SAGEConv(hidden, hidden) for et in EDGE_TYPES_NO_BURST}, aggr="sum")
        self.time_conv2 = TimeAwareConv(hidden, hidden, d_time, heads)
        self.bn1 = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Sequential(nn.Linear(hidden*2,64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64,1))
    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        burst_ei   = data["review", "burst", "review"].edge_index
        delta_t    = data["review", "burst", "review"].edge_attr.squeeze(-1)
        time_emb   = self.time_encoder(delta_t) if burst_ei.shape[1] > 0 else torch.zeros(0, D_TIME, device=x.device)
        nb_ei = {et: data.edge_index_dict[et] for et in EDGE_TYPES_NO_BURST}
        x_dict = self.conv1_nonburst({"review": x}, nb_ei)
        x_burst1 = self.time_conv1(x, burst_ei, time_emb)
        x1 = self.drop(F.relu(self.bn1(x_dict["review"] + x_burst1)))
        x_dict2 = self.conv2_nonburst({"review": x1}, nb_ei)
        x_burst2 = self.time_conv2(x1, burst_ei, time_emb)
        x2 = self.drop(F.relu(self.bn2(x_dict2["review"] + x_burst2)))
        return self.cls(torch.cat([x_dict2["review"], x_burst2], dim=-1)).squeeze(-1)


# ── 핵심: test-only 엣지 마스킹 ───────────────────────────────────────────────
def mask_to_test_only(data: HeteroData, test_mask: torch.Tensor) -> HeteroData:
    """
    test 추론용 데이터: 양 끝점이 모두 test 노드인 엣지만 유지
    train→test, test→train, train→train 엣지를 모두 제거
    """
    data_test = copy.deepcopy(data)
    for et in EDGE_TYPES:
        ei = data[et].edge_index
        src, dst = ei[0], ei[1]
        mask = test_mask[src] & test_mask[dst]  # 양 끝이 test인 엣지만
        data_test[et].edge_index = ei[:, mask]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            data_test[et].edge_attr = data[et].edge_attr[mask]
    return data_test


def evaluate_inductive(model, data_full, data_test_only, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(data_test_only)  # test-only 그래프로 추론
        probs  = torch.sigmoid(logits[test_mask]).cpu().numpy()
        labels = data_full["review"].y[test_mask].cpu().numpy()
    pr_auc   = average_precision_score(labels, probs)
    preds    = (probs >= 0.5).astype(int)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"PR-AUC": round(pr_auc, 4), "Macro-F1": round(macro_f1, 4)}


# ── 실행 ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("방법 A: Test-only 엣지 마스킹 재평가")
print("=" * 60)

data_full = torch.load(GRAPH / "hetero_graph.pt", weights_only=False).to(DEVICE)
test_mask = data_full["review"].test_mask

# test-only 서브그래프 생성
data_test_only = mask_to_test_only(data_full, test_mask).to(DEVICE)

# 엣지 수 비교
print("\n[엣지 수 비교]")
for et in EDGE_TYPES:
    full_n = data_full[et].edge_index.shape[1]
    test_n = data_test_only[et].edge_index.shape[1]
    print(f"  {et[1]:8s}  전체={full_n:>7,}  test-only={test_n:>7,}  "
          f"({test_n/full_n*100:.1f}%)")

# 모델 목록
model_classes = {
    "HeteroSAGE":  HeteroSAGE(FEAT_DIM, HIDDEN),
    "HeteroGAT":   HeteroGAT(FEAT_DIM, HIDDEN),
    "HeteroBWGNN": HeteroBWGNN(FEAT_DIM, HIDDEN),
    "TGATLite":    TGATLite(FEAT_DIM, HIDDEN),
}

# 기존 로그 로드
df_log = pd.read_csv(RES / "experiment_log.csv")

print("\n[재평가 결과]")
print(f"{'모델':<14} {'기존 PR-AUC':>12} {'기존 F1':>10} {'새 PR-AUC':>10} {'새 F1':>10}")
print("-" * 60)

inductive_rows = []
for name, model_inst in model_classes.items():
    pt_path = MOD / f"{name}_best.pt"
    if not pt_path.exists():
        print(f"  {name}: 모델 파일 없음, 스킵")
        continue
    state = torch.load(pt_path, weights_only=True)
    model_inst.load_state_dict(state)
    model_inst = model_inst.to(DEVICE)

    new_metrics = evaluate_inductive(model_inst, data_full, data_test_only, test_mask)

    old_row = df_log[df_log["model"] == name]
    old_pr  = old_row["pr_auc"].values[0]  if len(old_row) else 0.0
    old_f1  = old_row["macro_f1"].values[0] if len(old_row) else 0.0

    print(f"  {name:<14} {old_pr:>12.4f} {old_f1:>10.4f} "
          f"{new_metrics['PR-AUC']:>10.4f} {new_metrics['Macro-F1']:>10.4f}")

    inductive_rows.append({
        "model":        name,
        "pr_auc":       new_metrics["PR-AUC"],
        "macro_f1":     new_metrics["Macro-F1"],
        "params":       old_row["params"].values[0] if len(old_row) else 0,
        "train_sec":    old_row["train_sec"].values[0] if len(old_row) else 0,
        "notes":        "test-only 엣지 마스킹 (방법 A 인덕티브 평가)",
    })

# inductive 결과 저장
df_inductive = pd.DataFrame(inductive_rows)
df_inductive.to_csv(RES / "experiment_log_inductive.csv", index=False)

print("\n=== 최종 요약 ===")
best = df_inductive.loc[df_inductive["pr_auc"].idxmax()]
print(f"베스트: {best['model']}  PR-AUC={best['pr_auc']}  Macro-F1={best['macro_f1']}")
go_nogo = "GO ✅" if best["pr_auc"] >= 0.70 else "NO-GO ❌"
print(f"Go/No-Go (PR-AUC ≥ 0.70): {go_nogo}")
print(f"\n저장: {RES / 'experiment_log_inductive.csv'}")
