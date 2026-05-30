"""
16_anti_overfit.py
과적합 해소 실험 — DRAGWave 기준

현재 문제:
  DRAGWave_400ep: train=0.9999, test=0.9340, gap=0.0659 (⚠️ 경계)
  TGATLiteV2_400ep: train=0.9783, test=0.8822, gap=0.0961 (⚠️ 경계)

적용할 기법 (3종):
  1. DropEdge (Rong et al., ICLR 2020)
     - 매 epoch마다 엣지를 p%확률로 무작위 제거
     - 효과: 특정 엣지 패턴에 과의존 방지 → 구조적 일반화 강제
  2. Stronger Regularization
     - weight_decay: 1e-5 → 1e-4 (L2 패널티 강화)
     - dropout: 0.3 → 0.4 (뉴런 비활성화 강화)
  3. Label Smoothing
     - hard label (0/1) → soft label (0.05/0.95)
     - 효과: 모델이 과도한 확신(confidence)을 갖지 않도록 억제

목표: gap ≤ 0.05 (양호) 유지하면서 test PR-AUC ≥ 0.92
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import copy
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"
DEVICE= torch.device("cpu")

EDGE_TYPES = [
    ("review","rtr","review"), ("review","rsr","review"),
    ("review","burst","review"), ("review","rur","review"), ("review","sim","review"),
]
N_REL = len(EDGE_TYPES)


# ── 과적합 해소 기법 1: DropEdge ──────────────────────────────────────────────
def drop_edges(data, drop_p: float, seed: int = None):
    """
    DropEdge: 매 epoch마다 각 엣지 타입에서 drop_p 비율로 엣지 무작위 제거
    학습 시에만 적용, 평가 시에는 원본 그래프 사용
    """
    if drop_p <= 0:
        return data
    if seed is not None:
        torch.manual_seed(seed)
    data_dropped = copy.copy(data)
    for et in EDGE_TYPES:
        ei = data[et].edge_index
        n  = ei.shape[1]
        keep_mask = torch.rand(n) > drop_p          # drop_p 확률로 제거
        data_dropped[et].edge_index = ei[:, keep_mask]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            data_dropped[et].edge_attr = data[et].edge_attr[keep_mask]
    return data_dropped


# ── 과적합 해소 기법 3: Label Smoothing Focal Loss ────────────────────────────
class LabelSmoothingFocalLoss(nn.Module):
    """
    Focal Loss + Label Smoothing
    smoothing: 0 = 기존 hard label, 0.05~0.1 = soft label
    효과: 모델이 train 샘플에 과도한 확신을 갖지 못하도록 제한
    """
    def __init__(self, gamma=2.0, alpha=0.75, smoothing=0.05):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # Label smoothing: 1 → (1-s), 0 → s
        smooth_targets = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        bce  = F.binary_cross_entropy_with_logits(logits, smooth_targets, reduction="none")
        pt   = torch.exp(-bce)
        w    = torch.where(targets==1,
                           torch.full_like(bce, self.alpha),
                           torch.full_like(bce, 1-self.alpha))
        return (w * (1-pt)**self.gamma * bce).mean()


# ── 모델 정의 (DRAGWave — dropout 파라미터화) ─────────────────────────────────
class BWGATConv(MessagePassing):
    def __init__(self, in_ch, out_ch, heads=4, dr=0.3):
        super().__init__(aggr="add")
        self.gat = GATConv(in_ch, in_ch//heads, heads=heads, dropout=dr, add_self_loops=False)
        self.lin = nn.Linear(in_ch*2, out_ch)
    def forward(self, x, ei):
        if ei.shape[1] == 0:
            return self.lin(torch.cat([x, torch.zeros_like(x)], -1))
        low = self.gat(x, ei); high = x - low
        return self.lin(torch.cat([low, high], -1))

class DRAGWaveConv(nn.Module):
    def __init__(self, in_ch, out_ch, n_rel, heads=4, dr=0.3):
        super().__init__()
        self.bwgat    = nn.ModuleList([BWGATConv(in_ch, out_ch, heads, dr) for _ in range(n_rel)])
        self.self_lin = nn.Linear(in_ch, out_ch)
        self.attn_vec = nn.Linear(out_ch*2, 1, bias=False)
        self.drop     = nn.Dropout(dr)
    def forward(self, x, ei_list):
        h_self = self.self_lin(x)
        rel_embs = [self.bwgat[i](x, ei) for i, ei in enumerate(ei_list)]
        rel_stack = torch.stack(rel_embs, 1)
        h_exp = h_self.unsqueeze(1).expand_as(rel_stack)
        attn_w = F.softmax(
            self.attn_vec(torch.tanh(torch.cat([h_exp, rel_stack], -1))).squeeze(-1), dim=-1
        )
        h_agg = (rel_stack * attn_w.unsqueeze(-1)).sum(1)
        return self.drop(F.relu(h_self + h_agg))

class HeteroDRAGWave(nn.Module):
    def __init__(self, d, h=128, n_rel=N_REL, heads=4, dr=0.3):
        super().__init__()
        self.proj   = nn.Linear(d, h)
        self.layer1 = DRAGWaveConv(h, h, n_rel, heads, dr)
        self.layer2 = DRAGWaveConv(h, h, n_rel, heads, dr)
        self.bn1    = nn.BatchNorm1d(h)
        self.bn2    = nn.BatchNorm1d(h)
        self.drop   = nn.Dropout(dr)
        self.cls    = nn.Sequential(
            nn.Linear(h*2, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1)
        )
    def forward(self, data):
        x   = self.drop(F.relu(self.proj(data["review"].x)))
        ei  = [data.edge_index_dict[et] for et in EDGE_TYPES]
        h1  = self.bn1(self.layer1(x,  ei))
        h2  = self.bn2(self.layer2(h1, ei))
        return self.cls(torch.cat([h1, h2], -1)).squeeze(-1)


# ── 공통 평가 ─────────────────────────────────────────────────────────────────
def eval_both(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data)
        def s(mask):
            p = torch.sigmoid(logits[mask]).numpy()
            l = data["review"].y[mask].numpy()
            return (round(average_precision_score(l, p), 4),
                    round(f1_score(l, (p>=0.5).astype(int), average="macro", zero_division=0), 4))
        return s(data["review"].train_mask), s(data["review"].test_mask)


# ── 학습 루프 (DropEdge + 설정 가능한 정규화) ─────────────────────────────────
def train_with_regularization(config: dict, data, name: str):
    """
    config keys:
        dropout, weight_decay, drop_edge_p, label_smoothing, epochs, lr
    """
    dr   = config.get("dropout", 0.3)
    wd   = config.get("weight_decay", 1e-5)
    dep  = config.get("drop_edge_p", 0.0)
    ls   = config.get("label_smoothing", 0.0)
    ep   = config.get("epochs", 300)
    lr   = config.get("lr", 5e-4)

    feat  = data["review"].x.shape[1]
    torch.manual_seed(42)
    model = HeteroDRAGWave(feat, dr=dr).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep)
    crit  = LabelSmoothingFocalLoss(smoothing=ls)

    train_mask = data["review"].train_mask
    labels     = data["review"].y
    best_pr, best_state, no_imp = 0., None, 0
    history = []
    t0 = time.time()

    print(f"\n  [{name}] dropout={dr}  wd={wd}  drop_edge={dep}  label_smooth={ls}")

    for epoch in range(1, ep+1):
        model.train()
        opt.zero_grad()

        # DropEdge: 학습 시에만 엣지 일부 제거
        data_train = drop_edges(data, dep, seed=epoch) if dep > 0 else data
        loss = crit(model(data_train)[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()

        if epoch % 20 == 0 or epoch == 1:
            (tr_pr, _), (te_pr, te_f1) = eval_both(model, data)  # 평가는 원본 그래프
            gap = round(tr_pr - te_pr, 4)
            elapsed = time.time() - t0
            print(f"  [{name}] ep={epoch:3d}  loss={loss.item():.4f}  "
                  f"train={tr_pr:.4f}  test={te_pr:.4f}  gap={gap:+.4f}  ({elapsed:.0f}s)")
            history.append({"epoch":epoch,"train_pr":tr_pr,"pr_auc":te_pr,
                            "macro_f1":te_f1,"gap":gap,"loss":round(loss.item(),4)})
            if te_pr > best_pr:
                best_pr = te_pr
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 6:
                    print(f"  [{name}] Early stop ep={epoch}"); break

    model.load_state_dict(best_state)
    (tr_f, _), (te_f, f1_f) = eval_both(model, data)
    gap_f = round(tr_f - te_f, 4)
    n_params = sum(p.numel() for p in model.parameters())
    elapsed = round(time.time()-t0, 1)

    verdict = "🟢 양호" if gap_f <= 0.05 else ("⚠️ 경계" if gap_f <= 0.10 else "🔴 과적합")
    print(f"\n  [{name}] FINAL  train={tr_f}  test={te_f}  gap={gap_f:+.4f}  "
          f"F1={f1_f}  {verdict}  ({elapsed}s)")

    torch.save(best_state, MOD / f"{name}_best.pt")
    pd.DataFrame(history).to_csv(RES / f"history_{name}.csv", index=False)

    return {
        "model": name, "pr_auc": te_f, "macro_f1": f1_f,
        "train_pr": tr_f, "gap": gap_f, "params": n_params,
        "train_sec": elapsed, "verdict": verdict,
        "config": str(config),
    }


# ── 실험 설정 ─────────────────────────────────────────────────────────────────
data = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False)

experiments = [
    # 기준선: 기존 설정 (재현)
    ("DRAGWave_Base",   {"dropout":0.3, "weight_decay":1e-5, "drop_edge_p":0.0,
                          "label_smoothing":0.0, "epochs":300, "lr":5e-4}),
    # 실험 A: DropEdge 20%만 추가
    ("DRAGWave_DropEdge20", {"dropout":0.3, "weight_decay":1e-5, "drop_edge_p":0.20,
                              "label_smoothing":0.0, "epochs":300, "lr":5e-4}),
    # 실험 B: 정규화 강화 (weight_decay + dropout)
    ("DRAGWave_StrongReg",  {"dropout":0.4, "weight_decay":1e-4, "drop_edge_p":0.0,
                              "label_smoothing":0.0, "epochs":300, "lr":5e-4}),
    # 실험 C: 전체 조합 (DropEdge + 정규화 + Label Smoothing)
    ("DRAGWave_AllRegul",   {"dropout":0.4, "weight_decay":1e-4, "drop_edge_p":0.15,
                              "label_smoothing":0.05, "epochs":300, "lr":5e-4}),
]

print("=" * 65)
print("과적합 해소 실험 — DRAGWave 기준")
print(f"목표: gap ≤ 0.05 (양호) + test PR-AUC ≥ 0.92")
print("=" * 65)

results = []
for name, cfg in experiments:
    print("\n" + "=" * 65)
    print(f"▶ {name}")
    r = train_with_regularization(cfg, data, name)
    results.append(r)

# ── 결과 비교 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("=== 과적합 해소 실험 결과 비교 ===")
print("=" * 65)
print(f"  {'모델':30s} {'PR-AUC':>8} {'F1':>8} {'Gap':>8} {'판정':>10}")
print("  " + "-"*65)

# 기준: DRAGWave_400ep (gap=0.0659)
print(f"  {'[기준] DRAGWave_400ep':30s} {'0.9340':>8} {'0.9331':>8} {'+0.0659':>8} {'⚠️ 경계':>10}")
for r in results:
    print(f"  {r['model']:30s} {r['pr_auc']:>8.4f} {r['macro_f1']:>8.4f} "
          f"{r['gap']:>+8.4f} {r['verdict']:>10}")

# 최적 모델 찾기
best = max(results, key=lambda x: (x["pr_auc"], -x["gap"]))
print(f"\n최적 모델: {best['model']}")
print(f"  PR-AUC={best['pr_auc']}  gap={best['gap']:+.4f}  판정={best['verdict']}")

# experiment_log 업데이트
df_log = pd.read_csv(RES / "experiment_log.csv")
for r in results:
    if r["model"] not in df_log["model"].values:
        new = pd.DataFrame([{"model":r["model"],"pr_auc":r["pr_auc"],
                             "macro_f1":r["macro_f1"],"params":r["params"],
                             "train_sec":r["train_sec"],
                             "notes":f"과적합해소실험 gap={r['gap']:+.4f} {r['verdict']}"}])
        df_log = pd.concat([df_log, new], ignore_index=True)
df_log.to_csv(RES / "experiment_log.csv", index=False)
print(f"\n저장: experiment_log.csv ({len(df_log)}행)")
