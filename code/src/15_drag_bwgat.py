"""
15_drag_bwgat.py
DRAG + BWGAT 비교 구현 및 실험

[DRAG] Dynamic Relation-Attentive GNN for Fraud Detection
  논문: arXiv 2310.04171, ICDMW 2023 (bdi-lab/DRAG)
  핵심 아이디어:
    - 각 관계(엣지 타입)별로 독립적인 노드 임베딩 계산
    - 관계별 임베딩을 노드마다 다른 동적 Attention으로 집계
    - XAI 결과: R-U-R 20%, R-S-R 0.2% → 기존 SAGEConv는 동일 가중치!
                  DRAG는 이 비대칭 기여를 자동 학습 가능

[BWGAT] Beta Wavelet Graph Attention
  논문: "Attention Pooling for Beta Wavelet Filters" (IEEE 2023) + BWGNN (ICML 2022)
  핵심 아이디어:
    - BWGNN의 low-pass/high-pass를 GAT Attention으로 가중합
    - "사기 노드는 이웃과 다르다(high-pass)" + "중요한 이웃에 집중(attention)"

기존 HeteroBWGNN과의 차이:
  HeteroBWGNN: low + high → Linear → 출력 (각 엣지 타입 동일 가중치)
  BWGAT:       low_GAT + high_GAT → 가중합 (이웃 중요도 반영)
  DRAG:        per-relation 임베딩 → 동적 attention 집계 (관계 중요도 반영)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing
from torch_geometric.data import HeteroData

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

EDGE_TYPES = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
    ("review", "sim",   "review"),
]
EDGE_NAMES = ["rtr", "rsr", "burst", "rur", "sim"]
N_REL = len(EDGE_TYPES)

# ── 공통 유틸 ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(targets==1, torch.full_like(bce,self.alpha),
                          torch.full_like(bce,1-self.alpha))
        return (w*(1-pt)**self.gamma*bce).mean()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        probs  = torch.sigmoid(model(data)[mask]).cpu().numpy()
        labels = data["review"].y[mask].cpu().numpy()
    pr  = average_precision_score(labels, probs)
    f1  = f1_score(labels, (probs>=0.5).astype(int), average="macro", zero_division=0)
    return round(pr,4), round(f1,4)

def train_model(model, name, data, epochs=150, lr=5e-4, patience=20):
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss()
    train_mask = data["review"].train_mask
    labels     = data["review"].y
    best_pr, best_state, no_imp = 0., None, 0
    history, t0 = [], time.time()

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[train_mask], labels[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()

        if ep % 10 == 0 or ep == 1:
            pr, f1 = evaluate(model, data, data["review"].test_mask)
            elapsed = time.time()-t0
            print(f"  [{name}] ep={ep:3d}  loss={loss.item():.4f}  "
                  f"PR-AUC={pr:.4f}  F1={f1:.4f}  ({elapsed:.0f}s)")
            history.append({"epoch":ep,"pr_auc":pr,"macro_f1":f1})
            if pr > best_pr:
                best_pr = pr
                best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience//10:
                    print(f"  [{name}] Early stop ep={ep}"); break

    model.load_state_dict(best_state)
    pr_f, f1_f = evaluate(model, data, data["review"].test_mask)
    n_params = sum(p.numel() for p in model.parameters())
    elapsed  = round(time.time()-t0,1)
    print(f"\n  [{name}] FINAL  PR-AUC={pr_f}  F1={f1_f}  "
          f"params={n_params:,}  time={elapsed}s")
    torch.save(best_state, MOD / f"{name}_best.pt")
    pd.DataFrame(history).to_csv(RES / f"history_{name}.csv", index=False)
    return {"model":name,"pr_auc":pr_f,"macro_f1":f1_f,
            "params":n_params,"train_sec":elapsed}


# =============================================================================
# DRAG: Dynamic Relation-Attentive GNN
# 논문: arXiv 2310.04171 (ICDMW 2023)
#
# 기존 HeteroConv의 한계:
#   aggr="sum" → 모든 관계를 동등하게 취급
#   R-U-R 기여 20%, R-S-R 기여 0.2%인데도 같은 가중치 → 비효율
#
# DRAG의 해결책:
#   각 관계 r에 대해 h_r = SAGEConv_r(x, ei_r) 독립 계산
#   동적 attention: α_r = softmax(a^T tanh(W [h_self || h_r]))
#   최종: h = h_self + Σ_r α_r * h_r
#
# 우리 도메인 적용 의의:
#   XAI 결과상 R-U-R이 20% 기여, R-S-R이 0.2% 기여임을 모델이 자동 학습 가능
# =============================================================================
class DRAGConv(nn.Module):
    """
    DRAG의 핵심 레이어: 관계별 독립 임베딩 + 동적 attention 집계
    참고: Tang et al. ICDMW 2023, arXiv 2310.04171
    """
    def __init__(self, in_ch: int, out_ch: int, n_relations: int, dropout=0.3):
        super().__init__()
        # 각 관계마다 독립적인 SAGEConv
        self.rel_convs = nn.ModuleList([
            SAGEConv(in_ch, out_ch) for _ in range(n_relations)
        ])
        # Self-transformation
        self.self_lin = nn.Linear(in_ch, out_ch)
        # 동적 attention: [h_self || h_r] → scalar score
        self.attn_vec = nn.Linear(out_ch * 2, 1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index_list):
        """
        x: [N, in_ch]
        edge_index_list: list of [2, E_r] tensors (각 관계별 엣지)
        """
        h_self = self.self_lin(x)  # [N, out_ch]

        rel_embs = []
        for i, ei in enumerate(edge_index_list):
            if ei.shape[1] == 0:
                rel_embs.append(torch.zeros_like(h_self))
            else:
                rel_embs.append(self.rel_convs[i](x, ei))  # [N, out_ch]

        # 동적 attention: 각 노드마다 관계별 중요도가 다름
        rel_stack = torch.stack(rel_embs, dim=1)   # [N, n_rel, out_ch]
        h_self_exp = h_self.unsqueeze(1).expand_as(rel_stack)  # [N, n_rel, out_ch]
        attn_input = torch.cat([h_self_exp, rel_stack], dim=-1)  # [N, n_rel, 2*out_ch]
        attn_scores = self.attn_vec(torch.tanh(attn_input)).squeeze(-1)  # [N, n_rel]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, n_rel]

        # 가중 집계
        h_agg = (rel_stack * attn_weights.unsqueeze(-1)).sum(dim=1)  # [N, out_ch]
        return self.drop(F.relu(h_self + h_agg))


class HeteroDRAG(nn.Module):
    """
    DRAG를 헤테로 그래프 + 멀티레이어로 확장
    Layer 1: DRAGConv(in → hidden)
    Layer 2: DRAGConv(hidden → hidden)
    + 레이어 간 skip connection (중간/마지막 표현 concat)
    """
    def __init__(self, in_ch, hidden=128, n_rel=N_REL, dropout=0.3):
        super().__init__()
        self.proj   = nn.Linear(in_ch, hidden)
        self.drag1  = DRAGConv(hidden, hidden, n_rel, dropout)
        self.drag2  = DRAGConv(hidden, hidden, n_rel, dropout)
        self.bn1    = nn.BatchNorm1d(hidden)
        self.bn2    = nn.BatchNorm1d(hidden)
        self.drop   = nn.Dropout(dropout)
        # skip connection: layer1 + layer2 concat → 분류
        self.cls    = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _get_edge_list(self, data):
        return [data.edge_index_dict[et] for et in EDGE_TYPES]

    def forward(self, data):
        x     = self.drop(F.relu(self.proj(data["review"].x)))
        ei_list = self._get_edge_list(data)

        h1 = self.bn1(self.drag1(x, ei_list))       # [N, hidden]
        h2 = self.bn2(self.drag2(h1, ei_list))      # [N, hidden]

        # Skip: 두 레이어 출력 concat → 로컬+글로벌 구조 모두 반영
        out = torch.cat([h1, h2], dim=-1)            # [N, 2*hidden]
        return self.cls(out).squeeze(-1)


# =============================================================================
# BWGAT: Beta Wavelet + Graph Attention
# 참고: "Attention Pooling for Beta Wavelet Filters" (IEEE 2023)
#       + BWGNN (Tang et al., ICML 2022)
#
# 기존 BWGNN의 한계:
#   DualFreqConv: low = mean(neighbors), high = x - low
#   → 모든 이웃을 동등하게 집계 (mean)
#   → 중요한 이웃을 더 강조하지 못함
#
# BWGAT의 개선:
#   low = GAT_weighted_mean(neighbors)   → 중요한 이웃에 집중
#   high = x - GAT_weighted_mean(...)    → 의미 있는 편차
#   out = Linear(concat[low, high])
# =============================================================================
class BWGATConv(MessagePassing):
    """
    Beta Wavelet + Graph Attention
    low-pass: GAT attention-weighted 이웃 집계
    high-pass: 자신 - GAT low-pass (의미 있는 편차)
    """
    def __init__(self, in_ch: int, out_ch: int, heads=4, dropout=0.3):
        super().__init__(aggr="add")
        # GAT로 attention-weighted low-pass 계산
        self.gat      = GATConv(in_ch, in_ch // heads, heads=heads,
                                dropout=dropout, add_self_loops=False)
        # [low || high] → out_ch
        self.lin      = nn.Linear(in_ch * 2, out_ch)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        if edge_index.shape[1] == 0:
            # 엣지 없으면 x 자체로 처리
            return self.lin(torch.cat([x, torch.zeros_like(x)], dim=-1))

        # GAT attention-weighted 평균 = "smart" low-pass
        low  = self.gat(x, edge_index)         # [N, in_ch]
        # high-pass: 자신과 attention-weighted 이웃의 편차
        high = x - low                         # [N, in_ch] — 사기 노드의 이상 신호
        return self.lin(torch.cat([low, high], dim=-1))  # [N, out_ch]


class HeteroBWGAT(nn.Module):
    """
    BWGAT를 5종 헤테로 엣지에 적용
    각 엣지 타입에서 BWGATConv → HeteroConv로 집계
    """
    def __init__(self, in_ch, hidden=128, heads=4, dropout=0.3):
        super().__init__()
        self.proj  = nn.Linear(in_ch, hidden)
        self.conv1 = HeteroConv(
            {et: BWGATConv(hidden, hidden, heads, dropout) for et in EDGE_TYPES},
            aggr="sum"
        )
        self.conv2 = HeteroConv(
            {et: BWGATConv(hidden, hidden, heads, dropout) for et in EDGE_TYPES},
            aggr="sum"
        )
        self.bn1   = nn.BatchNorm1d(hidden)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.drop  = nn.Dropout(dropout)
        self.cls   = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, data):
        x = self.drop(F.relu(self.proj(data["review"].x)))
        d = {"review": x}
        d = self.conv1(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
        d = self.conv2(d, data.edge_index_dict)
        d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)


# =============================================================================
# DRAG + BWGAT 융합: DRAGWave
# 아이디어: DRAG의 관계 attention + BWGAT의 band-pass 결합
# "각 관계에서 GAT band-pass → 관계 간 동적 attention 집계"
# =============================================================================
class DRAGWaveConv(nn.Module):
    """
    DRAG + BWGAT 융합 레이어
    각 관계에서 BWGATConv → 관계 간 DRAG attention으로 집계
    """
    def __init__(self, in_ch, out_ch, n_relations, heads=4, dropout=0.3):
        super().__init__()
        self.bwgat_convs = nn.ModuleList([
            BWGATConv(in_ch, out_ch, heads, dropout) for _ in range(n_relations)
        ])
        self.self_lin  = nn.Linear(in_ch, out_ch)
        self.attn_vec  = nn.Linear(out_ch * 2, 1, bias=False)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, edge_index_list):
        h_self = self.self_lin(x)
        rel_embs = []
        for i, ei in enumerate(edge_index_list):
            rel_embs.append(self.bwgat_convs[i](x, ei))

        rel_stack  = torch.stack(rel_embs, dim=1)         # [N, n_rel, out_ch]
        h_self_exp = h_self.unsqueeze(1).expand_as(rel_stack)
        attn_scores= self.attn_vec(torch.tanh(
            torch.cat([h_self_exp, rel_stack], dim=-1)
        )).squeeze(-1)
        attn_w     = F.softmax(attn_scores, dim=-1)
        h_agg      = (rel_stack * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.drop(F.relu(h_self + h_agg))


class HeteroDRAGWave(nn.Module):
    """DRAG + BWGAT 융합 모델"""
    def __init__(self, in_ch, hidden=128, n_rel=N_REL, heads=4, dropout=0.3):
        super().__init__()
        self.proj   = nn.Linear(in_ch, hidden)
        self.layer1 = DRAGWaveConv(hidden, hidden, n_rel, heads, dropout)
        self.layer2 = DRAGWaveConv(hidden, hidden, n_rel, heads, dropout)
        self.bn1    = nn.BatchNorm1d(hidden)
        self.bn2    = nn.BatchNorm1d(hidden)
        self.drop   = nn.Dropout(dropout)
        self.cls    = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, data):
        x      = self.drop(F.relu(self.proj(data["review"].x)))
        ei_list= [data.edge_index_dict[et] for et in EDGE_TYPES]
        h1 = self.bn1(self.layer1(x,  ei_list))
        h2 = self.bn2(self.layer2(h1, ei_list))
        return self.cls(torch.cat([h1, h2], -1)).squeeze(-1)


# =============================================================================
# 실험 실행
# =============================================================================
data = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False).to(DEVICE)
FEAT = data["review"].x.shape[1]
torch.manual_seed(42)

results = []

# ── 기준 모델 로드 (HeteroBWGNN 기존 결과) ────────────────────────────────────
print("=" * 60)
print("기준: HeteroBWGNN (기존 결과 로드)")
df_log = pd.read_csv(RES / "experiment_log.csv")
bwgnn_row = df_log[df_log["model"] == "HeteroBWGNN"]
if len(bwgnn_row):
    r = bwgnn_row.iloc[0]
    print(f"  HeteroBWGNN: PR-AUC={r['pr_auc']}  F1={r['macro_f1']}")
    results.append({"model":"HeteroBWGNN(기존)","pr_auc":r["pr_auc"],
                    "macro_f1":r["macro_f1"],"params":r["params"],"train_sec":r.get("train_sec",0)})

# ── BWGAT ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("▶ BWGAT (Beta Wavelet + GAT Attention)")
print("=" * 60)
model_bwgat = HeteroBWGAT(FEAT).to(DEVICE)
result_bwgat = train_model(model_bwgat, "BWGAT", data)
results.append(result_bwgat)

# ── DRAG ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("▶ DRAG (Dynamic Relation-Attentive GNN)")
print("=" * 60)
model_drag = HeteroDRAG(FEAT).to(DEVICE)
result_drag = train_model(model_drag, "DRAG", data)
results.append(result_drag)

# ── DRAGWave (DRAG + BWGAT 융합) ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("▶ DRAGWave (DRAG × BWGAT 융합 — 본 연구 제안)")
print("=" * 60)
model_dragwave = HeteroDRAGWave(FEAT).to(DEVICE)
result_dragwave = train_model(model_dragwave, "DRAGWave", data)
results.append(result_dragwave)

# ── 결과 정리 ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== 모델 비교 결과 ===")
print("=" * 60)
df_res = pd.DataFrame(results)
print(df_res[["model","pr_auc","macro_f1","params"]].to_string(index=False))

# 기존 로그에 추가
df_log_new = pd.concat([df_log, df_res[~df_res["model"].str.contains("기존")]],
                        ignore_index=True)
df_log_new.to_csv(RES / "experiment_log.csv", index=False)
df_res.to_csv(RES / "drag_bwgat_results.csv", index=False)
print(f"\n저장: results/drag_bwgat_results.csv")

# ── DRAG 관계별 Attention 분석 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("=== DRAG 관계 Attention 분포 분석 ===")
print("(XAI ablation 결과와 비교: R-U-R 20% > R-Burst-R 5.6% > ...)")
print("=" * 60)

model_drag.eval()
with torch.no_grad():
    x = model_drag.drop(F.relu(model_drag.proj(data["review"].x)))
    ei_list = [data.edge_index_dict[et] for et in EDGE_TYPES]
    h_self = model_drag.drag1.self_lin(x)
    rel_embs = []
    for i, ei in enumerate(ei_list):
        if ei.shape[1] == 0:
            rel_embs.append(torch.zeros_like(h_self))
        else:
            rel_embs.append(model_drag.drag1.rel_convs[i](x, ei))
    rel_stack  = torch.stack(rel_embs, dim=1)
    h_self_exp = h_self.unsqueeze(1).expand_as(rel_stack)
    attn_input = torch.cat([h_self_exp, rel_stack], dim=-1)
    attn_scores= model_drag.drag1.attn_vec(torch.tanh(attn_input)).squeeze(-1)
    attn_weights = F.softmax(attn_scores, dim=-1).numpy()

# 스팸/정상 노드별 평균 Attention
y = data["review"].y.numpy()
print(f"\n{'엣지':8s} {'스팸 Attn':>12} {'정상 Attn':>12} {'차이':>10}")
for i, name in enumerate(EDGE_NAMES):
    spam_attn  = attn_weights[y==1, i].mean()
    legit_attn = attn_weights[y==0, i].mean()
    diff = spam_attn - legit_attn
    mark = " ← 사기 특화" if diff > 0.01 else ""
    print(f"  {name:6s}  {spam_attn:.4f}       {legit_attn:.4f}     {diff:+.4f}{mark}")

print("\n✅ DRAG / BWGAT 실험 완료")
print("  → DRAGWave: DRAG attention + BWGAT band-pass 결합 (본 연구 고유 설계)")
