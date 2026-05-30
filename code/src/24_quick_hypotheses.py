"""
24_quick_hypotheses.py
학습 불필요 빠른 가설 검증 (H3, H4, H6)

H3. Simple Heuristic vs GNN
    "Burst 연결 수만으로 분류 vs GNN" — GNN의 가치 정량화

H4. 관계별 앙상블 투표
    "R-U-R은 DRAGWave, Burst는 BWGNN, 결합하면?"
    각 모델이 특정 엣지 신호에 특화되었다는 가설 검증

H6. 캠페인 복구율 평가 (Campaign Recovery Rate)
    "탐지된 사기를 캠페인 단위로 묶었을 때 실제 캠페인이 얼마나 복구되는가"
    PR-AUC(개별 리뷰) 외에 캠페인 단위 탐지 정확도 측정
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
MOD   = BASE / "models"
RES   = BASE / "results"
DEVICE= torch.device("cpu")

EDGE_TYPES=[("review","rtr","review"),("review","rsr","review"),
            ("review","burst","review"),("review","rur","review"),("review","sim","review")]
N_REL=5

# ── 모델 로드 헬퍼 ─────────────────────────────────────────────────────────────
class DualFreqConv(MessagePassing):
    def __init__(self,a,b): super().__init__(aggr="mean"); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei): low=self.propagate(ei,x=x); return self.lin(torch.cat([low,x-low],-1))
    def message(self,x_j): return x_j

class BWGATConv(MessagePassing):
    def __init__(self,a,b,h=4,dr=0.3):
        super().__init__(aggr="add")
        self.gat=GATConv(a,a//h,heads=h,dropout=dr,add_self_loops=False); self.lin=nn.Linear(a*2,b)
    def forward(self,x,ei):
        if ei.shape[1]==0: return self.lin(torch.cat([x,torch.zeros_like(x)],-1))
        low=self.gat(x,ei); return self.lin(torch.cat([low,x-low],-1))

class DRAGWaveConv(nn.Module):
    def __init__(self,a,b,nr,h=4,dr=0.3):
        super().__init__()
        self.bwgat=nn.ModuleList([BWGATConv(a,b,h,dr) for _ in range(nr)])
        self.self_lin=nn.Linear(a,b); self.attn_vec=nn.Linear(b*2,1,bias=False); self.drop=nn.Dropout(dr)
    def forward(self,x,ei_list):
        hs=self.self_lin(x); re=[self.bwgat[i](x,ei) for i,ei in enumerate(ei_list)]
        rs=torch.stack(re,1); he=hs.unsqueeze(1).expand_as(rs)
        aw=F.softmax(self.attn_vec(torch.tanh(torch.cat([he,rs],-1))).squeeze(-1),dim=-1)
        return self.drop(F.relu(hs+(rs*aw.unsqueeze(-1)).sum(1)))

class HeteroDRAGWave(nn.Module):
    def __init__(self,d,h=128,nr=N_REL,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h); self.layer1=DRAGWaveConv(h,h,nr,dr=dr); self.layer2=DRAGWaveConv(h,h,nr,dr=dr)
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h*2,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x)))
        ei=[data.edge_index_dict[et] for et in EDGE_TYPES]
        h1=self.bn1(self.layer1(x,ei)); h2=self.bn2(self.layer2(h1,ei))
        return self.cls(torch.cat([h1,h2],-1)).squeeze(-1)

class HeteroBWGNN(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj=nn.Linear(d,h)
        self.conv1=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.conv2=HeteroConv({et:DualFreqConv(h,h) for et in EDGE_TYPES},aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self,data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

# 데이터 로드
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]
y    = data["review"].y.numpy()
test_mask = data["review"].test_mask
y_test    = y[test_mask.numpy()]

def get_probs(model, data, mask):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(data)[mask]).numpy()

def score(probs, labels):
    pr = round(average_precision_score(labels, probs), 4)
    f1 = round(f1_score(labels, (probs>=0.5).astype(int), average="macro", zero_division=0), 4)
    return pr, f1

results_all = []

# =============================================================================
# H3. Simple Heuristic vs GNN
# =============================================================================
print("="*65)
print("H3. Simple Heuristic Baseline vs GNN")
print("="*65)

burst_ei  = data["review","burst","review"].edge_index.numpy()
sim_ei    = data["review","sim","review"].edge_index.numpy()
rur_ei    = data["review","rur","review"].edge_index.numpy()
N         = len(y)

# 노드별 연결 수 계산
burst_deg = np.bincount(burst_ei[0], minlength=N).astype(float)
sim_deg   = np.bincount(sim_ei[0],   minlength=N).astype(float)
rur_deg   = np.bincount(rur_ei[0],   minlength=N).astype(float)

# Heuristic 1: Burst degree만으로 분류
max_burst = burst_deg.max() + 1
burst_score = burst_deg / max_burst

# Heuristic 2: Burst + Sim 가중합 (XAI 비율 참고)
xai_weights = {"burst": 5.63, "rur": 19.97, "sim": 0.91}
total_w = sum(xai_weights.values())
combo_score = (burst_deg * xai_weights["burst"] +
               rur_deg   * xai_weights["rur"]   +
               sim_deg   * xai_weights["sim"]) / (N * total_w)
combo_score = (combo_score - combo_score.min()) / (combo_score.max() - combo_score.min() + 1e-8)

# Heuristic 3: R-Sim-R 연결 여부만 (복붙 탐지)
sim_flag = (sim_deg > 0).astype(float)

heuristics = {
    "Burst_Degree":          burst_score,
    "XAI_Weighted_Degree":   combo_score,
    "SimR_Flag":             sim_flag,
}

print(f"  {'방법':30s} {'PR-AUC':>8} {'Macro-F1':>10}")
print("  " + "-"*50)
h3_results = []
for name, scores in heuristics.items():
    p = scores[test_mask.numpy()]
    pr, f1 = score(p, y_test)
    print(f"  {name:30s} {pr:>8.4f} {f1:>10.4f}")
    h3_results.append({"method":name,"pr_auc":pr,"macro_f1":f1,"type":"heuristic"})

# GNN 성능 로드 (비교 기준)
df_log = pd.read_csv(RES/"experiment_log.csv")
gnn_pr  = df_log[df_log["model"]=="DRAGWave_400ep"]["pr_auc"].values[0]
gnn_f1  = df_log[df_log["model"]=="DRAGWave_400ep"]["macro_f1"].values[0]
print(f"  {'DRAGWave_400ep (GNN)':30s} {gnn_pr:>8.4f} {gnn_f1:>10.4f}  ← GNN")

best_h = max(h3_results, key=lambda x: x["pr_auc"])
print(f"\n  결론: 최고 휴리스틱({best_h['method']}) vs GNN")
print(f"    PR-AUC 차이: GNN이 +{gnn_pr - best_h['pr_auc']:.4f} 우위")
print(f"    → GNN은 단순 연결 수보다 {(gnn_pr/best_h['pr_auc']-1)*100:.1f}% 더 정확함")

# =============================================================================
# H4. 관계별 앙상블 투표
# XAI 기여도 기반 가중치: RUR 20%, BURST 5.6%, RTR 2.3%, SIM 0.9%, RSR 0.2%
# =============================================================================
print("\n" + "="*65)
print("H4. 관계별 앙상블 투표")
print("="*65)

# 모델 로드
dw_model = HeteroDRAGWave(feat)
bw_model  = HeteroBWGNN(feat)

try:
    dw_model.load_state_dict(torch.load(MOD/"DRAGWave_400ep_best.pt", weights_only=True))
    bw_model.load_state_dict(torch.load(MOD/"HeteroBWGNN_boost_best.pt", weights_only=True))
    p_dw = get_probs(dw_model, data, test_mask)
    p_bw = get_probs(bw_model,  data, test_mask)

    print("  기존 앙상블 방식: 단순 가중 평균")
    combinations = [
        ("DRAGWave 단독",          p_dw,                    None),
        ("BWGNN 단독",             p_bw,                    None),
        ("단순 0.5/0.5",           0.5*p_dw + 0.5*p_bw,     None),
        ("기존 0.7/0.3",           0.7*p_dw + 0.3*p_bw,     None),
        ("XAI 가중(DW↑ BW↓)",     0.6*p_dw + 0.4*p_bw,     None),
    ]

    # 관계별 앙상블: DRAGWave는 Attention 학습 모델이라 관계 분리 가능
    # 실제 관계별 기여를 proxy로 추정: DW가 RUR에, BW가 BURST에 더 강함
    xai = {"rur": 19.97, "burst": 5.63, "rtr": 2.31, "sim": 0.91, "rsr": 0.20}
    total_xai = sum(xai.values())

    # DRAGWave를 높은 기여 관계에, BWGNN을 낮은 기여 관계에 가중
    # 실용적 근사: RUR 고기여 → DW 강조, BURST 중기여 → 균형, SIM/RSR → BW 강조
    # 가중치: DW(RUR+SIM) 0.65, BW(BURST+RTR+RSR) 0.35
    dw_weight = (xai["rur"] + xai["sim"]) / total_xai          # ≈ 0.72
    bw_weight = (xai["burst"] + xai["rtr"] + xai["rsr"]) / total_xai  # ≈ 0.28

    print(f"\n  XAI 기반 이론 가중치: DW={dw_weight:.2f}, BW={bw_weight:.2f}")
    p_xai_ens = dw_weight * p_dw + bw_weight * p_bw
    combinations.append(("XAI 관계 가중 앙상블", p_xai_ens, None))

    print(f"\n  {'방법':30s} {'PR-AUC':>8} {'Macro-F1':>10}")
    print("  " + "-"*52)
    h4_results = []
    for name, probs, _ in combinations:
        pr, f1 = score(probs, y_test)
        mark = " ★ BEST" if pr >= max(score(p,y_test)[0] for _,p,_ in combinations) else ""
        print(f"  {name:30s} {pr:>8.4f} {f1:>10.4f}{mark}")
        h4_results.append({"method":name,"pr_auc":pr,"macro_f1":f1,"type":"ensemble"})

    best_ens = max(h4_results, key=lambda x: x["pr_auc"])
    print(f"\n  결론: 최적 앙상블 = {best_ens['method']} (PR-AUC={best_ens['pr_auc']})")

except Exception as e:
    print(f"  모델 로드 실패: {e}")
    h4_results = []

# =============================================================================
# H6. 캠페인 복구율 평가
# =============================================================================
print("\n" + "="*65)
print("H6. 캠페인 복구율 평가 (Campaign Recovery Rate)")
print("="*65)

camps_path = RES/"fraud_campaigns.csv"
if camps_path.exists() and len(h4_results) > 0:
    df_camps = pd.read_csv(camps_path)
    suspected = df_camps[df_camps["is_campaign"]].copy()

    # DRAGWave 예측 사기 확률 (전체 노드)
    dw_model.eval()
    with torch.no_grad():
        all_probs_dw = torch.sigmoid(dw_model(data)).numpy()

    # 캠페인별 평균 사기 확률 & 탐지 여부
    import ast
    camp_results = []
    for _, row in suspected.iterrows():
        try:
            nodes = ast.literal_eval(row["nodes"])
        except:
            nodes = []
        if not nodes: continue

        valid_nodes = [n for n in nodes if n < len(all_probs_dw)]
        if not valid_nodes: continue

        avg_prob = float(np.mean(all_probs_dw[valid_nodes]))
        actual_spam_ratio = float(row["spam_ratio"])
        detected = avg_prob >= 0.5  # 캠페인 평균 확률 0.5 이상이면 탐지

        camp_results.append({
            "campaign_id":     int(row["campaign_id"]),
            "n_nodes":         int(row["n_nodes"]),
            "actual_spam_ratio": actual_spam_ratio,
            "avg_fraud_prob":  round(avg_prob, 4),
            "detected":        detected,
        })

    df_camp_eval = pd.DataFrame(camp_results)
    n_detected   = df_camp_eval["detected"].sum()
    n_total      = len(df_camp_eval)
    recovery_rate= n_detected / n_total if n_total > 0 else 0

    print(f"  분석 대상 의심 캠페인: {n_total}개")
    print(f"  탐지된 캠페인: {n_detected}개 (복구율 {recovery_rate:.1%})")
    print(f"\n  규모별 복구율:")
    for size_range, label in [((3,10),"소형(3~9)"), ((10,30),"중형(10~29)"), ((30,999),"대형(30+)")]:
        sub = df_camp_eval[(df_camp_eval["n_nodes"]>=size_range[0]) &
                            (df_camp_eval["n_nodes"]<size_range[1])]
        if len(sub) > 0:
            r = sub["detected"].mean()
            print(f"    {label:12s}: {len(sub)}개 중 {sub['detected'].sum()}개 탐지 ({r:.1%})")

    # 실제 스팸 비율 vs 모델 예측 상관관계
    if len(df_camp_eval) > 1:
        corr = np.corrcoef(df_camp_eval["actual_spam_ratio"], df_camp_eval["avg_fraud_prob"])[0,1]
        print(f"\n  실제 스팸 비율 ↔ 모델 예측 상관계수: {corr:.4f}")
        print(f"  → {'강한 양의 상관 (모델이 캠페인 스팸 비율을 잘 예측)' if corr > 0.5 else '보통 상관'}")

    df_camp_eval.to_csv(RES/"campaign_recovery_eval.csv", index=False)
    print(f"\n  저장: results/campaign_recovery_eval.csv")

    h6_result = {
        "n_campaigns_analyzed": n_total,
        "n_detected": int(n_detected),
        "recovery_rate": round(recovery_rate, 4),
        "correlation_spam_vs_prob": round(float(corr), 4) if 'corr' in dir() else None
    }
else:
    print("  캠페인 데이터 없음")
    h6_result = {}

# =============================================================================
# 결과 저장
# =============================================================================
all_results = {
    "H3_heuristic_vs_gnn": {
        "heuristics": h3_results,
        "gnn_pr_auc": float(gnn_pr),
        "gnn_advantage": round(float(gnn_pr) - max(r["pr_auc"] for r in h3_results), 4),
        "conclusion": f"GNN은 최고 휴리스틱 대비 PR-AUC +{float(gnn_pr) - max(r['pr_auc'] for r in h3_results):.4f} 우위"
    },
    "H4_ensemble_voting": {
        "combinations": h4_results,
        "best_method": max(h4_results, key=lambda x: x["pr_auc"])["method"] if h4_results else None,
        "best_pr_auc": max(h4_results, key=lambda x: x["pr_auc"])["pr_auc"] if h4_results else None,
    },
    "H6_campaign_recovery": h6_result,
}

with open(RES/"quick_hypotheses_results.json","w",encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n" + "="*65)
print("=== 빠른 가설 검증 완료 ===")
print("="*65)
print(f"H3 결론: GNN이 최고 휴리스틱보다 PR-AUC +{float(gnn_pr)-max(r['pr_auc'] for r in h3_results):.4f} 높음")
if h4_results:
    best = max(h4_results, key=lambda x: x["pr_auc"])
    print(f"H4 결론: 최적 앙상블 = {best['method']} (PR-AUC={best['pr_auc']})")
if h6_result.get("recovery_rate"):
    print(f"H6 결론: 캠페인 복구율 = {h6_result['recovery_rate']:.1%}")
print(f"\n저장: results/quick_hypotheses_results.json")
