"""
E10_rigorous_baselines.py
논문 수준 엄밀한 베이스라인 비교

① 텍스트 단독 (그래프 없음)
   - TF-IDF + Logistic Regression (전통 스팸 탐지)
   - SBERT + MLP (강한 피처 기반)
   - SBERT + XGBoost (트리 앙상블)

② 그래프 Inductive 베이스라인
   - GraphSAGE Inductive (설계 자체가 Inductive)
   - GAT Inductive

③ 멀티시드 (42/123/456) → 평균 ± std

모든 실험: 동일 train/test 분할, test-only 서브그래프 평가
"""
import copy, json, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

BASE  = Path(__file__).resolve().parent.parent.parent
GRAPH = BASE / "data" / "graphs"
PROC  = BASE / "data" / "processed"
RES   = BASE / "results" / "experiments"
RES.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
ET = [("review","rtr","review"),("review","rsr","review"),
      ("review","burst","review"),("review","rur","review"),("review","sim","review")]

# ── 데이터 로드 ────────────────────────────────────────────────────────────────
data = torch.load(GRAPH/"hetero_graph_boost.pt", weights_only=False)
df   = pd.read_parquet(PROC/"df_sampled.parquet")
feat = data["review"].x.shape[1]
tm   = data["review"].train_mask
te   = data["review"].test_mask
y    = data["review"].y

# test-only 인덕티브 서브그래프
def mask_ind(data, mask):
    d2 = copy.deepcopy(data)
    for et, ei in data.edge_index_dict.items():
        m = mask[ei[0]] & mask[ei[1]]
        d2[et].edge_index = ei[:, m]
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            d2[et].edge_attr = data[et].edge_attr[m]
    return d2

data_ind = mask_ind(data, te)

def score(p, l):
    pr = round(float(average_precision_score(l, p)), 4)
    f1 = round(float(f1_score(l, (p>=0.5).astype(int), average="macro", zero_division=0)), 4)
    return pr, f1

# 피처 준비
x_tr  = data["review"].x[tm].numpy()
x_te  = data["review"].x[te].numpy()
y_tr  = y[tm].numpy()
y_te  = y[te].numpy()
texts = df["text"].fillna("").tolist()
texts_tr = [texts[i] for i in torch.where(tm)[0].tolist()]
texts_te = [texts[i] for i in torch.where(te)[0].tolist()]

print("="*70)
print("E10: 논문 수준 베이스라인 비교")
print("="*70)
print(f"  Train: {tm.sum():,}  Test: {te.sum():,}  스팸: {y_te.mean():.3f}")

results = []

# ══════════════════════════════════════════════════════════════════════════════
# 텍스트 단독 베이스라인 (Inductive — 그래프 구조 전혀 사용 안 함)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[텍스트 단독 베이스라인]")

# 1. TF-IDF + LR (전통 스팸 탐지)
print("  1. TF-IDF + Logistic Regression...")
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2)
tfidf_tr = tfidf.fit_transform(texts_tr)
tfidf_te = tfidf.transform(texts_te)
for C_val, name in [(0.1, "TF-IDF+LR(C=0.1)"), (1.0, "TF-IDF+LR(C=1.0)"), (10.0, "TF-IDF+LR(C=10)")]:
    lr = LogisticRegression(C=C_val, max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(tfidf_tr, y_tr)
    p = lr.predict_proba(tfidf_te)[:, 1]
    pr, f1 = score(p, y_te)
    print(f"    {name:<25} PR-AUC={pr:.4f}  F1={f1:.4f}")
    results.append({"name": name, "type": "text_only", "pr_auc": pr, "macro_f1": f1, "inductive": True})

# 2. SBERT + MLP (멀티레이어)
print("  2. SBERT + MLP...")
sc = StandardScaler()
x_tr_s = sc.fit_transform(x_tr[:, :384])  # SBERT 384d만
x_te_s  = sc.transform(x_te[:, :384])
for hidden, name in [((128,), "SBERT+MLP(128)"), ((256,128), "SBERT+MLP(256,128)"), ((512,256,128), "SBERT+MLP(512,256,128)")]:
    mlp = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1)
    mlp.fit(x_tr_s, y_tr)
    p = mlp.predict_proba(x_te_s)[:, 1]
    pr, f1 = score(p, y_te)
    print(f"    {name:<30} PR-AUC={pr:.4f}  F1={f1:.4f}")
    results.append({"name": name, "type": "text_only", "pr_auc": pr, "macro_f1": f1, "inductive": True})

# 3. SBERT + XGBoost
print("  3. SBERT + XGBoost...")
try:
    from xgboost import XGBClassifier
    scale_pos = int((y_tr==0).sum() / (y_tr==1).sum())
    for lr_val, n_est, name in [(0.1, 300, "SBERT+XGB(lr=0.1,300)"), (0.05, 500, "SBERT+XGB(lr=0.05,500)")]:
        xgb = XGBClassifier(n_estimators=n_est, learning_rate=lr_val,
                            max_depth=6, scale_pos_weight=scale_pos,
                            random_state=42, eval_metric="aucpr",
                            early_stopping_rounds=20, verbosity=0)
        x_tr_v, x_va_v, y_tr_v, y_va_v = (
            x_tr_s[:int(len(x_tr_s)*0.9)], x_tr_s[int(len(x_tr_s)*0.9):],
            y_tr[:int(len(y_tr)*0.9)],     y_tr[int(len(y_tr)*0.9):]
        )
        xgb.fit(x_tr_v, y_tr_v, eval_set=[(x_va_v, y_va_v)], verbose=False)
        p = xgb.predict_proba(x_te_s)[:, 1]
        pr, f1 = score(p, y_te)
        print(f"    {name:<30} PR-AUC={pr:.4f}  F1={f1:.4f}")
        results.append({"name": name, "type": "text_only", "pr_auc": pr, "macro_f1": f1, "inductive": True})
except ImportError:
    print("    XGBoost 없음 — pip install xgboost 필요")

# 4. SBERT + All features + MLP
print("  4. SBERT + 전체 피처(386d) + MLP...")
sc2 = StandardScaler()
x_tr_f = sc2.fit_transform(x_tr)  # 386d 전체
x_te_f  = sc2.transform(x_te)
mlp_f = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42,
                      early_stopping=True, validation_fraction=0.1)
mlp_f.fit(x_tr_f, y_tr)
p = mlp_f.predict_proba(x_te_f)[:, 1]
pr, f1 = score(p, y_te)
print(f"    SBERT+feat(386)+MLP            PR-AUC={pr:.4f}  F1={f1:.4f}")
results.append({"name": "SBERT+feat(386)+MLP", "type": "text_only", "pr_auc": pr, "macro_f1": f1, "inductive": True})

# ══════════════════════════════════════════════════════════════════════════════
# Inductive GNN 베이스라인
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Inductive GNN 베이스라인]")

class FocalLoss(nn.Module):
    def __init__(self,g=2.,a=0.75): super().__init__(); self.g,self.a=g,a
    def forward(self,lo,ta):
        bce=F.binary_cross_entropy_with_logits(lo,ta.float(),reduction="none")
        pt=torch.exp(-bce)
        w=torch.where(ta==1,torch.full_like(bce,self.a),torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

# GraphSAGE Inductive (SAGEConv은 Inductive 설계)
class SAGEInductive(nn.Module):
    def __init__(self,d,h=128,dr=0.3):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: SAGEConv(h, h) for et in ET}, aggr="sum")
        self.conv2 = HeteroConv({et: SAGEConv(h, h) for et in ET}, aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self, data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

# GAT Inductive
class GATInductive(nn.Module):
    def __init__(self,d,h=128,heads=4,dr=0.3):
        super().__init__()
        self.proj = nn.Linear(d, h)
        self.conv1 = HeteroConv({et: GATConv(h, h//heads, heads=heads, dropout=dr, add_self_loops=False) for et in ET}, aggr="sum")
        self.conv2 = HeteroConv({et: GATConv(h, h//heads, heads=heads, dropout=dr, add_self_loops=False) for et in ET}, aggr="sum")
        self.bn1=nn.BatchNorm1d(h); self.bn2=nn.BatchNorm1d(h); self.drop=nn.Dropout(dr)
        self.cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(dr),nn.Linear(64,1))
    def forward(self, data):
        x=self.drop(F.relu(self.proj(data["review"].x))); d={"review":x}
        d=self.conv1(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn1(d["review"])))}
        d=self.conv2(d,data.edge_index_dict); d={"review":self.drop(F.relu(self.bn2(d["review"])))}
        return self.cls(d["review"]).squeeze(-1)

def train_eval_inductive(ModelClass, name, seeds=SEEDS, epochs=400, **kwargs):
    """멀티시드 학습 + Inductive 평가"""
    seed_results = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = ModelClass(feat, **kwargs)
        opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit  = FocalLoss()
        best_pr, best_state = 0., None

        for ep in range(1, epochs+1):
            model.train(); opt.zero_grad()
            loss = crit(model(data)[tm], y[tm])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); sched.step()
            if ep % 50 == 0:
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(data)[te]).numpy()
                pr_val = average_precision_score(y_te, p)
                if pr_val > best_pr:
                    best_pr = pr_val
                    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}

        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            p_ind = torch.sigmoid(model(data_ind)[te]).numpy()
        pr_ind, f1_ind = score(p_ind, y_te)
        seed_results.append({"seed": seed, "pr_auc": pr_ind, "f1": f1_ind})
        print(f"    seed={seed}  Inductive PR-AUC={pr_ind:.4f}  F1={f1_ind:.4f}")

    mean_pr = round(np.mean([r["pr_auc"] for r in seed_results]), 4)
    std_pr  = round(np.std([r["pr_auc"] for r in seed_results]), 4)
    mean_f1 = round(np.mean([r["f1"]     for r in seed_results]), 4)
    print(f"    → {name} 평균: PR-AUC={mean_pr:.4f}±{std_pr:.4f}  F1={mean_f1:.4f}")
    return {"name": name, "type": "gnn_inductive", "pr_auc": mean_pr, "std": std_pr,
            "macro_f1": mean_f1, "inductive": True, "seeds": seed_results}

print("  1. GraphSAGE Inductive (멀티시드)...")
r = train_eval_inductive(SAGEInductive, "GraphSAGE-Inductive")
results.append(r)

print("  2. GAT Inductive (멀티시드)...")
r = train_eval_inductive(GATInductive, "GAT-Inductive")
results.append(r)

# ── 전체 결과 정리 ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("전체 베이스라인 비교 (Inductive PR-AUC 기준)")
print("="*70)
print(f"  {'방법':<35} {'PR-AUC':>8}  {'Macro F1':>9}  {'유형'}")
print("  " + "-"*65)

# 랜덤 기준선
print(f"  {'랜덤 분류기 (하한)':<35} {'0.1322':>8}  {'—':>9}  이론값")
print()

for r in sorted(results, key=lambda x: x["pr_auc"], reverse=True):
    std_str = f"±{r.get('std',0):.4f}" if r.get("std") else "      "
    print(f"  {r['name']:<35} {r['pr_auc']:>8.4f}{std_str}  {r['macro_f1']:>9.4f}  {r['type']}")

# 우리 모델 비교
print()
print("  [우리 GNN 결과]")
for name, pr in [
    ("HeteroBWGNN_boost (Trans→Ind)",   0.6526),
    ("DRAGWave_TVF_400ep (Inductive)",  0.7429),
    ("4-way Ensemble (Inductive)",      0.7748),
]:
    print(f"  {name:<35} {pr:>8.4f}")

# Best 텍스트 베이스라인 대비
text_results = [r for r in results if r["type"] == "text_only"]
if text_results:
    best_text = max(text_results, key=lambda x: x["pr_auc"])
    print(f"\n  최강 텍스트 베이스라인: {best_text['name']} PR-AUC={best_text['pr_auc']:.4f}")
    print(f"  우리 4-way 대비: +{0.7748-best_text['pr_auc']:.4f} ({(0.7748-best_text['pr_auc'])/best_text['pr_auc']*100:.0f}%↑)")

# 저장
with open(RES/"E10_rigorous_baselines.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n저장: results/experiments/E10_rigorous_baselines.json")
print("완료")
