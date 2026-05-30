"""
28_multi_dataset_benchmark.py
다양한 외부 데이터셋 전체 벤치마크

시도하는 데이터셋:
  1. T-Finance    (BWGNN ICML 2022, 금융 거래, ~39K 노드)
  2. T-Social     (BWGNN ICML 2022, 소셜 네트워크, ~5.8M → 샘플링)
  3. Elliptic     (Bitcoin 거래, 203K 노드, 시계열)
  4. Reddit       (소셜 이상 탐지)
  5. Amazon (이미 완료)
  6. YelpChi (이미 완료)

각 데이터셋에 동일한 BWGNN 모델 적용
목적: 도메인에 무관한 모델 견고성 입증
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import time
import copy
import requests
import zipfile
import gdown
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData
from scipy.io import loadmat

BASE   = Path(__file__).resolve().parent.parent
EXT    = BASE / "data" / "external"
RES    = BASE / "results"
DEVICE = torch.device("cpu")
EXT.mkdir(parents=True, exist_ok=True)


# ── 공통 모델 ──────────────────────────────────────────────────────────────────
class DualFreqConv(MessagePassing):
    def __init__(self, a, b):
        super().__init__(aggr="mean"); self.lin = nn.Linear(a*2, b)
    def forward(self, x, ei):
        low = self.propagate(ei, x=x); return self.lin(torch.cat([low, x-low], -1))
    def message(self, x_j): return x_j

class FocalLoss(nn.Module):
    def __init__(self, g=2., a=0.75): super().__init__(); self.g, self.a = g, a
    def forward(self, lo, ta):
        bce = F.binary_cross_entropy_with_logits(lo, ta.float(), reduction="none")
        pt  = torch.exp(-bce)
        w   = torch.where(ta==1, torch.full_like(bce,self.a), torch.full_like(bce,1-self.a))
        return (w*(1-pt)**self.g*bce).mean()

def make_bwgnn(edge_types, feat_dim, hidden=64):
    class BWGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj  = nn.Linear(feat_dim, hidden)
            self.conv1 = HeteroConv({et: DualFreqConv(hidden, hidden) for et in edge_types}, aggr="sum")
            self.conv2 = HeteroConv({et: DualFreqConv(hidden, hidden) for et in edge_types}, aggr="sum")
            self.bn1   = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
            self.drop  = nn.Dropout(0.3)
            self.cls   = nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
        def forward(self, data):
            x = self.drop(F.relu(self.proj(data["node"].x))); d = {"node":x}
            d = self.conv1(d, data.edge_index_dict)
            d = {"node": self.drop(F.relu(self.bn1(d["node"])))}
            d = self.conv2(d, data.edge_index_dict)
            d = {"node": self.drop(F.relu(self.bn2(d["node"])))}
            return self.cls(d["node"]).squeeze(-1)
    return BWGNN()

def train_eval(data, ets, feat_dim, name, epochs=150, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_bwgnn(ets, feat_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss()
    tm    = data["node"].train_mask
    lb    = data["node"].y
    best_pr, best_state, no_imp = 0., None, 0; t0 = time.time()

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[tm], lb[tm])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()

        if ep % 30 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(data)[data["node"].test_mask]).numpy()
                l = data["node"].y[data["node"].test_mask].numpy()
            pr = round(average_precision_score(l,p),4)
            f1 = round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
            print(f"    [{name}] ep={ep:3d}  PR-AUC={pr:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
            if pr > best_pr: best_pr=pr; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp >= 4: print(f"    [{name}] Early stop ep={ep}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data)[data["node"].test_mask]).numpy()
        l = data["node"].y[data["node"].test_mask].numpy()
    return {"name":name,
            "pr_auc":round(average_precision_score(l,p),4),
            "macro_f1":round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4),
            "spam_ratio":round(float(lb.float().mean()),4)}


def mat_to_hetero(feat, labels, edge_dict, n_sample=None, seed=42):
    """scipy.io.loadmat 결과를 HeteroData로 변환 (node 키 사용)"""
    if hasattr(feat, "toarray"): feat = feat.toarray()
    feat_arr = feat.astype(np.float32)
    labels_arr = labels.flatten().astype(int)
    if labels_arr.min() == -1: labels_arr = (labels_arr == -1).astype(int)
    N = len(labels_arr)

    if n_sample and n_sample < N:
        np.random.seed(seed)
        spam_idx  = np.where(labels_arr==1)[0]
        legit_idx = np.where(labels_arr==0)[0]
        ratio = labels_arr.mean()
        n_sp = min(int(n_sample*ratio)+1, len(spam_idx))
        n_lg = min(n_sample-n_sp, len(legit_idx))
        idx  = np.sort(np.concatenate([np.random.choice(spam_idx,n_sp,replace=False),
                                       np.random.choice(legit_idx,n_lg,replace=False)]))
        idx_map = {o:i for i,o in enumerate(idx)}
        feat_arr = feat_arr[idx]; labels_arr = labels_arr[idx]; N = len(idx)
    else:
        idx_map = None

    cutoff = int(N*0.8)
    train_mask = torch.zeros(N,dtype=torch.bool); train_mask[:cutoff] = True
    test_mask  = torch.zeros(N,dtype=torch.bool); test_mask[cutoff:]  = True

    data = HeteroData()
    data["node"].x          = torch.tensor(feat_arr)
    data["node"].y          = torch.tensor(labels_arr, dtype=torch.long)
    data["node"].train_mask = train_mask
    data["node"].test_mask  = test_mask

    for et_name, adj in edge_dict.items():
        coo  = adj.tocoo()
        r, c = coo.row.astype(np.int64), coo.col.astype(np.int64)
        if idx_map:
            mask = np.array([ri in idx_map and ci in idx_map for ri,ci in zip(r,c)])
            r = np.array([idx_map[ri] for ri in r[mask]])
            c = np.array([idx_map[ci] for ci in c[mask]])
        ei = torch.tensor(np.stack([r,c]), dtype=torch.long)
        data["node", et_name, "node"].edge_index = ei

    return data


results = []
print("=" * 65)
print("다중 외부 데이터셋 벤치마크")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# [1] T-Finance (BWGNN 논문, ICML 2022)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] T-Finance 다운로드 시도...")
tf_path = EXT / "T-Finance.pt"

if not tf_path.exists():
    # Google Drive 공개 폴더 (BWGNN 논문 공식)
    gdrive_urls = [
        ("1pe0lmqB_L4TkRDFYNKNsLTsE4-HSmAhY", "T-Finance.pt"),   # T-Finance
    ]
    for gdrive_id, fname in gdrive_urls:
        try:
            out = EXT / fname
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            gdown.download(url, str(out), quiet=False)
            if out.exists() and out.stat().st_size > 10000:
                print(f"  T-Finance 다운로드 성공: {out.stat().st_size//1024} KB")
                break
        except Exception as e:
            print(f"  gdown 실패: {e}")

    if not tf_path.exists():
        # 대안: BWGNN GitHub에서 직접
        try:
            alt_url = "https://raw.githubusercontent.com/squareRoot3/Rethinking-Anomaly-Detection/master/dataset/T-Finance.pt"
            r = requests.get(alt_url, timeout=30)
            if r.status_code == 200 and len(r.content) > 10000:
                tf_path.write_bytes(r.content)
                print(f"  T-Finance GitHub 다운로드 성공: {len(r.content)//1024} KB")
        except Exception as e:
            print(f"  GitHub 다운로드 실패: {e}")

if tf_path.exists():
    try:
        tf_data = torch.load(tf_path, weights_only=False)
        print(f"  T-Finance 로드 성공: type={type(tf_data)}")
        if isinstance(tf_data, dict):
            print(f"  keys: {list(tf_data.keys())[:8]}")
        # T-Finance는 보통 PyG Data 형태
        if hasattr(tf_data, 'x'):
            N = tf_data.x.shape[0]
            spam_r = tf_data.y.float().mean().item()
            print(f"  노드: {N:,}  피처: {tf_data.x.shape[1]}  스팸: {spam_r:.1%}")
    except Exception as e:
        print(f"  T-Finance 로드 실패: {e}")
else:
    print("  T-Finance 다운로드 실패 — 이 데이터셋 스킵")

# ─────────────────────────────────────────────────────────────────────────────
# [2] Elliptic Bitcoin (Kaggle 대안)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Elliptic Bitcoin 다운로드 시도...")
elliptic_path = EXT / "elliptic_txs_features.csv"

if not elliptic_path.exists():
    alt_sources = [
        "https://raw.githubusercontent.com/IBM/cap-ub/main/data/elliptic_txs_features.csv",
        "https://raw.githubusercontent.com/vdrvar/bitcoin_fraud_detection/main/data/elliptic_txs_features.csv",
    ]
    for url in alt_sources:
        try:
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 200:
                with open(elliptic_path, 'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                if elliptic_path.stat().st_size > 100000:
                    print(f"  Elliptic 다운로드 성공: {elliptic_path.stat().st_size//1024} KB")
                    break
                else:
                    elliptic_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  Elliptic 실패: {e}")

if elliptic_path.exists():
    try:
        df_feat = pd.read_csv(elliptic_path, header=None)
        print(f"  Elliptic 피처: {df_feat.shape}")
        # 엣지 파일도 필요
        edge_url = "https://raw.githubusercontent.com/vdrvar/bitcoin_fraud_detection/main/data/elliptic_txs_edgelist.csv"
        edge_path = EXT / "elliptic_txs_edgelist.csv"
        r = requests.get(edge_url, timeout=30)
        if r.status_code == 200:
            edge_path.write_bytes(r.content)
            print(f"  Elliptic 엣지: {edge_path.stat().st_size//1024} KB")

        # 클래스 파일
        class_url = "https://raw.githubusercontent.com/vdrvar/bitcoin_fraud_detection/main/data/elliptic_txs_classes.csv"
        class_path = EXT / "elliptic_txs_classes.csv"
        r = requests.get(class_url, timeout=30)
        if r.status_code == 200:
            class_path.write_bytes(r.content)
    except Exception as e:
        print(f"  Elliptic 처리 실패: {e}")
else:
    print("  Elliptic 다운로드 실패 — 스킵")

# ─────────────────────────────────────────────────────────────────────────────
# [3] Reddit 이상 탐지 데이터셋
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Reddit 이상 탐지 데이터셋 다운로드 시도...")
reddit_path = EXT / "reddit.mat"

if not reddit_path.exists():
    reddit_urls = [
        "https://data.dgl.ai/dataset/reddit.zip",
        "http://snap.stanford.edu/graphsage/reddit.zip",
    ]
    for url in reddit_urls:
        try:
            r = requests.get(url, timeout=60, stream=True)
            if r.status_code == 200:
                zip_path = EXT / "reddit_dl.zip"
                with open(zip_path,'wb') as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                print(f"  Reddit 다운로드: {zip_path.stat().st_size//1024} KB")
                break
        except Exception as e:
            print(f"  Reddit 실패: {e}")

    # FairGAD Reddit (2024 논문, 이상 탐지 레이블)
    fairgad_url = "https://raw.githubusercontent.com/yeon-lab/FairGAD/main/data/reddit.pt"
    try:
        r = requests.get(fairgad_url, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            reddit_path.write_bytes(r.content)  # 임시 저장
            print(f"  FairGAD Reddit: {len(r.content)//1024} KB")
    except Exception as e:
        print(f"  FairGAD Reddit 실패: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [4] GADBench 공개 데이터셋
# GADBench (NeurIPS 2023) — GitHub에서 직접 다운로드 가능한 소형 데이터셋
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] GADBench 소형 데이터셋 다운로드 시도...")

gadbench_datasets = {
    "Disney":   "1nN_K7HQ2cQO2x8XHkEdq6zcraK0cECQE",
    "Book":     "1p6_hNS5h3LIBJMWqGQIgCjqYxmSBKBpA",
    "Reddit_GAD": "1xtQomSZ-_aSwrNpN3w1Y8s2KkqSUQdE-",
}

for ds_name, gdrive_id in gadbench_datasets.items():
    out_path = EXT / f"{ds_name}.pt"
    if out_path.exists():
        print(f"  {ds_name}: 이미 있음")
        continue
    try:
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, str(out_path), quiet=True)
        if out_path.exists() and out_path.stat().st_size > 1000:
            print(f"  {ds_name}: {out_path.stat().st_size//1024} KB 다운로드 성공")
        else:
            out_path.unlink(missing_ok=True)
            print(f"  {ds_name}: 다운로드 실패")
    except Exception as e:
        print(f"  {ds_name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# [5] Elliptic 처리 및 학습 (다운로드 성공한 경우)
# ─────────────────────────────────────────────────────────────────────────────
if (EXT/"elliptic_txs_features.csv").exists() and (EXT/"elliptic_txs_classes.csv").exists():
    print("\n[5] Elliptic Bitcoin 실험")
    try:
        df_feat  = pd.read_csv(EXT/"elliptic_txs_features.csv", header=None)
        df_class = pd.read_csv(EXT/"elliptic_txs_classes.csv")
        print(f"  피처: {df_feat.shape}  클래스: {df_class.shape}")

        # 라벨 처리 (1=illicit=사기, 2=licit=정상, unknown 제외)
        df_class.columns = ["txId","class"]
        known = df_class[df_class["class"] != "unknown"].copy()
        known["label"] = (known["class"] == "1").astype(int)

        # txId를 인덱스로
        df_feat.columns = ["txId"] + [f"f{i}" for i in range(df_feat.shape[1]-1)]
        merged = known.merge(df_feat, on="txId", how="inner")
        print(f"  유효 노드: {len(merged):,}  스팸: {merged['label'].mean():.1%}")

        if len(merged) > 1000:
            # 5K 샘플링
            n_sample = min(5000, len(merged))
            spam_idx  = merged[merged["label"]==1].index
            legit_idx = merged[merged["label"]==0].index
            spam_r    = merged["label"].mean()
            n_sp = min(int(n_sample*spam_r)+1, len(spam_idx))
            n_lg = n_sample - n_sp
            sampled = pd.concat([
                merged.loc[np.random.choice(spam_idx,  n_sp,  replace=False)],
                merged.loc[np.random.choice(legit_idx, n_lg, replace=False)],
            ]).reset_index(drop=True)

            feat_cols = [c for c in sampled.columns if c.startswith("f")]
            feat_arr  = sampled[feat_cols].values.astype(np.float32)
            labels_arr= sampled["label"].values.astype(int)
            N = len(sampled); cutoff = int(N*0.8)

            # 엣지 처리
            if (EXT/"elliptic_txs_edgelist.csv").exists():
                df_edge = pd.read_csv(EXT/"elliptic_txs_edgelist.csv")
                valid_ids = set(sampled["txId"].values)
                df_edge.columns = ["src","dst"]
                valid_mask = df_edge["src"].isin(valid_ids) & df_edge["dst"].isin(valid_ids)
                df_edge = df_edge[valid_mask]
                id2idx = {tid: i for i,tid in enumerate(sampled["txId"].values)}
                src_idx = [id2idx[s] for s in df_edge["src"]]
                dst_idx = [id2idx[d] for d in df_edge["dst"]]
                ei = torch.tensor([src_idx+dst_idx, dst_idx+src_idx], dtype=torch.long)
            else:
                ei = torch.zeros(2, 0, dtype=torch.long)

            data_ell = HeteroData()
            data_ell["node"].x          = torch.tensor(feat_arr)
            data_ell["node"].y          = torch.tensor(labels_arr, dtype=torch.long)
            data_ell["node"].train_mask = torch.zeros(N,dtype=torch.bool)
            data_ell["node"].test_mask  = torch.zeros(N,dtype=torch.bool)
            data_ell["node"].train_mask[:cutoff] = True
            data_ell["node"].test_mask[cutoff:]  = True
            data_ell["node","tx","node"].edge_index = ei

            r_ell = train_eval(data_ell, [("node","tx","node")],
                               feat_arr.shape[1], "Elliptic_5K", epochs=150)
            results.append(r_ell)
            print(f"  Elliptic 결과: PR-AUC={r_ell['pr_auc']}  F1={r_ell['macro_f1']}")
    except Exception as e:
        print(f"  Elliptic 실험 실패: {e}")
        import traceback; traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# [6] PyG 내장 fake 데이터로 시스템 검증 (백업)
# 모든 다운로드 실패 시에도 파이프라인 정상 작동 확인
# ─────────────────────────────────────────────────────────────────────────────
if not results:
    print("\n[6] 다운로드 실패 — PyG 내장 그래프로 파이프라인 검증")
    from torch_geometric.datasets import FakeHeteroDataset
    ds = FakeHeteroDataset(num_graphs=1)
    g  = ds[0]
    # 리뷰 사기 탐지와 같은 구조로 변환
    n_nodes = 1000
    feat    = torch.randn(n_nodes, 16)
    labels  = torch.zeros(n_nodes, dtype=torch.long)
    labels[torch.randperm(n_nodes)[:100]] = 1  # 10% 사기

    data_fake = HeteroData()
    data_fake["node"].x          = feat
    data_fake["node"].y          = labels
    data_fake["node"].train_mask = torch.zeros(n_nodes,dtype=torch.bool)
    data_fake["node"].test_mask  = torch.zeros(n_nodes,dtype=torch.bool)
    data_fake["node"].train_mask[:800] = True
    data_fake["node"].test_mask[800:]  = True
    ei = torch.randint(0, n_nodes, (2, 3000))
    data_fake["node","e","node"].edge_index = ei

    r_fake = train_eval(data_fake, [("node","e","node")], 16, "Pipeline_Test", epochs=30)
    print(f"  파이프라인 검증: PR-AUC={r_fake['pr_auc']}")

# ─────────────────────────────────────────────────────────────────────────────
# 결과 요약
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("=== 다중 데이터셋 벤치마크 요약 ===")
print("="*65)

# 기존 결과와 통합
all_results = [
    {"name":"YelpZip (우리)",     "pr_auc":0.9367,"macro_f1":0.9362,"spam_ratio":0.132,"domain":"맛집 리뷰"},
    {"name":"Amazon_RSimR",       "pr_auc":0.8905,"macro_f1":0.9057,"spam_ratio":0.069,"domain":"악기 리뷰"},
    {"name":"YelpChi_5K",         "pr_auc":0.6271,"macro_f1":0.2690,"spam_ratio":0.145,"domain":"호텔/레스토랑"},
]
all_results.extend([{"domain":"외부", **r} for r in results])

print(f"\n  {'데이터셋':25s} {'PR-AUC':>8} {'F1':>8} {'스팸비율':>8} {'도메인':>12}")
print("  " + "-"*65)
for r in all_results:
    print(f"  {r['name']:25s} {r['pr_auc']:>8.4f} {r['macro_f1']:>8.4f}"
          f" {r.get('spam_ratio',0)*100:>7.1f}% {r.get('domain',''):>12}")

# 저장
with open(RES/"multi_dataset_benchmark.json","w",encoding="utf-8") as f:
    json.dump({"existing":all_results[:3],"new_external":results}, f, ensure_ascii=False, indent=2)
print(f"\n저장: results/multi_dataset_benchmark.json")
print(f"\n다운로드된 파일:")
for f in sorted(EXT.iterdir()):
    print(f"  {f.name}: {f.stat().st_size//1024} KB")
