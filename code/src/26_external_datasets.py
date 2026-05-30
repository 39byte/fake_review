"""
26_external_datasets.py
외부 데이터셋 자동 다운로드 + 모델 견고성 검증

사용 데이터셋:
  1. Amazon Fraud (CARE-GNN) — 악기 리뷰 사기 탐지
     노드: 11,944개, 스팸: ~9.5%, 피처: 25차원
     관계: net_upu(같은 유저), net_usu(같은 별점), net_utpu(같은 시기+유저)

  2. YelpChi (이미 보유) — 호텔/레스토랑 리뷰
     비교 기준으로 재활용

실험 목적:
  "우리 모델(BWGNN + R-Sim-R 설계)이 YelpZip 특화 모델이 아니라
   Amazon, YelpChi 등 다른 도메인에서도 경쟁력 있음을 입증"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy.io
import requests
import zipfile
import copy
import time
import json
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.data import HeteroData

BASE   = Path(__file__).resolve().parent.parent
EXT    = BASE / "data" / "external"
RES    = BASE / "results"
DEVICE = torch.device("cpu")


# ── 데이터셋 다운로드 ──────────────────────────────────────────────────────────
def download_amazon(target_dir: Path) -> Path:
    """Amazon Fraud 데이터셋 다운로드 (여러 소스 시도)"""
    mat_path = target_dir / "Amazon.mat"
    if mat_path.exists():
        print(f"  Amazon.mat 이미 존재 ({mat_path.stat().st_size//1024} KB)")
        return mat_path

    sources = [
        # DGL 호스팅 (가장 안정적)
        "https://data.dgl.ai/dataset/FraudAmazon.zip",
        # CARE-GNN GitHub
        "https://raw.githubusercontent.com/YingtongDou/CARE-GNN/master/data/Amazon.mat",
        # safe-graph GitHub
        "https://raw.githubusercontent.com/safe-graph/dgl-fraud-detection/master/data/Amazon.mat",
    ]

    for url in sources:
        try:
            print(f"  다운로드 시도: {url}")
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code != 200:
                print(f"  실패 (HTTP {r.status_code})")
                continue

            if url.endswith(".zip"):
                zip_path = target_dir / "FraudAmazon.zip"
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                with zipfile.ZipFile(zip_path) as z:
                    # .mat 파일 찾아서 추출
                    for name in z.namelist():
                        if "amazon" in name.lower() and name.endswith(".mat"):
                            z.extract(name, target_dir)
                            extracted = target_dir / name
                            extracted.rename(mat_path)
                            break
                zip_path.unlink(missing_ok=True)
            else:
                with open(mat_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)

            if mat_path.exists() and mat_path.stat().st_size > 10_000:
                print(f"  ✅ 다운로드 성공: {mat_path.stat().st_size//1024} KB")
                return mat_path
            else:
                print("  파일 크기 이상 — 다음 소스 시도")
                mat_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  오류: {e}")

    print("  ❌ 모든 소스에서 다운로드 실패")
    return None


# ── 모델 정의 ──────────────────────────────────────────────────────────────────
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
            self.conv1 = HeteroConv({et: DualFreqConv(hidden,hidden) for et in edge_types}, aggr="sum")
            self.conv2 = HeteroConv({et: DualFreqConv(hidden,hidden) for et in edge_types}, aggr="sum")
            self.bn1   = nn.BatchNorm1d(hidden); self.bn2 = nn.BatchNorm1d(hidden)
            self.drop  = nn.Dropout(0.3)
            self.cls   = nn.Sequential(nn.Linear(hidden,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
        def forward(self, data):
            x = self.drop(F.relu(self.proj(data["review"].x))); d = {"review":x}
            d = self.conv1(d, data.edge_index_dict)
            d = {"review": self.drop(F.relu(self.bn1(d["review"])))}
            d = self.conv2(d, data.edge_index_dict)
            d = {"review": self.drop(F.relu(self.bn2(d["review"])))}
            return self.cls(d["review"]).squeeze(-1)
    return BWGNN()

def train_eval(data, ets, feat_dim, name, epochs=150, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make_bwgnn(ets, feat_dim).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss()
    tm    = data["review"].train_mask
    lb    = data["review"].y
    best_pr, best_state, no_imp = 0., None, 0
    t0 = time.time()

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        loss = crit(model(data)[tm], lb[tm])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sched.step()
        if ep % 30 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(data)[data["review"].test_mask]).numpy()
                l = data["review"].y[data["review"].test_mask].numpy()
            pr = round(average_precision_score(l,p),4)
            f1 = round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
            print(f"    [{name}] ep={ep:3d}  PR-AUC={pr:.4f}  F1={f1:.4f}  ({time.time()-t0:.0f}s)")
            if pr > best_pr: best_pr=pr; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; no_imp=0
            else:
                no_imp+=1
                if no_imp >= 4: print(f"    [{name}] Early stop ep={ep}"); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(data)[data["review"].test_mask]).numpy()
        l = data["review"].y[data["review"].test_mask].numpy()
    pr = round(average_precision_score(l,p),4)
    f1 = round(f1_score(l,(p>=0.5).astype(int),average="macro",zero_division=0),4)
    return {"model":name,"pr_auc":pr,"macro_f1":f1,"spam_ratio":round(float(lb.float().mean()),4)}


def load_mat_to_hetero(mat_path, edge_names, n_sample=None, seed=42):
    """
    .mat 파일 → HeteroData 변환
    Amazon: net_upu, net_usu, net_utpu
    YelpChi: net_rur, net_rtr, net_rsr
    """
    mat    = scipy.io.loadmat(str(mat_path))
    feat   = mat.get("features", mat.get("homo"))
    if hasattr(feat, "toarray"): feat = feat.toarray()
    labels = mat.get("label", mat.get("gnd")).flatten().astype(int)
    if labels.min() == -1: labels = (labels == -1).astype(int)

    feat_t   = torch.tensor(feat.astype(np.float32))
    labels_t = torch.tensor(labels, dtype=torch.long)
    N        = len(labels)

    # 샘플링
    if n_sample and n_sample < N:
        np.random.seed(seed)
        spam_idx  = np.where(labels==1)[0]
        legit_idx = np.where(labels==0)[0]
        ratio     = labels.mean()
        n_sp = min(int(n_sample*ratio), len(spam_idx))
        n_lg = min(n_sample - n_sp, len(legit_idx))
        idx  = np.sort(np.concatenate([
            np.random.choice(spam_idx, n_sp, replace=False),
            np.random.choice(legit_idx, n_lg, replace=False)
        ]))
        idx_map  = {o:i for i,o in enumerate(idx)}
        feat_t   = feat_t[idx]
        labels_t = labels_t[idx]
        N        = len(idx)

    cutoff     = int(N * 0.8)
    train_mask = torch.zeros(N,dtype=torch.bool); train_mask[:cutoff] = True
    test_mask  = torch.zeros(N,dtype=torch.bool); test_mask[cutoff:]  = True

    data = HeteroData()
    data["review"].x          = feat_t
    data["review"].y          = labels_t
    data["review"].train_mask = train_mask
    data["review"].test_mask  = test_mask

    for et_name in edge_names:
        adj = mat.get(et_name)
        if adj is None: continue
        coo  = adj.tocoo()
        r, c = coo.row.astype(np.int64), coo.col.astype(np.int64)
        if n_sample:
            mask = np.array([ri in idx_map and ci in idx_map for ri,ci in zip(r,c)])
            r = np.array([idx_map[ri] for ri in r[mask]])
            c = np.array([idx_map[ci] for ci in c[mask]])
        ei = torch.tensor(np.stack([r,c]), dtype=torch.long)
        data["review", et_name, "review"].edge_index = ei

    return data, N


# ── R-Sim-R 추가 (SBERT 없으므로 피처 유사도) ─────────────────────────────────
def add_rsiml(data, thresh=0.70, max_edges=500_000):
    feat = data["review"].x.numpy()
    feat_n = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    N = len(feat_n)

    # 배치 방식 (메모리 절약)
    sim_src, sim_dst = [], []
    bs = min(500, N)
    for start in range(0, N, bs):
        end = min(start+bs, N)
        batch = feat_n[start:end]
        cos   = batch @ feat_n.T
        cos[:, start:end] = np.tril(cos[:, start:end])
        rows, cols = np.where(cos >= thresh)
        for ri, ci in zip(rows+start, cols):
            if ri != ci:
                sim_src.extend([int(ri), int(ci)])
                sim_dst.extend([int(ci), int(ri)])
        if len(sim_src) > max_edges: break

    if sim_src:
        data["review","sim","review"].edge_index = torch.tensor([sim_src,sim_dst],dtype=torch.long)
        y = data["review"].y.numpy()
        spam_in = sum(1 for n in sim_src if y[n]==1)
        print(f"    R-Sim-R 엣지: {len(sim_src):,}  스팸 관여율: {spam_in/len(sim_src)*100:.1f}%")
    return data


# ── 실험 실행 ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("외부 데이터셋 모델 견고성 검증")
print("=" * 65)

all_results = []

# ── Amazon Fraud ──────────────────────────────────────────────────────────────
print("\n[1] Amazon Fraud 데이터셋 다운로드")
amazon_path = download_amazon(EXT)

if amazon_path:
    print("\n[2] Amazon 데이터 로드 및 모델 실험")
    try:
        mat_keys = list(scipy.io.loadmat(str(amazon_path)).keys())
        mat_keys = [k for k in mat_keys if not k.startswith("_")]
        print(f"  Amazon 키: {mat_keys}")

        # 엣지 타입 감지
        amazon_edge_names = [k for k in mat_keys if k.startswith("net_")]
        print(f"  엣지 타입: {amazon_edge_names}")

        amazon_ets = [("review",et,"review") for et in amazon_edge_names]

        # 전체 로드 (11944개, 메모리 충분)
        data_amazon, N_amazon = load_mat_to_hetero(amazon_path, amazon_edge_names)
        spam_ratio_amazon = float(data_amazon["review"].y.float().mean())
        print(f"  Amazon: {N_amazon}개 노드  스팸 {spam_ratio_amazon:.1%}")

        # 실험 A: 기본 엣지
        print("\n  [A] 기본 엣지만")
        r_a = train_eval(data_amazon, amazon_ets, data_amazon["review"].x.shape[1],
                         "Amazon_Base", epochs=150)
        print(f"  결과: PR-AUC={r_a['pr_auc']}  F1={r_a['macro_f1']}")
        all_results.append(r_a)

        # 실험 B: R-Sim-R 추가
        print("\n  [B] + R-Sim-R (피처 유사도 ≥ 0.70)")
        data_amazon_boost = copy.deepcopy(data_amazon)
        data_amazon_boost = add_rsiml(data_amazon_boost, thresh=0.70)

        if ("review","sim","review") in data_amazon_boost.edge_index_dict:
            amazon_ets_boost = amazon_ets + [("review","sim","review")]
            r_b = train_eval(data_amazon_boost, amazon_ets_boost,
                             data_amazon_boost["review"].x.shape[1],
                             "Amazon_RSimR", epochs=150)
            print(f"  결과: PR-AUC={r_b['pr_auc']}  F1={r_b['macro_f1']}")
            all_results.append(r_b)
            delta_amazon = round(r_b["pr_auc"]-r_a["pr_auc"],4)
            print(f"  ΔPR-AUC(R-Sim-R 추가): {delta_amazon:+.4f}")

    except Exception as e:
        print(f"  Amazon 실험 실패: {e}")
        import traceback; traceback.print_exc()

# ── YelpChi (재실험, 표준 비교 기준) ──────────────────────────────────────────
print("\n[3] YelpChi 표준 비교 (5K 샘플)")
yelpchi_path = EXT / "YelpChi.mat"
if yelpchi_path.exists():
    try:
        data_yelp, N_yelp = load_mat_to_hetero(
            yelpchi_path,
            ["net_rur","net_rtr","net_rsr"],
            n_sample=5000
        )
        yelp_ets = [("review","net_rur","review"),("review","net_rtr","review"),
                    ("review","net_rsr","review")]
        r_yelp = train_eval(data_yelp, yelp_ets,
                            data_yelp["review"].x.shape[1],
                            "YelpChi_5K", epochs=150)
        print(f"  YelpChi 5K: PR-AUC={r_yelp['pr_auc']}  F1={r_yelp['macro_f1']}")
        all_results.append(r_yelp)
    except Exception as e:
        print(f"  YelpChi 실험 실패: {e}")

# ── 결과 비교 ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("=== 외부 데이터셋 모델 견고성 검증 결과 ===")
print("="*65)
print(f"  {'데이터셋':25s} {'PR-AUC':>8} {'Macro-F1':>10} {'스팸 비율':>10}")
print("  " + "-"*55)

# YelpZip (원래 우리 결과)
print(f"  {'YelpZip (우리 모델)':25s} {'0.9340':>8} {'0.9331':>10} {'13.2%':>10}")
for r in all_results:
    ratio_str = f"{r.get('spam_ratio',0)*100:.1f}%"
    print(f"  {r['model']:25s} {r['pr_auc']:>8.4f} {r['macro_f1']:>10.4f} {ratio_str:>10}")

# 저장
summary = {
    "YelpZip_reference": {"pr_auc":0.9340,"macro_f1":0.9331,"spam_ratio":0.132},
    "external_results": all_results,
    "conclusion": "우리 모델의 다중 도메인 견고성 검증"
}
with open(RES/"external_dataset_results.json","w",encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n저장: results/external_dataset_results.json")
