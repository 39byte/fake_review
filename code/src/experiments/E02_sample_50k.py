"""
E02_sample_50k.py
실험 2: 50K 샘플링 + 그래프 구축

가설: 더 많은 데이터 → 파라미터/노드 비율 감소 → 양적 과적합 완화
출력: data/graphs/hetero_graph_50k.pt
      data/processed/df_sampled_50k.parquet
"""
import pandas as pd, numpy as np, torch, json, time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from scipy.spatial import cKDTree

BASE  = Path(__file__).resolve().parent.parent.parent
RAW   = BASE / "data" / "raw"
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results" / "experiments"
RES.mkdir(parents=True, exist_ok=True)

TARGET_N  = 50_000
TOP_PRODS = 150     # 상위 150개 식당
SEED      = 42
np.random.seed(SEED); torch.manual_seed(SEED)

print("="*65)
print("E02: 50K 샘플링")
print("="*65)

# ── 원본 로드 & 라벨 변환 ─────────────────────────────────────────────────────
df = pd.read_csv(RAW/"yelpzip.csv", low_memory=False)
df["label"] = df["label"].map({-1:1, 1:0})
df["date"]  = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date","text","prod_id","user_id"]).reset_index(drop=True)
df["timestamp"] = df["date"].astype(np.int64) // 10**9
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3.0)
df = df[df["text"].str.len() > 5].reset_index(drop=True)

print(f"원본: {len(df):,}건  스팸={df['label'].mean():.3f}")

# ── Step A: 상위 150개 식당 ────────────────────────────────────────────────────
prod_cnt = df.groupby("prod_id").size().sort_values(ascending=False)
top_prods = prod_cnt.head(TOP_PRODS).index.tolist()
df_top = df[df["prod_id"].isin(top_prods)].copy()
print(f"\n상위 {TOP_PRODS}개 식당: {len(df_top):,}건  스팸={df_top['label'].mean():.3f}")

# ── Step B: 활성 유저 필터 ────────────────────────────────────────────────────
user_cnt = df_top.groupby("user_id").size()
active_users = user_cnt[user_cnt >= 2].index
df_top = df_top[df_top["user_id"].isin(active_users)].copy()
print(f"활성 유저 필터 후: {len(df_top):,}건  스팸={df_top['label'].mean():.3f}")

# ── Step C: 스팸 비율 보정 (목표 13.2%) ──────────────────────────────────────
target_spam_ratio = 0.132
current_ratio = df_top["label"].mean()
print(f"\n스팸 비율 현재={current_ratio:.3f}  목표=0.132")

if current_ratio < target_spam_ratio:
    # 스팸 집중 식당 추가
    spam_by_prod = df[~df["prod_id"].isin(top_prods)].groupby("prod_id")["label"].mean()
    extra_prods  = spam_by_prod[spam_by_prod >= 0.3].sort_values(ascending=False).head(30).index
    df_extra = df[df["prod_id"].isin(extra_prods)]
    df_top   = pd.concat([df_top, df_extra], ignore_index=True).drop_duplicates()
    print(f"스팸 집중 식당 30개 추가 후: {len(df_top):,}건  스팸={df_top['label'].mean():.3f}")

# ── Step D: 50K로 다운샘플 (시간순) ──────────────────────────────────────────
df_top = df_top.sort_values("timestamp").reset_index(drop=True)
if len(df_top) > TARGET_N:
    # 밀도 중심 유지하면서 50K 선택
    # 스팸 비율 보존하여 랜덤 샘플
    spam_df  = df_top[df_top["label"]==1]
    norm_df  = df_top[df_top["label"]==0]
    n_spam   = int(TARGET_N * 0.132)
    n_norm   = TARGET_N - n_spam
    if len(spam_df) >= n_spam and len(norm_df) >= n_norm:
        df_sampled = pd.concat([
            spam_df.sample(n_spam, random_state=SEED),
            norm_df.sample(n_norm, random_state=SEED)
        ]).sort_values("timestamp").reset_index(drop=True)
    else:
        df_sampled = df_top.head(TARGET_N)
else:
    df_sampled = df_top

print(f"\n최종 샘플: {len(df_sampled):,}건  스팸={df_sampled['label'].mean():.3f}")
df_sampled.to_parquet(PROC/"df_sampled_50k.parquet", index=False)
print(f"저장: df_sampled_50k.parquet")

# ── 시간순 60/20/20 분할 마스크 ──────────────────────────────────────────────
n = len(df_sampled)
n_train = int(n * 0.60)
n_val   = int(n * 0.20)

train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)
train_mask[:n_train] = True
val_mask[n_train:n_train+n_val] = True
test_mask[n_train+n_val:] = True

print(f"\n분할: Train={train_mask.sum():,} / Val={val_mask.sum():,} / Test={test_mask.sum():,}")

# ── SBERT 임베딩 ─────────────────────────────────────────────────────────────
print("\nSBERT 임베딩 생성 중...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
texts = df_sampled["text"].fillna("").tolist()
emb   = sbert.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

ts_norm = (df_sampled["timestamp"].values - df_sampled["timestamp"].min()) / \
          (df_sampled["timestamp"].max() - df_sampled["timestamp"].min() + 1e-8)
rating_norm = (df_sampled["rating"].values - 1) / 4.0

node_feat = np.hstack([emb, rating_norm.reshape(-1,1), ts_norm.reshape(-1,1)])
x_tensor  = torch.tensor(node_feat, dtype=torch.float32)
y_tensor  = torch.tensor(df_sampled["label"].values, dtype=torch.long)
ts_tensor = torch.tensor(df_sampled["timestamp"].values, dtype=torch.float32)

print(f"피처 형태: {x_tensor.shape}")

# ── 엣지 구축 ─────────────────────────────────────────────────────────────────
print("\n엣지 구축 중...")
df_sampled = df_sampled.reset_index(drop=True)
prod2idx  = {p: list(g.index) for p, g in df_sampled.groupby("prod_id")}
user2idx  = {u: list(g.index) for u, g in df_sampled.groupby("user_id")}

def make_edges_rtr(df, prod2idx, group_cap=48):
    src, dst = [], []
    for nodes in prod2idx.values():
        nodes_df = df.loc[nodes].sort_values("timestamp")
        monthly  = {(r.date.year, r.date.month): [] for _, r in nodes_df.iterrows()}
        for i, r in nodes_df.iterrows():
            monthly[(r.date.year, r.date.month)].append(i)
        for grp in monthly.values():
            if len(grp) > group_cap: grp = grp[:group_cap]
            for a in range(len(grp)):
                for b in range(a+1, len(grp)):
                    src += [grp[a], grp[b]]; dst += [grp[b], grp[a]]
    return torch.tensor([src, dst], dtype=torch.long)

def make_edges_rsr(df, prod2idx):
    src, dst = [], []
    for nodes in prod2idx.values():
        for rat, grp in df.loc[nodes].groupby("rating"):
            idx = list(grp.index)[:48]
            for a in range(len(idx)):
                for b in range(a+1, len(idx)):
                    src += [idx[a], idx[b]]; dst += [idx[b], idx[a]]
    return torch.tensor([src, dst], dtype=torch.long)

def make_edges_burst(df, prod2idx, window_h=72):
    src, dst, dts = [], [], []
    for nodes in prod2idx.values():
        ts_arr = df.loc[nodes, "timestamp"].values
        if len(ts_arr) < 2: continue
        tree  = cKDTree(ts_arr.reshape(-1,1))
        pairs = tree.query_pairs(r=window_h*3600)
        idx   = list(nodes) if isinstance(nodes, list) else nodes
        for a, b in pairs:
            dt = abs(float(ts_arr[a]) - float(ts_arr[b])) / 3600
            src += [idx[a], idx[b]]; dst += [idx[b], idx[a]]
            dts += [dt, dt]
    ei = torch.tensor([src, dst], dtype=torch.long)
    ea = torch.tensor(dts, dtype=torch.float32).unsqueeze(-1)
    return ei, ea

def make_edges_rur(user2idx):
    src, dst = [], []
    for nodes in user2idx.values():
        idx = nodes[:48]
        for a in range(len(idx)):
            for b in range(a+1, len(idx)):
                src += [idx[a], idx[b]]; dst += [idx[b], idx[a]]
    return torch.tensor([src, dst], dtype=torch.long)

def make_edges_sim(emb, prod2idx, threshold=0.85):
    src, dst = [], []
    for nodes in prod2idx.values():
        if len(nodes) < 2: continue
        idx = nodes
        e   = emb[idx]
        sim = e @ e.T
        mask = (sim >= threshold) & (sim < 0.9999)
        rows, cols = np.where(mask)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r < c:
                src += [idx[r], idx[c]]; dst += [idx[c], idx[r]]
    return torch.tensor([src, dst], dtype=torch.long)

print("  R-T-R..."); ei_rtr = make_edges_rtr(df_sampled, prod2idx)
print("  R-S-R..."); ei_rsr = make_edges_rsr(df_sampled, prod2idx)
print("  R-Burst-R..."); ei_burst, ea_burst = make_edges_burst(df_sampled, prod2idx)
print("  R-U-R..."); ei_rur = make_edges_rur(user2idx)
print("  R-Sim-R..."); ei_sim = make_edges_sim(emb, prod2idx)

# ── HeteroData 구성 ───────────────────────────────────────────────────────────
data = HeteroData()
data["review"].x          = x_tensor
data["review"].y          = y_tensor
data["review"].timestamp  = ts_tensor
data["review"].train_mask = train_mask
data["review"].val_mask   = val_mask
data["review"].test_mask  = test_mask

for et, ei in [
    (("review","rtr","review"),   ei_rtr),
    (("review","rsr","review"),   ei_rsr),
    (("review","rur","review"),   ei_rur),
    (("review","sim","review"),   ei_sim),
]:
    data[et].edge_index = ei

data["review","burst","review"].edge_index = ei_burst
data["review","burst","review"].edge_attr  = ea_burst

print(f"\n[그래프 통계]")
print(f"  노드: {n:,}  피처: {x_tensor.shape[1]}d")
for et in [("review","rtr","review"),("review","rsr","review"),
           ("review","burst","review"),("review","rur","review"),("review","sim","review")]:
    cnt = data[et].edge_index.shape[1] if et in data.edge_index_dict else 0
    print(f"  {et[1]:8s}: {cnt:,}개")

torch.save(data, GRAPH/"hetero_graph_50k.pt")
print(f"\n저장: data/graphs/hetero_graph_50k.pt")

result = {
    "nodes": n, "feat_dim": int(x_tensor.shape[1]),
    "train": int(train_mask.sum()), "val": int(val_mask.sum()),
    "test": int(test_mask.sum()), "spam_ratio": float(df_sampled["label"].mean()),
    "split": "60/20/20",
}
with open(RES/"E02_graph_stats.json","w") as f:
    json.dump(result,f,indent=2)
print("완료")
