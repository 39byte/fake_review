"""
debug_check.py
팀원 환경 설정 오류 빠른 진단
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path

BASE  = Path(__file__).resolve().parent.parent
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"

print("=" * 55)
print("환경 설정 진단")
print("=" * 55)

# 1. 라벨 변환 확인
print("\n[1] 라벨 분포 확인")
df = pd.read_parquet(PROC / "df_sampled.parquet")
vc = df["label"].value_counts().sort_index()
print(f"  label 값: {vc.to_dict()}")
if set(df["label"].unique()) == {0, 1}:
    spam_n = (df["label"] == 1).sum()
    print(f"  ✅ 정상 — spam=1: {spam_n}건 ({spam_n/len(df)*100:.1f}%)")
elif set(df["label"].unique()) == {-1, 1}:
    print("  ❌ 라벨 변환 미적용! fake=-1/real=1 그대로 — 반드시 0/1로 변환 필요")
else:
    print(f"  ⚠️  예상치 못한 라벨 값: {df['label'].unique()}")

# 2. 그래프 파일 확인
print("\n[2] 그래프 파일 확인")
for fname in ["hetero_graph.pt", "hetero_graph_boost.pt"]:
    p = GRAPH / fname
    if p.exists():
        g = torch.load(p, weights_only=False)
        n_nodes = g["review"].x.shape[0]
        feat_dim = g["review"].x.shape[1]
        edge_types = list(g.edge_index_dict.keys())
        n_sim = g["review","sim","review"].edge_index.shape[1] if ("review","sim","review") in g.edge_index_dict else 0
        print(f"  {fname}: 노드={n_nodes}, 피처={feat_dim}d, 엣지타입={len(edge_types)}, R-Sim-R={n_sim}개")
    else:
        print(f"  {fname}: 파일 없음")

# 3. 마스크 확인
print("\n[3] Train/Test 마스크 확인")
g = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False)
tm = g["review"].train_mask
tsm = g["review"].test_mask
y = g["review"].y

print(f"  train: {tm.sum().item()}건  |  test: {tsm.sum().item()}건")
print(f"  train 스팸비율: {y[tm].float().mean().item():.3f}")
print(f"  test  스팸비율: {y[tsm].float().mean().item():.3f}")

if tm.sum().item() == 24000 and tsm.sum().item() == 6000:
    print("  ✅ 마스크 정상 (24K/6K)")
else:
    print("  ❌ 마스크 비정상! 재확인 필요")

# 4. 라벨 방향 확인
print("\n[4] 라벨 방향 확인")
labels = g["review"].y
spam_count = labels.sum().item()
print(f"  전체 스팸(label=1): {spam_count}건 ({spam_count/len(labels)*100:.1f}%)")
if abs(spam_count/len(labels) - 0.132) < 0.02:
    print("  ✅ 스팸 비율 정상 (~13.2%)")
else:
    print(f"  ❌ 스팸 비율 이상 ({spam_count/len(labels)*100:.1f}%) — 라벨 방향 확인 필요")

# 5. tag 컬럼 포함 여부
print("\n[5] 피처 누수 확인 (tag 컬럼)")
feat_dim = g["review"].x.shape[1]
print(f"  피처 차원: {feat_dim}d")
if feat_dim == 386:
    print("  ✅ 정상 (SBERT 384 + rating 1 + timestamp 1)")
elif feat_dim == 387:
    print("  ❌ tag 컬럼 포함됨! 정답 누수 발생 — 제거 필요")
else:
    print(f"  ⚠️  예상 외 차원 ({feat_dim}d) — 피처 구성 확인 필요")

print("\n" + "=" * 55)
print("진단 완료. 위 ❌ 항목을 수정하세요.")
