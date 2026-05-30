"""50K 확장 가능성 및 train/val/test 분할 분석"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
raw = pd.read_csv(BASE / "data" / "raw" / "yelpzip.csv", low_memory=False)
raw["label"] = raw["label"].map({-1: 1, 1: 0})

print("=== 원본 데이터 규모 ===")
print(f"전체 리뷰: {len(raw):,}건")
print(f"식당 수:   {raw['prod_id'].nunique():,}개")
print(f"유저 수:   {raw['user_id'].nunique():,}개")
print()

# 식당별 리뷰 수 분포
prod_cnt = raw.groupby("prod_id").size().sort_values(ascending=False)

print("=== 상위 N개 식당으로 구성 가능한 서브그래프 규모 ===")
print(f"  {'식당':>6}   {'리뷰수':>8}   {'스팸비율':>8}   {'비고'}")
print("  " + "-"*50)
for n in [50, 100, 150, 200]:
    subset = raw[raw["prod_id"].isin(prod_cnt.head(n).index)]
    spam_r = subset["label"].mean()
    note = ""
    if abs(spam_r - 0.132) > 0.02:
        note = "  ← 스팸 비율 이탈"
    print(f"  {n:4d}개   {len(subset):8,}건   {spam_r:.3f} ({spam_r*100:.1f}%)   {note}")

print()
print("=== 50K 기준 Train/Val/Test 분할 시나리오 ===")
# 상위 몇 개 식당이 50K에 가까운지 확인
for n in [100, 130, 150, 170, 200]:
    subset = raw[raw["prod_id"].isin(prod_cnt.head(n).index)]
    total = len(subset)
    train = int(total * 0.6)
    val   = int(total * 0.2)
    test  = total - train - val
    print(f"  식당 {n:3d}개  총 {total:,}건  →  Train {train:,} / Val {val:,} / Test {test:,}  (60/20/20)")

print()
print("=== 현재 30K 대비 50K 확장의 이점 ===")
print("  현재 (30K, 80/20):")
print("    Train 24,000 / Test 6,000 / Val 없음 / 식당 100개")
print()
print("  확장 시 (50K, 60/20/20):")
print("    Train ~30,000 / Val ~10,000 / Test ~10,000 / 식당 ~150개")
print("    - Train 크기 비슷 + Val 확보로 Early stopping 개선")
print("    - Test 크기 증가 → 평가 신뢰도 향상")
print("    - 그래프 밀도: 노드 증가만큼 엣지도 증가")
