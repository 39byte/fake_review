"""
01_eda_sampling.py
YelpZip EDA + 밀도 중심 서브그래프 샘플링 (30K 노드 목표)
대회 규정: 10K~50K 노드, 무작위 추출 금지, 시간순 80/20 분할
"""

import pandas as pd
import numpy as np
import torch
import os
import json
from pathlib import Path
from collections import Counter

BASE  = Path(__file__).resolve().parent.parent
RAW   = BASE / "data" / "raw"
PROC  = BASE / "data" / "processed"
RES   = BASE / "results"
PROC.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

# ── 1. 로드 & 라벨 변환 ──────────────────────────────────────────────────────
print("="*60)
print("[1] 데이터 로드 & 라벨 변환")
print("="*60)

df = pd.read_csv(RAW / "yelpzip.csv", low_memory=False)
print(f"원본 shape : {df.shape}")
print(f"컬럼       : {df.columns.tolist()}")
print(f"\n--- 첫 3행 ---")
print(df.head(3).to_string())

# 필수 라벨 변환: 사기=-1→1, 정상=1→0
print(f"\n원본 label 분포: {dict(Counter(df['label'].tolist()))}")
df['label'] = df['label'].map({-1: 1, 1: 0})
print(f"변환 label 분포: {dict(Counter(df['label'].tolist()))}")
spam_ratio = df['label'].mean()
print(f"스팸 비율: {spam_ratio:.3f} ({spam_ratio*100:.1f}%)")

# ── 2. 기본 EDA ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[2] 기본 EDA")
print("="*60)

print(f"\n결측치:\n{df.isnull().sum()}")
print(f"\ndtype:\n{df.dtypes}")

# date 컬럼 파싱 (실제 컬럼명: date 소문자)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(f"\ndate 파싱 실패: {df['date'].isnull().sum()}건")
print(f"date 범위: {df['date'].min()} ~ {df['date'].max()}")
print(f"연도 분포:\n{df['date'].dt.year.value_counts().sort_index()}")

# timestamp (초 단위)
df['timestamp'] = df['date'].astype('int64') // 10**9
df['timestamp'] = df['timestamp'].fillna(0).astype('int64')

# rating 분포
print(f"\nrating 분포:\n{df['rating'].value_counts().sort_index()}")

# prod_id (식당) 통계
prod_counts = df.groupby('prod_id').size().sort_values(ascending=False)
print(f"\n식당(prod_id) 수         : {len(prod_counts)}")
print(f"식당당 리뷰 수 (상위 10):\n{prod_counts.head(10).to_string()}")
print(f"식당당 리뷰 수 통계:\n{prod_counts.describe()}")

# user_id 통계
user_counts = df.groupby('user_id').size().sort_values(ascending=False)
print(f"\n유저(user_id) 수         : {len(user_counts)}")
print(f"유저당 리뷰 수 (상위 10):\n{user_counts.head(10).to_string()}")
print(f"유저당 리뷰 수 통계:\n{user_counts.describe()}")

# ── 3. 시간 분포 분석 ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[3] 시간 분포 분석 (TGAT 시간 split 설계)")
print("="*60)

df_valid_date = df[df['date'].notnull()].copy()
yearly = df_valid_date.groupby(df_valid_date['date'].dt.year).agg(
    count=('label', 'size'),
    spam_count=('label', 'sum')
).assign(spam_ratio=lambda x: x['spam_count'] / x['count'])
print(f"\n연도별 리뷰 수 & 스팸 비율:\n{yearly.to_string()}")

# 80/20 시간 분할 지점
df_sorted = df_valid_date.sort_values('date')
n_total = len(df_sorted)
cutoff_idx = int(n_total * 0.8)
cutoff_date = df_sorted.iloc[cutoff_idx]['date']
print(f"\n시간순 80/20 분할 기준일: {cutoff_date.date()}")
print(f"  train+valid: {cutoff_idx:,}건 ({cutoff_idx/n_total:.1%})")
print(f"  test       : {n_total - cutoff_idx:,}건 ({(n_total-cutoff_idx)/n_total:.1%})")

# ── 4. 샘플링 전략 (전략 A+B+C 혼합, 30K 목표, 스팸 비율 보존) ─────────────
print("\n" + "="*60)
print("[4] 밀도 중심 샘플링 (전략 A+B+C 혼합, 스팸 비율 보존)")
print("="*60)

TARGET_NODES = 30_000

# Step 1: 리뷰 집중 상위 100개 식당 풀 구성 (전략 A — 범위 넓혀서 스팸 다양성 확보)
top_prods = prod_counts.head(100).index.tolist()
df_step1 = df[df['prod_id'].isin(top_prods)].copy()
print(f"Step 1 - 상위 100개 식당 리뷰 수: {len(df_step1):,}")
print(f"         스팸 비율: {df_step1['label'].mean():.3f}")

# Step 2: 상위 식당 내에서 활성 유저 선택 (전략 B, ≥2 리뷰)
user_in_top_prods = df_step1.groupby('user_id').size().sort_values(ascending=False)
active_users = user_in_top_prods[user_in_top_prods >= 2].index.tolist()
df_step2 = df[df['user_id'].isin(active_users) & df['prod_id'].isin(top_prods)].copy()
print(f"Step 2 - 활성 유저(≥2 리뷰) + 상위 식당: {len(df_step2):,}")
print(f"         스팸 비율: {df_step2['label'].mean():.3f}")

# Step 3: 전체 기간에서 스팸 비율 보존하며 30K 추출
# 스팸/정상을 각각 목표 비율로 추출 (원본 스팸 비율 13.2% 근사)
target_spam_ratio = spam_ratio  # 0.132
n_spam_target   = int(TARGET_NODES * target_spam_ratio)
n_normal_target = TARGET_NODES - n_spam_target

df_spam   = df_step2[df_step2['label'] == 1].copy()
df_normal = df_step2[df_step2['label'] == 0].copy()

print(f"\n풀 내 스팸: {len(df_spam):,}, 정상: {len(df_normal):,}")

# 스팸이 부족하면 상위 100개 외 식당도 포함 (전략 C — 시간축 확장)
if len(df_spam) < n_spam_target:
    # 스팸 리뷰가 많은 식당 추가
    prod_spam_counts = df[df['label']==1].groupby('prod_id').size().sort_values(ascending=False)
    extra_prods = prod_spam_counts.head(150).index.tolist()
    df_spam_extra = df[df['prod_id'].isin(extra_prods) & (df['label']==1)].copy()
    df_spam = pd.concat([df_spam, df_spam_extra]).drop_duplicates()
    print(f"스팸 보충 후: {len(df_spam):,}")

# 시간순으로 정렬 후 비율 보존 샘플링 (burst 패턴 보존 위해 시간순 유지)
df_spam_sample   = df_spam.sort_values('date').tail(n_spam_target).copy()
df_normal_sample = df_normal.sort_values('date').tail(n_normal_target).copy()

df_sample = pd.concat([df_spam_sample, df_normal_sample]).sort_values('date').reset_index(drop=True)

print(f"\n최종 샘플 크기: {len(df_sample):,}")
print(f"스팸 비율 (목표: {target_spam_ratio:.3f}): {df_sample['label'].mean():.3f}")
print(f"식당 수: {df_sample['prod_id'].nunique()}")
print(f"유저 수: {df_sample['user_id'].nunique()}")

# ── 5. 시간 분할 적용 ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[5] 시간순 분할 적용")
print("="*60)

df_sample = df_sample.sort_values('date').reset_index(drop=True)
n_sample = len(df_sample)
train_cutoff = int(n_sample * 0.8)

df_sample['split'] = 'test'
df_sample.loc[:train_cutoff-1, 'split'] = 'train'

train_cut_date = df_sample.iloc[train_cutoff - 1]['date']
print(f"train/test 분할 기준일: {train_cut_date.date()}")
print(f"train: {(df_sample['split']=='train').sum():,}건")
print(f"test : {(df_sample['split']=='test').sum():,}건")
print(f"train 스팸 비율: {df_sample[df_sample['split']=='train']['label'].mean():.3f}")
print(f"test  스팸 비율: {df_sample[df_sample['split']=='test']['label'].mean():.3f}")

# ── 6. 노드 인덱스 부여 & 저장 ───────────────────────────────────────────────
print("\n" + "="*60)
print("[6] 저장")
print("="*60)

df_sample = df_sample.reset_index(drop=True)
df_sample['node_id'] = df_sample.index  # GNN 노드 인덱스

df_sample.to_parquet(PROC / "df_sampled.parquet", index=False)
print(f"저장: {PROC / 'df_sampled.parquet'} ({len(df_sample):,}행)")

# ── 7. EDA 요약 저장 ─────────────────────────────────────────────────────────
eda_summary = {
    "원본_총_리뷰수": len(df),
    "샘플_리뷰수": len(df_sample),
    "스팸_비율_원본": round(float(spam_ratio), 4),
    "스팸_비율_샘플": round(float(df_sample['label'].mean()), 4),
    "식당_수": int(df_sample['prod_id'].nunique()),
    "유저_수": int(df_sample['user_id'].nunique()),
    "train_비율": 0.8,
    "test_비율": 0.2,
    "train_cut_date": str(train_cut_date.date()),
    "날짜_범위_min": str(df_sample['date'].min().date()),
    "날짜_범위_max": str(df_sample['date'].max().date()),
    "rating_분포": df_sample['rating'].value_counts().sort_index().to_dict(),
    "random_state": "N/A — 시간순 정렬 분할 (재현 가능)",
}

with open(RES / "eda_summary.json", "w", encoding="utf-8") as f:
    json.dump(eda_summary, f, ensure_ascii=False, indent=2)
print(f"저장: {RES / 'eda_summary.json'}")

print("\n" + "="*60)
print("✅ EDA + 샘플링 완료")
print(f"   → 다음 단계: 02_features.py (SBERT 임베딩)")
print("="*60)
