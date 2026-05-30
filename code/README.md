# itda GNN 사기 리뷰 탐지 — 코드 제출 패키지

## 프로젝트 개요

**목표**: Graph Neural Network(GNN)를 활용한 조직적 사기(스팸) 리뷰 네트워크 탐지

YelpZip 데이터셋(약 60만 건 이상의 식당 리뷰)에서 밀도 중심 서브그래프를 샘플링하고,
리뷰 간 다양한 관계를 헤테로 그래프로 모델링하여 사기 리뷰를 탐지한다.
예선(GNN 모델링) → 본선(설명가능 시각화 대시보드) 2단계 구성.

- **대회**: itda 학술제 (팀명: 먹스타)
- **메인 데이터셋**: YelpZip (Kaggle) — 라벨: 사기 `-1→1`, 정상 `1→0`
- **참고 데이터셋**: YelpChi, Amazon (엣지 구조 설계 참고)

---

## 전체 파이프라인

```
[1] EDA & 샘플링          01_eda_sampling.py
      ↓
[2] 피처 생성             02_features.py
      ↓
[3] 헤테로 그래프 구축    03_graph_build.py
      ↓
[4] 베이스라인 학습       04_train_baseline.py
      ↓
[5] 핵심 모델 학습        09_boost.py  →  15_drag_bwgat.py
      ↓
[6] 최종 앙상블           29_performance_boost.py  /  32_ensemble_4way.py
      ↓
[7] XAI & 시각화          12_xai_attribution.py  →  13_fraud_network_viz.py
      ↓
[8] 대시보드              dashboard/app.py
```

---

### [1] EDA & 밀도 중심 샘플링 (`01_eda_sampling.py`)

- 원본 데이터 로드 및 라벨 변환 (`-1→1 사기`, `1→0 정상`)
- **밀도 중심 샘플링**: 리뷰 수 상위 100개 식당(`prod_id`) 중심으로 30,000 노드 추출
  - 무작위 추출 금지 — 노드 간 연결성을 보존해야 GNN 학습 가능
- 시간순 **80/20 분할** (train 80% / test 20%), `random_state=42`
- 출력: `data/processed/df_sampled.parquet`

### [2] 노드 피처 생성 (`02_features.py`)

| 피처 | 방법 | 차원 |
|------|------|------|
| 리뷰 텍스트 임베딩 | SBERT `all-MiniLM-L6-v2`, L2 정규화 | 384d |
| 별점 | `[1,5] → [0,1]` min-max 정규화 | 1d |
| 타임스탬프 | min-max 정규화 | 1d |
| **합계 노드 피처** | | **386d** |

- 출력: `data/processed/sbert_embeddings.pt`, `node_features.pt`

### [3] 헤테로 그래프 구축 (`03_graph_build.py`)

5종 엣지 관계로 리뷰-리뷰 헤테로 그래프(`HeteroData`) 구성:

| 엣지 타입 | 연결 조건 | 비고 |
|-----------|-----------|------|
| **R-T-R** | 동일 `prod_id` + 동일 연-월 | 대회 기본 Relation |
| **R-S-R** | 동일 `prod_id` + 동일 별점 | 대회 기본 Relation |
| **R-U-R** | 동일 `user_id` (슬라이딩 윈도우 w=3) | 대회 기본 Relation |
| **R-Burst-R** | 동일 `prod_id` + 72시간 이내 (cKDTree, Δt 엣지 피처) | **커스텀**: 리뷰 캠페인 탐지 |
| **R-Sim-R** | 동일 `prod_id` + SBERT 코사인 유사도 ≥ 0.85 | **커스텀**: 복붙 리뷰 탐지 |

- 그룹당 최대 32개 노드 샘플링으로 O(n²) 엣지 폭발 방지
- 출력: `data/graphs/hetero_graph.pt`

### [4] 베이스라인 학습 (`04_train_baseline.py`)

- `HeteroSAGE`, `HeteroGAT`, `HeteroBWGNN` 3종 베이스라인
- 손실함수: Focal Loss (γ=2.0, α=0.75, 스팸 비율 13.2% 반영)
- 옵티마이저: AdamW (lr=5e-4, weight_decay=1e-5), Cosine Annealing LR

### [5] 핵심 모델 학습

#### R-Sim-R 추가 + Warm Restart (`09_boost.py`)
- 5번째 커스텀 엣지 R-Sim-R을 그래프에 추가
- 전체 모델 400 epoch Warm Restart 재학습 (LR=2e-4)

#### DRAG / BWGAT / DRAGWave (`15_drag_bwgat.py`) — 핵심 모델
- **DRAG** (Dynamic Relation-Attentive GNN, arXiv:2310.04171): 엣지 타입별 독립 임베딩 계산 후 노드마다 다른 동적 Attention으로 집계 → R-U-R 기여 20% vs R-S-R 0.2%의 비대칭 관계 자동 학습
- **BWGAT** (Beta Wavelet GAT): BWGNN의 low-pass / high-pass 필터를 GAT Attention으로 가중합 → "사기 노드는 이웃과 다르다(high-pass)" 특성 반영
- **DRAGWave**: DRAG의 관계 Attention + BWGAT의 주파수 필터를 결합한 최종 모델

### [6] 실험 및 앙상블

| 파일 | 내용 |
|------|------|
| `07_ablation.py` | 엣지 타입별 기여도 ablation 실험 |
| `20_fair_comparison.py` | 400 epoch 공정 비교 실험 |
| `27_future_directions.py` | DRAGWave_NoRSR / TVF(시간 속도 피처) 확장 |
| `29_performance_boost.py` | **3-way 앙상블** (Transductive 최종) |
| `32_ensemble_4way.py` | **4-way 인덕티브 앙상블** (배포 환경 시뮬레이션) |

**3-way 앙상블 구성** (`29_performance_boost.py`):
- DRAGWave_NoRSR + HeteroBWGNN_boost + DRAGWave_TVF → **PR-AUC 0.9419**

**4-way 인덕티브 앙상블 구성** (`32_ensemble_4way.py`):
- DRAGWave_TVF_400ep (inductive gap 18.5%) + DRAGWave_NoRSR + HeteroBWGNN_boost + BWGAT → **PR-AUC 0.7748**

### [7] XAI & 시각화

- **엣지 기여도** (`12_xai_attribution.py`): 엣지 타입 하나씩 제거 시 fraud_prob 변화 측정
- **Burst Δt 분포**: 사기 노드 주변 burst 엣지의 Δt가 정상 대비 짧음을 시각화
- **피처 이상도**: SBERT 임베딩이 정상 평균과 떨어진 거리 기반 설명
- **네트워크 시각화** (`13_fraud_network_viz.py`): 사기 리뷰 클러스터 구조 시각화

---

## 최종 성능

| 설정 | PR-AUC | Macro-F1 | 모델 |
|------|--------|----------|------|
| Transductive | **0.9419** | **0.9386** | 3-way Ensemble |
| Inductive | **0.7748** | — | 4-way Ensemble (배포 환경 시뮬레이션) |

---

## 프로젝트 구조

```
code/
├── src/
│   ├── config.py                  # 경로 및 공통 하이퍼파라미터 중앙 관리
│   ├── 01_eda_sampling.py         # YelpZip EDA + 30K 밀도 중심 샘플링
│   ├── 02_features.py             # SBERT 임베딩 + 노드 피처 (386d)
│   ├── 03_graph_build.py          # 5종 엣지 헤테로 그래프 구축
│   ├── 04_train_baseline.py       # HeteroSAGE / GAT / BWGNN 베이스라인
│   ├── 05_train_tgat.py           # TGATLite (Bochner 시간 인코딩)
│   ├── 09_boost.py                # R-Sim-R 추가 + Warm Restart 400ep
│   ├── 12_xai_attribution.py      # 엣지 기여도 XAI 분석
│   ├── 13_fraud_network_viz.py    # 사기 네트워크 구조 시각화
│   ├── 15_drag_bwgat.py           # DRAG / BWGAT / DRAGWave 핵심 모델
│   ├── 20_fair_comparison.py      # 400ep 공정 비교 실험
│   ├── 27_future_directions.py    # DRAGWave_NoRSR / TVF 확장 실험
│   ├── 29_performance_boost.py    # 3-way Ensemble (Transductive 최종)
│   ├── 32_ensemble_4way.py        # 4-way Inductive Ensemble
│   └── experiments/               # 과적합 완화 실험 (E01~E10c)
├── dashboard/
│   └── app.py                     # Streamlit 대시보드
├── requirements.txt
└── README.md
```

---

## 실행 순서

```bash
# 환경 설치
pip install -r requirements.txt

# [1] EDA & 샘플링
python src/01_eda_sampling.py

# [2] 피처 생성
python src/02_features.py

# [3] 헤테로 그래프 구축
python src/03_graph_build.py

# [4] 베이스라인 학습
python src/04_train_baseline.py

# [5] R-Sim-R 추가 + Warm Restart
python src/09_boost.py

# [6] 핵심 모델 (DRAG / BWGAT / DRAGWave)
python src/15_drag_bwgat.py

# [7] DRAGWave NoRSR / TVF 확장
python src/27_future_directions.py

# [8] 3-way Ensemble (Transductive 최종)
python src/29_performance_boost.py

# [9] 4-way Inductive Ensemble
python src/32_ensemble_4way.py

# [10] 대시보드 실행
streamlit run dashboard/app.py
```

---

## 대시보드

- **URL**: https://itda-gnn-dashboard-dbpfa9uth9hdkmnqyjh4cm.streamlit.app
- **GitHub**: https://github.com/ejongmin/itda-gnn-dashboard

---

## 재현성

- `seed=42` 전역 고정 (`config.py` → `set_seed(42)`)
- CPU 환경에서 완전 재현 보장 (`torch.backends.cudnn.deterministic = True`)
- 데이터 분할: 시간순 80/20, `random_state=42`
