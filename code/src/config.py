"""
config.py
프로젝트 경로 및 공통 설정 중앙 관리
절대경로 하드코딩 제거 → 어느 컴퓨터에서든 동작
"""

import random
import numpy as np
import torch
from pathlib import Path

# 프로젝트 루트: 이 파일(src/config.py)의 상위 디렉토리
BASE = Path(__file__).resolve().parent.parent

# 데이터 경로
RAW   = BASE / "data" / "raw"
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"
EXT   = BASE / "data" / "external"

# 결과 경로
RES  = BASE / "results"
MOD  = BASE / "models"
REP  = BASE / "reports"

# 디렉토리 자동 생성
for d in [RAW, PROC, GRAPH, EXT, RES, MOD, REP]:
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    """완전한 재현성 보장 — 모든 난수 생성기 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CPU에서 deterministic 연산 강제 (성능 약간 저하 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 학습 공통 하이퍼파라미터 (근거 주석 포함)
TRAIN_CONFIG = {
    "hidden_dim":   128,    # GNN 사기 탐지 표준값 (BWGNN, CARE-GNN 동일)
    "lr":           5e-4,   # AdamW 권장 범위 중간값
    "weight_decay": 1e-5,   # L2 정규화 기본값
    "dropout":      0.3,    # GNN 표준 dropout
    "epochs":       400,    # DRAGWave 완전 수렴에 필요한 epoch
    "patience":     30,     # Early stopping patience (epoch 단위)
    "focal_gamma":  2.0,    # Focal Loss 원논문(Lin et al. 2017) 권장값
    "focal_alpha":  0.75,   # 스팸 비율 13.2% 기반 → 소수 클래스 가중치
    "seed":         42,
    "batch_mode":   "full", # 30K 노드 전체 배치
    "grad_clip":    1.0,    # 기울기 폭발 방지
}

# 그래프 설계 파라미터 (근거 주석 포함)
GRAPH_CONFIG = {
    "burst_window_h": 72,   # 리뷰 캠페인 실행 주기 상한 (업계 통념 24~72h)
    "top_n_prods":    100,  # 그래프 밀도 최대화 (Top-150/200과 동일 결과 확인)
    "rtr_group_cap":  32,   # O(n²) 방지: GraphSAGE 권장 이웃 크기 25~50
    "rur_window":     3,    # 헤비 유저 슬라이딩 윈도우 (최근 행동 중시)
    "sim_threshold":  0.85, # R-Sim-R: 스팸 1.9배 관여율 최적 임계값
    "target_nodes":   30_000,
    "train_ratio":    0.8,
}

EDGE_TYPES = [
    ("review", "rtr",   "review"),
    ("review", "rsr",   "review"),
    ("review", "burst", "review"),
    ("review", "rur",   "review"),
    ("review", "sim",   "review"),
]
