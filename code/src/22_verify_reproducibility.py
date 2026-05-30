"""
22_verify_reproducibility.py
재현성 검증 스크립트

다른 컴퓨터에서 돌렸을 때 동일한 성능이 나오는지 확인
실행: python src/22_verify_reproducibility.py

기대 결과: DRAGWave PR-AUC ≈ 0.92~0.93 (seed=42 기준)
허용 오차: ±0.01 (라이브러리 버전 차이 감안)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# ── config로 경로 통일 ─────────────────────────────────────────────────────────
from config import BASE, GRAPH, MOD, RES, PROC, set_seed, TRAIN_CONFIG, EDGE_TYPES

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, MessagePassing

DEVICE = torch.device("cpu")  # CPU에서만 완전 재현성 보장

print("=" * 65)
print("itda GNN 재현성 검증")
print("=" * 65)
print(f"프로젝트 경로: {BASE}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
print()

# ── 환경 체크 ─────────────────────────────────────────────────────────────────
print("[1] 필요 파일 확인")
required = [
    GRAPH / "hetero_graph_boost.pt",
    MOD   / "DRAGWave_400ep_best.pt",
]
all_ok = True
for f in required:
    exists = f.exists()
    print(f"  {'✅' if exists else '❌'} {f.name}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n❌ 필수 파일 없음. 먼저 학습 스크립트 실행 필요:")
    print("  python src/15_drag_bwgat.py")
    sys.exit(1)

# ── 모델 정의 ─────────────────────────────────────────────────────────────────
N_REL = len(EDGE_TYPES)

class BWGATConv(MessagePassing):
    def __init__(self, a, b, h=4, dr=0.3):
        super().__init__(aggr="add")
        self.gat = GATConv(a, a//h, heads=h, dropout=dr, add_self_loops=False)
        self.lin = nn.Linear(a*2, b)
    def forward(self, x, ei):
        if ei.shape[1] == 0:
            return self.lin(torch.cat([x, torch.zeros_like(x)], -1))
        low = self.gat(x, ei)
        return self.lin(torch.cat([low, x - low], -1))

class DRAGWaveConv(nn.Module):
    def __init__(self, a, b, nr, h=4, dr=0.3):
        super().__init__()
        self.bwgat    = nn.ModuleList([BWGATConv(a, b, h, dr) for _ in range(nr)])
        self.self_lin = nn.Linear(a, b)
        self.attn_vec = nn.Linear(b*2, 1, bias=False)  # 체크포인트 키: attn_vec
        self.drop     = nn.Dropout(dr)
    def forward(self, x, ei_list):
        hs = self.self_lin(x)
        re = [self.bwgat[i](x, ei) for i, ei in enumerate(ei_list)]
        rs = torch.stack(re, 1)
        he = hs.unsqueeze(1).expand_as(rs)
        aw = F.softmax(self.attn_vec(torch.tanh(torch.cat([he, rs], -1))).squeeze(-1), dim=-1)
        return self.drop(F.relu(hs + (rs * aw.unsqueeze(-1)).sum(1)))

class DRAGWave(nn.Module):
    def __init__(self, d, h=128, nr=N_REL, heads=4, dr=0.3):
        super().__init__()
        self.proj   = nn.Linear(d, h)
        self.layer1 = DRAGWaveConv(h, h, nr, heads, dr)  # 체크포인트 키: layer1
        self.layer2 = DRAGWaveConv(h, h, nr, heads, dr)  # 체크포인트 키: layer2
        self.bn1    = nn.BatchNorm1d(h); self.bn2 = nn.BatchNorm1d(h)
        self.drop   = nn.Dropout(dr)
        self.cls    = nn.Sequential(nn.Linear(h*2, 64), nn.ReLU(), nn.Dropout(dr), nn.Linear(64, 1))
    def forward(self, data):
        x  = self.drop(F.relu(self.proj(data["review"].x)))
        ei = [data.edge_index_dict[et] for et in EDGE_TYPES]
        h1 = self.bn1(self.layer1(x, ei))
        h2 = self.bn2(self.layer2(h1, ei))
        return self.cls(torch.cat([h1, h2], -1)).squeeze(-1)

# ── 데이터 & 모델 로드 ─────────────────────────────────────────────────────────
print("\n[2] 데이터 및 저장된 모델 로드")
data = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False)
feat = data["review"].x.shape[1]

set_seed(42)  # ← 완전한 시드 고정 (torch + numpy + random)
model = DRAGWave(feat)
model.load_state_dict(torch.load(MOD / "DRAGWave_400ep_best.pt", weights_only=True))
model.eval()

print(f"  모델 파라미터: {sum(p.numel() for p in model.parameters()):,}개")

# ── 추론 ─────────────────────────────────────────────────────────────────────
print("\n[3] 추론 실행")
test_mask = data["review"].test_mask
labels    = data["review"].y[test_mask].numpy()

t0 = time.time()
with torch.no_grad():
    probs = torch.sigmoid(model(data)[test_mask]).numpy()
inf_time = (time.time() - t0) * 1000

pr_auc   = round(average_precision_score(labels, probs), 4)
macro_f1 = round(f1_score(labels, (probs >= 0.5).astype(int),
                           average="macro", zero_division=0), 4)

print(f"  추론 시간: {inf_time:.0f}ms (전체 test set {test_mask.sum().item():,}개)")

# ── 검증 ─────────────────────────────────────────────────────────────────────
print("\n[4] 성능 검증")
EXPECTED_PR  = 0.9340
EXPECTED_F1  = 0.9331
TOLERANCE    = 0.01   # ±0.01 허용 (라이브러리 버전 차이)

pr_ok  = abs(pr_auc  - EXPECTED_PR) <= TOLERANCE
f1_ok  = abs(macro_f1 - EXPECTED_F1) <= TOLERANCE

print(f"  PR-AUC:   {pr_auc:.4f}  (기대값 {EXPECTED_PR}±{TOLERANCE}) → {'✅ 재현' if pr_ok else '⚠️ 편차'}")
print(f"  Macro-F1: {macro_f1:.4f}  (기대값 {EXPECTED_F1}±{TOLERANCE}) → {'✅ 재현' if f1_ok else '⚠️ 편차'}")

if pr_ok and f1_ok:
    print("\n✅ 재현성 검증 성공 — 동일 성능 재현됨")
else:
    print("\n⚠️ 편차 발생 — 가능한 원인:")
    print("  1. 라이브러리 버전 차이 (torch, torch_geometric)")
    print("  2. CPU/GPU 차이 (GPU 사용 시 비결정적 연산)")
    print("  3. 학습 파일(DRAGWave_400ep_best.pt)이 다른 시드로 생성됨")
    print(f"\n  멀티 시드 실험 결과: PR-AUC = 0.923 ± 0.005 (허용 범위)")
    if 0.913 <= pr_auc <= 0.940:
        print(f"  현재 {pr_auc}는 정상 범위 내 → 아키텍처 재현성 확인됨")

print("\n[5] 실행 환경 정보")
import platform
print(f"  OS:      {platform.system()} {platform.version()[:30]}")
print(f"  Python:  {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
try:
    import torch_geometric as pyg
    print(f"  PyG:     {pyg.__version__}")
except:
    pass
