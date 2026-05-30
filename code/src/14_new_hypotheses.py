"""
14_new_hypotheses.py
itda GNN 본선 대비 — 새 가설 3개 구현 스케치

가설 1: Temporal Velocity Feature (TVF)
  - 핵심: 노드 피처에 "리뷰 속도" 정보를 직접 주입
  - Δt_mean, Δt_std, burst_rank 등 시간 통계를 노드 피처로 추가
  - 난이도: 하 | 예상 임팩트: 인덕티브 PR-AUC +0.03~0.05

가설 2: Cascading Suspicion Propagation (CSP)
  - 핵심: 사기 확률을 그래프 위에서 반복 전파 (GNN 이후 post-processing)
  - PageRank 방식으로 이웃 노드의 사기 점수를 전파해 캠페인 단위 탐지 강화
  - 난이도: 중 | 예상 임팩트: Recall +5%p, 특히 sparse 스팸 클러스터

가설 3: Contrastive Temporal Pretraining (CTP)
  - 핵심: 라벨 없이 burst 패턴으로 대조 학습 pretrain → fine-tune
  - 같은 burst 그룹 내 리뷰를 positive pair로 정의
  - 인덕티브 갭 해소 기대 (0.92→0.66 문제 직접 공략)
  - 난이도: 상 | 예상 임팩트: 인덕티브 PR-AUC 0.66→0.75+ 목표
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv

BASE  = Path(__file__).resolve().parent.parent
PROC  = BASE / "data" / "processed"
GRAPH = BASE / "data" / "graphs"
RES   = BASE / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════════
# 가설 1: Temporal Velocity Feature (TVF)
# ──────────────────────────────────────────────────────────────────────────────
# 현재 한계: 시간 정보가 오직 R-Burst-R 엣지의 edge_attr(Δt)로만 존재.
#   → 인덕티브 세팅에서 burst 엣지가 잘려나가면 시간 신호 완전 소실.
#
# 제안: 각 리뷰 노드에 "velocity 피처"를 직접 추가.
#   - burst_degree   : 해당 노드의 72h 내 연결 이웃 수 (burst graph의 degree)
#   - delta_t_mean   : burst 이웃들과의 평균 Δt (단위: 시간)
#   - delta_t_min    : burst 이웃들과의 최소 Δt (가장 빠른 공모)
#   - burst_rank     : 같은 식당 내 시간순 순위 (0=가장 빠름, 1=가장 늦음)
#   - hour_of_day    : 등록 시각 (0~23), sin/cos 인코딩
#   - day_of_week    : 요일 (0=월), sin/cos 인코딩
#
# 근거: 인덕티브 설정에서도 노드 피처는 항상 사용 가능.
#   엣지가 없어도 "이 리뷰는 새벽 2시에, 같은 식당의 3번째 리뷰로 등록됨"
#   → 이 자체가 사기 신호다.
# ══════════════════════════════════════════════════════════════════════════════

def build_temporal_velocity_features(df: pd.DataFrame) -> torch.Tensor:
    """
    입력: df_sampled.parquet (columns: node_id, prod_id, timestamp, date)
    출력: velocity_features [N, 8] tensor
      [burst_degree, dt_mean, dt_min, dt_std, burst_rank, hour_sin, hour_cos, dow_sin]
    """
    N = len(df)
    feats = np.zeros((N, 8), dtype=np.float32)

    BURST_SEC = 72 * 3600
    df = df.copy().reset_index(drop=True)
    ts = df["timestamp"].values.astype(np.float64)

    # (1) burst degree, Δt 통계
    burst_degree  = np.zeros(N, dtype=np.float32)
    dt_sum        = np.zeros(N, dtype=np.float32)
    dt_min_arr    = np.full(N, np.inf, dtype=np.float32)
    dt_sq_sum     = np.zeros(N, dtype=np.float32)
    burst_count   = np.zeros(N, dtype=np.int32)

    for _, grp in df.groupby("prod_id"):
        nids = grp["node_id"].values
        ts_g = grp["timestamp"].values.astype(np.float64)
        n = len(nids)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                dt_sec = abs(ts_g[i] - ts_g[j])
                if dt_sec <= BURST_SEC:
                    dt_h = dt_sec / 3600.0
                    ni, nj = int(nids[i]), int(nids[j])
                    burst_degree[ni] += 1
                    burst_degree[nj] += 1
                    dt_sum[ni] += dt_h
                    dt_sum[nj] += dt_h
                    dt_sq_sum[ni] += dt_h ** 2
                    dt_sq_sum[nj] += dt_h ** 2
                    burst_count[ni] += 1
                    burst_count[nj] += 1
                    if dt_h < dt_min_arr[ni]:
                        dt_min_arr[ni] = dt_h
                    if dt_h < dt_min_arr[nj]:
                        dt_min_arr[nj] = dt_h

    safe_count = np.maximum(burst_count, 1).astype(np.float32)
    dt_mean = dt_sum / safe_count
    dt_var  = np.maximum(dt_sq_sum / safe_count - dt_mean ** 2, 0.0)
    dt_std  = np.sqrt(dt_var)
    dt_min_arr[dt_min_arr == np.inf] = 0.0

    # (2) burst_rank: 같은 식당 내 시간순 순위 (0=가장 빠름)
    burst_rank = np.zeros(N, dtype=np.float32)
    for _, grp in df.groupby("prod_id"):
        nids = grp["node_id"].values
        ts_g = grp["timestamp"].values.astype(np.float64)
        order = np.argsort(ts_g)
        n = len(nids)
        for rank_i, orig_i in enumerate(order):
            burst_rank[int(nids[orig_i])] = rank_i / max(n - 1, 1)  # [0, 1] 정규화

    # (3) 주기적 시간 인코딩 (sin/cos)
    # timestamp → datetime
    datetimes = pd.to_datetime(ts, unit="s", utc=True)
    hours   = datetimes.hour.values.astype(np.float32)
    dow     = datetimes.dayofweek.values.astype(np.float32)
    hour_sin = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    dow_sin  = np.sin(2 * np.pi * dow / 7).astype(np.float32)

    # (4) 정규화 (burst_degree → log1p, dt → /72)
    feats[:, 0] = np.log1p(burst_degree)
    feats[:, 1] = dt_mean / 72.0
    feats[:, 2] = dt_min_arr / 72.0
    feats[:, 3] = dt_std / 72.0
    feats[:, 4] = burst_rank
    feats[:, 5] = hour_sin
    feats[:, 6] = hour_cos
    feats[:, 7] = dow_sin

    return torch.tensor(feats, dtype=torch.float32)


def add_velocity_features_to_graph(graph_path: Path, proc_path: Path) -> HeteroData:
    """
    기존 hetero_graph.pt에 velocity_features를 concat해서 반환.
    노드 피처: [원본 386d] + [velocity 8d] = 394d
    """
    data = torch.load(graph_path, weights_only=False)
    df   = pd.read_parquet(proc_path / "df_sampled.parquet")
    vel  = build_temporal_velocity_features(df)  # [N, 8]

    data["review"].x = torch.cat([data["review"].x, vel], dim=-1)
    print(f"[TVF] 노드 피처: {386} → {data['review'].x.shape[1]}")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# 가설 2: Cascading Suspicion Propagation (CSP)
# ──────────────────────────────────────────────────────────────────────────────
# 현재 한계: GNN은 1~2-hop 이웃을 집계하지만 캠페인 단위의 "의심 전파"가
#   충분히 반영되지 않음. 특히 R-Sim-R 클러스터에서 중간 노드가 걸러지면
#   전파가 끊김.
#
# 제안: GNN 추론 후 사기 확률 p_i를 그래프 위에서 K번 반복 전파.
#   p_i^(k+1) = α * p_i^(0) + (1-α) * mean_{j∈N(i)} p_j^(k)
#
#   - p_i^(0): GNN이 출력한 원래 사기 확률 (anchor)
#   - α: anchor 강도 (0.3 권장) — 너무 작으면 전파 과잉
#   - K: 전파 횟수 (3~5)
#   - N(i): 모든 엣지 타입의 이웃 합집합
#
# 근거: 사기 캠페인은 연결된 리뷰들이 공동 운명이다.
#   한 노드의 사기 증거가 강하면 이웃들도 재심사해야 한다.
#   이는 GNN 구조 변경 없이 추론 후처리로 구현 가능 → 난이도 낮음.
#
# 기대 효과: Recall 향상 (현재 ~0.93 → 0.95+), False Negative 감소
# ══════════════════════════════════════════════════════════════════════════════

class CascadingSuspicionPropagation:
    """
    GNN 추론 후 사기 확률 전파 (post-processing, 학습 불필요).

    사용법:
        csp = CascadingSuspicionPropagation(alpha=0.3, K=4)
        refined_probs = csp(initial_probs, data)  # [N] tensor
    """
    def __init__(self, alpha: float = 0.3, K: int = 4):
        self.alpha = alpha  # anchor 강도
        self.K     = K      # 전파 반복 횟수

    def __call__(self, probs: torch.Tensor, data: HeteroData) -> torch.Tensor:
        """
        probs: [N] float tensor (GNN 출력 sigmoid 확률)
        data : HeteroData (엣지 정보 포함)
        return: [N] refined 확률
        """
        N = probs.shape[0]
        p = probs.clone().to(DEVICE)  # 현재 확률
        p0 = p.clone()                # anchor (GNN 원래 확률)

        # 모든 엣지 타입의 인접 행렬을 sparse COO로 합산
        all_src, all_dst = [], []
        for et in data.edge_index_dict:
            ei = data.edge_index_dict[et]
            all_src.append(ei[0])
            all_dst.append(ei[1])

        if len(all_src) == 0:
            return p

        src = torch.cat(all_src).to(DEVICE)
        dst = torch.cat(all_dst).to(DEVICE)

        # 중복 제거 (undirected 처리)
        edge_cat = torch.stack([src, dst], dim=0)
        edge_cat = torch.unique(edge_cat, dim=1)
        src_u, dst_u = edge_cat[0], edge_cat[1]

        # degree (이웃 수)
        deg = torch.zeros(N, device=DEVICE)
        deg.scatter_add_(0, dst_u, torch.ones(dst_u.shape[0], device=DEVICE))
        deg = deg.clamp(min=1.0)

        for _ in range(self.K):
            # 이웃 확률 합산
            neighbor_sum = torch.zeros(N, device=DEVICE)
            neighbor_sum.scatter_add_(0, dst_u, p[src_u])
            neighbor_mean = neighbor_sum / deg
            # anchor + 이웃 평균
            p = self.alpha * p0 + (1 - self.alpha) * neighbor_mean

        return p.cpu()

    def evaluate(self, probs: torch.Tensor, data: HeteroData, mask: torch.Tensor,
                 labels: torch.Tensor) -> dict:
        """CSP 전후 성능 비교"""
        refined = self(probs, data)
        pr_before = average_precision_score(labels[mask].numpy(), probs[mask].numpy())
        pr_after  = average_precision_score(labels[mask].numpy(), refined[mask].numpy())
        f1_before = f1_score(labels[mask].numpy(), (probs[mask] >= 0.5).numpy(),
                             average="macro", zero_division=0)
        f1_after  = f1_score(labels[mask].numpy(), (refined[mask] >= 0.5).numpy(),
                             average="macro", zero_division=0)
        print(f"[CSP] PR-AUC: {pr_before:.4f} → {pr_after:.4f} "
              f"({pr_after - pr_before:+.4f})")
        print(f"[CSP] Macro-F1: {f1_before:.4f} → {f1_after:.4f} "
              f"({f1_after - f1_before:+.4f})")
        return {
            "pr_before": pr_before, "pr_after": pr_after,
            "f1_before": f1_before, "f1_after": f1_after,
            "refined_probs": refined,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 가설 3: Contrastive Temporal Pretraining (CTP)
# ──────────────────────────────────────────────────────────────────────────────
# 현재 한계 (인덕티브 갭 직접 공략):
#   트랜스덕티브 PR-AUC=0.92 vs 인덕티브 PR-AUC=0.66 (갭=0.26)
#   근본 원인: GNN이 train-test 연결 엣지에 과도하게 의존.
#   학습 때 본 구조가 test에서 없어지면 성능 급락.
#
# 제안: 라벨 없이 "burst 패턴"만으로 대조 학습 사전학습(pretrain).
#
#   Positive pair 정의:
#     같은 R-Burst-R 클러스터 내 두 리뷰 (시간적으로 가까운 공모 의심 쌍)
#   Negative pair 정의:
#     다른 식당의 리뷰, 또는 72시간 이상 떨어진 리뷰
#
#   손실함수: InfoNCE (NT-Xent)
#     L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) ]
#
#   학습 절차:
#     1. Pretrain: 라벨 없이 burst 쌍으로 인코더 학습 (자기지도)
#     2. Fine-tune: 사전학습된 인코더 + 분류 헤드로 라벨 학습
#
#   핵심 직관:
#     "사기 그룹 내 리뷰들은 시간적으로 가까울수록 유사한 표현을 가져야 한다"
#     → 이 사전지식을 구조 정보 없이 피처 공간에 주입
#     → test 노드가 고립되어도 노드 피처 자체가 사기 패턴을 인코딩
#
# 기대 효과: 인덕티브 PR-AUC 0.66 → 0.73+ 목표
# ══════════════════════════════════════════════════════════════════════════════

class BurstContrastiveEncoder(nn.Module):
    """
    대조 학습용 인코더.
    입력: 노드 피처 [N, in_ch]
    출력: 정규화된 임베딩 [N, proj_dim]
    """
    def __init__(self, in_ch: int, hidden: int = 128, proj_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        # Projection head (SimCLR 방식: pretrain 후 제거)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """fine-tune 시 사용하는 표현 (projection head 제외)"""
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """pretrain 시 사용 (projection head 포함)"""
        h = self.encoder(x)
        z = self.proj_head(h)
        return F.normalize(z, dim=-1)  # 단위 구 위로 정규화


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor,
                 tau: float = 0.07) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    z_i, z_j: [B, D] 각각 positive pair의 앵커/포지티브 임베딩

    같은 배치 내 다른 쌍을 자동으로 negative로 사용.
    """
    B = z_i.shape[0]
    # [2B, D] concat
    z = torch.cat([z_i, z_j], dim=0)
    # [2B, 2B] 코사인 유사도 행렬
    sim = torch.mm(z, z.T) / tau
    # 자기 자신 제거 (대각 -inf)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # positive 인덱스: i의 positive는 i+B, i+B의 positive는 i
    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B)
    ]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss


def build_burst_pairs(data: HeteroData, max_pairs: int = 8192,
                      seed: int = 42) -> tuple:
    """
    R-Burst-R 엣지에서 대조 학습용 positive pair 샘플링.

    반환: (anchor_ids, positive_ids) — 각각 [P] long tensor
    """
    rng = np.random.default_rng(seed)
    ei = data["review", "burst", "review"].edge_index
    ea = data["review", "burst", "review"].edge_attr.squeeze(-1)  # Δt

    # 짧은 Δt 쌍 우선 (가장 강한 burst 신호)
    dt_np = ea.cpu().numpy()
    src_np = ei[0].cpu().numpy()
    dst_np = ei[1].cpu().numpy()

    # 단방향만 (src < dst)
    mask_dir = src_np < dst_np
    src_np, dst_np, dt_np = src_np[mask_dir], dst_np[mask_dir], dt_np[mask_dir]

    # Δt 기준 정렬 → 상위 max_pairs 선택
    order = np.argsort(dt_np)
    n_select = min(max_pairs, len(order))
    order = order[:n_select]

    anchor_ids   = torch.tensor(src_np[order], dtype=torch.long)
    positive_ids = torch.tensor(dst_np[order], dtype=torch.long)
    return anchor_ids, positive_ids


def pretrain_ctp(data: HeteroData, epochs: int = 50, batch_size: int = 512,
                 lr: float = 1e-3, tau: float = 0.07, seed: int = 42) -> BurstContrastiveEncoder:
    """
    대조 학습 사전학습 메인 함수.

    사용법:
        encoder = pretrain_ctp(data, epochs=50)
        # → encoder.encode(x) 로 표현 추출 후 분류 헤드 fine-tune
    """
    torch.manual_seed(seed)
    in_ch = data["review"].x.shape[1]
    encoder = BurstContrastiveEncoder(in_ch).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    x = data["review"].x.to(DEVICE)

    anchor_ids, pos_ids = build_burst_pairs(data, max_pairs=16384, seed=seed)
    n_pairs = len(anchor_ids)
    print(f"[CTP] Pretrain pairs: {n_pairs:,}  epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        encoder.train()
        # 배치 셔플
        perm = torch.randperm(n_pairs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_pairs, batch_size):
            idx = perm[start: start + batch_size]
            if len(idx) < 2:
                continue
            a_idx = anchor_ids[idx].to(DEVICE)
            p_idx = pos_ids[idx].to(DEVICE)

            z_a = encoder.project(x[a_idx])
            z_p = encoder.project(x[p_idx])
            loss = nt_xent_loss(z_a, z_p, tau=tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"  [CTP] epoch={epoch:3d}  loss={avg_loss:.4f}")

    print("[CTP] Pretrain 완료.")
    return encoder


class CTPClassifier(nn.Module):
    """
    사전학습된 인코더 위에 분류 헤드를 붙인 fine-tune 모델.
    인덕티브 설정에서 엣지 없이도 노드 피처만으로 동작.
    """
    def __init__(self, encoder: BurstContrastiveEncoder, hidden: int = 128,
                 freeze_encoder: bool = False, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.cls = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder.encode(x)      # [N, hidden]
        return self.cls(h).squeeze(-1)  # [N]


def finetune_ctp(encoder: BurstContrastiveEncoder, data: HeteroData,
                 epochs: int = 100, lr: float = 5e-4,
                 freeze_encoder: bool = False) -> dict:
    """
    사전학습 인코더 fine-tune + 평가.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.75):
            super().__init__()
            self.gamma, self.alpha = gamma, alpha
        def forward(self, logits, targets):
            bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
            pt  = torch.exp(-bce)
            w   = torch.where(targets == 1,
                              torch.full_like(bce, self.alpha),
                              torch.full_like(bce, 1 - self.alpha))
            return (w * (1 - pt) ** self.gamma * bce).mean()

    model = CTPClassifier(encoder, freeze_encoder=freeze_encoder).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch   = CosineAnnealingLR(opt, T_max=epochs)
    crit  = FocalLoss()

    x  = data["review"].x.to(DEVICE)
    y  = data["review"].y.to(DEVICE)
    tm = data["review"].train_mask
    vm = data["review"].test_mask

    best_pr, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = crit(model(x)[tm], y[tm])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                probs  = torch.sigmoid(model(x)[vm]).cpu().numpy()
                labels = y[vm].cpu().numpy()
            pr  = average_precision_score(labels, probs)
            f1  = f1_score(labels, probs >= 0.5, average="macro", zero_division=0)
            print(f"  [CTP-FT] ep={epoch:3d}  loss={loss.item():.4f}  "
                  f"PR-AUC={pr:.4f}  F1={f1:.4f}")
            if pr > best_pr:
                best_pr = pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs  = torch.sigmoid(model(x)[vm]).cpu().numpy()
        labels = y[vm].cpu().numpy()
    pr  = average_precision_score(labels, probs)
    f1  = f1_score(labels, probs >= 0.5, average="macro", zero_division=0)
    print(f"\n[CTP] FINAL PR-AUC={pr:.4f}  Macro-F1={f1:.4f}")
    return {"PR-AUC": round(pr, 4), "Macro-F1": round(f1, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 실행 진입점
# ══════════════════════════════════════════════════════════════════════════════

def run_hypothesis_1():
    """가설 1: Temporal Velocity Feature — 노드 피처 강화 후 그래프 저장"""
    print("\n" + "=" * 60)
    print("가설 1: Temporal Velocity Feature (TVF)")
    print("=" * 60)
    data = add_velocity_features_to_graph(
        GRAPH / "hetero_graph.pt", PROC
    )
    save_path = GRAPH / "hetero_graph_tvf.pt"
    torch.save(data, save_path)
    print(f"저장: {save_path}")
    print("다음 단계: 04_train_baseline.py / 09_boost.py 에서 hetero_graph_tvf.pt 사용")
    return data


def run_hypothesis_2(probs: torch.Tensor = None):
    """
    가설 2: Cascading Suspicion Propagation
    probs를 넣으면 CSP 전후 성능 비교, 없으면 데모 실행.
    """
    print("\n" + "=" * 60)
    print("가설 2: Cascading Suspicion Propagation (CSP)")
    print("=" * 60)
    data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)

    if probs is None:
        print("[CSP] probs가 없어 랜덤 확률로 데모 실행")
        N = data["review"].x.shape[0]
        probs = torch.sigmoid(torch.randn(N) * 2.0)  # 임의 확률

    csp = CascadingSuspicionPropagation(alpha=0.3, K=4)

    test_mask = data["review"].test_mask
    labels    = data["review"].y
    result    = csp.evaluate(probs, data, test_mask, labels)
    return result


def run_hypothesis_3():
    """가설 3: Contrastive Temporal Pretraining"""
    print("\n" + "=" * 60)
    print("가설 3: Contrastive Temporal Pretraining (CTP)")
    print("=" * 60)
    data = torch.load(GRAPH / "hetero_graph.pt", weights_only=False)
    data = data.to(DEVICE)

    # Step 1: Pretrain
    encoder = pretrain_ctp(data, epochs=50, batch_size=512, tau=0.07)

    # Step 2: Fine-tune (인코더 고정 vs 전체 학습 비교)
    print("\n[CTP] Fine-tune (인코더 동결 — feature extractor mode)")
    result_frozen = finetune_ctp(encoder, data, epochs=100, freeze_encoder=True)

    print("\n[CTP] Fine-tune (전체 학습 — end-to-end mode)")
    result_e2e = finetune_ctp(encoder, data, epochs=100, freeze_encoder=False)

    # 결과 저장
    results = pd.DataFrame([
        {"model": "CTP_frozen",  **result_frozen,
         "notes": "CTP pretrain → frozen encoder fine-tune"},
        {"model": "CTP_e2e",     **result_e2e,
         "notes": "CTP pretrain → end-to-end fine-tune"},
    ])
    save_path = RES / "experiment_log_ctp.csv"
    results.to_csv(save_path, index=False)
    print(f"\n결과 저장: {save_path}")
    return results


if __name__ == "__main__":
    import sys

    # 기본 실행: 모든 가설 순차 실행
    run_hyp = sys.argv[1] if len(sys.argv) > 1 else "all"

    if run_hyp in ("1", "all"):
        run_hypothesis_1()

    if run_hyp in ("2", "all"):
        # 실제 사용 시: GNN probs 를 직접 넘길 것
        # 예) from src.09_boost import get_probs; run_hypothesis_2(probs=get_probs())
        run_hypothesis_2(probs=None)

    if run_hyp in ("3", "all"):
        run_hypothesis_3()

    print("\n" + "=" * 60)
    print("14_new_hypotheses.py 완료")
    print("=" * 60)
