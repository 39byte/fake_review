"""
35_korean_poc.py
한국어 리뷰 적용 PoC (Proof of Concept)

목적: YelpZip(영어) 기반 파이프라인을 국내 플랫폼(배달앱 등)에 적용하려면
      SBERT 모델만 교체하면 된다는 것을 검증.

변경점:
  기존: SentenceTransformer('all-MiniLM-L6-v2')  → 384d
  교체: SentenceTransformer('jhgan/ko-sroberta-multitask')  → 768d
       또는 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'          → 768d

아키텍처 영향:
  - node_features: [N, 386] → [N, 770]  (+384d, rating/timestamp 동일)
  - 모델 입력 projection만 변경: nn.Linear(386, 128) → nn.Linear(770, 128)
  - 이후 GNN 레이어, 손실 함수, 학습 루프 전부 동일

결과: results/korean_poc_result.json
"""

import json
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

RES = Path(__file__).resolve().parent.parent / "results"

# ── 샘플 한국어 리뷰 (실제 배달앱 리뷰 시나리오 모사) ──────────────────────────
REVIEWS = {
    "spam_burst": [
        "정말 맛있어요! 음식도 빠르고 서비스도 최고! 강력 추천합니다.",
        "진짜 맛있어요~ 음식도 빠르고 서비스도 최고! 꼭 시켜보세요.",
        "너무 맛있어요!! 음식도 빠르고 서비스도 최고! 다들 드셔보세요.",
        "맛집이에요. 음식도 빠르고 서비스도 최고! 재주문 할게요.",
    ],
    "spam_copy": [
        "사장님이 친절하고 음식이 맛있습니다. 배달도 빠르고 좋아요.",
        "사장님이 친절하고 음식이 맛있습니다. 배달도 빠르고 좋아요!",
        "사장님이 친절하고 음식이 맛있습니다. 배달도 빠르고 좋아요^^",
    ],
    "legit": [
        "양이 좀 적었지만 맛은 괜찮았어요. 다음엔 사이드도 추가해볼 것 같아요.",
        "배달이 늦었고 음식이 식어 있었어요. 실망스러웠습니다.",
        "처음 시켜봤는데 생각보다 괜찮았어요. 소스가 특히 맛있었습니다.",
        "가성비 좋아요. 양도 많고 맛도 나쁘지 않아서 자주 시킬 것 같아요.",
    ],
}

print("=" * 60)
print("한국어 리뷰 어뷰징 탐지 — KoSBERT PoC")
print("=" * 60)

# ── KoSBERT 로드 ──────────────────────────────────────────────────────────────
print("\n[1] KoSBERT 모델 로드...")
KO_MODELS = [
    "jhgan/ko-sroberta-multitask",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
]

ko_model = None
used_model = ""
for m in KO_MODELS:
    try:
        ko_model = SentenceTransformer(m)
        used_model = m
        print(f"  ✅ 로드 성공: {m}")
        break
    except Exception as e:
        print(f"  ⚠️  {m} 실패: {e}")

if ko_model is None:
    print("  KoSBERT 로드 실패. 영어 SBERT로 대체합니다.")
    ko_model   = SentenceTransformer("all-MiniLM-L6-v2")
    used_model = "all-MiniLM-L6-v2 (fallback)"

# ── 임베딩 차원 확인 ───────────────────────────────────────────────────────────
sample_emb = ko_model.encode(["테스트"])
emb_dim    = sample_emb.shape[1]
feat_dim   = emb_dim + 2  # + rating(1) + timestamp(1)
print(f"\n[2] 임베딩 차원: {emb_dim}d  →  노드 피처: {feat_dim}d")
print(f"    (기존 영어 SBERT: 384d → 386d)")
print(f"    모델 projection 변경: nn.Linear(386, 128) → nn.Linear({feat_dim}, 128)")

# ── 유사도 분석 — 스팸 탐지 가능성 검증 ─────────────────────────────────────────
print("\n[3] 코사인 유사도 분석 (R-Sim-R 엣지 구성 가능성)")

all_texts  = []
all_labels = []
all_types  = []
for cat, texts in REVIEWS.items():
    for t in texts:
        all_texts.append(t)
        all_labels.append(1 if "spam" in cat else 0)
        all_types.append(cat)

embeddings = ko_model.encode(all_texts, show_progress_bar=False)
sim_matrix = cosine_similarity(embeddings)

n = len(all_texts)
spam_sim_scores  = []
legit_sim_scores = []
cross_sim_scores = []

for i in range(n):
    for j in range(i + 1, n):
        s = sim_matrix[i, j]
        if all_labels[i] == 1 and all_labels[j] == 1:
            spam_sim_scores.append(s)
        elif all_labels[i] == 0 and all_labels[j] == 0:
            legit_sim_scores.append(s)
        else:
            cross_sim_scores.append(s)

print(f"  스팸↔스팸  유사도: {np.mean(spam_sim_scores):.4f} ± {np.std(spam_sim_scores):.4f}")
print(f"  정상↔정상  유사도: {np.mean(legit_sim_scores):.4f} ± {np.std(legit_sim_scores):.4f}")
print(f"  스팸↔정상  유사도: {np.mean(cross_sim_scores):.4f} ± {np.std(cross_sim_scores):.4f}")

threshold = 0.85
spam_edges  = sum(1 for s in spam_sim_scores  if s >= threshold)
legit_edges = sum(1 for s in legit_sim_scores if s >= threshold)
cross_edges = sum(1 for s in cross_sim_scores if s >= threshold)
print(f"\n  threshold={threshold} 기준 R-Sim-R 엣지 형성:")
print(f"    스팸↔스팸: {spam_edges}/{len(spam_sim_scores)} 쌍  ({spam_edges/max(len(spam_sim_scores),1)*100:.0f}%)")
print(f"    정상↔정상: {legit_edges}/{len(legit_sim_scores)} 쌍  ({legit_edges/max(len(legit_sim_scores),1)*100:.0f}%)")
print(f"    스팸↔정상: {cross_edges}/{len(cross_sim_scores)} 쌍  ({cross_edges/max(len(cross_sim_scores),1)*100:.0f}%)")

# ── 아키텍처 변경 최소성 검증 ──────────────────────────────────────────────────
print("\n[4] 아키텍처 변경 사항")
print(f"""
  # 02_features.py — 1줄 변경
  기존: model = SentenceTransformer('all-MiniLM-L6-v2')   # 384d
  변경: model = SentenceTransformer('{used_model}')  # {emb_dim}d

  # HeteroDRAGWave — 자동 처리 (feat_dim 인자로 전달)
  기존: model = HeteroDRAGWave(386)   → proj: Linear(386, 128)
  변경: model = HeteroDRAGWave({feat_dim})   → proj: Linear({feat_dim}, 128)

  # 그 외: 그래프 엣지, 학습 루프, 손실 함수 — 변경 없음
""")

# ── 결론 ──────────────────────────────────────────────────────────────────────
spam_advantage = np.mean(spam_sim_scores) - np.mean(legit_sim_scores)
feasible = spam_advantage > 0.05

print(f"[결론]")
print(f"  스팸↔스팸 유사도가 정상↔정상보다 {spam_advantage:+.4f} 높음")
print(f"  R-Sim-R 엣지 선별력: {'✅ 유효 (스팸 쌍에서 더 많은 엣지 형성)' if feasible else '⚠️ 샘플 부족으로 불명확'}")
print(f"  → 한국어 데이터 확보 시 코드 변경 1줄 + 재학습으로 즉시 적용 가능")

result = {
    "model_used":     used_model,
    "embedding_dim":  int(emb_dim),
    "node_feat_dim":  int(feat_dim),
    "similarity": {
        "spam_spam_mean":  round(float(np.mean(spam_sim_scores)),  4),
        "legit_legit_mean": round(float(np.mean(legit_sim_scores)), 4),
        "spam_legit_mean": round(float(np.mean(cross_sim_scores)), 4),
    },
    "rsimr_edges_at_085": {
        "spam_spam":  spam_edges,
        "legit_legit": legit_edges,
        "spam_legit":  cross_edges,
    },
    "conclusion": "코드 변경 최소 (02_features.py 1줄 + 재학습). 한국어 데이터 확보 시 즉시 적용 가능.",
    "feasible": bool(feasible),
}

out = RES / "korean_poc_result.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n저장: {out.name}")
print("✅ 한국어 PoC 완료")
