"""
E06_compare.py
전체 실험 결과 비교 — 기준 대비 개선 여부 확인
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent.parent / "results" / "experiments"

BASELINE = {
    "name": "기준 (BWGNN_boost, 30K, 80/20)",
    "n_params": 387329, "n_train": 24000, "param_ratio": 16.1,
    "train_pr": 0.9999, "test_pr": 0.9242, "test_f1": 0.9079,
    "gap": 0.0757, "inductive_pr": 0.6526, "inductive_f1": 0.7515,
}

all_results = [BASELINE]

for fname in ["E01_result.json", "E03_E05_results.json"]:
    p = RES / fname
    if p.exists():
        data = json.load(open(p))
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)

print("="*90)
print("전체 실험 결과 비교")
print("="*90)
print(f"  {'실험':<30} {'params':>8} {'비율':>5} {'Test':>7} {'Gap':>7} {'Inductive':>10}")
print("  " + "-"*80)

for r in all_results:
    name = r.get("name","?")[:30]
    pr   = r.get("param_ratio", r.get("n_params",0)/max(r.get("n_train",1),1))
    te   = r.get("test_pr", 0)
    gap  = r.get("gap", 0)
    ind  = r.get("inductive_pr", 0)
    np_  = r.get("n_params", 0)

    # 기준 대비 화살표
    te_mark  = "↑" if te  > 0.9242 else ("↓" if te  < 0.9200 else "→")
    gap_mark = "↓" if gap < 0.070  else ("↑" if gap > 0.080  else "→")
    ind_mark = "↑" if ind > 0.6526 else ("↓" if ind < 0.62   else "→")

    print(f"  {name:<30} {np_:>8,} {pr:>5.1f} "
          f"{te:>7.4f}{te_mark} {gap:>+7.4f}{gap_mark} {ind:>10.4f}{ind_mark}")

print()
print("  ↑ = 기준 대비 향상   ↓ = 악화   → = 유사")
print()
print("  [결론 요약]")

best_te  = max(all_results, key=lambda r: r.get("test_pr",0))
best_ind = max(all_results, key=lambda r: r.get("inductive_pr",0))
best_gap = min(all_results, key=lambda r: abs(r.get("gap",1)))

print(f"  Test PR-AUC 최고:   {best_te['name']}  {best_te['test_pr']:.4f}")
print(f"  Inductive 최고:     {best_ind['name']}  {best_ind['inductive_pr']:.4f}")
print(f"  Gap 최소:           {best_gap['name']}  {best_gap['gap']:+.4f}")
