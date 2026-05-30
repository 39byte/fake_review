# itda GNN 사기 탐지 — 코드 제출 패키지

## 프로젝트 구조
src/01_eda_sampling.py     : YelpZip EDA + 30K 밀도 중심 샘플링
src/02_features.py         : SBERT 임베딩 + 노드 피처 (386d) 생성
src/03_graph_build.py      : 4종 기본 엣지 헤테로 그래프 구축
src/04_train_baseline.py   : HeteroSAGE / GAT / BWGNN 베이스라인 학습
src/05_train_tgat.py       : TGATLite (Bochner 시간 인코딩) 학습
src/09_boost.py            : R-Sim-R 추가 + Warm Restart 400ep 부스트
src/15_drag_bwgat.py       : DRAG / BWGAT / DRAGWave 학습 (핵심 모델)
src/20_fair_comparison.py  : 400ep 공정 비교 실험
src/27_future_directions.py: DRAGWave_NoRSR / TVF 확장 실험
src/29_performance_boost.py: 3-way Ensemble 최종 성능 (PR-AUC 0.9419)
src/32_ensemble_4way.py    : 4-way Inductive Ensemble (0.7748)
src/39_strict_inductive_train.py: Strict Inductive Training 실험
src/experiments/           : 과적합 완화 실험 (E01~E10c)

## 최종 성능
- Transductive PR-AUC: 0.9419 / Macro-F1: 0.9386 (3-way Ensemble)
- Inductive  PR-AUC: 0.7748 (4-way Ensemble, 배포 환경 시뮬레이션)

## 대시보드
URL: https://itda-gnn-dashboard-dbpfa9uth9hdkmnqyjh4cm.streamlit.app
GitHub: https://github.com/ejongmin/itda-gnn-dashboard

## 실행 순서
python src/01_eda_sampling.py
python src/02_features.py
python src/03_graph_build.py
python src/04_train_baseline.py
python src/09_boost.py
python src/15_drag_bwgat.py
python src/27_future_directions.py  (DRAGWave_NoRSR)
python src/29_performance_boost.py  (3-way Ensemble 최종)
python src/32_ensemble_4way.py      (4-way Inductive Ensemble)

## 재현성
seed=42 고정 (config.py set_seed(42))
CPU 환경에서 완전 재현 보장
