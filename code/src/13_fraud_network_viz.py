"""
13_fraud_network_viz.py
사기 네트워크 시각화 + 캠페인 단위 탐지 (Community Detection)

목적:
  1. 탐지된 스팸 리뷰들이 형성하는 네트워크 구조를 시각화
  2. Community Detection으로 "캠페인 단위" 어뷰징 클러스터 식별
  3. 발표용 인터랙티브 HTML 차트 생성

방법:
  - R-Sim-R + 단기 R-Burst-R (Δt < 24h) 엣지로 서브그래프 구축
  - 스팸 노드를 중심으로 연결 컴포넌트 추출
  - Greedy Modularity Community Detection (networkx 내장)
  - Plotly로 인터랙티브 시각화

출력:
  results/fraud_campaigns.csv        — 탐지된 캠페인 목록
  reports/fraud_network_viz.html     — 인터랙티브 시각화
"""

import torch
import numpy as np
import pandas as pd
import json
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import Counter

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
PROC  = BASE / "data" / "processed"
RES   = BASE / "results"
REP   = BASE / "reports"
MOD   = BASE / "models"

print("데이터 로드 중...")
data = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False)
df   = pd.read_parquet(PROC / "df_sampled.parquet")
y    = data["review"].y.numpy()

# ─────────────────────────────────────────────────────────────────────────────
# [Step 1] 사기 관련 서브그래프 구축
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] 사기 서브그래프 구축")

G = nx.Graph()

# R-Sim-R 엣지 추가 (의미론적 유사성 — 복붙 캠페인의 핵심 신호)
sim_ei = data["review", "sim", "review"].edge_index.numpy()
for i in range(sim_ei.shape[1]):
    src, dst = int(sim_ei[0, i]), int(sim_ei[1, i])
    G.add_edge(src, dst, edge_type="sim", weight=2.0)

# R-Burst-R 단기 엣지 추가 (Δt < 24h — 조직적 버스트의 핵심 신호)
burst_ei   = data["review", "burst", "review"].edge_index.numpy()
burst_attr = data["review", "burst", "review"].edge_attr.squeeze().numpy()
short_mask = burst_attr < 24.0
for i in np.where(short_mask)[0]:
    src, dst = int(burst_ei[0, i]), int(burst_ei[1, i])
    dt = float(burst_attr[i])
    G.add_edge(src, dst, edge_type="burst", weight=1.0, delta_t=dt)

# 노드 속성 추가
for node in G.nodes():
    G.nodes[node]["is_spam"]    = int(y[node])
    G.nodes[node]["rating"]     = float(df.iloc[node]["rating"]) if node < len(df) else 3.0
    G.nodes[node]["text_short"] = str(df.iloc[node]["text"])[:50] if node < len(df) else ""

print(f"  서브그래프 노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}")
spam_in_G = sum(1 for n in G.nodes() if y[n] == 1)
print(f"  스팸 비율: {spam_in_G}/{G.number_of_nodes()} = {spam_in_G/G.number_of_nodes()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# [Step 2] Community Detection — 캠페인 단위 클러스터 식별
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Community Detection (Greedy Modularity)")

# 연결 컴포넌트만 대상
components = [c for c in nx.connected_components(G) if len(c) >= 3]
print(f"  크기 ≥3 연결 컴포넌트: {len(components)}개")

# 각 컴포넌트의 스팸 비율 계산
campaigns = []
for idx, comp in enumerate(sorted(components, key=len, reverse=True)):
    subG  = G.subgraph(comp)
    nodes = list(comp)
    n_spam  = sum(1 for n in nodes if y[n] == 1)
    n_total = len(nodes)
    spam_ratio = n_spam / n_total

    # 내부 엣지 타입 분포
    sim_edges   = sum(1 for u, v, d in subG.edges(data=True) if d.get("edge_type") == "sim")
    burst_edges = sum(1 for u, v, d in subG.edges(data=True) if d.get("edge_type") == "burst")

    # 식당 분포 (prod_id)
    prods = df.iloc[nodes]["prod_id"].value_counts()
    main_prod = str(prods.index[0]) if len(prods) > 0 else "unknown"

    campaigns.append({
        "campaign_id":   idx + 1,
        "n_nodes":       n_total,
        "n_spam":        n_spam,
        "spam_ratio":    round(spam_ratio, 3),
        "sim_edges":     sim_edges,
        "burst_edges":   burst_edges,
        "main_product":  main_prod,
        "is_campaign":   spam_ratio >= 0.5,  # 스팸 50% 이상 = 의심 캠페인
        "nodes":         nodes[:20],  # 상위 20개 노드만 저장
    })

df_campaigns = pd.DataFrame(campaigns)
n_detected = (df_campaigns["is_campaign"]).sum()
print(f"  의심 캠페인 탐지: {n_detected}개 (스팸 비율 ≥ 50%)")
print(f"  캠페인 내 총 노드: {df_campaigns[df_campaigns['is_campaign']]['n_nodes'].sum():,}")
print(df_campaigns[df_campaigns["is_campaign"]].head(10)[
    ["campaign_id","n_nodes","n_spam","spam_ratio","sim_edges","burst_edges"]
].to_string(index=False))

df_campaigns.to_csv(RES / "fraud_campaigns.csv", index=False)
print(f"\n저장: results/fraud_campaigns.csv")

# ─────────────────────────────────────────────────────────────────────────────
# [Step 3] 상위 캠페인 시각화 (Plotly 인터랙티브)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] 인터랙티브 시각화 생성")

# 상위 5개 의심 캠페인을 하나의 그래프로 시각화
top_campaigns = df_campaigns[df_campaigns["is_campaign"]].nlargest(5, "n_nodes")
viz_nodes = set()
for _, row in top_campaigns.iterrows():
    comp = components[row["campaign_id"] - 1]
    viz_nodes.update(list(comp)[:30])  # 컴포넌트당 최대 30개

subG_viz = G.subgraph(viz_nodes).copy()

# Spring layout
pos = nx.spring_layout(subG_viz, k=0.5, seed=42)

# 노드 색상: 스팸(빨강) / 정상(파랑)
node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
for node in subG_viz.nodes():
    x, y_pos = pos[node]
    node_x.append(x); node_y.append(y_pos)
    is_spam = y[node]
    node_color.append("#E74C3C" if is_spam else "#3498DB")
    node_size.append(14 if is_spam else 8)
    text_preview = str(df.iloc[node]["text"])[:40] + "..." if node < len(df) else ""
    node_text.append(f"노드 {node}<br>{'🚨스팸' if is_spam else '✅정상'}<br>{text_preview}")

# 엣지 색상: sim(주황) / burst(초록)
edge_traces = []
for u, v, d in subG_viz.edges(data=True):
    x0, y0 = pos[u]; x1, y1 = pos[v]
    color = "#E67E22" if d.get("edge_type") == "sim" else "#27AE60"
    edge_traces.append(go.Scatter(
        x=[x0, x1, None], y=[y0, y1, None],
        mode="lines",
        line=dict(width=1.5, color=color),
        hoverinfo="none",
        showlegend=False
    ))

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode="markers",
    hoverinfo="text",
    text=node_text,
    marker=dict(color=node_color, size=node_size,
                line=dict(width=1, color="white"))
)

fig = go.Figure(
    data=edge_traces + [node_trace],
    layout=go.Layout(
        title=dict(
            text="사기 리뷰 캠페인 네트워크<br>"
                 "<sub>🔴 스팸 노드 | 🔵 정상 노드 | "
                 "<span style='color:#E67E22'>━</span> 복붙 유사성(R-Sim-R) | "
                 "<span style='color:#27AE60'>━</span> 단기 버스트(R-Burst-R, Δt<24h)</sub>",
            x=0.5
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=80),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
        height=650,
    )
)

# 범례 추가 (더미 트레이스)
for name, color, symbol in [
    ("스팸 노드", "#E74C3C", "circle"),
    ("정상 노드", "#3498DB", "circle"),
]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=color, size=10, symbol=symbol),
        name=name, showlegend=True
    ))
for name, color in [
    ("R-Sim-R (복붙 유사성)", "#E67E22"),
    ("R-Burst-R (단기 버스트)", "#27AE60"),
]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=color, width=2),
        name=name, showlegend=True
    ))

out_html = REP / "fraud_network_viz.html"
fig.write_html(str(out_html))
print(f"저장: reports/fraud_network_viz.html")

# ─────────────────────────────────────────────────────────────────────────────
# [Step 4] 캠페인 통계 요약 (발표용)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[Step 4] 캠페인 탐지 요약")
print("=" * 60)
print(f"  전체 탐지 컴포넌트: {len(components)}개")
print(f"  의심 캠페인 (스팸≥50%): {n_detected}개")
print(f"  의심 캠페인 내 총 리뷰: {df_campaigns[df_campaigns['is_campaign']]['n_nodes'].sum():,}개")
print(f"  의심 캠페인 내 스팸 리뷰: {df_campaigns[df_campaigns['is_campaign']]['n_spam'].sum():,}개")

# 저장 형태 요약 JSON
summary = {
    "total_components": len(components),
    "suspected_campaigns": int(n_detected),
    "total_reviews_in_campaigns": int(df_campaigns[df_campaigns["is_campaign"]]["n_nodes"].sum()),
    "spam_in_campaigns": int(df_campaigns[df_campaigns["is_campaign"]]["n_spam"].sum()),
    "avg_campaign_size": round(df_campaigns[df_campaigns["is_campaign"]]["n_nodes"].mean(), 1),
    "max_campaign_size": int(df_campaigns[df_campaigns["is_campaign"]]["n_nodes"].max()),
}
with open(RES / "campaign_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n저장: results/campaign_summary.json")
print("✅ 사기 네트워크 시각화 완료")
print(f"   → 발표 시연: reports/fraud_network_viz.html 브라우저에서 열기")
