"""
17_network_comparison_viz.py
시각적 설명력 (20점) — 정상 vs 사기 네트워크 구조 비교 시각화

채점 기준: "정상 네트워크와 사기(스팸) 네트워크의 구조적 특징과
            관계성이 직관적으로 명확하게 구분되도록 시각화했는지?"

생성 파일:
  reports/viz_01_network_compare.html   — 정상 vs 사기 나란히 비교
  reports/viz_02_burst_timeline.html    — 버스트 패턴 시계열 비교
  reports/viz_03_similarity_dist.html   — SBERT 유사도 분포
  reports/viz_04_xai_radar.html         — 엣지 기여도 레이더 차트
  reports/viz_05_campaign_stats.html    — 캠페인 클러스터 특성
"""

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
from collections import defaultdict

BASE  = Path(__file__).resolve().parent.parent
GRAPH = BASE / "data" / "graphs"
PROC  = BASE / "data" / "processed"
RES   = BASE / "results"
REP   = BASE / "reports"

print("데이터 로드...")
data = torch.load(GRAPH / "hetero_graph_boost.pt", weights_only=False)
df   = pd.read_parquet(PROC / "df_sampled.parquet")
y    = data["review"].y.numpy()
burst_ei   = data["review","burst","review"].edge_index.numpy()
burst_dt   = data["review","burst","review"].edge_attr.squeeze().numpy()
sim_ei     = data["review","sim","review"].edge_index.numpy()
rur_ei     = data["review","rur","review"].edge_index.numpy()

SPAM_COLOR  = "#E74C3C"
LEGIT_COLOR = "#3498DB"
BG_COLOR    = "#1a1a2e"
CARD_COLOR  = "#16213e"

spam_idx  = set(np.where(y == 1)[0])
legit_idx = set(np.where(y == 0)[0])

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 1] 정상 vs 사기 네트워크 나란히 비교
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 1] 정상 vs 사기 네트워크 비교...")

def build_subgraph(node_set, max_nodes=40):
    """지정 노드 집합에서 R-Sim-R + R-Burst-R 서브그래프 구축"""
    G = nx.Graph()
    nodes_list = list(node_set)[:max_nodes]
    node_set_sample = set(nodes_list)
    for i in range(sim_ei.shape[1]):
        s, d = int(sim_ei[0,i]), int(sim_ei[1,i])
        if s in node_set_sample and d in node_set_sample:
            G.add_edge(s, d, edge_type="sim")
    for i in range(burst_ei.shape[1]):
        s, d = int(burst_ei[0,i]), int(burst_ei[1,i])
        if s in node_set_sample and d in node_set_sample and burst_dt[i] < 24:
            G.add_edge(s, d, edge_type="burst")
    for n in nodes_list:
        G.add_node(n)
    return G

# 사기 클러스터 (캠페인 8번 — 90노드, spam_ratio=0.656)
df_camps = pd.read_csv(RES / "fraud_campaigns.csv")
top_camp  = df_camps[df_camps["is_campaign"]].nlargest(1, "n_nodes").iloc[0]

import ast
try:
    camp_nodes = ast.literal_eval(top_camp["nodes"])
except:
    camp_nodes = list(spam_idx)[:40]

spam_nodes_sample  = [n for n in camp_nodes if n in spam_idx][:25]
legit_nodes_sample = list(legit_idx)[:25]

G_spam  = build_subgraph(set(spam_nodes_sample),  25)
G_legit = build_subgraph(set(legit_nodes_sample), 25)

def graph_to_traces(G, node_color, title_text):
    if G.number_of_nodes() == 0:
        return [], []
    pos = nx.spring_layout(G, seed=42, k=0.8)
    edge_traces = []
    for u, v, d in G.edges(data=True):
        x0,y0 = pos[u]; x1,y1 = pos[v]
        ec = "#E67E22" if d.get("edge_type")=="sim" else "#27AE60"
        edge_traces.append(go.Scatter(x=[x0,x1,None], y=[y0,y1,None],
            mode="lines", line=dict(width=1.5,color=ec), hoverinfo="none", showlegend=False))
    nx_list = list(G.nodes())
    nx_arr  = np.array([pos[n] for n in nx_list])
    node_trace = go.Scatter(
        x=nx_arr[:,0], y=nx_arr[:,1], mode="markers",
        marker=dict(color=node_color, size=12, line=dict(width=1,color="white")),
        text=[f"노드 {n}" for n in nx_list], hoverinfo="text", showlegend=False
    )
    return edge_traces, node_trace

spam_edges,  spam_nodes  = graph_to_traces(G_spam,  SPAM_COLOR,  "사기 클러스터")
legit_edges, legit_nodes = graph_to_traces(G_legit, LEGIT_COLOR, "정상 네트워크")

fig1 = make_subplots(rows=1, cols=2,
    subplot_titles=["🚨 사기(Spam) 리뷰 클러스터", "✅ 정상(Legit) 리뷰 네트워크"],
    horizontal_spacing=0.05)

for et in spam_edges:
    fig1.add_trace(et, row=1, col=1)
if spam_nodes:
    fig1.add_trace(spam_nodes, row=1, col=1)
for et in legit_edges:
    fig1.add_trace(et, row=1, col=2)
if legit_nodes:
    fig1.add_trace(legit_nodes, row=1, col=2)

# 범례
for name, color in [("스팸 노드", SPAM_COLOR), ("정상 노드", LEGIT_COLOR)]:
    fig1.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
        marker=dict(color=color,size=10), name=name, showlegend=True))
for name, color in [("R-Sim-R (복붙 유사성)","#E67E22"), ("R-Burst-R (단기 버스트)","#27AE60")]:
    fig1.add_trace(go.Scatter(x=[None],y=[None],mode="lines",
        line=dict(color=color,width=2), name=name, showlegend=True))

fig1.update_layout(
    title=dict(text="사기 vs 정상 리뷰 네트워크 구조 비교<br>"
               "<sub>사기 클러스터: R-Sim-R(복붙) + R-Burst-R(단기집중) 엣지가 밀집</sub>",
               x=0.5, font=dict(size=16)),
    paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
    font=dict(color="white"), height=550, showlegend=True,
    legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
)
for i in [1,2]:
    fig1.update_xaxes(showgrid=False, showticklabels=False, row=1, col=i)
    fig1.update_yaxes(showgrid=False, showticklabels=False, row=1, col=i)

fig1.write_html(str(REP / "viz_01_network_compare.html"))
print("  저장: viz_01_network_compare.html")

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 2] 버스트 패턴 시계열 비교 (정상 vs 사기)
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 2] 버스트 Δt 분포 비교...")

spam_dt  = burst_dt[np.array([s in spam_idx  for s in burst_ei[0]])]
legit_dt = burst_dt[np.array([s in legit_idx for s in burst_ei[0]])]

fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=["Burst Δt 분포 (0~72h)", "누적 분포 (CDF)"],
    horizontal_spacing=0.12)

bins = np.arange(0, 73, 3)
spam_hist,  _ = np.histogram(spam_dt,  bins=bins, density=True)
legit_hist, _ = np.histogram(legit_dt, bins=bins, density=True)
bin_mids = (bins[:-1] + bins[1:]) / 2

fig2.add_trace(go.Bar(x=bin_mids, y=spam_hist,  name="🚨 사기",  marker_color=SPAM_COLOR,  opacity=0.7), row=1,col=1)
fig2.add_trace(go.Bar(x=bin_mids, y=legit_hist, name="✅ 정상",  marker_color=LEGIT_COLOR, opacity=0.7), row=1,col=1)

spam_sorted  = np.sort(spam_dt);  spam_cdf  = np.arange(1,len(spam_sorted)+1)/len(spam_sorted)
legit_sorted = np.sort(legit_dt); legit_cdf = np.arange(1,len(legit_sorted)+1)/len(legit_sorted)
fig2.add_trace(go.Scatter(x=spam_sorted,  y=spam_cdf,  name="🚨 사기 CDF",  line=dict(color=SPAM_COLOR, width=2)), row=1,col=2)
fig2.add_trace(go.Scatter(x=legit_sorted, y=legit_cdf, name="✅ 정상 CDF",  line=dict(color=LEGIT_COLOR,width=2)), row=1,col=2)
fig2.add_vline(x=12, line_dash="dash", line_color="yellow", annotation_text="12h", row=1, col=2)

fig2.update_layout(
    title=dict(text="Burst 시간 간격(Δt) 분포 — 사기 vs 정상<br>"
               "<sub>두 분포는 전체적으로 유사하나, 초단기(0~6h) 집중 패턴이 사기의 핵심 신호</sub>",
               x=0.5, font=dict(size=16)),
    paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
    font=dict(color="white"), height=480, barmode="overlay",
)
fig2.update_xaxes(title_text="시간 간격 Δt (시간)", row=1, col=1)
fig2.update_xaxes(title_text="시간 간격 Δt (시간)", row=1, col=2)
fig2.update_yaxes(title_text="밀도", row=1, col=1)
fig2.update_yaxes(title_text="누적 확률", row=1, col=2)
fig2.write_html(str(REP / "viz_02_burst_timeline.html"))
print("  저장: viz_02_burst_timeline.html")

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 3] SBERT 유사도 분포 — R-Sim-R 엣지의 효과
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 3] SBERT 유사도 분포...")

emb = torch.load(PROC / "sbert_embeddings.pt", weights_only=True).numpy()

# 스팸/정상 쌍별 코사인 유사도 샘플링
np.random.seed(42)
n_sample = 2000
spam_list  = list(spam_idx)
legit_list = list(legit_idx)

def sample_cosine(idx_list, n):
    pairs = np.random.choice(len(idx_list), (n, 2), replace=True)
    sims  = []
    for i, j in pairs:
        if i != j:
            a, b = emb[idx_list[i]], emb[idx_list[j]]
            sims.append(float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-8)))
    return np.array(sims)

spam_sim  = sample_cosine(spam_list,  n_sample)
legit_sim = sample_cosine(legit_list, n_sample)
cross_sim  = []
for _ in range(n_sample):
    a = emb[np.random.choice(spam_list)]
    b = emb[np.random.choice(legit_list)]
    cross_sim.append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)))
cross_sim = np.array(cross_sim)

fig3 = go.Figure()
for vals, name, color in [
    (spam_sim,  "🚨 스팸-스팸 쌍",   SPAM_COLOR),
    (legit_sim, "✅ 정상-정상 쌍",   LEGIT_COLOR),
    (cross_sim, "🔀 스팸-정상 혼합", "#9B59B6"),
]:
    fig3.add_trace(go.Histogram(x=vals, name=name, nbinsx=50,
        marker_color=color, opacity=0.65, histnorm="probability density"))

fig3.add_vline(x=0.85, line_dash="dash", line_color="yellow", line_width=2,
    annotation_text="R-Sim-R 임계값 (0.85)", annotation_font_color="yellow",
    annotation_position="top right")

fig3.update_layout(
    title=dict(text="리뷰 쌍 간 SBERT 코사인 유사도 분포<br>"
               "<sub>스팸-스팸 쌍이 0.85 이상 구간에서 가장 많음 → R-Sim-R 엣지 설계의 근거</sub>",
               x=0.5, font=dict(size=16)),
    xaxis_title="코사인 유사도",
    yaxis_title="밀도",
    barmode="overlay",
    paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
    font=dict(color="white"), height=480,
)
fig3.write_html(str(REP / "viz_03_similarity_dist.html"))
print("  저장: viz_03_similarity_dist.html")

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 4] XAI 엣지 기여도 — 레이더 + 막대 복합
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 4] XAI 엣지 기여도 시각화...")

attr_path = RES / "xai_edge_attribution.csv"
if attr_path.exists():
    df_attr = pd.read_csv(attr_path)
    fig4 = make_subplots(rows=1, cols=2,
        specs=[[{"type":"polar"}, {"type":"xy"}]],
        subplot_titles=["엣지별 기여도 (레이더)", "ΔPR-AUC (막대)"],
        horizontal_spacing=0.15)

    edge_names = df_attr["edge_type"].tolist()
    contribs   = df_attr["contribution_pct"].abs().tolist()
    deltas     = df_attr["delta_pr_auc"].tolist()

    fig4.add_trace(go.Scatterpolar(
        r=contribs + [contribs[0]], theta=edge_names + [edge_names[0]],
        fill="toself", fillcolor=f"rgba(231,76,60,0.3)",
        line=dict(color=SPAM_COLOR, width=2), name="기여율 (%)"), row=1, col=1)

    colors = [SPAM_COLOR if d > 0 else "#95A5A6" for d in deltas]
    fig4.add_trace(go.Bar(
        x=df_attr["edge_type"], y=df_attr["delta_pr_auc"],
        marker_color=colors,
        text=[f"{v:+.4f}" for v in deltas], textposition="outside",
        name="ΔPR-AUC"), row=1, col=2)

    fig4.update_layout(
        title=dict(text="엣지 타입별 사기 탐지 기여도 (XAI Ablation)<br>"
                   "<sub>R-U-R(사용자 재활용) 20% > R-Burst-R(단기집중) 5.6% > R-T-R > R-Sim-R > R-S-R</sub>",
                   x=0.5, font=dict(size=16)),
        paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
        polar=dict(bgcolor=CARD_COLOR, radialaxis=dict(color="white", gridcolor="#333"),
                   angularaxis=dict(color="white")),
        font=dict(color="white"), height=480, showlegend=False,
    )
    fig4.write_html(str(REP / "viz_04_xai_radar.html"))
    print("  저장: viz_04_xai_radar.html")

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 5] 사기 캠페인 클러스터 통계 — 탐지된 91개 캠페인
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 5] 캠페인 클러스터 통계...")

df_camps = pd.read_csv(RES / "fraud_campaigns.csv")
suspected = df_camps[df_camps["is_campaign"]].copy()

fig5 = make_subplots(rows=2, cols=2,
    subplot_titles=["캠페인 규모 분포 (노드 수)", "캠페인 스팸 비율 분포",
                    "유사도(R-Sim-R) vs 버스트(R-Burst-R) 엣지",
                    "상위 10개 캠페인 상세"],
    vertical_spacing=0.15, horizontal_spacing=0.12)

# 규모 분포
fig5.add_trace(go.Histogram(x=suspected["n_nodes"], nbinsx=20,
    marker_color=SPAM_COLOR, opacity=0.8, name="캠페인 규모"), row=1,col=1)
# 스팸 비율 분포
fig5.add_trace(go.Histogram(x=suspected["spam_ratio"], nbinsx=15,
    marker_color="#E67E22", opacity=0.8, name="스팸 비율"), row=1,col=2)
# 산점도
fig5.add_trace(go.Scatter(
    x=suspected["sim_edges"], y=suspected["burst_edges"],
    mode="markers",
    marker=dict(color=suspected["spam_ratio"], colorscale="Reds", size=10,
                showscale=True, colorbar=dict(title="스팸 비율",len=0.45,y=0.2)),
    text=[f"캠페인{r['campaign_id']}<br>규모:{r['n_nodes']}<br>스팸:{r['spam_ratio']:.0%}"
          for _,r in suspected.iterrows()],
    hoverinfo="text", name="캠페인"), row=2,col=1)
# 상위 10개 막대
top10 = suspected.nlargest(10,"n_nodes")
fig5.add_trace(go.Bar(
    x=[f"캠페인{i}" for i in top10["campaign_id"]],
    y=top10["n_nodes"],
    marker_color=[SPAM_COLOR if r>0.6 else "#E67E22" for r in top10["spam_ratio"]],
    text=[f"스팸 {r:.0%}" for r in top10["spam_ratio"]], textposition="outside",
    name="규모"), row=2,col=2)

fig5.update_layout(
    title=dict(text=f"탐지된 사기 캠페인 분석 — {len(suspected)}개 의심 캠페인<br>"
               "<sub>스팸 비율 ≥50% 클러스터 = 조직적 어뷰징 캠페인으로 분류</sub>",
               x=0.5, font=dict(size=16)),
    paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
    font=dict(color="white"), height=700, showlegend=False,
)
fig5.update_xaxes(gridcolor="#333"); fig5.update_yaxes(gridcolor="#333")
fig5.write_html(str(REP / "viz_05_campaign_stats.html"))
print("  저장: viz_05_campaign_stats.html")

# ─────────────────────────────────────────────────────────────────────────────
# [VIZ 6] 모델 성능 비교 — 발전 과정 스토리텔링
# ─────────────────────────────────────────────────────────────────────────────
print("[VIZ 6] 모델 발전 과정 시각화...")

df_log = pd.read_csv(RES / "experiment_log.csv").sort_values("pr_auc")

# 카테고리 분류
def get_cat(m):
    if "DRAGWave" in m: return "DRAGWave (본 연구)"
    if "DRAG" in m: return "DRAG"
    if "BWGAT" in m: return "BWGAT"
    if "TGATLiteV2" in m: return "TGATLiteV2 (본 연구)"
    if "BWGNN" in m or "BWGNN" in m: return "BWGNN 계열"
    if "SAGEConv" in m or "SAGE" in m: return "정적 GNN"
    return "기타"

cat_colors = {
    "DRAGWave (본 연구)": "#E74C3C",
    "DRAG": "#E67E22",
    "BWGAT": "#F39C12",
    "TGATLiteV2 (본 연구)": "#3498DB",
    "BWGNN 계열": "#27AE60",
    "정적 GNN": "#95A5A6",
    "기타": "#BDC3C7",
}

fig6 = go.Figure()
for cat, color in cat_colors.items():
    mask = df_log["model"].apply(lambda m: get_cat(m)==cat)
    sub  = df_log[mask]
    if len(sub)==0: continue
    fig6.add_trace(go.Scatter(
        x=sub["macro_f1"], y=sub["pr_auc"],
        mode="markers+text",
        marker=dict(color=color, size=14, line=dict(width=1,color="white")),
        text=sub["model"].apply(lambda m: m.replace("_400ep","★").replace("HeteroBWGNN","BWGNN")),
        textposition="top center", textfont=dict(size=9),
        name=cat, hoverinfo="text",
        hovertext=[f"{r['model']}<br>PR-AUC={r['pr_auc']:.4f}<br>F1={r['macro_f1']:.4f}"
                   for _,r in sub.iterrows()]
    ))

# 목표선
fig6.add_hline(y=0.90, line_dash="dot", line_color="yellow",
    annotation_text="PR-AUC 0.90 기준선", annotation_position="left")

fig6.update_layout(
    title=dict(text="모델 성능 비교 — PR-AUC vs Macro-F1<br>"
               "<sub>★ = 400 epoch 완전 수렴 버전 / 우상단이 최고 성능</sub>",
               x=0.5, font=dict(size=16)),
    xaxis_title="Macro-F1",
    yaxis_title="PR-AUC",
    paper_bgcolor=BG_COLOR, plot_bgcolor=CARD_COLOR,
    font=dict(color="white"), height=550,
    legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1),
    xaxis=dict(range=[0.68,1.0], gridcolor="#333"),
    yaxis=dict(range=[0.68,1.0], gridcolor="#333"),
)
fig6.write_html(str(REP / "viz_06_model_comparison.html"))
print("  저장: viz_06_model_comparison.html")

print("\n✅ 시각화 6종 완료")
print("  → reports/viz_01~06_*.html")
