"""
itda GNN 사기 탐지 대시보드
학술대회 최적화 디자인 — F패턴 레이아웃 · 세마틱 색상 · 벤치마크 라인
"""

import json, numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent
DATA = BASE / "data"
RPT  = BASE / "reports"

# ── 색상 체계 (연구결과 기반 세마틱 팔레트) ───────────────────────────────────
C = {
    "bg"     : "#0C0E14",   # Soft Black (순검정 X)
    "surf"   : "#161B25",   # Card surface
    "surf2"  : "#1E2535",   # Elevated card
    "border" : "#2A3347",   # Subtle border
    # Semantic
    "primary": "#4F9CF9",   # Blue — 신뢰 / 정보
    "success": "#10B981",   # Green — 성공 / 정상
    "warn"   : "#F59E0B",   # Amber — 경고
    "danger" : "#EF4444",   # Red — 위험 / 스팸
    "purple" : "#8B5CF6",   # Violet — XAI / 분석
    "teal"   : "#14B8A6",   # Teal — 강조
    # Text
    "text"   : "#E8EDF5",
    "text2"  : "#94A3B8",
    "muted"  : "#4B5563",
}

CHART = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="Inter, Noto Sans KR, sans-serif", color=C["text"], size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    hoverlabel=dict(bgcolor=C["surf2"], font_color=C["text"], bordercolor=C["border"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
)

AXIS = dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
            linecolor=C["border"], tickcolor=C["border"], tickfont_color=C["text2"])

st.set_page_config(
    page_title="itda GNN | 리뷰 어뷰징 탐지",
    page_icon="🛡️", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;600;700&display=swap');

html,body,[class*="css"]{{font-family:'Inter','Noto Sans KR',sans-serif;}}
.main,.block-container{{background:{C['bg']}!important; padding-top:0!important;}}

/* ── 상단 헤더 바 ── */
.top-bar{{
  background:{C['surf']}; border-bottom:1px solid {C['border']};
  padding:12px 24px; display:flex; align-items:center; justify-content:space-between;
  margin:-1rem -1rem 1.5rem -1rem;
}}
.top-logo{{font-size:18px;font-weight:700;color:{C['primary']};letter-spacing:-0.3px;}}
.top-sub{{font-size:12px;color:{C['text2']};margin-top:2px;}}
.top-stat{{text-align:center;}}
.top-stat-val{{font-size:20px;font-weight:700;}}
.top-stat-lbl{{font-size:10px;color:{C['text2']};text-transform:uppercase;letter-spacing:.6px;}}

/* ── KPI 카드 ── */
.kpi{{
  background:{C['surf']}; border:1px solid {C['border']};
  border-radius:10px; padding:16px 18px; position:relative; overflow:hidden;
}}
.kpi::before{{
  content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background:var(--accent);
}}
.kpi-val{{font-size:30px;font-weight:700;line-height:1.1;margin:4px 0 2px;}}
.kpi-lbl{{font-size:11px;font-weight:600;color:{C['text2']};
           text-transform:uppercase;letter-spacing:.7px;}}
.kpi-sub{{font-size:11px;color:{C['muted']};margin-top:4px;}}
.kpi-delta-up{{color:{C['success']};font-size:12px;font-weight:600;}}
.kpi-delta-dn{{color:{C['danger']};font-size:12px;font-weight:600;}}

/* ── 섹션 타이틀 ── */
.sec{{
  font-size:13px;font-weight:700;color:{C['text']};
  border-left:3px solid var(--accent,{C['primary']});
  padding-left:10px; margin:20px 0 12px; letter-spacing:.2px;
}}

/* ── 인사이트 박스 ── */
.insight{{
  background:{C['surf2']};border-left:3px solid var(--accent,{C['primary']});
  border-radius:0 8px 8px 0; padding:12px 14px; margin:8px 0;
  font-size:12.5px; line-height:1.7; color:{C['text']};
}}

/* ── 배지 ── */
.badge{{
  display:inline-block; padding:2px 9px; border-radius:99px;
  font-size:11px; font-weight:600; line-height:1.8;
}}
.badge-blue  {{background:{C['primary']}22;color:{C['primary']};border:1px solid {C['primary']}44;}}
.badge-green {{background:{C['success']}22;color:{C['success']};border:1px solid {C['success']}44;}}
.badge-red   {{background:{C['danger']}22; color:{C['danger']}; border:1px solid {C['danger']}44;}}
.badge-amber {{background:{C['warn']}22;   color:{C['warn']};   border:1px solid {C['warn']}44;}}

/* ── 리뷰 박스 ── */
.review-box{{
  background:{C['surf2']};border:1px solid {C['border']};border-left:3px solid {C['primary']};
  border-radius:0 8px 8px 0;padding:14px 16px;
  font-size:13px;line-height:1.8;color:{C['text']};
}}

/* 사이드바 */
section[data-testid="stSidebar"]>div:first-child{{
  background:linear-gradient(180deg,{C['surf']} 0%,{C['bg']} 100%)!important;
  border-right:1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"]{{font-size:13px;font-weight:600;color:{C['text2']};}}
.stTabs [aria-selected="true"]{{color:{C['primary']}!important;border-bottom-color:{C['primary']}!important;}}
div[data-testid="metric-container"]{{background:{C['surf']};border-radius:8px;padding:12px;}}
</style>
""", unsafe_allow_html=True)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
@st.cache_data
def load():
    pr   = np.load(DATA/"all_probs.npy")
    lbl  = np.load(DATA/"all_labels.npy")
    tm   = np.load(DATA/"test_mask.npy")
    bei  = np.load(DATA/"burst_ei.npy")
    bdt  = np.load(DATA/"burst_dt.npy")
    sei  = np.load(DATA/"sim_ei.npy")
    rur  = np.load(DATA/"rur_ei.npy")
    df   = pd.read_parquet(DATA/"df_sampled.parquet")
    return pr, lbl, tm, bei, bdt, sei, rur, df

@st.cache_data
def load_res():
    log  = pd.read_csv(DATA/"experiment_log.csv")
    logi = pd.read_csv(DATA/"experiment_log_inductive.csv")
    camps= pd.read_csv(DATA/"fraud_campaigns.csv")
    if "actual_spam_ratio" in camps.columns:
        camps = camps.rename(columns={"actual_spam_ratio":"spam_ratio"})
    if "detected" in camps.columns:
        camps = camps.rename(columns={"detected":"is_campaign"})
    attr = pd.read_csv(DATA/"xai_edge_attribution.csv")
    return log, logi, camps, attr

def jload(n):
    p = DATA/n; return json.load(open(p,"r",encoding="utf-8")) if p.exists() else {}

probs,labels,tmask,bei,bdt,sei,rur,df = load()
log,logi,camps,attr = load_res()
eda  = jload("eda_summary.json")
csum = jload("campaign_summary.json")
kf   = jload("inductive_kfold_result.json")
e4   = jload("ensemble_4way_result.json")


# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding:20px 0 16px;text-align:center;'>
      <div style='font-size:36px;margin-bottom:6px;'>🛡️</div>
      <div style='font-size:15px;font-weight:700;color:{C["primary"]};'>itda GNN</div>
      <div style='font-size:11px;color:{C["text2"]};margin-top:3px;'>리뷰 어뷰징 탐지 시스템</div>
    </div>
    <div style='height:1px;background:{C["border"]};margin:0 0 16px;'></div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  개요 및 성과",
        "🎯  실시간 탐지 데모",
        "🕸  캠페인 분석",
        "📊  모델 성능 비교",
        "🔬  XAI 탐지 근거",
    ], label_visibility="collapsed")

    st.markdown(f"<div style='height:1px;background:{C['border']};margin:16px 0;'></div>", unsafe_allow_html=True)

    # 핵심 지표 사이드바
    for lbl_s, val_s, clr_s in [
        ("PR-AUC",   "0.9419", C["primary"]),
        ("Macro F1", "0.9386", C["success"]),
        ("인덕티브", "0.7748", C["purple"]),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:8px 4px;border-bottom:1px solid {C["border"]}22;'>
          <span style='font-size:11px;color:{C["text2"]};'>{lbl_s}</span>
          <span style='font-size:14px;font-weight:700;color:{clr_s};'>{val_s}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:10px;color:{C['muted']};text-align:center;margin-top:16px;'>YelpZip · 30K 노드 · 963K 엣지</div>",
                unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 🏠  개요 및 성과
# ═════════════════════════════════════════════════════════════════════════════
if "개요" in page:
    # 상단 헤더
    st.markdown(f"""
    <div class='top-bar'>
      <div>
        <div class='top-logo'>헤테로 시간 인식 GNN 기반 조직적 리뷰 어뷰징 탐지</div>
        <div class='top-sub'>YelpZip 608K 리뷰 · 5종 도메인 특화 엣지 · DRAGWave 융합 아키텍처 · itda 학술대회 본선</div>
      </div>
      <div style='display:flex;gap:28px;'>
        <div class='top-stat'><div class='top-stat-val' style='color:{C["primary"]};'>0.9419</div><div class='top-stat-lbl'>PR-AUC</div></div>
        <div class='top-stat'><div class='top-stat-val' style='color:{C["success"]};'>0.9386</div><div class='top-stat-lbl'>Macro F1</div></div>
        <div class='top-stat'><div class='top-stat-val' style='color:{C["purple"]};'>0.7748</div><div class='top-stat-lbl'>인덕티브</div></div>
        <div class='top-stat'><div class='top-stat-val' style='color:{C["warn"]};'>91개</div><div class='top-stat-lbl'>캠페인 탐지</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # F-패턴: KPI 5개 (상단 전체폭)
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (c1, "PR-AUC",      "0.9419", "+9.4%p vs 기준",  C["primary"],  "3-way Ensemble"),
        (c2, "Macro F1",    "0.9386", "+9.9%p vs 기준",  C["success"],  "3-way Ensemble"),
        (c3, "인덕티브",    "0.7748", "+18.7%p vs 기준", C["purple"],   "4-way Ensemble"),
        (c4, "캠페인 탐지", "91개",   "1,584개 컴포넌트",C["warn"],     "R-Sim-R 클러스터"),
        (c5, "추론 속도",   "5.6ms",  "초당 179개 처리", C["teal"],     "CPU 기준"),
    ]
    for col, lbl_k, val_k, sub_k, clr_k, tip_k in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi' style='--accent:{clr_k};'>
              <div class='kpi-lbl'>{lbl_k}</div>
              <div class='kpi-val' style='color:{clr_k};'>{val_k}</div>
              <div class='kpi-delta-up'>↑ {sub_k}</div>
              <div class='kpi-sub'>{tip_k}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 중단: 좌(2/3) — 엣지 분석 차트 · 우(1/3) — 데이터 분포
    col_l, col_r = st.columns([2, 1], gap="large")

    with col_l:
        st.markdown(f"<div class='sec' style='--accent:{C[\"primary\"]};'>📐 5종 엣지 설계 — 기여도 & 규모</div>",
                    unsafe_allow_html=True)

        if not attr.empty:
            edge_colors = [C["purple"],C["teal"],C["primary"],C["warn"],C["muted"]]
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(
                name="엣지 수", x=attr["edge_type"], y=attr["n_edges"],
                marker_color=edge_colors, opacity=0.75,
                text=attr["n_edges"].apply(lambda v:f"{v:,}"),
                textposition="outside", textfont_size=10,
            ), secondary_y=False)
            fig.add_trace(go.Scatter(
                name="XAI 기여도(%)", x=attr["edge_type"], y=attr["contribution_pct"],
                mode="lines+markers+text",
                line=dict(color=C["danger"], width=2.5),
                marker=dict(size=10, color=C["danger"],
                            line=dict(color=C["bg"], width=2)),
                text=[f"{v:.1f}%" for v in attr["contribution_pct"]],
                textposition="top center", textfont=dict(size=10, color=C["danger"]),
            ), secondary_y=True)
            fig.update_xaxes(**AXIS)
            fig.update_yaxes(title_text="엣지 수", secondary_y=False, **AXIS)
            fig.update_yaxes(title_text="XAI 기여도 (%)", secondary_y=True,
                             showgrid=False, tickcolor=C["border"], tickfont_color=C["text2"])
            fig.update_layout(**CHART, height=280, legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)

        # 모델 발전 4단계 (가로 타임라인)
        st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>🚀 모델 발전 — 4단계 진화 경로</div>",
                    unsafe_allow_html=True)
        stages = [
            ("Stage 1","정적 GNN",    "HeteroSAGE\nBWGNN",    "0.830", C["muted"]),
            ("Stage 2","시간 인식",   "TGATLite\nBochner",    "0.738", C["primary"]),
            ("Stage 3","Dual Memory", "TGATLiteV2\nR-Sim-R",  "0.924", C["teal"]),
            ("Stage 4★","DRAGWave",   "DRAG×BWGAT\n3-way",   "0.942", C["success"]),
        ]
        sc1,sc2,sc3,sc4 = st.columns(4)
        for col_s,(stg,title,sub,pr,clr) in zip([sc1,sc2,sc3,sc4],stages):
            with col_s:
                st.markdown(f"""
                <div style='background:{C["surf"]};border:1px solid {C["border"]};
                            border-top:3px solid {clr};border-radius:8px;
                            padding:14px 12px;text-align:center;'>
                  <div style='font-size:10px;font-weight:700;color:{clr};
                               text-transform:uppercase;letter-spacing:.5px;'>{stg}</div>
                  <div style='font-size:13px;font-weight:700;color:{C["text"]};
                               margin:5px 0 3px;'>{title}</div>
                  <div style='font-size:10px;color:{C["text2"]};white-space:pre-line;
                               margin-bottom:8px;'>{sub}</div>
                  <div style='font-size:18px;font-weight:700;color:{clr};'>PR {pr}</div>
                </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"<div class='sec' style='--accent:{C[\"warn\"]};'>📊 데이터 분포</div>",
                    unsafe_allow_html=True)
        # 클래스 비율
        fig_sp = go.Figure(go.Pie(
            labels=["정상 (86.8%)", "스팸 (13.2%)"],
            values=[86.8, 13.2], hole=0.62,
            marker_colors=[C["primary"], C["danger"]],
            textfont_size=11, showlegend=True,
        ))
        fig_sp.update_layout(**CHART, height=170,
                             legend=dict(orientation="h", y=-0.1),
                             annotations=[dict(text="클래스", x=0.5, y=0.5,
                                              showarrow=False, font_size=12,
                                              font_color=C["text"])])
        st.plotly_chart(fig_sp, use_container_width=True)

        # 별점 분포
        rd = eda.get("rating_분포",{"1.0":930,"2.0":1887,"3.0":5016,"4.0":11517,"5.0":10650})
        fig_r = go.Figure(go.Bar(
            x=list(rd.keys()), y=list(rd.values()),
            marker_color=[C["danger"],C["warn"],C["muted"],C["primary"],C["success"]],
            opacity=0.85, text=list(rd.values()),
            textposition="outside", textfont_size=9,
        ))
        fig_r.update_layout(**CHART, height=170, xaxis_title="별점",
                            yaxis=dict(**AXIS, title="리뷰 수"),
                            xaxis=dict(**AXIS))
        st.plotly_chart(fig_r, use_container_width=True)

        # 데이터 요약
        st.markdown(f"""
        <div style='background:{C["surf"]};border:1px solid {C["border"]};
                    border-radius:8px;padding:12px 14px;font-size:12px;
                    line-height:2;color:{C["text2"]};'>
          📁 원본: <b style='color:{C["text"]}'>608,458건</b><br>
          📊 샘플: <b style='color:{C["text"]}'>30,000건</b> (4.9%)<br>
          🏪 식당: <b style='color:{C["text"]}'>100개</b><br>
          👤 유저: <b style='color:{C["text"]}'>12,524명</b><br>
          📅 기간: <b style='color:{C["text"]}'>2011~2015</b>
        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# 🎯  실시간 탐지 데모
# ═════════════════════════════════════════════════════════════════════════════
elif "실시간" in page:
    st.markdown(f"""
    <div class='top-bar'>
      <div>
        <div class='top-logo'>🎯 실시간 리뷰 사기 탐지 데모</div>
        <div class='top-sub'>GNN 사전 추론 결과 기반 · Test Set 6,000건 대상</div>
      </div>
    </div>""", unsafe_allow_html=True)

    ctrl, result = st.columns([1, 2], gap="large")

    with ctrl:
        st.markdown(f"<div class='sec' style='--accent:{C[\"primary\"]};'>🔧 노드 선택</div>",
                    unsafe_allow_html=True)
        lf = st.selectbox("필터", ["전체","실제 스팸만","실제 정상만"], label_visibility="collapsed")

        if st.button("🎲 랜덤 선택", use_container_width=True, type="primary"):
            idx = np.where(tmask)[0]
            if lf=="실제 스팸만":   idx=idx[labels[idx]==1]
            elif lf=="실제 정상만": idx=idx[labels[idx]==0]
            st.session_state["sel"] = int(np.random.choice(idx))

        nid = st.number_input("노드 ID 직접 입력", 0, len(df)-1, 0, label_visibility="collapsed")
        if st.button("이 노드 분석", use_container_width=True):
            st.session_state["sel"] = int(nid)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>📈 Test Set 요약</div>",
                    unsafe_allow_html=True)
        tp = probs[tmask]
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;'>
          <div class='kpi' style='--accent:{C["danger"]};padding:12px;'>
            <div class='kpi-lbl'>탐지율</div>
            <div class='kpi-val' style='font-size:22px;color:{C["danger"]};'>{(tp>0.5).mean():.1%}</div>
          </div>
          <div class='kpi' style='--accent:{C["primary"]};padding:12px;'>
            <div class='kpi-lbl'>평균 확률</div>
            <div class='kpi-val' style='font-size:22px;color:{C["primary"]};'>{tp.mean():.3f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    with result:
        sel = st.session_state.get("sel")
        if sel is None:
            st.markdown(f"""
            <div style='height:260px;display:flex;align-items:center;justify-content:center;
                        background:{C["surf"]};border:1px dashed {C["border"]};border-radius:10px;'>
              <div style='text-align:center;color:{C["muted"]};'>
                <div style='font-size:44px;'>🔍</div>
                <div style='margin-top:10px;font-size:13px;'>왼쪽에서 리뷰를 선택하세요</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            fp = float(probs[sel])
            is_spam   = fp >= 0.5
            real_spam = int(labels[sel]) == 1
            color = C["danger"] if is_spam else C["success"]

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fp*100,
                number={"suffix":"%","font":{"size":40,"color":color}},
                title={"text":"사기 확률","font":{"size":13,"color":C["text2"]}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":C["border"],
                            "tickfont":{"color":C["text2"],"size":10}},
                    "bar":{"color":color,"thickness":0.22},
                    "bgcolor":C["surf2"], "bordercolor":C["border"],
                    "steps":[
                        {"range":[0,30],  "color":"rgba(16,185,129,0.08)"},
                        {"range":[30,60], "color":"rgba(245,158,11,0.08)"},
                        {"range":[60,100],"color":"rgba(239,68,68,0.08)"},
                    ],
                    "threshold":{"line":{"color":C["danger"],"width":2.5},"value":50},
                },
            ))
            fig_g.update_layout(**CHART, height=220)
            st.plotly_chart(fig_g, use_container_width=True)

            ok = is_spam == real_spam
            v1,v2,v3 = st.columns(3)
            for col_c, lbl_c, val_c, clr_c in [
                (v1,"GNN 판정","🚨 사기" if is_spam else "✅ 정상",C["danger"] if is_spam else C["success"]),
                (v2,"실제 라벨","🚨 스팸" if real_spam else "✅ 정상",C["danger"] if real_spam else C["success"]),
                (v3,"판정 결과","✓ 정확" if ok else "✗ 오분류",C["success"] if ok else C["danger"]),
            ]:
                with col_c:
                    st.markdown(f"""
                    <div style='background:{clr_c}18;border:1px solid {clr_c}44;
                                border-radius:8px;padding:10px;text-align:center;'>
                      <div style='font-size:15px;font-weight:700;color:{clr_c};'>{val_c}</div>
                      <div style='font-size:10px;color:{C["text2"]};margin-top:3px;'>{lbl_c}</div>
                    </div>""", unsafe_allow_html=True)

    if sel is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>🔎 탐지 근거 — 노드 {sel}</div>",
                    unsafe_allow_html=True)
        nb  = np.where((bei[0]==sel)|(bei[1]==sel))[0]
        ns  = np.where((sei[0]==sel)|(sei[1]==sel))[0]
        nu  = np.where((rur[0]==sel)|(rur[1]==sel))[0]
        adt = float(bdt[nb].mean()) if len(nb)>0 else 0.

        m1,m2,m3,m4 = st.columns(4)
        for col_m, lbl_m, val_m, sub_m, clr_m in [
            (m1,"R-U-R 연결",    f"{len(nu)}개", "동일 계정 (기여도 20%)",  C["purple"]),
            (m2,"R-Burst-R",    f"{len(nb)}개", "72h 이내 버스트",          C["teal"]),
            (m3,"평균 Δt",      f"{adt:.1f}h",  "↓ 낮을수록 의심",          C["warn"]),
            (m4,"R-Sim-R",      f"{len(ns)}개", "복붙 유사도 연결",          C["primary"]),
        ]:
            with col_m:
                st.markdown(f"""
                <div class='kpi' style='--accent:{clr_m};padding:12px;'>
                  <div class='kpi-lbl'>{lbl_m}</div>
                  <div class='kpi-val' style='font-size:22px;color:{clr_m};'>{val_m}</div>
                  <div class='kpi-sub'>{sub_m}</div>
                </div>""", unsafe_allow_html=True)

        dl, dr = st.columns([3,2])
        with dl:
            row = df.iloc[sel] if sel < len(df) else None
            if row is not None:
                st.markdown(f"<div class='sec' style='--accent:{C[\"primary\"]};'>리뷰 텍스트</div>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='review-box'>{str(row.get('text',''))[:420]}</div>",
                            unsafe_allow_html=True)

        with dr:
            st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>신호 강도</div>",
                        unsafe_allow_html=True)
            sig = [("R-U-R (20%)", len(nu), C["purple"]),
                   ("R-Burst-R (6%)", len(nb), C["teal"]),
                   ("R-Sim-R (1%)", len(ns), C["primary"])]
            fig_s = go.Figure([go.Bar(
                y=[n for n,_,_ in sig], x=[c for _,c,_ in sig],
                orientation="h",
                marker_color=[cl for _,_,cl in sig],
                opacity=0.85,
                text=[f"{c}개" for _,c,_ in sig], textposition="outside",
                textfont=dict(size=11),
            )])
            fig_s.update_layout(**CHART, height=160, showlegend=False,
                                xaxis=dict(**AXIS, range=[0, max([c for _,c,_ in sig],default=1)*1.5+1]),
                                yaxis=dict(**AXIS))
            st.plotly_chart(fig_s, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# 🕸  캠페인 분석
# ═════════════════════════════════════════════════════════════════════════════
elif "캠페인" in page:
    st.markdown(f"""
    <div class='top-bar'>
      <div>
        <div class='top-logo'>🕸 사기 캠페인 탐지기</div>
        <div class='top-sub'>R-Sim-R + R-Burst-R 기반 조직적 어뷰징 클러스터 식별</div>
      </div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4 = st.columns(4)
    for col_k, lbl_k, val_k, sub_k, clr_k in [
        (k1,"전체 컴포넌트", f"{csum.get('total_components',1584):,}개", "연결된 리뷰 그룹", C["primary"]),
        (k2,"의심 캠페인",   f"{csum.get('suspected_campaigns',91)}개",  "스팸 비율 ≥ 50%", C["danger"]),
        (k3,"캠페인 내 리뷰",f"{csum.get('total_reviews_in_campaigns',464)}건","조직적 어뷰징 대상",C["warn"]),
        (k4,"최대 캠페인",   f"{csum.get('max_campaign_size',90)}개",     "단일 최대 클러스터",C["teal"]),
    ]:
        with col_k:
            st.markdown(f"""
            <div class='kpi' style='--accent:{clr_k};'>
              <div class='kpi-lbl'>{lbl_k}</div>
              <div class='kpi-val' style='color:{clr_k};'>{val_k}</div>
              <div class='kpi-sub'>{sub_k}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([3,2], gap="large")

    with cl:
        st.markdown(f"<div class='sec' style='--accent:{C[\"danger\"]};'>🗂 의심 캠페인 목록</div>",
                    unsafe_allow_html=True)
        if not camps.empty:
            sp_col = "spam_ratio" if "spam_ratio" in camps.columns else camps.columns[2]
            fl_col = "is_campaign" if "is_campaign" in camps.columns else None
            dc = camps[camps[fl_col]==True].copy() if fl_col else camps[camps[sp_col]>=0.5].copy()
            dc["스팸 비율"] = dc[sp_col].apply(lambda v:f"{v:.0%}")
            show = [c for c in ["campaign_id","n_nodes","스팸 비율","avg_fraud_prob"] if c in dc.columns]
            st.dataframe(
                dc[show].rename(columns={"campaign_id":"캠페인 ID","n_nodes":"리뷰 수","avg_fraud_prob":"평균 사기 확률"})
                .head(25).style.background_gradient(subset=["평균 사기 확률"] if "평균 사기 확률" in [c.replace("avg_fraud_prob","평균 사기 확률") for c in show] else [],
                                                    cmap="Reds"),
                use_container_width=True, height=360
            )

    with cr:
        st.markdown(f"<div class='sec' style='--accent:{C[\"warn\"]};'>📊 캠페인 크기 분포</div>",
                    unsafe_allow_html=True)
        if not camps.empty and "n_nodes" in camps.columns:
            fig_h = go.Figure(go.Histogram(
                x=camps["n_nodes"], nbinsx=18,
                marker_color=C["warn"], opacity=0.8,
                marker_line=dict(color=C["bg"], width=0.5),
            ))
            fig_h.update_layout(**CHART, height=190,
                                xaxis=dict(**AXIS, title="캠페인 내 리뷰 수"),
                                yaxis=dict(**AXIS, title="빈도"))
            fig_h.add_vline(x=camps["n_nodes"].mean(), line_dash="dot",
                            line_color=C["primary"],
                            annotation_text=f"평균 {camps['n_nodes'].mean():.1f}개",
                            annotation_position="top right",
                            annotation_font_color=C["primary"])
            st.plotly_chart(fig_h, use_container_width=True)

        sp2 = "spam_ratio" if "spam_ratio" in camps.columns else None
        if sp2:
            dc2 = camps.dropna(subset=[sp2]).copy()
            dc2["spam_ratio"] = dc2[sp2]
            vals = [(dc2["spam_ratio"]==1.0).sum(),
                    ((dc2["spam_ratio"]>=0.75)&(dc2["spam_ratio"]<1.0)).sum(),
                    ((dc2["spam_ratio"]>=0.5)&(dc2["spam_ratio"]<0.75)).sum(),
                    (dc2["spam_ratio"]<0.5).sum()]
            fig_d = go.Figure(go.Pie(
                labels=["100%","75~99%","50~74%","< 50%"],
                values=vals, hole=0.58,
                marker_colors=[C["danger"],C["warn"],C["primary"],C["muted"]],
                textfont_size=11,
            ))
            fig_d.update_layout(**CHART, height=185,
                                annotations=[dict(text="스팸\n비율",x=0.5,y=0.5,
                                               showarrow=False,font_size=11,
                                               font_color=C["text"])])
            st.plotly_chart(fig_d, use_container_width=True)

    viz = RPT/"fraud_network_viz.html"
    if viz.exists():
        st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>🌐 사기 네트워크 시각화 — 🔴 스팸 · 🔵 정상</div>",
                    unsafe_allow_html=True)
        st.components.v1.html(open(viz,encoding="utf-8").read(), height=560)


# ═════════════════════════════════════════════════════════════════════════════
# 📊  모델 성능 비교
# ═════════════════════════════════════════════════════════════════════════════
elif "모델" in page:
    st.markdown(f"""
    <div class='top-bar'>
      <div>
        <div class='top-logo'>📊 모델 성능 비교</div>
        <div class='top-sub'>트랜스덕티브 · 인덕티브 · 학습 수렴 분석</div>
      </div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏆 트랜스덕티브 성능", "🌍 인덕티브 (배포 환경)", "📈 학습 곡선"])

    with tab1:
        st.markdown(f"<div class='sec' style='--accent:{C[\"primary\"]};'>PR-AUC 및 Macro F1 순위 (상위 14개)</div>",
                    unsafe_allow_html=True)

        lf = log.sort_values("pr_auc", ascending=False).head(14)
        lfs = lf.sort_values("pr_auc", ascending=True)

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["PR-AUC ↑ 높을수록 좋음", "Macro F1 ↑ 높을수록 좋음"],
                            horizontal_spacing=0.12)
        for i, (metric, goal) in enumerate([("pr_auc",0.85),("macro_f1",0.85)]):
            vals = lfs[metric].values
            bar_c = [C["success"] if v==vals.max() else
                     (C["primary"] if v>=0.90 else C["muted"]) for v in vals]
            fig.add_trace(go.Bar(
                y=lfs["model"], x=vals, orientation="h",
                marker_color=bar_c, opacity=0.85,
                text=[f"{v:.4f}" for v in vals],
                textposition="outside", textfont_size=10,
                showlegend=False,
            ), row=1, col=i+1)
            # 목표 기준선
            fig.add_vline(x=goal, line_dash="dot", line_color=C["warn"],
                          line_width=1.5, row=1, col=i+1)

        fig.update_xaxes(range=[0.55, 1.06], row=1, col=1, **AXIS)
        fig.update_xaxes(range=[0.55, 1.06], row=1, col=2, **AXIS)
        fig.update_yaxes(**AXIS)
        fig.update_layout(**CHART, height=440)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class='insight' style='--accent:{C["success"]};'>
          ✅ <b>3-way Ensemble</b> PR-AUC 0.9419 / F1 0.9386 달성 (베이스라인 0.83 대비 +11.2%p) ·
          점선 = 목표 기준 0.85
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div class='sec' style='--accent:{C[\"muted\"]};margin-top:20px;'>전체 실험 결과</div>",
                    unsafe_allow_html=True)
        disp = log[["model","pr_auc","macro_f1","params","notes"]].copy()
        disp.columns = ["모델","PR-AUC","Macro-F1","파라미터","비고"]
        st.dataframe(disp, use_container_width=True, height=280)

    with tab2:
        st.markdown(f"<div class='sec' style='--accent:{C[\"purple\"]};'>인덕티브 PR-AUC — Train→Test 엣지 완전 차단 후 평가</div>",
                    unsafe_allow_html=True)
        st.markdown("<span class='badge badge-amber'>⚠ 실제 배포 환경 시뮬레이션</span>",
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        extra = []
        for nm, sc in e4.get("individual_scores", {}).items():
            if nm not in logi["model"].values:
                extra.append({"model":nm,"pr_auc":sc["pr_auc"],"macro_f1":sc["macro_f1"]})
        extra.append({
            "model":"4-way Ensemble ★",
            "pr_auc":e4.get("inductive_pr_auc",0.7748),
            "macro_f1":e4.get("inductive_macro_f1",0.8158),
        })
        li = pd.concat([logi, pd.DataFrame(extra)], ignore_index=True)
        li = li.drop_duplicates("model").sort_values("pr_auc", ascending=True)

        bar_c2 = [C["success"] if v>=0.75 else
                  (C["primary"] if v>=0.65 else
                   (C["warn"] if v>=0.5 else C["danger"])) for v in li["pr_auc"]]

        fig2 = go.Figure(go.Bar(
            y=li["model"], x=li["pr_auc"], orientation="h",
            marker_color=bar_c2, opacity=0.85,
            text=[f"{v:.4f}" for v in li["pr_auc"]],
            textposition="outside", textfont_size=10,
        ))
        fig2.add_vline(x=0.7,    line_dash="dot", line_color=C["warn"],
                       line_width=1.5, annotation_text="0.7 기준선",
                       annotation_font_color=C["warn"])
        fig2.add_vline(x=0.7748, line_dash="dash", line_color=C["success"],
                       line_width=1.5, annotation_text="4-way 0.7748",
                       annotation_position="bottom right",
                       annotation_font_color=C["success"])
        fig2.update_xaxes(range=[0, 1.12], **AXIS)
        fig2.update_yaxes(**AXIS)
        fig2.update_layout(**CHART, height=max(320, len(li)*26+80),
                           title="인덕티브 PR-AUC (배포 환경 기준)")
        st.plotly_chart(fig2, use_container_width=True)

        # Gap 비교
        merged = pd.merge(
            log[["model","pr_auc"]].rename(columns={"pr_auc":"transductive"}),
            logi[["model","pr_auc"]].rename(columns={"pr_auc":"inductive"}),
            on="model", how="inner"
        ).sort_values("transductive", ascending=False).head(8)

        fig3 = go.Figure([
            go.Bar(name="트랜스덕티브", x=merged["model"], y=merged["transductive"],
                   marker_color=C["primary"], opacity=0.8),
            go.Bar(name="인덕티브",     x=merged["model"], y=merged["inductive"],
                   marker_color=C["teal"], opacity=0.8),
        ])
        fig3.add_hline(y=0.7, line_dash="dot", line_color=C["warn"], line_width=1)
        fig3.update_xaxes(**AXIS); fig3.update_yaxes(**AXIS, range=[0,1.05])
        fig3.update_layout(**CHART, height=300, barmode="group",
                           title="트랜스덕티브 vs 인덕티브 Gap 비교",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        st.markdown(f"<div class='sec' style='--accent:{C[\"teal\"]};'>주요 모델 PR-AUC 학습 수렴 곡선</div>",
                    unsafe_allow_html=True)
        st.info("학습 곡선 데이터는 로컬 환경에서만 제공됩니다.")


# ═════════════════════════════════════════════════════════════════════════════
# 🔬  XAI 탐지 근거
# ═════════════════════════════════════════════════════════════════════════════
elif "XAI" in page:
    st.markdown(f"""
    <div class='top-bar'>
      <div>
        <div class='top-logo'>🔬 XAI — 설명 가능한 사기 탐지</div>
        <div class='top-sub'>Edge Ablation 기여도 분석 · 어뷰저 전략 진화 발견</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if not attr.empty:
        xa1, xa2 = st.columns([3,2], gap="large")

        with xa1:
            st.markdown(f"<div class='sec' style='--accent:{C[\"purple\"]};'>엣지 제거 시 PR-AUC 하락 = 기여도</div>",
                        unsafe_allow_html=True)
            a2 = attr.sort_values("contribution_pct", ascending=True)
            edge_c = [C["purple"] if v==a2["contribution_pct"].max() else
                      (C["teal"] if v>=3 else (C["primary"] if v>=1 else C["muted"]))
                      for v in a2["contribution_pct"]]
            fig_x = go.Figure(go.Bar(
                y=a2["edge_type"], x=a2["contribution_pct"],
                orientation="h", marker_color=edge_c, opacity=0.85,
                text=[f"{v:.2f}%" for v in a2["contribution_pct"]],
                textposition="outside", textfont_size=11,
            ))
            # 평균 기여도 기준선
            avg_c = attr["contribution_pct"].mean()
            fig_x.add_vline(x=avg_c, line_dash="dot", line_color=C["warn"],
                            line_width=1.5,
                            annotation_text=f"평균 {avg_c:.1f}%",
                            annotation_font_color=C["warn"])
            fig_x.update_xaxes(**AXIS, title="기여도 (%)")
            fig_x.update_yaxes(**AXIS)
            fig_x.update_layout(**CHART, height=260)
            st.plotly_chart(fig_x, use_container_width=True)

            st.dataframe(
                attr[["edge_type","n_edges","base_pr_auc","ablated_pr_auc","contribution_pct"]]
                .rename(columns={"edge_type":"엣지","n_edges":"수","base_pr_auc":"기준",
                                 "ablated_pr_auc":"제거 후","contribution_pct":"기여도(%)"}),
                use_container_width=True, hide_index=True,
            )

        with xa2:
            st.markdown(f"<div class='sec' style='--accent:{C[\"purple\"]};'>기여도 구성 비율</div>",
                        unsafe_allow_html=True)
            ec = [C["purple"],C["teal"],C["primary"],C["warn"],C["muted"]]
            fig_d = go.Figure(go.Pie(
                labels=attr["edge_type"], values=attr["contribution_pct"],
                hole=0.58, marker_colors=ec, textfont_size=11,
                textinfo="label+percent",
            ))
            fig_d.update_layout(**CHART, height=240,
                                annotations=[dict(text="기여도",x=0.5,y=0.5,
                                               showarrow=False,font_size=13,
                                               font_color=C["text"])])
            st.plotly_chart(fig_d, use_container_width=True)

            st.markdown(f"""
            <div class='insight' style='--accent:{C["purple"]};'>
              <b style='color:{C["purple"]}'>💡 핵심 발견</b><br>
              • <b>R-U-R 19.97%</b> — 동일 계정 반복이 최대 신호<br>
              • R-Burst-R 5.63% — 72h 집중 2위<br>
              • <b style='color:{C["muted"]}'>R-S-R 0.20%</b> — 317K 엣지지만 기여 최저<br>
              → DRAGWave가 이 비대칭을 자동 학습
            </div>""", unsafe_allow_html=True)

    # 어뷰저 진화 분석
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec' style='--accent:{C[\"warn\"]};'>⏳ 어뷰저 전략 진화 — K-Fold 시간 분할 분석</div>",
                unsafe_allow_html=True)

    if kf and "kfold" in kf:
        kdf = pd.DataFrame(kf["kfold"]).dropna(subset=["pr_auc"])
        kdf["spam_pct"] = kdf["spam_ratio"] * 100

        fig_k = make_subplots(specs=[[{"secondary_y":True}]])
        bar_ck = [C["success"] if v>=0.7 else (C["warn"] if v>=0.4 else C["danger"])
                  for v in kdf["pr_auc"]]
        fig_k.add_trace(go.Bar(
            x=[f"Fold {r['fold']}" for _,r in kdf.iterrows()],
            y=kdf["pr_auc"], name="인덕티브 PR-AUC",
            marker_color=bar_ck, opacity=0.82,
            text=[f"{v:.3f}" for v in kdf["pr_auc"]],
            textposition="outside", textfont_size=11,
        ), secondary_y=False)
        fig_k.add_trace(go.Scatter(
            x=[f"Fold {r['fold']}" for _,r in kdf.iterrows()],
            y=kdf["spam_pct"], name="스팸 밀도 (%)",
            mode="lines+markers+text",
            line=dict(color=C["purple"], width=2.5, dash="dot"),
            marker=dict(size=9,color=C["purple"],line=dict(color=C["bg"],width=2)),
            text=[f"{v:.1f}%" for v in kdf["spam_pct"]],
            textposition="top center", textfont=dict(size=10,color=C["purple"]),
        ), secondary_y=True)
        # 기준선
        fig_k.add_hline(y=0.7, line_dash="dot", line_color=C["success"],
                        line_width=1.5, secondary_y=False,
                        annotation_text="PR-AUC 0.7 기준",
                        annotation_font_color=C["success"])
        fig_k.update_yaxes(title_text="인덕티브 PR-AUC", secondary_y=False,
                           range=[0,1.1], **AXIS)
        fig_k.update_yaxes(title_text="스팸 밀도 (%)", secondary_y=True,
                           range=[0,45], showgrid=False,
                           tickcolor=C["border"], tickfont_color=C["text2"])
        fig_k.update_xaxes(**AXIS)
        fig_k.update_layout(**CHART, height=320,
                            title="시간 구간별 PR-AUC vs 스팸 밀도",
                            legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig_k, use_container_width=True)

        c_ins, c_sum = st.columns([2,1])
        with c_ins:
            st.markdown(f"""
            <div class='insight' style='--accent:{C["warn"]};'>
              <b style='color:{C["warn"]}'>🔍 발견: 어뷰저 전략 진화</b><br>
              초기 구간(스팸 29.4%) → 후기 구간(6~12%)으로 스팸 밀도가 급감.
              Time-decay 재학습 포함 3가지 완화 시도 모두 PR-AUC 개선 없음 (최대 Δ+0.006).
              <b>이는 모델 실패가 아닌, 플랫폼 탐지 강화에 적응한 어뷰저 전략 진화의 증거다.</b>
            </div>""", unsafe_allow_html=True)
        with c_sum:
            if "summary" in kf:
                sm = kf["summary"]
                st.markdown(f"""
                <div class='kpi' style='--accent:{C["purple"]};'>
                  <div class='kpi-lbl'>K-Fold Mean PR-AUC</div>
                  <div class='kpi-val' style='color:{C["purple"]};'>{sm.get("mean_pr_auc","—")}</div>
                  <div class='kpi-sub'>std = {sm.get("std_pr_auc","—")}</div>
                </div>""", unsafe_allow_html=True)

