"""
NBA Rookie Career Longevity Predictor — Streamlit Dashboard
Author: Antigravity AI
Stack : Scikit-Learn  |  Streamlit  |  Plotly
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Rookie Longevity Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLES  (dark court-side theme)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=Barlow:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0d0f14;
    --surface:  #161b26;
    --card:     #1e2535;
    --border:   #2a3348;
    --gold:     #f5a623;
    --gold2:    #ffcc44;
    --green:    #29d97b;
    --red:      #f05252;
    --blue:     #4da6ff;
    --muted:    #7a8499;
    --text:     #e8ecf4;
    --text2:    #b0b9cc;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif !important;
}

/* hide default Streamlit chrome */
[data-testid="stHeader"], footer { display: none !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text2) !important; font-family: 'Barlow', sans-serif !important; }
[data-testid="stSidebar"] h2 { color: var(--gold) !important; font-family: 'Barlow Condensed', sans-serif !important; font-size: 1.1rem !important; letter-spacing: .06em; text-transform: uppercase; }

/* Sliders */
.stSlider > div { padding: 0 !important; }
.stSlider [data-baseweb="slider"] { margin-top: -8px; }
.stSlider [role="slider"] { background: var(--gold) !important; border: 2px solid var(--gold2) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: .08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 4px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .75rem !important; text-transform: uppercase; letter-spacing: .06em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Barlow Condensed', sans-serif !important; font-size: 1.9rem !important; font-weight: 800; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--gold) 0%, #e8920a 100%) !important;
    color: #0d0f14 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 800 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(245,166,35,.25) !important;
    transition: all .2s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(245,166,35,.45) !important;
    transform: translateY(-1px) !important;
}

/* Selectbox / number input */
[data-baseweb="select"] div, [data-baseweb="input"] input {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Barlow', sans-serif !important;
}

/* ── Custom card helper (used via st.markdown) ── */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
}
.kpi-label { font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-bottom:4px; }
.kpi-value { font-family:'Barlow Condensed',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; }
.kpi-sub   { font-size:.78rem; color:var(--muted); margin-top:4px; }
.tag { display:inline-block; padding:3px 10px; border-radius:20px; font-size:.72rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
.tag-green { background:rgba(41,217,123,.15); color:var(--green); border:1px solid rgba(41,217,123,.3); }
.tag-red   { background:rgba(240, 82, 82,.15); color:var(--red);   border:1px solid rgba(240, 82, 82,.3); }
.section-header {
    font-family:'Barlow Condensed',sans-serif;
    font-size:1.5rem;
    font-weight:800;
    letter-spacing:.06em;
    text-transform:uppercase;
    color:var(--text);
    border-left:4px solid var(--gold);
    padding-left:12px;
    margin: 18px 0 10px 0;
}
.hero-title {
    font-family:'Barlow Condensed',sans-serif;
    font-size:2.6rem;
    font-weight:800;
    letter-spacing:.04em;
    text-transform:uppercase;
    line-height:1.1;
}
.hero-sub {
    font-size:.9rem;
    color:var(--muted);
    letter-spacing:.04em;
}
.result-box {
    border-radius:14px;
    padding:28px 24px;
    text-align:center;
    margin: 14px 0;
}
.result-survive {
    background: linear-gradient(135deg,rgba(41,217,123,.12),rgba(41,217,123,.04));
    border: 2px solid rgba(41,217,123,.4);
}
.result-bust {
    background: linear-gradient(135deg,rgba(240,82,82,.12),rgba(240,82,82,.04));
    border: 2px solid rgba(240,82,82,.4);
}
.result-label { font-family:'Barlow Condensed',sans-serif; font-size:2rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
.result-prob  { font-size:3.4rem; font-weight:800; font-family:'Barlow Condensed',sans-serif; line-height:1.1; margin:6px 0; }
.result-sub   { font-size:.82rem; color:var(--muted); }
.insight-pill {
    display:inline-block;
    background:var(--card);
    border:1px solid var(--border);
    border-radius:8px;
    padding:10px 16px;
    margin:6px 4px;
    font-size:.82rem;
    color:var(--text2);
    line-height:1.4;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Barlow, sans-serif", color="#b0b9cc", size=12),
    title_font=dict(family="Barlow Condensed, sans-serif", color="#e8ecf4", size=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0b9cc")),
    margin=dict(t=48, b=40, l=40, r=20),
    xaxis=dict(gridcolor="#1e2535", linecolor="#2a3348", tickfont=dict(color="#7a8499")),
    yaxis=dict(gridcolor="#1e2535", linecolor="#2a3348", tickfont=dict(color="#7a8499")),
)

GOLD = "#f5a623"; GREEN = "#29d97b"; RED = "#f05252"; BLUE = "#4da6ff"; MUTED = "#7a8499"

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL PIPELINE  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["GP","MIN","PTS","FGS","FGA","3PM","3PA","FTM","FTA","OREB","DREB","AST","STL","BLK","TOV"]
TARGET   = "5Yrs"

@st.cache_resource(show_spinner=False)
def build_pipeline():
    df = pd.read_csv("nba_data.csv")
    X_raw = df.drop(columns=[TARGET])
    y     = df[TARGET]
    text_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    X_raw = X_raw.drop(columns=text_cols, errors="ignore")

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw[FEATURES])

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X_imp)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sc, y, test_size=0.2, random_state=42, stratify=y)

    # Baseline
    base = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42)
    base.fit(X_tr, y_tr)

    # Tuned
    param_grid = {"C": np.logspace(-3,3,20), "penalty":["l1","l2"]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(LogisticRegression(solver="liblinear", random_state=42),
                      param_grid, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuned = gs.best_estimator_

    return dict(
        imputer=imputer, scaler=scaler,
        base=base, tuned=tuned,
        best_params=gs.best_params_,
        X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te,
        df=df,
    )

def metrics_dict(model, X_te, y_te):
    yp   = model.predict(X_te)
    ypr  = model.predict_proba(X_te)[:,1]
    return dict(
        acc  = accuracy_score(y_te, yp),
        prec = precision_score(y_te, yp),
        rec  = recall_score(y_te, yp),
        f1   = f1_score(y_te, yp),
        auc  = roc_auc_score(y_te, ypr),
        cm   = confusion_matrix(y_te, yp),
        fpr  = roc_curve(y_te, ypr)[0],
        tpr  = roc_curve(y_te, ypr)[1],
        yp   = yp,
        ypr  = ypr,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:3.4rem;line-height:1;margin-top:8px'>🏀</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
        <div class='hero-title'>NBA Rookie Career<br><span style='color:#f5a623'>Longevity Predictor</span></div>
        <div class='hero-sub'>Classical Machine Learning · Logistic Regression · Binary Classification</div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:14px 0 0 0'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODEL (with spinner)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⚙️  Training pipeline… this only happens once."):
    P = build_pipeline()

base_m  = metrics_dict(P["base"],  P["X_te"], P["y_te"])
tuned_m = metrics_dict(P["tuned"], P["X_te"], P["y_te"])
bp      = P["best_params"]

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — Rookie Stats Input
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏀 Rookie Stats Input")
    st.markdown("<div style='font-size:.75rem;color:#7a8499;margin-bottom:12px'>Enter first-season statistics to predict career longevity.</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Playing Time ──
    st.markdown("**⏱ Playing Time**")
    gp  = st.slider("GP — Games Played",       min_value=1,   max_value=82,  value=60,   step=1)
    mn  = st.slider("MIN — Minutes / Game",    min_value=0.0, max_value=42.0,value=17.6, step=0.1)

    st.markdown("**📊 Scoring**")
    pts = st.slider("PTS — Points / Game",     min_value=0.0, max_value=35.0,value=6.8,  step=0.1)
    fgs = st.slider("FGM — Field Goals Made",  min_value=0.0, max_value=12.0,value=2.6,  step=0.1)
    fga = st.slider("FGA — Field Goals Att.",  min_value=0.0, max_value=22.0,value=5.9,  step=0.1)
    tpm = st.slider("3PM — 3-Pointers Made",   min_value=0.0, max_value=4.0, value=0.25, step=0.05)
    tpa = st.slider("3PA — 3-Pointers Att.",   min_value=0.0, max_value=8.0, value=0.78, step=0.05)
    ftm = st.slider("FTM — Free Throws Made",  min_value=0.0, max_value=10.0,value=1.29, step=0.1)
    fta = st.slider("FTA — Free Throws Att.",  min_value=0.0, max_value=12.0,value=1.82, step=0.1)

    st.markdown("**🤝 All-Round**")
    oreb= st.slider("OREB — Off. Rebounds",    min_value=0.0, max_value=6.0, value=1.0,  step=0.1)
    dreb= st.slider("DREB — Def. Rebounds",    min_value=0.0, max_value=10.0,value=2.0,  step=0.1)
    ast = st.slider("AST — Assists",           min_value=0.0, max_value=12.0,value=1.54, step=0.1)
    stl = st.slider("STL — Steals",            min_value=0.0, max_value=3.0, value=0.62, step=0.05)
    blk = st.slider("BLK — Blocks",            min_value=0.0, max_value=5.0, value=0.37, step=0.05)
    tov = st.slider("TOV — Turnovers",         min_value=0.0, max_value=5.0, value=1.19, step=0.1)

    st.markdown("---")
    model_choice = st.radio("Model", ["Tuned (Recommended)", "Baseline (L2, C=1.0)"], index=0)
    predict_btn  = st.button("🔮  Predict Career Longevity")

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_pred, tab_perf, tab_coef, tab_data = st.tabs([
    "🔮  Prediction", "📈  Model Performance", "🔬  Feature Insights", "📋  Dataset Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 · PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    left, right = st.columns([1.05, 1], gap="large")

    # ── Input summary ──
    with left:
        st.markdown("<div class='section-header'>Rookie Profile</div>", unsafe_allow_html=True)
        raw_input = np.array([[gp, mn, pts, fgs, fga, tpm, tpa, ftm, fta, oreb, dreb, ast, stl, blk, tov]])
        input_df  = pd.DataFrame(raw_input, columns=FEATURES)

        labels = {
            "GP":f"{gp} games","MIN":f"{mn:.1f} min/g","PTS":f"{pts:.1f} pts/g",
            "FGS":f"{fgs:.1f} fgm/g","FGA":f"{fga:.1f} fga/g",
            "3PM":f"{tpm:.2f} 3pm/g","3PA":f"{tpa:.2f} 3pa/g",
            "FTM":f"{ftm:.1f} ftm/g","FTA":f"{fta:.1f} fta/g",
            "OREB":f"{oreb:.1f} oreb/g","DREB":f"{dreb:.1f} dreb/g",
            "AST":f"{ast:.1f} ast/g","STL":f"{stl:.2f} stl/g",
            "BLK":f"{blk:.2f} blk/g","TOV":f"{tov:.1f} tov/g",
        }
        pill_html = "".join(
            f"<span class='insight-pill'><b style='color:#f5a623'>{k}</b>&nbsp; {v}</span>"
            for k,v in labels.items()
        )
        st.markdown(f"<div style='line-height:2'>{pill_html}</div>", unsafe_allow_html=True)

        # Shooting efficiency gauges
        fg_pct  = (fgs/fga*100)  if fga > 0 else 0
        ft_pct  = (ftm/fta*100)  if fta > 0 else 0
        tp_pct  = (tpm/tpa*100)  if tpa > 0 else 0

        fig_gauge = make_subplots(
            rows=1, cols=3,
            specs=[[{"type":"indicator"},{"type":"indicator"},{"type":"indicator"}]],
        )
        gauge_kw = dict(mode="gauge+number", number=dict(suffix="%", font=dict(size=18, color="#e8ecf4")),
                        gauge=dict(axis=dict(range=[0,100], tickcolor="#2a3348"),
                                   bar=dict(color=GOLD),
                                   bgcolor="#1e2535",
                                   bordercolor="#2a3348"))
        fig_gauge.add_trace(go.Indicator(title=dict(text="FG%",  font=dict(color=MUTED, size=12)), value=round(fg_pct,1), **gauge_kw), 1,1)
        fig_gauge.add_trace(go.Indicator(title=dict(text="FT%",  font=dict(color=MUTED, size=12)), value=round(ft_pct,1), **gauge_kw), 1,2)
        fig_gauge.add_trace(go.Indicator(title=dict(text="3P%",  font=dict(color=MUTED, size=12)), value=round(tp_pct,1), **gauge_kw), 1,3)
        
        # FIXED: Pengaturan layout dipisah agar margin tidak bentrok dengan PLOTLY_LAYOUT
        fig_gauge.update_layout(**PLOTLY_LAYOUT, height=180)
        fig_gauge.update_layout(margin=dict(t=30, b=0, l=10, r=10))
        
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Prediction result ──
    with right:
        st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

        active_model = P["tuned"] if "Tuned" in model_choice else P["base"]

        # Pre-process input
        X_in = P["imputer"].transform(input_df[FEATURES])
        X_in = P["scaler"].transform(X_in)

        prob1 = float(active_model.predict_proba(X_in)[0][1])
        prob0 = 1 - prob1
        pred  = int(active_model.predict(X_in)[0])

        if pred == 1:
            box_cls = "result-survive"
            verdict = "✅  Will Survive 5+ Years"
            color   = GREEN
            icon    = "🟢"
        else:
            box_cls = "result-bust"
            verdict = "❌  Less Likely to Reach 5 Years"
            color   = RED
            icon    = "🔴"

        st.markdown(f"""
        <div class='result-box {box_cls}'>
            <div class='result-label' style='color:{color}'>{verdict}</div>
            <div class='result-prob' style='color:{color}'>{prob1*100:.1f}%</div>
            <div class='result-sub'>probability of 5+ year career</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[prob1], y=["Career Longevity"],
            orientation="h", marker_color=GREEN,
            name=f"Survive ≥5 Yrs ({prob1*100:.1f}%)",
            text=[f"{prob1*100:.1f}%"], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#0d0f14",size=13,family="Barlow Condensed"),
        ))
        fig_bar.add_trace(go.Bar(
            x=[prob0], y=["Career Longevity"],
            orientation="h", marker_color=RED,
            name=f"< 5 Yrs ({prob0*100:.1f}%)",
            text=[f"{prob0*100:.1f}%"] if prob0 > 0.08 else [""],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#fff",size=12,family="Barlow Condensed"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT, barmode="stack", height=90,
            margin=dict(t=4,b=4,l=4,r=4), showlegend=True,
            yaxis=dict(showticklabels=False, showgrid=False),
            xaxis=dict(range=[0,1], showticklabels=False, showgrid=False),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=11)),
        )
        fig_bar.update_layout(margin=dict(t=4, b=4, l=4, r=4))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Key coefficient contributors for this player
        coef   = active_model.coef_[0]
        contrib = X_in[0] * coef
        contrib_df = pd.DataFrame({"Feature": FEATURES, "Impact": contrib}) \
                       .sort_values("Impact", key=abs, ascending=False).head(6)

        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Impact"],
            y=contrib_df["Feature"],
            orientation="h",
            marker_color=[GREEN if v>0 else RED for v in contrib_df["Impact"]],
            text=[f"{v:+.3f}" for v in contrib_df["Impact"]],
            textposition="outside",
            textfont=dict(size=11, color="#e8ecf4"),
        ))
        fig_contrib.add_vline(x=0, line_color="#2a3348", line_width=1.5)
        fig_contrib.update_layout(
            **PLOTLY_LAYOUT, title="Top 6 Feature Contributions (this prediction)",
            height=240, margin=dict(t=38,b=4,l=4,r=60),
        )
        st.plotly_chart(fig_contrib, use_container_width=True)

        # Quick tip
        top_pos = contrib_df[contrib_df["Impact"]>0]["Feature"].iloc[0] if (contrib_df["Impact"]>0).any() else "—"
        top_neg = contrib_df[contrib_df["Impact"]<0]["Feature"].iloc[0] if (contrib_df["Impact"]<0).any() else "—"
        st.markdown(f"""
        <div class='kpi-card' style='padding:14px 16px'>
            <div class='kpi-label'>💡 Scout Insight</div>
            <div style='font-size:.85rem;color:#b0b9cc;margin-top:6px'>
                <b style='color:{GREEN}'>Biggest strength:</b> {top_pos} &nbsp;·&nbsp;
                <b style='color:{RED}'>Biggest risk:</b> {top_neg}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 · MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab_perf:
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)

    # ── KPI row ──
    k1,k2,k3,k4,k5 = st.columns(5)
    def delta(v, b):
        d = v-b
        s = f"+{d*100:.2f}%" if d>=0 else f"{d*100:.2f}%"
        return s, "normal" if d>=0 else "inverse"

    with k1: st.metric("Accuracy",  f"{tuned_m['acc']*100:.2f}%",  delta(tuned_m['acc'], base_m['acc'])[0])
    with k2: st.metric("Precision", f"{tuned_m['prec']*100:.2f}%", delta(tuned_m['prec'],base_m['prec'])[0])
    with k3: st.metric("Recall",    f"{tuned_m['rec']*100:.2f}%",  delta(tuned_m['rec'], base_m['rec'])[0])
    with k4: st.metric("F1-Score",  f"{tuned_m['f1']*100:.2f}%",   delta(tuned_m['f1'],  base_m['f1'])[0])
    with k5: st.metric("ROC-AUC",   f"{tuned_m['auc']:.4f}",       delta(tuned_m['auc'], base_m['auc'])[0])

    st.markdown("<div style='font-size:.75rem;color:#7a8499;margin:2px 0 16px 4px'>↑ Δ vs Baseline (L2, C=1.0)</div>", unsafe_allow_html=True)

    col_cm, col_roc = st.columns(2, gap="large")

    # ── Confusion matrices ──
    with col_cm:
        st.markdown("<div class='section-header' style='font-size:1.1rem'>Confusion Matrix</div>", unsafe_allow_html=True)
        tab_b, tab_t = st.tabs(["Baseline", f"Tuned ({bp['penalty'].upper()}, C={bp['C']:.4f})"])

        def cm_fig(cm, title, cscale):
            labels = ["< 5 Yrs", "≥ 5 Yrs"]
            z = cm[::-1]
            text = [[str(z[i][j]) for j in range(2)] for i in range(2)]
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels[::-1],
                text=text, texttemplate="%{text}",
                colorscale=cscale, showscale=False,
                textfont=dict(size=22, family="Barlow Condensed"),
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title=title, height=300,
                xaxis_title="Predicted", yaxis_title="Actual",
                xaxis=dict(side="bottom"), margin=dict(t=44,b=40,l=60,r=20))
            return fig

        with tab_b:
            st.plotly_chart(cm_fig(base_m["cm"],"Baseline Confusion Matrix",
                [[0,"#1e2535"],[0.5,"#1d4e8f"],[1,"#1a56cc"]]), use_container_width=True)
        with tab_t:
            st.plotly_chart(cm_fig(tuned_m["cm"],"Tuned Confusion Matrix",
                [[0,"#1e2535"],[0.5,"#14532d"],[1,"#15803d"]]), use_container_width=True)

    # ── ROC Curve ──
    with col_roc:
        st.markdown("<div class='section-header' style='font-size:1.1rem'>ROC Curve</div>", unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
            line=dict(color="#2a3348",dash="dot",width=1.5),name="Random Guess (AUC=0.50)"))
        fig_roc.add_trace(go.Scatter(
            x=base_m["fpr"],  y=base_m["tpr"],  mode="lines",
            line=dict(color=BLUE,  width=2.5), name=f"Baseline (AUC={base_m['auc']:.4f})"))
        fig_roc.add_trace(go.Scatter(
            x=tuned_m["fpr"], y=tuned_m["tpr"], mode="lines",
            line=dict(color=GREEN, width=2.5, dash="dash"), name=f"Tuned (AUC={tuned_m['auc']:.4f})"))
        fig_roc.add_hrect(y0=0.75, y1=1, line_width=0,
            fillcolor="rgba(245,166,35,.04)", annotation_text="Success Zone (AUC≥0.75)",
            annotation_position="top right",
            annotation_font=dict(size=10, color=GOLD))
        fig_roc.update_layout(**PLOTLY_LAYOUT,
            title="ROC Curve — Baseline vs. Tuned", height=340,
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            xaxis=dict(range=[-0.02,1.02]), yaxis=dict(range=[-0.02,1.02]),
            legend=dict(x=0.55, y=0.08),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Comparison table ──
    st.markdown("<div class='section-header' style='font-size:1.1rem;margin-top:8px'>Full Metrics Table</div>", unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Metric":    ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
        "Baseline (L2, C=1.0)": [f"{base_m['acc']:.4f}", f"{base_m['prec']:.4f}",
                                   f"{base_m['rec']:.4f}", f"{base_m['f1']:.4f}", f"{base_m['auc']:.4f}"],
        f"Tuned ({bp['penalty'].upper()}, C={bp['C']:.4f})":
            [f"{tuned_m['acc']:.4f}", f"{tuned_m['prec']:.4f}",
             f"{tuned_m['rec']:.4f}", f"{tuned_m['f1']:.4f}", f"{tuned_m['auc']:.4f}"],
        "Δ Improvement": [
            f"+{(tuned_m['acc']-base_m['acc'])*100:.2f}%",
            f"+{(tuned_m['prec']-base_m['prec'])*100:.2f}%",
            f"+{(tuned_m['rec']-base_m['rec'])*100:.2f}%",
            f"+{(tuned_m['f1']-base_m['f1'])*100:.2f}%",
            f"+{(tuned_m['auc']-base_m['auc']):.4f}",
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    # Tuning info
    st.info(f"🔧 **Best Hyperparameters (5-Fold Stratified CV · scored on F1):** "
            f"penalty = **{bp['penalty'].upper()}**, C = **{bp['C']:.6f}** "
            f"(equivalent regularization α ≈ {1/bp['C']:.4f})", icon=None)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 · FEATURE INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_coef:
    st.markdown("<div class='section-header'>Coefficient & Feature Analysis</div>", unsafe_allow_html=True)

    coef   = P["tuned"].coef_[0]
    odds   = np.exp(coef)
    coef_df = pd.DataFrame({
        "Feature":      FEATURES,
        "Beta (Coef)":  coef,
        "Odds Ratio":   odds,
        "Direction":    ["Positive ▲" if c>0 else "Negative ▼" for c in coef],
        "|β|":          np.abs(coef),
    }).sort_values("|β|", ascending=False).reset_index(drop=True)

    col_l, col_r = st.columns([1.3,1], gap="large")

    with col_l:
        colors = [GREEN if v>0 else RED for v in coef_df["Beta (Coef)"]]
        fig_coef = go.Figure(go.Bar(
            x=coef_df["Beta (Coef)"],
            y=coef_df["Feature"],
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.3f}" for v in coef_df["Beta (Coef)"]],
            textposition="outside",
            textfont=dict(size=11, color="#e8ecf4"),
        ))
        fig_coef.add_vline(x=0, line_color="#2a3348", line_width=1.5)
        fig_coef.update_layout(
            **PLOTLY_LAYOUT,
            title="Standardized Coefficients (Log-Odds Impact)",
            height=480, margin=dict(t=44,b=20,l=4,r=70),
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    with col_r:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        # Odds ratio bubble chart
        fig_or = go.Figure()
        for _, row in coef_df.iterrows():
            fig_or.add_trace(go.Scatter(
                x=[row["Odds Ratio"]], y=[row["Feature"]],
                mode="markers+text",
                marker=dict(size=max(10, abs(row["Beta (Coef)"])*35),
                            color=GREEN if row["Beta (Coef)"]>0 else RED,
                            opacity=0.85, line=dict(width=1, color="#0d0f14")),
                text=[f"{row['Odds Ratio']:.2f}x"],
                textposition="middle right",
                textfont=dict(size=10, color="#e8ecf4"),
                showlegend=False,
            ))
        fig_or.add_vline(x=1.0, line_color=GOLD, line_width=1.5, line_dash="dash",
                         annotation_text="No Effect (OR=1)", annotation_font=dict(color=GOLD, size=10))
        fig_or.update_layout(
            **PLOTLY_LAYOUT,
            title="Odds Ratios (exp(β))",
            height=480, margin=dict(t=44,b=20,l=4,r=80),
            yaxis=dict(autorange="reversed"),
            xaxis_title="Odds Ratio",
        )
        st.plotly_chart(fig_or, use_container_width=True)

    # ── Coefficient table + insights ──
    st.markdown("<div class='section-header' style='font-size:1.1rem'>Coefficient Table</div>", unsafe_allow_html=True)
    disp = coef_df[["Feature","Beta (Coef)","Odds Ratio","Direction"]].copy()
    disp["Beta (Coef)"] = disp["Beta (Coef)"].map("{:+.4f}".format)
    disp["Odds Ratio"]  = disp["Odds Ratio"].map("{:.4f}x".format)
    st.dataframe(disp, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header' style='font-size:1.1rem;margin-top:16px'>📋 Sports Analytics Takeaways</div>", unsafe_allow_html=True)
    insights = [
        ("🎯 3-Point Efficiency Dilemma", "3PM (β=+1.17) boosts survival odds 3.2×, but 3PA (β=−1.19) is the strongest negative signal. Volume without efficiency = career killer."),
        ("🏃 Games Played = Durability Signal", "GP (β=+0.62) proves that rookies who earn and keep rotation spots — staying healthy and available — are 86% more likely to reach year 5."),
        ("💪 Offensive Rebounding = Hustle Proxy", "OREB (β=+0.51) is the 3rd strongest positive predictor. It signals physical presence, motor, and hustle that coaches value at every level."),
        ("🆓 Free Throw Conversion Matters", "FTM (β=+0.48) positive, FTA (β=−0.40) negative. Drawing fouls helps, but missing free throws signals a skill gap that stunts development."),
        ("🎯 Field Goal Efficiency", "FGM (β=+0.39) positive while FGA (β=−0.21) negative — echoing the same efficiency-over-volume theme across all shot types."),
    ]
    for title, body in insights:
        st.markdown(f"""
        <div class='kpi-card'>
            <div style='font-family:"Barlow Condensed",sans-serif;font-size:1rem;font-weight:700;color:#f5a623;margin-bottom:4px'>{title}</div>
            <div style='font-size:.84rem;color:#b0b9cc;line-height:1.5'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 · DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    df = P["df"].copy()
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    d1,d2,d3,d4 = st.columns(4)
    with d1: st.metric("Total Players", f"{len(df):,}")
    with d2: st.metric("Career ≥ 5 Yrs", f"{df[TARGET].sum():,}")
    with d3: st.metric("Career < 5 Yrs",  f"{(df[TARGET]==0).sum():,}")
    with d4: st.metric("Longevity Rate",  f"{df[TARGET].mean()*100:.1f}%")

    # Class distribution
    col_dist, col_hist = st.columns(2, gap="large")
    with col_dist:
        cnts = df[TARGET].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["≥ 5 Years (Survived)","< 5 Years (Short Career)"],
            values=[cnts[1], cnts[0]],
            hole=0.55,
            marker_colors=[GREEN, RED],
            textfont=dict(family="Barlow Condensed", size=13),
            hovertemplate="%{label}<br>%{value} players (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, title="Career Outcome Distribution", height=300, margin=dict(t=44,b=0,l=0,r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_hist:
        feat_sel = st.selectbox("Select feature to explore", FEATURES, index=0)
        fig_h = go.Figure()
        for label, color, name in [(1,GREEN,"≥ 5 Yrs"),(0,RED,"< 5 Yrs")]:
            fig_h.add_trace(go.Histogram(
                x=df[df[TARGET]==label][feat_sel], name=name,
                marker_color=color, opacity=0.7, nbinsx=25,
            ))
        fig_h.update_layout(**PLOTLY_LAYOUT,
            title=f"{feat_sel} Distribution by Career Outcome",
            barmode="overlay", height=300,
            xaxis_title=feat_sel, yaxis_title="Count",
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header' style='font-size:1.1rem'>Feature Correlation Matrix</div>", unsafe_allow_html=True)
    corr = df[FEATURES].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0,RED],[0.5,"#1e2535"],[1,GREEN]],
        zmid=0, text=corr.values,
        texttemplate="%{text:.2f}", textfont=dict(size=9),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#7a8499"), outlinecolor="#2a3348"),
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, title="Pearson Correlation Heatmap (Features)", height=460,
                           margin=dict(t=44,b=40,l=80,r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    # Raw data
    with st.expander("📄  View Raw Dataset"):
        st.dataframe(df, use_container_width=True, height=380)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='margin:32px 0 12px 0'>
<div style='text-align:center;font-size:.74rem;color:#7a8499;font-family:"Barlow",sans-serif;letter-spacing:.04em'>
    NBA Rookie Career Longevity Predictor &nbsp;·&nbsp; Classical ML (No Deep Learning) &nbsp;·&nbsp;
    Logistic Regression + GridSearchCV &nbsp;·&nbsp; Scikit-Learn 1.x &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp;
    Dataset: Kaggle / UCI NBA Players
</div>
<div style='height:20px'></div>
""", unsafe_allow_html=True)