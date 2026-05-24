"""
╔══════════════════════════════════════════════════════════════════╗
║   NBA ROOKIE CAREER LONGEVITY PREDICTOR                         ║
║   Single-file Streamlit App  ·  No external .pkl required       ║
║   Mock ML via position-aware weighted scoring model             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import math
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NBA Rookie Longevity Predictor",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  ·  Retro-Premium NBA Theme
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --court:    #0a0c10;
    --panel:    #111318;
    --card:     #181c24;
    --border:   #252c3a;
    --gold:     #f0a500;
    --gold-lt:  #ffd166;
    --green:    #06d6a0;
    --red:      #ef476f;
    --blue:     #118ab2;
    --text:     #dce3ef;
    --muted:    #68748a;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--court) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"], footer { display: none !important; }

[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stSlider label {
    font-size: .72rem !important;
    letter-spacing: .07em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] .stSlider [role="slider"] {
    background: var(--gold) !important;
    border: 2px solid var(--gold-lt) !important;
}

[data-testid="stTabs"] button {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.05rem !important;
    letter-spacing: .1em !important;
    color: var(--muted) !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
    padding: 8px 22px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; }

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .68rem !important;
    text-transform: uppercase !important;
    letter-spacing: .08em !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    letter-spacing: .04em !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--gold), #d48e00) !important;
    color: #0a0c10 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.15rem !important;
    letter-spacing: .12em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(240,165,0,.28) !important;
}
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

.nba-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    letter-spacing: .08em;
    line-height: 1;
    color: var(--text);
}
.nba-sub {
    font-size: .78rem;
    color: var(--muted);
    letter-spacing: .05em;
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    margin-top: 4px;
}
.section-label {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.3rem;
    letter-spacing: .1em;
    color: var(--text);
    border-left: 3px solid var(--gold);
    padding-left: 10px;
    margin: 20px 0 12px 0;
}
.verdict-box { border-radius: 14px; padding: 30px 24px 24px; text-align: center; }
.verdict-survive {
    background: linear-gradient(160deg,rgba(6,214,160,.13),rgba(6,214,160,.04));
    border: 2px solid rgba(6,214,160,.4);
}
.verdict-bust {
    background: linear-gradient(160deg,rgba(239,71,111,.13),rgba(239,71,111,.04));
    border: 2px solid rgba(239,71,111,.4);
}
.verdict-label { font-family: 'Bebas Neue', sans-serif; font-size: 1.6rem; letter-spacing: .15em; }
.verdict-prob  { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; line-height: 1.05; }
.verdict-sub   { font-size: .78rem; color: var(--muted); margin-top: 4px;
                 font-family: 'DM Mono', monospace; letter-spacing: .05em; }
.stat-pill {
    display: inline-block;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 12px; margin: 3px;
    font-size: .78rem; color: var(--text); font-family: 'DM Mono', monospace;
}
.stat-pill b { color: var(--gold); }
.factor-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}
.factor-title { font-family: 'Bebas Neue', sans-serif; font-size: 1rem; letter-spacing: .08em; margin-bottom: 4px; }
.factor-body  { font-size: .8rem; color: var(--muted); line-height: 1.55; }
.pos-badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-family: 'Bebas Neue', sans-serif; font-size: .9rem; letter-spacing: .1em;
    background: rgba(240,165,0,.15); border: 1px solid rgba(240,165,0,.4);
    color: var(--gold); margin-bottom: 4px;
}
.info-box {
    background: rgba(17,138,178,.1); border: 1px solid rgba(17,138,178,.35);
    border-radius: 8px; padding: 12px 16px; font-size: .81rem;
    color: #90cce8; line-height: 1.5; margin: 10px 0;
}
.warning-box {
    background: rgba(240,165,0,.08); border: 1px solid rgba(240,165,0,.3);
    border-radius: 8px; padding: 12px 16px; font-size: .81rem;
    color: #f0c060; line-height: 1.5; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  PLOTLY BASE THEME
# ══════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#68748a", size=11),
    title_font=dict(family="Bebas Neue, sans-serif", color="#dce3ef", size=15),
    xaxis=dict(gridcolor="#181c24", linecolor="#252c3a"),
    yaxis=dict(gridcolor="#181c24", linecolor="#252c3a"),
)
GOLD="#f0a500"; GREEN="#06d6a0"; RED="#ef476f"; BLUE="#118ab2"; MUTED="#68748a"

# ══════════════════════════════════════════════════════════════════
#  POSITION CONFIG — dynamic slider ranges & defaults
#  Each stat: (min, max, default, step, label)
# ══════════════════════════════════════════════════════════════════
POSITIONS = {
    "Point Guard (PG)": {
        "abbr":"PG","emoji":"🎯","color":BLUE,
        "desc":"Floor general. Distributes the ball, pushes pace, and creates for others.",
        "stats":{
            "GP": (10,82,65,1,"Games Played"),
            "MIN":(5,40,26,0.5,"Minutes / Game"),
            "PTS":(2,35,14,0.5,"Points / Game"),
            "REB":(0,8,3.5,0.5,"Rebounds / Game"),
            "AST":(2,14,6.5,0.5,"Assists / Game"),
            "STL":(0,3.5,1.4,0.1,"Steals / Game"),
            "BLK":(0,1.5,0.2,0.1,"Blocks / Game"),
            "TOV":(0.5,6,2.8,0.1,"Turnovers / Game"),
            "FG%":(30,60,44,0.5,"Field Goal %"),
        },
    },
    "Shooting Guard (SG)": {
        "abbr":"SG","emoji":"🎯","color":"#8b5cf6",
        "desc":"Primary scorer off the wing. Strong off-ball movement and shooting.",
        "stats":{
            "GP": (10,82,62,1,"Games Played"),
            "MIN":(5,40,24,0.5,"Minutes / Game"),
            "PTS":(2,40,15,0.5,"Points / Game"),
            "REB":(0,8,3.5,0.5,"Rebounds / Game"),
            "AST":(0.5,9,3.5,0.5,"Assists / Game"),
            "STL":(0,3.5,1.2,0.1,"Steals / Game"),
            "BLK":(0,2.0,0.3,0.1,"Blocks / Game"),
            "TOV":(0.3,5,2.2,0.1,"Turnovers / Game"),
            "FG%":(30,55,43,0.5,"Field Goal %"),
        },
    },
    "Small Forward (SF)": {
        "abbr":"SF","emoji":"⚖️","color":GREEN,
        "desc":"The versatile all-rounder. Balanced across all stat categories.",
        "stats":{
            "GP": (10,82,63,1,"Games Played"),
            "MIN":(5,40,24,0.5,"Minutes / Game"),
            "PTS":(2,38,13,0.5,"Points / Game"),
            "REB":(1,12,5.5,0.5,"Rebounds / Game"),
            "AST":(0.5,8,2.8,0.5,"Assists / Game"),
            "STL":(0,3.5,1.1,0.1,"Steals / Game"),
            "BLK":(0,3.0,0.5,0.1,"Blocks / Game"),
            "TOV":(0.3,5,2.0,0.1,"Turnovers / Game"),
            "FG%":(30,60,45,0.5,"Field Goal %"),
        },
    },
    "Power Forward (PF)": {
        "abbr":"PF","emoji":"💪","color":GOLD,
        "desc":"Interior presence with range. High-value rebounder and scorer near the paint.",
        "stats":{
            "GP": (10,82,60,1,"Games Played"),
            "MIN":(5,40,22,0.5,"Minutes / Game"),
            "PTS":(2,35,12,0.5,"Points / Game"),
            "REB":(2,16,7.5,0.5,"Rebounds / Game"),
            "AST":(0.3,6,2.0,0.5,"Assists / Game"),
            "STL":(0,2.5,0.9,0.1,"Steals / Game"),
            "BLK":(0,4.0,1.0,0.1,"Blocks / Game"),
            "TOV":(0.3,4,1.8,0.1,"Turnovers / Game"),
            "FG%":(35,68,49,0.5,"Field Goal %"),
        },
    },
    "Center (C)": {
        "abbr":"C","emoji":"🏰","color":RED,
        "desc":"Paint anchor. Dominates boards and rim protection. High FG% due to proximity.",
        "stats":{
            "GP": (10,82,58,1,"Games Played"),
            "MIN":(5,40,22,0.5,"Minutes / Game"),
            "PTS":(2,32,12,0.5,"Points / Game"),
            "REB":(3,18,9.0,0.5,"Rebounds / Game"),
            "AST":(0.2,5,1.5,0.5,"Assists / Game"),
            "STL":(0,2.0,0.7,0.1,"Steals / Game"),
            "BLK":(0.2,5.0,1.6,0.1,"Blocks / Game"),
            "TOV":(0.3,4,1.7,0.1,"Turnovers / Game"),
            "FG%":(40,75,54,0.5,"Field Goal %"),
        },
    },
}

# ══════════════════════════════════════════════════════════════════
#  LEAGUE AVERAGE REFERENCE DATA (by position)
# ══════════════════════════════════════════════════════════════════
LEAGUE_AVG = {
    "Point Guard (PG)":    dict(GP=62,MIN=25,PTS=13,REB=3.5,AST=6.0,STL=1.3,BLK=0.2,TOV=2.7,FG_pct=43),
    "Shooting Guard (SG)": dict(GP=60,MIN=24,PTS=14,REB=3.5,AST=3.2,STL=1.1,BLK=0.3,TOV=2.1,FG_pct=43),
    "Small Forward (SF)":  dict(GP=60,MIN=24,PTS=12,REB=5.2,AST=2.6,STL=1.0,BLK=0.5,TOV=1.9,FG_pct=45),
    "Power Forward (PF)":  dict(GP=58,MIN=22,PTS=11,REB=7.0,AST=1.8,STL=0.8,BLK=0.9,TOV=1.7,FG_pct=48),
    "Center (C)":          dict(GP=56,MIN=22,PTS=11,REB=8.5,AST=1.4,STL=0.6,BLK=1.5,TOV=1.6,FG_pct=54),
}

# ══════════════════════════════════════════════════════════════════
#  MOCK ML MODEL — predict_survival()
#  Position-aware weighted scoring that simulates a logistic model.
#  All edge cases wrapped in try/except to prevent app crash.
# ══════════════════════════════════════════════════════════════════
def predict_survival(stats: dict, position_key: str) -> dict:
    """
    Simulates a Logistic Regression prediction using a deterministic
    weighted scoring formula.

    Returns:
        probability (float)  : 0–1 survival probability
        prediction  (int)    : 1 = survive, 0 = bust
        factors     (list)   : [(title, delta, explanation), ...]
        score_raw   (float)  : pre-sigmoid composite score
        error       (str)    : present only if something went wrong
    """
    try:
        pos = POSITIONS[position_key]["abbr"]

        # ── Safe type coercion ──
        def to_float(val, fallback=0.0):
            try:
                v = float(val)
                return v if math.isfinite(v) else fallback
            except (TypeError, ValueError):
                return fallback

        gp    = to_float(stats.get("GP"),  60)
        mn    = to_float(stats.get("MIN"), 20)
        pts   = to_float(stats.get("PTS"), 10)
        reb   = to_float(stats.get("REB"),  5)
        ast   = to_float(stats.get("AST"),  3)
        stl   = to_float(stats.get("STL"),  1)
        blk   = to_float(stats.get("BLK"), 0.5)
        tov   = to_float(stats.get("TOV"),  2)
        fgpct = to_float(stats.get("FG%"), 45)

        # ── Safe division ──
        def safe_div(a, b, fallback=0.0):
            try:
                if b == 0 or not math.isfinite(b):
                    return fallback
                result = a / b
                return result if math.isfinite(result) else fallback
            except Exception:
                return fallback

        # ── Derived ratios ──
        ast_tov = safe_div(ast, max(tov, 0.01), fallback=1.5)
        pts_min = safe_div(pts, max(mn,  0.01), fallback=0.5)
        reb_min = safe_div(reb, max(mn,  0.01), fallback=0.25)

        # ── Base score (intercept) ──
        score = -1.20

        # 1. Games Played — durability
        gp_norm = (gp - 41) / 20.0
        score  += 0.70 * gp_norm

        # 2. Minutes — rotation security
        mn_norm = (mn - 18) / 8.0
        score  += 0.45 * mn_norm

        # 3. Scoring efficiency
        score  += 0.60 * (pts_min - 0.55)

        # 4. FG% — shooting quality
        score  += 0.04 * (fgpct - 45)

        # 5. Position-specific contributions
        if pos in ("PG", "SG"):
            score += 0.55 * (ast_tov - 1.8)
            score += 0.25 * (stl - 1.0)
            score -= 0.30 * max(0, tov - 3.5)
            score += 0.20 * (ast - 4.0)

        elif pos == "SF":
            score += 0.30 * (ast_tov - 1.5)
            score += 0.35 * (reb_min - 0.22)
            score += 0.25 * (stl - 0.9)
            score += 0.20 * (blk - 0.4)

        else:  # PF / C
            score += 0.50 * reb_min
            score += 0.45 * (blk - 0.8)
            score += 0.015 * (fgpct - 50)
            score -= 0.20 * max(0, tov - 3.0)

        # 6. Universal red flags
        pts_safe = max(pts, 0.01)
        tov_pts_ratio = safe_div(tov, pts_safe, fallback=0.0)
        score -= 0.20 * max(0, tov_pts_ratio - 0.25)
        if gp < 20:
            score -= 0.80

        # 7. Clamp + sigmoid
        score   = max(-6.0, min(6.0, score))
        prob    = 1.0 / (1.0 + math.exp(-score))

        # ── Build factor explanations ──
        factors = []

        # Games Played
        if gp >= 70:
            factors.append(("✅ Elite Durability", +1,
                f"{int(gp)} games played — iron-man availability. Teams reward presence."))
        elif gp >= 55:
            factors.append(("🟡 Moderate Availability", 0,
                f"{int(gp)} games — decent, but not yet a rotational lock."))
        else:
            factors.append(("⚠️ Low Games Played", -1,
                f"Only {int(gp)} games — small sample size, potential injury concern."))

        # Scoring efficiency
        if pts_min >= 0.75:
            factors.append(("✅ High Scoring Efficiency", +1,
                f"{pts:.1f} pts / {mn:.1f} min = {pts_min:.2f} pts per minute — elite rate."))
        elif pts_min >= 0.50:
            factors.append(("🟡 Average Scoring Rate", 0,
                f"{pts_min:.2f} pts/min — functional contributor but not a standout scorer."))
        else:
            factors.append(("⚠️ Low Scoring Output", -1,
                f"{pts:.1f} pts in {mn:.1f} min — needs more offensive production to stick."))

        # Shooting
        if fgpct >= 50:
            factors.append(("✅ Efficient Shooter", +1,
                f"{fgpct:.1f}% FG — above average efficiency, maximises possessions."))
        elif fgpct >= 42:
            factors.append(("🟡 League-Average FG%", 0,
                f"{fgpct:.1f}% — acceptable, but a few percentage points of growth needed."))
        else:
            factors.append(("⚠️ Poor Shooting Efficiency", -1,
                f"{fgpct:.1f}% FG — below league average; hurts team offensive rating."))

        # Position-specific key signal
        if pos in ("PG", "SG"):
            if ast_tov >= 2.2:
                factors.append(("✅ Excellent Ball Security", +1,
                    f"AST/TOV = {ast_tov:.2f} — great care with the ball for a guard."))
            elif ast_tov >= 1.5:
                factors.append(("🟡 Decent Playmaking Ratio", 0,
                    f"AST/TOV = {ast_tov:.2f} — solid but has room to improve decision-making."))
            else:
                factors.append(("⚠️ Turnover-Prone Guard", -1,
                    f"AST/TOV = {ast_tov:.2f} — critical weakness; turnover liability for a PG/SG."))
        else:
            if reb_min >= 0.35:
                factors.append(("✅ Dominant Rebounder", +1,
                    f"{reb:.1f} reb / {mn:.1f} min = {reb_min:.2f}/min — elite glass presence."))
            elif reb_min >= 0.20:
                factors.append(("🟡 Adequate Rebounding", 0,
                    f"{reb_min:.2f} reb/min — contributes but not a standout interior force."))
            else:
                factors.append(("⚠️ Weak Rebounding", -1,
                    f"Only {reb:.1f} boards in {mn:.1f} min — major concern for a big man."))

        return dict(
            probability = round(max(0.0, min(1.0, prob)), 4),
            prediction  = int(prob >= 0.50),
            factors     = factors,
            score_raw   = round(score, 4),
        )

    except ZeroDivisionError as e:
        return dict(probability=0.5, prediction=0, factors=[],
                    score_raw=0.0, error=f"Division error: {e}")
    except (TypeError, ValueError) as e:
        return dict(probability=0.5, prediction=0, factors=[],
                    score_raw=0.0, error=f"Input type error: {e}")
    except Exception as e:
        return dict(probability=0.5, prediction=0, factors=[],
                    score_raw=0.0, error=f"Unexpected error in model: {e}")

# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
c_icon, c_title = st.columns([0.07, 0.93])
with c_icon:
    st.markdown("<div style='font-size:3rem;line-height:1;padding-top:8px'>🏀</div>",
                unsafe_allow_html=True)
with c_title:
    st.markdown("""
    <div class='nba-header'>NBA Rookie Career
    <span style='color:#f0a500'>Longevity Predictor</span></div>
    <div class='nba-sub'>
    position-aware mock ml model &nbsp;·&nbsp;
    single-file app &nbsp;·&nbsp; no external .pkl required
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — PLAYER BUILDER
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='font-family:"Bebas Neue",sans-serif;font-size:1.4rem;
    letter-spacing:.1em;color:#f0a500;margin-bottom:4px'>
    🏀 Player Builder
    </div>
    <div style='font-size:.7rem;color:#68748a;margin-bottom:14px;
    font-family:"DM Mono",monospace;letter-spacing:.05em'>
    CONFIGURE ROOKIE PROFILE
    </div>
    """, unsafe_allow_html=True)

    # ── Identity inputs ──
    raw_name = st.text_input(
        "PLAYER NAME",
        value=st.session_state.get("player_name_input", "Rookie Player"),
        placeholder="e.g. Bronny James",
        key="player_name_input",
    )
    player_name = (raw_name.strip() or "Rookie Player")

    position_key = st.selectbox(
        "POSITION / ROLE",
        options=list(POSITIONS.keys()),
        index=list(POSITIONS.keys()).index(st.session_state.get("position_key", list(POSITIONS.keys())[0])),
        key="position_key",
    )

    pos_cfg  = POSITIONS[position_key]
    pos_abbr = pos_cfg["abbr"]
    pos_col  = pos_cfg["color"]
    stat_cfg = pos_cfg["stats"]

    st.markdown(f"""
    <div class='info-box' style='margin:8px 0 14px'>
        <b style='color:#f0a500'>{pos_cfg["emoji"]} {pos_abbr}</b> —
        {pos_cfg["desc"]}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Dynamic stat sliders by group ──
    slider_vals = {}
    groups = {
        "⏱ PLAYING TIME": ["GP", "MIN"],
        "📊 SCORING":      ["PTS", "FG%"],
        "🤝 CONTRIBUTION": ["REB", "AST", "STL", "BLK", "TOV"],
    }

    for grp_label, keys in groups.items():
        st.markdown(f"**{grp_label}**")
        for k in keys:
            mn, mx, df, step, label = stat_cfg[k]
            slider_vals[k] = st.slider(
                label,
                min_value=float(mn),
                max_value=float(mx),
                value=float(df),
                step=float(step),
                format="%.1f",
                key=f"sl_{pos_abbr}_{k}",
            )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.button("🔮  PREDICT CAREER LONGEVITY")  # decorative — prediction is always live

# ══════════════════════════════════════════════════════════════════
#  RUN PREDICTION — always live, never crashes the app
# ══════════════════════════════════════════════════════════════════
try:
    result = predict_survival(slider_vals, position_key)
except Exception as e:
    st.error(f"⚠️ Critical prediction failure: {e}")
    st.stop()

if "error" in result:
    st.error(f"⚠️ Model error: {result['error']}  "
             f"Please adjust the sliders and try again.")
    st.stop()

prob      = result["probability"]
pred      = result["prediction"]
factors   = result["factors"]
score_raw = result["score_raw"]

# Confidence tier label
if   prob >= 0.80: conf, conf_col = "VERY HIGH CONFIDENCE", GREEN
elif prob >= 0.65: conf, conf_col = "HIGH CONFIDENCE",       GREEN
elif prob >= 0.50: conf, conf_col = "MODERATE CONFIDENCE",   GOLD
elif prob >= 0.35: conf, conf_col = "LOW CONFIDENCE",        GOLD
else:              conf, conf_col = "VERY LOW CONFIDENCE",   RED

# ══════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Prediction", "📡  Radar Analysis",
    "🔬  Score Breakdown", "📖  How It Works",
])

# ──────────────────────────────────────────────────────────────────
#  TAB 1 · PREDICTION
# ──────────────────────────────────────────────────────────────────
with tab1:
    col_l, col_r = st.columns([1, 1.05], gap="large")

    with col_l:
        box_cls = "verdict-survive" if pred == 1 else "verdict-bust"
        verdict = "✅  WILL SURVIVE 5+ YEARS" if pred == 1 else "❌  LIKELY WON'T REACH 5 YRS"
        vcolor  = GREEN if pred == 1 else RED

        st.markdown(f"""
        <div class='verdict-box {box_cls}'>
            <div style='margin-bottom:6px'>
                <span class='pos-badge'>{pos_cfg["emoji"]} {pos_abbr}</span>
            </div>
            <div style='font-size:1rem;color:#dce3ef;margin-bottom:2px'>
                {player_name}
            </div>
            <div class='verdict-label' style='color:{vcolor}'>{verdict}</div>
            <div class='verdict-prob'  style='color:{vcolor}'>{prob*100:.1f}%</div>
            <div class='verdict-sub'>probability of 5-year career survival</div>
            <div style='margin-top:10px;font-size:.72rem;color:{conf_col};
                        font-family:"DM Mono",monospace;letter-spacing:.06em'>
                {conf}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability stacked bar
        try:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=[prob], y=[""],
                orientation="h", marker_color=GREEN, name=f"Survive {prob*100:.1f}%",
                text=f"{prob*100:.1f}%" if prob > 0.08 else "",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=13, family="Bebas Neue", color="#0a0c10"),
            ))
            fig_bar.add_trace(go.Bar(
                x=[1-prob], y=[""],
                orientation="h", marker_color=RED, name=f"Bust {(1-prob)*100:.1f}%",
                text=f"{(1-prob)*100:.1f}%" if (1-prob) > 0.08 else "",
                textposition="inside", insidetextanchor="middle",
                textfont=dict(size=12, family="Bebas Neue", color="#fff"),
            ))
            fig_bar.update_layout(
                **PL, barmode="stack", height=78,
                margin=dict(t=6,b=6,l=4,r=4), showlegend=True,
                yaxis=dict(showticklabels=False,showgrid=False,zeroline=False),
                xaxis=dict(range=[0,1],showticklabels=False,showgrid=False,zeroline=False),
                legend=dict(orientation="h",y=-0.6,x=0.5,xanchor="center",font=dict(size=10)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Bar chart render error (non-critical): {e}")

        # Stat pills
        st.markdown("<div class='section-label'>Quick Stats</div>", unsafe_allow_html=True)
        labels_map = {"GP":"Games","MIN":"Min","PTS":"Pts","REB":"Reb",
                      "AST":"Ast","STL":"Stl","BLK":"Blk","TOV":"Tov","FG%":"FG%"}
        pills = "".join(
            f"<span class='stat-pill'><b>{labels_map.get(k,k)}</b>&nbsp;"
            f"{v:.1f}{'%' if k=='FG%' else ''}</span>"
            for k, v in slider_vals.items()
        )
        st.markdown(f"<div style='line-height:2.4'>{pills}</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-label'>Key Factors</div>", unsafe_allow_html=True)

        for title, delta, body in factors:
            try:
                bg  = ("rgba(6,214,160,.08)"  if delta > 0 else
                       "rgba(239,71,111,.08)" if delta < 0 else "rgba(240,165,0,.06)")
                bdr = ("rgba(6,214,160,.3)"   if delta > 0 else
                       "rgba(239,71,111,.3)"  if delta < 0 else "rgba(240,165,0,.25)")
                tc  = GREEN if delta > 0 else (RED if delta < 0 else GOLD)
                st.markdown(f"""
                <div class='factor-card' style='background:{bg};border-color:{bdr}'>
                    <div class='factor-title' style='color:{tc}'>{title}</div>
                    <div class='factor-body'>{body}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                pass  # skip bad factor silently

        st.markdown("<div class='section-label'>Derived Metrics</div>", unsafe_allow_html=True)
        try:
            tov_s  = max(float(slider_vals.get("TOV", 1)), 0.01)
            min_s  = max(float(slider_vals.get("MIN", 1)), 0.01)
            ast_tov = round(float(slider_vals.get("AST",1)) / tov_s, 2)
            pts_min = round(float(slider_vals.get("PTS",0)) / min_s, 2)
            reb_min = round(float(slider_vals.get("REB",0)) / min_s, 2)

            m1, m2, m3 = st.columns(3)
            with m1: st.metric("AST / TOV", f"{ast_tov:.2f}")
            with m2: st.metric("PTS / MIN",  f"{pts_min:.2f}")
            with m3: st.metric("REB / MIN",  f"{reb_min:.2f}")
        except Exception as e:
            st.warning(f"Derived metrics error: {e}")

# ──────────────────────────────────────────────────────────────────
#  TAB 2 · RADAR ANALYSIS
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-label'>Player vs. Position League Average</div>",
                unsafe_allow_html=True)
    try:
        avg = LEAGUE_AVG[position_key]
        radar_keys = ["GP","MIN","PTS","REB","AST","STL","BLK","TOV","FG%"]
        avg_raw = [avg["GP"],avg["MIN"],avg["PTS"],avg["REB"],
                   avg["AST"],avg["STL"],avg["BLK"],avg["TOV"],avg["FG_pct"]]
        plr_raw = [float(slider_vals.get(k, 0)) for k in radar_keys]
        radar_labels = ["GP","MIN","PTS","REB","AST","STL","BLK","TOV↓","FG%"]

        norm_plr, norm_avg = [], []
        for i, k in enumerate(radar_keys):
            mn_, mx_, _, _, _ = stat_cfg[k]
            rng = max(mx_ - mn_, 0.01)
            if k == "TOV":  # invert: lower TOV = better
                norm_plr.append(round(1 - (plr_raw[i] - mn_) / rng, 3))
                norm_avg.append(round(1 - (avg_raw[i]  - mn_) / rng, 3))
            else:
                norm_plr.append(round((plr_raw[i] - mn_) / rng, 3))
                norm_avg.append(round((avg_raw[i]  - mn_) / rng, 3))

        lc = radar_labels + [radar_labels[0]]
        np_ = norm_plr + [norm_plr[0]]
        na_ = norm_avg + [norm_avg[0]]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=na_, theta=lc, fill="toself",
            fillcolor="rgba(17,138,178,.12)",
            line=dict(color=BLUE, width=1.8, dash="dash"),
            name=f"{pos_abbr} League Avg",
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=np_, theta=lc, fill="toself",
            fillcolor="rgba(240,165,0,.16)",
            line=dict(color=GOLD, width=2.5),
            name=player_name,
        ))
        fig_r.update_layout(
            **PL,
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                                gridcolor="#252c3a", linecolor="#252c3a"),
                angularaxis=dict(tickfont=dict(color="#dce3ef",size=12,
                                               family="Bebas Neue"),
                                 linecolor="#252c3a", gridcolor="#252c3a"),
            ),
            title=f"{player_name}  vs  {pos_abbr} League Average",
            height=440, legend=dict(orientation="h",y=-0.08,x=0.5,xanchor="center"),
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Comparison table
        st.markdown("<div class='section-label'>Stat Comparison Table</div>",
                    unsafe_allow_html=True)
        rows = []
        for k, la in zip(radar_keys, avg_raw):
            try:
                pv   = float(slider_vals.get(k, 0))
                diff = round(pv - la, 2)
                rows.append({
                    "Stat": k,
                    player_name: round(pv, 1),
                    "League Avg (Position)": round(la, 1),
                    "Δ": f"{'↑' if diff>0 else '↓' if diff<0 else '='} {abs(diff):.1f}",
                })
            except Exception:
                pass
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    except KeyError as e:
        st.error(f"Position data missing: {e}")
    except Exception as e:
        st.error(f"Radar chart error: {e}")

# ──────────────────────────────────────────────────────────────────
#  TAB 3 · SCORE BREAKDOWN
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-label'>Model Score Decomposition</div>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class='info-box'>
        Raw composite score: <b style='color:#f0a500'>{score_raw:+.4f}</b>
        &nbsp;→&nbsp; Sigmoid output: <b style='color:#f0a500'>{prob*100:.2f}%</b>
        &nbsp;·&nbsp; Decision threshold: <b>50%</b>
        &nbsp;·&nbsp; Verdict: <b style='color:{"#06d6a0" if pred else "#ef476f"}'>
        {"✅ SURVIVE" if pred else "❌ BUST"}</b>
    </div>
    """, unsafe_allow_html=True)

    # Gauge
    try:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=34,color="#dce3ef",family="Bebas Neue")),
            gauge=dict(
                axis=dict(range=[0,100], tickcolor="#252c3a",
                          tickfont=dict(color="#68748a")),
                bar=dict(color=GREEN if pred else RED, thickness=0.22),
                bgcolor="#181c24", bordercolor="#252c3a",
                steps=[
                    dict(range=[0,35],  color="rgba(239,71,111,.12)"),
                    dict(range=[35,50], color="rgba(240,165,0,.08)"),
                    dict(range=[50,65], color="rgba(6,214,160,.08)"),
                    dict(range=[65,100],color="rgba(6,214,160,.16)"),
                ],
                threshold=dict(line=dict(color=GOLD,width=2),thickness=0.75,value=50),
            ),
            title=dict(text="Career Survival Probability",
                       font=dict(family="Bebas Neue",size=16,color="#dce3ef")),
        ))
        fig_g.update_layout(**PL, height=300, margin=dict(t=44,b=10,l=40,r=40))
        st.plotly_chart(fig_g, use_container_width=True)
    except Exception as e:
        st.warning(f"Gauge error: {e}")

    # Sensitivity analysis
    st.markdown("<div class='section-label'>What-If Sensitivity</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    How much does the survival probability change if we tweak each
    stat by ±5 steps, holding everything else fixed?
    </div>
    """, unsafe_allow_html=True)

    try:
        sens_keys = (["GP","MIN","PTS","FG%","AST","TOV"]
                     if pos_abbr in ("PG","SG")
                     else ["GP","MIN","PTS","FG%","REB","BLK","TOV"])
        up_deltas, dn_deltas, s_labels = [], [], []

        for k in sens_keys:
            if k not in stat_cfg:
                continue
            mn_, mx_, _, step_, lbl_ = stat_cfg[k]
            bump = step_ * 5

            up_s = dict(slider_vals); up_s[k] = min(mx_, float(slider_vals[k]) + bump)
            dn_s = dict(slider_vals); dn_s[k] = max(mn_, float(slider_vals[k]) - bump)

            ru = predict_survival(up_s, position_key)
            rd = predict_survival(dn_s, position_key)

            if "error" not in ru and "error" not in rd:
                up_deltas.append(round((ru["probability"] - prob) * 100, 2))
                dn_deltas.append(round((rd["probability"] - prob) * 100, 2))
                s_labels.append(lbl_)

        if s_labels:
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Bar(
                y=s_labels, x=up_deltas, orientation="h",
                name=f"+{bump:.1f} bump", marker_color=GREEN,
                text=[f"{d:+.1f}%" for d in up_deltas],
                textposition="outside", textfont=dict(size=10,color="#dce3ef"),
            ))
            fig_sens.add_trace(go.Bar(
                y=s_labels, x=dn_deltas, orientation="h",
                name=f"−{bump:.1f} drop", marker_color=RED,
                text=[f"{d:+.1f}%" for d in dn_deltas],
                textposition="outside", textfont=dict(size=10,color="#dce3ef"),
            ))
            fig_sens.add_vline(x=0, line_color="#252c3a", line_width=1.5)
            fig_sens.update_layout(
                **PL, barmode="group",
                title="Probability Change per Stat Tweak (±5 steps)",
                height=340, margin=dict(t=44,b=10,l=4,r=70),
                xaxis_title="Δ Probability (%)",
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Not enough valid stat perturbations to build sensitivity chart.")
    except Exception as e:
        st.error(f"Sensitivity analysis error: {e}")

# ──────────────────────────────────────────────────────────────────
#  TAB 4 · HOW IT WORKS
# ──────────────────────────────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("<div class='section-label'>Pipeline Overview</div>",
                    unsafe_allow_html=True)
        steps = [
            ("1. Identity Input",
             "Player name (text) and position (selectbox). Position choice immediately "
             "re-configures all slider ranges and defaults to keep stats realistic."),
            ("2. Position-Locked Sliders",
             "Each position has unique min/max/default per stat. A Center can't "
             "default to 10 AST; a PG can't have 16 REB as a max. Prevents "
             "physically impossible inputs from reaching the model."),
            ("3. Derived Ratios",
             "AST/TOV (playmaking efficiency), PTS/MIN (scoring rate), "
             "REB/MIN (rebounding dominance). These are more predictive "
             "than raw totals and mirror what scouts actually measure."),
            ("4. Weighted Scoring",
             "Each factor adds or subtracts a weighted delta from a base "
             "intercept of −1.2. Universal weights (GP, MIN, PTS/MIN, FG%) "
             "apply to all positions; position-specific weights are layered on top."),
            ("5. Clamp + Sigmoid",
             "Raw score clamped to [−6, +6] then passed through "
             "σ(x) = 1/(1+e^−x) → probability in [0,1]. "
             "Threshold at 0.50 gives the binary prediction."),
            ("6. Error Safety",
             "Every division is guarded with safe_div(). All type conversions "
             "use to_float() with fallbacks. The entire model is wrapped in "
             "try/except so a bad input will show st.error() — never crash the app."),
        ]
        for title, body in steps:
            st.markdown(f"""
            <div class='factor-card'>
                <div class='factor-title' style='color:#f0a500'>{title}</div>
                <div class='factor-body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-label'>Position Weight Summary</div>",
                    unsafe_allow_html=True)

        weight_table = {
            "All Positions (Universal)": [
                ("GP (z-score)",  "+0.70", "Biggest durability signal for all roles."),
                ("MIN (z-score)", "+0.45", "Earning rotation minutes is critical."),
                ("PTS/MIN",       "+0.60", "Scoring efficiency beats raw PPG."),
                ("FG%",           "+0.04", "Per percentage point above 45%."),
            ],
            "Guards (PG / SG)": [
                ("AST/TOV",       "+0.55", "Premier guard predictor."),
                ("Steals",        "+0.25", "Perimeter defense rewarded."),
                ("Excess TOV",    "−0.30", "TOV > 3.5/g penalised heavily."),
                ("Raw AST",       "+0.20", "Bonus for volume passing."),
            ],
            "Wings (SF)": [
                ("AST/TOV",       "+0.30", "Playmaking expected but not primary."),
                ("REB/MIN",       "+0.35", "Versatile boards are key."),
                ("STL + BLK",     "+0.25 / +0.20", "Two-way impact valued."),
            ],
            "Bigs (PF / C)": [
                ("REB/MIN",       "+0.50", "Core survival predictor."),
                ("Blocks",        "+0.45", "Rim protection is irreplaceable."),
                ("FG% bonus",     "+0.015/pt", "Reward proximity efficiency."),
                ("Excess TOV",    "−0.20", "Turnovers penalised for bigs too."),
            ],
        }

        for role, items in weight_table.items():
            color = (BLUE if "Guard" in role else
                     GREEN if "Wing" in role else
                     GOLD  if "Big"  in role else "#f0a500")
            st.markdown(f"""
            <div class='factor-card' style='border-left:3px solid {color};padding-left:14px;margin-bottom:6px'>
                <div class='factor-title' style='color:{color};margin-bottom:6px'>{role}</div>
            """, unsafe_allow_html=True)
            for stat, wt, note in items:
                st.markdown(f"""
                <div style='font-size:.77rem;color:#68748a;margin:2px 0'>
                    <span style='color:#dce3ef;font-family:"DM Mono",monospace'>{stat}</span>
                    &nbsp;<span style='color:{color};font-weight:600'>{wt}</span>
                    &nbsp;— {note}
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='warning-box' style='margin-top:14px'>
            ⚠️ <b>Important note:</b> This is a deterministic scoring
            function — not a statistically trained model. It is calibrated
            to produce realistic-feeling outputs but is not fitted on real
            NBA career data. To use a real model, replace the body of
            <code>predict_survival()</code> with a
            <code>joblib.load("model.pkl").predict_proba()</code> call.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<hr>
<div style='text-align:center;font-family:"DM Mono",monospace;font-size:.68rem;
            color:#68748a;letter-spacing:.06em;padding-bottom:18px'>
    NBA ROOKIE LONGEVITY PREDICTOR &nbsp;·&nbsp;
    SINGLE-FILE STREAMLIT APP &nbsp;·&nbsp;
    POSITION-AWARE MOCK ML &nbsp;·&nbsp; NO EXTERNAL .PKL REQUIRED
</div>
""", unsafe_allow_html=True)