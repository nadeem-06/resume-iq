import os
import tempfile
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from backend import (
    extract_resume_text,
    extract_resume_entities,
    parse_job_description,
    score_resume,
)

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  (dark, editorial, refined)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Root theme ── */
:root {
    --bg:        #0b0f1a;
    --surface:   #111827;
    --card:      #161d2e;
    --border:    #1f2d45;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --green:     #34d399;
    --red:       #f87171;
    --yellow:    #fbbf24;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* Global resets */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e3a5f, #1a2744);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
}
.hero-title {
    font-family: var(--font-head) !important;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, var(--accent) 60%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.7rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto;
}

/* ── Cards ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: var(--font-head);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.9rem;
}

/* ── Score circle ── */
.score-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 0 1.5rem;
}
.score-circle {
    width: 160px;
    height: 160px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: var(--font-head);
    margin-bottom: 1rem;
    position: relative;
}
.score-number {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
}
.score-denom {
    font-size: 0.9rem;
    color: var(--muted);
    font-family: var(--font-mono);
}
.verdict-pill {
    display: inline-block;
    padding: 0.5rem 1.6rem;
    border-radius: 100px;
    font-family: var(--font-head);
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.verdict-yes  { background: rgba(52,211,153,0.15); color: var(--green); border: 1px solid rgba(52,211,153,0.4); }
.verdict-no   { background: rgba(248,113,113,0.12); color: var(--red);   border: 1px solid rgba(248,113,113,0.35); }
.verdict-maybe{ background: rgba(251,191,36,0.12);  color: var(--yellow);border: 1px solid rgba(251,191,36,0.35); }

/* ── Skill pills ── */
.pill-group { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.5rem; }
.pill {
    font-family: var(--font-mono);
    font-size: 0.74rem;
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-weight: 500;
}
.pill-match  { background: rgba(52,211,153,0.12);  color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
.pill-miss   { background: rgba(248,113,113,0.1);  color: #fca5a5; border: 1px solid rgba(248,113,113,0.25); }
.pill-extra  { background: rgba(129,140,248,0.1);  color: #a5b4fc; border: 1px solid rgba(129,140,248,0.25); }

/* ── Info rows ── */
.info-row {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
}
.info-row:last-child { border-bottom: none; }
.info-key {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    min-width: 130px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.info-val {
    font-size: 0.9rem;
    color: var(--text);
    font-weight: 400;
}

/* ── Upload / textarea ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
}
textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important; }

/* ── Button ── */
[data-testid="baseButton-primary"] button,
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Warning / info banners ── */
[data-testid="stAlert"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-head);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.8rem 0 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Plotly chart bg ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER RENDERERS
# ─────────────────────────────────────────────

def render_pills(skills: list[str], pill_class: str):
    if not skills:
        st.markdown(f'<span style="color:#64748b;font-size:0.85rem;">None</span>', unsafe_allow_html=True)
        return
    pills_html = "".join(f'<span class="pill {pill_class}">{s}</span>' for s in skills)
    st.markdown(f'<div class="pill-group">{pills_html}</div>', unsafe_allow_html=True)


def score_color(score: float) -> str:
    if score >= 70:
        return "#34d399"
    elif score >= 45:
        return "#fbbf24"
    return "#f87171"


def verdict_html(score: float) -> str:
    if score >= 70:
        return '<span class="verdict-pill verdict-yes">✓ Shortlisted</span>'
    elif score >= 45:
        return '<span class="verdict-pill verdict-maybe">~ Under Review</span>'
    return '<span class="verdict-pill verdict-no">✗ Not Shortlisted</span>'


def make_gauge(score: float) -> go.Figure:
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": color, "family": "Syne"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#1f2d45",
                "tickfont": {"color": "#64748b", "size": 10, "family": "DM Mono"},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#161d2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  45], "color": "rgba(248,113,113,0.08)"},
                {"range": [45, 70], "color": "rgba(251,191,36,0.08)"},
                {"range": [70, 100],"color": "rgba(52,211,153,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        height=220,
        font={"family": "DM Sans"},
    )
    return fig


def make_score_bar_chart(skill_s, exp_s, edu_s) -> go.Figure:
    """Horizontal bar chart for score breakdown."""
    categories = ["Education<br>(max 15)", "Experience<br>(max 25)", "Skills<br>(max 60)"]
    values     = [edu_s, exp_s, skill_s]
    maxes      = [15, 25, 60]
    colors     = ["#34d399", "#818cf8", "#38bdf8"]

    fig = go.Figure()

    # Background bars (max capacity)
    fig.add_trace(go.Bar(
        y=categories, x=maxes,
        orientation="h",
        marker_color=["rgba(52,211,153,0.08)", "rgba(129,140,248,0.08)", "rgba(56,189,248,0.08)"],
        showlegend=False,
        name="Max",
        hoverinfo="skip",
    ))

    # Actual score bars
    fig.add_trace(go.Bar(
        y=categories, x=values,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1f} pts" for v in values],
        textposition="outside",
        textfont={"family": "Syne", "size": 13, "color": "#e2e8f0"},
        showlegend=False,
        name="Score",
    ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=15, b=0),
        height=220,
        xaxis=dict(
            gridcolor="#1f2d45", zeroline=False,
            tickfont={"color": "#64748b", "family": "DM Mono", "size": 10},
            range=[0, max(maxes) * 1.3],
        ),
        yaxis=dict(
            tickfont={"color": "#94a3b8", "family": "DM Sans", "size": 11},
        ),
        bargap=0.35,
    )
    return fig


def make_eval_metrics_bar_chart(metrics: dict, candidate_exp: float, required_exp: float) -> go.Figure:
    """Grouped bar chart for evaluation metrics."""

    precision = metrics["precision"] * 100
    recall    = metrics["recall"] * 100
    f1        = metrics["f1_score"] * 100
    smt       = metrics["skill_match_rate"]

    req_exp = required_exp if required_exp > 0 else 1
    exp_fit = min(candidate_exp / req_exp, 1.0) * 100 if req_exp > 0 else (100.0 if candidate_exp > 0 else 50.0)

    metric_names = ["Precision", "Recall", "F1 Score", "Skill Match Rate", "Experience Fit"]
    metric_values = [precision, recall, f1, smt, exp_fit]

    bar_colors = []
    for val in metric_values:
        if val >= 70:
            bar_colors.append("rgba(52,211,153,0.85)")
        elif val >= 40:
            bar_colors.append("rgba(251,191,36,0.85)")
        else:
            bar_colors.append("rgba(248,113,113,0.80)")

    fig = go.Figure()

    # Reference line at 100%
    fig.add_trace(go.Bar(
        x=metric_names,
        y=[100] * len(metric_names),
        marker_color=[
            "rgba(52,211,153,0.06)", "rgba(129,140,248,0.06)", "rgba(56,189,248,0.06)",
            "rgba(251,191,36,0.06)", "rgba(52,211,153,0.06)"
        ],
        showlegend=False,
        hoverinfo="skip",
        name="Max",
    ))

    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color=bar_colors,
        marker_line_width=0,
        text=[f"{v:.1f}%" for v in metric_values],
        textposition="outside",
        textfont={"family": "Syne", "size": 12, "color": "#e2e8f0"},
        showlegend=False,
        name="Score",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=25, b=0),
        height=300,
        yaxis=dict(
            gridcolor="#1f2d45",
            zeroline=False,
            tickfont={"color": "#64748b", "family": "DM Mono", "size": 10},
            range=[0, 130],
            ticksuffix="%",
        ),
        xaxis=dict(
            tickfont={"color": "#94a3b8", "family": "DM Sans", "size": 11},
        ),
        bargap=0.3,
    )
    return fig


def make_radar(metrics: dict) -> go.Figure:
    cats = ["Skill Precision", "Skill Recall", "F1 Score",
            "Experience Fit", "Coverage"]

    precision     = metrics["precision"]
    recall        = metrics["recall"]
    f1            = metrics["f1_score"]
    req_exp       = metrics.get("required_experience_raw", 0)
    cand_exp_raw  = metrics.get("candidate_experience_raw", 0)
    exp_fit       = min(cand_exp_raw / req_exp, 1.0) if req_exp > 0 else (1.0 if cand_exp_raw > 0 else 0.5)
    coverage      = metrics["coverage_score"] / 100

    values = [precision, recall, f1, exp_fit, coverage]
    values_pct = [round(v * 100, 1) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(56,189,248,0.1)",
        line=dict(color="#38bdf8", width=2),
        name="Resume",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[100] * (len(cats) + 1),
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(31,45,69,0.3)",
        line=dict(color="#1f2d45", width=1, dash="dot"),
        name="Perfect",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont={"color": "#64748b", "size": 8, "family": "DM Mono"},
                gridcolor="#1f2d45", linecolor="#1f2d45",
            ),
            angularaxis=dict(
                tickfont={"color": "#94a3b8", "size": 10, "family": "DM Sans"},
                linecolor="#1f2d45", gridcolor="#1f2d45",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#64748b", family="DM Sans", size=10)),
        margin=dict(l=30, r=30, t=30, b=30),
        height=300,
    )
    return fig


def make_skill_donut(matched: int, missing: int, extra: int) -> go.Figure:
    labels = ["Matched", "Missing", "Extra Skills"]
    values = [matched, missing, extra]
    colors = ["rgba(52,211,153,0.85)", "rgba(248,113,113,0.75)", "rgba(129,140,248,0.75)"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62,
        marker=dict(colors=colors, line=dict(color="#161d2e", width=2)),
        textfont=dict(family="DM Sans", size=11, color="#e2e8f0"),
        hovertemplate="%{label}: %{value} skills<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=10),
        height=220,
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", family="DM Sans", size=10),
                    orientation="v", x=1.02, y=0.5),
        annotations=[dict(
            text=f"<b>{matched}</b><br><span style='font-size:10px'>matched</span>",
            x=0.5, y=0.5, font_size=18, showarrow=False,
            font=dict(color="#34d399", family="Syne"),
        )],
    )
    return fig


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧠 NLP · BERT · Semantic Matching</div>
    <h1 class="hero-title">ResumeIQ</h1>
    <p class="hero-sub">AI-powered resume screening with precision evaluation metrics</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Input area ──────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="card-title">📄 Upload Resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.markdown(
            f'<div style="color:#34d399;font-family:\'DM Mono\',monospace;font-size:0.8rem;margin-top:0.4rem;">'
            f'✓ {uploaded_file.name} &nbsp;·&nbsp; {uploaded_file.size/1024:.1f} KB</div>',
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown('<div class="card-title">📝 Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area(
        "Paste job description",
        height=180,
        placeholder="Paste the full job description here — skills, experience requirements, responsibilities...",
        label_visibility="collapsed",
    )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    analyze_clicked = st.button("🚀 Analyze Resume", type="primary")

st.markdown("---")

# ── Analysis ────────────────────────────────
if analyze_clicked:
    if not uploaded_file or not jd_text.strip():
        st.warning("⚠️  Please upload a PDF resume **and** enter a job description before analyzing.")
        st.stop()

    with st.spinner("Running NLP pipeline — extracting entities, matching skills, computing metrics…"):
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        resume_text  = extract_resume_text(tmp_path)
        resume_data  = extract_resume_entities(resume_text)
        job_data     = parse_job_description(jd_text)
        final_score, breakdown = score_resume(resume_data, job_data)
        os.unlink(tmp_path)

    # Enrich metrics with raw exp values for radar chart
    metrics = breakdown["evaluation_metrics"]
    metrics["candidate_experience_raw"] = resume_data["experience_years"]
    metrics["required_experience_raw"]  = job_data["min_experience"]

    sc = score_color(final_score)

    # ══════════════════════════════════════════
    #  ROW 1 — Score + Breakdown bar + Radar
    # ══════════════════════════════════════════
    st.markdown('<div class="section-label">📊 Results Overview</div>', unsafe_allow_html=True)

    r1a, r1b, r1c = st.columns([1, 1.2, 1.2], gap="large")

    with r1a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Overall Score</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(final_score), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div style="text-align:center;margin-top:-0.5rem">{verdict_html(final_score)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with r1b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Score Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(
            make_score_bar_chart(
                breakdown["skill_score"],
                breakdown["experience_score"],
                breakdown["education_score"],
            ),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Performance Radar</div>', unsafe_allow_html=True)
        st.plotly_chart(make_radar(metrics), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    #  ROW 2 — Evaluation Metrics Bar Chart
    # ══════════════════════════════════════════
    st.markdown('<div class="section-label">📐 Evaluation Metrics</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Precision · Recall · F1 · Skill Match Rate · Experience Fit</div>', unsafe_allow_html=True)
    st.plotly_chart(
        make_eval_metrics_bar_chart(metrics, resume_data["experience_years"], job_data["min_experience"]),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Compact summary row below chart
    exp_gap    = metrics["experience_gap"]
    gap_prefix = "+" if exp_gap >= 0 else ""
    gap_color  = "#34d399" if exp_gap >= 0 else "#f87171"

    st.markdown(
        f'<div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:0.8rem;padding-top:0.8rem;border-top:1px solid #1f2d45;">'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;">Resume Skills: '
        f'<b style="color:#38bdf8">{metrics["total_resume_skills"]}</b></span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;">JD Required: '
        f'<b style="color:#818cf8">{metrics["total_required_skills"]}</b></span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;">Matched: '
        f'<b style="color:#34d399">{metrics["total_matched_skills"]}</b></span>'
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#64748b;">Experience Gap: '
        f'<b style="color:{gap_color}">{gap_prefix}{exp_gap} yrs</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Metric explanations
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
**Precision** — Of all skills listed in the resume, what fraction are actually required by the job?  
High precision = the candidate's skills are highly relevant to this specific role.

**Recall** — Of all skills the job requires, what fraction does the resume cover?  
High recall = the candidate covers most of what the employer needs.

**F1 Score** — The harmonic mean of Precision and Recall (0–100%).  
Best single metric for skill match quality. Penalises both missing required skills and irrelevant extras.

**Skill Match Rate** — Same as Recall expressed as a percentage. Directly answers "how many JD skills did this resume tick?"

**Experience Fit** — How well the candidate's experience meets the requirement (capped at 100%).

**Score weights:** Skills 60 pts · Experience 25 pts · Education 15 pts = 100 pts total.  
Shortlisted threshold: ≥ 70 pts.
        """)

    # ══════════════════════════════════════════
    #  ROW 3 — Skill Analysis
    # ══════════════════════════════════════════
    st.markdown('<div class="section-label">🧠 Skill Analysis</div>', unsafe_allow_html=True)

    sa1, sa2 = st.columns([1, 1.6], gap="large")

    with sa1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Skill Distribution</div>', unsafe_allow_html=True)
        matched_c = len(breakdown["matched_skills"])
        missing_c = len(breakdown["missing_skills"])
        extra_c   = len(breakdown.get("extra_skills", []))
        st.plotly_chart(
            make_skill_donut(matched_c, missing_c, extra_c),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with sa2:
        tab1, tab2, tab3 = st.tabs(["✅ Matched", "❌ Missing", "➕ Extra"])
        with tab1:
            st.markdown(
                f'<div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">'
                f'{matched_c} skills match the job description</div>',
                unsafe_allow_html=True,
            )
            render_pills(breakdown["matched_skills"], "pill-match")
        with tab2:
            st.markdown(
                f'<div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">'
                f'{missing_c} required skills not found in resume</div>',
                unsafe_allow_html=True,
            )
            render_pills(breakdown["missing_skills"], "pill-miss")
        with tab3:
            st.markdown(
                f'<div style="color:#64748b;font-size:0.8rem;margin-bottom:0.5rem">'
                f'{extra_c} additional skills beyond what JD requires</div>',
                unsafe_allow_html=True,
            )
            render_pills(breakdown.get("extra_skills", []), "pill-extra")

    # ══════════════════════════════════════════
    #  ROW 4 — Candidate Profile + JD Summary
    # ══════════════════════════════════════════
    st.markdown('<div class="section-label">📋 Profile & Job Summary</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2, gap="large")

    with p1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Candidate Profile</div>', unsafe_allow_html=True)

        def info_row(key, val):
            if val:
                st.markdown(
                    f'<div class="info-row">'
                    f'<span class="info-key">{key}</span>'
                    f'<span class="info-val">{val}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        info_row("Name",        resume_data.get("name") or "—")
        info_row("Email",       resume_data.get("email") or "—")
        info_row("Phone",       resume_data.get("phone") or "—")
        info_row("Experience",  f"{resume_data['experience_years']} years")
        info_row("Designation", resume_data.get("designation") or "—")

        edu = resume_data.get("education", [])
        if edu:
            st.markdown(
                f'<div class="info-row">'
                f'<span class="info-key">Education</span>'
                f'<span class="info-val">{edu[0]}'
                + (f'<br><span style="color:#64748b;font-size:0.8rem">+{len(edu)-1} more</span>' if len(edu) > 1 else '')
                + '</span></div>',
                unsafe_allow_html=True,
            )

        companies = resume_data.get("companies", [])
        if companies:
            info_row("Companies", ", ".join(companies[:3]) + ("…" if len(companies) > 3 else ""))

        st.markdown('</div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Job Requirements</div>', unsafe_allow_html=True)
        info_row("Min Experience", f"{job_data['min_experience']} years")
        info_row("Required Skills", f"{len(job_data['required_skills'])} skills identified")
        st.markdown(
            '<div style="margin-top:0.8rem"><div class="metric-label" style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;">All Required Skills</div></div>',
            unsafe_allow_html=True,
        )
        render_pills(job_data["required_skills"], "pill-extra")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ──────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#64748b;font-family:\'DM Mono\',monospace;font-size:0.72rem;">'
        'ResumeIQ · NLP + BERT + Streamlit · Rule-based fallback when model unavailable'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    # Empty-state hint
    st.markdown("""
<div style="text-align:center;padding:3rem 1rem;color:#64748b">
    <div style="font-size:3rem;margin-bottom:1rem">📄</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;color:#475569;margin-bottom:0.5rem">
        Upload a resume and paste a job description to begin
    </div>
    <div style="font-size:0.85rem">
        The system will extract skills, experience, and education — then score the match.
    </div>
</div>
""", unsafe_allow_html=True)