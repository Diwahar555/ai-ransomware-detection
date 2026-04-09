"""
training_monitor.py  —  Live Training Progress Monitor
Run with:  streamlit run app/training_monitor.py
"""

import json
import os
import sys
import time

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROGRESS_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "training_progress.json"
)

MODEL_ORDER = ["Random Forest", "XGBoost", "SVM", "DNN", "LSTM", "Meta-Ensemble"]
MODEL_ICONS = {
    "Random Forest":  "🌲",
    "XGBoost":        "⚡",
    "SVM":            "🔵",
    "DNN":            "🧠",
    "LSTM":           "🔄",
    "Meta-Ensemble":  "🎯",
}
MODEL_DESC = {
    "Random Forest":  "200 trees · depth 20 · balanced weights",
    "XGBoost":        "300 estimators · hist method · learning 0.05",
    "SVM":            "Calibrated LinearSVC · 30K subset · sigmoid",
    "DNN":            "256→128→64→32→1 · batch norm · dropout",
    "LSTM":           "128+64 units · 2-layer recurrent · dropout",
    "Meta-Ensemble":  "Logistic stacking of all 5 model outputs",
}

st.set_page_config(
    page_title="Live Training Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1528 40%, #0a1520 100%);
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 1300px; padding-top: 1.5rem; }

/* Hero */
.hero {
    background: linear-gradient(135deg,
        rgba(99,179,237,0.08) 0%,
        rgba(154,117,234,0.06) 50%,
        rgba(72,199,142,0.06) 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #63b3ed; margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #e2e8f0 0%, #63b3ed 60%, #9a75ea 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem 0;
}
.hero-sub { color: #8097b1; font-size: 0.98rem; line-height: 1.6; }

/* Status pill */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.82rem; font-weight: 600; letter-spacing: 0.04em;
    margin-top: 1rem;
}
.status-starting { background: rgba(99,179,237,0.15); border: 1px solid rgba(99,179,237,0.3); color: #63b3ed; }
.status-training { background: rgba(236,190,60,0.15); border: 1px solid rgba(236,190,60,0.3); color: #ecbe3c; }
.status-done     { background: rgba(72,199,142,0.15); border: 1px solid rgba(72,199,142,0.3); color: #48c78e; }
.status-error    { background: rgba(252,110,110,0.15); border: 1px solid rgba(252,110,110,0.3); color: #fc6e6e; }
.pulse { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         background: currentColor; animation: pulse 1.4s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(0.7)} }

/* Overall progress bar */
.progress-wrap {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 1.5rem 2rem; margin-bottom: 1.2rem;
}
.progress-label { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.7rem; }
.progress-label-left { font-size: 0.82rem; color: #8097b1; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.progress-label-right { font-size: 1.4rem; font-weight: 800; font-family: 'JetBrains Mono', monospace;
                         background: linear-gradient(135deg, #63b3ed, #9a75ea);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.progress-track { background: rgba(255,255,255,0.07); border-radius: 99px; height: 12px; overflow: hidden; }
.progress-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, #63b3ed 0%, #9a75ea 60%, #48c78e 100%);
    transition: width 0.6s ease;
    box-shadow: 0 0 12px rgba(99,179,237,0.5);
}

/* Model cards grid */
.model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.model-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.model-card.state-pending  { border-color: rgba(255,255,255,0.07); }
.model-card.state-training { border-color: rgba(236,190,60,0.4); background: rgba(236,190,60,0.04);
                              box-shadow: 0 0 20px rgba(236,190,60,0.08); }
.model-card.state-done     { border-color: rgba(72,199,142,0.35); background: rgba(72,199,142,0.04); }
.model-card-bar {
    position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 18px 18px 0 0;
}
.model-card-bar.state-pending  { background: rgba(255,255,255,0.1); }
.model-card-bar.state-training {
    background: linear-gradient(90deg, #ecbe3c, #f6a821);
    animation: shimmer 1.5s infinite;
    background-size: 200% 100%;
}
@keyframes shimmer { 0%{background-position:200%} 100%{background-position:-200%} }
.model-card-bar.state-done     { background: linear-gradient(90deg, #48c78e, #36d399); }

.model-icon   { font-size: 1.8rem; margin-bottom: 0.5rem; }
.model-name   { font-size: 1.05rem; font-weight: 700; color: #e2e8f0; }
.model-desc   { font-size: 0.78rem; color: #637a93; margin-top: 0.2rem; line-height: 1.5; }
.model-status-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-size: 0.75rem; font-weight: 600; border-radius: 999px;
    padding: 0.25rem 0.7rem; margin-top: 0.7rem;
}
.badge-pending  { background: rgba(255,255,255,0.06); color: #637a93; }
.badge-training { background: rgba(236,190,60,0.18); color: #ecbe3c; }
.badge-done     { background: rgba(72,199,142,0.18); color: #48c78e; }

.model-metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.4rem; margin-top: 0.8rem; }
.metric-mini { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 0.45rem 0.5rem; text-align: center; }
.metric-mini-label { font-size: 0.64rem; color: #637a93; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-mini-value { font-size: 0.9rem; font-weight: 700; color: #e2e8f0; font-family: 'JetBrains Mono', monospace; margin-top: 0.1rem; }
.metric-mini-value.highlight { color: #63b3ed; }
.elapsed { font-size: 0.72rem; color: #526070; margin-top: 0.5rem; }

/* Results table section */
.results-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.results-title { font-size: 0.78rem; color: #63b3ed; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 700; margin-bottom: 1rem; }
.results-table { width: 100%; border-collapse: collapse; }
.results-table th {
    text-align: left; font-size: 0.72rem; color: #637a93;
    text-transform: uppercase; letter-spacing: 0.10em;
    padding: 0.45rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.07);
    font-weight: 600;
}
.results-table td {
    padding: 0.55rem 0.8rem;
    font-size: 0.9rem; font-family: 'JetBrains Mono', monospace;
    color: #c8d8e8; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.results-table td:first-child { font-family: 'Inter', sans-serif; font-weight: 600; color: #e2e8f0; }
.results-table tr:hover td { background: rgba(99,179,237,0.04); }
.best-row td { color: #48c78e !important; }
.best-row td:first-child { color: #48c78e !important; }

/* Log terminal */
.log-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    max-height: 260px;
    overflow-y: auto;
}
.log-line { padding: 0.18rem 0; line-height: 1.6; }
.log-line .ts { color: #3d5266; margin-right: 0.6rem; }
.log-line .msg { color: #8fb8d4; }
.log-line .msg.done { color: #48c78e; }
.log-line .msg.error { color: #fc6e6e; }
.log-line .msg.start { color: #ecbe3c; }

/* No file */
.no-file {
    text-align: center; padding: 4rem 2rem;
    color: #526070; font-size: 1rem; border-radius: 20px;
    border: 2px dashed rgba(255,255,255,0.07);
}
.no-file .big { font-size: 3rem; margin-bottom: 1rem; }

/* Streamlit overrides */
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stMarkdown { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


def fmt_pct(v):
    try:
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "—"


def elapsed_str(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def load_progress():
    if not os.path.exists(PROGRESS_LOG):
        return None
    try:
        with open(PROGRESS_LOG, "r") as f:
            return json.load(f)
    except Exception:
        return None


def render_hero(data):
    status = data.get("status", "starting")
    elapsed = time.time() - data.get("started_at", time.time())
    cur = data.get("current_model") or "—"
    idx = data.get("current_model_index", 0)
    total = data.get("total_models", 6)

    status_map = {
        "starting": ("starting", "⏳ Initializing"),
        "training": ("training", f"🔥 Training  ·  {cur}"),
        "done":     ("done",     "✅ All models complete"),
        "error":    ("error",    "❌ Error occurred"),
    }
    sc, sl = status_map.get(status, ("starting", status))

    st.markdown(f"""
    <div class="hero">
        <div class="hero-eyebrow">🛡️ Ransomware Detection · AI Training Pipeline</div>
        <h1 class="hero-title">Live Training Monitor</h1>
        <p class="hero-sub">Real-time progress for all 6 models — Random Forest, XGBoost, SVM, DNN, LSTM, Meta-Ensemble</p>
        <div class="status-pill status-{sc}">
            <span class="pulse"></span>
            {sl} · Elapsed {elapsed_str(elapsed)}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_overall_progress(data):
    done_count = sum(1 for m in MODEL_ORDER if data.get("models", {}).get(m, {}).get("status") == "done")
    total = len(MODEL_ORDER)
    pct = int(done_count / total * 100)

    st.markdown(f"""
    <div class="progress-wrap">
        <div class="progress-label">
            <span class="progress-label-left">Overall Progress · {done_count} / {total} models complete</span>
            <span class="progress-label-right">{pct}%</span>
        </div>
        <div class="progress-track">
            <div class="progress-fill" style="width:{pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_cards(data):
    models_data = data.get("models", {})
    now = time.time()
    cards_html = '<div class="model-grid">'

    for name in MODEL_ORDER:
        info = models_data.get(name, {})
        st_val = info.get("status", "pending")
        metrics = info.get("metrics", {})
        icon = MODEL_ICONS.get(name, "🤖")
        desc = MODEL_DESC.get(name, "")

        # Badge
        if st_val == "done":
            badge = '<span class="model-status-badge badge-done">✔ Done</span>'
        elif st_val == "training":
            elapsed_s = now - info.get("started_at", now)
            badge = f'<span class="model-status-badge badge-training"><span class="pulse"></span> Training · {elapsed_str(elapsed_s)}</span>'
        else:
            badge = '<span class="model-status-badge badge-pending">⏸ Pending</span>'

        # Metrics block
        metrics_html = ""
        if metrics:
            f1  = fmt_pct(metrics.get("f1", 0))
            acc = fmt_pct(metrics.get("accuracy", 0))
            auc = fmt_pct(metrics.get("auc", 0))
            metrics_html = f"""
            <div class="model-metrics">
                <div class="metric-mini">
                    <div class="metric-mini-label">F1</div>
                    <div class="metric-mini-value highlight">{f1}</div>
                </div>
                <div class="metric-mini">
                    <div class="metric-mini-label">Acc</div>
                    <div class="metric-mini-value">{acc}</div>
                </div>
                <div class="metric-mini">
                    <div class="metric-mini-label">AUC</div>
                    <div class="metric-mini-value">{auc}</div>
                </div>
            </div>
            """
            elapsed_val = info.get("elapsed", 0)
            metrics_html += f'<div class="elapsed">⏱ Completed in {elapsed_str(elapsed_val)}</div>'

        cards_html += f"""
        <div class="model-card state-{st_val}">
            <div class="model-card-bar state-{st_val}"></div>
            <div class="model-icon">{icon}</div>
            <div class="model-name">{name}</div>
            <div class="model-desc">{desc}</div>
            {badge}
            {metrics_html}
        </div>
        """

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_results_table(data):
    results = data.get("results", [])
    if not results:
        return

    best_f1 = max(r.get("f1", 0) for r in results)

    rows_html = ""
    for r in results:
        name = r.get("model", "?")
        f1   = r.get("f1", 0)
        acc  = fmt_pct(r.get("accuracy", 0))
        prec = fmt_pct(r.get("precision", 0))
        rec  = fmt_pct(r.get("recall", 0))
        f1s  = fmt_pct(f1)
        auc  = fmt_pct(r.get("auc", 0))
        cls  = "best-row" if f1 == best_f1 else ""
        star = " ★" if f1 == best_f1 else ""
        rows_html += f"""
        <tr class="{cls}">
            <td>{name}{star}</td>
            <td>{acc}</td>
            <td>{prec}</td>
            <td>{rec}</td>
            <td>{f1s}</td>
            <td>{auc}</td>
        </tr>
        """

    st.markdown(f"""
    <div class="results-section">
        <div class="results-title">📊 Model Results So Far</div>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>AUC</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def render_log(data):
    log = data.get("log", [])
    if not log:
        return
    lines_html = ""
    for entry in reversed(log[-40:]):
        t   = entry.get("t", 0)
        msg = entry.get("msg", "")
        cls = "done" if "✔" in msg or "🎉" in msg else ("error" if "❌" in msg else ("start" if "▶" in msg else ""))
        ts_str = elapsed_str(t)
        lines_html += f'<div class="log-line"><span class="ts">[{ts_str}]</span><span class="msg {cls}">{msg}</span></div>'

    st.markdown(f"""
    <div class="results-section">
        <div class="results-title">📋 Training Log</div>
        <div class="log-box">{lines_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Main ───────────────────────────────────────────────────────────────────
data = load_progress()

if data is None:
    st.markdown("""
    <div class="no-file">
        <div class="big">🔍</div>
        <b>No training in progress</b><br><br>
        Start training first:<br>
        <code>python run_500k_training.py</code>
        <br><br>
        <small>This page auto-refreshes every 4 seconds once training begins.</small>
    </div>
    """, unsafe_allow_html=True)
else:
    render_hero(data)
    render_overall_progress(data)
    render_model_cards(data)
    render_results_table(data)
    render_log(data)

    status = data.get("status", "starting")
    if status not in ("done", "error"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("🔄 Auto-refreshing every 4 seconds…")
        time.sleep(4)
        st.rerun()
    else:
        if status == "done":
            st.success("🎉 Training complete! Run `streamlit run app/behavioral_dashboard.py` to see the full dashboard.")
        else:
            st.error("❌ Training ended with an error. Check the terminal for details.")
