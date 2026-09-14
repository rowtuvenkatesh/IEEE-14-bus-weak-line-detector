import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="IEEE 14-Bus Weak Line Detector", layout="wide", page_icon="\u26A1")

# ============================================================
# LOAD MODELS (real, trained artifacts -- no re-training here)
# ============================================================
clf = joblib.load('gbm_classifier.pkl')
reg = joblib.load('severity_regressor.pkl')
FEATURE_COLS = joblib.load('feature_cols.pkl')
threshold = joblib.load('threshold.pkl')

with open('dashboard_data.json') as f:
    SCENARIO_DATA = json.load(f)['scenarios']


# Embedded small research-finding samples (200-point identity proof, 57-step
# trend trace) -- kept inline so no extra file is needed for deployment.
RESEARCH = {"lp_sample":[0.7568,0.0,0.4022,0.1359,0.5912,0.165,0.8259,0.0,0.0004,0.7758,0.0,0.0096,1.0,0.0,0.0,0.0362,0.0245,0.7965,0.0,0.0,1.0,0.6662,0.0,0.0,0.0,0.0466,0.2298,0.0158,0.3177,0.0,0.0,1.0,0.0,0.0,0.4909,0.9852,0.9,0.7991,0.0,0.9599,0.0,0.5575,0.7684,0.0,0.9627,0.3113,0.0026,0.2788,0.5844,0.0,0.2582,0.459,0.0,0.0203,0.9883,0.0558,0.0,0.4481,0.0,0.0,0.0016,0.9845,0.3682,1.0,0.8945,1.0,0.3944,1.0,0.908,0.1342,1.0,0.5689,0.0,0.0,1.0,0.0,0.7008,0.6428,0.0,0.0,0.0,0.9736,1.0,0.0,0.9882,0.0,1.0,0.2774,0.0051,0.0,0.712,0.8483,0.0,0.0,0.3972,0.0,0.0,0.0,0.9923,0.1309,0.2198,0.0,0.9921,0.5231,0.1819,0.9078,0.0,0.9774,0.97,0.5732,0.9727,0.0,0.4007,0.0963,0.6043,0.7604,0.9999,0.0,0.0,0.0163,0.1967,0.0,0.869,0.0,0.0008,0.7829,0.6001,0.762,0.0,1.0,0.9274,0.9993,0.9997,0.19,0.977,0.7954,0.4286,0.5785,0.2234,0.457,0.964,0.3571,0.0559,0.0,0.0429,1.0,1.0,0.8508,0.0097,0.0,0.6195,0.4105,0.8282,0.2808,0.99,0.7541,0.9563,0.272,0.0,0.0,0.0296,0.0,0.9643,0.0,0.0707,0.0,0.4187,0.677,0.7522,0.8189,1.0,0.0,0.0158,0.5654,0.8602,0.0,0.9099,0.0,0.0,0.2718,0.9609,0.0365,0.8707,0.8832,1.0,0.036,0.0539,0.8461,0.5077,0.0047,0.2615,0.0,0.0258,0.1138,0.5695,0.0,1.0,0.0806,0.1778,0.0],"bvsi_sample":[0.7568,0.0,0.4022,0.1359,0.5912,0.165,0.8259,0.0,0.0004,0.7758,0.0,0.0096,1.0,0.0,0.0,0.0362,0.0245,0.7965,0.0,0.0,1.0,0.6662,0.0,0.0,0.0,0.0466,0.2298,0.0158,0.3177,0.0,0.0,1.0,0.0,0.0,0.4909,0.9852,0.9,0.7991,0.0,0.9599,0.0,0.5575,0.7684,0.0,0.9627,0.3113,0.0026,0.2788,0.5844,0.0,0.2582,0.459,0.0,0.0203,0.9883,0.0558,0.0,0.4481,0.0,0.0,0.0016,0.9845,0.3682,1.0,0.8945,1.0,0.3944,1.0,0.908,0.1342,1.0,0.5689,0.0,0.0,1.0,0.0,0.7008,0.6428,0.0,0.0,0.0,0.9736,1.0,0.0,0.9882,0.0,1.0,0.2774,0.0051,0.0,0.712,0.8483,0.0,0.0,0.3972,0.0,0.0,0.0,0.9923,0.1309,0.2198,0.0,0.9921,0.5231,0.1819,0.9078,0.0,0.9774,0.97,0.5732,0.9727,0.0,0.4007,0.0963,0.6043,0.7604,0.9999,0.0,0.0,0.0163,0.1967,0.0,0.869,0.0,0.0008,0.7829,0.6001,0.762,0.0,1.0,0.9274,0.9993,0.9997,0.19,0.977,0.7954,0.4286,0.5785,0.2234,0.457,0.964,0.3571,0.0559,0.0,0.0429,1.0,1.0,0.8508,0.0097,0.0,0.6195,0.4105,0.8282,0.2808,0.99,0.7541,0.9563,0.272,0.0,0.0,0.0296,0.0,0.9643,0.0,0.0707,0.0,0.4187,0.677,0.7522,0.8189,1.0,0.0,0.0158,0.5654,0.8602,0.0,0.9099,0.0,0.0,0.2718,0.9609,0.0365,0.8707,0.8832,1.0,0.036,0.0539,0.8461,0.5077,0.0047,0.2615,0.0,0.0258,0.1138,0.5695,0.0,1.0,0.0806,0.1778,0.0],"lambda_trend":[0.0,0.0154,0.0462,0.1077,0.2304,0.3831,0.5349,0.6858,0.8357,0.9845,1.1321,1.2784,1.4233,1.5667,1.7085,1.8484,1.9864,2.1223,2.2492,2.3695,2.4831,2.5906,2.6923,2.7886,2.8797,2.9659,3.0475,3.1246,3.1976,3.2667,3.3319,3.3935,3.4516,3.5064,3.558,3.6066,3.6522,3.6949,3.735,3.7723,3.8072,3.8395,3.8694,3.8969,3.9222,3.9452,3.966,3.9847,4.0012,4.0157,4.0281,4.0385,4.0468,4.0531,4.0575,4.0598,4.0603],"nvsi_trend":[0.0987,0.098,0.097,0.0963,0.1007,0.1153,0.1372,0.1635,0.1924,0.2229,0.2544,0.2864,0.3189,0.3514,0.384,0.4165,0.4488,0.4807,0.5107,0.5393,0.5664,0.5921,0.6165,0.6396,0.6615,0.6823,0.7021,0.7208,0.7385,0.7554,0.7714,0.7866,0.801,0.8147,0.8277,0.84,0.8517,0.8628,0.8733,0.8833,0.8928,0.9018,0.9103,0.9184,0.926,0.9333,0.9401,0.9466,0.9527,0.9585,0.964,0.9691,0.9739,0.9784,0.9826,0.9866,0.9891]}

METRICS = {
    'classifier': {
        'threshold': 0.218,
        'precision': 0.758, 'recall': 0.901, 'f1': 0.824, 'roc_auc': 0.987,
        'confusion_matrix': [[6250, 96], [33, 301]]
    },
    'regressor': {'mae': 0.0024, 'rmse': 0.0085, 'r2': 0.999},
    'feature_importance': {
        'dV': 0.267, 'Qto': 0.223, 'Pfrom': 0.092, 'Pto': 0.089, 'lambda': 0.070,
        'Qloss': 0.067, 'dist_to_collapse': 0.058, 'Vj': 0.060, 'Qfrom': 0.044,
        'Vi': 0.0104, 'X': 0.0065, 'Ploss': 0.0075, 'Z': 0.0036, 'R': 0.0023
    },
    'vsi_ranking': {
        'NVSI':   {'sat': 0.4,  'corr': 0.99},
        'L':      {'sat': 3.6,  'corr': 0.85},
        'FVSI':   {'sat': 6.6,  'corr': 0.80},
        'VQILine':{'sat': 6.6,  'corr': 0.80},
        'NLSI':   {'sat': 4.6,  'corr': 0.79},
        'VSI2':   {'sat': 13.0, 'corr': 0.78},
        'VLSI':   {'sat': 1.7,  'corr': 0.76},
        'Lij':    {'sat': 11.0, 'corr': 0.73},
        'Lmn':    {'sat': 11.0, 'corr': 0.73},
        'LQP':    {'sat': 2.1,  'corr': 0.35},
        'BVSI':   {'sat': 7.2,  'corr': 0.32},
        'Lp':     {'sat': 7.2,  'corr': 0.32},
    }
}

BUS_POS = {
    1:(1.2,5.9), 2:(1.2,4.5), 3:(1.2,2.6), 4:(3.0,3.4), 5:(3.0,5.2),
    6:(4.8,2.6), 7:(4.8,4.2), 8:(4.8,5.8), 9:(6.6,3.9), 10:(6.6,2.4),
    11:(5.4,1.4), 12:(7.2,1.4), 13:(7.8,2.0), 14:(8.0,3.1)
}
SCENARIO_LABELS = {
    "9":  "Contingency — line 8 tripped (4→7)",
    "10": "Contingency — line 9 tripped (4→9)",
    "17": "Contingency — line 16 tripped (9→10)",
    "19": "Contingency — line 18 tripped (10→11)",
    "25": "Randomized loading direction A",
    "29": "Randomized loading direction B"
}

def risk_color(p):
    stops = [(0.0, (58,69,86)), (threshold, (234,179,8)), (1.0, (239,68,68))]
    for (t0,c0), (t1,c1) in zip(stops, stops[1:]):
        if t0 <= p <= t1 or (p < 0 and t0 == 0) or (p > 1 and t1 == 1):
            tt = 0 if t1==t0 else max(0,min(1,(p-t0)/(t1-t0)))
            r = c0[0]+(c1[0]-c0[0])*tt; g = c0[1]+(c1[1]-c0[1])*tt; b = c0[2]+(c1[2]-c0[2])*tt
            return f'rgb({r:.0f},{g:.0f},{b:.0f})'
    return 'rgb(239,68,68)'

# ============================================================
# HEADER
# ============================================================
st.title("\u26A1 IEEE 14-Bus Weak Line Detector")
st.caption("Machine-learning voltage stability monitoring \u2014 Gradient Boosting classifier + Random Forest severity regressor, "
           "trained on a 31,660-sample multi-scenario continuation power flow dataset.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Classifier F1", f"{METRICS['classifier']['f1']:.3f}")
k2.metric("ROC-AUC", f"{METRICS['classifier']['roc_auc']:.3f}")
k3.metric("Recall (critical)", f"{METRICS['classifier']['recall']:.1%}")
k4.metric("Regressor R\u00b2", f"{METRICS['regressor']['r2']:.3f}")
k5.metric("Decision threshold", f"{threshold:.3f}")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001F5FA\uFE0F  Network Status & One-Line Diagram",
    "\U0001F3AF  Live What-If Prediction",
    "\U0001F4CA  Model Performance",
    "\U0001F52C  VSI Research Insights",
])

# ============================================================
# TAB 1: REAL-TIME NETWORK STATUS + ONE-LINE DIAGRAM
# ============================================================
with tab1:
    st.subheader("All 20 lines, live status for a selected operating point")
    st.caption("Replaying real predictions on held-out test scenarios never seen during training.")

    scenario_ids = list(SCENARIO_DATA.keys())
    c_sc, c_step = st.columns([2, 3])
    with c_sc:
        scenario_id = st.selectbox("Scenario", scenario_ids, format_func=lambda s: SCENARIO_LABELS.get(s, s))
    scenario = SCENARIO_DATA[scenario_id]
    n_steps = len(scenario['steps'])
    with c_step:
        step_idx = st.slider("Loading step", 0, n_steps - 1, 0, key="step_slider")
    step = scenario['steps'][step_idx]

    lines = step['lines']
    top = max(lines, key=lambda l: l[4])
    n_flagged = sum(1 for l in lines if l[5] == 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Loading factor \u03bb", f"{step['lam']:.3f}")
    m2.metric("Distance to collapse", f"{step['dtc']:.3f}")
    m3.metric("Top predicted risk", f"L{top[0]} \u2014 {top[4]*100:.1f}%")
    m4.metric("Lines flagged critical", f"{n_flagged} / 20")

    col_diag, col_rank = st.columns([3, 2])

    with col_diag:
        fig = go.Figure()
        for line_id, f, t, actual, proba_l, pred, *_ in lines:
            x1, y1 = BUS_POS[f]; x2, y2 = BUS_POS[t]
            width = 3 + proba_l * 7
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2], mode='lines',
                line=dict(color=risk_color(proba_l), width=width),
                hoverinfo='text',
                text=f"Line {line_id} ({f}\u2192{t})<br>Predicted risk: {proba_l*100:.1f}%<br>Model call: {'CRITICAL' if pred else 'not critical'}",
                showlegend=False
            ))
            if actual == 1:
                fig.add_trace(go.Scatter(
                    x=[(x1+x2)/2], y=[(y1+y2)/2], mode='markers',
                    marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(color='black', width=3)),
                    hoverinfo='text', text=f"Ground truth weak line: L{line_id}", showlegend=False
                ))
        bx = [BUS_POS[b][0] for b in BUS_POS]; by = [BUS_POS[b][1] for b in BUS_POS]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode='markers+text', text=[str(b) for b in BUS_POS],
            textposition='middle center', textfont=dict(color='white', size=10),
            marker=dict(size=28, color='#1f4e79', symbol='square'),
            hoverinfo='skip', showlegend=False
        ))
        fig.update_layout(
            height=480, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Line color/thickness = predicted risk (grey \u2192 amber \u2192 red). Black ring = actual ground-truth weak line.")

    with col_rank:
        st.markdown("**All 20 lines, ranked by predicted risk**")
        ranked = sorted(lines, key=lambda l: -l[4])
        rank_df = pd.DataFrame(
            [(f"L{l[0]}", f"{l[1]}\u2192{l[2]}", l[4], "\u2713" if l[3] else "") for l in ranked],
            columns=["Line", "Path", "Risk", "Actual"]
        )
        st.dataframe(
            rank_df, use_container_width=True, hide_index=True, height=440,
            column_config={"Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.1f%%")}
        )

# ============================================================
# TAB 2: LIVE WHAT-IF PREDICTION
# ============================================================
with tab2:
    st.subheader("Manually enter a line's electrical state")
    st.caption("Predict weak-line status and severity for any hypothetical measurement, not limited to the replayed scenarios.")

    RANGES = {
        'R': (0.0, 0.221, 0.055), 'X': (0.042, 0.556, 0.195), 'Z': (0.044, 0.556, 0.206),
        'Vi': (0.726, 1.085, 1.039), 'Vj': (0.694, 1.090, 0.995), 'dV': (-0.240, 0.295, 0.016),
        'Pfrom': (-159.572, 620.911, 24.617), 'Qfrom': (-79.289, 162.431, 6.232),
        'Pto': (-553.727, 164.704, -23.588), 'Qto': (-80.022, 286.539, 0.000),
        'Ploss': (0.0, 88.964, 0.376), 'Qloss': (-3.681, 325.685, 5.103),
        'lambda': (0.013, 4.622, 2.784), 'dist_to_collapse': (0.0, 4.080, 0.777),
    }
    cols = st.columns(3)
    values = {}
    for i, feat in enumerate(FEATURE_COLS):
        lo, hi, default = RANGES.get(feat, (0.0, 1.0, 0.0))
        with cols[i % 3]:
            values[feat] = st.slider(feat, float(lo), float(hi), float(default), key=f"slider_{feat}")

    X_input = pd.DataFrame([values])[FEATURE_COLS]
    proba = clf.predict_proba(X_input)[0, 1]
    is_critical = proba >= threshold
    severity = reg.predict(X_input)[0]

    st.markdown("#### Prediction")
    c1, c2, c3 = st.columns(3)
    c1.metric("Critical-line probability", f"{proba*100:.1f}%")
    c2.metric(f"Model call (threshold {threshold:.3f})", "CRITICAL" if is_critical else "not critical")
    c3.metric("Predicted severity (NVSI)", f"{severity:.3f}")

    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=proba*100,
        number={'suffix': "%"},
        gauge={'axis': {'range': [0,100]},
               'bar': {'color': risk_color(proba)},
               'steps': [{'range':[0,threshold*100],'color':'#e5e7eb'},
                         {'range':[threshold*100,100],'color':'#fecaca'}],
               'threshold': {'line':{'color':'black','width':3},'value':threshold*100}}
    ))
    gauge.update_layout(height=250, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(gauge, use_container_width=True)

    if is_critical:
        st.error("This line is flagged as a likely weak point under these conditions.")
    else:
        st.success("This line is not flagged as critical under these conditions.")

# ============================================================
# TAB 3: MODEL PERFORMANCE & ANALYTICS
# ============================================================
with tab3:
    st.subheader("Classifier performance (held-out, scenario-disjoint test set)")

    cA, cB = st.columns(2)
    with cA:
        cm = np.array(METRICS['classifier']['confusion_matrix'])
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=['Not critical','Critical'], y=['Not critical','Critical'],
            colorscale='Blues', text=cm, texttemplate="%{text}", showscale=False
        ))
        fig_cm.update_layout(title="Confusion matrix", height=380,
                              xaxis_title="Predicted", yaxis_title="Actual",
                              yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_cm, use_container_width=True)

    with cB:
        met = METRICS['classifier']
        met_df = pd.DataFrame({
            'Metric': ['Precision','Recall','F1-score','ROC-AUC'],
            'Value': [met['precision'], met['recall'], met['f1'], met['roc_auc']]
        })
        fig_met = px.bar(met_df, x='Metric', y='Value', range_y=[0,1], text='Value',
                          color='Value', color_continuous_scale='Blues')
        fig_met.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_met.update_layout(title="Classifier metrics", height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_met, use_container_width=True)

    st.subheader("Feature importance")
    fi = METRICS['feature_importance']
    fi_df = pd.DataFrame(sorted(fi.items(), key=lambda x: x[1]), columns=['Feature','Importance'])
    fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues')
    fig_fi.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)
    st.caption("Voltage drop (dV) and reactive power flow (Qto) dominate \u2014 consistent with established voltage stability theory.")

    st.subheader("Regressor performance (severity_score / NVSI)")
    r1, r2, r3 = st.columns(3)
    r1.metric("R\u00b2", f"{METRICS['regressor']['r2']:.4f}")
    r2.metric("MAE", f"{METRICS['regressor']['mae']:.4f}")
    r3.metric("RMSE", f"{METRICS['regressor']['rmse']:.4f}")
    st.caption("Near-exact fit is expected: severity_score is a deterministic function of measured features (X, Vi, Pto, Qto), not a noisy label.")

# ============================================================
# TAB 4: VSI RESEARCH INSIGHTS
# ============================================================
with tab4:
    st.subheader("Reliability of all 12 published voltage stability indices")
    st.caption("Scored on saturation rate and trend correlation with the loading factor, across all 30 simulated scenarios.")

    vsi_df = pd.DataFrame(METRICS['vsi_ranking']).T.reset_index().rename(columns={'index':'Index'})
    vsi_df = vsi_df.sort_values('corr', ascending=False)
    fig_vsi = go.Figure()
    fig_vsi.add_trace(go.Bar(x=vsi_df['Index'], y=vsi_df['corr'], name='Trend correlation', marker_color='#1f4e79'))
    fig_vsi.add_trace(go.Bar(x=vsi_df['Index'], y=vsi_df['sat']/100, name='Saturation rate', marker_color='#c0504d'))
    fig_vsi.update_layout(barmode='group', height=420, yaxis_title='Score (0-1 scale)',
                           legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig_vsi, use_container_width=True)
    st.info("**NVSI** is the most reliable index (0.99 trend correlation, 0.4% saturation) and is used as the ground-truth "
            "severity label throughout this app. **BVSI** and **Lp** tie for the weakest trend correlation \u2014 not a coincidence, "
            "see below.")

    st.subheader("Key finding: BVSI and Lp are algebraically identical")
    c_scat, c_trend = st.columns(2)
    with c_scat:
        fig_id = go.Figure(go.Scatter(
            x=RESEARCH['lp_sample'], y=RESEARCH['bvsi_sample'], mode='markers',
            marker=dict(size=5, color='#1f4e79', opacity=0.6), name='Sampled operating points'
        ))
        fig_id.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='red', dash='dash'), name='y = x'))
        fig_id.update_layout(title="BVSI vs. Lp (200 sampled points)", height=380,
                              xaxis_title="Lp", yaxis_title="BVSI")
        st.plotly_chart(fig_id, use_container_width=True)
        st.caption("Every point lies exactly on y=x (max deviation ~5.6\u00d710\u207b\u2079, floating-point precision).")
    with c_trend:
        fig_tr = go.Figure(go.Scatter(
            x=RESEARCH['lambda_trend'], y=RESEARCH['nvsi_trend'], mode='lines',
            line=dict(color='#1f4e79', width=2)
        ))
        fig_tr.add_hline(y=1.0, line_dash='dot', line_color='gray')
        fig_tr.update_layout(title="NVSI vs. loading factor (line 10, base scenario)", height=380,
                              xaxis_title="Loading factor \u03bb", yaxis_title="NVSI")
        st.plotly_chart(fig_tr, use_container_width=True)
        st.caption("NVSI rises smoothly toward the collapse point \u2014 the behavior expected of a trustworthy index.")

    st.markdown("""
    **Why this matters:** since \u03b8=atan2(X,R), the identity cos(\u03b8\u2212\u03b4) = [Rcos\u03b4+Xsin\u03b4]/Z shows that
    **Lp \u2261 BVSI** and **Lij \u2261 Lmn** are the same formula in different variables, and **VQILine \u2261 FVSI** via
    |Im(1/(R+jX))| = X/Z\u00b2. Of the 12 indices compared in the literature, only 9 are mathematically distinct.
    """)

st.divider()
st.caption(
    "IEEE 14-bus system \u2022 Gradient Boosting classifier (F1=0.824, ROC-AUC=0.987) \u2022 "
    "Random Forest severity regressor (R\u00b2=0.999) \u2022 Features are raw electrical measurements only, "
    "no VSI formula computed at inference time."
)
