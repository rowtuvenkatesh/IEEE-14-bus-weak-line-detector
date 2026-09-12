import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Weak Line Detector", layout="wide")

# ---- Load the real trained models (no conversion, exact Python objects) ----
clf = joblib.load('gbm_classifier.pkl')
reg = joblib.load('severity_regressor.pkl')
FEATURE_COLS = joblib.load('feature_cols.pkl')
threshold = joblib.load('threshold.pkl')

st.title("IEEE 14-bus Weak Line Detector")
st.caption("Live prediction using the actual trained Gradient Boosting classifier and Random Forest regressor.")

# ---- Input controls, one per feature, using the real dataset's 2nd/98th
#      percentile range (trims extreme outliers) with the median as default ----
RANGES = {
    # feature:            (min slider,  max slider,  default = dataset median)
    'R':                (0.0,     0.221,   0.055),
    'X':                (0.042,   0.556,   0.195),
    'Z':                (0.044,   0.556,   0.206),
    'Vi':               (0.726,   1.085,   1.039),
    'Vj':               (0.694,   1.090,   0.995),
    'dV':               (-0.240,  0.295,   0.016),
    'Pfrom':            (-159.572, 620.911, 24.617),
    'Qfrom':            (-79.289,  162.431,  6.232),
    'Pto':              (-553.727, 164.704, -23.588),
    'Qto':              (-80.022,  286.539,  0.000),
    'Ploss':            (0.0,     88.964,   0.376),
    'Qloss':            (-3.681,  325.685,  5.103),
    'lambda':           (0.013,   4.622,    2.784),
    'dist_to_collapse': (0.0,     4.080,    0.777),
}

st.subheader("Input line/system state")
cols = st.columns(3)
values = {}
for i, feat in enumerate(FEATURE_COLS):
    lo, hi, default = RANGES.get(feat, (0.0, 1.0, 0.0))
    with cols[i % 3]:
        values[feat] = st.slider(feat, float(lo), float(hi), float(default))

X_input = pd.DataFrame([values])[FEATURE_COLS]

# ---- Predict ----
proba = clf.predict_proba(X_input)[0, 1]
is_critical = proba >= threshold
severity = reg.predict(X_input)[0]

st.subheader("Prediction")
c1, c2, c3 = st.columns(3)
c1.metric("Critical-line probability", f"{proba*100:.1f}%")
c2.metric("Model call (threshold {:.3f})".format(threshold), "CRITICAL" if is_critical else "not critical")
c3.metric("Predicted severity (NVSI)", f"{severity:.3f}")

if is_critical:
    st.error("This line is flagged as a likely weak point under these conditions.")
else:
    st.success("This line is not flagged as critical under these conditions.")

st.divider()
st.caption(
    "Model: Gradient Boosting classifier (F1=0.824, ROC-AUC~0.98) and Random Forest "
    "regressor (R²=0.999) trained on the IEEE 14-bus multi-scenario CPF dataset. "
    "Features are raw electrical measurements only -- no VSI formula is computed at inference."
)
