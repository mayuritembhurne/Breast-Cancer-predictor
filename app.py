import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource
def load_assets():
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    stats = pd.read_csv("feature_stats.csv", index_col=0)
    examples = pd.read_csv("examples.csv", index_col=0)
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, stats, examples, metrics


model, scaler, stats, examples, metrics = load_assets()
features = list(stats.index)


def key(name):
    return f"in_{name}"


# Start every slider at the dataset median the first time the app loads
for f in features:
    if key(f) not in st.session_state:
        st.session_state[key(f)] = float(stats.loc[f, "median"])


def load_example(row_name):
    for f in features:
        st.session_state[key(f)] = float(examples.loc[row_name, f])


def reset_inputs():
    for f in features:
        st.session_state[key(f)] = float(stats.loc[f, "median"])


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this app")
    st.write(
        "This app uses a machine-learning model trained on the **Wisconsin "
        "Breast Cancer dataset** (569 tumor samples, 30 measurements each) "
        "to predict whether a tumor is **benign** or **malignant**."
    )
    st.subheader("Model performance")
    st.write(f"Model: **{metrics['model_name']}**")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")
    c1.metric("Precision", f"{metrics['precision']:.1%}")
    c2.metric("F1 score", f"{metrics['f1']:.1%}")
    st.caption("Measured on a held-out test set of 114 samples the model never saw during training.")
    st.warning(
        "Educational project only. This is **not** a medical device and must "
        "never be used to make real diagnostic decisions."
    )

# ------------------------------------------------------------------- main
st.title("🩺 Breast Cancer Diagnosis Predictor")
st.write(
    "Set the tumor measurements below (or load an example), then click "
    "**Predict**. All 30 values come from digitized images of a fine-needle "
    "aspirate of a breast mass."
)

b1, b2, b3, _ = st.columns([1.2, 1.2, 1, 3])
b1.button("Load typical benign case", on_click=load_example, args=("Typical benign",))
b2.button("Load typical malignant case", on_click=load_example, args=("Typical malignant",))
b3.button("Reset to median", on_click=reset_inputs)

groups = {
    "Mean values": [f for f in features if f.startswith("mean ")],
    "Standard error": [f for f in features if f.endswith(" error")],
    "Worst values": [f for f in features if f.startswith("worst ")],
}

tabs = st.tabs(list(groups.keys()))
for tab, (group_name, group_features) in zip(tabs, groups.items()):
    with tab:
        cols = st.columns(2)
        for i, f in enumerate(group_features):
            lo, hi = float(stats.loc[f, "min"]), float(stats.loc[f, "max"])
            with cols[i % 2]:
                st.slider(
                    f.capitalize(),
                    min_value=lo,
                    max_value=hi,
                    step=(hi - lo) / 1000,
                    format="%.4f",
                    key=key(f),
                )

st.divider()

if st.button("Predict", type="primary"):
    values = pd.DataFrame([[st.session_state[key(f)] for f in features]], columns=features)
    scaled = scaler.transform(values)
    prediction = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]

    # In this dataset: 0 = malignant, 1 = benign
    p_malignant = float(probabilities[0])
    p_benign = float(probabilities[1])

    if prediction == 1:
        st.success(f"### Prediction: BENIGN  ({p_benign:.1%} confidence)")
    else:
        st.error(f"### Prediction: MALIGNANT  ({p_malignant:.1%} confidence)")

    r1, r2 = st.columns(2)
    r1.metric("Probability benign", f"{p_benign:.1%}")
    r2.metric("Probability malignant", f"{p_malignant:.1%}")
    st.progress(p_malignant, text="Malignancy risk score")

    st.caption(
        "Remember: this is a classroom model. Real diagnosis requires "
        "clinical examination, imaging, and pathology by qualified professionals."
    )

with st.expander("How does the model decide? (top 10 influential features)"):
    coefs = pd.Series(model.coef_[0], index=features)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).head(10).index)
    st.bar_chart(top)
    st.caption(
        "Positive values push the prediction toward **benign**; negative "
        "values push it toward **malignant**. Features were standardized, so "
        "the bars are directly comparable."
    )
