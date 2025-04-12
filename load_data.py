
import streamlit as st
import pandas as pd
import altair as alt
import os

st.set_page_config(page_title="Model Comparison Dashboard", layout="wide")

# Title
st.title("🤖 Model Evaluation & Comparison Dashboard")

# Load data
data = pd.read_csv("model_metrics.csv")

# --- 1. METRICS TABLE ---
st.subheader("📊 Overall Performance Metrics")
st.dataframe(data)

# --- 2. BAR CHARTS FOR EACH METRIC ---
st.subheader("📈 Metric Comparisons")

metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
for metric in metric_columns:
    st.markdown(f"### 🔹 {metric}")
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X("Model", sort="-y"),
        y=alt.Y(metric, scale=alt.Scale(domain=[0, 1])),
        color="Model"
    ).properties(width=600, height=300)
    st.altair_chart(chart, use_container_width=True)

# --- 3. CONFUSION MATRICES & CLASSIFICATION REPORTS ---
st.subheader("🧩 Detailed Per-Model Insights")

model_selection = st.selectbox("Select a model to view details:", data["Model"])

# Show Confusion Matrix
st.markdown("#### 📌 Confusion Matrix")
cm_path = f"confusion_images/{model_selection}.png"
if os.path.exists(cm_path):
    st.image(cm_path, caption=f"{model_selection} - Confusion Matrix" )
else:
    st.warning("Confusion matrix image not found.")



