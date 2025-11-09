import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Page Configuration
st.set_page_config(page_title="🌾 Smart Agriculture Dashboard", layout="wide")

st.title("🌿 Smart Agriculture Data Dashboard")

# ✅ Folder containing CSV
output_path = "/home/adminvarshitha/bigdata/projects/output_enhanced_fixed"

# ✅ Find the CSV automatically
csv_files = [f for f in os.listdir(output_path) if f.endswith(".csv")]
if not csv_files:
    st.error("⚠️ No CSV files found in output folder.")
else:
    file_path = os.path.join(output_path, csv_files[0])
    df = pd.read_csv(file_path)

    st.success(f"✅ Loaded dataset: **{csv_files[0]}**")
    st.write("### 📊 Sample Data")
    st.dataframe(df.head(10))

    # ✅ Dataset Summary
    st.markdown("### 📈 Dataset Summary")
    st.dataframe(df.describe())

    # ✅ Correlation Heatmap (Enhanced)
    st.markdown("### 🔍 Feature Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Correlation"},
        square=True,
        linewidths=0.5,
        ax=ax
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)

    # ✅ Plant Health Distribution
    if "PlantHealth" in df.columns:
        st.markdown("### 🌱 Plant Health Distribution")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.histplot(df["PlantHealth"], kde=True, bins=12, color="#4CAF50", ax=ax2)
        ax2.set_xlabel("Plant Health", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        st.pyplot(fig2)

    # ✅ Scatter: GreenColorIntensity vs PlantHealth
    if "GreenColorIntensity" in df.columns and "PlantHealth" in df.columns:
        st.markdown("### 🍃 Green Color Intensity vs Plant Health")
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=df,
            x="GreenColorIntensity",
            y="PlantHealth",
            color="#388E3C",
            s=70,
            alpha=0.7,
            ax=ax3
        )
        ax3.set_xlabel("Green Color Intensity", fontsize=12)
        ax3.set_ylabel("Plant Health", fontsize=12)
        st.pyplot(fig3)

    # ✅ Average Plant Health by Leaf Edge Type
    if "LeafEdgeType" in df.columns and "PlantHealth" in df.columns:
        st.markdown("### 🧩 Average Plant Health by Leaf Edge Type")
        avg_health = df.groupby("LeafEdgeType")["PlantHealth"].mean().reset_index()
        avg_health = avg_health.sort_values("PlantHealth", ascending=False)
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        sns.barplot(x="LeafEdgeType", y="PlantHealth", data=avg_health, palette="Greens", ax=ax4)
        ax4.set_xlabel("Leaf Edge Type", fontsize=12)
        ax4.set_ylabel("Average Plant Health", fontsize=12)
        st.pyplot(fig4)

    st.success("✅ Dashboard rendering complete.")
