import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("🌍 Disaster Management Prediction Dashboard")

# Load the data (replace with your actual CSV path)
data = pd.read_csv("results.csv")

st.subheader("📊 Sample Data")
st.dataframe(data.head())

st.subheader("📈 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, cmap="YlGnBu", ax=ax)
st.pyplot(fig)
