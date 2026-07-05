import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Decision Copilot")

st.title("🚀 AI Decision Copilot")

uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"]
)

if uploaded_file is None:
    st.info("CSV 파일을 업로드해주세요.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.success("파일 업로드 완료!")

st.subheader("데이터 정보")
col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", df.shape[0])
with col2:
    st.metric("Columns", df.shape[1])

st.subheader("Columns")
st.write(df.columns.tolist())

st.subheader("Data Types")
st.dataframe(df.dtypes.reset_index())

missing = df.isnull().sum()

st.subheader("Missing Values")
st.dataframe(missing.reset_index())

st.subheader("Summary")
st.dataframe(df.describe())

st.subheader("Preview")
st.dataframe(df.head())