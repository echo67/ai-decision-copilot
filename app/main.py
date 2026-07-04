import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Decision Copilot")

st.title("🚀 AI Decision Copilot")

uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("파일 업로드 완료!")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")

    st.dataframe(df.head())

else:
    print("CSV 파일을 업로드해주세요.")