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


# import matplotlib.pyplot as plt
# numeric_cols = df.select_dtypes(include="number").columns
# selected_col = st.selectbox(
#     "수치형 컬럼 선택",
#     numeric_cols
# )
# fig, ax = plt.subplots()
# ax.hist(df[selected_col].dropna())
# ax.set_title(selected_col)
# ax.set_xlabel(selected_col)
# ax.set_ylabel("Count")
# st.pyplot(fig)

import plotly.express as px

plot_type = st.selectbox(
    "그래프 종류",
    [
        "Histogram",
        "Box Plot",
        "Violin Plot"
    ]
)

numeric_cols = df.select_dtypes(include="number").columns

selected_col = st.selectbox(
    "수치형 컬럼 선택",
    numeric_cols
)

if plot_type == "Histogram":
    fig = px.histogram(
        df,
        x=selected_col,
        nbins=30,
        title=f"{selected_col} Distribution"
    )
elif plot_type == "Box Plot":
    fig = px.box(
        df,
        y=selected_col,
        title=f"{selected_col} Box Plot"
    )
elif plot_type == "Violin Plot":
    fig = px.violin(
        df,
        y=selected_col,
        box=True,
        title=f"{selected_col} Violin Plot"
    )

fig.update_layout(
    xaxis_title=selected_col,
    yaxis_title="Count",
    template="plotly_white",
    height = 500
)

st.plotly_chart(
    fig,
    use_container_width=True
)



