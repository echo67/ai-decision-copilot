# %%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI Decision Copilot", layout="wide")

st.title("🚀 AI Decision Copilot")

uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"]
)

if uploaded_file is None:
    st.info("CSV 파일을 업로드해주세요.")
    st.stop()

uploaded_file = "/Users/eunah/Dropbox/패스트캠퍼스/Part 8 (Unicode Encoding Conflict). 정형 데이터/Chapter 03. TabNet 활용 회귀 - 부동산 가격 예측/test.csv"
df = pd.read_csv(uploaded_file)

st.success("파일 업로드 완료!")

st.subheader("데이터 정보")
rows, cols = df.shape
missing = df.isna().sum().sum()
duplicate = df.duplicated().sum()
numeric = len(df.select_dtypes(include="number").columns)
categorical = len(df.select_dtypes(exclude="number").columns)
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", rows)
with col2:
    st.metric("Columns", cols)
with col3:
    st.metric("Missing", missing)

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Duplicate", duplicate)
with col5:
    st.metric("Numeric", numeric)
with col6:
    st.metric("Categorical", categorical)

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
numeric_cols = df.select_dtypes(include="number").columns
st.sidebar.header("Visualization Settings")

plot_type = st.sidebar.selectbox(
    "그래프 종류",
    [
        "Histogram",
        "Box Plot",
        "Violin Plot"
    ]
)

selected_col = st.sidebar.selectbox(
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

numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()


st.subheader("Correlation Matrix")
st.dataframe(corr)

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto"
)

fig.update_layout(
    title="Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


threshold = st.sidebar.slider(
    "Correlation Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.80,
    step=0.05
)

strong_corr = []
cols = corr.columns

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):  # 중복(A-B, B-A) 제거
        value = corr.iloc[i, j]
        if abs(value) >= threshold:
            strong_corr.append({
                "Variable 1": cols[i],
                "Variable 2": cols[j],
                "Correlation": round(value, 3)
            })

strong_corr_df = (
    pd.DataFrame(strong_corr)
      .sort_values(
         by="Correlation",
          key=lambda x: x.abs(),
          ascending=False
      )
      .reset_index(drop=True)
)

st.subheader("Strong Correlation")
st.dataframe(
    strong_corr_df,
    use_container_width=True
)


