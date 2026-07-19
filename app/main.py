# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


# =========================================================
# 1. Page Configuration
# =========================================================
st.set_page_config(
    page_title="AI Decision Copilot",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 AI Decision Copilot")
st.caption(
    "데이터 탐색, 분포 진단, 상관관계 분석, "
    "분류 모델링 및 SHAP 기반 의사결정 지원"
)


# =========================================================
# 2. Utility Functions
# =========================================================
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """
    업로드된 CSV 파일을 읽습니다.
    여러 인코딩을 순차적으로 시도합니다.
    """
    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp949",
        "euc-kr",
    ]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(
                file,
                encoding=encoding,
            )
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "CSV 파일의 인코딩을 확인할 수 없습니다. "
        "UTF-8 또는 CP949 형식으로 저장한 후 다시 업로드해주세요."
    )


def make_strong_correlation_table(
    correlation_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    절댓값이 threshold 이상인 상관관계 변수 쌍을 반환합니다.
    자기상관과 중복 조합은 제외합니다.
    """
    results = []
    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = correlation_matrix.iloc[i, j]

            if (
                pd.notna(correlation)
                and abs(correlation) >= threshold
            ):
                results.append(
                    {
                        "Variable 1": columns[i],
                        "Variable 2": columns[j],
                        "Correlation": correlation,
                        "Absolute Correlation": abs(
                            correlation
                        ),
                    }
                )

    if not results:
        return pd.DataFrame(
            columns=[
                "Variable 1",
                "Variable 2",
                "Correlation",
                "Absolute Correlation",
            ]
        )

    return (
        pd.DataFrame(results)
        .sort_values(
            by="Absolute Correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_distribution_health(
    series: pd.Series,
) -> dict:
    """
    선택한 수치형 변수의 분포 통계와
    탐색적 Distribution Health Score를 계산합니다.
    """
    clean_series = series.dropna()
    valid_count = len(clean_series)

    if valid_count == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
            "missing_rate": 100.0,
            "zero_rate": np.nan,
            "outlier_count": 0,
            "outlier_rate": np.nan,
            "score": 0,
            "level": "🔴 Poor",
        }

    mean = clean_series.mean()
    median = clean_series.median()
    std = clean_series.std()
    skew = clean_series.skew()
    kurtosis = clean_series.kurt()

    missing_rate = series.isna().mean() * 100
    zero_rate = clean_series.eq(0).mean() * 100

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (clean_series < lower_bound)
        | (clean_series > upper_bound)
    )

    outlier_count = int(outlier_mask.sum())
    outlier_rate = (
        outlier_count / valid_count * 100
    )

    score = 100

    if missing_rate > 5:
        score -= 20

    if zero_rate > 70:
        score -= 15

    if outlier_rate > 5:
        score -= 20

    if pd.notna(skew) and abs(skew) > 2:
        score -= 15

    if (
        pd.notna(kurtosis)
        and abs(kurtosis) > 7
    ):
        score -= 10

    score = max(score, 0)

    if score >= 90:
        level = "🟢 Excellent"
    elif score >= 70:
        level = "🟡 Good"
    elif score >= 50:
        level = "🟠 Needs Review"
    else:
        level = "🔴 Poor"

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "missing_rate": missing_rate,
        "zero_rate": zero_rate,
        "outlier_count": outlier_count,
        "outlier_rate": outlier_rate,
        "score": score,
        "level": level,
    }


def get_confidence_level(
    probability: float,
) -> tuple[str, str]:
    """
    예측 확률에 따른 Confidence 수준과 상태 색상을 반환합니다.
    """
    if probability >= 0.95:
        return (
            "🟢 High Confidence",
            "success",
        )

    if probability >= 0.70:
        return (
            "🟡 Medium Confidence",
            "warning",
        )

    return (
        "🔴 Low Confidence",
        "error",
    )

def render_insight_card(
    title: str,
    value: str,
    subtitle: str = "",
) -> None:
    html = f"""
<div style="
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    min-height: 145px;
    background-color: rgba(128, 128, 128, 0.05);
">
    <div style="
        font-size: 14px;
        color: #777777;
        margin-bottom: 10px;
    ">
        {title}
    </div>
    <div style="
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 8px;
        word-break: break-word;
    ">
        {value}
    </div>
    <div style="
        font-size: 13px;
        color: #777777;
    ">
        {subtitle}
    </div>
</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =========================================================
# 3. File Upload
# =========================================================
uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "분석을 시작하려면 CSV 파일을 업로드해주세요."
    )
    st.stop()

try:
    df = load_csv(uploaded_file)

except Exception as error:
    st.error(
        f"파일을 읽는 중 오류가 발생했습니다: {error}"
    )
    st.stop()

if df.empty:
    st.warning(
        "업로드한 CSV 파일에 데이터가 없습니다."
    )
    st.stop()


# 새로운 파일이 업로드되면 이전 모델 결과 제거
current_file_key = (
    uploaded_file.name,
    uploaded_file.size,
)

previous_file_key = st.session_state.get(
    "uploaded_file_key"
)

if (
    previous_file_key is not None
    and previous_file_key != current_file_key
):
    st.session_state.pop(
        "model_result",
        None,
    )

st.session_state["uploaded_file_key"] = (
    current_file_key
)

st.success("파일 업로드 완료!")


# =========================================================
# 4. Data Overview
# =========================================================
st.header("1. Data Overview")

rows, columns = df.shape
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

numeric_columns = (
    df.select_dtypes(include="number")
    .columns
    .tolist()
)

categorical_columns = (
    df.select_dtypes(exclude="number")
    .columns
    .tolist()
)

overview_columns = st.columns(6)

overview_metrics = [
    ("Rows", rows),
    ("Columns", columns),
    ("Missing", missing_count),
    ("Duplicates", duplicate_count),
    ("Numeric", len(numeric_columns)),
    ("Categorical", len(categorical_columns)),
]

for column, (label, value) in zip(
    overview_columns,
    overview_metrics,
):
    with column:
        st.metric(
            label,
            f"{value:,}",
        )


overview_tab, missing_tab, summary_tab, preview_tab = (
    st.tabs(
        [
            "Column Information",
            "Missing Values",
            "Summary",
            "Preview",
        ]
    )
)

with overview_tab:
    column_information = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": (
                df.dtypes.astype(str).values
            ),
            "Non-Null Count": (
                df.notna().sum().values
            ),
            "Unique Count": (
                df.nunique(dropna=True).values
            ),
        }
    )

    st.dataframe(
        column_information,
        use_container_width=True,
        hide_index=True,
    )

with missing_tab:
    missing_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": (
                df.isna().sum().values
            ),
            "Missing Rate (%)": (
                df.isna().mean().values * 100
            ).round(2),
        }
    ).sort_values(
        by="Missing Count",
        ascending=False,
    )

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True,
    )

with summary_tab:
    st.dataframe(
        df.describe(
            include="all"
        ).T,
        use_container_width=True,
    )

with preview_tab:
    max_preview_rows = min(
        100,
        len(df),
    )

    preview_rows = st.slider(
        "미리보기 행 수",
        min_value=1,
        max_value=max_preview_rows,
        value=min(
            10,
            max_preview_rows,
        ),
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True,
    )


# =========================================================
# 5. Visualization Settings
# =========================================================
if not numeric_columns:
    st.warning(
        "수치형 컬럼이 없어 분석을 수행할 수 없습니다."
    )
    st.stop()

st.sidebar.header(
    "Visualization Settings"
)

plot_type = st.sidebar.selectbox(
    "그래프 종류",
    [
        "Histogram",
        "Box Plot",
        "Violin Plot",
    ],
)

selected_column = st.sidebar.selectbox(
    "수치형 컬럼",
    numeric_columns,
)

correlation_threshold = st.sidebar.slider(
    "Correlation Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.80,
    step=0.05,
)


# =========================================================
# 6. Distribution Visualization
# =========================================================
st.header(
    "2. Distribution Visualization"
)

if plot_type == "Histogram":
    number_of_bins = st.sidebar.slider(
        "Histogram Bins",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
    )

    distribution_figure = px.histogram(
        df,
        x=selected_column,
        nbins=number_of_bins,
        title=(
            f"{selected_column} Distribution"
        ),
    )

    distribution_figure.update_layout(
        xaxis_title=selected_column,
        yaxis_title="Count",
    )

elif plot_type == "Box Plot":
    distribution_figure = px.box(
        df,
        y=selected_column,
        points="outliers",
        title=(
            f"{selected_column} Box Plot"
        ),
    )

    distribution_figure.update_layout(
        xaxis_title="",
        yaxis_title=selected_column,
    )

else:
    distribution_figure = px.violin(
        df,
        y=selected_column,
        box=True,
        points="outliers",
        title=(
            f"{selected_column} Violin Plot"
        ),
    )

    distribution_figure.update_layout(
        xaxis_title="",
        yaxis_title=selected_column,
    )

distribution_figure.update_layout(
    template="plotly_white",
    height=500,
)

st.plotly_chart(
    distribution_figure,
    use_container_width=True,
)


# =========================================================
# 7. Distribution Analysis
# =========================================================
st.header("3. Distribution Analysis")

distribution_result = (
    calculate_distribution_health(
        df[selected_column]
    )
)

first_metric_row = st.columns(4)

with first_metric_row[0]:
    st.metric(
        "Mean",
        (
            f"{distribution_result['mean']:.3f}"
            if pd.notna(
                distribution_result["mean"]
            )
            else "N/A"
        ),
    )

with first_metric_row[1]:
    st.metric(
        "Median",
        (
            f"{distribution_result['median']:.3f}"
            if pd.notna(
                distribution_result["median"]
            )
            else "N/A"
        ),
    )

with first_metric_row[2]:
    st.metric(
        "Standard Deviation",
        (
            f"{distribution_result['std']:.3f}"
            if pd.notna(
                distribution_result["std"]
            )
            else "N/A"
        ),
    )

with first_metric_row[3]:
    st.metric(
        "Skewness",
        (
            f"{distribution_result['skew']:.3f}"
            if pd.notna(
                distribution_result["skew"]
            )
            else "N/A"
        ),
    )


second_metric_row = st.columns(4)

with second_metric_row[0]:
    st.metric(
        "Kurtosis",
        (
            f"{distribution_result['kurtosis']:.3f}"
            if pd.notna(
                distribution_result["kurtosis"]
            )
            else "N/A"
        ),
    )

with second_metric_row[1]:
    st.metric(
        "Zero Rate",
        (
            f"{distribution_result['zero_rate']:.2f}%"
            if pd.notna(
                distribution_result["zero_rate"]
            )
            else "N/A"
        ),
    )

with second_metric_row[2]:
    st.metric(
        "Missing Rate",
        f"{distribution_result['missing_rate']:.2f}%",
    )

with second_metric_row[3]:
    st.metric(
        "Outlier Count",
        f"{distribution_result['outlier_count']:,}",
        delta=(
            f"{distribution_result['outlier_rate']:.2f}%"
        ),
        delta_color="off",
    )


if (
    pd.notna(distribution_result["skew"])
    and abs(distribution_result["skew"]) > 1
):
    skew_direction = (
        "오른쪽"
        if distribution_result["skew"] > 0
        else "왼쪽"
    )

    st.info(
        f"선택한 변수는 {skew_direction}으로 "
        "치우친 분포입니다."
    )

if (
    pd.notna(
        distribution_result["zero_rate"]
    )
    and distribution_result["zero_rate"] > 50
):
    st.warning(
        "유효 관측치 중 0의 비율이 50%를 초과합니다. "
        "Zero-inflated 데이터 가능성을 검토해보세요."
    )


st.subheader("Distribution Health")

health_column1, health_column2 = (
    st.columns([1, 3])
)

with health_column1:
    st.metric(
        "Health Score",
        (
            f"{distribution_result['score']} / 100"
        ),
    )

with health_column2:
    st.markdown(
        f"### {distribution_result['level']}"
    )

    st.progress(
        distribution_result["score"] / 100
    )

st.caption(
    "Distribution Health Score는 결측률, 0 비율, "
    "이상치 비율, 왜도 및 첨도를 이용한 탐색적 점수입니다."
)


# =========================================================
# 8. Correlation Analysis
# =========================================================
st.header("4. Correlation Analysis")

numeric_df = df[numeric_columns]
correlation_matrix = numeric_df.corr()

if len(numeric_columns) < 2:
    st.info(
        "상관관계 분석에는 최소 2개의 수치형 변수가 필요합니다."
    )

else:
    correlation_tab, heatmap_tab, strong_tab = (
        st.tabs(
            [
                "Correlation Matrix",
                "Correlation Heatmap",
                "Strong Correlations",
            ]
        )
    )

    with correlation_tab:
        st.dataframe(
            correlation_matrix.style.format(
                "{:.3f}"
            ),
            use_container_width=True,
        )

    with heatmap_tab:
        correlation_figure = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
            aspect="auto",
            title="Correlation Heatmap",
        )

        correlation_figure.update_layout(
            template="plotly_white",
            height=max(
                500,
                len(numeric_columns) * 35,
            ),
        )

        st.plotly_chart(
            correlation_figure,
            use_container_width=True,
        )

    with strong_tab:
        strong_correlation_df = (
            make_strong_correlation_table(
                correlation_matrix,
                correlation_threshold,
            )
        )

        if strong_correlation_df.empty:
            st.info(
                f"절댓값이 {correlation_threshold:.2f} 이상인 "
                "상관관계가 없습니다."
            )

        else:
            st.dataframe(
                strong_correlation_df.style.format(
                    {
                        "Correlation": "{:.3f}",
                        "Absolute Correlation": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# 9. Classification Model Settings
# =========================================================
st.header("5. Classification Model")

st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "모델 종류",
    [
        "Logistic Regression",
        "Random Forest",
    ],
)

target_column = st.sidebar.selectbox(
    "Target",
    numeric_columns,
)

test_size = st.sidebar.slider(
    "Test Data Ratio",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

target_unique_count = (
    df[target_column]
    .nunique(dropna=True)
)

st.write(
    f"선택한 Target: **{target_column}**"
)

st.write(
    f"Target 고유값 수: **{target_unique_count:,}**"
)

if target_unique_count < 2:
    st.error(
        "Target에 하나의 값만 존재합니다."
    )

elif target_unique_count > 20:
    st.warning(
        "Target의 고유값 수가 많습니다. "
        "연속형 변수라면 분류가 아니라 회귀 모델을 사용해야 합니다."
    )


feature_candidates = [
    column
    for column in numeric_columns
    if column != target_column
]

selected_features = st.multiselect(
    "학습에 사용할 Feature",
    options=feature_candidates,
    default=feature_candidates,
)


# =========================================================
# 10. Model Training
# =========================================================
train_button = st.button(
    "Train Model",
    type="primary",
    use_container_width=True,
)

if train_button:
    if target_unique_count < 2:
        st.error(
            "Target에 최소 2개의 클래스가 필요합니다."
        )
        st.stop()

    if not selected_features:
        st.error(
            "최소 하나 이상의 Feature를 선택해주세요."
        )
        st.stop()

    model_data = df[
        selected_features + [target_column]
    ].dropna()

    if len(model_data) < 10:
        st.error(
            "결측값 제거 후 학습 데이터가 너무 적습니다."
        )
        st.stop()

    X = model_data[selected_features]
    y = model_data[target_column]

    class_counts = y.value_counts()

    if class_counts.min() < 2:
        st.error(
            "일부 클래스의 관측치가 1개뿐입니다."
        )
        st.stop()

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )

    except ValueError as error:
        st.error(
            f"Train/Test 분할 오류: {error}"
        )
        st.stop()

    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )

    else:
        model = LogisticRegression(
            max_iter=2_000,
            random_state=42,
            class_weight="balanced",
        )

    try:
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

    except Exception as error:
        st.error(
            f"모델 학습 중 오류가 발생했습니다: {error}"
        )
        st.stop()

    st.session_state["model_result"] = {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y": y,
        "predictions": predictions,
        "selected_features": selected_features,
        "model_type": model_type,
        "target_column": target_column,
        "test_size": test_size,
    }

    st.success(
        f"{model_type} 학습 완료: "
        f"Train {len(X_train):,}건 / "
        f"Test {len(X_test):,}건"
    )


# =========================================================
# 11. Saved Model Results
# =========================================================
if "model_result" not in st.session_state:
    st.info(
        "모델 설정을 완료한 후 Train Model을 눌러주세요."
    )
    st.stop()


result = st.session_state["model_result"]

model = result["model"]
X_test = result["X_test"]
y_test = result["y_test"]
y = result["y"]
predictions = result["predictions"]
selected_features = result["selected_features"]
trained_model_type = result["model_type"]
trained_target_column = result["target_column"]


st.caption(
    f"현재 표시 중인 모델: {trained_model_type} | "
    f"Target: {trained_target_column}"
)


# =========================================================
# 12. Model Performance
# =========================================================
accuracy = accuracy_score(
    y_test,
    predictions,
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

metric_results = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
}


if (
    hasattr(model, "predict_proba")
    and y.nunique() == 2
):
    positive_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    auc = roc_auc_score(
        y_test,
        positive_probabilities,
    )

    metric_results["ROC-AUC"] = auc

elif (
    hasattr(model, "predict_proba")
    and y.nunique() > 2
):
    try:
        multiclass_probabilities = (
            model.predict_proba(X_test)
        )

        auc = roc_auc_score(
            y_test,
            multiclass_probabilities,
            multi_class="ovr",
            average="weighted",
            labels=model.classes_,
        )

        metric_results["ROC-AUC"] = auc

    except ValueError:
        pass


st.subheader("Model Performance")

performance_columns = st.columns(
    len(metric_results)
)

for column, (
    metric_name,
    metric_value,
) in zip(
    performance_columns,
    metric_results.items(),
):
    with column:
        st.metric(
            metric_name,
            f"{metric_value:.3f}",
        )


if recall < 0.70:
    st.warning(
        "Weighted Recall이 0.70 미만입니다. "
        "일부 실제 클래스를 놓칠 가능성이 있습니다."
    )


# =========================================================
# 13. Class Distribution
# =========================================================
st.subheader("Class Distribution")

class_distribution = (
    y.value_counts(dropna=False)
    .rename_axis("Class")
    .reset_index(name="Count")
)

class_distribution["Rate (%)"] = (
    class_distribution["Count"]
    / class_distribution["Count"].sum()
    * 100
).round(2)

st.dataframe(
    class_distribution,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# 14. Confusion Matrix
# =========================================================
st.subheader("Confusion Matrix")

confusion_matrix_values = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_,
)

confusion_figure = px.imshow(
    confusion_matrix_values,
    text_auto=True,
    x=[
        f"Predicted {class_name}"
        for class_name in model.classes_
    ],
    y=[
        f"Actual {class_name}"
        for class_name in model.classes_
    ],
    title="Confusion Matrix",
    aspect="auto",
    color_continuous_scale="Blues",
)

confusion_figure.update_layout(
    template="plotly_white",
    coloraxis_showscale=False,
    height=max(
        450,
        len(model.classes_) * 60,
    ),
)

st.plotly_chart(
    confusion_figure,
    use_container_width=True,
)


# =========================================================
# 15. Binary Error Analysis
# =========================================================
if y.nunique() == 2:
    tn, fp, fn, tp = (
        confusion_matrix_values.ravel()
    )

    confusion_columns = st.columns(4)

    with confusion_columns[0]:
        st.metric(
            "True Positive",
            f"{tp:,}",
        )

    with confusion_columns[1]:
        st.metric(
            "False Positive",
            f"{fp:,}",
        )

    with confusion_columns[2]:
        st.metric(
            "False Negative",
            f"{fn:,}",
        )

    with confusion_columns[3]:
        st.metric(
            "True Negative",
            f"{tn:,}",
        )


    result_df = X_test.copy()

    result_df["Actual"] = (
        y_test.to_numpy()
    )

    result_df["Prediction"] = predictions

    all_probabilities = model.predict_proba(
        X_test
    )

    predicted_class_indices = np.argmax(
        all_probabilities,
        axis=1,
    )

    result_df["Prediction Probability"] = (
        all_probabilities[
            np.arange(len(X_test)),
            predicted_class_indices,
        ]
    )

    negative_class = model.classes_[0]
    positive_class = model.classes_[1]

    false_positive_df = result_df[
        (
            result_df["Actual"]
            == negative_class
        )
        & (
            result_df["Prediction"]
            == positive_class
        )
    ]

    false_negative_df = result_df[
        (
            result_df["Actual"]
            == positive_class
        )
        & (
            result_df["Prediction"]
            == negative_class
        )
    ]


    # -----------------------------------------------------
    # FP vs FN Feature Comparison
    # -----------------------------------------------------
    st.subheader(
        "FP vs FN Feature Comparison"
    )

    comparison_features = [
        column
        for column in selected_features
        if (
            column in result_df.columns
            and pd.api.types.is_numeric_dtype(
                result_df[column]
            )
        )
    ]

    if (
        false_positive_df.empty
        and false_negative_df.empty
    ):
        st.success(
            "False Positive와 False Negative가 없습니다."
        )

    elif not comparison_features:
        st.info(
            "비교할 수치형 Feature가 없습니다."
        )

    else:
        fp_feature_mean = (
            false_positive_df[
                comparison_features
            ]
            .mean()
            .rename(
                "False Positive Mean"
            )
        )

        fn_feature_mean = (
            false_negative_df[
                comparison_features
            ]
            .mean()
            .rename(
                "False Negative Mean"
            )
        )

        feature_comparison_df = pd.concat(
            [
                fp_feature_mean,
                fn_feature_mean,
            ],
            axis=1,
        )

        feature_comparison_df[
            "Mean Difference"
        ] = (
            feature_comparison_df[
                "False Positive Mean"
            ]
            - feature_comparison_df[
                "False Negative Mean"
            ]
        )

        feature_comparison_df[
            "Absolute Difference"
        ] = (
            feature_comparison_df[
                "Mean Difference"
            ].abs()
        )

        feature_comparison_df = (
            feature_comparison_df
            .reset_index()
            .rename(
                columns={
                    "index": "Feature"
                }
            )
            .sort_values(
                by="Absolute Difference",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.caption(
            f"False Positive {len(false_positive_df):,}건과 "
            f"False Negative {len(false_negative_df):,}건의 "
            "Feature 평균 비교"
        )

        st.dataframe(
            feature_comparison_df.style.format(
                {
                    "False Positive Mean": "{:.3f}",
                    "False Negative Mean": "{:.3f}",
                    "Mean Difference": "{:.3f}",
                    "Absolute Difference": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


    # -----------------------------------------------------
    # Error Samples
    # -----------------------------------------------------
    st.subheader("Error Analysis")

    error_type = st.selectbox(
        "Error Type",
        [
            "False Positive",
            "False Negative",
        ],
        key="classification_error_type",
    )

    if error_type == "False Positive":
        st.write(
            f"False Positive: "
            f"**{len(false_positive_df):,}건**"
        )

        if false_positive_df.empty:
            st.success(
                "False Positive가 없습니다."
            )
        else:
            st.dataframe(
                false_positive_df,
                use_container_width=True,
            )

    else:
        st.write(
            f"False Negative: "
            f"**{len(false_negative_df):,}건**"
        )

        if false_negative_df.empty:
            st.success(
                "False Negative가 없습니다."
            )
        else:
            st.dataframe(
                false_negative_df,
                use_container_width=True,
            )


# =========================================================
# 16. Random Forest Feature Importance
# =========================================================
if trained_model_type == "Random Forest":
    st.subheader("Feature Importance")

    feature_importance_df = (
        pd.DataFrame(
            {
                "Feature": selected_features,
                "Importance": (
                    model.feature_importances_
                ),
            }
        )
        .sort_values(
            by="Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance_threshold = (
        st.sidebar.slider(
            "Importance Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.01,
            step=0.01,
            key=(
                "feature_importance_threshold"
            ),
        )
    )

    filtered_feature_importance = (
        feature_importance_df[
            feature_importance_df[
                "Importance"
            ] >= importance_threshold
        ].copy()
    )

    if filtered_feature_importance.empty:
        st.info(
            f"Importance가 {importance_threshold:.2f} 이상인 "
            "Feature가 없습니다."
        )

    else:
        importance_figure = px.bar(
            filtered_feature_importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title=(
                "Feature Importance "
                f"(Threshold ≥ {importance_threshold:.2f})"
            ),
        )

        importance_figure.update_layout(
            yaxis={
                "categoryorder":
                "total ascending"
            },
            template="plotly_white",
            height=max(
                400,
                len(
                    filtered_feature_importance
                )
                * 35,
            ),
        )

        st.plotly_chart(
            importance_figure,
            use_container_width=True,
        )

        st.dataframe(
            filtered_feature_importance.style.format(
                {
                    "Importance": "{:.5f}"
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# 17. SHAP Analysis
# =========================================================
if trained_model_type == "Random Forest":
    st.header("6. SHAP Analysis")

    if X_test.empty:
        st.warning(
            "SHAP 분석에 사용할 테스트 데이터가 없습니다."
        )

    else:
        # TreeExplainer 생성 여부
        try:
            explainer = shap.TreeExplainer(model)

        except Exception as error:
            st.exception(error)
            st.error(
                "SHAP TreeExplainer를 생성하지 못했습니다."
            )

        else:
            # =================================================
            # 17-1. Global SHAP Analysis
            # =================================================
            st.subheader("Global SHAP Analysis")

            max_shap_samples = min(
                500,
                len(X_test),
            )

            shap_sample_size = st.sidebar.slider(
                "SHAP Sample Size",
                min_value=1,
                max_value=max_shap_samples,
                value=min(
                    100,
                    max_shap_samples,
                ),
                step=1,
                key="shap_sample_size",
            )

            shap_class = st.sidebar.selectbox(
                "Global SHAP 분석 클래스",
                options=list(model.classes_),
                index=(
                    1
                    if len(model.classes_) == 2
                    else 0
                ),
                key="global_shap_class",
            )

            shap_class_index = int(
                np.where(
                    model.classes_ == shap_class
                )[0][0]
            )

            run_global_shap = st.button(
                "Run Global SHAP",
                use_container_width=True,
                key="run_global_shap",
            )

            if run_global_shap:
                X_shap = X_test.sample(
                    n=shap_sample_size,
                    random_state=42,
                )

                try:
                    with st.spinner(
                        "Global SHAP 값을 계산하고 있습니다."
                    ):
                        global_shap_result = explainer(
                            X_shap
                        )

                    global_values = np.asarray(
                        global_shap_result.values
                    )

                    # Random Forest 분류
                    # shape: samples × features × classes
                    if global_values.ndim == 3:
                        global_shap_array = global_values[
                            :,
                            :,
                            shap_class_index,
                        ]

                    # shape: samples × features
                    elif global_values.ndim == 2:
                        global_shap_array = global_values

                    else:
                        raise ValueError(
                            "지원하지 않는 Global SHAP 배열 형태입니다. "
                            f"shape={global_values.shape}"
                        )

                    # 계산 결과 저장
                    st.session_state[
                        "global_shap_result"
                    ] = {
                        "X_shap": X_shap,
                        "shap_array": global_shap_array,
                        "shap_class": shap_class,
                    }

                    st.success(
                        "Global SHAP 계산이 완료되었습니다."
                    )

                except Exception as error:
                    st.exception(error)
                    st.error(
                        "Global SHAP 계산에 실패했습니다."
                    )

            # 저장된 Global SHAP 결과 출력
            if (
                "global_shap_result"
                in st.session_state
            ):
                global_result = st.session_state[
                    "global_shap_result"
                ]

                X_shap = global_result["X_shap"]
                global_shap_array = global_result[
                    "shap_array"
                ]
                analyzed_class = global_result[
                    "shap_class"
                ]

                st.caption(
                    f"현재 표시 중인 SHAP 클래스: "
                    f"{analyzed_class}"
                )

                # ---------------------------------------------
                # Summary Plot
                # ---------------------------------------------
                try:
                    st.markdown(
                        "#### SHAP Summary Plot"
                    )

                    plt.figure(
                        figsize=(10, 6)
                    )

                    shap.summary_plot(
                        global_shap_array,
                        X_shap,
                        show=False,
                    )

                    summary_figure = plt.gcf()
                    plt.tight_layout()

                    st.pyplot(
                        summary_figure,
                        clear_figure=True,
                    )

                    plt.close(
                        summary_figure
                    )

                except Exception as error:
                    st.exception(error)
                    st.warning(
                        "SHAP Summary Plot을 생성하지 못했습니다."
                    )

                # ---------------------------------------------
                # Global SHAP Importance
                # ---------------------------------------------
                try:
                    mean_absolute_shap = np.abs(
                        global_shap_array
                    ).mean(axis=0)

                    shap_importance_df = (
                        pd.DataFrame(
                            {
                                "Feature": X_shap.columns,
                                "Mean Absolute SHAP":
                                mean_absolute_shap,
                            }
                        )
                        .sort_values(
                            by="Mean Absolute SHAP",
                            ascending=False,
                        )
                        .reset_index(drop=True)
                    )

                    max_shap_importance = float(
                        shap_importance_df[
                            "Mean Absolute SHAP"
                        ].max()
                    )

                    if (
                        np.isfinite(
                            max_shap_importance
                        )
                        and max_shap_importance > 0
                    ):
                        shap_threshold = (
                            st.sidebar.slider(
                                "SHAP Importance Threshold",
                                min_value=0.0,
                                max_value=max_shap_importance,
                                value=0.0,
                                step=max(
                                    max_shap_importance
                                    / 100,
                                    0.0001,
                                ),
                                key=(
                                    "shap_importance_threshold"
                                ),
                            )
                        )

                    else:
                        shap_threshold = 0.0

                    filtered_shap_df = (
                        shap_importance_df[
                            shap_importance_df[
                                "Mean Absolute SHAP"
                            ] >= shap_threshold
                        ]
                        .copy()
                    )

                    st.markdown(
                        "#### Global SHAP Feature Importance"
                    )

                    if filtered_shap_df.empty:
                        st.info(
                            "Threshold를 만족하는 "
                            "Feature가 없습니다."
                        )

                    else:
                        shap_bar_figure = px.bar(
                            filtered_shap_df,
                            x="Mean Absolute SHAP",
                            y="Feature",
                            orientation="h",
                            title=(
                                "Global SHAP "
                                "Feature Importance"
                            ),
                        )

                        shap_bar_figure.update_layout(
                            yaxis={
                                "categoryorder":
                                "total ascending"
                            },
                            template="plotly_white",
                            height=max(
                                400,
                                len(
                                    filtered_shap_df
                                )
                                * 35,
                            ),
                        )

                        st.plotly_chart(
                            shap_bar_figure,
                            use_container_width=True,
                        )

                        st.dataframe(
                            filtered_shap_df.style.format(
                                {
                                    "Mean Absolute SHAP":
                                    "{:.5f}",
                                }
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:
                    st.exception(error)
                    st.warning(
                        "Global SHAP 중요도 테이블을 "
                        "생성하지 못했습니다."
                    )

            else:
                st.info(
                    "Global SHAP은 계산량이 크므로 "
                    "'Run Global SHAP' 버튼을 눌러 실행하세요."
                )


            # =================================================
            # 17-2. Individual Prediction Analysis
            # =================================================
            st.subheader(
                "Individual Prediction Analysis"
            )

            sample_index = st.sidebar.number_input(
                "Sample Index",
                min_value=0,
                max_value=len(X_test) - 1,
                value=0,
                step=1,
                key="shap_sample_index",
            )

            sample = X_test.iloc[
                [sample_index]
            ]

            actual = y_test.iloc[
                sample_index
            ]

            prediction = model.predict(
                sample
            )[0]

            sample_probabilities = (
                model.predict_proba(sample)[0]
            )

            predicted_class_index = int(
                np.where(
                    model.classes_ == prediction
                )[0][0]
            )

            predicted_probability = float(
                sample_probabilities[
                    predicted_class_index
                ]
            )

            is_correct = (
                actual == prediction
            )

            correct_status = (
                "✅ Correct"
                if is_correct
                else "❌ Incorrect"
            )

            (
                confidence_level,
                confidence_status,
            ) = get_confidence_level(
                predicted_probability
            )

            sample_metric_columns = st.columns(4)

            with sample_metric_columns[0]:
                st.metric(
                    "Actual",
                    str(actual),
                )

            with sample_metric_columns[1]:
                st.metric(
                    "Prediction",
                    str(prediction),
                )

            with sample_metric_columns[2]:
                st.metric(
                    "Probability",
                    f"{predicted_probability:.3f}",
                )

            with sample_metric_columns[3]:
                st.metric(
                    "Correct 여부",
                    correct_status,
                )

            st.caption(
                f"Test position: {sample_index} | "
                f"Original index: {sample.index[0]}"
            )

            if is_correct:
                st.success(
                    f"실제값 {actual}을(를) "
                    "올바르게 예측했습니다."
                )
            else:
                st.error(
                    f"실제값은 {actual}이지만 "
                    f"{prediction}(으)로 잘못 예측했습니다."
                )


            # =================================================
            # 17-3. Individual SHAP
            # =================================================
            try:
                sample_shap_result = explainer(
                    sample
                )

                sample_values = np.asarray(
                    sample_shap_result.values
                )

                if sample_values.ndim == 3:
                    sample_shap_values = (
                        sample_values[
                            0,
                            :,
                            predicted_class_index,
                        ]
                    )

                    base_values = np.asarray(
                        sample_shap_result.base_values
                    )

                    if base_values.ndim == 2:
                        sample_base_value = float(
                            base_values[
                                0,
                                predicted_class_index,
                            ]
                        )
                    else:
                        sample_base_value = float(
                            base_values.reshape(-1)[
                                predicted_class_index
                            ]
                        )

                elif sample_values.ndim == 2:
                    sample_shap_values = (
                        sample_values[0]
                    )

                    sample_base_value = float(
                        np.asarray(
                            sample_shap_result.base_values
                        ).reshape(-1)[0]
                    )

                else:
                    raise ValueError(
                        "지원하지 않는 Individual SHAP "
                        f"배열 형태입니다: {sample_values.shape}"
                    )


                # ---------------------------------------------
                # Waterfall Plot
                # ---------------------------------------------
                st.markdown(
                    "#### SHAP Waterfall Plot"
                )

                sample_explanation = shap.Explanation(
                    values=sample_shap_values,
                    base_values=sample_base_value,
                    data=sample.iloc[
                        0
                    ].to_numpy(),
                    feature_names=(
                        sample.columns.tolist()
                    ),
                )

                shap.plots.waterfall(
                    sample_explanation,
                    max_display=10,
                    show=False,
                )

                waterfall_figure = plt.gcf()

                st.pyplot(
                    waterfall_figure,
                    clear_figure=True,
                )

                plt.close(
                    waterfall_figure
                )


                # ---------------------------------------------
                # Contribution Table
                # ---------------------------------------------
                st.markdown(
                    "#### Feature Contributions"
                )

                contribution_df = pd.DataFrame(
                    {
                        "Feature": sample.columns,
                        "Feature Value":
                        sample.iloc[0].to_numpy(),
                        "SHAP":
                        sample_shap_values,
                    }
                )

                contribution_df[
                    "Absolute SHAP"
                ] = contribution_df[
                    "SHAP"
                ].abs()

                contribution_df[
                    "Direction"
                ] = np.where(
                    contribution_df[
                        "SHAP"
                    ] > 0,
                    "Increase",
                    np.where(
                        contribution_df[
                            "SHAP"
                        ] < 0,
                        "Decrease",
                        "Neutral",
                    ),
                )

                contribution_df = (
                    contribution_df
                    .sort_values(
                        by="Absolute SHAP",
                        ascending=False,
                    )
                    .reset_index(drop=True)
                )

                top_contribution_count = st.slider(
                    "표시할 Contribution 개수",
                    min_value=1,
                    max_value=len(
                        contribution_df
                    ),
                    value=min(
                        10,
                        len(contribution_df),
                    ),
                    step=1,
                    key="shap_contribution_count",
                )

                st.dataframe(
                    contribution_df
                    .head(
                        top_contribution_count
                    )
                    .style.format(
                        {
                            "Feature Value": "{:.4f}",
                            "SHAP": "{:.5f}",
                            "Absolute SHAP": "{:.5f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


                positive_contribution_df = (
                    contribution_df[
                        contribution_df[
                            "SHAP"
                        ] > 0
                    ].copy()
                )

                negative_contribution_df = (
                    contribution_df[
                        contribution_df[
                            "SHAP"
                        ] < 0
                    ].copy()
                )


                positive_tab, negative_tab = (
                    st.tabs(
                        [
                            "Positive Contributions",
                            "Negative Contributions",
                        ]
                    )
                )

                with positive_tab:
                    if (
                        positive_contribution_df
                        .empty
                    ):
                        st.info(
                            "양의 SHAP 기여가 없습니다."
                        )
                    else:
                        st.dataframe(
                            positive_contribution_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                with negative_tab:
                    if (
                        negative_contribution_df
                        .empty
                    ):
                        st.info(
                            "음의 SHAP 기여가 없습니다."
                        )
                    else:
                        st.dataframe(
                            negative_contribution_df,
                            use_container_width=True,
                            hide_index=True,
                        )


                # =================================================
                # 17-4. AI Insight
                # =================================================
                top_positive = (
                    positive_contribution_df
                    .head(3)
                )

                top_negative = (
                    negative_contribution_df
                    .head(3)
                )

                top_contributing_feature = (
                    contribution_df.iloc[0][
                        "Feature"
                    ]
                    if not contribution_df.empty
                    else "N/A"
                )

                top_contributing_shap = (
                    contribution_df.iloc[0][
                        "SHAP"
                    ]
                    if not contribution_df.empty
                    else np.nan
                )

                top_increasing_feature = (
                    positive_contribution_df.iloc[0]["Feature"]
                    if not positive_contribution_df.empty
                    else "N/A"
                )

                top_increasing_shap = (
                    positive_contribution_df.iloc[0]["SHAP"]
                    if not positive_contribution_df.empty
                    else np.nan
                )

                top_decreasing_feature = (
                    negative_contribution_df.iloc[0]["Feature"]
                    if not negative_contribution_df.empty
                    else "N/A"
                )

                top_decreasing_shap = (
                    negative_contribution_df.iloc[0]["SHAP"]
                    if not negative_contribution_df.empty
                    else np.nan
                )

                insight_messages = []

                for _, row in (
                    top_positive.iterrows()
                ):
                    insight_messages.append(
                        f"**{row['Feature']}**가 "
                        "현재 예측을 증가시키는 주요 요인입니다. "
                        f"(SHAP: {row['SHAP']:.4f})"
                    )

                for _, row in (
                    top_negative.iterrows()
                ):
                    insight_messages.append(
                        f"**{row['Feature']}**는 "
                        "현재 예측을 감소시키는 방향으로 작용했습니다. "
                        f"(SHAP: {row['SHAP']:.4f})"
                    )


                # =================================================
                # 17-5. Recommended Action
                # =================================================
                recommendations = []

                if not top_positive.empty:
                    primary_feature = (
                        top_positive.iloc[0][
                            "Feature"
                        ]
                    )

                    recommendations.append(
                        f"{primary_feature} 값을 "
                        "우선 확인하세요."
                    )

                if predicted_probability >= 0.95:
                    recommendations.append(
                        "높은 신뢰도의 예측입니다. "
                        "우선 점검을 권장합니다."
                    )

                elif predicted_probability >= 0.70:
                    recommendations.append(
                        "중간 수준의 신뢰도입니다. "
                        "주요 Feature와 원본 데이터를 "
                        "함께 확인하세요."
                    )

                else:
                    recommendations.append(
                        "예측 신뢰도가 낮습니다. "
                        "추가 데이터 또는 현업 확인이 필요합니다."
                    )

                if not is_correct:
                    recommendations.append(
                        "현재 샘플은 오분류 사례입니다. "
                        "모델 개선 분석 대상으로 검토하세요."
                    )

                recommended_action = (
                    recommendations[0]
                    if recommendations
                    else "추가 확인 필요"
                )


                # =================================================
                # 17-6. Insight Cards
                # =================================================
                st.subheader("AI Insight Summary")

                card_columns = st.columns(4)

                with card_columns[0]:
                    render_insight_card(
                        title="Confidence",
                        value=confidence_level,
                        subtitle=(
                            f"Prediction probability: "
                            f"{predicted_probability:.3f}"
                        ),
                    )

                with card_columns[1]:
                    render_insight_card(
                        title="Top Increasing Feature",
                        value=str(top_increasing_feature),
                        subtitle=(
                            f"SHAP: {top_increasing_shap:.4f}"
                            if pd.notna(top_increasing_shap)
                            else "No positive contribution"
                        ),
                    )

                with card_columns[2]:
                    render_insight_card(
                        title="Top Decreasing Feature",
                        value=str(top_decreasing_feature),
                        subtitle=(
                            f"SHAP: {top_decreasing_shap:.4f}"
                            if pd.notna(top_decreasing_shap)
                            else "No negative contribution"
                        ),
                    )

                with card_columns[3]:
                    render_insight_card(
                        title="Recommended Action",
                        value=recommended_action,
                        subtitle="SHAP 기반 탐색적 제안",
                    )

                st.subheader("AI Insight")

                if insight_messages:
                    for message in insight_messages:
                        st.markdown(
                            f"- {message}"
                        )
                else:
                    st.info(
                        "표시할 SHAP 기여 요인이 없습니다."
                    )


                st.subheader(
                    "Recommended Action"
                )

                for recommendation in recommendations:
                    st.markdown(
                        f"✅ {recommendation}"
                    )


            except Exception as error:
                st.exception(error)

                st.warning(
                    "Individual SHAP 분석에 실패했습니다. "
                    "위 오류 내용을 확인해주세요."
                )

# %%