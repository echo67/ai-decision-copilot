# %%
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
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
st.caption("데이터 탐색, 분포 진단, 상관관계 분석 및 기본 모델링")


# =========================================================
# 2. Utility Functions
# =========================================================
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    """
    업로드된 CSV 파일을 읽습니다.
    일반적인 인코딩 오류를 고려해 여러 인코딩을 순차적으로 시도합니다.
    """
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        "CSV 파일의 인코딩을 확인할 수 없습니다. "
        "UTF-8 또는 CP949 형식으로 저장한 후 다시 업로드해주세요."
    )


def make_strong_correlation_table(
    corr_matrix: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    절댓값이 threshold 이상인 변수 쌍을 반환합니다.
    자기상관과 중복 조합은 제외합니다.
    """
    results = []
    columns = corr_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            correlation = corr_matrix.iloc[i, j]

            if pd.notna(correlation) and abs(correlation) >= threshold:
                results.append(
                    {
                        "Variable 1": columns[i],
                        "Variable 2": columns[j],
                        "Correlation": round(correlation, 3),
                        "Absolute Correlation": round(abs(correlation), 3),
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


def calculate_distribution_health(series: pd.Series) -> dict:
    """
    선택한 수치형 변수의 요약 통계와 Distribution Health Score를 계산합니다.

    참고:
    이 점수는 통계적으로 검증된 표준 지표가 아니라
    탐색적 데이터 진단을 위한 사용자 정의 휴리스틱입니다.
    """
    clean_series = series.dropna()
    total_count = len(series)
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
    outlier_rate = outlier_count / valid_count * 100

    score = 100

    if missing_rate > 5:
        score -= 20

    if zero_rate > 70:
        score -= 15

    if outlier_rate > 5:
        score -= 20

    if pd.notna(skew) and abs(skew) > 2:
        score -= 15

    if pd.notna(kurtosis) and abs(kurtosis) > 7:
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


# =========================================================
# 3. File Upload
# =========================================================
uploaded_file = st.file_uploader(
    "CSV 파일을 업로드하세요",
    type=["csv"],
)

if uploaded_file is None:
    st.info("분석을 시작하려면 CSV 파일을 업로드해주세요.")
    st.stop()

try:
    df = load_csv(uploaded_file)
except Exception as error:
    st.error(f"파일을 읽는 중 오류가 발생했습니다: {error}")
    st.stop()

if df.empty:
    st.warning("업로드한 CSV 파일에 데이터가 없습니다.")
    st.stop()

st.success("파일 업로드 완료!")


# =========================================================
# 4. Basic Data Information
# =========================================================
st.header("1. Data Overview")

rows, columns = df.shape
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())

numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

metric_columns = st.columns(6)

overview_metrics = [
    ("Rows", rows),
    ("Columns", columns),
    ("Missing", missing_count),
    ("Duplicates", duplicate_count),
    ("Numeric", len(numeric_columns)),
    ("Categorical", len(categorical_columns)),
]

for column, (label, value) in zip(metric_columns, overview_metrics):
    with column:
        st.metric(label, f"{value:,}")


overview_tab, missing_tab, summary_tab, preview_tab = st.tabs(
    [
        "Column Information",
        "Missing Values",
        "Summary",
        "Preview",
    ]
)

with overview_tab:
    column_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.notna().sum().values,
            "Unique Count": df.nunique(dropna=True).values,
        }
    )

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True,
    )

with missing_tab:
    missing_table = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
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
        df.describe(include="all").T,
        use_container_width=True,
    )

with preview_tab:
    preview_rows = st.slider(
        "미리보기 행 수",
        min_value=5,
        max_value=min(100, len(df)),
        value=min(10, len(df)),
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True,
    )


# =========================================================
# 5. Sidebar Visualization Settings
# =========================================================
st.sidebar.header("Visualization Settings")

if not numeric_columns:
    st.warning(
        "수치형 컬럼이 없어 분포 분석, 상관관계 분석 및 모델링을 수행할 수 없습니다."
    )
    st.stop()

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
st.header("2. Distribution Visualization")

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
        title=f"{selected_column} Distribution",
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
        title=f"{selected_column} Box Plot",
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
        title=f"{selected_column} Violin Plot",
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

distribution_result = calculate_distribution_health(
    df[selected_column]
)

first_metric_row = st.columns(4)

with first_metric_row[0]:
    st.metric(
        "Mean",
        (
            f"{distribution_result['mean']:.3f}"
            if pd.notna(distribution_result["mean"])
            else "N/A"
        ),
    )

with first_metric_row[1]:
    st.metric(
        "Median",
        (
            f"{distribution_result['median']:.3f}"
            if pd.notna(distribution_result["median"])
            else "N/A"
        ),
    )

with first_metric_row[2]:
    st.metric(
        "Standard Deviation",
        (
            f"{distribution_result['std']:.3f}"
            if pd.notna(distribution_result["std"])
            else "N/A"
        ),
    )

with first_metric_row[3]:
    st.metric(
        "Skewness",
        (
            f"{distribution_result['skew']:.3f}"
            if pd.notna(distribution_result["skew"])
            else "N/A"
        ),
    )

second_metric_row = st.columns(4)

with second_metric_row[0]:
    st.metric(
        "Kurtosis",
        (
            f"{distribution_result['kurtosis']:.3f}"
            if pd.notna(distribution_result["kurtosis"])
            else "N/A"
        ),
    )

with second_metric_row[1]:
    st.metric(
        "Zero Rate",
        f"{distribution_result['zero_rate']:.2f}%",
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
        delta=f"{distribution_result['outlier_rate']:.2f}%",
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
        f"선택한 변수는 {skew_direction}으로 치우친 분포입니다."
    )

if (
    pd.notna(distribution_result["zero_rate"])
    and distribution_result["zero_rate"] > 50
):
    st.warning(
        "전체 유효 관측치 중 0의 비율이 50%를 초과합니다. "
        "Zero-inflated 데이터일 가능성을 검토해보세요."
    )

st.subheader("Distribution Health")

health_column1, health_column2 = st.columns([1, 3])

with health_column1:
    st.metric(
        "Health Score",
        f"{distribution_result['score']} / 100",
    )

with health_column2:
    st.markdown(
        f"### {distribution_result['level']}"
    )

    st.progress(
        distribution_result["score"] / 100
    )

st.caption(
    "Distribution Health Score는 결측률, 0 비율, 이상치 비율, "
    "왜도 및 첨도를 이용한 탐색적 휴리스틱 점수입니다."
)


# =========================================================
# 8. Correlation Analysis
# =========================================================
st.header("4. Correlation Analysis")

numeric_df = df[numeric_columns]
correlation_matrix = numeric_df.corr()

if correlation_matrix.empty:
    st.info("상관관계를 계산할 수 있는 수치형 변수가 없습니다.")

elif len(numeric_columns) < 2:
    st.info("상관관계 분석에는 최소 2개의 수치형 변수가 필요합니다.")

else:
    correlation_tab, heatmap_tab, strong_tab = st.tabs(
        [
            "Correlation Matrix",
            "Correlation Heatmap",
            "Strong Correlations",
        ]
    )

    with correlation_tab:
        st.dataframe(
            correlation_matrix.style.format("{:.3f}"),
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
            height=max(500, len(numeric_columns) * 35),
        )

        st.plotly_chart(
            correlation_figure,
            use_container_width=True,
        )

    with strong_tab:
        strong_correlation_df = make_strong_correlation_table(
            correlation_matrix,
            correlation_threshold,
        )

        if strong_correlation_df.empty:
            st.info(
                f"절댓값이 {correlation_threshold:.2f} 이상인 "
                "상관관계가 없습니다."
            )

        else:
            st.dataframe(
                strong_correlation_df,
                use_container_width=True,
                hide_index=True,
            )


# =========================================================
# 9. Model Selection
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

target_unique_count = df[target_column].nunique(dropna=True)

st.write(f"선택한 Target: **{target_column}**")
st.write(f"Target 고유값 수: **{target_unique_count:,}**")

if target_unique_count < 2:
    st.error(
        "Target에 하나의 값만 존재하여 모델을 학습할 수 없습니다."
    )

elif target_unique_count > 20:
    st.warning(
        "선택한 Target의 고유값이 많습니다. "
        "현재 모델은 분류 모델이므로 연속형 Target에는 적절하지 않습니다. "
        "연속형 변수라면 Regression 모델을 사용해야 합니다."
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

if st.button(
    "Train Model",
    type="primary",
    use_container_width=True,
):
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
            "일부 클래스의 관측치가 1개뿐이므로 "
            "Train/Test 분할을 수행할 수 없습니다."
        )
        st.stop()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
    except ValueError as error:
        st.error(
            f"Train/Test 분할 중 오류가 발생했습니다: {error}"
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
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

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
    }

if "model_result" in st.session_state:
    result = st.session_state["model_result"]
    model = result["model"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    y= result["y"]
    predictions = result["predictions"]
    selected_features = result["selected_features"]
    trained_model_type = result["model_type"]

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

    # Binary classification ROC-AUC
    if (
        hasattr(model, "predict_proba")
        and y.nunique() == 2
    ):
        prediction_probabilities = model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            prediction_probabilities,
        )

        metric_results["ROC-AUC"] = auc

    # Multiclass ROC-AUC
    elif (
        hasattr(model, "predict_proba")
        and y.nunique() > 2
    ):
        try:
            prediction_probabilities = model.predict_proba(
                X_test
            )

            auc = roc_auc_score(
                y_test,
                prediction_probabilities,
                multi_class="ovr",
                average="weighted",
                labels=model.classes_,
            )

            metric_results["ROC-AUC"] = auc

        except ValueError:
            pass

    st.subheader("Model Performance")

    result_columns = st.columns(
        len(metric_results)
    )

    for column, (metric_name, metric_value) in zip(
        result_columns,
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
    # Confusion Matrix and Error Analysis
    # =========================================================
    st.subheader("Confusion Matrix")

    if y.nunique() == 2:
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_,
        )

        tn, fp, fn, tp = cm.ravel()

        confusion_columns = st.columns(4)

        with confusion_columns[0]:
            st.metric("True Positive", f"{tp:,}")

        with confusion_columns[1]:
            st.metric("False Positive", f"{fp:,}")

        with confusion_columns[2]:
            st.metric("False Negative", f"{fn:,}")

        with confusion_columns[3]:
            st.metric("True Negative", f"{tn:,}")

        # Confusion matrix 시각화
        confusion_figure = px.imshow(
            cm,
            text_auto=True,
            x=[
                f"Predicted {model.classes_[0]}",
                f"Predicted {model.classes_[1]}",
            ],
            y=[
                f"Actual {model.classes_[0]}",
                f"Actual {model.classes_[1]}",
            ],
            title="Confusion Matrix",
            aspect="auto",
            color_continuous_scale="Blues",
        )

        confusion_figure.update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            height=450,
        )

        st.plotly_chart(
            confusion_figure,
            use_container_width=True,
        )

        # 예측 결과 데이터 생성
        result_df = X_test.copy()

        result_df["Actual"] = y_test.to_numpy()
        result_df["Prediction"] = predictions

        if "prediction_probabilities" in locals():
            result_df["Prediction Probability"] = (
                prediction_probabilities
            )

        negative_class = model.classes_[0]
        positive_class = model.classes_[1]

        # False Positive
        false_positive_df = result_df[
            (result_df["Actual"] == negative_class)
            & (result_df["Prediction"] == positive_class)
        ]

        # False Negative
        false_negative_df = result_df[
            (result_df["Actual"] == positive_class)
            & (result_df["Prediction"] == negative_class)
        ]

        # =========================================================
        # False Positive vs False Negative Feature Comparison
        # =========================================================
        st.subheader("FP vs FN Feature Comparison")

        # Actual, Prediction을 제외한 수치형 Feature만 선택
        comparison_features = [
            column
            for column in selected_features
            if column in result_df.columns
            and pd.api.types.is_numeric_dtype(result_df[column])
        ]

        if false_positive_df.empty and false_negative_df.empty:
            st.success(
                "False Positive와 False Negative가 모두 없어 "
                "Feature 평균을 비교할 데이터가 없습니다."
            )

        elif not comparison_features:
            st.info("평균을 비교할 수치형 Feature가 없습니다.")

        else:
            fp_feature_mean = (
                false_positive_df[comparison_features]
                .mean()
                .rename("False Positive Mean")
            )

            fn_feature_mean = (
                false_negative_df[comparison_features]
                .mean()
                .rename("False Negative Mean")
            )

            feature_comparison_df = pd.concat(
                [
                    fp_feature_mean,
                    fn_feature_mean,
                ],
                axis=1,
            )

            # FP 평균과 FN 평균의 차이
            feature_comparison_df["Mean Difference"] = (
                feature_comparison_df["False Positive Mean"]
                - feature_comparison_df["False Negative Mean"]
            )

            feature_comparison_df["Absolute Difference"] = (
                feature_comparison_df["Mean Difference"].abs()
            )

            feature_comparison_df = (
                feature_comparison_df
                .reset_index()
                .rename(columns={"index": "Feature"})
                .sort_values(
                    by="Absolute Difference",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            st.caption(
                f"False Positive {len(false_positive_df):,}건과 "
                f"False Negative {len(false_negative_df):,}건의 "
                "Feature 평균을 비교합니다."
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
                f"실제 클래스는 `{negative_class}`이지만 "
                f"`{positive_class}`로 잘못 예측한 데이터: "
                f"**{len(false_positive_df):,}건**"
            )

            if false_positive_df.empty:
                st.success("False Positive가 없습니다.")
            else:
                st.dataframe(
                    false_positive_df,
                    use_container_width=True,
                )

        else:
            st.write(
                f"실제 클래스는 `{positive_class}`이지만 "
                f"`{negative_class}`로 잘못 예측한 데이터: "
                f"**{len(false_negative_df):,}건**"
            )

            if false_negative_df.empty:
                st.success("False Negative가 없습니다.")
            else:
                st.dataframe(
                    false_negative_df,
                    use_container_width=True,
                )

    else:
        st.info(
            "TP, FP, FN, TN 기반 오류 분석은 현재 이진 분류에서만 제공합니다. "
            "다중 클래스 문제에서는 전체 Confusion Matrix를 확인하세요."
        )

        multiclass_cm = confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_,
        )

        multiclass_figure = px.imshow(
            multiclass_cm,
            text_auto=True,
            x=[
                f"Predicted {class_name}"
                for class_name in model.classes_
            ],
            y=[
                f"Actual {class_name}"
                for class_name in model.classes_
            ],
            title="Multiclass Confusion Matrix",
            aspect="auto",
            color_continuous_scale="Blues",
        )

        multiclass_figure.update_layout(
            template="plotly_white",
            coloraxis_showscale=False,
            height=max(
                450,
                len(model.classes_) * 60,
            ),
        )

        st.plotly_chart(
            multiclass_figure,
            use_container_width=True,
        )

if (
    "model_result" in st.session_state
    and st.session_state["model_result"]["model_type"]
    == "Random Forest"
):
        st.subheader("Feature Importance")
        feature_importance = (
            pd.DataFrame(
                {
                    "Feature": selected_features,
                    "Importance": model.feature_importances_,
                }
            )
            .sort_values(
                by="Importance",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        importance_threshold = st.sidebar.slider(
            "Importance Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.01,
            step=0.01,
            key="feature_importance_threshold",
        )
        filtered_feature_importance = feature_importance[
            feature_importance["Importance"] >= importance_threshold
        ].copy()
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
                    "categoryorder": "total ascending",
                },
                template="plotly_white",
                height=max(
                    400,
                    len(filtered_feature_importance) * 35,
                ),
            )
            st.plotly_chart(
                importance_figure,
                use_container_width=True,
            )
            st.dataframe(
                filtered_feature_importance,
                use_container_width=True,
                hide_index=True,
            )


# %%