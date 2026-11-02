from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ==================================================
# 1. APP CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Employee Analytics | HR Intelligence",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "employee_attrition_clean.csv"

API_URL = "http://127.0.0.1:8000"

FEATURES = [
    "TotalMonthsOfExp",
    "TotalOrgsWorked",
    "MonthsInOrg",
    "LastPayIncrementBand",
    "AverageFeedback"
]


# ==================================================
# 2. LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + ["Attrition"]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return data


try:
    df = load_data()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("HR Intelligence")
st.sidebar.caption("Employee Analytics Platform")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Data Analysis",
        "Visualizations",
        "Attrition Prediction",
        "Employee Segmentation",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning + FastAPI")


# ==================================================
# 4. OVERVIEW
# ==================================================

if page == "Overview":

    st.title("📊 Employee Analytics Dashboard")

    st.write(
        "Explore employee data, analyze patterns, "
        "generate model predictions, and discover "
        "employee segments."
    )

    total_employees = len(df)
    missing_values = int(df.isnull().sum().sum())
    unique_classes = df["Attrition"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Employee Records",
        f"{total_employees:,}"
    )

    col2.metric(
        "Missing Values",
        f"{missing_values:,}"
    )

    col3.metric(
        "Distinct Target Classes",
        unique_classes
    )

    st.subheader("Target Class Distribution")

    class_counts = (
        df["Attrition"]
        .value_counts()
        .sort_index()
        .rename_axis("Target Class")
        .reset_index(name="Employee Count")
    )

    fig = px.bar(
        class_counts,
        x="Target Class",
        y="Employee Count",
        text="Employee Count",
        title="Employee Records by Target Class"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Verify the meaning of target labels 0 and 1 "
        "before interpreting them as attrition outcomes."
    )


# ==================================================
# 5. DATA ANALYSIS
# ==================================================

elif page == "Data Analysis":

    st.title("🔍 Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape**")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

    with col2:
        st.write("**Data Types**")
        st.dataframe(
            df.dtypes.astype(str).rename("Data Type"),
            use_container_width=True
        )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df[FEATURES].describe().T,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing_df = (
        df.isnull()
        .sum()
        .rename("Missing Count")
        .to_frame()
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.subheader("Target Class Counts")

    st.dataframe(
        df["Attrition"].value_counts().sort_index()
        .rename_axis("Target Class")
        .reset_index(name="Count"),
        use_container_width=True
    )


# ==================================================
# 6. VISUALIZATIONS
# ==================================================

elif page == "Visualizations":

    st.title("📈 Employee Data Visualizations")

    selected_feature = st.selectbox(
        "Select a feature",
        FEATURES
    )

    st.subheader(f"Distribution of {selected_feature}")

    fig1 = px.histogram(
        df,
        x=selected_feature,
        color="Attrition",
        barmode="overlay",
        title=f"{selected_feature} Distribution by Target Class"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.subheader("Feature Relationships")

    x_feature = st.selectbox(
        "X-axis",
        FEATURES,
        index=0
    )

    y_feature = st.selectbox(
        "Y-axis",
        FEATURES,
        index=2
    )

    fig2 = px.scatter(
        df,
        x=x_feature,
        y=y_feature,
        color="Attrition",
        title=f"{x_feature} vs {y_feature}"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader("Feature Correlation")

    correlation = df[FEATURES].corr()

    fig3 = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Between Employee Features"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ==================================================
# 7. ATTRITION PREDICTION
# ==================================================

elif page == "Attrition Prediction":

    st.title("🤖 Employee Prediction")

    st.write(
        "Enter employee details to request a prediction "
        "from the FastAPI machine learning service."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            experience = st.number_input(
                "Total Months of Experience",
                min_value=0.0,
                value=60.0
            )

            organizations = st.number_input(
                "Total Organizations Worked",
                min_value=0,
                value=3,
                step=1
            )

            months_in_org = st.number_input(
                "Months in Current Organization",
                min_value=0.0,
                value=24.0
            )

        with col2:
            pay_band = st.number_input(
                "Last Pay Increment Band",
                min_value=0.0,
                value=2.0
            )

            feedback = st.number_input(
                "Average Feedback",
                min_value=0.0,
                value=3.0
            )

        submitted = st.form_submit_button(
            "Predict Employee Class",
            use_container_width=True
        )

    if submitted:

        payload = {
            "TotalMonthsOfExp": experience,
            "TotalOrgsWorked": organizations,
            "MonthsInOrg": months_in_org,
            "LastPayIncrementBand": pay_band,
            "AverageFeedback": feedback
        }

        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15
            )

            if response.status_code == 200:

                result = response.json()

                st.success("Prediction completed")

                st.metric(
                    "Predicted Target Class",
                    result["predicted_class"]
                )

                st.write(
                    "**Model Version:**",
                    result["model_version"]
                )

                probabilities = result[
                    "class_probabilities"
                ]

                probability_df = pd.DataFrame({
                    "Target Class": list(probabilities.keys()),
                    "Estimated Probability": [
                        value * 100
                        for value in probabilities.values()
                    ]
                })

                fig4 = px.bar(
                    probability_df,
                    x="Target Class",
                    y="Estimated Probability",
                    text_auto=".2f",
                    title="Estimated Class Probabilities",
                    labels={
                        "Estimated Probability": "Probability (%)"
                    }
                )

                st.plotly_chart(
                    fig4,
                    use_container_width=True
                )

                st.info(
                    "The predicted class is a model output. "
                    "Verify target-label meanings before "
                    "interpreting it as employee attrition."
                )

            else:
                st.error(
                    f"API returned HTTP {response.status_code}"
                )

                st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to FastAPI. Start the backend "
                "in another terminal and try again."
            )

        except requests.exceptions.Timeout:
            st.error(
                "The prediction request timed out. "
                "Check whether the API is responding."
            )

        except requests.exceptions.RequestException as error:
            st.error(f"Request failed: {error}")


# ==================================================
# 8. EMPLOYEE SEGMENTATION
# ==================================================

elif page == "Employee Segmentation":

    st.title("👥 Employee Segmentation")

    st.write(
        "Use K-Means clustering to group employee records "
        "based on similarities in the selected features."
    )

    number_of_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=6,
        value=3
    )

    @st.cache_data
    def create_segments(data, k):

        clustering_data = data[FEATURES].copy()

        scaler = StandardScaler()

        scaled_data = scaler.fit_transform(
            clustering_data
        )

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        cluster_labels = kmeans.fit_predict(
            scaled_data
        )

        segmented = data.copy()
        segmented["Cluster"] = cluster_labels

        return segmented

    try:
        segmented_df = create_segments(
            df,
            number_of_clusters
        )

        st.subheader("Cluster Summary")

        cluster_summary = (
            segmented_df
            .groupby("Cluster")[FEATURES]
            .mean()
            .round(2)
        )

        st.dataframe(
            cluster_summary,
            use_container_width=True
        )

        st.subheader("Number of Records per Cluster")

        cluster_counts = (
            segmented_df["Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Employee Count")
        )

        fig5 = px.bar(
            cluster_counts,
            x="Cluster",
            y="Employee Count",
            text="Employee Count",
            title="Employee Distribution Across Clusters"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

        st.subheader("Cluster Comparison")

        x_feature = st.selectbox(
            "Select X-axis feature",
            FEATURES,
            index=0,
            key="cluster_x"
        )

        y_feature = st.selectbox(
            "Select Y-axis feature",
            FEATURES,
            index=2,
            key="cluster_y"
        )

        fig6 = px.scatter(
            segmented_df,
            x=x_feature,
            y=y_feature,
            color=segmented_df["Cluster"].astype(str),
            title="Employee Clusters",
            labels={"color": "Cluster"}
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

        st.download_button(
            "Download Segmented Dataset",
            data=segmented_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="employee_segments.csv",
            mime="text/csv"
        )

        st.caption(
            "Clusters are descriptive groups, not proof of "
            "attrition risk or employee performance."
        )

    except Exception as error:
        st.error(
            f"Unable to create employee segments: {error}"
        )


# ==================================================
# 9. ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.markdown("""
    ### Employee Attrition Prediction & HR Analytics Platform

    **Objective**

    Explore employee data, visualize feature patterns,
    generate machine learning predictions, and group
    employees using unsupervised learning.

    **Technologies**

    - Python
    - Pandas
    - Scikit-learn
    - Plotly
    - Streamlit
    - FastAPI
    - Pydantic
    - Joblib

    **Modules**

    1. Data Analysis
    2. Interactive Visualizations
    3. ML Prediction API
    4. Employee Segmentation
    5. Project Documentation

    **Limitations**

    - Prediction quality depends on the training data.
    - Target-label meanings must be verified.
    - Clusters do not establish causation.
    - Model predictions should not be used as the sole
      basis for employment decisions.
    """)