import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)

from xgboost import XGBClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Daniel Etele | AI Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .stApp {
        background: #f5f7fb;
    }

    /* Header */
    .hero {
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 50%,
            #0f766e 100%
        );
        padding: 38px 42px;
        border-radius: 22px;
        margin-bottom: 25px;
        color: white;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
        color: white;
    }

    .hero h2 {
        font-size: 20px;
        font-weight: 500;
        color: #ccfbf1;
        margin-top: 0;
    }

    .hero p {
        color: #e2e8f0;
        font-size: 16px;
        line-height: 1.7;
        max-width: 900px;
    }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        color: #0f172a;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #64748b;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Prediction */
    .prediction-positive {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-left: 6px solid #e11d48;
        padding: 25px;
        border-radius: 16px;
    }

    .prediction-negative {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        border-left: 6px solid #059669;
        padding: 25px;
        border-radius: 16px;
    }

    .prediction-title {
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .prediction-text {
        font-size: 15px;
        color: #475569;
        line-height: 1.7;
    }

    /* Section titles */
    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 20px;
    }

    /* Info boxes */
    .info-box {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px;
        line-height: 1.7;
        color: #334155;
    }

    .portfolio-box {
        background: #0f172a;
        color: white;
        padding: 25px;
        border-radius: 18px;
        margin-top: 20px;
    }

    .portfolio-box h3 {
        color: #5eead4;
    }

    .portfolio-box p {
        color: #cbd5e1;
        line-height: 1.7;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        padding: 35px 0 10px 0;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT CONFIG
# ============================================================

AUTHOR_NAME = "Daniel Etele"
AUTHOR_ROLE = "AI & ML Systems | Expert Systems | Cybersecurity | Lecturer"

GITHUB_URL = "https://github.com/tari007"

PROJECT_TITLE = "AI-Based Heart Disease Prediction System"

PROJECT_DESCRIPTION = (
    "A machine learning healthcare project that compares four classification "
    "models for predicting the likelihood of heart disease from patient-level "
    "clinical and demographic features."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Heart-Disease-Dataset.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


df = load_data()


# ============================================================
# FEATURES
# ============================================================

target_col = "target"

continuous_features = [
    "age",
    "resting bps",
    "cholesterol",
    "max heart rate",
    "oldpeak",
]

categorical_features = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope",
]

feature_columns = continuous_features + categorical_features


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(data):

    X = data[feature_columns]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, continuous_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    results = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
        ).ravel()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "F1 Score": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "Specificity": specificity,
            "ROC-AUC": roc_auc_score(
                y_test,
                y_prob,
            ),
            "PR-AUC": average_precision_score(
                y_test,
                y_prob,
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        }

        trained_models[name] = pipeline

    return (
        trained_models,
        results,
        X_train,
        X_test,
        y_train,
        y_test,
    )


(
    models,
    results,
    X_train,
    X_test,
    y_train,
    y_test,
) = train_models(df)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    f"""
    <div class="hero">

        <h1>🫀 {PROJECT_TITLE}</h1>

        <h2>{AUTHOR_NAME} · {AUTHOR_ROLE}</h2>

        <p>
            {PROJECT_DESCRIPTION}
            This interactive dashboard demonstrates the complete machine
            learning workflow from preprocessing and model evaluation to
            patient-level prediction and model interpretation.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP METRICS
# ============================================================

best_model = max(
    results,
    key=lambda model: results[model]["F1 Score"]
)

best_f1 = results[best_model]["F1 Score"]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Dataset</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-description">
                Patient records
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">Features</div>
            <div class="metric-value">13</div>
            <div class="metric-description">
                Clinical and demographic variables
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Models</div>
            <div class="metric-value">4</div>
            <div class="metric-description">
                Classification algorithms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Best F1</div>
            <div class="metric-value">{best_f1:.1%}</div>
            <div class="metric-description">
                {best_model}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 AI Model Lab")

    st.markdown(
        """
        This dashboard demonstrates a complete
        machine learning prediction workflow.
        """
    )

    st.divider()

    st.markdown("### Project")

    st.write("Heart Disease Prediction")

    st.markdown("### Developer")

    st.write(AUTHOR_NAME)

    st.markdown("### Focus Areas")

    st.write(
        "AI & ML Systems\n\n"
        "Expert Systems\n\n"
        "Healthcare AI\n\n"
        "Cybersecurity\n\n"
        "Data Science"
    )

    st.divider()

    st.markdown("### GitHub")

    st.markdown(
        f"[Visit GitHub Profile]({GITHUB_URL})"
    )

    st.divider()

    st.caption(
        "Educational AI/ML portfolio project. "
        "Not a clinically validated diagnostic system."
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🫀 Prediction",
        "📊 Model Performance",
        "🔍 Prediction Explanation",
        "🧠 About the Project",
    ]
)


# ============================================================
# TAB 1: PREDICTION
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Patient Risk Prediction</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Enter patient characteristics and allow the selected ML model '
        'to estimate the probability of heart disease.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.6, 1])

    with left:

        st.markdown("### Patient Information")

        c1, c2 = st.columns(2)

        with c1:

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=50,
                step=1,
            )

            sex = st.selectbox(
                "Sex",
                options=[0, 1],
                format_func=lambda x:
                    "Female (0)" if x == 0 else "Male (1)",
            )

            chest_pain = st.selectbox(
                "Chest Pain Type",
                options=sorted(
                    df["chest pain type"].dropna().unique()
                ),
            )

            resting_bps = st.number_input(
                "Resting Blood Pressure",
                min_value=50,
                max_value=250,
                value=130,
                step=1,
            )

            cholesterol = st.number_input(
                "Cholesterol",
                min_value=50,
                max_value=700,
                value=220,
                step=1,
            )

        with c2:

            fasting_bs = st.selectbox(
                "Fasting Blood Sugar",
                options=[0, 1],
                format_func=lambda x:
                    "≤ 120 mg/dl (0)"
                    if x == 0
                    else "> 120 mg/dl (1)",
            )

            resting_ecg = st.selectbox(
                "Resting ECG",
                options=sorted(
                    df["resting ecg"].dropna().unique()
                ),
            )

            max_hr = st.number_input(
                "Maximum Heart Rate",
                min_value=50,
                max_value=250,
                value=150,
                step=1,
            )

            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                options=[0, 1],
                format_func=lambda x:
                    "No (0)" if x == 0 else "Yes (1)",
            )

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
            )

            st_slope = st.selectbox(
                "ST Slope",
                options=sorted(
                    df["ST slope"].dropna().unique()
                ),
            )

    with right:

        st.markdown("### AI Configuration")

        selected_model = st.selectbox(
            "Prediction Model",
            list(models.keys()),
            index=list(models.keys()).index("Random Forest"),
        )

        st.info(
            f"""
            **Selected model**

            {selected_model}

            The model was trained using an 80/20 stratified
            train-test split with preprocessing performed
            inside the machine learning pipeline.
            """
        )

        st.markdown("### Prediction")

        predict_button = st.button(
            "🧠 Run AI Prediction",
            type="primary",
            use_container_width=True,
        )

    if predict_button:

        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "chest pain type": [chest_pain],
                "resting bps": [resting_bps],
                "cholesterol": [cholesterol],
                "fasting blood sugar": [fasting_bs],
                "resting ecg": [resting_ecg],
                "max heart rate": [max_hr],
                "exercise angina": [exercise_angina],
                "oldpeak": [oldpeak],
                "ST slope": [st_slope],
            }
        )

        model = models[selected_model]

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0][1]
        )

        st.session_state["last_input"] = input_data
        st.session_state["last_model"] = selected_model
        st.session_state["last_probability"] = probability
        st.session_state["last_prediction"] = prediction

        st.write("")

        if prediction == 1:

            st.markdown(
                f"""
                <div class="prediction-positive">

                    <div class="prediction-title">
                        ⚠️ Elevated Predicted Risk
                    </div>

                    <div class="prediction-text">
                        The selected machine learning model classified
                        this input as <strong>heart disease predicted</strong>.
                    </div>

                    <br>

                    <strong>Predicted probability:</strong>
                    {probability:.1%}

                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-negative">

                    <div class="prediction-title">
                        ✓ Lower Predicted Risk
                    </div>

                    <div class="prediction-text">
                        The selected machine learning model classified
                        this input as <strong>no heart disease predicted</strong>.
                    </div>

                    <br>

                    <strong>Predicted probability:</strong>
                    {probability:.1%}

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric(
                "Disease Probability",
                f"{probability:.1%}",
            )

        with col_b:
            st.metric(
                "Model",
                selected_model,
            )

        with col_c:
            st.metric(
                "Disease Recall",
                f"{results[selected_model]['Recall']:.1%}",
            )

        st.warning(
            "Educational use only. This prediction is not a clinical "
            "diagnosis and must not be used to make medical decisions."
        )


# ============================================================
# TAB 2: MODEL PERFORMANCE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Model Performance Laboratory</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Comparative evaluation of the four machine learning classifiers '
        'on the held-out test set.'
        '</div>',
        unsafe_allow_html=True,
    )

    results_df = pd.DataFrame(results).T

    display_metrics = results_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Specificity",
            "ROC-AUC",
            "PR-AUC",
        ]
    ].copy()

    st.dataframe(
        display_metrics.style.format(
            "{:.2%}"
        ),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">Performance Comparison</div>',
        unsafe_allow_html=True,
    )

    metric_to_plot = st.selectbox(
        "Select metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Specificity",
            "ROC-AUC",
            "PR-AUC",
        ],
    )

    chart_df = (
        results_df[metric_to_plot]
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        chart_df.index,
        chart_df.values,
    )

    ax.set_title(
        f"{metric_to_plot} Comparison",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_ylabel(metric_to_plot)

    ax.set_ylim(
        0,
        min(
            1.05,
            max(chart_df.values) + 0.15
        ),
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    for bar, value in zip(
        bars,
        chart_df.values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">ROC Curve</div>',
        unsafe_allow_html=True,
    )

    fig2, ax2 = plt.subplots(figsize=(9, 6))

    for name, model in models.items():

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            y_prob,
        )

        auc = roc_auc_score(
            y_test,
            y_prob,
        )

        ax2.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc:.3f})",
        )

    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax2.set_xlabel(
        "False Positive Rate"
    )

    ax2.set_ylabel(
        "True Positive Rate"
    )

    ax2.set_title(
        "Receiver Operating Characteristic",
        fontsize=16,
        fontweight="bold",
    )

    ax2.legend()

    ax2.grid(
        alpha=0.2
    )

    plt.tight_layout()

    st.pyplot(
        fig2,
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-title">Confusion Matrix Details</div>',
        unsafe_allow_html=True,
    )

    cm_model = st.selectbox(
        "Select model",
        list(models.keys()),
        key="cm_model",
    )

    cm_results = results[cm_model]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "True Negatives",
            cm_results["TN"],
        )

    with c2:
        st.metric(
            "False Positives",
            cm_results["FP"],
        )

    with c3:
        st.metric(
            "False Negatives",
            cm_results["FN"],
        )

    with c4:
        st.metric(
            "True Positives",
            cm_results["TP"],
        )


# ============================================================
# TAB 3: PREDICTION EXPLANATION
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Prediction Explanation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'A lightweight local sensitivity analysis showing how changing '
        'individual patient inputs can alter the model probability.'
        '</div>',
        unsafe_allow_html=True,
    )

    if "last_input" not in st.session_state:

        st.info(
            "Run a prediction first. The explanation panel will "
            "then analyse the most recent prediction."
        )

    else:

        input_data = st.session_state["last_input"]
        selected_model = st.session_state["last_model"]

        model = models[selected_model]

        baseline_probability = float(
            model.predict_proba(input_data)[0][1]
        )

        explanation_rows = []

        for feature in feature_columns:

            modified = input_data.copy()

            original_value = modified.iloc[0][feature]

            if feature in continuous_features:

                std_value = df[feature].std()

                if pd.isna(std_value) or std_value == 0:
                    continue

                modified_value = original_value + std_value

                modified.iloc[0, modified.columns.get_loc(feature)] = (
                    modified_value
                )

            else:

                values = (
                    df[feature]
                    .dropna()
                    .unique()
                    .tolist()
                )

                alternatives = [
                    value
                    for value in values
                    if value != original_value
                ]

                if not alternatives:
                    continue

                modified_value = alternatives[0]

                modified.iloc[
                    0,
                    modified.columns.get_loc(feature)
                ] = modified_value

            new_probability = float(
                model.predict_proba(modified)[0][1]
            )

            change = (
                new_probability
                - baseline_probability
            )

            explanation_rows.append(
                {
                    "Feature": feature,
                    "Baseline": original_value,
                    "Sensitivity Test": modified_value,
                    "Probability Change": change,
                }
            )

        explanation_df = pd.DataFrame(
            explanation_rows
        )

        explanation_df = explanation_df.sort_values(
            "Probability Change",
            key=lambda x: abs(x),
            ascending=False,
        )

        st.markdown(
            f"""
            <div class="info-box">

            <strong>Model:</strong> {selected_model}<br>

            <strong>Baseline disease probability:</strong>
            {baseline_probability:.1%}<br><br>

            The table below tests how the model probability changes when
            individual inputs are modified while the other inputs remain
            unchanged.

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.dataframe(
            explanation_df.style.format(
                {
                    "Probability Change": "{:+.2%}",
                }
            ),
            use_container_width=True,
        )

        st.info(
            "Interpretation note: this is a local sensitivity analysis. "
            "It is not a causal explanation and should not be interpreted "
            "as clinical feature importance."
        )


# ============================================================
# TAB 4: ABOUT
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">About This AI Project</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="info-box">

        <h3>{PROJECT_TITLE}</h3>

        <p>
        This project demonstrates the application of supervised machine
        learning to a heart disease classification problem.
        </p>

        <p>
        The system compares Logistic Regression, Decision Tree,
        Random Forest and XGBoost classifiers.
        </p>

        <p>
        Data preprocessing is integrated into the machine learning
        pipeline. Continuous variables use median imputation and
        standardisation. Categorical variables use most-frequent
        imputation and one-hot encoding.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Machine Learning Pipeline")

        st.markdown(
            """
            **1. Data ingestion**

            Patient dataset is loaded from CSV.

            **2. Data cleaning**

            Legacy row-number column is removed.

            **3. Feature preprocessing**

            Numerical and categorical features are processed separately.

            **4. Model training**

            Four supervised classification models are trained.

            **5. Evaluation**

            Models are evaluated using accuracy, precision, recall,
            F1, specificity, ROC-AUC and PR-AUC.

            **6. Prediction**

            The trained pipeline accepts new patient characteristics
            and returns a probability estimate.
            """
        )

    with col2:

        st.markdown("### Dataset Features")

        feature_table = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Type": [
                    "Continuous"
                    if f in continuous_features
                    else "Categorical"
                    for f in feature_columns
                ],
            }
        )

        st.dataframe(
            feature_table,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown(
        f"""
        <div class="portfolio-box">

            <h3>👨🏽‍💻 About the Developer</h3>

            <p>
            <strong>{AUTHOR_NAME}</strong> is an AI-focused academic and
            technology professional working across AI & ML Systems,
            Expert Systems, Cybersecurity, Web Development and teaching.
            </p>

            <p>
            This project is part of an AI/ML portfolio demonstrating
            practical model development, evaluation, deployment and
            explainability.
            </p>

            <p>
            GitHub:
            <a href="{GITHUB_URL}"
               target="_blank"
               style="color:#5eead4;">
               github.com/tari007
            </a>
            </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.error(
        "IMPORTANT: This application is an educational machine-learning "
        "project. It is not a clinically validated diagnostic tool. "
        "Do not use its predictions to make medical decisions."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">

        AI/ML Healthcare Portfolio Project ·
        Developed by {AUTHOR_NAME} ·
        Streamlit · Scikit-learn · XGBoost

        <br><br>

        © 2026 {AUTHOR_NAME}

    </div>
    """,
    unsafe_allow_html=True,
)
