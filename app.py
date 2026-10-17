import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CLEAN DASHBOARD STYLE
# =========================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(59, 130, 246, 0.10),
                transparent 28%
            ),
            #07111f;
        color: #e8eef7;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #091525;
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 12px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 18px;
    }

    .metric-card {
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(15, 31, 53, 0.78);
        border-radius: 16px;
        padding: 18px;
        min-height: 105px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    .result-card {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 31, 53, 0.82);
        border-radius: 18px;
        padding: 22px;
        min-height: 135px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .result-value {
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 10px;
    }

    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.22);
        border-radius: 12px;
        padding: 13px 15px;
        color: #f5d28b;
        font-size: 0.82rem;
        line-height: 1.6;
        margin: 10px 0 20px;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 31, 53, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 15px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DATASET CONFIGURATION
# =========================================================

TARGET = "target"

CONTINUOUS = [
    "age",
    "resting bps",
    "cholesterol",
    "max heart rate",
    "oldpeak",
]

CATEGORICAL = [
    "sex",
    "chest pain type",
    "fasting blood sugar",
    "resting ecg",
    "exercise angina",
    "ST slope",
]

FEATURES = CONTINUOUS + CATEGORICAL

DATA_FILE = "Heart-Disease-Dataset.csv"


def get_data_path():
    """Locate the dataset in the application directory."""
    if os.path.exists(DATA_FILE):
        return DATA_FILE

    app_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(app_directory, DATA_FILE)


@st.cache_data
def load_data():
    """Load the dataset."""
    data = pd.read_csv(get_data_path())

    # Remove the row-number column if it exists.
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return data


# =========================================================
# MODEL TRAINING AND EVALUATION
# =========================================================

@st.cache_resource
def train_models(data):
    """
    Train the four models using the same 80/20 stratified split.
    Preprocessing is fitted on the training data only.
    """

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocessor = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_preprocessor, CONTINUOUS),
            ("categorical", categorical_preprocessor, CATEGORICAL),
        ]
    )

    model_definitions = {
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

    results = {}

    for model_name, classifier in model_definitions.items():

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                labels=[0, 1],
                zero_division=0,
            )
        )

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test,
            probabilities,
        )

        results[model_name] = {
            "model": model,
            "accuracy": accuracy_score(y_test, predictions),
            "disease_precision": precision[1],
            "disease_recall": recall[1],
            "disease_f1": f1[1],
            "specificity": recall[0],
            "roc_auc": roc_auc_score(y_test, probabilities),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "fpr": false_positive_rate,
            "tpr": true_positive_rate,
        }

    return results, X_test, y_test


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def metric_card(label, value, note=""):
    """Display a dashboard metric card."""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readable_feature(feature):
    """Convert dataset column names into readable labels."""

    labels = {
        "age": "Age",
        "sex": "Sex",
        "chest pain type": "Chest pain type",
        "resting bps": "Resting blood pressure",
        "cholesterol": "Cholesterol",
        "fasting blood sugar": "Fasting blood sugar",
        "resting ecg": "Resting ECG",
        "max heart rate": "Maximum heart rate",
        "exercise angina": "Exercise-induced angina",
        "oldpeak": "Oldpeak",
        "ST slope": "ST slope",
    }

    return labels.get(feature, feature)


def explain_prediction(model, patient, reference_row):
    """
    Compare the patient's prediction with predictions obtained
    by replacing one input at a time with a reference value.

    This is local sensitivity analysis, not causal explanation.
    """

    base_probability = float(
        model.predict_proba(patient)[0, 1]
    )

    impacts = []

    for feature in FEATURES:

        altered_patient = patient.copy()

        altered_patient[feature] = reference_row[feature]

        altered_probability = float(
            model.predict_proba(altered_patient)[0, 1]
        )

        impacts.append(
            {
                "Feature": feature,
                "Impact": base_probability - altered_probability,
                "Reference probability": altered_probability,
            }
        )

    explanation = pd.DataFrame(impacts)

    explanation["Absolute impact"] = explanation["Impact"].abs()

    explanation = explanation.sort_values(
        "Absolute impact",
        ascending=False,
    )

    return base_probability, explanation


# =========================================================
# LOAD DATA AND TRAIN MODELS
# =========================================================

try:
    df = load_data()

    with st.spinner("Preparing data and evaluating models..."):
        results, X_test, y_test = train_models(df)

except Exception as error:
    st.error(
        "The application could not load the dataset or train the models."
    )
    st.exception(error)
    st.stop()


# Create the model comparison table.
comparison = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Accuracy": result["accuracy"],
            "Disease Precision": result["disease_precision"],
            "Disease Recall": result["disease_recall"],
            "Disease F1": result["disease_f1"],
            "Specificity": result["specificity"],
            "ROC-AUC": result["roc_auc"],
            "TP": result["tp"],
            "FN": result["fn"],
            "FP": result["fp"],
            "TN": result["tn"],
        }
        for model_name, result in results.items()
    ]
).sort_values(
    "Disease F1",
    ascending=False,
)

best_model = comparison.iloc[0]["Model"]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🫀 Prediction Dashboard")

    st.caption(
        "Explore patient predictions and compare machine-learning models."
    )

    st.divider()

    st.markdown("### Dashboard sections")

    st.markdown(
        """
        - Patient prediction
        - Model performance
        - Prediction explanation
        - Dataset and methodology
        """
    )

    st.divider()

    st.caption(
        "Educational machine-learning demonstration. "
        "Not a medical diagnostic service."
    )


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important medical notice:</strong>
        This application is an educational machine-learning project.
        It is not a clinically validated diagnostic tool.
        Predictions must not be used to make medical decisions,
        diagnose a condition, or replace advice from a qualified
        healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

st.markdown("### Dashboard overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "Patient records",
        f"{len(df):,}",
        "Records in the dataset",
    )

with c2:
    metric_card(
        "Input variables",
        f"{len(FEATURES)}",
        "Variables used by the models",
    )

with c3:
    metric_card(
        "Models evaluated",
        "4",
        "Classification algorithms",
    )

with c4:
    metric_card(
        "Test records",
        f"{len(y_test):,}",
        "Stratified holdout set",
    )


st.markdown(
    "<div class='section-title'>Explore the dashboard</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        Enter patient information, compare model performance,
        and inspect how changes to individual inputs affect
        the model's predicted probability.
    </div>
    """,
    unsafe_allow_html=True,
)


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔎 Prediction",
        "📊 Model Performance",
        "🧠 Prediction Explanation",
        "📘 Methodology",
    ]
)


# =========================================================
# TAB 1: PATIENT PREDICTION
# =========================================================

with tab1:

    st.markdown("### Enter patient information")

    st.caption(
        "Select values for the variables used by the model."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=53,
        )

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda value: (
                "Female (0)" if value == 0 else "Male (1)"
            ),
        )

        chest_pain = st.selectbox(
            "Chest pain type",
            [1, 2, 3, 4],
            format_func=lambda value: {
                1: "1 · Typical angina",
                2: "2 · Atypical angina",
                3: "3 · Non-anginal pain",
                4: "4 · Asymptomatic",
            }[value],
        )

        resting_bps = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=80.0,
            max_value=250.0,
            value=130.0,
        )

    with col2:

        cholesterol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=50.0,
            max_value=700.0,
            value=240.0,
        )

        fasting_bs = st.selectbox(
            "Fasting blood sugar",
            [0, 1],
            format_func=lambda value: (
                "0 · At or below 120 mg/dl"
                if value == 0
                else "1 · Above 120 mg/dl"
            ),
        )

        resting_ecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2],
            format_func=lambda value: {
                0: "0 · Normal",
                1: "1 · ST-T wave abnormality",
                2: "2 · Left ventricular hypertrophy",
            }[value],
        )

        max_hr = st.number_input(
            "Maximum heart rate (bpm)",
            min_value=50.0,
            max_value=250.0,
            value=145.0,
        )

    with col3:

        exercise_angina = st.selectbox(
            "Exercise-induced angina",
            [0, 1],
            format_func=lambda value: (
                "0 · No" if value == 0 else "1 · Yes"
            ),
        )

        oldpeak = st.number_input(
            "Oldpeak (ST depression)",
            min_value=-2.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )

        st_slope = st.selectbox(
            "ST slope",
            [0, 1, 2, 3],
        )

        model_name = st.selectbox(
            "Prediction model",
            list(results.keys()),
            index=list(results.keys()).index(best_model),
        )

    patient = pd.DataFrame(
        [
            {
                "age": age,
                "sex": sex,
                "chest pain type": chest_pain,
                "resting bps": resting_bps,
                "cholesterol": cholesterol,
                "fasting blood sugar": fasting_bs,
                "resting ecg": resting_ecg,
                "max heart rate": max_hr,
                "exercise angina": exercise_angina,
                "oldpeak": oldpeak,
                "ST slope": st_slope,
            }
        ]
    )[FEATURES]

    if st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    ):

        st.session_state["patient"] = patient
        st.session_state["model_name"] = model_name
        st.session_state["predicted"] = True

    if st.session_state.get("predicted", False):

        patient = st.session_state["patient"]
        selected_model_name = st.session_state["model_name"]

        selected_model = results[selected_model_name]["model"]

        prediction = int(selected_model.predict(patient)[0])

        probability = float(
            selected_model.predict_proba(patient)[0, 1]
        )

        st.divider()

        r1, r2, r3 = st.columns(3)

        with r1:

            classification = (
                "Class 1 · Heart disease"
                if prediction == 1
                else "Class 0 · No heart disease"
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Model classification
                    </div>
                    <div class="result-value">
                        {classification}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Model-estimated probability
                    </div>
                    <div class="result-value">
                        {probability:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:

            recall = results[
                selected_model_name
            ]["disease_recall"]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Selected model
                    </div>
                    <div class="result-value"
                         style="font-size: 1.25rem;">
                        {selected_model_name}
                    </div>
                    <div class="metric-note">
                        Test-set disease recall: {recall:.1%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Predicted probability")

        st.progress(
            min(max(probability, 0.0), 1.0),
            text=f"Class 1 probability: {probability:.1%}",
        )

        st.info(
            "This result reflects the selected model's output. "
            "It is not a confirmed diagnosis or a validated "
            "estimate of an individual's actual medical risk."
        )


# =========================================================
# TAB 2: MODEL PERFORMANCE
# =========================================================

with tab2:

    st.markdown("### Model performance comparison")

    st.caption(
        "The following metrics were calculated on the held-out "
        "20% stratified test set."
    )

    display_columns = [
        "Model",
        "Accuracy",
        "Disease Precision",
        "Disease Recall",
        "Disease F1",
        "Specificity",
        "ROC-AUC",
    ]

    st.dataframe(
        comparison[display_columns].style.format(
            {
                column: "{:.3f}"
                for column in display_columns[1:]
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Compare disease-class metrics")

    fig, ax = plt.subplots(figsize=(11, 5))

    chart_data = comparison.set_index("Model")[
        [
            "Disease Precision",
            "Disease Recall",
            "Disease F1",
        ]
    ]

    chart_data.plot(
        kind="bar",
        ax=ax,
        width=0.78,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Disease-class metrics on the test set")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncols=3)

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### ROC curves")

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, result in results.items():

        ax.plot(
            result["fpr"],
            result["tpr"],
            label=f"{name} · AUC {result['roc_auc']:.3f}",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Confusion matrices")

    matrix_columns = st.columns(4)

    for column, (_, row) in zip(
        matrix_columns,
        comparison.iterrows(),
    ):

        with column:

            st.markdown(f"**{row['Model']}**")

            matrix = np.array(
                [
                    [row["TN"], row["FP"]],
                    [row["FN"], row["TP"]],
                ],
                dtype=int,
            )

            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                ),
                use_container_width=True,
            )

    st.caption(
        "TP = correctly predicted heart disease; "
        "TN = correctly predicted no heart disease; "
        "FP = false positive; FN = false negative."
    )


# =========================================================
# TAB 3: PREDICTION EXPLANATION
# =========================================================

with tab3:

    st.markdown("### Understand the model output")

    st.write(
        "This section uses local sensitivity analysis. It changes "
        "one input at a time to a reference value and measures "
        "how the model's predicted probability changes."
    )

    st.warning(
        "This analysis is not a causal explanation, clinical "
        "feature importance, or proof that a particular feature "
        "caused the prediction."
    )

    if not st.session_state.get("predicted", False):

        st.info(
            "Run a prediction in the Prediction tab to view "
            "the sensitivity analysis."
        )

    else:

        patient = st.session_state["patient"]

        selected_model_name = st.session_state["model_name"]

        selected_model = results[
            selected_model_name
        ]["model"]

        # Use medians for numeric features and the mode for
        # coded categorical features as the reference profile.
        reference = X_test.median(
            numeric_only=True
        ).to_frame().T

        for feature in CATEGORICAL:
            reference[feature] = X_test[feature].mode().iloc[0]

        reference = reference[FEATURES]

        base_probability, explanation = explain_prediction(
            selected_model,
            patient,
            reference.iloc[0],
        )

        top_features = explanation.head(6).copy()

        top_features["Feature"] = top_features[
            "Feature"
        ].map(readable_feature)

        top_features = top_features.sort_values("Impact")

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            top_features["Feature"],
            top_features["Impact"] * 100,
        )

        ax.axvline(0, linewidth=1)

        ax.set_xlabel(
            "Change in class 1 probability (percentage points)"
        )

        ax.set_title(
            f"Local sensitivity analysis · {selected_model_name}"
        )

        ax.grid(axis="x", alpha=0.2)

        plt.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("### Input sensitivity details")

        display_explanation = explanation.head(8).copy()

        display_explanation["Feature"] = (
            display_explanation["Feature"].map(readable_feature)
        )

        display_explanation["Impact"] = (
            display_explanation["Impact"].map(
                lambda value: f"{value:+.1%}"
            )
        )

        display_explanation["Reference probability"] = (
            display_explanation["Reference probability"].map(
                lambda value: f"{value:.1%}"
            )
        )

        st.dataframe(
            display_explanation[
                [
                    "Feature",
                    "Impact",
                    "Reference probability",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            f"Original class 1 probability: {base_probability:.1%}. "
            "A positive impact means the patient's input raises "
            "the model output relative to the reference profile. "
            "A negative impact means it lowers the output."
        )


# =========================================================
# TAB 4: METHODOLOGY
# =========================================================

with tab4:

    st.markdown("### Machine-learning workflow")

    st.write(
        "The application compares four classification algorithms "
        "using the same dataset and stratified train/test split."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("#### Preprocessing")

        st.write(
            "Continuous variables use median imputation and "
            "standardisation. Coded categorical variables use "
            "most-frequent imputation and one-hot encoding."
        )

    with col2:

        st.markdown("#### Evaluation")

        st.write(
            "The data is split into 80% training and 20% testing. "
            "The split is stratified and uses random_state=42."
        )

    with col3:

        st.markdown("#### Models")

        st.write(
            "Logistic Regression, Decision Tree, Random Forest "
            "and XGBoost."
        )

    st.markdown("### Dataset variables")

    variable_table = pd.DataFrame(
        {
            "Variable": [
                "age",
                "sex",
                "chest pain type",
                "resting bps",
                "cholesterol",
                "fasting blood sugar",
                "resting ecg",
                "max heart rate",
                "exercise angina",
                "oldpeak",
                "ST slope",
                "target",
            ],
            "Meaning": [
                "Age in years",
                "0 = female, 1 = male",
                "Coded chest pain category",
                "Resting blood pressure (mm Hg)",
                "Serum cholesterol (mg/dl)",
                "0 = at or below 120 mg/dl; 1 = above 120 mg/dl",
                "Resting electrocardiogram result",
                "Maximum heart rate achieved",
                "0 = no; 1 = yes",
                "ST depression",
                "Slope of the peak exercise ST segment",
                "0 = no heart disease; 1 = heart disease",
            ],
        }
    )

    st.dataframe(
        variable_table,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### How to interpret the metrics")

    st.write(
        "**Accuracy:** The share of all test records classified correctly."
    )

    st.write(
        "**Precision:** Of records classified as heart disease, "
        "the share that actually belong to class 1 in the test set."
    )

    st.write(
        "**Recall:** Of test records labelled as heart disease, "
        "the share the model identifies correctly."
    )

    st.write(
        "**Specificity:** Of test records labelled as no heart "
        "disease, the share the model identifies correctly."
    )

    st.write(
        "**ROC-AUC:** A measure of how well the model ranks "
        "positive cases above negative cases across thresholds."
    )

    st.caption(
        "These scores describe performance on this particular test "
        "split. They do not establish clinical validity or guarantee "
        "performance on other patient populations."
    )
