

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Rejection Predictor", page_icon="🏦", layout="centered")

FIELD_LABELS = {
    "AGE": "Age",
    "EDUC": "Years of Education",
    "KIDS": "Number of Children",
    "INCOME": "Annual Household Income ($)",
    "NETWORTH": "Net Worth ($)",
    "ASSET": "Total Assets ($)",
    "DEBT": "Total Debt ($)",
    "DEBT2INC": "Debt-to-Income Ratio",
    "CCBAL": "Credit Card Balance ($)",
    "HOMEEQ": "Home Equity ($)",
    "MARRIED": "Marital Status",
    "LF": "In the Labor Force?",
    "HHSEX": "Sex",
    "LATE": "Late on a Payment in the Past Year?",
    "LATE60": "60+ Days Late on a Payment?",
    "BNKRUPLAST5": "Bankruptcy in the Past 5 Years?",
    "FORECLLAST5": "Foreclosure in the Past 5 Years?",
    "RACE": "Race/Ethnicity",
}

RACE_LABELS = {1: "White (non-Hispanic)", 2: "Black (non-Hispanic)", 3: "Hispanic", 5: "Other"}


YES_NO_FIELDS = {
    "LF": (1, 0),
    "LATE": (1, 0),
    "LATE60": (1, 0),
    "BNKRUPLAST5": (1, 0),
    "FORECLLAST5": (1, 0),
}


@st.cache_resource
def load_model(path: str = "model.pkl"):
    return joblib.load(path)


def numeric_input(col, numeric_stats):
    stats = numeric_stats[col]
    label = FIELD_LABELS.get(col, col)
    lo, hi, default = stats["min"], stats["max"], stats["mean"]
    if col in ("AGE", "EDUC", "KIDS"):
        return st.number_input(
            label, min_value=int(lo), max_value=int(hi) + 5,
            value=int(round(default)), step=1, key=col,
        )
    return st.number_input(
        label, min_value=float(min(lo, 0)), max_value=float(hi) * 1.2,
        value=float(round(default, 2)), step=100.0 if hi > 1000 else 0.1, key=col,
    )


def main():
    st.title("🏦 Loan / Credit Rejection Predictor")
    st.write(
        "Enter an applicant's profile below to estimate the likelihood "
        "their credit application would be turned down, based on patterns "
        "from the Survey of Consumer Finances."
    )

    try:
        data = load_model()
    except FileNotFoundError:
        st.error(
            "Could not find 'model.pkl'. Please run `python train_model.py` "
            "first (with 'SCFP2019.csv' in the same folder) to generate "
            "the trained model file."
        )
        return

    pipeline = data["pipeline"]
    feature_columns = data["feature_columns"]
    numeric_stats = data["numeric_stats"]
    categorical_options = data["categorical_options"]
    model_name = data.get("model_name")
    accuracy = data.get("accuracy")
    roc_auc = data.get("roc_auc")
    precision = data.get("precision")
    recall = data.get("recall")

    responses = {}

    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "💰 Financials", "📋 Credit History"])

    with tab1:
        for col in ["AGE", "EDUC", "KIDS"]:
            responses[col] = numeric_input(col, numeric_stats)

        married_choice = st.radio(FIELD_LABELS["MARRIED"], options=["Married / Living with partner", "Neither"], horizontal=True)
        responses["MARRIED"] = 1 if married_choice == "Married / Living with partner" else 2

        sex_choice = st.radio(FIELD_LABELS["HHSEX"], options=["Male", "Female"], horizontal=True)
        responses["HHSEX"] = 1 if sex_choice == "Male" else 2

        race_choice = st.selectbox(
            FIELD_LABELS["RACE"],
            options=categorical_options["RACE"],
            format_func=lambda x: RACE_LABELS.get(int(x), str(x)),
        )
        responses["RACE"] = race_choice

        lf_choice = st.radio(FIELD_LABELS["LF"], options=["Yes", "No"], horizontal=True)
        responses["LF"] = 1 if lf_choice == "Yes" else 0

    with tab2:
        for col in ["INCOME", "NETWORTH", "ASSET", "DEBT", "CCBAL", "HOMEEQ"]:
            responses[col] = numeric_input(col, numeric_stats)
        responses["DEBT2INC"] = st.number_input(
            FIELD_LABELS["DEBT2INC"], min_value=0.0, max_value=20.0,
            value=float(round(numeric_stats["DEBT2INC"]["mean"], 2)), step=0.1,
        )

    with tab3:
        for col in ["LATE", "LATE60", "BNKRUPLAST5", "FORECLLAST5"]:
            choice = st.radio(FIELD_LABELS[col], options=["Yes", "No"], horizontal=True, key=col)
            responses[col] = 1 if choice == "Yes" else 0

    st.divider()

    if st.button("Predict Rejection Risk", type="primary"):
        input_df = pd.DataFrame([responses])[feature_columns]

        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ **High Risk of Rejection** (Estimated probability: {proba:.1%})")
        else:
            st.success(f"✅ **Likely to be Approved** (Estimated probability of rejection: {proba:.1%})")

        st.progress(min(max(proba, 0.0), 1.0))

    st.divider()
    with st.expander("ℹ️ About this model"):
        st.write(
            f"""
            - **Model type:** {model_name or "Classifier"} (chosen automatically from
              Logistic Regression / Random Forest / Gradient Boosting via cross-validation,
              then tuned with GridSearchCV)
            - **Data:** Survey of Consumer Finances 2019, restricted to respondents who
              actually applied for credit, deduplicated to one row per household
              (the raw file has 5 near-identical "implicate" rows per household).
            - **Target:** `TURNDOWN` -- was actually turned down for credit, or given
              less credit than requested, in the past 5 years.
            {f"- **Test Accuracy:** {accuracy:.1%}" if accuracy is not None else ""}
            {f"- **Test ROC-AUC:** {roc_auc:.4f}" if roc_auc is not None else ""}
            {f"- **Precision:** {precision:.1%}" if precision is not None else ""}
            {f"- **Recall:** {recall:.1%} (share of actual rejections the model catches)" if recall is not None else ""}
            """
        )
        st.caption(
            "Note: recall is modest, meaning the model misses a good share of real "
            "rejections -- this reflects genuine class imbalance in the data (~18% "
            "of applicants are rejected) rather than a bug. Treat predictions as a "
            "rough risk signal, not a certainty."
        )


if __name__ == "__main__":
    main()
