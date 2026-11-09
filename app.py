# app.py
import os
import streamlit as st
import pandas as pd
import joblib
 
st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")
 
DATA_FILE = "student_data.csv"
MODEL_FILE = "student_model.pkl"
 
 
@st.cache_resource(show_spinner="Preparing data and model (first run only)...")
def prepare():
    """Load data/model if they exist; otherwise download the data and train the model."""
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
 
    # 1) Data
    if not os.path.exists(DATA_FILE):
        from ucimlrepo import fetch_ucirepo
        student = fetch_ucirepo(id=320)
        raw = pd.concat([student.data.features, student.data.targets], axis=1)
        raw.to_csv(DATA_FILE, index=False)
    df = pd.read_csv(DATA_FILE)
 
    # 2) Same split as train_model.py (used for the evaluation summary)
    clean = df.drop_duplicates()
    y = (clean["G3"] < 10).astype(int)
    X = clean.drop(columns=["G3"])  # G3 is never a predictor
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
 
    # 3) Model
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(exclude="number").columns.tolist()
        preprocessor = ColumnTransformer([
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols),
        ])
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight="balanced"
            ))
        ])
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
 
    preds = model.predict(X_test)
    acc = round(accuracy_score(y_test, preds), 3)
    report = classification_report(
        y_test, preds, target_names=["Not at risk", "At risk"], zero_division=0
    )
    return model, df, acc, report
 
 
model, data, accuracy, report = prepare()
 
st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")
 
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))
 
with st.expander("Model evaluation (20% hold-out test set)"):
    st.write(f"Accuracy: **{accuracy}**")
    st.code(report)
    st.caption("Do not interpret accuracy alone; check precision, recall and F1 for the 'At risk' class.")
 
st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)
 
st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")
 
template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)
 
uploaded = st.file_uploader("Choose student CSV", type=["csv"])
 
if uploaded is not None:
    new_data = pd.read_csv(uploaded)
    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]
 
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        predictions = model.predict(new_data[required_columns])
        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK"
            for value in predictions
        ]
 
        st.subheader("Prediction results")
        st.dataframe(results, use_container_width=True)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))
 
        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv"
        )
 
st.caption("Predictions are not diagnoses and must not be the sole basis for real student decisions.")
