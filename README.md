# Student Performance Prediction System

A classroom prototype that estimates whether a student is at risk of failing the final Mathematics grade, built for the Data Mining laboratory (Python, Streamlit, GitHub).

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of students who may need additional academic support, using historical student information.

## Dataset
UCI Machine Learning Repository, Student Performance dataset (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
Downloaded automatically through the `ucimlrepo` package.

## Prediction target
- `G3 < 10` -> **AT RISK** (1)
- `G3 >= 10` -> **NOT AT RISK** (0)

`G3` is the target only and is never used as an input feature (to avoid target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, Git/GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py        # downloads data, trains model, creates student_data.csv and student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions. You can download the results as CSV.

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- Small dataset; accuracy alone is not a reliable measure, so review precision, recall and F1 for the "At risk" class.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
