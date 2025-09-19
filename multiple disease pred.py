# -*- coding: utf-8 -*-
"""
Multiple Disease Prediction Streamlit App

Clean, consolidated rewrite:
- Uses pathlib and a safe loader to open model files from the local `saved models` folder
- Records model load errors and displays descriptive status in the sidebar
- Uses typed Streamlit inputs (`number_input`, `selectbox`) with sensible defaults
- Wraps predictions with a safe helper that reports exceptions to the UI

Place the three .sav files in the `saved models` directory next to this script:
 - diabetes_model.sav
 - heart_disease_model.sav
 - parkinsons_model.sav

Run with:
    python -m streamlit run "multiple disease pred.py"
"""

from pathlib import Path
from typing import Optional, Dict
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import os
import time


# --- Configuration ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved models"
MODEL_ERRORS: Dict[str, str] = {}


def _get_file_info(path: Path) -> str:
    try:
        stat = path.stat()
        size_kb = stat.st_size / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        return f"{size_kb:.1f} KB, modified {mtime}"
    except Exception:
        return "(no info)"


def load_model(filename: str) -> Optional[object]:
    """Load a pickled model from the saved models folder.

    On error, record a descriptive message in MODEL_ERRORS and return None.
    """
    path = MODEL_DIR / filename
    if not path.exists():
        MODEL_ERRORS[filename] = f"File not found: {path}"
        return None
    try:
        with path.open("rb") as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        MODEL_ERRORS[filename] = str(e)
        return None


def safe_predict(model, features):
    try:
        return model.predict([features])
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        return None


# --- Load models at startup -----------------------------------------------
DIABETES_FILE = "diabetes_model.sav"
HEART_FILE = "heart_disease_model.sav"
PARK_FILE = "parkinsons_model.sav"

_diabetes_model = load_model(DIABETES_FILE)
_heart_model = load_model(HEART_FILE)
_park_model = load_model(PARK_FILE)


# --- Sidebar: Navigation -----------------------------------
with st.sidebar:
    # Keep a compact navigation menu; model-status info intentionally removed
    selected = option_menu(
        "Multiple Disease Prediction System",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction"],
        icons=["activity", "heart", "person"],
        menu_icon="cast",
        default_index=0,
    )


# --- Pages ---------------------------------------------------------------
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    c1, c2, c3 = st.columns(3)
    with c1:
        pregnancies = st.number_input("Number of Pregnancies", value=2, min_value=0, step=1)
        skin = st.number_input("Skin Thickness", value=20, min_value=0, step=1)
        dpf = st.number_input("Diabetes Pedigree Function", value=0.5, min_value=0.0, step=0.01, format="%.4f")
    with c2:
        glucose = st.number_input("Glucose Level", value=120, min_value=0, step=1)
        insulin = st.number_input("Insulin Level", value=79.0, min_value=0.0, step=1.0)
        age = st.number_input("Age", value=45, min_value=0, step=1)
    with c3:
        bp = st.number_input("Blood Pressure", value=70, min_value=0, step=1)
        bmi = st.number_input("BMI", value=28.0, min_value=0.0, step=0.1)

    if st.button("Diabetes Test Result"):
        if _diabetes_model is None:
            st.error("Diabetes model is not loaded. Check sidebar for details.")
        else:
            features = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
            res = safe_predict(_diabetes_model, features)
            if res is not None:
                st.success("The person is diabetic" if int(res[0]) == 1 else "The person is not diabetic")


if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", value=54, min_value=0, step=1)
        trestbps = st.number_input("Resting Blood Pressure", value=130, min_value=0, step=1)
        restecg = st.number_input("Resting ECG result", value=1, min_value=0, step=1)
    with c2:
        sex = st.number_input("Sex (0 = female, 1 = male)", value=1, min_value=0, max_value=1, step=1)
        chol = st.number_input("Serum Cholesterol (mg/dl)", value=250, min_value=0, step=1)
        thalach = st.number_input("Maximum Heart Rate achieved", value=150, min_value=0, step=1)
    with c3:
        cp = st.number_input("Chest Pain type (0-3)", value=3, min_value=0, step=1)
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (0/1)", value=0, min_value=0, max_value=1, step=1)
        exang = st.number_input("Exercise Induced Angina (0/1)", value=0, min_value=0, max_value=1, step=1)

    oldpeak = st.number_input("ST depression induced by exercise", value=1.0, min_value=0.0, step=0.1)
    slope = st.number_input("Slope of the peak exercise ST segment", value=2, min_value=0, step=1)
    ca = st.number_input("Major vessels colored by flourosopy (0-3)", value=0, min_value=0, step=1)

    thal_map = {"0 - normal": 0, "1 - fixed defect": 1, "2 - reversible defect": 2}
    thal_label = st.selectbox("Thalassemia (thal)", list(thal_map.keys()), index=2)
    thal = thal_map[thal_label]

    if st.button("Heart Disease Test Result"):
        if _heart_model is None:
            st.error("Heart disease model is not loaded. Check sidebar for details.")
        else:
            features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            res = safe_predict(_heart_model, features)
            if res is not None:
                st.success("The person is having heart disease" if int(res[0]) == 1 else "The person does not have any heart disease")


if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', value=119.992, min_value=0.0, step=0.001, format="%.3f")
        RAP = st.number_input('MDVP:RAP', value=0.00370, min_value=0.0, step=1e-6, format="%.6f")
        APQ3 = st.number_input('Shimmer:APQ3', value=0.11, min_value=0.0, step=0.01, format="%.2f")
        HNR = st.number_input('HNR', value=21.0, min_value=0.0, step=0.1)
        D2 = st.number_input('D2', value=2.1, min_value=0.0, step=0.01)
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', value=157.302, min_value=0.0, step=0.001, format="%.3f")
        PPQ = st.number_input('MDVP:PPQ', value=0.00401, min_value=0.0, step=1e-6, format="%.6f")
        APQ5 = st.number_input('Shimmer:APQ5', value=0.16, min_value=0.0, step=0.01, format="%.2f")
        RPDE = st.number_input('RPDE', value=0.65, min_value=0.0, step=0.01)
        PPE = st.number_input('PPE', value=0.2, min_value=0.0, step=0.01)
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', value=74.997, min_value=0.0, step=0.001, format="%.3f")
        DDP = st.number_input('Jitter:DDP', value=0.00631, min_value=0.0, step=1e-6, format="%.6f")
        APQ = st.number_input('MDVP:APQ', value=0.14, min_value=0.0, step=0.01, format="%.2f")
        DFA = st.number_input('DFA', value=0.71, min_value=0.0, step=0.01)
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.00784, min_value=0.0, step=1e-6, format="%.6f")
        Shimmer = st.number_input('MDVP:Shimmer', value=0.24, min_value=0.0, step=0.01, format="%.2f")
        DDA = st.number_input('Shimmer:DDA', value=0.17, min_value=0.0, step=0.01, format="%.2f")
        spread1 = st.number_input('spread1', value=-4.0, step=0.1)
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.00007, min_value=0.0, step=1e-7, format="%.7f")
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=2.0, min_value=0.0, step=0.1)
        NHR = st.number_input('NHR', value=0.022, min_value=0.0, step=1e-4, format="%.4f")
        spread2 = st.number_input('spread2', value=2.0, step=0.1)

    if st.button("Parkinson's Test Result"):
        if _park_model is None:
            st.error("Parkinsons model is not loaded. Check sidebar for details.")
        else:
            features = [
                fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                RPDE, DFA, spread1, spread2 if 'spread2' in locals() else 2.0, D2, PPE
            ]
            res = safe_predict(_park_model, features)
            if res is not None:
                st.success("The person has Parkinson's disease" if int(res[0]) == 1 else "The person does not have Parkinson's disease")



