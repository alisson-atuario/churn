import sys
from pathlib import Path

# Caminho absoluto para a pasta utils
project_utils = Path(__file__).parent 
print(project_utils)
sys.path.append(str(project_utils))


import streamlit as st
import joblib
from paths import DATA_DIR, MODEL_DIR

@st.cache_resource
def load_model():
    """Carrega o modelo treinado."""
    return joblib.load(MODEL_DIR/'best_model_lgb.joblib')

@st.cache_resource
def load_calibrator():
    """Carrega calibrador Venn-Ambers"""
    try:
        return joblib.load(MODEL_DIR/'venn_abers_calibrator.joblib')
    except FileNotFoundError:
        return None

@st.cache_data
def load_dataset():
    """Carrega dataset tratado."""
    return joblib.load(DATA_DIR / 'raw' /'data_raw.joblib')

@st.cache_data
def load_test_data():
    """Carrega dados de teste."""
    X_test = joblib.load(DATA_DIR / 'processed' / 'X_test.joblib')
    y_test = joblib.load(DATA_DIR / 'processed' / 'y_test.joblib')
    return X_test, y_test

@st.cache_data
def load_cal_data():
    """Carrega dados de calibração."""
    X_cal = joblib.load(DATA_DIR / 'processed' / 'X_cal.joblib')
    y_cal = joblib.load(DATA_DIR / 'processed' / 'y_cal.joblib')
    return X_cal, y_cal


