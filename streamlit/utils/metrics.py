from pathlib import Path
import joblib
import sys

# Caminho absoluto para a pasta utils
project_utils = Path(__file__).parent 
sys.path.append(str(project_utils))

from matplotlib.pyplot import available_backends
from matplotlib.style import available
import numpy as np
import pandas as pd
from paths import MODEL_DIR,DATA_DIR,SRC_DIR
from data_loader import load_dataset, load_cal_data, load_test_data, load_model
from sklearn.metrics import brier_score_loss, fbeta_score, f1_score

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

#%% Home
def calculate_churn_rate(data):
    """Calcula a taxa de churn atual do dataset histórico"""
    if 'Churn' in data.columns:
        churn_column = data['Churn']
        # Verifica se é binário (0/1) ou categórico ('Yes'/'No')
        if churn_column.dtype == 'object':
            churn_rate = (churn_column == 'Yes').mean() * 100
        else:
            churn_rate = churn_column.mean() * 100
        return round(churn_rate, 1)
    return 0.0

def calculate_total_customers(data):
    """Calcula o número total de clientes"""
    return len(data)

def calculate_churned_customers(data):
    """Calcula o número de clientes que deram churn"""
    if 'Churn' in data.columns:
        churn_column = data['Churn']
        if churn_column.dtype == 'object':
            return (churn_column == 'Yes').sum()
        else:
            return churn_column.sum()
    return 0

def calculate_value_lost_historical(data):
    """Calcula o valor mensal perdido com churn (histórico real)"""
    if 'MonthlyCharges' in data.columns and 'Churn' in data.columns:
        # Filtra apenas clientes que deram churn
        churned_customers = data[data['Churn'] == 'Yes'] if data['Churn'].dtype == 'object' else data[data['Churn'] == 1]
        value_lost = churned_customers['MonthlyCharges'].sum()
        return round(value_lost, 2)
    return 0.0

def calculate_average_tenure(data):
    """Calcula o tempo médio de permanência dos clientes"""
    if 'tenure' in data.columns:
        return round(data['tenure'].mean(), 1)
    return 0.0

def calculate_avg_monthly_charge(data):
    """Calcula a média de gastos mensais"""
    if 'MonthlyCharges' in data.columns:
        return round(data['MonthlyCharges'].mean(), 2)
    return 0.0

def get_main_kpis():
    """Retorna as principais KPIs baseadas em dados históricos reais"""
    # Carrega dados históricos
    data = load_dataset()
    
    # Calcula métricas baseadas no histórico real
    churn_rate = calculate_churn_rate(data)
    total_customers = calculate_total_customers(data)
    churned_customers = calculate_churned_customers(data)
    value_lost = calculate_value_lost_historical(data)
    avg_tenure = calculate_average_tenure(data)
    avg_monthly_charge = calculate_avg_monthly_charge(data)
    
    return {
        'churn_rate': churn_rate,
        'total_customers': total_customers,
        'churned_customers': churned_customers,
        'value_lost': value_lost,
        'avg_tenure': avg_tenure,
        'avg_monthly_charge': avg_monthly_charge
    }

def get_trend_data():
    """Gera dados de tendência de churn por tempo de permanência"""
    data = load_dataset()
    
    # Cria bins de tempo de permanência (em meses)
    bins = [0, 12, 24, 36, 48, 60, 72, float('inf')]
    labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72', '73+']
    
    data['tenure_group'] = pd.cut(data['tenure'], bins=bins, labels=labels, right=False)
    
    # Calcula taxa de churn por grupo de tempo
    trend_data = data.groupby('tenure_group').agg(
        total_customers=('Churn', 'count'),
        churned_customers=('Churn', lambda x: (x == 'Yes').sum() if x.dtype == 'object' else x.sum()),
        avg_monthly_charge=('MonthlyCharges', 'mean')
    ).reset_index()
    
    trend_data['churn_rate'] = (trend_data['churned_customers'] / trend_data['total_customers'] * 100).round(1)
    trend_data['avg_monthly_charge'] = trend_data['avg_monthly_charge'].round(2)
    
    return trend_data

def calculate_value_lost_distribution(data):
    """
    Calcula a distribuição do valor perdido por faixa de gasto mensal
    Retorna o percentual do valor perdido que vem de clientes com alto gasto mensal
    """
    if 'MonthlyCharges' not in data.columns or 'Churn' not in data.columns:
        return 0.0
    
    # Filtra apenas clientes que deram churn
    if data['Churn'].dtype == 'object':
        churned_data = data[data['Churn'] == 'Yes']
    else:
        churned_data = data[data['Churn'] == 1]
    
    if len(churned_data) == 0:
        return 0.0
    
    # Define faixas de gasto mensal (alto gasto = acima da mediana)
    median_charge = churned_data['MonthlyCharges'].median()
    
    # Calcula valor total perdido
    total_value_lost = churned_data['MonthlyCharges'].sum()
    
    # Calcula valor perdido de clientes com alto gasto mensal
    high_spenders = churned_data[churned_data['MonthlyCharges'] > median_charge]
    high_spender_value_lost = high_spenders['MonthlyCharges'].sum()
    
    # Calcula percentual
    if total_value_lost > 0:
        percent_high_spenders = (high_spender_value_lost / total_value_lost) * 100
        return round(percent_high_spenders, 1)

#%% Exploratory
def data_raw_treatment():
    """Trata os dados brutos para análise"""
    data = load_dataset()
    
    # Substitui espaços vazios por Nan
    for coluna in data.columns:
        data[coluna] = data[coluna].apply(lambda x: x if x != ' ' else np.nan)
    
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['Partner'] = pd.to_numeric(data['Partner'],errors='coerce') 
    return data

def calculate_churn_rate(data):
    """ Calcula a taxa de Churn"""
    churn_freq = data['Churn'].value_counts(normalize=True)
    churn_rate = churn_freq*100
    print(churn_rate)
    return round(churn_rate, 2)

def data_categoric():
    """ Retorna um dataframe categórico """
    data = load_dataset()
    # Aplica a função lambda para todas as colunas do DataFrame
    for coluna in data.columns:
        data[coluna] = data[coluna].apply(lambda x: x if x != ' ' else np.nan)

    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['Partner'] = pd.to_numeric(data['Partner'],errors='coerce')

    for col in ['tenure','MonthlyCharges','TotalCharges']:
        categories, quantis = pd.qcut(data[col], q=5, retbins=True, precision=0)
        data[col+'_cat'] = categories
    
    categoric_values = data.select_dtypes(exclude=[np.number])
    categoric_values.reset_index(names='before_index',drop=True,inplace=True)
    categoric_values.drop(columns=['customerID'], inplace=True)

    return categoric_values

#%% Predictions

def get_model_predictions():
    """Retorna predições do modelo nos dados de teste"""
    X_cal, y_cal = load_cal_data()
    X_test, y_test = load_test_data()
    model = load_model()
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    y_pred_cal = model.predict(X_cal)
    y_pred_proba_cal = model.predict_proba(X_cal)[:, 1]
    
    return X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal

def calculate_model_metrics(y_test, y_pred, y_pred_proba):
    """Calcula todas as métricas de performance do modelo"""
    # Métricas básicas
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Curvas
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'auc_score': auc_score,
        'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr, thresholds_roc),
        'pr_curve': (precision, recall, thresholds_pr)
    }

def get_confusion_matrix_values(conf_matrix):
    """Retorna valores interpretados da matriz de confusão"""
    tn, fp, fn, tp = conf_matrix.ravel()
    return {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    }

def create_risk_segments(y_pred_proba, y_test):
    """Cria segmentação de clientes por nível de risco"""
    df_risk = pd.DataFrame({
        'probability': y_pred_proba,
        'actual_churn': y_test.values if hasattr(y_test, 'values') else y_test
    })
    
    df_risk['risk_segment'] = pd.cut(
        df_risk['probability'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Baixo Risco', 'Médio Risco', 'Alto Risco']
    )
    
    risk_summary = df_risk.groupby('risk_segment').agg({
        'probability': ['count', 'mean'],
        'actual_churn': 'sum'
    }).round(3)
    
    risk_summary.columns = ['Total Clientes', 'Prob. Média', 'Churns Reais']
    risk_summary['Taxa Churn Real (%)'] = (
        risk_summary['Churns Reais'] / risk_summary['Total Clientes'] * 100
    ).round(1)
    
    return risk_summary

def get_client_prediction_details(X_test, y_test, y_pred, y_pred_proba, sample_idx):
    """Retorna detalhes de predição para um cliente específico"""
    return {
        'probability': float(y_pred_proba[sample_idx]),
        'prediction': int(y_pred[sample_idx]),
        'actual': int(y_test.iloc[sample_idx]),
        'features': X_test.iloc[sample_idx].to_dict()
    }

def predict_single_customer(customer_data):
    """Faz predição para um único cliente"""
    model = load_model()
    prob = model.predict_proba(customer_data)[0, 1]
    prediction = 1 if prob >= 0.5 else 0
    
    return {
        'probability': prob,
        'prediction': prediction
    }

# SHAP Analysis
def load_shap_values():
    """Carrega valores SHAP salvos"""
    path = DATA_DIR/ 'processed' / 'shap_values.joblib'
    shap_data = joblib.load(path)
    return (
        shap_data['shap_values'],
        shap_data['expected_value'],
        shap_data['feature_names']
    )

def get_feature_importance(shap_values, feature_names):
    """Calcula importância das features baseada em valores SHAP"""
    importance = np.abs(shap_values).mean(0)
    importance_dict = dict(zip(feature_names, importance))
    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

def explain_prediction(shap_values, expected_value, feature_names, sample_idx=0, top_n=None):
    """Gera explicação textual para uma previsão específica"""
    values = shap_values[sample_idx]
    sorted_idx = np.argsort(np.abs(values))[::-1]
    
    if top_n is not None:
        sorted_idx = sorted_idx[:top_n]
    
    features = []
    for idx in sorted_idx:
        impact = "aumenta" if values[idx] > 0 else "diminui"
        features.append({
            'feature': feature_names[idx],
            'impact': impact,
            'value': abs(values[idx])
        })
    
    # Adiciona informação do valor base e predição final
    final_prediction = expected_value + values.sum()
    
    return {
        'features': features,
        'base_value': expected_value,        # ← Agora usa!
        'final_prediction': final_prediction  # ← E calcula predição final
    }

#%% Calibration
def threshold_evaluation(y_test, y_proba_raw, y_proba_cal=None):

    # Calcula métricas para diferentes thresholds
    thresholds = np.linspace(0.1, 0.9, 50)
    precisions = []
    recalls = []
    f1_scores = []
    f2_scores = []
    
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    proba_to_use = y_proba_cal if y_proba_cal is not None else y_proba_raw
    
    for t in thresholds:
        y_pred_t = (proba_to_use > t).astype(int)
        
        # Calcula métricas
        tp = ((y_pred_t == 1) & (y_test_array == 1)).sum()
        fp = ((y_pred_t == 1) & (y_test_array == 0)).sum()
        fn = ((y_pred_t == 0) & (y_test_array == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score(y_test_array, y_pred_t, zero_division=0))
        f2_scores.append(fbeta_score(y_test_array, y_pred_t, beta=2, zero_division=0))

    # Encontra threshold ótimo (F2)
    best_threshold = thresholds[np.argmax(f2_scores)]

    return thresholds, precisions, recalls, f1_scores, f2_scores, best_threshold, y_test_array, proba_to_use
