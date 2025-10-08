# Page: 02_predictions.py
from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))


import streamlit as st
from utils.data_loader import load_test_data

from utils.metrics import (
    get_model_predictions,
    calculate_model_metrics,
    get_confusion_matrix_values,
    create_risk_segments
)
from utils.plots import (
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
    plot_probability_distribution_by_class,
    plot_risk_segments,
    plot_metrics_comparison_bar
)

st.set_page_config(
    page_title="Predições de Churn",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Performance do Modelo de Predição")
st.markdown("""
Avaliação completa do modelo LightGBM: métricas de classificação, análise de erros 
e segmentação de clientes por nível de risco.
""")

# Carrega dados

def load_prediction_data():
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
    metrics = calculate_model_metrics(y_cal, y_pred_cal, y_pred_proba_cal)
    return X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal, metrics

try:
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal, metrics = load_prediction_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()


#%% 1. KPIs Principais
st.header("1. Métricas de Performance")

with st.expander("Indicadores Chave", expanded=True):
    class_report = metrics['classification_report']
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-ROC", f"{metrics['auc_score']:.3f}")
    col2.metric("Precisão (Churn)", f"{class_report['1']['precision']:.3f}")
    col3.metric("Recall (Churn)", f"{class_report['1']['recall']:.3f}")
    col4.metric("F1-Score (Churn)", f"{class_report['1']['f1-score']:.3f}")
    
    st.info("""
    **Interpretação:**
    - **AUC-ROC**: Capacidade de separar churn de não-churn (>0.75 = bom)
    - **Precisão**: Dos alertados como churn, quantos realmente cancelaram
    - **Recall**: Dos que cancelaram, quantos conseguimos detectar
    - **F1-Score**: Balanço entre precisão e recall
    """)

#%% 2. Comparação Visual de Métricas
st.header("2. Comparação de Métricas por Classe")

with st.expander("Visualização", expanded=True):
    st.plotly_chart(plot_metrics_comparison_bar(class_report))

#%% 3. Matriz de Confusão
st.header("3. Análise de Erros - Matriz de Confusão")

with st.expander("Visualização", expanded=True):
    col_a, col_b = st.columns([2, 3])
    
    with col_a:
        st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix']))
    
    with col_b:
        st.markdown("### Interpretação dos Resultados")
        cm_values = get_confusion_matrix_values(metrics['confusion_matrix'])
        
        st.markdown(f"""
        **Verdadeiros Negativos (TN): {cm_values['true_negative']}**  
        Clientes que não cancelaram e foram corretamente identificados ✅
        
        **Falsos Positivos (FP): {cm_values['false_positive']}**  
        Clientes que não cancelaram, mas modelo alertou como risco  
        Custo: esforço de retenção desperdiçado
        
        **Falsos Negativos (FN): {cm_values['false_negative']}**  
        Clientes que cancelaram mas não foram detectados  
        Custo crítico: perda de receita
        
        **Verdadeiros Positivos (TP): {cm_values['true_positive']}**  
        Clientes em risco corretamente identificados ✅  
        Oportunidade de ação preventiva
        """)
        
        # Calcula custos
        total_churn = cm_values['true_positive'] + cm_values['false_negative']
        taxa_deteccao = (cm_values['true_positive'] / total_churn * 100) if total_churn > 0 else 0
        
        st.success(f"Taxa de Detecção de Churn: **{taxa_deteccao:.1f}%**")

#%% 4. Curvas de Performance
st.header("4. Curvas ROC e Precision-Recall")

with st.expander("Visualização", expanded=True):
    st.plotly_chart(plot_roc_and_pr_curves(metrics))
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Curva ROC**  
        Trade-off entre detectar churns (TPR) vs alertas falsos (FPR).  
        AUC próximo de 1.0 = excelente separação.
        """)
    with col2:
        st.markdown("""
        **Curva Precision-Recall**  
        Mais relevante em dados desbalanceados.  
        Mostra qualidade dos alertas de churn.
        """)

#%% 5. Distribuição de Probabilidades
st.header("5. Distribuição das Probabilidades Preditas")

with st.expander("Visualização", expanded=True):
    st.plotly_chart(plot_probability_distribution_by_class(y_test, y_pred_proba))
    
    st.info("""
    **Análise ideal:**
    - Pico azul (não-churn) à esquerda: modelo confiante que não vão cancelar
    - Pico vermelho (churn) à direita: modelo confiante que vão cancelar
    - Boa separação = modelo consegue distinguir bem os grupos
    """)

#%% 6. Segmentação de Risco
st.header("6. Segmentação de Clientes por Nível de Risco")

with st.expander("Análise", expanded=True):
    risk_summary = create_risk_segments(y_pred_proba, y_test)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.plotly_chart(plot_risk_segments(risk_summary))
    
    with col2:
        st.markdown("### Resumo por Segmento")
        st.dataframe(risk_summary)
    
    # Insights dinâmicos
    alto_risco = risk_summary.loc['Alto Risco']
    medio_risco = risk_summary.loc['Médio Risco']
    baixo_risco = risk_summary.loc['Baixo Risco']
    
    st.success(f"""
    **Recomendações de Ação:**
    
    🔴 **Alto Risco**: {int(alto_risco['Total Clientes'])} clientes 
    ({alto_risco['Taxa Churn Real (%)']:.1f}% taxa real de churn)  
    → Ação imediata: contato proativo, ofertas personalizadas
    
    🟡 **Médio Risco**: {int(medio_risco['Total Clientes'])} clientes 
    ({medio_risco['Taxa Churn Real (%)']:.1f}% taxa real de churn)  
    → Monitoramento: campanhas preventivas, pesquisas de satisfação
    
    🟢 **Baixo Risco**: {int(baixo_risco['Total Clientes'])} clientes 
    ({baixo_risco['Taxa Churn Real (%)']:.1f}% taxa real de churn)  
    → Manutenção: comunicação regular, programas de fidelidade
    """)

#%% 7. Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("← Voltar para Exploratory"):
        st.switch_page("pages/01_exploratory.py")

with col_nav2:
    if st.button("Ir para Simulador →"):
        st.switch_page("pages/03_simulator.py")

with col_nav3:
    if st.button("🏠 Home"):
        st.switch_page("00_home.py")

X_test, y_test = load_test_data()
print(len(X_test))

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i>
</div>
""", unsafe_allow_html=True)