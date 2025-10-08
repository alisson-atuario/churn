from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_model, load_calibrator
from utils.metrics import get_model_predictions, calculate_model_metrics, threshold_evaluation
from utils.plots import plot_calibration_curve, plot_threshold_evaluation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import brier_score_loss, fbeta_score, f1_score
from sklearn.calibration import calibration_curve
from utilities import calculate_ece

st.set_page_config(
    page_title="Calibração do Modelo",
    page_icon="🎯",
    layout="wide"
)

st.title("Calibração e Otimização de Probabilidades")
st.markdown("""
Análise da **confiabilidade das probabilidades** e otimização do threshold de decisão.
Entenda como a calibração Venn-ABERS melhora a precisão das predições.
""")

# Carrega dados
def load_calibration_data():
    model = load_model()
    calibrator = load_calibrator()
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
    
    # Probabilidades sem calibração
    y_test_proba_raw = model.predict_proba(X_test)[:, 1]
    
    # Probabilidades calibradas (se disponível)
    y_test_proba_cal = None
    if calibrator:
        y_test_proba_cal = calibrator.predict_proba(model.predict_proba(X_test))[:, 1]
    
    return y_test, y_pred, y_test_proba_raw, y_test_proba_cal, calibrator

try:
    y_test, y_pred, y_proba_raw, y_proba_cal, calibrator = load_calibration_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()

#%% 1. O Que é Calibração?
st.header("1. O Que é Calibração de Probabilidades?")

with st.expander("Conceito Fundamental", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### O Problema
        
        Modelos de ML frequentemente produzem **probabilidades não calibradas**:
        
        - Modelo diz: "Este cliente tem 80% de chance de churn"
        - Realidade: De 100 clientes com "80% de chance", apenas 60 realmente dão churn
        
        **Isso é um problema porque:**
        - ❌ Decisões de negócio baseadas em probabilidades erradas
        - ❌ Dificuldade em definir thresholds adequados
        - ❌ Perda de confiança nas predições
        """)
    
    with col2:
        st.markdown("""
        ### A Solução
        
        **Calibração** ajusta as probabilidades para refletir frequências reais:
        
        - Modelo calibrado: "80% de chance"
        - Realidade: ~80 de 100 clientes realmente dão churn ✓
        
        **Vantagens:**
        - ✅ Probabilidades interpretáveis
        - ✅ Melhor tomada de decisão
        - ✅ Comparação justa entre modelos
        """)
    
    st.info("""
    **Exemplo Prático:**
    
    Se o modelo diz que 100 clientes têm 70% de chance de churn:
    - **Sem calibração**: Talvez 55 ou 85 realmente deem churn (impreciso)
    - **Com calibração**: Aproximadamente 70 dão churn (confiável)
    """)

#%% 2. Comparação: Com vs Sem Calibração
st.header("2. Impacto da Calibração")

with st.expander("Métricas de Calibração", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    # Calcula ECE (Expected Calibration Error)
    ece_raw = calculate_ece(y_test.values if hasattr(y_test, 'values') else y_test, y_proba_raw)
    brier_raw = brier_score_loss(y_test, y_proba_raw)
    
    with col1:
        st.metric(
            "ECE Sem Calibração",
            f"{ece_raw:.4f}",
            help="Expected Calibration Error - quanto menor, melhor"
        )
    
    with col2:
        st.metric(
            "Brier Score Sem Calibração",
            f"{brier_raw:.4f}",
            help="Penaliza desvios da probabilidade real - quanto menor, melhor"
        )
    
    with col3:
        if y_proba_cal is not None:
            ece_cal = calculate_ece(y_test.values if hasattr(y_test, 'values') else y_test, y_proba_cal)
            improvement = (1 - ece_cal/ece_raw) * 100
            st.metric(
                "Melhoria com Calibração",
                f"+{improvement:.1f}%",
                delta=f"ECE: {ece_cal:.4f}",
                delta_color="normal"
            )
        else:
            st.warning("Calibrador não disponível")
    
    if y_proba_cal is not None:
        st.success(f"""
        **Resultado:** A calibração reduziu o erro de calibração em **{improvement:.1f}%**, 
        tornando as probabilidades mais confiáveis para tomada de decisão.
        """)

#%% 3. Curva de Calibração
st.header("3. Curva de Calibração")

with st.expander("Visualização", expanded=True):
    # Calcula curvas
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_proba_raw, n_bins=10)
    
    # Plota curva
    fig = plot_calibration_curve(prob_pred_raw, prob_true_raw, y_proba_cal, y_test)
    
    st.plotly_chart(fig)
    
    st.info("""
    **Como interpretar:**
    
    - **Linha cinza tracejada**: Calibração perfeita (ideal)
    - **Linha vermelha**: Modelo sem calibração
    - **Linha azul**: Modelo calibrado
    
    **Quanto mais próximo da linha perfeita, melhor!**
    
    Exemplo: Se o modelo prevê 0.6 (60%), idealmente ~60% desses casos devem ser churns reais.
    """)

#%% 4. Otimização de Threshold
st.header("4. Otimização de Threshold")

with st.expander("Trade-off Precision vs Recall", expanded=True):
    st.markdown("""
    ### Por que otimizar o threshold?
    
    O threshold padrão (0.5) nem sempre é ideal para negócios. Dependendo do custo:
    - **Perder um cliente** (Falso Negativo)
    - **Oferta desnecessária** (Falso Positivo)
    
    Podemos ajustar o threshold para maximizar resultados.
    """)
    
    # Calcula métricas para diferentes thresholds
    thresholds, precisions, recalls, f1_scores, f2_scores, best_threshold, y_test_array, proba_to_use = threshold_evaluation(y_test, y_proba_raw, y_proba_cal)
    
    # Plota gráfico
    fig = plot_threshold_evaluation(thresholds, precisions, recalls, f1_scores, f2_scores, best_threshold)
    
    st.plotly_chart(fig)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Threshold Padrão", "0.50")
        st.caption("Balanço neutro")
    
    with col2:
        st.metric("Threshold Ótimo (F2)", f"{best_threshold:.2f}")
        st.caption("Prioriza detectar mais churns (Recall)")
    
    with col3:
        improvement_idx = np.where(thresholds == best_threshold)[0][0] if best_threshold in thresholds else np.argmin(np.abs(thresholds - best_threshold))
        f2_improvement = (f2_scores[improvement_idx] / f2_scores[np.where(thresholds >= 0.5)[0][0]] - 1) * 100
        st.metric("Ganho em F2-Score", f"+{f2_improvement:.1f}%")
        st.caption("vs threshold 0.5")
    
    st.warning(f"""
    **💡 Recomendação:** Use threshold **{best_threshold:.2f}** em produção.
    
    **Por quê F2 e não F1?**
    - F2 dá **peso 2x maior ao Recall**
    - No churn, **perder um cliente custa mais** do que uma ação de retenção desnecessária
    - Melhor detectar 10 churns com 3 falsos positivos do que detectar 7 churns com 1 falso positivo
    """)

#%% 5. Simulador de Threshold
st.header("5. Simulador Interativo")

with st.expander("Teste Diferentes Thresholds", expanded=True):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configuração")
        
        threshold_sim = st.slider(
            "Threshold de Decisão",
            min_value=0.1,
            max_value=0.9,
            value=float(best_threshold),
            step=0.01,
            help="Ajuste para ver o impacto nas métricas"
        )
        
        # Calcula métricas para threshold selecionado
        y_pred_sim = (proba_to_use > threshold_sim).astype(int)
        
        tp_sim = ((y_pred_sim == 1) & (y_test_array == 1)).sum()
        fp_sim = ((y_pred_sim == 1) & (y_test_array == 0)).sum()
        fn_sim = ((y_pred_sim == 0) & (y_test_array == 1)).sum()
        tn_sim = ((y_pred_sim == 0) & (y_test_array == 0)).sum()
        
        precision_sim = tp_sim / (tp_sim + fp_sim) if (tp_sim + fp_sim) > 0 else 0
        recall_sim = tp_sim / (tp_sim + fn_sim) if (tp_sim + fn_sim) > 0 else 0
        f1_sim = 2 * (precision_sim * recall_sim) / (precision_sim + recall_sim) if (precision_sim + recall_sim) > 0 else 0
        
        st.markdown("### Métricas")
        st.metric("Precision", f"{precision_sim:.3f}")
        st.metric("Recall", f"{recall_sim:.3f}")
        st.metric("F1-Score", f"{f1_sim:.3f}")
    
    with col2:
        st.markdown("### Matriz de Confusão")
        
        # Cria heatmap
        cm_data = [[tn_sim, fp_sim], [fn_sim, tp_sim]]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=['Não Churn', 'Churn'],
            y=['Não Churn', 'Churn'],
            text=cm_data,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale='Blues'
        ))
        
        fig_cm.update_layout(
            title=f'Threshold = {threshold_sim:.2f}',
            xaxis_title='Predito',
            yaxis_title='Real',
            height=400
        )
        
        st.plotly_chart(fig_cm)
        
        # Interpretação
        total_churns = tp_sim + fn_sim
        detected_pct = (tp_sim / total_churns * 100) if total_churns > 0 else 0
        
        total_alerts = tp_sim + fp_sim
        precision_pct = (tp_sim / total_alerts * 100) if total_alerts > 0 else 0
        
        st.info(f"""
        **Interpretação:**
        - ✅ Detectamos **{detected_pct:.1f}%** dos churns reais ({tp_sim} de {total_churns})
        - ⚠️ **{precision_pct:.1f}%** dos alertas são corretos ({tp_sim} de {total_alerts})
        - ❌ **{fn_sim}** churns não detectados (custo alto!)
        - 💰 **{fp_sim}** ações desnecessárias (custo baixo)
        """)

#%% 6. Venn-ABERS: Metodologia
st.header("6. Metodologia: Venn-ABERS")

with st.expander("Como Funciona a Calibração", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Algoritmo
        
        1. **Treina modelo base** (LightGBM)
        2. **Divide dados**:
           - Train: Para treinar modelo
           - Calibration: Para calibrar probabilidades
           - Test: Para validar
        3. **Aplica Venn-ABERS**:
           - Usa "conformal prediction"
           - Não assume distribuição específica
           - Funciona com poucos dados
        4. **Otimiza threshold** via F2-score
        """)
    
    with col2:
        st.markdown("""
        ### Vantagens
        
        - ✅ **Não-paramétrico**: Sem suposições sobre distribuição
        - ✅ **Intervalos de confiança**: Além da probabilidade pontual
        - ✅ **Eficiente**: Funciona com datasets pequenos
        - ✅ **Teórico**: Garantias matemáticas de validade
        
        ### Alternativas
        
        - Platt Scaling (assume distribuição sigmoide)
        - Isotonic Regression (assume monotonia)
        - Temperature Scaling (deep learning)
        """)
    
    if calibrator:
        st.success("✅ Calibrador Venn-ABERS ativo no sistema")
    else:
        st.warning("⚠️ Calibrador não carregado. Execute notebook de calibração.")

#%% 7. Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("← Voltar para Insights Estratégicos"):
        st.switch_page("pages/04_insights.py")

with col_nav2:
    if st.button("Ir para Documentação Técnica →"):
        st.switch_page("pages/06_technical.py")

with col_nav3:
    if st.button("🏠 Home"):
        st.switch_page("00_home.py")


# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i>
</div>
""", unsafe_allow_html=True)