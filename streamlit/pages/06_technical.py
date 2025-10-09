# Page: 05_technical.py
from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_model, load_calibrator#, load_test_data
from utils.metrics import get_model_predictions, calculate_model_metrics
import joblib

st.set_page_config(
    page_title="Documentação Técnica",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Documentação Técnica do Modelo")
st.markdown("""
Detalhes técnicos sobre arquitetura, hiperparâmetros, processo de treinamento e calibração do modelo.
""")

# Carrega modelo e dados
try:
    model = load_model()
    calibrator = load_calibrator()
    print(calibrator)
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()

#%% 1. Visão Geral da Arquitetura
st.header("1. Arquitetura do Sistema")

with st.expander("Pipeline de Machine Learning", expanded=True):
    st.markdown("""
    ```mermaid
    graph LR
        A[Dados Brutos] --> B[Pré-processamento]
        B --> C[Feature Engineering]
        C --> D[Split: Train/Cal/Test]
        D --> E[Undersampling]
        E --> F[LightGBM + RandomizedSearch]
        F --> G[Venn-ABERS Calibration]
        G --> H[Modelo Final]
        H --> I[Predições + SHAP]
    ```
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Etapas de Processamento:**
        
        1. **Limpeza de Dados**
           - Tratamento de valores ausentes
           - Conversão de tipos
           - Remoção de outliers
        
        2. **Feature Engineering**
           - Encoding categórico
           - Normalização numérica
           - Criação de features derivadas
        
        3. **Balanceamento**
           - RandomUnderSampler no treino
           - Mantém distribuição original no teste
        """)
    
    with col2:
        st.markdown("""
        **Validação e Teste:**
        
        1. **Split Estratificado**
           - Train: 70% (com undersampling)
           - Calibration: 15%
           - Test: 15%
        
        2. **Calibração**
           - Venn-ABERS para probabilidades
           - Otimização de threshold (F2-score)
        
        3. **Explicabilidade**
           - SHAP values para interpretação
           - Feature importance global e local
        """)

#%% 2. Algoritmo e Hiperparâmetros
st.header("2. Algoritmo: LightGBM")

with st.expander("Detalhes do Modelo", expanded=True):
    st.markdown("""
    **LightGBM (Light Gradient Boosting Machine)**
    
    - **Tipo**: Gradient Boosting Decision Tree (GBDT)
    - **Vantagens**: Rápido, eficiente com categóricas, baixo overfitting
    - **Ideal para**: Dados tabulares com classes desbalanceadas
    """)
    
    # Pega os parâmetros do modelo
    params = model.get_params()
    
    # Organiza parâmetros por categoria
    param_categories = {
        'Estrutura da Árvore': ['num_leaves', 'max_depth', 'min_child_samples', 'min_child_weight'],
        'Aprendizado': ['learning_rate', 'n_estimators', 'boosting_type'],
        'Regularização': ['lambda_l1', 'lambda_l2', 'min_split_gain'],
        'Amostragem': ['feature_fraction', 'bagging_fraction', 'bagging_freq']
    }
    
    tabs = st.tabs(list(param_categories.keys()))
    
    for tab, (category, param_names) in zip(tabs, param_categories.items()):
        with tab:
            params_df = pd.DataFrame([
                {'Parâmetro': name, 'Valor': params.get(name, 'N/A')}
                for name in param_names if name in params
            ])
            st.dataframe(params_df, hide_index=True)

#%% 3. Processo de Otimização
st.header("3. Otimização de Hiperparâmetros")

with st.expander("RandomizedSearchCV", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Configuração da Busca:**
        
        - **Método**: RandomizedSearchCV
        - **Iterações**: 50 combinações
        - **Cross-Validation**: 5-fold
        - **Métrica**: neg_log_loss
        - **Paralelização**: n_jobs=-1
        """)
    
    with col2:
        st.markdown("""
        **Espaço de Busca:**
        
        - **num_leaves**: 20-100
        - **max_depth**: 3-12
        - **learning_rate**: 0.01-0.31
        - **n_estimators**: 100-1000
        - **regularização**: L1/L2 (0-1)
        """)
    
    st.info("""
    **Por que RandomizedSearch ao invés de GridSearch?**
    
    - ✅ Mais eficiente: explora espaço de busca mais amplo com menos iterações
    - ✅ Menor custo computacional: 50 fits vs 100.000+ do Grid
    - ✅ Resultados comparáveis: estudos mostram performance similar
    """)

#%% 4. Calibração de Probabilidades
st.header("4. Calibração: Venn-ABERS")

with st.expander("Metodologia de Calibração", expanded=True):
    st.markdown("""
    ### O que é Calibração?
    
    Calibração ajusta as probabilidades preditas para que reflitam a **verdadeira frequência** de eventos.
    
    **Exemplo:**
    - Modelo diz: "80% de chance de churn"
    - Calibrado: "De 100 clientes com essa probabilidade, 80 realmente deram churn"
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Venn-ABERS Calibrator**
        
        - **Tipo**: Não-paramétrico
        - **Vantagens**:
          - Funciona com poucos dados
          - Sem pressupostos sobre distribuição
          - Gera intervalos de confiança
        
        - **Processo**:
          1. Treina em conjunto de calibração
          2. Ajusta probabilidades via conformal prediction
          3. Otimiza threshold por F2-score (prioriza Recall)
        """)
    
    with col2:
        if calibrator:
            st.success("✅ Modelo calibrado disponível")
            st.markdown("""
            **Métricas de Calibração:**
            
            - ECE (Expected Calibration Error): Mede desvio médio
            - Brier Score: Penaliza previsões erradas
            - Curva de calibração: Visualiza ajuste
            
            **Threshold Otimizado:**
            - Padrão: 0.5
            - Otimizado: ~0.42 (depende do F2-score)
            - Favorece Recall (menos clientes escapam)
            """)
        else:
            st.warning("⚠️ Calibrador não carregado")

#%% 5. Explicabilidade (SHAP)
st.header("5. Explicabilidade: SHAP Values")

with st.expander("Metodologia SHAP", expanded=True):
    st.markdown("""
    ### SHAP (SHapley Additive exPlanations)
    
    Baseado em **teoria dos jogos**, calcula a contribuição marginal de cada feature para a predição.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Como Funciona:**
        
        1. **Valor Base**: Predição média do modelo
        2. **Contribuições**: Cada feature empurra para cima/baixo
        3. **Predição Final**: Base + Σ(contribuições)
        
        **Exemplo:**
        ```
        Base value: 0.265 (26.5% de churn médio)
        + Contract_Monthly: +0.180
        + Tenure_<12m: +0.120
        - PaymentMethod_Auto: -0.050
        ─────────────────────────
        = Predição final: 0.515 (51.5%)
        ```
        """)
    
    with col2:
        st.markdown("""
        **Tipos de Análise:**
        
        - **Global**: Importância média de todas features
        - **Local**: Contribuição para cliente específico
        - **Dependência**: Como feature impacta predição
        - **Interações**: Efeitos combinados
        
        **Vantagens:**
        - Matematicamente rigoroso
        - Consistente e justo
        - Agnóstico ao modelo
        """)

#%% 6. Métricas de Performance
st.header("6. Métricas de Avaliação")

with st.expander("Indicadores Chave", expanded=True):
    metrics = calculate_model_metrics(y_cal, y_pred_cal, y_pred_proba_cal)
    class_report = metrics['classification_report']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AUC-ROC", f"{metrics['auc_score']:.3f}")
        st.caption("Capacidade de distinguir classes")
    
    with col2:
        st.metric("F1-Score (Churn)", f"{class_report['1']['f1-score']:.3f}")
        st.caption("Balanço Precision/Recall")
    
    with col3:
        st.metric("Recall (Churn)", f"{class_report['1']['recall']:.3f}")
        st.caption("Taxa de detecção de churn")
    
    st.markdown("""
    **Métricas Complementares:**
    
    | Métrica | Valor | Interpretação |
    |---------|-------|---------------|
    | Precision | {precision:.3f} | {precision_pct:.1f}% dos alertas são corretos |
    | Recall | {recall:.3f} | {recall_pct:.1f}% dos churns são detectados |
    | F1-Score | {f1:.3f} | Balanço entre os dois |
    | AUC-ROC | {auc:.3f} | Discriminação geral do modelo |
    """.format(
        precision=class_report['1']['precision'],
        recall=class_report['1']['recall'],
        f1=class_report['1']['f1-score'],
        auc=metrics['auc_score'],
        precision_pct=class_report['1']['precision']*100,
        recall_pct=class_report['1']['recall']*100
    ))

#%% 7. Limitações e Melhorias Futuras
st.header("7. Limitações e Roadmap")

with st.expander("Considerações Importantes", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Limitações Atuais:**
        
        - **Dados estáticos**: Modelo não se atualiza automaticamente
        - **Features limitadas**: Apenas dados demográficos e contratuais
        - **Temporal**: Não captura sazonalidade
        - **Desbalanceamento**: Mesmo com undersampling, há viés
        """)
    
    with col2:
        st.markdown("""
        **Melhorias Futuras:**
        
        - ✨ **Re-treinamento automático**: Pipeline MLOps
        - ✨ **Features comportamentais**: Uso, reclamações, NPS
        - ✨ **Modelos temporais**: LSTM para séries temporais
        - ✨ **Ensemble**: Combinar múltiplos algoritmos
        - ✨ **A/B Testing**: Validar estratégias de retenção
        """)

#%% 8. Versionamento
st.header("8. Informações de Versão")

with st.expander("Metadados do Modelo", expanded=True):
    metadata = {
        'Modelo': 'LightGBM v4.5.0',
        'Calibração': 'Venn-ABERS' if calibrator else 'Não aplicada',
        'Data de Treinamento': '2025-01-15',
        'Dataset': f'{len(X_cal) + len(X_test)} amostras',
        'Features': f'{X_test.shape[1]} variáveis',
        'Versão do Código': 'v1.2.0'
    }
    
    df_metadata = pd.DataFrame(list(metadata.items()), columns=['Item', 'Valor'])
    st.dataframe(df_metadata, hide_index=True)

# Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("← Voltar para Calibração"):
        st.switch_page("pages/05_calibration.py")

with col_nav2:
    if st.button("Ir para Configurações →"):
        st.switch_page("pages/07_settings.py")

with col_nav3:
    if st.button("🏠 Home"):
        st.switch_page("pages/00_home.py")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i><br>
    <a href='https://alisson-atuario.github.io/' target='_blank' style='color: #3498db; text-decoration: none;'>(Ver portfólio e outros projetos)</a>
</div>
""", unsafe_allow_html=True)