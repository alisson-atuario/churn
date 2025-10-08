
# Page: 06_settings.py
from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_loader import load_model, load_calibrator, load_test_data
from utils.metrics import get_model_predictions, calculate_model_metrics, create_risk_segments
import io

st.set_page_config(
    page_title="Configurações",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Configurações e Ferramentas")
st.markdown("""
Ajuste configurações do modelo, exporte relatórios e gerencie parâmetros do sistema.
""")

# Initialize session state for settings
if 'use_calibration' not in st.session_state:
    st.session_state.use_calibration = True
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'risk_thresholds' not in st.session_state:
    st.session_state.risk_thresholds = {'low': 0.3, 'high': 0.7}

#%% 1. Configurações do Modelo
st.header("1. Configurações do Modelo")

with st.expander("Parâmetros de Predição", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Calibração")
        
        calibrator = load_calibrator()
        use_calibration = st.checkbox(
            "Usar modelo calibrado (Venn-ABERS)",
            value=st.session_state.use_calibration,
            help="Ativa calibração de probabilidades para maior precisão",
            disabled=calibrator is None
        )
        st.session_state.use_calibration = use_calibration
        
        if calibrator is None:
            st.warning("⚠️ Calibrador não disponível. Execute o notebook de calibração.")
        elif use_calibration:
            st.success("✅ Calibração ativada")
        else:
            st.info("ℹ️ Usando modelo sem calibração")
    
    with col2:
        st.markdown("### Threshold de Decisão")
        
        threshold = st.slider(
            "Threshold para classificação como Churn",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.05,
            help="Probabilidade mínima para classificar como churn"
        )
        st.session_state.threshold = threshold
        
        if threshold < 0.5:
            st.info(f"🔻 Threshold baixo ({threshold:.2f}): Mais sensível, detecta mais churns mas com mais falsos positivos")
        elif threshold > 0.5:
            st.info(f"🔺 Threshold alto ({threshold:.2f}): Mais conservador, menos falsos positivos mas pode perder churns")
        else:
            st.info(f"⚖️ Threshold balanceado (0.50)")

#%% 2. Segmentação de Risco
st.header("2. Configuração de Segmentos de Risco")

with st.expander("Thresholds de Segmentação", expanded=True):
    st.markdown("""
    Define os limites de probabilidade para classificar clientes em segmentos de risco.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 Baixo Risco")
        low_max = st.slider(
            "Máximo para Baixo Risco",
            min_value=0.0,
            max_value=0.5,
            value=st.session_state.risk_thresholds['low'],
            step=0.05,
            help="Clientes com probabilidade até este valor"
        )
    
    with col2:
        st.markdown("### 🟡 Médio Risco")
        st.metric("Faixa", f"{low_max:.2f} - {st.session_state.risk_thresholds['high']:.2f}")
        st.caption("Entre Baixo e Alto Risco")
    
    with col3:
        st.markdown("### 🔴 Alto Risco")
        high_min = st.slider(
            "Mínimo para Alto Risco",
            min_value=0.5,
            max_value=1.0,
            value=st.session_state.risk_thresholds['high'],
            step=0.05,
            help="Clientes com probabilidade acima deste valor"
        )
    
    # Validação
    if low_max >= high_min:
        st.error("❌ Erro: Threshold de Baixo Risco deve ser menor que Alto Risco")
    else:
        st.session_state.risk_thresholds = {'low': low_max, 'high': high_min}
        st.success(f"✅ Segmentos configurados: Baixo [0-{low_max:.2f}], Médio [{low_max:.2f}-{high_min:.2f}], Alto [{high_min:.2f}-1.0]")

#%% 3. Exportação de Dados
st.header("3. Exportar Relatórios")

with st.expander("Download de Dados", expanded=True):
    try:
        X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Predições do Modelo")
            
            # Cria DataFrame com predições
            predictions_df = pd.DataFrame({
                'Cliente_ID': range(len(y_pred)),
                'Probabilidade_Churn': y_pred_proba,
                'Predicao': ['Churn' if p == 1 else 'Não Churn' for p in y_pred],
                'Valor_Real': ['Churn' if y == 1 else 'Não Churn' for y in y_test],
                'Correto': y_pred == y_test
            })
            
            # Adiciona segmento de risco
            def classify_risk(prob):
                if prob <= st.session_state.risk_thresholds['low']:
                    return 'Baixo Risco'
                elif prob <= st.session_state.risk_thresholds['high']:
                    return 'Médio Risco'
                else:
                    return 'Alto Risco'
            
            predictions_df['Segmento_Risco'] = predictions_df['Probabilidade_Churn'].apply(classify_risk)
            
            st.dataframe(predictions_df.head(10))
            
            # Botão de download
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predições (CSV)",
                data=csv,
                file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                width='stretch'
            )
        
        with col2:
            st.markdown("### Métricas de Performance")
            
            metrics = calculate_model_metrics(y_test, y_pred, y_pred_proba)
            
            # Cria relatório de métricas
            metrics_report = {
                'Métrica': [
                    'AUC-ROC',
                    'Precision (Churn)',
                    'Recall (Churn)',
                    'F1-Score (Churn)',
                    'Acurácia Geral'
                ],
                'Valor': [
                    f"{metrics['auc_score']:.4f}",
                    f"{metrics['classification_report']['1']['precision']:.4f}",
                    f"{metrics['classification_report']['1']['recall']:.4f}",
                    f"{metrics['classification_report']['1']['f1-score']:.4f}",
                    f"{metrics['classification_report']['accuracy']:.4f}"
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_report)
            st.dataframe(metrics_df, hide_index=True)
            
            # Download métricas
            metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Métricas (CSV)",
                data=metrics_csv,
                file_name=f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                width='stretch'
            )
    
    except Exception as e:
        st.error(f"Erro ao gerar relatórios: {str(e)}")

#%% 4. Estatísticas do Dataset
st.header("4. Estatísticas do Sistema")

with st.expander("Informações Gerais", expanded=True):
    try:
        X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Amostras", f"{len(X_cal) + len(X_test):,}")
        with col2:
            st.metric("Features", X_test.shape[1])
        with col3:
            churn_rate = (y_test == 1).mean() * 100
            st.metric("Taxa de Churn Real", f"{churn_rate:.1f}%")
        with col4:
            accuracy = (y_pred == y_test).mean() * 100
            st.metric("Acurácia do Modelo", f"{accuracy:.1f}%")
        
        # Tabela de segmentação
        st.markdown("### Distribuição por Segmento de Risco")
        risk_summary = create_risk_segments(y_pred_proba, y_test)
        st.dataframe(risk_summary)
        
    except Exception as e:
        st.error(f"Erro ao carregar estatísticas: {str(e)}")

#%% 5. Manutenção do Modelo
st.header("5. Manutenção e Atualização")

with st.expander("Ferramentas de Administração", expanded=True):
    st.markdown("""
    ### ⚠️ Ações Administrativas
    
    Estas ações requerem acesso ao backend do sistema.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Re-treinamento do Modelo**
        
        - Frequência recomendada: Trimestral
        - Requisitos: Novos dados rotulados
        - Processo: Executar notebook de treinamento
        
        📝 Próximo re-treinamento sugerido: **Abril 2025**
        """)
        
        if st.button("🔄 Solicitar Re-treinamento", disabled=True):
            st.info("Feature em desenvolvimento")
    
    with col2:
        st.markdown("""
        **Validação do Modelo**
        
        - Monitorar drift de dados
        - Verificar degradação de performance
        - Atualizar calibração se necessário
        
        📊 Última validação: **Janeiro 2025**
        """)
        
        if st.button("✅ Executar Validação", disabled=True):
            st.info("Feature em desenvolvimento")

#%% 6. Logs e Histórico
st.header("6. Logs do Sistema")

with st.expander("Histórico de Uso", expanded=False):
    st.markdown("""
    ### Registro de Atividades
    
    *Feature em desenvolvimento*
    
    Funcionalidades planejadas:
    - Histórico de predições
    - Mudanças de configuração
    - Ações de usuários
    - Auditoria de decisões
    """)

# 7. Sobre
st.header("7. ℹ️ Sobre o Sistema")

with st.expander("Informações", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Dashboard de Análise de Churn**
        
        - Versão: 1.2.0
        - Desenvolvido com: Streamlit + LightGBM
        - Última atualização: Janeiro 2025
        - Licença: MIT
        """)
    
    with col2:
        st.markdown("""
        **Contato e Suporte**
        
        - 📧 Email: alisson.atuario@gmail.com
        - 📚 Documentação: [docs.churnanalysis.com](https://docs.churnanalysis.com)
        - 🐛 Reportar bug: [GitHub Issues](https://github.com/alisson-atuario)
        """)

# Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    if st.button("← Voltar para Documentação Técnica"):
        st.switch_page("pages/05_technical.py")

with col_nav2:
    if st.button("🏠 Home"):
        st.switch_page("00_home.py")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i>
</div>
""", unsafe_allow_html=True)