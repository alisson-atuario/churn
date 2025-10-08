# Page: 03_simulator.py
from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
import pandas as pd
from utils.metrics import (
    get_model_predictions,
    get_client_prediction_details,
    load_shap_values,
    explain_prediction
)
from utils.plots import (
    plot_shap_waterfall,
    plot_shap_dependence
)

st.set_page_config(
    page_title="Simulador de Churn",
    page_icon="🔍",
    layout="wide"
)

st.title("Simulador - Análise Individual de Clientes")
st.markdown("""
Explore predições específicas e entenda **por que** cada cliente tem determinado risco de churn.
Use análise SHAP para ver o impacto de cada característica.
""")

# Carrega dados
def load_simulator_data():
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal = get_model_predictions()
    shap_values, expected_value, feature_names = load_shap_values()
    return X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal, shap_values, expected_value, feature_names

try:
    X_cal, y_cal, X_test, y_test, y_pred, y_pred_proba, y_pred_cal, y_pred_proba_cal, shap_values, expected_value, feature_names = load_simulator_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()

# Sidebar - Seleção de Cliente
st.sidebar.header("Seletor de Cliente")
# Opção 1: Por índice
sample_idx = st.sidebar.slider(
    "Selecione pelo índice:",
    min_value=0, # 318 ára debug, posteriormente será alterado para 0
    max_value=X_cal.shape[0] - 1,
    value=0
)

# Opção 2: Filtro por risco
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros Rápidos:**")

risk_filter = st.sidebar.radio(
    "Mostrar clientes:",
    ["Todos", "Alto Risco (>70%)", "Médio Risco (30-70%)", "Baixo Risco (<30%)"]
)

if risk_filter != "Todos":
    if risk_filter == "Alto Risco (>70%)":
        filtered_idx = [i for i, p in enumerate(y_pred_proba_cal) if p > 0.7]
    elif risk_filter == "Médio Risco (30-70%)":
        filtered_idx = [i for i, p in enumerate(y_pred_proba_cal) if 0.3 <= p <= 0.7]
    else:  # Baixo Risco
        filtered_idx = [i for i, p in enumerate(y_pred_proba_cal) if p < 0.3]
    
    if filtered_idx:
        st.sidebar.success(f"Encontrados {len(filtered_idx)} clientes neste segmento")
        sample_idx = st.sidebar.selectbox(
            "Escolha um cliente:",
            filtered_idx,
            format_func=lambda x: f"Cliente {x} (prob: {y_pred_proba_cal[x]:.1%})"
        )
    else:
        st.sidebar.warning("Nenhum cliente neste filtro")

# Detalhes do cliente selecionado
client_details = get_client_prediction_details(X_cal, y_cal, y_pred_cal, y_pred_proba_cal, sample_idx)

# Header com informações principais
st.header(f"Cliente {sample_idx}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    prob = client_details['probability']
    color = "🔴" if prob > 0.7 else "🟡" if prob > 0.3 else "🟢"
    st.metric("Risco de Churn", f"{color} {prob:.1%}")

with col2:
    pred_label = "Churn" if client_details['prediction'] == 1 else "Não Churn"
    st.metric("Predição do Modelo", pred_label)

with col3:
    actual_label = "Churn" if client_details['actual'] == 1 else "Não Churn"
    delta_color = "off" if client_details['prediction'] == client_details['actual'] else "inverse"
    st.metric("Valor Real", actual_label)

with col4:
    acertou = "✅ Acertou" if client_details['prediction'] == client_details['actual'] else "❌ Errou"
    st.metric("Status", acertou)

#%% 1. Explicação SHAP (Waterfall)
st.header("1. Análise de Contribuições - SHAP Waterfall")

with st.expander("Visualização", expanded=True):
    st.plotly_chart(
        plot_shap_waterfall(shap_values, expected_value, feature_names, sample_idx),
    )
    
    st.info("""
    **Como interpretar:**
    - **Base Value**: Probabilidade média de churn (linha de partida)
    - **Barras vermelhas**: Features que AUMENTAM risco de churn
    - **Barras azuis**: Features que DIMINUEM risco de churn
    - **Final Value**: Probabilidade final calculada para este cliente
    
    Quanto maior a barra, maior o impacto daquela característica.
    """)

#%% 2. Explicação Textual
st.header("2. Principais Fatores de Risco")

with st.expander("Explicação Detalhada", expanded=True):
    explanation = explain_prediction(shap_values, expected_value, feature_names, sample_idx, top_n=5)
    
    st.markdown("### Top 5 Features Mais Impactantes:")
    
    for i, feature in enumerate(explanation['features'], 1):
        impact_emoji = "🔺" if feature['impact'] == "aumenta" else "🔻"
        col_a, col_b = st.columns([3, 1])
        
        with col_a:
            st.markdown(f"**{i}. {feature['feature']}** {impact_emoji}")
            st.markdown(f"*{feature['impact'].capitalize()} o risco de churn*")
        
        with col_b:
            st.metric("Impacto", f"{feature['value']:.3f}")

#%% 3. Características do Cliente
st.header("3. Perfil Completo do Cliente")

with st.expander("Ver todas as características", expanded=False):
    # Organiza features em 3 colunas
    features_df = pd.DataFrame.from_dict(
        client_details['features'], 
        orient='index', 
        columns=['Valor']
    )
    features_df.index.name = 'Feature'
    
    st.dataframe(features_df)

#%% 4. Análise de Dependência (Feature Específica)
st.header("4. Análise de Dependência - Como uma Feature Impacta?")

with st.expander("Visualização", expanded=False):
    selected_feature = st.selectbox(
        "Selecione uma feature para análise detalhada:",
        feature_names,
        index=0
    )
    
    feature_idx = feature_names.index(selected_feature)
    
    st.plotly_chart(
        plot_shap_dependence(shap_values, X_cal.values, feature_names, feature_idx),
        width='stretch'
    )
    
    st.info(f"""
    **Interpretação:**
    - Cada ponto é um cliente
    - **Eixo X**: Valor da feature "{selected_feature}"
    - **Eixo Y**: Impacto SHAP (quanto aumenta/diminui probabilidade de churn)
    - **Cor**: Valor da feature (vermelho = alto, azul = baixo)
    
    Padrão crescente = quanto maior o valor, maior o risco de churn.
    """)

#%% 5. Comparação com outros clientes
st.header("5. Comparação com Clientes Similares")

with st.expander("Análise Comparativa", expanded=False):
    # Encontra clientes com probabilidade similar (±10%)
    similar_range = 0.1
    similar_clients = [
        i for i, p in enumerate(y_pred_proba_cal) 
        if abs(p - client_details['probability']) < similar_range and i != sample_idx
    ][:5]  # Pega até 5 clientes similares
    
    if similar_clients:
        st.markdown(f"**Clientes com risco similar (±{similar_range*100:.0f}%):**")
        
        comparison_data = []
        for idx in similar_clients:
            comparison_data.append({
                'Cliente': f"#{idx}",
                'Probabilidade': f"{y_pred_proba_cal[idx]:.1%}",
                'Predição': "Churn" if y_pred_cal[idx] == 1 else "Não Churn",
                'Real': "Churn" if y_cal.iloc[idx] == 1 else "Não Churn"
            })
        
        st.dataframe(pd.DataFrame(comparison_data))
    else:
        st.info("Nenhum cliente com risco muito similar encontrado no dataset.")

#%% 6. Ações Recomendadas
st.header("6. Recomendações de Ação")

with st.expander("Estratégias de Retenção", expanded=True):
    prob = client_details['probability']
    
    if prob > 0.7:
        st.error("""
        **🔴 ALTO RISCO - Ação Imediata Necessária**
        
        Recomendações:
        - Contato proativo da equipe de retenção
        - Oferta personalizada com desconto/benefício
        - Investigar pontos de insatisfação específicos
        - Agendar reunião para entender necessidades
        - Considerar upgrade de serviço ou mudança de plano
        """)
    elif prob > 0.3:
        st.warning("""
        **🟡 MÉDIO RISCO - Monitoramento Ativo**
        
        Recomendações:
        - Incluir em campanha preventiva de retenção
        - Enviar pesquisa de satisfação
        - Oferecer suporte técnico proativo
        - Comunicar novidades e melhorias do serviço
        - Avaliar histórico de atendimento
        """)
    else:
        st.success("""
        **🟢 BAIXO RISCO - Manutenção de Relacionamento**
        
        Recomendações:
        - Manter comunicação regular
        - Incluir em programas de fidelidade
        - Oferecer upgrades quando relevante
        - Pedir feedback sobre experiência positiva
        - Considerar como embaixador da marca
        """)

#%% Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("← Voltar para Predictions"):
        st.switch_page("pages/02_predictions.py")

with col_nav2:
    if st.button("Ver Insights Globais →"):
        st.switch_page("pages/04_insights.py")

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