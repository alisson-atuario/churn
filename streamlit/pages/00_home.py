# Page: 00_home.py
import sys
from pathlib import Path

# Caminho absoluto para a pasta utils
project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
# import pandas as pd
from utils.data_loader import load_dataset
from utils.metrics import get_main_kpis, calculate_value_lost_distribution
from utils.plots import plot_churn_trend, plot_churn_distribution
# from utils.paths import DATA_DIR

# Configuração da página
st.set_page_config(
    page_title="Análise de Churn",
    page_icon="📊",
    layout="wide"
)

# Título principal com logo
col1, col2 = st.columns([1, 4])
# with col1:
#     st.image("streamlit/assets/logo_P&B.png", width=30)
with col2:
    st.title("Modelagem e Análise de Churn")

# st.image("streamlit/assets/theme.png", width=500)

# Visão Geral do Projeto
st.markdown("""
## Sobre o Projeto
Este dashboard apresenta uma análise completa de churn (cancelamento) de clientes,
baseada em dados históricos reais. Aqui você encontra métricas gerais, tendências
e insights sobre o comportamento dos clientes.
""")

# Carrega e calcula as KPIs
kpis = get_main_kpis()

# KPIs principais
st.markdown("## 📊 KPIs Principais")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Taxa de Churn",
        value=f"{kpis['churn_rate'].get('Yes', 0):.2f}%",
        help="Percentual de clientes que cancelaram o serviço"
    )

with col2:
    st.metric(
        label="Total de Clientes",
        value=f"{kpis['total_customers']:,}",
        help="Número total de clientes na base histórica"
    )

with col3:
    st.metric(
        label="Clientes com Churn",
        value=f"{kpis['churned_customers']:,}",
        help="Número absoluto de clientes que cancelaram"
    )

# Segunda linha de KPIs
col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        label="Valor Mensal Perdido",
        value=f"R$ {kpis['value_lost']:,.2f}",
        help="Valor total perdido mensalmente com clientes que cancelaram"
    )

with col5:
    st.metric(
        label="Tempo Médio (meses)",
        value=f"{kpis['avg_tenure']}",
        help="Tempo médio de permanência dos clientes"
    )

with col6:
    st.metric(
        label="Gasto Mensal Médio",
        value=f"R$ {kpis['avg_monthly_charge']:,.2f}",
        help="Valor médio mensal gasto por cliente"
    )

# Gráfico de tendência
st.markdown("## 📈 Tendência de Churn por Tempo de Permanência")
st.plotly_chart(plot_churn_trend(), width='stretch')

# Explicação do gráfico
st.info("""
💡 **Sobre o gráfico**: Este gráfico mostra como a taxa de churn varia de acordo com o tempo 
de permanência dos clientes (em meses). A barra vermelha indica a taxa de churn percentual, 
enquanto a linha azul mostra o número total de clientes em cada faixa de tempo.
""")

# Principais Insights (baseados na análise dos dados)
valor_perdido = calculate_value_lost_distribution(load_dataset())
st.markdown(f"""
## 🔍 Principais Insights

- **Perfil de Risco**: Clientes com contratos mensais apresentam maior taxa de churn
- **Tempo de Permanência**: Clientes mais novos (<12 meses) têm maior propensão a cancelar
- **Serviços**: Clientes com internet fibra ótica tendem a ter maior valor mensal
- **Valor**: Cerca de {valor_perdido:.1f}% do valor perdido vem de clientes com alto gasto mensal
""")

# Navegação Rápida
st.markdown("## Navegação Rápida")
col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("📊 Análise Exploratória", width='stretch'):
        st.switch_page("pages/01_exploratory.py")

with col_nav2:
    if st.button("🎯 Previsões", width='stretch'):
        st.switch_page("pages/02_predictions.py")

with col_nav3:
    if st.button("💡 Recomendações", width='stretch'):
        st.switch_page("pages/04_insights.py")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i>
</div>
""", unsafe_allow_html=True)