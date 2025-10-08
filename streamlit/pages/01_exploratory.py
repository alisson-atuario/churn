# Page: 01_exploratory.py
from pathlib import Path
import sys

# Caminho absoluto para a pasta utils
project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
from utils.data_loader import load_dataset
from utils.metrics import calculate_churn_rate, data_categoric,data_raw_treatment,data_categoric
from utils.plots import plot_churn_distribution, plot_histograms_by_churn, plot_numeric_cols_distribution,generate_graph_analyses

data = load_dataset()
churn_rate = calculate_churn_rate(data)

# Configuração da página
st.set_page_config(
    page_title="Análise Exploratória",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título da Página
st.title("Análise Exploratória do Churn")
st.markdown(f"""
Análise de {data.shape[0]} clientes com ~{churn_rate['Yes']}% de taxa de churn, 
explorando variáveis como tempo de permanência, gastos mensais, 
tipo de contrato e serviços de internet para identificar padrões de cancelamento
""")

#%% 1) Distribuição de Churn
st.header("1. Distribuição de Churn")   
with st.expander("Visualização",expanded=True):
    st.plotly_chart(plot_churn_distribution())
    st.info(f"""
    💡 Observa-se um desbalanceamento: ~{churn_rate['No']}% "No" e ~{churn_rate['Yes']}% "Yes". 
    Esse desbalanceamento deve ser levado em conta ao treinar modelos preditivos (pode impactar métricas como acurácia).
    """)

#%% 2) Variáveis Numéricas
st.header("2. Análise de Variáveis Numéricas")
with st.expander("Visualização",expanded=True):

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data = data_raw_treatment()
    available_numeric = [c for c in numeric_cols if c in data.columns]

    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    data = data_raw_treatment()
    available_numeric = [c for c in numeric_cols if c in data.columns]

    # Seletor com todas selecionadas por padrão
    selected_vars = st.multiselect(
        "Selecione as variáveis para análise:",
        options=available_numeric,
        default=available_numeric,  # Todas selecionadas por padrão
        key="var_selector"
    )

    for col in selected_vars:
        st.subheader(f"{col}")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.plotly_chart(plot_numeric_cols_distribution(col), width='stretch')
        
        with col_b:
            st.plotly_chart(plot_histograms_by_churn(col), width='stretch')
            # st.info("""💡 **Principais Achados da Análise Categórica**  
            #         - Clientes com **contratos mensais** e **até 12 meses de permanência** apresentam risco muito maior de churn.  
            #         - O método de pagamento também pesa: **Electronic check** é mais associado ao cancelamento.  
            #         - Entre os serviços, **fibra ótica** e ausência de **segurança/atendimento técnico** elevam a taxa de churn.  
            #         """)

    # Mensagem se nenhuma selecionada
    if not selected_vars:
        st.info("Selecione pelo menos uma variável para visualizar os gráficos.")
    else:
        st.info(
        """
        **Principais padrões observados nos gráficos de violino:**
        - **Tenure (Tempo de Contrato):** O churn é muito mais frequente nos **primeiros meses (< 12)**.
        A partir de 24 meses, a taxa de churn cai bastante — mostrando a importância dos **primeiros
        meses como fase crítica de retenção**.

        - **MonthlyCharges (Gasto Mensal):** Clientes com churn apresentam maior concentração de valores
        entre **70–100**, indicando que **quanto maior o gasto, maior a probabilidade de saída**.

        - **TotalCharges (Valor Total):** Clientes que cancelaram possuem valores totais mais baixos,
        já que interromperam o contrato cedo. Em contrapartida, clientes fiéis chegam a acumular
        valores acima de **8k**.
        """,icon="💡"
      )

#%% 3) Variáveis Categóricas
st.header("3. Análise de Variáveis Categóricas")
with st.expander("Visualização",expanded=True):

    categoric_values = data_categoric()
    available_categoric = categoric_values.columns.tolist()
    available_categoric.remove('Churn')

    # Seletor com todas selecionadas por padrão
    selected_categoric = st.multiselect(
        "Selecione as variáveis para análise:",
        options=available_categoric,
        default=['Contract','tenure_cat','PaymentMethod','InternetService'],  # selecionadas por padrão
        key="var_selector_categoric"
    )

    # Mensagem se nenhuma selecionada
    if not selected_categoric:
        st.info("Selecione pelo menos uma variável para visualizar os gráficos.")
    else:
        for i in range(0,len(selected_categoric),2):
            col_a, col_b = st.columns(2)

            # Primeira Coluna
            with col_a:
                col_name = selected_categoric[i]
                st.subheader(f"{col_name}")
                st.plotly_chart(generate_graph_analyses(col_name), 
                width='stretch',key=f"plot_{col_name}_{i}")

             # Segunda Coluna
            with col_b:
                if i+1 < len(selected_categoric):
                    col_name = selected_categoric[i+1]
                    st.subheader(f"{col_name}")
                    st.plotly_chart(generate_graph_analyses(col_name),
                     width='stretch',key=f"plot_{col_name}_{i+1}")
            


        st.info(
        """
        **Principais Achados da Análise Categórica:**

        - Clientes com **contratos mensais** e **até 12 meses de permanência** apresentam risco muito maior de churn. Isso significa que clientes novos são muito mais vulneráveis.

        - O método de pagamento também pesa: **Electronic check** é mais associado ao cancelamento. Pagamentos menos automatizados possuem maior risco de cancelamento.  

        - Entre os serviços, **fibra ótica** e ausência de **segurança/atendimento técnico** elevam a taxa de churn. Isso pode indicar expectativa alta vs. insatisfação com o serviço.
        
        """,icon="💡"
        )

st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    if st.button("Ir para Previsões →", width='stretch'):
        st.switch_page("pages/02_predictions.py")

with col_nav2:
    if st.button("🏠 Home", width='stretch'):
        st.switch_page("00_home.py")

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <i>Dashboard de Análise de Churn | Desenvolvido por Alisson Ursulino</i>
</div>
""", unsafe_allow_html=True)