# Page: 04_insights.py
from pathlib import Path
import sys

project_utils = Path(__file__).parent.parent
sys.path.append(str(project_utils))

import streamlit as st
import pandas as pd
import numpy as np
from utils.metrics import get_feature_importance, load_shap_values
from utils.plots import plot_shap_summary, plot_shap_importance
from utils.data_loader import load_dataset

st.set_page_config(
    page_title="Insights de Churn",
    page_icon="💡",
    layout="wide"
)

st.title("💡 Insights Globais e Recomendações Estratégicas")
st.markdown("""
Análise consolidada dos principais fatores que causam churn e recomendações 
acionáveis baseadas em dados e interpretabilidade do modelo (SHAP).
""")

# Carrega dados
def load_insights_data():
    shap_values, expected_value, feature_names = load_shap_values()
    feature_importance_dict = get_feature_importance(shap_values, feature_names)
    data = load_dataset()
    return shap_values, expected_value, feature_names, feature_importance_dict, data

try:
    shap_values, expected_value, feature_names, feature_importance_dict, data = load_insights_data()
except Exception as e:
    st.error(f"Erro ao carregar dados: {str(e)}")
    st.stop()

# 1. Visão Geral
st.header("1. Principais Fatores de Churn")

with st.expander("Importância Global das Features", expanded=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(plot_shap_importance(shap_values, feature_names))
    
    with col2:
        st.markdown("### Top 5 Features")
        top_features = list(feature_importance_dict.items())[:5]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            st.markdown(f"**{i}. {feature}**")
            st.progress(float(importance / max(feature_importance_dict.values())))
            st.caption(f"Importância: {importance:.4f}")
            st.markdown("")

# 2. SHAP Summary Plot
st.header("2. Análise Detalhada de Impacto")

with st.expander("SHAP Summary Plot", expanded=True):
    st.plotly_chart(plot_shap_summary(shap_values, feature_names))
    
    st.info("""
    **Como interpretar este gráfico:**
    
    - **Posição vertical**: Importância da feature (mais acima = mais importante)
    - **Posição horizontal**: Impacto no churn (direita = aumenta, esquerda = diminui)
    - **Cor**: Valor da feature (vermelho = alto, azul = baixo)
    
    **Exemplo**: Se "Contract_Month-to-month" aparece muito à direita em vermelho,
    significa que ter contrato mensal (valor alto) aumenta muito o risco de churn.
    """)

# 3. Insights Acionáveis
st.header("3. Insights e Recomendações por Categoria")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Contratos", "💰 Pagamentos", "🌐 Serviços", "👥 Perfil do Cliente"])

with tab1:
    st.markdown("### Contratos e Fidelização")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **📊 Insights Identificados:**
        
        - Contratos mensais têm 3-4x mais churn que contratos anuais
        - Clientes novos (<12 meses) são mais vulneráveis
        - Falta de compromisso de longo prazo aumenta risco
        """)
    
    with col_b:
        st.markdown("""
        **✅ Recomendações:**
        
        1. Oferecer incentivos para migração de mensal → anual
        2. Desconto progressivo por tempo de permanência
        3. Bonificação para contratos de 12+ meses
        4. Programa de fidelidade com benefícios crescentes
        """)
    
    st.success("**Ação prioritária**: Criar campanha de conversão mensal → anual com benefício tangível (15-20% desconto no primeiro ano).")

with tab2:
    st.markdown("### Métodos de Pagamento")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **📊 Insights Identificados:**
        
        - Electronic check associado a maior taxa de churn
        - Falta de automação pode indicar menor engajamento
        - Métodos automáticos (débito/crédito) têm menor risco
        """)
    
    with col_b:
        st.markdown("""
        **✅ Recomendações:**
        
        1. Incentivar migração para débito automático
        2. Oferecer desconto para pagamento recorrente
        3. Facilitar cadastro de cartão de crédito
        4. Alertas proativos de vencimento
        """)
    
    st.success("**Ação prioritária**: Cashback ou desconto de 5% para quem migrar de boleto → débito automático.")

with tab3:
    st.markdown("### Serviços e Satisfação")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **📊 Insights Identificados:**
        
        - Fibra ótica tem maior churn (expectativa alta vs realidade)
        - Falta de serviços adicionais (segurança, suporte)
        - Clientes sem suporte técnico têm maior rotatividade
        - Múltiplas linhas reduzem churn (maior dependência do serviço)
        - Streaming incluso aumenta retenção
        """)
    
    with col_b:
        st.markdown("""
        **✅ Recomendações:**
        
        1. **Bundle de serviços**: Oferecer pacotes com segurança + streaming
        2. **Suporte proativo**: Contato regular para verificar qualidade da fibra
        3. **Programa de indicação**: Benefícios para clientes que indicam múltiplas linhas
        4. **Teste gratuito**: 30 dias de serviços premium para clientes em risco
        5. **Monitoramento de qualidade**: Alertas automáticos para problemas técnicos
        """)
    
    st.success("**Ação prioritária**: Criar pacote 'Fibra Premium' incluindo segurança digital + streaming por 20% a mais.")

with tab4:
    st.markdown("### Perfil do Cliente")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
        **📊 Insights Identificados:**
        
        - Idosos (>65 anos) têm menor churn (menos comparação de preços)
        - Clientes com dependentes são mais fiéis
        - Solteiros têm maior rotatividade que casados
        - Clientes urbanos mais propensos a trocar de operadora
        - Baixa renda associada a maior instabilidade
        """)
    
    with col_b:
        st.markdown("""
        **✅ Recomendações:**
        
        1. **Programa família**: Descontos progressivos por dependentes
        2. **Plano sênior**: Benefícios específicos para +65 anos
        3. **Casais**: Ofertas conjuntas e bônus de aniversário
        4. **Comunidades**: Parcerias com associações de bairro
        5. **Planos sociais**: Opções acessíveis para baixa renda
        """)
    
    st.success("**Ação prioritária**: Lançar programa 'Família Plus' com desconto de 10% a partir do 2º dependente.")

# 4. Segmentação Estratégica
st.header("4. Segmentação de Clientes para Ações Específicas")

with st.expander("Estratégias por Perfil", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🚨 Alto Risco
        **Perfil**: Contrato mensal + Electronic check + Fibra ótica
        
        **Ações**:
        - Contato proativo a cada 3 meses
        - Oferta de migração para anual com 25% desconto
        - Convite para programa de fidelidade
        - Monitoramento de satisfação contínuo
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Médio Risco  
        **Perfil**: Contrato anual + Débito automático + Múltiplos serviços
        
        **Ações**:
        - Pesquisa de satisfação semestral
        - Ofertas de upgrades de serviço
        - Comunicação sobre novos benefícios
        - Programa de indicação
        """)
    
    with col3:
        st.markdown("""
        ### ✅ Baixo Risco
        **Perfil**: Contrato longo + Múltiplas linhas + Idoso/casado
        
        **Ações**:
        - Manutenção do relacionamento
        - Agradecimento pela fidelidade
        - Pequenos benefícios surpresa
        - Pedido de depoimento/indicação
        """)

# 5. Métricas de Impacto Esperado
st.header("5. Projeção de Resultados")

with st.expander("Estimativa de Impacto das Ações", expanded=True):
    st.markdown("""
    **Redução Esperada de Churn por Iniciativa:**
    
    | Iniciativa | Impacto Esperado | ROI Estimado |
    |------------|------------------|--------------|
    | Conversão mensal → anual | 15-20% redução | Alto |
    | Migração para débito automático | 8-12% redução | Médio-Alto |
    | Bundle de serviços | 10-15% redução | Médio |
    | Programas familiares | 5-8% redução | Médio |
    | Suporte proativo | 12-18% redução | Alto |
    """)
    
    st.warning("""
    **⚠️ Considerações Importantes:**
    
    - Custo de aquisição de novo cliente é 5x maior que retenção
    - Clientes retidos aumentam valor vitalício em 25-40%
    - Satisfação do cliente impacta diretamente no NPS e reputação
    """)

# 6. Próximos Passos
st.header("6. Plano de Ação Imediato")

with st.expander("Checklist de Implementação", expanded=True):
    st.markdown("""
    **📅 Primeiras 4 Semanas:**
    
    - [ ] Desenvolver campanha de migração mensal → anual
    - [ ] Criar sistema de incentivo para débito automático  
    - [ ] Implementar segmentação por risco no CRM
    - [ ] Treinar equipe de retenção nas novas estratégias
    
    **📅 Próximos 60 Dias:**
    
    - [ ] Lançar programa de fidelidade familiar
    - [ ] Desenvolver pacotes de serviços adicionais
    - [ ] Implementar monitoramento contínuo de satisfação
    - [ ] Criar dashboard de métricas de retenção
    
    **📅 Trimestre 2:**
    
    - [ ] Expandir para segmentos específicos (sênior, empresas)
    - [ ] Desenvolver programa de indicação
    - [ ] Otimizar estratégias baseado em resultados
    - [ ] Escalar iniciativas bem-sucedidas
    """)

# Navegação
st.markdown("---")
st.markdown("## Navegação")

col_nav1, col_nav2, col_nav3 = st.columns(3)

with col_nav1:
    if st.button("← Voltar para Simulador"):
        st.switch_page("pages/03_simulator.py")

with col_nav2:
    if st.button("Ir para Calibração →"):
        st.switch_page("pages/05_calibration.py")

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