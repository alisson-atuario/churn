import streamlit as st

st.set_page_config(
    page_title="Dashboard de Churn",
    page_icon="📊", 
    layout="wide"
)

pg = st.navigation([
    st.Page("pages/00_home.py", title="Home", icon=":material/home:"),
    st.Page("pages/01_exploratory.py", title="Análise Exploratória", icon=":material/search:"),
    st.Page("pages/02_predictions.py", title="Previsões", icon=":material/target:"),
    st.Page("pages/03_simulator.py", title="Simulador", icon=":material/lightbulb_2:"),
    st.Page("pages/04_insights.py", title="Insights", icon=":material/monitoring:"),
    st.Page("pages/05_calibration.py", title="Calibração", icon=":material/tactic:"),
    st.Page("pages/06_technical.py", title="Técnico", icon=":material/analytics:"),
    st.Page("pages/07_settings.py", title="Configurações", icon=":material/settings:"),
])

pg.run()