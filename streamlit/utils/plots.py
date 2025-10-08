import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs.surface.contours import Y
from plotly.subplots import make_subplots
from metrics import get_trend_data, data_raw_treatment, calculate_churn_rate, data_categoric
from sklearn.calibration import calibration_curve

#%% Home

def plot_churn_trend():
    """Cria gráfico de tendência de churn por tempo de permanência"""
    trend_data = get_trend_data()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Gráfico de barras para taxa de churn
    fig.add_trace(
        go.Bar(
            x=trend_data['tenure_group'],
            y=trend_data['churn_rate'],
            name='Taxa de Churn (%)',
            marker_color='indianred',
            opacity=0.7
        ),
        secondary_y=False
    )
    
    # Linha para número total de clientes
    fig.add_trace(
        go.Scatter(
            x=trend_data['tenure_group'],
            y=trend_data['total_customers'],
            name='Total de Clientes',
            line=dict(color='royalblue', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    # Configurações do layout
    fig.update_layout(
        title='Tendência de Churn por Tempo de Permanência',
        xaxis_title='Tempo de Permanência (meses)',
        yaxis_title='Taxa de Churn (%)',
        yaxis2_title='Número de Clientes',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig

#%% Exploratory
def plot_churn_distribution():
    """Cria gráfico de distribuição de churn (básico)"""
    data = data_raw_treatment()
    churn_freq = calculate_churn_rate(data)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Gráfico de barras para taxa de churn
    fig.add_trace(
        go.Bar(
            x=['No', 'Yes'],
            y=churn_freq,
            name='Distribuição de Churn (%)',
            marker_color='indianred',
            opacity=0.7,
            width=0.4
        ),
        secondary_y=False
        )
    # Linha para o Numero Total de Churns
    fig.add_trace(
        go.Scatter(
            x=['No', 'Yes'],
            y=data['Churn'].value_counts(),
            name='Total de Churns',
            # line=dict(color='royalblue', width=3),
            mode='markers'
        ),
        secondary_y=True
    )
    # Configuração do Layout
    fig.update_layout(
        title='Distribuição de Churn',
        xaxis_title='Churn',
        yaxis_title='Taxa de Churn (%)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_numeric_cols_distribution(col):
    """Cria gráfico de distribuição de variáveis numéricas"""
    data = data_raw_treatment()

    fig_box = go.Figure()

    fig_box.add_trace(
        go.Violin(
            y=data.loc[data['Churn']=='Yes', col],
            name='Churn',
            line=dict(color='indianred'),
            box_visible=True,
            meanline_visible=True
        )
    )       

    fig_box.add_trace(
        go.Violin(
            y=data.loc[data['Churn']=='No', col],
            name='No Churn',
            line=dict(color='royalblue'),
            box_visible=True,
            meanline_visible=True
        )
    )       

    fig_box.update_layout(
        title=f'Distribuição de {col} por Churn',
        xaxis_title=col,
        yaxis_title='Densidade',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig_box

def plot_histograms_by_churn(col):
    """Cria gráfico de histograma de variáveis numéricas por churn"""
    data = data_raw_treatment()

    fig_hist = px.histogram(
        data,
        x=col,
        color='Churn',
        barnorm='percent',
        color_discrete_map={'Yes': 'indianred', 'No': 'royalblue'},
        nbins=20,
        opacity=0.5,
        title=f'Histograma de {col} por Churn',
        labels={'Churn': 'Churn', col: col}
    )    

    fig_hist.update_layout(
        title=f'Histograma de {col} por Churn',
        xaxis_title=col,
        yaxis_title='Frequência',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig_hist

def generate_graph_analyses(col):
    """Gera gráfico de barras empilhadas com Plotly"""
    # Seus cálculos originais
    categoric_values = data_categoric()
    count_per_churn = categoric_values[[col, 'Churn']].value_counts(normalize=True).unstack()
    count_per_churn = (count_per_churn * 100).round(2)
    index_cat = [str(i) for i in count_per_churn.index.tolist()]
    
    # Cores
    colors = ['royalblue','indianred']
    
    # Cria figura Plotly
    fig = go.Figure()
    
    for i, (cat, cat_count) in enumerate(count_per_churn.items()):
        fig.add_trace(go.Bar(
            x=index_cat,
            y=cat_count.values,
            name=cat,  # ou use cat se preferir
            marker_color=colors[i],
            opacity=0.8,
            text=cat_count.values,
            textposition='inside',
            texttemplate='%{text:.1f}%'
        ))
    
    # Layout
    fig.update_layout(
        title=f'Proporção de Churn por {col}',
        xaxis_title=col,
        yaxis_title='Proporção (%)',
        barmode='stack',  # Barras empilhadas
        legend=dict(
            title='Churn',
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02
        ),
        hovermode='x unified'
    )
    
    return fig

#%% Predictions

def plot_confusion_matrix(conf_matrix):
    """Plota matriz de confusão"""
    labels = ['Não Churn', 'Churn']
    
    fig = go.Figure(data=go.Heatmap(
        z=conf_matrix,
        x=labels,
        y=labels,
        text=conf_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis_title='Predito',
        yaxis_title='Real',
        width=500,
        height=500,
        yaxis_autorange='reversed'
    )
    
    return fig

def plot_roc_and_pr_curves(metrics):
    """Cria gráfico combinado de curvas ROC e Precision-Recall"""
    fpr, tpr, _ = metrics['roc_curve']
    precision, recall, _ = metrics['pr_curve']
    auc_score = metrics['auc_score']
    avg_precision = metrics['avg_precision']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Curva ROC', 'Curva Precision-Recall')
    )
    
    # Curva ROC
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr, 
            name=f'ROC (AUC={auc_score:.3f})',
            line=dict(color='royalblue', width=2)
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], 
            name='Baseline',
            line=dict(color='gray', dash='dash')
        ),
        row=1, col=1
    )
    
    # Curva Precision-Recall
    fig.add_trace(
        go.Scatter(
            x=recall, y=precision, 
            name=f'PR (AP={avg_precision:.3f})',
            line=dict(color='indianred', width=2)
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)
    return fig

def plot_probability_distribution_by_class(y_test, y_pred_proba):
    """Plota distribuição de probabilidades separada por classe real"""
    y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
    
    df_probs = pd.DataFrame({
        'probability': y_pred_proba,
        'actual_churn': y_test_array
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_probs[df_probs['actual_churn'] == 0]['probability'],
        name='Não Churn (Real)',
        marker_color='royalblue',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=df_probs[df_probs['actual_churn'] == 1]['probability'],
        name='Churn (Real)',
        marker_color='indianred',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_vline(
        x=0.5, line_dash="dash", line_color="black",
        annotation_text="Threshold = 0.5"
    )
    
    fig.update_layout(
        title='Distribuição de Probabilidades por Classe Real',
        xaxis_title='Probabilidade de Churn',
        yaxis_title='Frequência',
        barmode='overlay',
        height=400
    )
    return fig

def plot_risk_segments(risk_summary):
    """Plota distribuição de clientes por segmento de risco"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_summary.index,
        y=risk_summary['Total Clientes'],
        name='Total Clientes',
        marker_color='lightblue',
        text=risk_summary['Total Clientes'].astype(int),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=risk_summary.index,
        y=risk_summary['Churns Reais'],
        name='Churns Reais',
        marker_color='indianred',
        text=risk_summary['Churns Reais'].astype(int),
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Distribuição de Clientes por Segmento de Risco',
        xaxis_title='Segmento',
        yaxis_title='Número de Clientes',
        barmode='group',
        height=400
    )
    return fig

def plot_metrics_comparison_bar(class_report):
    """Plota comparação de métricas por classe"""
    metrics_data = {
        'Precision': [class_report['0']['precision'], class_report['1']['precision']],
        'Recall': [class_report['0']['recall'], class_report['1']['recall']],
        'F1-Score': [class_report['0']['f1-score'], class_report['1']['f1-score']]
    }
    
    fig = go.Figure()
    classes = ['Não Churn', 'Churn']
    colors = ['royalblue', 'indianred']
    
    for i, cls in enumerate(classes):
        fig.add_trace(go.Bar(
            name=cls,
            x=list(metrics_data.keys()),
            y=[metrics_data[metric][i] for metric in metrics_data.keys()],
            marker_color=colors[i],
            text=[f"{metrics_data[metric][i]:.3f}" for metric in metrics_data.keys()],
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Comparação de Métricas por Classe',
        xaxis_title='Métrica',
        yaxis_title='Valor',
        barmode='group',
        height=400,
        yaxis_range=[0, 1.1]
    )
    return fig

#%% SHAP Analysis
def plot_shap_summary(shap_values, feature_names):
    """Cria gráfico de resumo SHAP (versão Plotly do summary plot)"""
    # Calcula importância média absoluta
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance_sorted = np.argsort(feature_importance)
    
    # Cria figura
    fig = go.Figure()
    
    # Adiciona scatter plot para cada feature
    for idx in feature_importance_sorted:
        fig.add_trace(go.Scatter(
            x=shap_values[:, idx],
            y=[feature_names[idx]] * len(shap_values),
            mode='markers',
            name=feature_names[idx],
            marker=dict(
                size=8,
                opacity=0.6,
                colorscale='RdBu',
                color=shap_values[:, idx],
                showscale=True,
                colorbar=dict(title='SHAP value')
            )
        ))
    
    fig.update_layout(
        title='SHAP Summary Plot - Impacto das Features',
        xaxis_title='SHAP value',
        yaxis_title='Features',
        showlegend=False,
        height=800
    )
    
    return fig

def plot_shap_importance(shap_values, feature_names):
    """Cria gráfico de barras de importância SHAP"""
    # Calcula importância média absoluta
    feature_importance = np.abs(shap_values).mean(0)
    sorted_idx = np.argsort(feature_importance)
    
    fig = go.Figure(go.Bar(
        x=feature_importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h'
    ))
    
    fig.update_layout(
        title='SHAP Feature Importance',
        xaxis_title='Importância Média (|SHAP value|)',
        yaxis_title='Features',
        height=600
    )
    
    return fig

def plot_shap_waterfall(shap_values, expected_value, feature_names, sample_idx=0):
    """Cria gráfico waterfall para explicação individual"""
    # Pega valores SHAP para um cliente específico
    values = shap_values[sample_idx]
    
    # Ordena por magnitude
    sorted_idx = np.argsort(np.abs(values))
    
    # Calcula valores cumulativos
    cumulative = np.cumsum(values[sorted_idx])
    total = cumulative[-1] + expected_value
    
    # Cria figura
    fig = go.Figure()
    
    # Adiciona barra para valor base (expected_value)
    fig.add_trace(go.Waterfall(
        name='expected_value',
        orientation='h',
        measure=['relative'],
        x=[expected_value],
        textposition='outside',
        text=[f'Base Value<br>{expected_value:.3f}'],
        y=['base'],
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': 'rgb(31, 119, 180)'}},
        increasing={'marker': {'color': 'rgb(31, 119, 180)'}}
    ))
    
    # Adiciona contribuições individuais
    for idx in sorted_idx:
        fig.add_trace(go.Waterfall(
            orientation='h',
            measure=['relative'],
            x=[values[idx]],
            textposition='outside',
            text=[f'{feature_names[idx]}<br>{values[idx]:.3f}'],
            y=[feature_names[idx]],
            connector={'line': {'color': 'rgb(63, 63, 63)'}},
            decreasing={'marker': {'color': 'indianred'}},
            increasing={'marker': {'color': 'royalblue'}}
        ))
    
    # Adiciona valor final
    fig.add_trace(go.Waterfall(
        orientation='h',
        measure=['total'],
        x=[total],
        textposition='outside',
        text=[f'Final Value<br>{total:.3f}'],
        y=['final'],
        connector={'line': {'color': 'rgb(63, 63, 63)'}},
        decreasing={'marker': {'color': 'rgb(214, 39, 40)'}},
        increasing={'marker': {'color': 'rgb(44, 160, 44)'}}
    ))
    
    fig.update_layout(
        title='SHAP Waterfall - Explicação Individual',
        showlegend=False,
        height=800,
        yaxis={'categoryorder': 'array', 'categoryarray': ['base'] + [feature_names[i] for i in sorted_idx] + ['final']}
    )
    
    return fig

def plot_shap_dependence(shap_values, feature_values, feature_names, feature_idx):
    """Cria gráfico de dependência SHAP para uma feature específica"""
    fig = go.Figure()

    # Pega os valores da feature
    feature_data = feature_values[:, feature_idx]

    # Converte para numérico se for categórico
    if not np.issubdtype(feature_data.dtype, np.number):
        # É categórico - converte para códigos
        feature_series = pd.Series(feature_data)
        color_values = feature_series.astype('category').cat.codes
        
        # Cria hover text com valores originais
        hover_text = [f"{feature_names[feature_idx]}: {val}" for val in feature_data]
    else:
        # Já é numérico
        color_values = feature_data
        hover_text = None

    fig.add_trace(go.Scatter(
        x=feature_data,
        y=shap_values[:, feature_idx],
        mode='markers',
        text=hover_text,
        marker=dict(
            size=8,
            opacity=0.6,
            color=color_values,
            colorscale='RdBu',
            showscale=True,
            colorbar=dict(title=feature_names[feature_idx])
        )
    ))
    
    fig.update_layout(
        title=f'SHAP Dependence Plot - {feature_names[feature_idx]}',
        xaxis_title=f'Feature Value ({feature_names[feature_idx]})',
        yaxis_title='SHAP value',
        showlegend=False
    )
    
    return fig

def plot_shap_decision(shap_values, expected_value, feature_names, n_samples=20):
    """Cria gráfico de decisão SHAP para múltiplos clientes"""
    # Seleciona amostras
    values = shap_values[:n_samples]
    
    # Calcula valores cumulativos
    cumsum = np.zeros((values.shape[0], values.shape[1] + 1))
    cumsum[:, 1:] = np.cumsum(values, axis=1)
    cumsum[:, 0] = expected_value
    
    # Cria figura
    fig = go.Figure()
    
    # Adiciona linhas para cada amostra
    for i in range(values.shape[0]):
        fig.add_trace(go.Scatter(
            x=list(range(values.shape[1] + 1)),
            y=cumsum[i],
            mode='lines+markers',
            name=f'Cliente {i+1}',
            marker=dict(size=8),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='SHAP Decision Plot - Comparação de Clientes',
        xaxis_title='Features',
        yaxis_title='Predição + SHAP values',
        xaxis=dict(
            ticktext=['Base'] + list(feature_names),
            tickvals=list(range(values.shape[1] + 1))
        ),
        height=600,
        showlegend=True
    )
    
    return fig

#%% Calibrations
def plot_calibration_curve(prob_pred_raw, prob_true_raw, y_proba_cal=None, y_test=None):
    fig = go.Figure()
    
    # Linha de calibração perfeita
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Calibração Perfeita',
        line=dict(dash='dash', color='gray', width=2)
    ))
    
    # Sem calibração
    fig.add_trace(go.Scatter(
        x=prob_pred_raw,
        y=prob_true_raw,
        mode='lines+markers',
        name='Sem Calibração',
        line=dict(color='indianred', width=3),
        marker=dict(size=10)
    ))
    
    # Com calibração (se disponível)
    if y_proba_cal is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, y_proba_cal, n_bins=10)
        fig.add_trace(go.Scatter(
            x=prob_pred_cal,
            y=prob_true_cal,
            mode='lines+markers',
            name='Com Calibração (Venn-ABERS)',
            line=dict(color='royalblue', width=3),
            marker=dict(size=10)
        ))
    
    fig.update_layout(
        title='Curva de Calibração - Comparação',
        xaxis_title='Probabilidade Predita',
        yaxis_title='Fração Real de Churns',
        hovermode='x unified',
        height=500
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    return fig

def plot_threshold_evaluation(thresholds, precisions, recalls, f1_scores, f2_scores, best_threshold):

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Trade-off Precision vs Recall', 'F-Scores por Threshold')
    )
    
    # Gráfico 1: Precision/Recall
    fig.add_trace(
        go.Scatter(x=thresholds, y=precisions, name='Precision', line=dict(color='indianred', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=recalls, name='Recall', line=dict(color='royalblue', width=2)),
        row=1, col=1
    )
    
    # Gráfico 2: F-Scores
    fig.add_trace(
        go.Scatter(x=thresholds, y=f1_scores, name='F1-Score', line=dict(color='green', width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=thresholds, y=f2_scores, name='F2-Score (prioriza Recall)', line=dict(color='purple', width=2)),
        row=1, col=2
    )

    # Marca threshold ótimo
    fig.add_vline(x=best_threshold, line_dash="dash", line_color="black", annotation_text=f"Ótimo: {best_threshold:.2f}", row=1, col=2)
    
    fig.update_xaxes(title_text="Threshold", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="Threshold", row=1, col=2)
    fig.update_yaxes(title_text="F-Score", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=True)

    return fig