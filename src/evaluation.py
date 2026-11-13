"""
Métricas, tabela única de resultados e figuras de avaliação (Etapa 5 em diante).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_poisson_deviance, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)

from src import config

# Métrica principal de cada problema (a que Optuna otimiza e as tabelas resumem)
MAIN_METRIC = {'multiclasse': 'f1_macro', 'binario': 'roc_auc', 'regressao': 'desvio_poisson'}
HIGHER_IS_BETTER = {'f1_macro': True, 'roc_auc': True, 'desvio_poisson': False}


# =============================================================================
# MÉTRICAS
# =============================================================================
def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Métricas globais de classificação; roc_auc só no binário (proba da classe 1)."""
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        out['roc_auc'] = roc_auc_score(y_true, y_proba)
        out['f1_positivo'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return out


def per_class_metrics(y_true, y_pred, labels) -> pd.DataFrame:
    return pd.DataFrame({
        'classe': labels,
        'precision': precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        'recall': recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        'f1': f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
        'suporte': [int((np.asarray(y_true) == c).sum()) for c in labels],
    })


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE, R² e desvio de Poisson (previsões negativas são truncadas para o desvio)."""
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': r2_score(y_true, y_pred),
        'desvio_poisson': mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None)),
    }


# =============================================================================
# TABELA ÚNICA DE RESULTADOS
# =============================================================================
def save_results(rows: list[dict], problema: str) -> pd.DataFrame:
    """
    Grava as linhas em outputs/tables/resultados.csv (formato longo: problema,
    modelo, conjunto_atributos, split, fold, metrica, valor, tempo_s e, na MLP,
    semente e epocas), substituindo só as linhas anteriores dos mesmos modelos
    no mesmo problema. Assim clássicos e MLP podem ser rodados em separado.
    """
    novo = pd.DataFrame(rows)
    if config.RESULTS_CSV.exists():
        antigo = pd.read_csv(config.RESULTS_CSV)
        substituir = (antigo['problema'] == problema) & antigo['modelo'].isin(novo['modelo'].unique())
        novo = pd.concat([antigo[~substituir], novo], ignore_index=True)
    novo.to_csv(config.RESULTS_CSV, index=False)
    return novo


def summarize(results: pd.DataFrame, problema: str, metrica: str) -> pd.DataFrame:
    """Média ± desvio da métrica por modelo × conjunto × split (desvio entre folds)."""
    r = results[(results['problema'] == problema) & (results['metrica'] == metrica)]
    agg = r.groupby(['conjunto_atributos', 'modelo', 'split'])['valor'].agg(['mean', 'std']).reset_index()
    agg['valor'] = agg.apply(lambda l: f"{l['mean']:.3f}" + (f" ± {l['std']:.3f}" if pd.notna(l['std']) else ''), axis=1)
    return agg.pivot(index=['conjunto_atributos', 'modelo'], columns='split', values='valor').reindex(columns=config.SPLITS)


def ablation_table(results: pd.DataFrame, problema: str, metrica: str) -> pd.DataFrame:
    """Ganho médio de cada conjunto de atributos sobre o 'base', por modelo × split (só os clássicos)."""
    r = results[(results['problema'] == problema) & (results['metrica'] == metrica)
                & ~results['modelo'].str.startswith('mlp')]
    medias = r.groupby(['modelo', 'split', 'conjunto_atributos'])['valor'].mean().unstack('conjunto_atributos')
    base = medias['base']
    ganho = medias.sub(base, axis=0)
    if not HIGHER_IS_BETTER[metrica]:
        ganho = -ganho
    return ganho.drop(columns='base').round(4)


def comparison_table(results: pd.DataFrame, problema: str, metrica: str) -> pd.DataFrame:
    """
    Clássicos × MLP no conjunto completo: média ± desvio da métrica principal
    em cada split (desvio entre folds e, na MLP, também entre sementes) e o
    tempo médio de treino por ajuste no split temporal.
    """
    r = results[(results['problema'] == problema) & (results['conjunto_atributos'] == 'completo')]
    m = r[r['metrica'] == metrica]
    agg = m.groupby(['modelo', 'split'])['valor'].agg(['mean', 'std']).reset_index()
    agg['valor'] = agg.apply(lambda l: f"{l['mean']:.3f}" + (f" ± {l['std']:.3f}" if pd.notna(l['std']) else ''), axis=1)
    tabela = agg.pivot(index='modelo', columns='split', values='valor').reindex(columns=config.SPLITS)
    tabela['tempo_treino_s'] = m[m['split'] == 'temporal'].groupby('modelo')['tempo_s'].mean().round(2)
    ordem = m[m['split'] == 'temporal'].groupby('modelo')['valor'].mean().sort_values(ascending=not HIGHER_IS_BETTER[metrica])
    return tabela.reindex(ordem.index)


# =============================================================================
# FIGURAS
# =============================================================================
def plot_learning_curves(curvas: pd.DataFrame, metrica: str, titulo: str, path):
    """Loss e métrica principal por época, treino × validação interna, uma linha por semente."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for seed, h in curvas.groupby('semente'):
        for ax, base in zip(axes, ['loss', 'metrica']):
            ax.plot(h['epoca'], h[f'{base}_treino'], color='steelblue', alpha=0.6, lw=1)
            ax.plot(h['epoca'], h[f'{base}_val'], color='darkorange', alpha=0.8, lw=1)
        axes[0].axvline(h.loc[h['loss_val'].idxmin(), 'epoca'], color='gray', ls=':', lw=0.8)
    axes[0].set_title('Loss (linha pontilhada: melhor época de cada semente)')
    axes[0].set_ylabel('loss')
    axes[1].set_title(f'{metrica}')
    axes[1].set_ylabel(metrica)
    for ax in axes:
        ax.set_xlabel('época')
        ax.plot([], [], color='steelblue', label='treino')
        ax.plot([], [], color='darkorange', label='validação interna')
        ax.legend()
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_comparison(results: pd.DataFrame, problema: str, metrica: str, path):
    """Métrica principal por modelo e split no conjunto completo (barra = média, erro = desvio)."""
    r = results[(results['problema'] == problema) & (results['conjunto_atributos'] == 'completo') & (results['metrica'] == metrica)]
    ordem = r[r['split'] == 'temporal'].groupby('modelo')['valor'].mean().sort_values(ascending=not HIGHER_IS_BETTER[metrica]).index
    plt.figure(figsize=(11, 4.5))
    sns.barplot(data=r, x='modelo', y='valor', hue='split', order=ordem, hue_order=config.SPLITS, errorbar='sd')
    plt.title(f"{problema}: {metrica} no conjunto completo ({'maior' if HIGHER_IS_BETTER[metrica] else 'menor'} é melhor)")
    plt.xlabel('')
    plt.ylabel(metrica)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion(y_true, y_pred, labels, titulo, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f'{titulo}: contagens')
    sns.heatmap(cm_pct, annot=True, fmt='.0f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title('% da classe real (recall na diagonal)')
    for ax in axes:
        ax.set_xlabel('previsto')
        ax.set_ylabel('real')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_regression_diagnostics(y_true, y_pred, titulo, path):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].scatter(y_pred, y_true + np.random.default_rng(config.SEED).uniform(-0.15, 0.15, len(y_true)), s=6, alpha=0.3)
    lim = max(y_true.max(), y_pred.max()) + 0.5
    axes[0].plot([0, lim], [0, lim], 'r--', lw=1)
    axes[0].set_title(f'{titulo}: real × previsto (real com jitter)')
    axes[0].set_xlabel('previsto')
    axes[0].set_ylabel('real (n)')
    res = y_true - y_pred
    axes[1].scatter(y_pred, res, s=6, alpha=0.3)
    axes[1].axhline(0, color='r', ls='--', lw=1)
    axes[1].set_title('Resíduos × previsto')
    axes[1].set_xlabel('previsto')
    axes[1].set_ylabel('real − previsto')
    # Calibração por faixa de previsão: média real × média prevista
    faixas = pd.qcut(y_pred, 10, duplicates='drop')
    cal = pd.DataFrame({'real': y_true, 'prev': y_pred}).groupby(faixas, observed=True).mean()
    axes[2].plot(cal['prev'], cal['real'], 'o-')
    axes[2].plot([0, cal.max().max()], [0, cal.max().max()], 'r--', lw=1)
    axes[2].set_title('Calibração por decil de previsão')
    axes[2].set_xlabel('média prevista')
    axes[2].set_ylabel('média real')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_importance(importancia: pd.DataFrame, titulo, path, top=20):
    imp = importancia.sort_values('importancia_media', ascending=True).tail(top)
    plt.figure(figsize=(8, 0.35 * len(imp) + 1.5))
    plt.barh(imp['atributo'], imp['importancia_media'], xerr=imp['importancia_desvio'], color='steelblue')
    plt.axvline(0, color='gray', lw=0.5)
    plt.title(titulo)
    plt.xlabel('queda da métrica ao embaralhar o atributo')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_ablation(ganho: pd.DataFrame, titulo, path):
    """Ganho de cada conjunto sobre o base, por modelo (barras) e split (painéis)."""
    g = ganho.reset_index().melt(id_vars=['modelo', 'split'], var_name='conjunto', value_name='ganho')
    splits = [s for s in config.SPLITS if s in g['split'].unique()]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.5 * len(splits), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, splits):
        sns.barplot(data=g[g['split'] == s], x='conjunto', y='ganho', hue='modelo', ax=ax)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(f'split {s}')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('ganho sobre base')
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
