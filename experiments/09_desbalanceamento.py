"""
Impacto do desbalanceamento das classes (pergunta 4 da discussão final):
cada modelo de classificação é treinado sem e com pesos de classe, no
conjunto completo e no teste temporal. Compara accuracy (que premia a classe
majoritária) com accuracy balanceada e F1 macro; no hotspot, também precisão
e recall da classe positiva. Os clássicos usam class_weight; a MLP, os pesos
na função de perda (média de 5 sementes). Não grava em resultados.csv.
"""
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.evaluation import classification_metrics
from src.mlp import evaluate_mlp
from src.models import SplitContext, classical_models, evaluate_models

optuna_script = importlib.import_module('experiments.07_optuna')
load_problem, DATASETS = optuna_script.load_problem, optuna_script.DATASETS
CLASSICOS = ['logistica', 'random_forest', 'hgb']
METRICAS = {'multiclasse': ['accuracy', 'balanced_accuracy', 'f1_macro'],
            'binario': ['accuracy', 'balanced_accuracy', 'f1_macro', 'precision_positivo', 'recall_positivo', 'roc_auc']}


def run_problem(problema: str, ctx: SplitContext) -> pd.DataFrame:
    df, y, classes = load_problem(problema)
    y_serie = pd.Series(y, index=df.index)
    metric_fn = classification_metrics if problema == 'binario' else (lambda yt, yp, pr: classification_metrics(yt, yp))
    rows = []
    for pesos, class_weight in [('sem pesos', None), ('com pesos', 'balanced')]:
        modelos = {k: v.set_params(class_weight=class_weight) for k, v in classical_models(problema).items() if k in CLASSICOS}
        r, _ = evaluate_models(df, y_serie, problema, DATASETS[problema], modelos, ['completo'], ['temporal'], ctx, metric_fn=metric_fn)
        r_mlp, _, _ = evaluate_mlp(df, y, problema, problema, DATASETS[problema], ['completo'], ['temporal'], ctx,
                                   params={'balanced': class_weight is not None}, classes=classes)
        rows += [{**linha, 'pesos': pesos} for linha in r + r_mlp]
    t = pd.DataFrame(rows)
    return t[t['metrica'].isin(METRICAS[problema])].groupby(['modelo', 'pesos', 'metrica'])['valor'].mean().unstack('metrica')[METRICAS[problema]]


def plot(tabela: pd.DataFrame, problema: str, path):
    metricas = METRICAS[problema][:3] + (['recall_positivo'] if problema == 'binario' else [])
    modelos = CLASSICOS + ['mlp']
    fig, axes = plt.subplots(1, len(metricas), figsize=(4.2 * len(metricas), 4.3), sharey=True)
    for ax, m in zip(axes, metricas):
        for k, (pesos, cor) in enumerate([('sem pesos', 'lightgray'), ('com pesos', 'steelblue')]):
            ax.bar(np.arange(len(modelos)) + (k - 0.5) * 0.4, [tabela.loc[(mod, pesos), m] for mod in modelos], 0.4, color=cor, label=pesos)
        ax.set_xticks(range(len(modelos)), modelos, rotation=20)
        ax.set_title(m)
    axes[0].legend()
    fig.suptitle(f'Desbalanceamento, {problema}: teste temporal sem e com pesos de classe')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


if __name__ == '__main__':
    torch.set_num_threads(config.TORCH_THREADS)
    config.MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ctx = SplitContext()
    for problema in ['multiclasse', 'binario']:
        tabela = run_problem(problema, ctx)
        tabela.round(4).to_csv(config.TABLES_DIR / f'desbalanceamento_{problema}.csv')
        plot(tabela, problema, config.MODEL_FIG_DIR / f'{problema}_desbalanceamento.png')
        print(f"\n{problema}: teste temporal, conjunto completo\n{tabela.round(3).to_string()}")
    print("\n[OK] desbalanceamento_<problema>.csv e figuras <problema>_desbalanceamento.png")
