"""
Etapa 5, regressão: número de atropelamentos n por segmento × mês.
Referências (média e histórico do segmento), linear, Poisson, random forest e
HGB com perda Poisson nos três splits, ablação, importância por permuta e
gráficos de diagnóstico.
"""
import importlib

import pandas as pd

from src import config
from src.evaluation import (
    MAIN_METRIC, ablation_table, plot_ablation, plot_importance, plot_regression_diagnostics,
    regression_metrics, save_results, summarize,
)
from src.models import SplitContext, TREE_MODELS, classical_models, evaluate_models, importance_by_permutation

PROBLEMA = 'regressao'
DATASET = 'segmentos_periodo'
segment_history_baseline = importlib.import_module('experiments.05_binario').segment_history_baseline


def run():
    config.MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SEGMENTOS_CSV)
    y = df['n'].astype(float)
    ctx = SplitContext()
    print(f"{len(df)} linhas; n médio {y.mean():.3f}, variância {y.var():.3f}")

    print("\n1. TREINO E AVALIAÇÃO")
    rows, preds = evaluate_models(
        df, y, PROBLEMA, DATASET, classical_models(PROBLEMA), list(config.ABLATION_SETS), config.SPLITS, ctx,
        metric_fn=lambda yt, yp, pr: regression_metrics(yt, yp), keep_predictions=('temporal', 'teste'))
    rows += segment_history_baseline(df, y, ctx, PROBLEMA)
    results = save_results(rows, PROBLEMA)
    for metrica in ['desvio_poisson', 'rmse', 'mae', 'r2']:
        resumo = summarize(results, PROBLEMA, metrica)
        resumo.to_csv(config.TABLES_DIR / f'resumo_{PROBLEMA}_{metrica}.csv')
        if metrica in ('desvio_poisson', 'rmse'):
            print(f"\n{metrica} (média ± desvio entre folds):\n{resumo.to_string()}")

    ganho = ablation_table(results, PROBLEMA, MAIN_METRIC[PROBLEMA])
    ganho = ganho[~ganho.index.get_level_values('modelo').isin(['media', 'historico_segmento'])]
    ganho.to_csv(config.TABLES_DIR / f'ablacao_{PROBLEMA}.csv')
    plot_ablation(ganho, 'Regressão: redução do desvio de Poisson sobre o conjunto base', config.MODEL_FIG_DIR / f'{PROBLEMA}_ablacao.png')
    print(f"\nRedução do desvio de Poisson sobre o base (positivo = melhor):\n{ganho.to_string()}")

    teste = results[(results['split'] == 'temporal') & (results['conjunto_atributos'] == 'completo')
                    & (results['metrica'] == MAIN_METRIC[PROBLEMA]) & results['modelo'].isin(preds)].set_index('modelo')['valor']
    melhor = teste.idxmin()
    melhor_arvore = teste[teste.index.isin(TREE_MODELS)].idxmin()
    print(f"\nMelhor no teste temporal (completo): {melhor} (desvio {teste[melhor]:.3f}); melhor de árvores: {melhor_arvore}")
    p = preds[melhor]
    plot_regression_diagnostics(y[p['mask']], p['y_pred'], f'{melhor} (teste temporal)', config.MODEL_FIG_DIR / f'{PROBLEMA}_diagnostico.png')
    p = preds[melhor_arvore]
    imp = importance_by_permutation(p['pipe'], df.loc[p['mask'], p['features']], y[p['mask']], scoring='neg_mean_poisson_deviance')
    imp.to_csv(config.TABLES_DIR / f'importancia_{PROBLEMA}.csv', index=False)
    plot_importance(imp, f'Regressão: importância por permuta ({melhor_arvore}, teste temporal, desvio de Poisson)',
                    config.MODEL_FIG_DIR / f'{PROBLEMA}_importancia.png')
    print(f"\nImportância por permuta ({melhor_arvore}):\n{imp.head(10).to_string(index=False)}")
    print(f"\n[OK] resultados em {config.RESULTS_CSV.relative_to(config.ROOT)}")


if __name__ == '__main__':
    run()
