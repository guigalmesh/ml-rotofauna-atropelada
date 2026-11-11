"""
Etapa 5, problema binário: hotspot (n >= HOTSPOT_MIN_N) por segmento × mês.
Referências (prevalência e histórico do segmento), logística, random forest e
HGB nos três splits, ablação, importância por permuta e sensibilidade ao
critério n >= 2.
"""
import numpy as np
import pandas as pd

from src import config
from src.evaluation import (
    MAIN_METRIC, ablation_table, classification_metrics, plot_ablation, plot_confusion,
    plot_importance, save_results, summarize,
)
from src.models import SplitContext, TREE_MODELS, classical_models, evaluate_models, importance_by_permutation, iter_splits

PROBLEMA = 'binario'
DATASET = 'segmentos_periodo'


def segment_history_baseline(df, y, ctx, problema):
    """
    Referência 'histórico do segmento': a fração positiva (ou média de n) do
    segmento no treino da rodada; segmento nunca visto (split espacial) recebe
    a média global do treino. Não usa atributos: mede o quanto só a identidade
    do segmento explica.
    """
    rows = []
    for split in config.SPLITS:
        for rodada, m_tr, m_val in iter_splits(df, split, ctx):
            media_seg = y[m_tr].groupby(df.loc[m_tr, 'segmento_id']).mean()
            proba = df.loc[m_val, 'segmento_id'].map(media_seg).fillna(y[m_tr].mean()).values
            if problema == 'binario':
                metricas = classification_metrics(y[m_val], (proba >= 0.5).astype(int), proba)
            else:
                from src.evaluation import regression_metrics
                metricas = regression_metrics(y[m_val], proba)
            for metrica, valor in metricas.items():
                rows.append({'problema': problema, 'modelo': 'historico_segmento', 'conjunto_atributos': 'completo',
                             'split': split, 'fold': rodada, 'metrica': metrica, 'valor': float(valor), 'tempo_s': 0.0})
    return rows


def run():
    config.MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SEGMENTOS_CSV)
    y = (df['n'] >= config.HOTSPOT_MIN_N).astype(int)
    ctx = SplitContext()
    print(f"{len(df)} linhas; positivos {y.mean() * 100:.1f}%")

    print("\n1. TREINO E AVALIAÇÃO")
    rows, preds = evaluate_models(
        df, y, PROBLEMA, DATASET, classical_models(PROBLEMA), list(config.ABLATION_SETS), config.SPLITS, ctx,
        metric_fn=classification_metrics, keep_predictions=('temporal', 'teste'))
    rows += segment_history_baseline(df, y, ctx, PROBLEMA)
    results = save_results(rows, PROBLEMA)
    resumo = summarize(results, PROBLEMA, MAIN_METRIC[PROBLEMA])
    resumo.to_csv(config.TABLES_DIR / f'resumo_{PROBLEMA}.csv')
    print(f"\nROC-AUC (média ± desvio entre folds):\n{resumo.to_string()}")
    resumo_f1 = summarize(results, PROBLEMA, 'f1_positivo')
    print(f"\nF1 da classe positiva:\n{resumo_f1[resumo_f1.index.get_level_values(0) == 'completo'].to_string()}")

    ganho = ablation_table(results, PROBLEMA, MAIN_METRIC[PROBLEMA])
    ganho = ganho[~ganho.index.get_level_values('modelo').isin(['prevalencia', 'historico_segmento'])]
    ganho.to_csv(config.TABLES_DIR / f'ablacao_{PROBLEMA}.csv')
    plot_ablation(ganho, 'Hotspot: ganho de ROC-AUC sobre o conjunto base', config.MODEL_FIG_DIR / f'{PROBLEMA}_ablacao.png')
    print(f"\nGanho de ROC-AUC sobre o base:\n{ganho.to_string()}")

    teste = results[(results['split'] == 'temporal') & (results['conjunto_atributos'] == 'completo')
                    & (results['metrica'] == MAIN_METRIC[PROBLEMA]) & results['modelo'].isin(preds)].set_index('modelo')['valor']
    melhor = teste.idxmax()
    melhor_arvore = teste[teste.index.isin(TREE_MODELS)].idxmax()
    print(f"\nMelhor no teste temporal (completo): {melhor} (ROC-AUC {teste[melhor]:.3f}); melhor de árvores: {melhor_arvore}")
    p = preds[melhor]
    plot_confusion(y[p['mask']], p['y_pred'], [0, 1], f'{melhor} (teste temporal, limiar 0,5)', config.MODEL_FIG_DIR / f'{PROBLEMA}_confusao.png')
    p = preds[melhor_arvore]
    imp = importance_by_permutation(p['pipe'], df.loc[p['mask'], p['features']], y[p['mask']], scoring='roc_auc')
    imp.to_csv(config.TABLES_DIR / f'importancia_{PROBLEMA}.csv', index=False)
    plot_importance(imp, f'Hotspot: importância por permuta ({melhor_arvore}, teste temporal, ROC-AUC)',
                    config.MODEL_FIG_DIR / f'{PROBLEMA}_importancia.png')
    print(f"\nImportância por permuta ({melhor_arvore}):\n{imp.head(10).to_string(index=False)}")

    # =========================================================================
    # SENSIBILIDADE: n >= 2 ("mês intenso"), só logística e HGB no conjunto completo
    # =========================================================================
    print(f"\n2. SENSIBILIDADE: hotspot = n >= {config.HOTSPOT_MIN_N_SENSITIVITY}")
    y2 = (df['n'] >= config.HOTSPOT_MIN_N_SENSITIVITY).astype(int)
    modelos = {k: v for k, v in classical_models(PROBLEMA).items() if k in ('logistica', 'hgb')}
    rows2, _ = evaluate_models(df, y2, f'{PROBLEMA}_n{config.HOTSPOT_MIN_N_SENSITIVITY}', DATASET, modelos, ['completo'],
                               config.SPLITS, ctx, metric_fn=classification_metrics)
    results = save_results(rows2, f'{PROBLEMA}_n{config.HOTSPOT_MIN_N_SENSITIVITY}')
    print(summarize(results, f'{PROBLEMA}_n{config.HOTSPOT_MIN_N_SENSITIVITY}', 'roc_auc').to_string())
    print(f"\n[OK] resultados em {config.RESULTS_CSV.relative_to(config.ROOT)}")


if __name__ == '__main__':
    run()
