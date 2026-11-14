"""
Etapas 5 e 6, problema multiclasse: classe taxonômica de cada ocorrência.
Referências ingênuas, regressão logística, random forest e HGB, nos três
splits, com ablação dos dados externos e importância por permuta; e a MLP
(--modelos mlp), comparada com os clássicos.
"""
import argparse

import joblib
import pandas as pd

from src import config
from src.evaluation import (
    MAIN_METRIC, ablation_table, classification_metrics, per_class_metrics, plot_ablation,
    plot_confusion, plot_importance, save_results, summarize,
)
from src.mlp import run_mlp_section
from src.models import SplitContext, TREE_MODELS, classical_models, evaluate_models, importance_by_permutation

PROBLEMA = 'multiclasse'
DATASET = 'ocorrencias'
CLASSES = ['birds', 'reptiles', 'mammals', 'amphibians']


def run_classicos(df, y, ctx):
    # =========================================================================
    # 1. MODELOS × CONJUNTOS × SPLITS
    # =========================================================================
    print("\n1. TREINO E AVALIAÇÃO")
    rows, preds = evaluate_models(
        df, y, PROBLEMA, DATASET, classical_models(PROBLEMA), list(config.ABLATION_SETS), config.SPLITS, ctx,
        metric_fn=lambda yt, yp, pr: classification_metrics(yt, yp), keep_predictions=('temporal', 'teste'))
    results = save_results(rows, PROBLEMA)
    # Só os clássicos: a tabela não muda se a MLP já estiver em resultados.csv (a MLP entra no comparativo)
    classicos = results[~results['modelo'].str.startswith('mlp')]
    resumo = summarize(classicos, PROBLEMA, MAIN_METRIC[PROBLEMA])
    resumo.to_csv(config.TABLES_DIR / f'resumo_{PROBLEMA}.csv')
    print(f"\nF1 macro (média ± desvio entre folds):\n{resumo.to_string()}")

    # =========================================================================
    # 2. ABLAÇÃO DOS DADOS EXTERNOS
    # =========================================================================
    ganho = ablation_table(results, PROBLEMA, MAIN_METRIC[PROBLEMA])
    ganho = ganho[~ganho.index.get_level_values('modelo').isin(['majoritaria', 'estratificada'])]
    ganho.to_csv(config.TABLES_DIR / f'ablacao_{PROBLEMA}.csv')
    plot_ablation(ganho, 'Multiclasse: ganho de F1 macro sobre o conjunto base', config.MODEL_FIG_DIR / f'{PROBLEMA}_ablacao.png')
    print(f"\nGanho de F1 macro sobre o base:\n{ganho.to_string()}")

    # =========================================================================
    # 3. MELHOR MODELO NO TESTE TEMPORAL: MATRIZ DE CONFUSÃO, POR CLASSE, IMPORTÂNCIA
    # =========================================================================
    teste = results[(results['problema'] == PROBLEMA) & (results['split'] == 'temporal') & (results['conjunto_atributos'] == 'completo')
                    & (results['metrica'] == MAIN_METRIC[PROBLEMA]) & results['modelo'].isin(preds)].set_index('modelo')['valor']
    melhor = teste.idxmax()
    melhor_arvore = teste[teste.index.isin(TREE_MODELS)].idxmax()
    print(f"\nMelhor no teste temporal (completo): {melhor} (F1 macro {teste[melhor]:.3f}); melhor de árvores: {melhor_arvore}")
    p = preds[melhor]
    y_teste = y[p['mask']]
    plot_confusion(y_teste, p['y_pred'], CLASSES, f'{melhor} (teste temporal)', config.MODEL_FIG_DIR / f'{PROBLEMA}_confusao.png')
    por_classe = per_class_metrics(y_teste, p['y_pred'], CLASSES).assign(modelo=melhor)
    por_classe.to_csv(config.TABLES_DIR / f'por_classe_{PROBLEMA}.csv', index=False)
    print(por_classe.to_string(index=False))

    p = preds[melhor_arvore]
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipe': p['pipe'], 'features': p['features']}, config.MODELS_DIR / f'{PROBLEMA}_{melhor_arvore}.joblib')
    imp = importance_by_permutation(p['pipe'], df.loc[p['mask'], p['features']], y[p['mask']], scoring='f1_macro')
    imp.to_csv(config.TABLES_DIR / f'importancia_{PROBLEMA}.csv', index=False)
    plot_importance(imp, f'Multiclasse: importância por permuta ({melhor_arvore}, teste temporal, F1 macro)',
                    config.MODEL_FIG_DIR / f'{PROBLEMA}_importancia.png')
    print(f"\nImportância por permuta ({melhor_arvore}):\n{imp.head(10).to_string(index=False)}")
    print(f"\n[OK] resultados em {config.RESULTS_CSV.relative_to(config.ROOT)}; figuras em {config.MODEL_FIG_DIR.relative_to(config.ROOT)}/")


def run(modelos: str):
    config.MODEL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.OCORRENCIAS_CSV, parse_dates=['data'])
    y = df['classe']
    ctx = SplitContext()
    print(f"{len(df)} ocorrências; treino temporal {(df['split_temporal'] == 'treino').sum()}, "
          f"teste {(df['split_temporal'] == 'teste').sum()}")
    if modelos in ('classicos', 'todos'):
        run_classicos(df, y, ctx)
    if modelos in ('mlp', 'todos'):
        # A MLP recebe a classe como índice 0–3, na ordem de CLASSES
        codigos = pd.Categorical(y, categories=CLASSES).codes
        run_mlp_section(df, codigos, PROBLEMA, PROBLEMA, DATASET, ctx, classes=CLASSES)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelos', choices=['classicos', 'mlp', 'todos'], default='todos',
                        help='quais modelos treinar (padrão: todos)')
    run(parser.parse_args().modelos)
