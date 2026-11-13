"""
Etapa 8: regularização e overfitting da MLP.

Cinco configurações (A a E em config.REG_CONFIGS), mudando uma coisa por vez,
nos três problemas, com as mesmas sementes e o mesmo recorte do teste
temporal da Etapa 6: ajuste até 2012, validação em 2013 (é ela que decide o
early stopping) e teste em 2014–2015. A configuração A treina até o fim, sem
early stopping, para a curva de overfitting aparecer; B a E param pela
validação. Para cada treino ficam a curva por época, a lacuna entre treino e
validação no modelo final e a métrica de teste.
"""
import argparse
import importlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src import config
from src.evaluation import HIGHER_IS_BETTER, MAIN_METRIC
from src.mlp import inner_validation_mask, main_metric, predict_mlp, to_dense, train_mlp
from src.models import SplitContext, feature_set, iter_splits, make_preprocessor

optuna_script = importlib.import_module('experiments.07_optuna')
load_problem, PROBLEMAS, DATASETS = optuna_script.load_problem, optuna_script.PROBLEMAS, optuna_script.DATASETS
CONJUNTO = 'completo'


def run_configs(problema: str, ctx: SplitContext):
    """Treina A–E com cada semente; devolve (uma linha por treino, curvas por época)."""
    metrica, maior = MAIN_METRIC[problema], HIGHER_IS_BETTER[MAIN_METRIC[problema]]
    df, y, classes = load_problem(problema)
    feats = feature_set(CONJUNTO, DATASETS[problema], include_history=True)
    _, m_tr, m_teste = next(iter_splits(df, 'temporal', ctx))
    m_fit, m_val = inner_validation_mask(df, m_tr, 'temporal', ctx, config.SEED)
    assert not (m_teste & (m_fit | m_val)).any() and not (m_fit & m_val).any()
    pre = make_preprocessor(feats, scale=True).fit(df.loc[m_fit, feats])
    X_fit, X_val, X_teste = (to_dense(pre.transform(df.loc[m, feats])) for m in (m_fit, m_val, m_teste))
    print(f"  ajuste {m_fit.sum()} linhas, validação {m_val.sum()}, teste {m_teste.sum()}; {X_fit.shape[1]} entradas")

    linhas, curvas = [], []
    for nome, mudancas in config.REG_CONFIGS.items():
        params = {**config.MLP_DEFAULTS, **mudancas}
        for seed in config.MLP_SEEDS_TEST:
            t0 = time.perf_counter()
            model, hist, epoca = train_mlp(X_fit, y[m_fit], X_val, y[m_val], problema, seed=seed,
                                           n_out=len(classes) if classes else None, **params)
            tempo = time.perf_counter() - t0
            final = hist[hist['epoca'] == epoca].iloc[0]   # época do modelo que fica (a melhor, ou a última em A)
            y_pred, y_proba = predict_mlp(model, X_teste, problema)
            teste = float(main_metric(problema, y[m_teste], y_pred, y_proba))
            # Lacuna positiva = pior na validação que no treino
            lacuna = (final['metrica_treino'] - final['metrica_val']) * (1 if maior else -1)
            linhas.append({'problema': problema, 'configuracao': nome, 'semente': seed, 'epoca_final': epoca,
                           'epocas_treinadas': len(hist), 'loss_treino': final['loss_treino'], 'loss_val': final['loss_val'],
                           'lacuna_loss': final['loss_val'] - final['loss_treino'], 'metrica_treino': final['metrica_treino'],
                           'metrica_val': final['metrica_val'], 'lacuna_metrica': lacuna, 'metrica_teste': teste,
                           'melhor_val_no_caminho': hist['metrica_val'].max() if maior else hist['metrica_val'].min(),
                           'tempo_s': round(tempo, 2)})
            curvas.append(hist.assign(problema=problema, configuracao=nome, semente=seed, epoca_final=epoca))
            print(f"  {nome:22s} semente {seed}: modelo da época {epoca:3d} de {len(hist):3d} | {metrica} treino "
                  f"{final['metrica_treino']:.3f} val {final['metrica_val']:.3f} teste {teste:.3f} | {tempo:5.1f}s", flush=True)
    return pd.DataFrame(linhas), pd.concat(curvas, ignore_index=True)


def summarize(resultados: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio entre sementes, por configuração."""
    colunas = ['epoca_final', 'metrica_treino', 'metrica_val', 'metrica_teste', 'lacuna_metrica', 'lacuna_loss', 'tempo_s']
    g = resultados.groupby('configuracao', sort=False)[colunas]
    return g.mean().join(g.std().add_suffix('_dp')).round(4)


def plot_curves(curvas: pd.DataFrame, problema: str, path):
    """Loss (em cima) e métrica (embaixo) por época, treino × validação, uma coluna por configuração."""
    metrica = MAIN_METRIC[problema]
    nomes = list(config.REG_CONFIGS)
    fig, axes = plt.subplots(2, len(nomes), figsize=(3.6 * len(nomes), 7), sharey='row')
    for j, nome in enumerate(nomes):
        c = curvas[curvas['configuracao'] == nome]
        for _, h in c.groupby('semente'):
            for i, base in enumerate(['loss', 'metrica']):
                axes[i, j].plot(h['epoca'], h[f'{base}_treino'], color='steelblue', alpha=0.6, lw=1)
                axes[i, j].plot(h['epoca'], h[f'{base}_val'], color='darkorange', alpha=0.8, lw=1)
            if config.REG_CONFIGS[nome]['early_stopping']:
                for ax in axes[:, j]:
                    ax.axvline(h['epoca_final'].iloc[0], color='gray', ls=':', lw=0.8)
        axes[0, j].set_title(nome, fontsize=11)
        axes[1, j].set_xlabel('época')
    axes[0, 0].set_ylabel('loss')
    axes[1, 0].set_ylabel(metrica)
    axes[0, 0].plot([], [], color='steelblue', label='treino')
    axes[0, 0].plot([], [], color='darkorange', label='validação (2013)')
    axes[0, 0].plot([], [], color='gray', ls=':', label='época em que o modelo fica')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'Regularização, MLP {problema}: treino × validação por época (5 sementes; eixo x próprio de cada configuração)')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_gap(resultados: pd.DataFrame, problema: str, path):
    """Métrica no treino, na validação e no teste do modelo final de cada configuração."""
    metrica = MAIN_METRIC[problema]
    longo = resultados.melt(id_vars=['configuracao', 'semente'], value_vars=['metrica_treino', 'metrica_val', 'metrica_teste'],
                            var_name='conjunto', value_name='valor')
    longo['conjunto'] = longo['conjunto'].map({'metrica_treino': 'treino', 'metrica_val': 'validação (2013)', 'metrica_teste': 'teste (2014–2015)'})
    m = longo.groupby(['configuracao', 'conjunto'], sort=False)['valor'].agg(['mean', 'std']).reset_index()
    conjuntos, cores = ['treino', 'validação (2013)', 'teste (2014–2015)'], ['steelblue', 'darkorange', 'seagreen']
    todas = list(config.REG_CONFIGS)
    # À esquerda as cinco; à direita só B–E, porque a escala de A esconde as diferenças entre elas
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [5, 4]})
    for ax, nomes in zip(axes, [todas, todas[1:]]):
        sub = m[m['configuracao'].isin(nomes)]
        for k, (conj, cor) in enumerate(zip(conjuntos, cores)):
            d = sub[sub['conjunto'] == conj].set_index('configuracao').reindex(nomes)
            ax.bar(np.arange(len(nomes)) + (k - 1) * 0.27, d['mean'], 0.27, yerr=d['std'], color=cor, label=conj, capsize=2)
        lo, hi = (sub['mean'] - sub['std']).min(), (sub['mean'] + sub['std']).max()
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_xticks(range(len(nomes)), [n.replace(' (32)', '') for n in nomes], fontsize=9)
    axes[0].set_ylabel(f"{metrica} ({'maior' if HIGHER_IS_BETTER[metrica] else 'menor'} é melhor)")
    axes[0].set_title('As cinco configurações')
    axes[1].set_title('Só B a E (eixo ampliado)')
    axes[0].legend(fontsize=9)
    fig.suptitle(f'Regularização, MLP {problema}: modelo final no treino, na validação e no teste (média ± desvio de 5 sementes)')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def run(problema: str, ctx: SplitContext):
    metrica = MAIN_METRIC[problema]
    print(f"\n{'=' * 100}\nREGULARIZAÇÃO: MLP {problema} ({metrica}; {'maior' if HIGHER_IS_BETTER[metrica] else 'menor'} é melhor)\n{'=' * 100}")
    resultados, curvas = run_configs(problema, ctx)
    resultados.to_csv(config.TABLES_DIR / f'regularizacao_{problema}.csv', index=False)
    curvas.to_csv(config.TABLES_DIR / f'regularizacao_curvas_{problema}.csv', index=False)
    resumo = summarize(resultados)
    resumo.to_csv(config.TABLES_DIR / f'regularizacao_resumo_{problema}.csv')
    plot_curves(curvas, problema, config.REG_FIG_DIR / f'{problema}_curvas.png')
    plot_gap(resultados, problema, config.REG_FIG_DIR / f'{problema}_treino_val_teste.png')
    print(f"\n  Resumo (média de {len(config.MLP_SEEDS_TEST)} sementes; lacuna positiva = pior na validação que no treino):")
    print(resumo[['epoca_final', 'metrica_treino', 'metrica_val', 'lacuna_metrica', 'metrica_teste', 'metrica_teste_dp', 'tempo_s']].to_string())
    print(f"\n[OK] {problema}: regularizacao_*_{problema}.csv em {config.TABLES_DIR.relative_to(config.ROOT)}/; "
          f"figuras em {config.REG_FIG_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problema', choices=PROBLEMAS + ['todos'], default='todos')
    args = parser.parse_args()
    torch.set_num_threads(config.TORCH_THREADS)
    config.REG_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ctx = SplitContext()
    for problema in (PROBLEMAS if args.problema == 'todos' else [args.problema]):
        run(problema, ctx)
