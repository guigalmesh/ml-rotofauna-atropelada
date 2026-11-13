"""
Etapa 7: otimização dos hiperparâmetros da MLP com Optuna.

Um estudo por problema (multiclasse, binário, regressão), no conjunto de
atributos completo. A função objetivo só recebe o treino temporal (<= 2013) e
imita o protocolo do teste um ano antes: ajusta até 2011, usa 2012 no early
stopping e mede a métrica principal em 2013, em média de 3 sementes. O teste
(2014–2015) só é lido depois do fim da busca, quando a melhor configuração é
retreinada com o protocolo da Etapa 6 (modelo `mlp_optuna`) e comparada com a
MLP original e com os clássicos.

Os estudos ficam em outputs/optuna.db: rodar de novo continua de onde parou;
--reiniciar apaga o estudo e recomeça.
"""
import argparse
import json
import time

import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import optuna
import pandas as pd
import torch

from src import config
from src.evaluation import (
    HIGHER_IS_BETTER, MAIN_METRIC, comparison_table, plot_comparison, plot_learning_curves, save_results,
)
from src.mlp import MLP, evaluate_mlp, main_metric, mlp_test_report, params_from_search, predict_mlp, to_dense, train_mlp
from src.models import SplitContext, feature_set, make_preprocessor

CLASSES = ['birds', 'reptiles', 'mammals', 'amphibians']
PROBLEMAS = ['multiclasse', 'binario', 'regressao']
DATASETS = {'multiclasse': 'ocorrencias', 'binario': 'segmentos_periodo', 'regressao': 'segmentos_periodo'}
CONJUNTO = 'completo'
ESCALA_LOG = {'n_units', 'lr', 'weight_decay'}
ESTADOS = {optuna.trial.TrialState.COMPLETE: 'completo', optuna.trial.TrialState.PRUNED: 'podado'}


# =============================================================================
# DADOS
# =============================================================================
def load_problem(problema: str):
    """(df, y, classes) do problema; y como vetor numérico, no formato que a MLP recebe."""
    if problema == 'multiclasse':
        df = pd.read_csv(config.OCORRENCIAS_CSV)
        return df, pd.Categorical(df['classe'], categories=CLASSES).codes, CLASSES
    df = pd.read_csv(config.SEGMENTOS_CSV)
    if problema == 'binario':
        return df, (df['n'] >= config.HOTSPOT_MIN_N).astype(int).values, None
    return df, df['n'].astype(float).values, None


def validation_arrays(df_tr: pd.DataFrame, y_tr: np.ndarray, feats: list[str]):
    """
    Recorte do treino temporal para o objective, um ano antes do protocolo do
    teste: ajuste até o antepenúltimo ano, early stopping no penúltimo e
    validação (valor objetivo) no último. O pré-processador é ajustado só no
    recorte de ajuste.
    """
    ano = df_tr['ano'].values
    assert ano.max() <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR, 'o objective recebeu linhas do teste'
    ano_val, ano_es = ano.max(), ano.max() - 1
    mascaras = (ano < ano_es, ano == ano_es, ano == ano_val)
    pre = make_preprocessor(feats, scale=True).fit(df_tr.loc[mascaras[0], feats])
    X = [to_dense(pre.transform(df_tr.loc[m, feats])) for m in mascaras]
    y = [y_tr[m] for m in mascaras]
    print(f"  objective: ajuste até {ano_es - 1} ({mascaras[0].sum()} linhas), early stopping em {ano_es} "
          f"({mascaras[1].sum()}), validação em {ano_val} ({mascaras[2].sum()}); {X[0].shape[1]} entradas")
    return X, y


# =============================================================================
# ESTUDO
# =============================================================================
def suggest(trial: optuna.Trial) -> dict:
    e = config.OPTUNA_SEARCH_SPACE
    return {
        'n_layers': trial.suggest_int('n_layers', *e['n_layers']),
        'n_units': trial.suggest_int('n_units', *e['n_units'], log=True),
        'activation': trial.suggest_categorical('activation', e['activation']),
        'optimizer': trial.suggest_categorical('optimizer', e['optimizer']),
        'lr': trial.suggest_float('lr', *e['lr'], log=True),
        'batch_size': trial.suggest_categorical('batch_size', e['batch_size']),
        'dropout': trial.suggest_float('dropout', *e['dropout']),
        'weight_decay': trial.suggest_float('weight_decay', *e['weight_decay'], log=True),
    }


def make_objective(task: str, X: list, y: list, n_out: int | None):
    """Média da métrica principal na validação entre as sementes; poda pela média parcial."""
    (X_fit, X_es, X_val), (y_fit, y_es, y_val) = X, y

    def objective(trial: optuna.Trial) -> float:
        params = {**config.MLP_DEFAULTS, **params_from_search(suggest(trial))}
        valores, epocas = [], []
        for passo, seed in enumerate(config.OPTUNA_SEEDS_TRIAL):
            model, _, epoca = train_mlp(X_fit, y_fit, X_es, y_es, task, seed=seed, n_out=n_out, **params)
            y_pred, y_proba = predict_mlp(model, X_val, task)
            valores.append(float(main_metric(task, y_val, y_pred, y_proba)))
            epocas.append(epoca)
            trial.set_user_attr('epocas', float(np.mean(epocas)))
            trial.report(float(np.mean(valores)), passo)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('desvio_sementes', float(np.std(valores, ddof=1)))
        return float(np.mean(valores))
    return objective


def get_study(problema: str, reiniciar: bool, podar: bool) -> optuna.Study:
    storage, nome = f'sqlite:///{config.OPTUNA_DB}', f'mlp_{problema}'
    if reiniciar:
        try:
            optuna.delete_study(study_name=nome, storage=storage)
        except KeyError:
            pass
    pruner = optuna.pruners.MedianPruner(n_startup_trials=config.OPTUNA_STARTUP_TRIALS) if podar else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=nome, storage=storage, load_if_exists=True,
        direction='maximize' if HIGHER_IS_BETTER[MAIN_METRIC[problema]] else 'minimize',
        sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=config.OPTUNA_STARTUP_TRIALS), pruner=pruner)
    if not study.trials:
        study.enqueue_trial(config.OPTUNA_FIRST_TRIAL)
    return study


def finished(study: optuna.Study) -> list:
    return [t for t in study.trials if t.state in ESTADOS]


def log_trial(metrica: str):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        p = trial.params
        feitos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        melhor = f"melhor {study.best_value:.4f} no trial {study.best_trial.number}" if feitos else 'sem melhor ainda'
        valor = trial.value if trial.value is not None else list(trial.intermediate_values.values())[-1]
        rede = ' → '.join(str(h) for h in params_from_search(p)['hidden'])
        print(f"  trial {trial.number:3d} {ESTADOS.get(trial.state, 'falhou'):8s} {metrica} {valor:9.4f} | {melhor} | "
              f"{rede:14s} {p['activation']:4s} {p['optimizer']:7s} lr {p['lr']:.1e} batch {p['batch_size']:3d} "
              f"dropout {p['dropout']:.2f} wd {p['weight_decay']:.1e} | época {trial.user_attrs.get('epocas', 0):5.1f} | "
              f"{trial.duration.total_seconds():5.1f}s", flush=True)
    return callback


# =============================================================================
# EXPORTAÇÃO DO ESTUDO
# =============================================================================
def trials_table(study: optuna.Study) -> pd.DataFrame:
    """Uma linha por trial; no podado, o valor é a média parcial das sementes treinadas."""
    linhas = []
    for t in finished(study):
        parciais = list(t.intermediate_values.values())
        linhas.append({'trial': t.number, 'estado': ESTADOS[t.state], 'valor': t.value if t.value is not None else parciais[-1],
                       **t.params, 'sementes_treinadas': len(parciais), 'epocas_media': t.user_attrs.get('epocas'),
                       'desvio_sementes': t.user_attrs.get('desvio_sementes'), 'duracao_s': round(t.duration.total_seconds(), 2)})
    return pd.DataFrame(linhas)


def visible_range(valores: pd.Series, maior_melhor: bool) -> tuple[float, float]:
    """Faixa do eixo y que mostra a parte boa da busca; trials que divergiram ficam fora."""
    melhor = valores.max() if maior_melhor else valores.min()
    corte = valores.quantile(0.1 if maior_melhor else 0.9)
    folga = 0.12 * abs(melhor - corte) + 1e-4
    return (corte - folga, melhor + folga) if maior_melhor else (melhor - folga, corte + folga)


def plot_history(tabela: pd.DataFrame, metrica: str, problema: str, path):
    maior = HIGHER_IS_BETTER[metrica]
    comp, pod = tabela[tabela['estado'] == 'completo'], tabela[tabela['estado'] == 'podado']
    acumulado = comp['valor'].cummax() if maior else comp['valor'].cummin()
    lo, hi = visible_range(comp['valor'], maior)
    fora = ((tabela['valor'] < lo) | (tabela['valor'] > hi)).sum()
    plt.figure(figsize=(11, 4.5))
    plt.scatter(pod['trial'], pod['valor'], marker='x', color='gray', s=28, label=f'podado ({len(pod)}; média parcial)')
    plt.scatter(comp['trial'], comp['valor'], color='steelblue', s=28, label=f'completo ({len(comp)})')
    plt.step(comp['trial'], acumulado, where='post', color='darkorange', lw=1.8, label='melhor até o trial')
    plt.axhline(tabela.loc[tabela['trial'] == 0, 'valor'].iloc[0], color='black', ls=':', lw=1, label='trial 0: configuração inicial')
    plt.ylim(lo, hi)
    plt.xlabel('trial')
    plt.ylabel(f"{metrica} na validação ({'maior' if maior else 'menor'} é melhor)")
    plt.title(f'Optuna, MLP {problema}: histórico da otimização' + (f' (trials fora da faixa do eixo: {fora})' if fora else ''))
    plt.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.14), frameon=False)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_slices(tabela: pd.DataFrame, metrica: str, problema: str, path):
    """Valor objetivo contra cada hiperparâmetro (trials completos; cor = ordem do trial)."""
    maior = HIGHER_IS_BETTER[metrica]
    comp = tabela[tabela['estado'] == 'completo']
    lo, hi = visible_range(comp['valor'], maior)
    melhor = comp.loc[comp['valor'].idxmax() if maior else comp['valor'].idxmin()]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    for ax, nome in zip(axes.ravel(), config.OPTUNA_SEARCH_SPACE):
        categorico = isinstance(config.OPTUNA_SEARCH_SPACE[nome], list)
        if categorico:
            niveis = [str(v) for v in config.OPTUNA_SEARCH_SPACE[nome]]
            x = comp[nome].astype(str).map({v: i for i, v in enumerate(niveis)})
            x = x + np.random.default_rng(config.SEED).uniform(-0.15, 0.15, len(x))
            ax.set_xticks(range(len(niveis)), niveis)
            x_melhor = niveis.index(str(melhor[nome]))
        else:
            x, x_melhor = comp[nome], melhor[nome]
            lo_x, hi_x = config.OPTUNA_SEARCH_SPACE[nome]
            if nome == 'n_layers':
                ax.set_xticks(range(lo_x, hi_x + 1))
            ax.set_xlim((lo_x / 1.4, hi_x * 1.4) if nome in ESCALA_LOG else (lo_x - 0.08 * (hi_x - lo_x), hi_x + 0.08 * (hi_x - lo_x)))
            if nome in ESCALA_LOG:
                ax.set_xscale('log')
                if nome == 'n_units':
                    ax.set_xticks([16, 32, 64, 128, 256], ['16', '32', '64', '128', '256'])
                ax.xaxis.set_minor_formatter(NullFormatter())
        pontos = ax.scatter(x, comp['valor'], c=comp['trial'], cmap='viridis', s=22)
        ax.scatter([x_melhor], [melhor['valor']], marker='*', color='red', s=160, zorder=3)
        ax.set_title(nome)
        ax.set_ylim(lo, hi)
    for ax in axes[:, 0]:
        ax.set_ylabel(metrica)
    fig.colorbar(pontos, ax=axes, label='trial', shrink=0.8)
    fig.suptitle(f"Optuna, MLP {problema}: {metrica} na validação por hiperparâmetro (estrela = melhor trial)")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_param_importance(importancia: pd.Series, problema: str, path):
    plt.figure(figsize=(8, 4))
    importancia.sort_values().plot.barh(color='steelblue')
    plt.xlabel('importância (fANOVA; fração da variância do valor objetivo)')
    plt.title(f'Optuna, MLP {problema}: importância dos hiperparâmetros')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def export_study(study: optuna.Study, problema: str) -> dict:
    metrica = MAIN_METRIC[problema]
    tabela = trials_table(study)
    tabela.to_csv(config.TABLES_DIR / f'optuna_trials_{problema}.csv', index=False)
    importancia = pd.Series(optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=config.SEED)), name='importancia')
    importancia.rename_axis('hiperparametro').round(4).to_csv(config.TABLES_DIR / f'optuna_importancia_{problema}.csv')
    plot_history(tabela, metrica, problema, config.OPTUNA_FIG_DIR / f'{problema}_historico.png')
    plot_slices(tabela, metrica, problema, config.OPTUNA_FIG_DIR / f'{problema}_hiperparametros.png')
    plot_param_importance(importancia, problema, config.OPTUNA_FIG_DIR / f'{problema}_importancia.png')

    melhor = study.best_trial
    resumo = {'problema': problema, 'metrica': metrica, 'trials': len(tabela),
              'completos': int((tabela['estado'] == 'completo').sum()), 'podados': int((tabela['estado'] == 'podado').sum()),
              'valor_trial_0': float(tabela.loc[tabela['trial'] == 0, 'valor'].iloc[0]), 'melhor_valor': float(melhor.value),
              'melhor_trial': melhor.number, 'tempo_busca_min': round(tabela['duracao_s'].sum() / 60, 1), **melhor.params}
    print(f"\n  {resumo['trials']} trials ({resumo['completos']} completos, {resumo['podados']} podados) em {resumo['tempo_busca_min']} min")
    print(f"  {metrica} na validação: trial 0 (configuração inicial) {resumo['valor_trial_0']:.4f} → "
          f"melhor {resumo['melhor_valor']:.4f} (trial {melhor.number})")
    print(f"  melhores hiperparâmetros: {melhor.params}")
    print(f"  importância: {importancia.round(3).to_dict()}")
    return resumo


def confirm_on_validation(problema: str, X: list, y: list, n_out: int | None, melhores: dict) -> dict:
    """
    Repete na validação a configuração inicial e a melhor com sementes que a
    busca não usou. Separa o ganho que se sustenta na validação do que foi
    sorte das 3 sementes do trial escolhido. Não usa o teste.
    """
    metrica, maior = MAIN_METRIC[problema], HIGHER_IS_BETTER[MAIN_METRIC[problema]]
    (X_fit, X_es, X_val), (y_fit, y_es, y_val) = X, y
    linhas = []
    for nome, p in [('inicial', config.OPTUNA_FIRST_TRIAL), ('otimizada', melhores)]:
        params = {**config.MLP_DEFAULTS, **params_from_search(p)}
        for seed in config.OPTUNA_SEEDS_CONFIRM:
            model, _, epoca = train_mlp(X_fit, y_fit, X_es, y_es, problema, seed=seed, n_out=n_out, **params)
            y_pred, y_proba = predict_mlp(model, X_val, problema)
            linhas.append({'configuracao': nome, 'semente': seed, 'valor': float(main_metric(problema, y_val, y_pred, y_proba)), 'epoca': epoca})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(config.TABLES_DIR / f'optuna_confirmacao_{problema}.csv', index=False)
    g = tabela.groupby('configuracao')['valor'].agg(['mean', 'std'])
    pares = tabela.pivot(index='semente', columns='configuracao', values='valor')
    vitorias = int(((pares['otimizada'] > pares['inicial']) if maior else (pares['otimizada'] < pares['inicial'])).sum())
    print(f"\n  Confirmação na validação com {len(pares)} sementes novas ({metrica}): inicial {g.loc['inicial', 'mean']:.4f} ± "
          f"{g.loc['inicial', 'std']:.4f}; otimizada {g.loc['otimizada', 'mean']:.4f} ± {g.loc['otimizada', 'std']:.4f}; "
          f"a otimizada vence em {vitorias} de {len(pares)} sementes")
    return {'confirmacao_inicial': float(g.loc['inicial', 'mean']), 'confirmacao_otimizada': float(g.loc['otimizada', 'mean']),
            'confirmacao_desvio_inicial': float(g.loc['inicial', 'std']), 'confirmacao_desvio_otimizada': float(g.loc['otimizada', 'std']),
            'confirmacao_vitorias': f'{vitorias} de {len(pares)}'}


def save_summary(resumo: dict):
    """optuna_resumo.csv e optuna_melhores.json: uma entrada por problema, substituída a cada execução."""
    path = config.TABLES_DIR / 'optuna_resumo.csv'
    antigo = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['problema'])
    novo = pd.concat([antigo[antigo['problema'] != resumo['problema']], pd.DataFrame([resumo])], ignore_index=True)
    inteiras = ['trials', 'completos', 'podados', 'melhor_trial', 'n_layers', 'n_units', 'batch_size', 'parametros_original', 'parametros_otimizada']
    novo[inteiras] = novo[inteiras].astype(int)
    novo.set_index('problema').reindex([p for p in PROBLEMAS if p in novo['problema'].values]).to_csv(path)
    path_json = config.TABLES_DIR / 'optuna_melhores.json'
    melhores = json.loads(path_json.read_text()) if path_json.exists() else {}
    melhores[resumo['problema']] = {k: resumo[k] for k in config.OPTUNA_SEARCH_SPACE}
    path_json.write_text(json.dumps({p: melhores[p] for p in PROBLEMAS if p in melhores}, indent=2) + '\n')


# =============================================================================
# AVALIAÇÃO FINAL: SÓ AQUI O TESTE É LIDO
# =============================================================================
def n_parameters(n_in: int, n_out: int, params: dict) -> int:
    return sum(p.numel() for p in MLP(n_in, n_out, params['hidden'], params['activation'], params['dropout']).parameters())


def final_evaluation(df, y, problema: str, classes, params: dict, ctx: SplitContext) -> dict:
    """Retreina a melhor configuração com o protocolo da Etapa 6 e compara com a MLP original e os clássicos."""
    metrica = MAIN_METRIC[problema]
    rows, historicos, artefato = evaluate_mlp(df, y, problema, problema, DATASETS[problema], [CONJUNTO], config.SPLITS, ctx,
                                              params=params, nome='mlp_optuna', classes=classes)
    results = save_results(rows, problema)
    curvas = pd.concat(historicos, ignore_index=True)
    curvas.to_csv(config.TABLES_DIR / f'curvas_mlp_optuna_{problema}.csv', index=False)
    plot_learning_curves(curvas, metrica, f'MLP otimizada, {problema}: treino × validação interna (teste temporal, conjunto completo)',
                         config.OPTUNA_FIG_DIR / f'{problema}_curvas_otimizada.png')
    joblib.dump(artefato, config.MODELS_DIR / f'{problema}_mlp_optuna.joblib')
    mlp_test_report(df, y, problema, artefato, nome='mlp_optuna')

    comparativo = comparison_table(results, problema, metrica)
    comparativo.to_csv(config.TABLES_DIR / f'comparativo_{problema}.csv')
    plot_comparison(results, problema, metrica, config.MODEL_FIG_DIR / f'{problema}_mlp_vs_classicos.png')

    # Original × otimizada: todas as métricas, por split, com tempo e época de parada
    r = results[(results['problema'] == problema) & results['modelo'].isin(['mlp', 'mlp_optuna'])
                & (results['conjunto_atributos'] == CONJUNTO)]
    par = (r.groupby(['metrica', 'split', 'modelo'])
            .agg(media=('valor', 'mean'), desvio=('valor', 'std'), tempo_s=('tempo_s', 'mean'), epocas=('epocas', 'mean'), treinos=('valor', 'size'))
            .round(4).reset_index())
    par.to_csv(config.TABLES_DIR / f'optuna_original_vs_otimizada_{problema}.csv', index=False)
    principal = par[par['metrica'] == metrica].drop(columns='metrica').set_index(['split', 'modelo']).reindex(config.SPLITS, level='split')
    print(f"\n  Original × otimizada ({metrica}; {'maior' if HIGHER_IS_BETTER[metrica] else 'menor'} é melhor):\n{principal.to_string()}")
    print(f"\n  Comparativo com os clássicos (conjunto completo):\n{comparativo.to_string()}")

    teste = principal.loc['temporal']
    n_out = len(classes) if classes else 1
    return {'teste_original': float(teste.loc['mlp', 'media']), 'teste_otimizada': float(teste.loc['mlp_optuna', 'media']),
            'desvio_teste_original': float(teste.loc['mlp', 'desvio']), 'desvio_teste_otimizada': float(teste.loc['mlp_optuna', 'desvio']),
            'tempo_s_original': float(teste.loc['mlp', 'tempo_s']), 'tempo_s_otimizada': float(teste.loc['mlp_optuna', 'tempo_s']),
            'parametros_original': n_parameters(artefato['n_in'], n_out, config.MLP_DEFAULTS),
            'parametros_otimizada': n_parameters(artefato['n_in'], n_out, params)}


# =============================================================================
# EXECUÇÃO
# =============================================================================
def run(problema: str, n_trials: int, reiniciar: bool, podar: bool, ctx: SplitContext):
    metrica = MAIN_METRIC[problema]
    print(f"\n{'=' * 100}\nOPTUNA: MLP {problema} (objetivo: {metrica} na validação, {'maximizar' if HIGHER_IS_BETTER[metrica] else 'minimizar'})\n{'=' * 100}")
    df, y, classes = load_problem(problema)
    feats = feature_set(CONJUNTO, DATASETS[problema], include_history=True)

    # 1. BUSCA: o objective recebe só o treino temporal
    treino = (df['split_temporal'] == 'treino').values
    X, y_partes = validation_arrays(df[treino].reset_index(drop=True), y[treino], feats)
    study = get_study(problema, reiniciar, podar)
    n_out = len(classes) if classes else None
    faltam = n_trials - len(finished(study))
    if faltam > 0:
        print(f"  {faltam} trials a rodar ({len(finished(study))} já no estudo); {len(config.OPTUNA_SEEDS_TRIAL)} sementes por trial; "
              f"poda {'pela mediana, após a 1ª semente' if podar else 'desligada'}")
        t0 = time.perf_counter()
        study.optimize(make_objective(problema, X, y_partes, n_out), n_trials=faltam, callbacks=[log_trial(metrica)])
        print(f"  busca: {(time.perf_counter() - t0) / 60:.1f} min")
    else:
        print(f"  estudo já tem {len(finished(study))} trials; pulando a busca")

    # 2. EXPORTAÇÃO E CONFIRMAÇÃO NA VALIDAÇÃO (ainda sem o teste)
    resumo = export_study(study, problema)
    resumo.update(confirm_on_validation(problema, X, y_partes, n_out, study.best_trial.params))

    # 3. AVALIAÇÃO FINAL (primeiro uso do teste)
    print(f"\n  Retreino da melhor configuração como 'mlp_optuna' (protocolo da Etapa 6)")
    params = params_from_search(study.best_trial.params)
    resumo.update(final_evaluation(df, y, problema, classes, params, ctx))
    save_summary(resumo)
    print(f"\n[OK] {problema}: tabelas optuna_*_{problema}.csv em {config.TABLES_DIR.relative_to(config.ROOT)}/; "
          f"figuras em {config.OPTUNA_FIG_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--problema', choices=PROBLEMAS + ['todos'], default='todos')
    parser.add_argument('--trials', type=int, default=config.OPTUNA_N_TRIALS, help=f'total de trials por estudo (padrão: {config.OPTUNA_N_TRIALS})')
    parser.add_argument('--reiniciar', action='store_true', help='apaga o estudo salvo e recomeça a busca')
    parser.add_argument('--sem-poda', action='store_true', help='desliga o MedianPruner')
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    torch.set_num_threads(config.TORCH_THREADS)
    for pasta in (config.OPTUNA_FIG_DIR, config.MODELS_DIR):
        pasta.mkdir(parents=True, exist_ok=True)
    ctx = SplitContext()
    for problema in (PROBLEMAS if args.problema == 'todos' else [args.problema]):
        run(problema, args.trials, args.reiniciar, not args.sem_poda, ctx)
