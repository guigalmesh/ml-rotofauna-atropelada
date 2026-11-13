"""
EDA dos atributos de modelagem (complemento da Etapa 3): estatísticas
descritivas, valores ausentes, outliers (regra de 1,5 × IQR) e correlação
entre atributos, para apontar redundâncias. Lê data/processed/ e usa só o
treino temporal, para que o teste não entre em nenhuma decisão.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.models import feature_set

LIMIAR_CORRELACAO = 0.8


def describe(df: pd.DataFrame, cols: list[str], dataset: str) -> pd.DataFrame:
    linhas = []
    for c in cols:
        s = df[c]
        if c in config.CATEGORICAL_FEATURES:
            linhas.append({'dataset': dataset, 'atributo': c, 'tipo': 'categórico', 'n': len(s), 'ausentes_pct': s.isna().mean() * 100,
                           'categorias': s.nunique(), 'mais_frequente': f"{s.mode()[0]} ({s.value_counts(normalize=True).iloc[0] * 100:.0f}%)"})
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        fora = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100
        linhas.append({'dataset': dataset, 'atributo': c, 'tipo': 'numérico', 'n': len(s), 'ausentes_pct': s.isna().mean() * 100,
                       'media': s.mean(), 'desvio': s.std(), 'min': s.min(), 'q1': q1, 'mediana': s.median(), 'q3': q3,
                       'max': s.max(), 'assimetria': s.skew(), 'outliers_iqr_pct': fora})
    return pd.DataFrame(linhas)


def high_correlations(corr: pd.DataFrame, dataset: str) -> pd.DataFrame:
    pares = corr.where(np.triu(np.ones(corr.shape, bool), k=1)).stack().rename('spearman').reset_index()
    pares.columns = ['atributo_1', 'atributo_2', 'spearman']
    pares = pares[pares['spearman'].abs() >= LIMIAR_CORRELACAO]
    return pares.assign(dataset=dataset).sort_values('spearman', key=abs, ascending=False)


def run():
    config.EDA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    oc = pd.read_csv(config.OCORRENCIAS_CSV)
    painel = pd.read_csv(config.SEGMENTOS_CSV)
    oc, painel = oc[oc['split_temporal'] == 'treino'], painel[painel['split_temporal'] == 'treino']
    f_oc = feature_set('completo', 'ocorrencias')
    f_painel = feature_set('completo', 'segmentos_periodo', include_history=True)
    print(f"Treino temporal: {len(oc)} ocorrências ({len(f_oc)} atributos); {len(painel)} linhas de painel ({len(f_painel)} atributos + alvo n)")

    # 1. ESTATÍSTICAS DESCRITIVAS, AUSENTES E OUTLIERS
    desc = pd.concat([describe(oc, f_oc, 'ocorrencias'), describe(painel, f_painel + ['n'], 'segmentos_periodo')], ignore_index=True)
    desc.round(3).to_csv(config.TABLES_DIR / 'estatisticas_descritivas.csv', index=False)
    num = desc[(desc['dataset'] == 'segmentos_periodo') & (desc['tipo'] == 'numérico')]
    print("\nPainel, atributos numéricos:")
    print(num[['atributo', 'ausentes_pct', 'media', 'desvio', 'min', 'mediana', 'max', 'assimetria', 'outliers_iqr_pct']].round(2).to_string(index=False))

    # 2. CORRELAÇÃO ENTRE ATRIBUTOS (Spearman: não supõe relação linear nem normalidade)
    altas = []
    fig, axes = plt.subplots(1, 2, figsize=(21, 9), gridspec_kw={'width_ratios': [1, 1.25]})
    for ax, (nome, df, feats) in zip(axes, [('ocorrências', oc, f_oc), ('painel segmento × mês', painel, f_painel + ['n'])]):
        cols = [c for c in feats if c not in config.CATEGORICAL_FEATURES]
        corr = df[cols].corr(method='spearman')
        altas.append(high_correlations(corr, nome))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, ax=ax, cbar=False, annot_kws={'size': 7})
        ax.set_title(f'Correlação de Spearman entre atributos numéricos: {nome} (treino)')
    plt.tight_layout()
    plt.savefig(config.EDA_FIG_DIR / '17_correlacao_atributos.png', dpi=200)
    plt.close()
    altas = pd.concat(altas, ignore_index=True)
    altas.round(3).to_csv(config.TABLES_DIR / 'correlacoes_altas.csv', index=False)
    print(f"\nPares com |Spearman| >= {LIMIAR_CORRELACAO}:\n{altas.round(2).to_string(index=False)}")
    alvo = painel[[c for c in f_painel if c not in config.CATEGORICAL_FEATURES] + ['n']].corr(method='spearman')['n'].drop('n')
    print(f"\nCorrelação de cada atributo com n (painel):\n{alvo.sort_values(key=abs, ascending=False).round(3).to_string()}")

    # 3. OUTLIERS: boxplots padronizados e a cauda do alvo
    cols = [c for c in f_painel if c not in config.CATEGORICAL_FEATURES and painel[c].nunique() > 2]
    z = (painel[cols] - painel[cols].mean()) / painel[cols].std()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={'width_ratios': [2.2, 1]})
    sns.boxplot(data=z, orient='h', ax=axes[0], fliersize=2, color='steelblue')
    axes[0].set_title('Atributos numéricos do painel, padronizados (treino): pontos além de 1,5 × IQR')
    axes[0].set_xlabel('desvios-padrão em relação à média')
    contagem = painel['n'].value_counts().sort_index()
    axes[1].bar(contagem.index, contagem.values, color='steelblue')
    axes[1].set_yscale('log')
    axes[1].set_title(f"Alvo n por segmento × mês (máx. {painel['n'].max()}; {(painel['n'] == 0).mean() * 100:.0f}% zeros)")
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('linhas (escala log)')
    plt.tight_layout()
    plt.savefig(config.EDA_FIG_DIR / '18_outliers_atributos.png', dpi=200)
    plt.close()
    print(f"\nLinhas com n >= 5: {(painel['n'] >= 5).sum()} ({(painel['n'] >= 5).mean() * 100:.2f}%); maior n: {painel['n'].max()}")
    print("\n[OK] estatisticas_descritivas.csv, correlacoes_altas.csv e figuras 17 e 18")


if __name__ == '__main__':
    run()
