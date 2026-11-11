import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.external import points_gdf

# EDA de revisão antes da Etapa 2: esforço de campo, zeros, classes raras,
# sazonalidade do teste e blocos espaciais. Tudo exploratório; nada aqui é
# ajustado e reaproveitado pelos modelos
OUTPUT_DIR = config.EDA_FIG_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

UCS = ['esecae', 'pnb', 'jbb_fal_ibge']
# Grade da proposta inicial de blocos, avaliada aqui como diagnóstico; os blocos
# adotados são trechos ao longo da via (config.SPATIAL_BLOCK_M), criados nos splits
GRADE_DIAGNOSTICO_M = 3000
MESES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']


def load_data():
    df = pd.read_csv(config.INTERIM_CSV, parse_dates=['data'])
    df['treino'] = df['ano'] <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR
    campanhas = pd.read_csv(config.CAMPANHAS_CSV, parse_dates=['data'])
    segmentos = gpd.read_file(config.EXTERNAL_PROCESSED_DIR / 'segmentos_osm.gpkg')
    return df, campanhas, segmentos


def monthly_effort(campanhas: pd.DataFrame) -> pd.Series:
    """Campanhas por mês na janela das campanhas, com 0 nos meses sem nenhuma."""
    meses = pd.period_range(*config.CLIMATE_WINDOW, freq='M').astype(str)
    return campanhas.groupby('mes').size().reindex(meses, fill_value=0)


def zero_fractions(df: pd.DataFrame, campanhas: pd.DataFrame) -> pd.DataFrame:
    """
    Fração de campanhas sem registro silvestre em cada UC, observada e
    esperada por Poisson e por binomial negativa (momentos da própria UC).
    Zeros observados bem acima da binomial negativa indicariam circuito não
    percorrido; abaixo ou perto dela, são zeros reais
    """
    contagens = (pd.crosstab(df['campanha'], df['uc'])
                   .reindex(index=campanhas['campanha'], columns=UCS, fill_value=0))
    linhas = []
    for uc in UCS:
        m, v = contagens[uc].mean(), contagens[uc].var()
        p, r = m / v, m ** 2 / (v - m)
        linhas.append({'uc': uc, 'media': m, 'variancia': v,
                       'observado': (contagens[uc] == 0).mean(),
                       'poisson': np.exp(-m), 'binomial_negativa': p ** r})
    return pd.DataFrame(linhas).set_index('uc'), contagens


def assign_segments(df: pd.DataFrame, segmentos: gpd.GeoDataFrame) -> pd.DataFrame:
    """Segmento mais próximo de cada registro, até ASSIGN_MAX_DIST_M."""
    pontos = points_gdf(df)
    junto = gpd.sjoin_nearest(pontos, segmentos[['segmento_id', 'geometry']],
                              how='left', max_distance=config.ASSIGN_MAX_DIST_M)
    junto = junto[~junto.index.duplicated()]
    return df.assign(segmento_id=junto['segmento_id'])


def rate_panel(df: pd.DataFrame, segmentos: gpd.GeoDataFrame, esforco: pd.Series) -> pd.DataFrame:
    """Painel segmento × mês com n (alvo) e n_campanhas (só informativo)."""
    df = df.dropna(subset=['segmento_id'])
    n = df.groupby(['segmento_id', df['data'].dt.to_period('M').astype(str)]).size()
    indice = pd.MultiIndex.from_product([segmentos['segmento_id'], esforco.index], names=['segmento_id', 'mes'])
    painel = n.reindex(indice, fill_value=0).rename('n').reset_index()
    painel['n_campanhas'] = painel['mes'].map(esforco)
    painel['treino'] = painel['mes'].str[:4].astype(int) <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR
    return painel


def run_review_eda():
    df, campanhas, segmentos = load_data()
    teste_inicio = f'{config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR + 1}-01'

    # =========================================================================
    # 1. ESFORÇO DE CAMPO
    # =========================================================================
    print("=" * 75)
    print("1. ESFORÇO DE CAMPO (campanhas por mês)")
    print("=" * 75)
    esforco = monthly_effort(campanhas)
    faltam = sorted(set(range(1, campanhas['campanha'].max() + 1)) - set(campanhas['campanha']))
    print(f"Campanhas na base: {len(campanhas)} (ids sem nenhum registro: {faltam})")
    print(f"Meses sem campanha: {(esforco == 0).sum()} de {len(esforco)}")
    print(f"Campanhas por mês: mín {esforco.min()}, mediana {esforco.median():.0f}, máx {esforco.max()}")
    print(f"Campanhas sem registro silvestre: {(campanhas['n_silvestres'] == 0).sum()}")

    # Gráfico 8: campanhas por mês, com o período de teste destacado
    fig, ax = plt.subplots(figsize=(14, 4))
    cores = ['steelblue' if m < teste_inicio else 'darkorange' for m in esforco.index]
    ax.bar(range(len(esforco)), esforco.values, color=cores)
    ax.set_xticks(range(0, len(esforco), 3))
    ax.set_xticklabels(esforco.index[::3], rotation=45, ha='right')
    ax.axhline(esforco.median(), color='gray', ls='--', lw=1, label=f'mediana ({esforco.median():.0f})')
    ax.set_title('Campanhas por Mês (azul: treino, laranja: teste)')
    ax.set_ylabel('Campanhas')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '08_esforco_campanhas_mes.png', dpi=300)
    plt.close()

    # Gráfico 9: atropelamentos × campanhas no mês (o esforço infla a contagem?)
    mensal = df.groupby(df['data'].dt.to_period('M').astype(str)).size().reindex(esforco.index, fill_value=0)
    corr = np.corrcoef(esforco, mensal)[0, 1]
    print(f"Correlação campanhas no mês × atropelamentos no mês: r = {corr:.2f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.boxplot(x=esforco.values, y=mensal.values, color='lightsteelblue', ax=axes[0])
    axes[0].set_title(f'Atropelamentos no Mês por Nº de Campanhas (r = {corr:.2f})')
    axes[0].set_xlabel('Campanhas no mês')
    axes[0].set_ylabel('Atropelamentos silvestres no mês')
    taxa_mensal = mensal / esforco
    sns.boxplot(x=esforco.values, y=taxa_mensal.values, color='lightgreen', ax=axes[1])
    axes[1].set_title('Atropelamentos por Campanha, por Nº de Campanhas')
    axes[1].set_xlabel('Campanhas no mês')
    axes[1].set_ylabel('Atropelamentos por campanha')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '09_esforco_contagem_taxa.png', dpi=300)
    plt.close()

    # =========================================================================
    # 2. ZERO OBSERVADO × CIRCUITO NÃO PERCORRIDO
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. ZEROS POR UC")
    print("=" * 75)
    zeros, contagens = zero_fractions(df, campanhas)
    print(zeros.round(3))
    n_ucs = (contagens > 0).sum(axis=1)
    total = contagens.sum(axis=1)
    print("\nRegistros silvestres por campanha, pelo nº de UCs com registro:")
    print(total.groupby(n_ucs).describe()[['count', 'mean', '50%']].round(1))

    # Gráfico 10: zeros por UC e registros por campanha pelo nº de UCs
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    (zeros[['observado', 'poisson', 'binomial_negativa']] * 100).plot.bar(
        ax=axes[0], color=['indianred', 'lightgray', 'gray'], rot=0)
    axes[0].set_title('% de Campanhas sem Registro na UC')
    axes[0].set_ylabel('% das campanhas')
    axes[0].set_xlabel('')
    axes[0].legend(['observado', 'esperado (Poisson)', 'esperado (binomial negativa)'])
    sns.boxplot(x=n_ucs.values, y=total.values, color='lightsteelblue', ax=axes[1])
    axes[1].set_title('Registros por Campanha pelo Nº de UCs com Registro')
    axes[1].set_xlabel('UCs com pelo menos um registro')
    axes[1].set_ylabel('Registros silvestres na campanha')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '10_zeros_por_uc.png', dpi=300)
    plt.close()

    # =========================================================================
    # 3. CLASSES RARAS (MULTICLASSE)
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. CLASSES NO TREINO E NO TESTE (split temporal)")
    print("=" * 75)
    classes = pd.crosstab(df['classe'], df['treino'].map({True: 'treino', False: 'teste'}))
    classes = classes[['treino', 'teste']].sort_values('treino', ascending=False)
    classes['% treino'] = (classes['treino'] / classes['treino'].sum() * 100).round(2)
    print(classes)

    # Gráfico 11: classes no treino e no teste (escala log)
    classes[['treino', 'teste']].plot.barh(figsize=(8, 4.5), color=['steelblue', 'darkorange'], logx=True)
    plt.gca().invert_yaxis()
    plt.title('Ocorrências por Classe no Treino (≤ 2013) e no Teste (escala log)')
    plt.xlabel('Ocorrências')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '11_classes_treino_teste.png', dpi=300)
    plt.close()

    # =========================================================================
    # 4. SAZONALIDADE E COBERTURA DO TESTE
    # =========================================================================
    print("\n" + "=" * 75)
    print("4. SAZONALIDADE E COBERTURA DO TESTE")
    print("=" * 75)
    sazonal = pd.DataFrame({'taxa': taxa_mensal})
    sazonal['ano'] = sazonal.index.str[:4].astype(int)
    sazonal['mes_num'] = sazonal.index.str[5:].astype(int)
    sazonal['conjunto'] = np.where(sazonal['ano'] <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR, 'treino', 'teste')
    cobertura = sazonal.groupby(['mes_num', 'conjunto']).size().unstack(fill_value=0)
    cobertura.index = MESES
    print("Meses de cada mês do calendário em cada conjunto:")
    print(cobertura[['treino', 'teste']].T)
    por_conjunto = sazonal.groupby('conjunto')['taxa'].mean()
    print(f"\nAtropelamentos por campanha: treino {por_conjunto['treino']:.2f}, teste {por_conjunto['teste']:.2f}")

    # Gráfico 12: taxa por mês do calendário, uma linha por ano
    plt.figure(figsize=(10, 4.5))
    for ano, grupo in sazonal.groupby('ano'):
        teste = ano > config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR
        plt.plot(grupo['mes_num'], grupo['taxa'], marker='o', lw=2 if teste else 1,
                 ls='--' if teste else '-', label=f'{ano}{" (teste)" if teste else ""}')
    plt.axvspan(4.5, 9.5, color='khaki', alpha=0.3, label='estação seca (mai–set)')
    plt.xticks(range(1, 13), MESES)
    plt.title('Atropelamentos Silvestres por Campanha ao Longo do Ano')
    plt.ylabel('Atropelamentos por campanha')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '12_sazonalidade_taxa.png', dpi=300)
    plt.close()

    # =========================================================================
    # 5. PAINEL SEGMENTO × MÊS (prévia do alvo n)
    # =========================================================================
    print("\n" + "=" * 75)
    print("5. PAINEL SEGMENTO × MÊS")
    print("=" * 75)
    df = assign_segments(df, segmentos)
    print(f"Registros sem segmento a ≤ {config.ASSIGN_MAX_DIST_M} m: {df['segmento_id'].isna().sum()}")
    painel = rate_panel(df, segmentos, esforco)
    resumo = painel.groupby('treino').agg(linhas=('n', 'size'), pct_zero=('n', lambda s: (s == 0).mean() * 100),
                                          n_medio=('n', 'mean'), n_max=('n', 'max'))
    resumo.index = resumo.index.map({True: 'treino', False: 'teste'})
    print(resumo.round(3))
    por_segmento = painel[painel['treino']].groupby('segmento_id')['n'].sum().sort_values(ascending=False)
    top = por_segmento.head(int(len(por_segmento) * 0.2)).sum() / por_segmento.sum() * 100
    print(f"Os 20% de segmentos com mais atropelamentos no treino concentram {top:.0f}% dos atropelamentos")

    # Gráfico 13: distribuição de n e concentração por segmento
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    positivos = painel[painel['n'] > 0]
    sns.histplot(data=positivos, x='n', hue=positivos['treino'].map({True: 'treino', False: 'teste'}).rename('conjunto'),
                 stat='density', common_norm=False, discrete=True, ax=axes[0])
    axes[0].set_title(f'Atropelamentos nos Segmento × Mês com Registro\n({100 - resumo.loc["treino", "pct_zero"]:.0f}% das linhas do treino)')
    axes[0].set_xlabel('Atropelamentos no segmento no mês (n)')
    acumulado = por_segmento.cumsum() / por_segmento.sum() * 100
    axes[1].plot(np.arange(1, len(acumulado) + 1) / len(acumulado) * 100, acumulado.values, color='indianred')
    axes[1].plot([0, 100], [0, 100], color='gray', ls='--', lw=1)
    axes[1].set_title('Concentração dos Atropelamentos entre os Segmentos (treino)')
    axes[1].set_xlabel('% dos segmentos (do maior para o menor)')
    axes[1].set_ylabel('% acumulado dos atropelamentos')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '13_painel_segmento_mes.png', dpi=300)
    plt.close()

    # =========================================================================
    # 6. BLOCOS ESPACIAIS
    # =========================================================================
    print("\n" + "=" * 75)
    print(f"6. BLOCOS ESPACIAIS DE {GRADE_DIAGNOSTICO_M / 1000:.0f} KM")
    print("=" * 75)
    bloco = (segmentos['x_centro'] // GRADE_DIAGNOSTICO_M).astype(int).astype(str) + '_' + \
            (segmentos['y_centro'] // GRADE_DIAGNOSTICO_M).astype(int).astype(str)
    segmentos = segmentos.assign(bloco=bloco)
    por_bloco = segmentos.groupby('bloco').size()
    print(f"Blocos com segmento: {len(por_bloco)}; segmentos por bloco: "
          f"mín {por_bloco.min()}, mediana {por_bloco.median():.0f}, máx {por_bloco.max()}")
    print(f"Blocos com 1 ou 2 segmentos: {(por_bloco <= 2).sum()}")
    # Segmentos com vizinho (a ≤ 60 m) em outro bloco: na validação espacial, ficam colados ao treino
    buffer = segmentos.buffer(config.SEGMENT_MAX_GAP_M)
    vizinhos = gpd.sjoin(gpd.GeoDataFrame(segmentos[['segmento_id', 'bloco']], geometry=buffer),
                         segmentos[['segmento_id', 'bloco', 'geometry']], predicate='intersects')
    fronteira = vizinhos[vizinhos['bloco_left'] != vizinhos['bloco_right']]['segmento_id_left'].nunique()
    print(f"Segmentos encostados num segmento de outro bloco: {fronteira} de {len(segmentos)} "
          f"({fronteira / len(segmentos) * 100:.0f}%)")

    # Gráfico 14: mapa dos blocos e segmentos por bloco
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    # Cores por bloco em ciclo da tab20 (o colormap categórico satura com mais de 20 blocos)
    codigo = segmentos['bloco'].astype('category').cat.codes
    segmentos.plot(color=[plt.cm.tab20(c % 20) for c in codigo], lw=2.5, ax=axes[0])
    x0, x1 = axes[0].get_xlim()
    y0, y1 = axes[0].get_ylim()
    for gx in np.arange(x0 // GRADE_DIAGNOSTICO_M * GRADE_DIAGNOSTICO_M, x1, GRADE_DIAGNOSTICO_M):
        axes[0].axvline(gx, color='lightgray', lw=0.5)
    for gy in np.arange(y0 // GRADE_DIAGNOSTICO_M * GRADE_DIAGNOSTICO_M, y1, GRADE_DIAGNOSTICO_M):
        axes[0].axhline(gy, color='lightgray', lw=0.5)
    axes[0].set_title(f'Segmentos Coloridos pelo Bloco de {GRADE_DIAGNOSTICO_M / 1000:.0f} km')
    axes[0].set_xlabel('X (UTM)')
    axes[0].set_ylabel('Y (UTM)')
    sns.countplot(x=por_bloco.values, color='steelblue', ax=axes[1])
    axes[1].set_title('Segmentos por Bloco')
    axes[1].set_xlabel('Segmentos no bloco')
    axes[1].set_ylabel('Blocos')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '14_blocos_espaciais.png', dpi=300)
    plt.close()

    # =========================================================================
    # 7. ALVOS DO BINÁRIO E DA REGRESSÃO (Etapa 3, parte dos alvos)
    # =========================================================================
    print("\n" + "=" * 75)
    print("7. ALVOS: HOTSPOT E n POR SEGMENTO")
    print("=" * 75)
    painel['cal'] = painel['mes'].str[5:].astype(int)
    painel['estacao'] = np.where(painel['cal'].between(5, 9), 'seca', 'chuvosa')
    linhas = []
    for limiar in [config.HOTSPOT_MIN_N, config.HOTSPOT_MIN_N_SENSITIVITY]:
        pos = painel['n'] >= limiar
        for conjunto, mask in [('treino', painel['treino']), ('teste', ~painel['treino'])]:
            for est in ['chuvosa', 'seca', 'todas']:
                m = mask if est == 'todas' else mask & (painel['estacao'] == est)
                linhas.append({'criterio': f'n >= {limiar}', 'conjunto': conjunto, 'estacao': est,
                               'pct_positivos': round(pos[m].mean() * 100, 1)})
    hotspot = pd.DataFrame(linhas)
    hotspot.to_csv(config.TABLES_DIR / 'proporcao_hotspot.csv', index=False)
    print(hotspot.pivot(index=['criterio', 'conjunto'], columns='estacao', values='pct_positivos'))
    total_seg = painel.groupby('segmento_id')['n'].sum()
    segmentos = segmentos.merge(total_seg.rename('n_total'), left_on='segmento_id', right_index=True)

    # Gráfico 15: % de positivos por critério e estação, e mapa dos segmentos pela contagem total
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1, 1.6]})
    barras = hotspot[(hotspot['conjunto'] == 'treino') & (hotspot['estacao'] != 'todas')]
    sns.barplot(data=barras, x='criterio', y='pct_positivos', hue='estacao', palette=['steelblue', 'khaki'], ax=axes[0])
    axes[0].set_title('% de Segmento × Mês Positivos no Treino')
    axes[0].set_ylabel('% positivos')
    axes[0].set_xlabel('')
    segmentos.plot(column='n_total', cmap='YlOrRd', lw=3, legend=True, ax=axes[1],
                   legend_kwds={'label': 'atropelamentos (2010–2015)', 'shrink': 0.7})
    axes[1].set_title('Atropelamentos Silvestres por Segmento de 500 m')
    axes[1].set_xlabel('X (UTM)')
    axes[1].set_ylabel('Y (UTM)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '15_hotspot_e_mapa_segmentos.png', dpi=300)
    plt.close()

    print(f"\n[OK] Gráficos 08–15 salvos em: {OUTPUT_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    run_review_eda()
