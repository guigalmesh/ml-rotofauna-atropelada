import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency

from src import config
from src.preprocessing import generate_quality_table

# EDA das ocorrências (Etapas 1 e 3). Os gráficos 08–14 (esforço, zeros, classes
# por split, sazonalidade, painel e blocos) ficam no 01d_eda_revisao.py
OUTPUT_DIR = config.EDA_FIG_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')

MESES = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
CLASSES = ['birds', 'reptiles', 'mammals', 'amphibians']


def add_temporal_columns_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Mês numérico e estação (seca = maio a setembro no DF) para as análises temporais."""
    df_temp = df.copy()
    df_temp['mes_num'] = df_temp['data'].dt.month
    df_temp['estacao'] = np.where(df_temp['mes_num'].between(5, 9), 'seca', 'chuvosa')
    return df_temp


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Cramér's V (associação entre duas categóricas) e p-valor do qui-quadrado."""
    tabela = pd.crosstab(a, b)
    chi2, p, _, _ = chi2_contingency(tabela)
    n = tabela.values.sum()
    k = min(tabela.shape) - 1
    return float(np.sqrt(chi2 / (n * k))), float(p)


def run_eda_pipeline():
    # =========================================================================
    # 1. RESUMO DA LIMPEZA (gerado pelo 01a_limpeza.py)
    # =========================================================================
    print("=" * 75)
    print("1. RESUMO DA LIMPEZA (RAW → INTERIM)")
    print("=" * 75)
    decisoes = pd.read_csv(config.DECISOES_LIMPEZA_CSV).set_index('etapa')
    print(decisoes[['linhas_antes', 'linhas_depois']])

    # Gráfico 1: silvestres × domésticos (contagens da etapa de filtragem)
    filtro = decisoes.loc['filtrar_silvestres']
    proporcao = pd.Series({
        'silvestre': filtro['linhas_depois'],
        'doméstico': filtro['linhas_antes'] - filtro['linhas_depois'],
    })
    print("\n--- 1.1. Proporção Fauna Silvestre vs. Doméstica ---")
    print(pd.DataFrame({'Quantidade': proporcao, '%': (proporcao / proporcao.sum() * 100).round(2)}))

    plt.figure(figsize=(6, 4))
    sns.barplot(x=proporcao.index, y=proporcao.values, hue=proporcao.index, palette='Set2', legend=False)
    plt.title('Proporção de Registros: Silvestres vs. Domésticos\n(Base Sem Duplicatas)')
    plt.xlabel('Categoria')
    plt.ylabel('Quantidade de Ocorrências')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_proporcao_silvestre_domestico.png', dpi=300)
    plt.close()

    # Qualidade da base original sem duplicatas (todas as colunas, silvestres + domésticos)
    qualidade = pd.read_csv(config.QUALIDADE_ORIGINAL_CSV, index_col='coluna')
    print("\n--- 1.2. Qualidade da Base Original (5.164 registros, todas as colunas) ---")
    print(qualidade)

    # Gráfico 7: % de 'nao_identificado' por coluna (justifica os descartes)
    nao_id = qualidade['% nao_identificado'].sort_values()
    nao_id = nao_id[nao_id > 0]
    plt.figure(figsize=(7, 4))
    nao_id.plot.barh(color='indianred')
    plt.title("% de 'nao_identificado' por Coluna\n(Base Original Sem Duplicatas)")
    plt.xlabel('% dos registros')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '07_nao_identificado_base_original.png', dpi=300)
    plt.close()

    # =========================================================================
    # 2. DATASET LIMPO (APENAS FAUNA SILVESTRE)
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. ANÁLISE SOBRE O DATASET LIMPO (APENAS FAUNA SILVESTRE)")
    print("=" * 75)
    df = pd.read_csv(config.INTERIM_CSV, parse_dates=['data'])
    df = add_temporal_columns_for_eda(df)
    campanhas = pd.read_csv(config.CAMPANHAS_CSV, parse_dates=['data'])
    print(f"Total de registros silvestres finais: {len(df)}\n")

    print("--- 2.1. Qualidade dos Dados (Apenas Silvestres) ---")
    print(generate_quality_table(df))

    print("\n--- 2.2. Estatísticas Descritivas Numéricas ---")
    print(df[['ano', 'x', 'y', 'mes_num']].describe().T.round(2))

    print("\n--- 2.3. Distribuição da Variável Alvo ('classe') ---")
    c_classe = df['classe'].value_counts()
    print(pd.DataFrame({'Quantidade': c_classe, '%': (c_classe / len(df) * 100).round(2)}))
    print(f"Razão majoritária / minoritária: {c_classe.max() / c_classe.min():.1f}×")

    especies = df[df['especie_nm'] != 'nao_identificado']['especie_nm'].value_counts()
    print(f"\nEspécies identificadas: {len(especies)}; raras (< 10 registros): {(especies < 10).sum()}")
    print(f"Sem espécie: {(df['especie_nm'] == 'nao_identificado').mean() * 100:.1f}% dos registros")
    volatinia = df['especie_nm'] == 'volatinia_jacarina'
    print(f"Volatinia jacarina: {volatinia.sum()} registros ({volatinia.mean() * 100:.1f}% dos silvestres, "
          f"{volatinia.sum() / (df['classe'] == 'birds').sum() * 100:.1f}% das aves)")

    # Gráfico 2: classe, top 15 espécies e participação da Volatinia por mês
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.countplot(data=df, y='classe', order=c_classe.index, hue='classe', ax=axes[0], palette='viridis', legend=False)
    axes[0].set_title('Distribuição por Classe (Silvestres)')
    axes[0].set_xlabel('Quantidade')
    axes[0].set_ylabel('')
    top15 = especies.head(15)
    sns.barplot(x=top15.values, y=top15.index, hue=top15.index, ax=axes[1], palette='magma', legend=False)
    axes[1].set_title('Top 15 Espécies Identificadas')
    axes[1].set_xlabel('Quantidade')
    axes[1].set_ylabel('')
    vol_mes = df.groupby('mes_num').apply(lambda g: (g['especie_nm'] == 'volatinia_jacarina').mean() * 100)
    axes[2].bar(vol_mes.index, vol_mes.values, color='slategray')
    axes[2].set_xticks(range(1, 13))
    axes[2].set_xticklabels(MESES)
    axes[2].set_title('Participação da Volatinia jacarina\nnos Registros do Mês')
    axes[2].set_ylabel('% dos registros silvestres')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_distribuicao_classes_especies.png', dpi=300)
    plt.close()

    # =========================================================================
    # 3. ESPACIAL
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. ESPACIAL")
    print("=" * 75)
    # Distância ao registro mais próximo (argumento para o split espacial)
    xy = df[['x', 'y']].values
    dist, idx = cKDTree(xy).query(xy, k=2)
    vizinho = pd.Series(dist[:, 1])
    print("Distância ao registro mais próximo (m):")
    print(vizinho.describe(percentiles=[.25, .5, .75, .9]).round(1))
    mesma_data = (df['data'].values == df['data'].values[idx[:, 1]]) & (vizinho == 0)
    mesmo_ponto = vizinho == 0
    print(f"Registros com outro no mesmo ponto exato: {mesmo_ponto.sum()}; destes, na mesma data: {mesma_data.sum()}")
    repetidos = df[mesma_data.values]
    pares = df.loc[idx[mesma_data.values, 1]]
    mesma_especie = (repetidos['especie_nm'].values == pares['especie_nm'].values)
    print(f"  mesma data e mesma espécie: {mesma_especie.sum()} (animais distintos no mesmo ponto, "
          f"já sem os duplicados exatos removidos na limpeza); espécies diferentes: {(~mesma_especie).sum()}")

    # Gráfico 3: distância ao vizinho mais próximo (substitui os boxplots de x e y)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(vizinho.clip(upper=500), bins=50, color='steelblue', ax=axes[0])
    axes[0].set_title('Distância ao Registro Mais Próximo (até 500 m)')
    axes[0].set_xlabel('metros')
    axes[0].axvline(vizinho.median(), color='red', ls='--', label=f'mediana {vizinho.median():.0f} m')
    axes[0].legend()
    ecdf = vizinho.sort_values().reset_index(drop=True)
    axes[1].plot(ecdf.values, np.arange(1, len(ecdf) + 1) / len(ecdf) * 100, color='steelblue')
    axes[1].set_xscale('log')
    axes[1].set_title('Distribuição Acumulada')
    axes[1].set_xlabel('metros (escala log)')
    axes[1].set_ylabel('% dos registros')
    for d in [50, 500, 5000]:
        axes[1].axvline(d, color='gray', lw=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_distancia_vizinho_mais_proximo.png', dpi=300)
    plt.close()

    # Composição de classes por UC, com qui-quadrado
    por_uc = pd.crosstab(df['uc'], df['classe'])[CLASSES]
    v_uc, p_uc = cramers_v(df['uc'], df['classe'])
    print("\nClasses por UC (%):")
    print((por_uc.div(por_uc.sum(axis=1), axis=0) * 100).round(1))
    print(f"Qui-quadrado classe × uc: p = {p_uc:.2e}; Cramér's V = {v_uc:.3f}")

    # Gráfico 4: mapa de densidade (hexágonos de ~1 km) e composição por UC
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.6, 1]})
    hb = axes[0].hexbin(df['x'], df['y'], gridsize=60, bins='log', cmap='YlOrRd', mincnt=1)
    axes[0].set_aspect('equal')
    axes[0].set_title('Densidade de Atropelamentos Silvestres\n(hexágonos de ~1 km, escala log)')
    axes[0].set_xlabel('X (UTM)')
    axes[0].set_ylabel('Y (UTM)')
    fig.colorbar(hb, ax=axes[0], label='registros')
    (por_uc.div(por_uc.sum(axis=1), axis=0) * 100).plot.bar(stacked=True, ax=axes[1], colormap='viridis', rot=0)
    axes[1].set_title(f"Composição de Classes por UC\n(Cramér's V = {v_uc:.2f})")
    axes[1].set_ylabel('% dos registros da UC')
    axes[1].set_xlabel('')
    axes[1].legend(title='classe', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_densidade_espacial_classes_uc.png', dpi=300)
    plt.close()

    # =========================================================================
    # 4. TEMPORAL
    # =========================================================================
    print("\n" + "=" * 75)
    print("4. TEMPORAL")
    print("=" * 75)
    ano_mes = pd.crosstab(df['ano'], df['mes_num']).reindex(columns=range(1, 13), fill_value=0)
    ano_mes.columns = MESES
    print("Registros por ano × mês (2010 começa em abril, 2015 termina em março):")
    print(ano_mes)
    esforco_ano = campanhas.groupby(campanhas['data'].dt.year).size()
    por_ano = df.groupby('ano').size()
    taxa_ano = (por_ano / esforco_ano).round(2)
    print("\nRegistros por campanha, por ano:")
    print(pd.DataFrame({'registros': por_ano, 'campanhas': esforco_ano, 'por_campanha': taxa_ano}))
    print("\nDia da semana (reflete o calendário das campanhas: sem sábado e domingo):")
    print(df['dia_da_sem'].value_counts())

    # Gráfico 5: heatmap ano × mês, registros por campanha por ano e dia da semana
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), gridspec_kw={'width_ratios': [2, 1, 1]})
    sns.heatmap(ano_mes, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], cbar=False)
    axes[0].set_title('Registros por Ano × Mês')
    axes[0].set_ylabel('')
    axes[1].bar(taxa_ano.index.astype(str), taxa_ano.values, color='steelblue')
    axes[1].set_title('Registros por Campanha, por Ano')
    axes[1].set_ylabel('registros / campanha')
    for i, (ano, n_c) in enumerate(esforco_ano.items()):
        axes[1].text(i, taxa_ano[ano] + 0.2, f'{n_c} camp.', ha='center', fontsize=8)
    ordem_dias = [d for d in ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom'] if d in df['dia_da_sem'].unique()]
    sns.countplot(data=df, x='dia_da_sem', order=ordem_dias, ax=axes[2], color='teal')
    axes[2].set_title('Dia da Semana\n(calendário das campanhas, não do risco)')
    axes[2].set_xlabel('')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_distribuicao_temporal.png', dpi=300)
    plt.close()

    # =========================================================================
    # 5. RELAÇÕES COM O ALVO MULTICLASSE
    # =========================================================================
    print("\n" + "=" * 75)
    print("5. RELAÇÕES COM O ALVO MULTICLASSE")
    print("=" * 75)
    assoc = []
    for nome, serie in [('uc', df['uc']), ('mes', df['mes_num']), ('estacao', df['estacao']),
                        ('ano', df['ano']), ('dia_da_sem', df['dia_da_sem'])]:
        v, p = cramers_v(serie, df['classe'])
        assoc.append({'atributo': nome, 'cramers_v': round(v, 3), 'p_qui2': p})
    assoc = pd.DataFrame(assoc).sort_values('cramers_v', ascending=False)
    assoc.to_csv(config.TABLES_DIR / 'associacao_classe.csv', index=False)
    print("Cramér's V entre classe e cada atributo categórico:")
    print(assoc.to_string(index=False))

    classe_mes = pd.crosstab(df['mes_num'], df['classe'])[CLASSES]
    classe_mes_pct = classe_mes.div(classe_mes.sum(axis=1), axis=0) * 100
    classe_mes_pct.index = MESES
    classe_est = pd.crosstab(df['estacao'], df['classe'])[CLASSES]
    print("\nClasses por estação (%):")
    print((classe_est.div(classe_est.sum(axis=1), axis=0) * 100).round(1))

    # Gráfico 6: composição de classes por mês e registros por mês de cada classe
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1.15, 1]})
    classe_mes_pct.plot.bar(stacked=True, ax=axes[0], colormap='viridis', rot=0, width=0.85)
    axes[0].axvspan(3.5, 8.5, color='khaki', alpha=0.25)
    axes[0].set_title("Composição de Classes por Mês (faixa = estação seca)")
    axes[0].set_ylabel('% dos registros do mês')
    axes[0].set_xlabel('')
    axes[0].legend(title='classe', bbox_to_anchor=(1.0, 1), loc='upper left', fontsize=8)
    (classe_mes / classe_mes.sum()).mul(100).plot(ax=axes[1], marker='o')
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(MESES)
    axes[1].axvspan(4.5, 9.5, color='khaki', alpha=0.25)
    axes[1].set_title('Distribuição Anual de Cada Classe\n(% dos registros da classe em cada mês)')
    axes[1].set_ylabel('% da classe')
    axes[1].set_xlabel('')
    axes[1].legend(title='classe')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_classe_por_mes_estacao.png', dpi=300)
    plt.close()

    print(f"\n[OK] Gráficos 01–07 salvos em: {OUTPUT_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    run_eda_pipeline()
