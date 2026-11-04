import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.preprocessing import generate_quality_table

# Configuração visual e diretório de saída para os gráficos do relatório
OUTPUT_DIR = config.EDA_FIG_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style='whitegrid')


def add_temporal_columns_for_eda(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai mês numérico e período climático (seco vs chuvoso no DF)
    para as análises temporais da EDA.
    """
    df_temp = df.copy()
    df_temp['mes_num'] = df_temp['data'].dt.month
    df_temp['periodo_climatico'] = df_temp['mes_num'].apply(
        lambda m: 'seco' if m in [5, 6, 7, 8, 9] else 'chuvoso'
    )
    return df_temp


def run_eda_pipeline():
    # =========================================================================
    # RESUMO DA LIMPEZA (gerado pelo 01a_limpeza.py)
    # =========================================================================
    print("=" * 75)
    print("1. RESUMO DA LIMPEZA (RAW → INTERIM)")
    print("=" * 75)
    decisoes = pd.read_csv(config.DECISOES_LIMPEZA_CSV).set_index('etapa')
    print(decisoes[['linhas_antes', 'linhas_depois']])

    # Gráfico 1: Silvestres vs Domésticos (contagens da etapa de filtragem)
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
    print(f"Atributos: {len(qualidade)}")
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
    # ANÁLISE DO DATASET LIMPO (APENAS FAUNA SILVESTRE)
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. ANÁLISE SOBRE O DATASET LIMPO (APENAS FAUNA SILVESTRE)")
    print("=" * 75)
    df_silvestre = pd.read_csv(config.INTERIM_CSV, parse_dates=['data'])
    df_silvestre = add_temporal_columns_for_eda(df_silvestre)
    print(f"Total de registros silvestres finais: {len(df_silvestre)}\n")

    print("--- 2.1. Qualidade dos Dados (Apenas Silvestres) ---")
    print(generate_quality_table(df_silvestre))

    print("\n--- 2.2. Estatísticas Descritivas Numéricas - describe() ---")
    num_cols = ['campanha', 'ano', 'x', 'y', 'mes_num']
    print(df_silvestre[num_cols].describe().T.round(2))

    print("\n--- 2.3. Estatísticas Descritivas Categóricas - describe() ---")
    resumo_cat = df_silvestre.select_dtypes(include=['object', 'str']).describe().T
    resumo_cat['% top'] = (resumo_cat['freq'] / resumo_cat['count'] * 100).round(2)
    print(resumo_cat)

    print("\n--- 2.4. Distribuição da Variável Alvo ('classe') e Desbalanceamento ---")
    c_classe = df_silvestre['classe'].value_counts()
    p_classe = (df_silvestre['classe'].value_counts(normalize=True) * 100).round(2)
    print(pd.DataFrame({'Quantidade': c_classe, '%': p_classe}))

    especies_counts = df_silvestre['especie_nm'].value_counts()
    print(f"\nTotal de espécies silvestres únicas: {len(especies_counts)}")
    print(f"Espécies raras (< 10 ocorrências):   {(especies_counts < 10).sum()} espécies")

    # =========================================================================
    # GERAÇÃO DE VISUALIZAÇÕES GRÁFICAS (APENAS SILVESTRES)
    # =========================================================================
    # Gráfico 2: Distribuição por Classe e Top 15 Espécies
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ordem_cls = df_silvestre['classe'].value_counts().index
    sns.countplot(data=df_silvestre, y='classe', order=ordem_cls, hue='classe', ax=axes[0], palette='viridis', legend=False)
    axes[0].set_title('Distribuição por Classe Taxonômica (Silvestres)')
    axes[0].set_xlabel('Quantidade')

    top15_esp = df_silvestre[df_silvestre['especie_nm'] != 'nao_identificado']['especie_nm'].value_counts().head(15)
    sns.barplot(x=top15_esp.values, y=top15_esp.index, hue=top15_esp.index, ax=axes[1], palette='magma', legend=False)
    axes[1].set_title('Top 15 Espécies Silvestres Identificadas')
    axes[1].set_xlabel('Quantidade')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_distribuicao_classes_especies.png', dpi=300)
    plt.close()

    # Gráfico 3: Boxplots das Coordenadas UTM (Verificação de Outliers)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=df_silvestre['x'], ax=axes[0], color='skyblue').set_title('Boxplot Coordenada X (UTM)')
    sns.boxplot(x=df_silvestre['y'], ax=axes[1], color='lightgreen').set_title('Boxplot Coordenada Y (UTM)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_boxplots_coordenadas_outliers.png', dpi=300)
    plt.close()

    # Gráfico 4: Distribuição Espacial (Mapa de Dispersão X vs Y)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_silvestre, x='x', y='y', hue='classe',
        alpha=0.6, s=20, palette='tab10'
    )
    plt.title('Distribuição Espacial dos Atropelamentos de Fauna Silvestre (UTM)')
    plt.xlabel('Coordenada X (Leste-Oeste)')
    plt.ylabel('Coordenada Y (Norte-Sul)')
    plt.legend(title='Classe', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_distribuicao_espacial_utm.png', dpi=300)
    plt.close()

    # Gráfico 5: Distribuição Temporal (Ano, Mês e Dia da Semana)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.countplot(data=df_silvestre, x='ano', ax=axes[0], color='steelblue').set_title('Ocorrências por Ano')
    sns.countplot(data=df_silvestre, x='mes_num', ax=axes[1], color='coral').set_title('Ocorrências por Mês (1-12)')
    ordem_dias = [d for d in ['seg', 'ter', 'qua', 'qui', 'sex', 'sab', 'dom'] if d in df_silvestre['dia_da_sem'].unique()]
    sns.countplot(data=df_silvestre, x='dia_da_sem', order=ordem_dias, ax=axes[2], color='teal').set_title('Ocorrências por Dia da Semana')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_distribuicao_temporal.png', dpi=300)
    plt.close()

    # Gráfico 6: Matriz de Correlação (Investigação de Atributos Redundantes)
    plt.figure(figsize=(7, 5))
    corr_cols = ['x', 'y', 'ano', 'mes_num', 'campanha']
    sns.heatmap(df_silvestre[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação das Variáveis Numéricas')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_matriz_correlacao.png', dpi=300)
    plt.close()

    print(f"\n[OK] Todos os 7 gráficos foram gerados e salvos em: {OUTPUT_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    run_eda_pipeline()
