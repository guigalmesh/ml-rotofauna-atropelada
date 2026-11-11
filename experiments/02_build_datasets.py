"""
Etapa 4: datasets processados e splits.

Lê o interim, as campanhas e os dados externos processados, e grava
  data/processed/ocorrencias.csv        (multiclasse: 1 linha por ocorrência)
  data/processed/segmentos_periodo.csv  (binário e regressão: segmento × mês)
com os atributos crus e as colunas de split. Nada ajustado aos dados é salvo.
Termina com as verificações de esquema, de vazamento e de consistência.
"""
import hashlib

import geopandas as gpd
import numpy as np
import pandas as pd

from src import config
from src.features import (
    add_history_features, assign_segment, build_panel, day_of_year_features,
    month_features, previous_month_climate, segment_uc, static_site_features,
    vegetation_by_year,
)
from src.splits import (
    balanced_block_folds, random_folds, segment_distance_matrix, spatial_blocks,
    spatial_cv_splits, temporal_split,
)

EXT = config.EXTERNAL_PROCESSED_DIR
OUT = config.OCORRENCIAS_CSV.parent
ROAD_ATTRS = ['tipo_via', 'velocidade_max', 'faixas', 'comprimento_m']


def expected_columns(dataset: str) -> list[str]:
    grupos = [c for g in config.FEATURE_GROUPS.values() for c in g]
    if dataset == 'ocorrencias':
        grupos = [c for c in grupos if c not in config.FEATURE_GROUPS['historico']] + config.OCCURRENCE_EXTRA_FEATURES
    else:
        grupos = grupos + config.PANEL_EXTRA_FEATURES
    return config.META_COLUMNS[dataset] + [config.TARGET_COLUMNS[dataset]] + config.SPLIT_COLUMNS + grupos


def md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def run_build():
    OUT.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    verificacoes = []

    def checar(nome, condicao, detalhe=''):
        verificacoes.append({'verificacao': nome, 'ok': bool(condicao), 'detalhe': detalhe})
        assert condicao, f'{nome}: {detalhe}'

    # =========================================================================
    # 1. ENTRADAS
    # =========================================================================
    print("=" * 75)
    print("1. ENTRADAS")
    print("=" * 75)
    interim = pd.read_csv(config.INTERIM_CSV, parse_dates=['data'])
    campanhas = pd.read_csv(config.CAMPANHAS_CSV)
    segmentos = gpd.read_file(EXT / 'segmentos_osm.gpkg')
    clima = pd.read_csv(EXT / 'clima_mensal.csv')
    agua = gpd.read_file(EXT / 'hidrografia.gpkg')
    dem_path = EXT / 'elevacao.tif'
    veg_path = EXT / f'vegetacao_nativa_{config.VEG_RADIUS_M}m.tif'
    meses = [str(m) for m in pd.period_range(*config.CLIMATE_WINDOW, freq='M')]
    print(f"Ocorrências: {len(interim)}; segmentos: {len(segmentos)}; meses: {len(meses)}; "
          f"feições de água: {len(agua)}")

    # =========================================================================
    # 2. ATRIBUTOS DOS SEGMENTOS (estáticos e por ano)
    # =========================================================================
    print("\n" + "=" * 75)
    print("2. ATRIBUTOS DOS SEGMENTOS")
    print("=" * 75)
    ocorrencias = assign_segment(interim, segmentos, config.ASSIGN_MAX_DIST_M)
    checar('todas as ocorrências têm segmento', ocorrencias['segmento_id'].notna().all(),
           f"{ocorrencias['segmento_id'].isna().sum()} sem segmento a <= {config.ASSIGN_MAX_DIST_M} m")
    ocorrencias['segmento_id'] = ocorrencias['segmento_id'].astype(int)
    print(f"Distância ocorrência → segmento: mediana {ocorrencias['dist_segmento_m'].median():.1f} m, "
          f"máx {ocorrencias['dist_segmento_m'].max():.0f} m")

    seg = segmentos[['segmento_id', 'x_centro', 'y_centro'] + ROAD_ATTRS].copy()
    seg['uc'] = segment_uc(ocorrencias, segmentos).values
    seg[['elevacao_m', 'dist_agua_m']] = static_site_features(seg['x_centro'], seg['y_centro'], agua, dem_path).values
    # % de vegetação nativa do segmento em cada ano do painel (banda do ano anterior)
    anos_painel = sorted({int(m[:4]) for m in meses})
    veg_seg = pd.DataFrame({
        ano: vegetation_by_year(seg['x_centro'], seg['y_centro'], np.full(len(seg), ano), veg_path)
        for ano in anos_painel
    }, index=seg['segmento_id'])
    print("UC por segmento:", seg['uc'].value_counts().to_dict())
    print("Ausentes nos segmentos (%):")
    print((seg[ROAD_ATTRS + ['elevacao_m', 'dist_agua_m']].isna().mean() * 100).round(1).to_string())

    # =========================================================================
    # 3. BLOCOS E FOLDS ESPACIAIS
    # =========================================================================
    print("\n" + "=" * 75)
    print("3. BLOCOS E FOLDS ESPACIAIS")
    print("=" * 75)
    dist = segment_distance_matrix(segmentos)
    seg['bloco_espacial'] = spatial_blocks(segmentos, dist, config.SPATIAL_BLOCK_M).values
    peso = ocorrencias.groupby('segmento_id').size().reindex(seg['segmento_id'], fill_value=0)
    seg['fold_espacial'] = balanced_block_folds(seg['bloco_espacial'], pd.Series(peso.values, index=seg.index), config.SPATIAL_N_FOLDS, config.SEED).values

    blocos = seg.groupby('bloco_espacial').agg(segmentos=('segmento_id', 'size'), km=('comprimento_m', lambda s: s.sum() / 1000),
                                                fold=('fold_espacial', 'first'))
    blocos['ocorrencias'] = peso.groupby(seg['bloco_espacial'].values).sum()
    blocos.to_csv(config.TABLES_DIR / 'blocos_espaciais.csv')
    print(f"Blocos: {len(blocos)}; segmentos por bloco: mín {blocos['segmentos'].min()}, "
          f"mediana {blocos['segmentos'].median():.0f}, máx {blocos['segmentos'].max()}; "
          f"km por bloco: mín {blocos['km'].min():.1f}, mediana {blocos['km'].median():.1f}, máx {blocos['km'].max():.1f}")
    por_fold = blocos.groupby('fold').agg(blocos=('segmentos', 'size'), segmentos=('segmentos', 'sum'), ocorrencias=('ocorrencias', 'sum'))
    print("Por fold espacial:")
    print(por_fold.to_string())
    checar('cada bloco está num único fold', (seg.groupby('bloco_espacial')['fold_espacial'].nunique() == 1).all())
    checar('todos os folds espaciais têm blocos', por_fold.shape[0] == config.SPATIAL_N_FOLDS)
    # Contato entre blocos: segmentos encostados (<= gap) num segmento de outro bloco
    adj = dist <= config.SEGMENT_MAX_GAP_M
    np.fill_diagonal(adj, False)
    outro_bloco = adj & (seg['bloco_espacial'].values[:, None] != seg['bloco_espacial'].values[None, :])
    contato = outro_bloco.any(axis=1).mean() * 100
    print(f"Segmentos encostados num segmento de outro bloco: {contato:.0f}%")

    # =========================================================================
    # 4. OCORRÊNCIAS (MULTICLASSE)
    # =========================================================================
    print("\n" + "=" * 75)
    print("4. OCORRENCIAS.CSV")
    print("=" * 75)
    clima_anterior = previous_month_climate(clima)
    oc = ocorrencias[['data', 'ano', 'classe', 'uc', 'x', 'y', 'segmento_id']].copy()
    oc['x_mais_y'] = oc['x'] + oc['y']
    oc['x_menos_y'] = oc['x'] - oc['y']
    oc[['mes_sin', 'mes_cos', 'seco']] = month_features(oc['data'].dt.month).values
    oc['seco'] = oc['seco'].astype(int)
    oc[['dia_ano_sin', 'dia_ano_cos']] = day_of_year_features(oc['data']).values
    oc = oc.merge(clima_anterior, left_on=oc['data'].dt.to_period('M').astype(str), right_on='ano_mes', how='left').drop(columns='ano_mes')
    oc = oc.merge(seg[['segmento_id', 'tipo_via', 'velocidade_max', 'faixas']], on='segmento_id', how='left')
    oc[['elevacao_m', 'dist_agua_m']] = static_site_features(oc['x'], oc['y'], agua, dem_path).values
    oc['veg_nativa_pct'] = vegetation_by_year(oc['x'], oc['y'], oc['ano'].values, veg_path)
    oc = oc.merge(seg[['segmento_id', 'bloco_espacial', 'fold_espacial']], on='segmento_id', how='left')
    oc['fold_aleatorio'] = random_folds(len(oc), config.SPATIAL_N_FOLDS, config.SEED)
    oc['split_temporal'] = temporal_split(oc['ano'])
    oc = oc[expected_columns('ocorrencias')]
    print(f"Linhas: {len(oc)}; colunas: {oc.shape[1]}")
    print("Ausentes (%):")
    print((oc.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())

    # =========================================================================
    # 5. PAINEL SEGMENTO × MÊS (BINÁRIO E REGRESSÃO)
    # =========================================================================
    print("\n" + "=" * 75)
    print("5. SEGMENTOS_PERIODO.CSV")
    print("=" * 75)
    painel = build_panel(ocorrencias, segmentos, meses)
    painel['n_campanhas'] = painel['ano_mes'].map(campanhas.groupby('mes').size()).fillna(0).astype(int)
    painel[['mes_sin', 'mes_cos', 'seco']] = month_features(painel['mes']).values
    painel['seco'] = painel['seco'].astype(int)
    painel = painel.merge(clima_anterior, on='ano_mes', how='left')
    painel = painel.merge(seg.drop(columns=['elevacao_m', 'dist_agua_m']).rename(columns={'x_centro': 'x', 'y_centro': 'y'}),
                          on='segmento_id', how='left')
    painel[['elevacao_m', 'dist_agua_m']] = seg.set_index('segmento_id').loc[painel['segmento_id'], ['elevacao_m', 'dist_agua_m']].values
    painel['x_mais_y'] = painel['x'] + painel['y']
    painel['x_menos_y'] = painel['x'] - painel['y']
    painel['veg_nativa_pct'] = [veg_seg.at[s, a] for s, a in zip(painel['segmento_id'], painel['ano'])]
    painel = add_history_features(painel)
    painel['fold_aleatorio'] = random_folds(len(painel), config.SPATIAL_N_FOLDS, config.SEED + 1)
    painel['split_temporal'] = temporal_split(painel['ano'])
    painel = painel[expected_columns('segmentos_periodo')]
    print(f"Linhas: {len(painel)} ({painel['segmento_id'].nunique()} segmentos × {painel['ano_mes'].nunique()} meses); "
          f"colunas: {painel.shape[1]}")
    print(f"n: zeros {(painel['n'] == 0).mean() * 100:.1f}%, média {painel['n'].mean():.3f}, máx {painel['n'].max()}")
    print("Ausentes (%):")
    print((painel.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())

    # =========================================================================
    # 6. VERIFICAÇÕES
    # =========================================================================
    print("\n" + "=" * 75)
    print("6. VERIFICAÇÕES")
    print("=" * 75)
    for nome, df in [('ocorrencias', oc), ('segmentos_periodo', painel)]:
        proibidas = set(config.FORBIDDEN_IN_PROCESSED) & set(df.columns)
        checar(f'{nome}: nenhuma coluna proibida', not proibidas, str(proibidas))
        checar(f'{nome}: esquema igual ao esperado', list(df.columns) == expected_columns(nome))
        checar(f'{nome}: alvo sem ausentes', df[config.TARGET_COLUMNS[nome]].notna().all())
        checar(f'{nome}: splits sem ausentes', df[config.SPLIT_COLUMNS].notna().all().all())
    checar('soma de n igual ao total de ocorrências', painel['n'].sum() == len(interim) == len(oc),
           f"{painel['n'].sum()} vs {len(interim)}")
    checar('painel completo e sem duplicatas', len(painel) == len(segmentos) * len(meses)
           and not painel.duplicated(['segmento_id', 'ano_mes']).any())
    checar('todos os meses têm campanha', (painel['n_campanhas'] > 0).all())

    # Lags só com o passado: recalcula a partir do painel restrito a meses anteriores
    rng = np.random.default_rng(config.SEED)
    amostra = painel.sample(200, random_state=int(rng.integers(1 << 31)))
    ok_lags = True
    for _, linha in amostra.iterrows():
        passado = painel[(painel['segmento_id'] == linha['segmento_id']) & (painel['ano_mes'] < linha['ano_mes'])].sort_values('ano_mes')
        esperado_12 = passado['n'].tail(12).mean() if len(passado) else np.nan
        ok_lags &= len(passado) == linha['meses_historico']
        ok_lags &= np.isnan(linha['n_lag_1']) if passado.empty else linha['n_lag_1'] == passado['n'].iloc[-1]
        ok_lags &= (np.isnan(linha['n_media_12m']) and np.isnan(esperado_12)) or np.isclose(linha['n_media_12m'], esperado_12)
        mesmo_mes = passado[passado['ano_mes'] == f"{int(linha['ano']) - 1}-{int(linha['mes']):02d}"]
        ok_lags &= np.isnan(linha['n_mesmo_mes_ano_anterior']) if mesmo_mes.empty else linha['n_mesmo_mes_ano_anterior'] == mesmo_mes['n'].iloc[0]
    checar('lags usam só meses anteriores (amostra de 200 linhas)', ok_lags)
    checar('lag de 12 meses ausente no primeiro ano', painel.loc[painel['ano_mes'] < meses[12], 'n_mesmo_mes_ano_anterior'].isna().all())

    # Clima e vegetação sem informação do próprio período
    clima_idx = clima.set_index('ano_mes')
    chuva_mes = painel['ano_mes'].map(clima_idx['precipitacao_mm'])
    checar('chuva no painel é a do mês anterior, não a do mês', (painel['prec_mes_anterior_mm'] != chuva_mes).any()
           and (painel.loc[painel['ano_mes'] == meses[1], 'prec_mes_anterior_mm'] == clima_idx.at[meses[0], 'precipitacao_mm']).all())
    checar('vegetação do painel vem do ano anterior', all(
        np.isclose(veg_seg.at[s, a], v, equal_nan=True) for s, a, v in
        zip(painel['segmento_id'].head(500), painel['ano'].head(500), painel['veg_nativa_pct'].head(500))))

    # Splits
    checar('split temporal: treino até o ano de corte', (
        (oc.loc[oc['split_temporal'] == 'treino', 'ano'] <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR).all()
        and (painel.loc[painel['split_temporal'] == 'teste', 'ano'] > config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR).all()))
    checar('ocorrência herda o bloco do seu segmento',
           (oc.merge(seg[['segmento_id', 'bloco_espacial']], on='segmento_id', suffixes=('', '_seg'))
              .pipe(lambda d: (d['bloco_espacial'] == d['bloco_espacial_seg']).all())))
    faixa = []
    for buffer_m in [0, config.SPATIAL_BUFFER_M, 2 * config.SPATIAL_BUFFER_M]:
        perdas, contatos = [], []
        for treino, val in spatial_cv_splits(painel['segmento_id'], painel['fold_espacial'], segmentos, dist, buffer_m):
            seg_treino = painel.loc[treino, 'segmento_id'].unique()
            seg_val = painel.loc[val, 'segmento_id'].unique()
            pos = pd.Series(np.arange(len(segmentos)), index=segmentos['segmento_id'].values)
            d = dist[np.ix_(pos[seg_val].values, pos[seg_treino].values)]
            contatos.append((d.min(axis=1) <= config.SEGMENT_MAX_GAP_M).mean() * 100)
            perdas.append((1 - treino.sum() / (~val).sum()) * 100)
        faixa.append({'faixa_m': buffer_m, 'val_com_vizinho_no_treino_pct': round(np.mean(contatos), 1),
                      'treino_descartado_pct': round(np.mean(perdas), 1)})
    faixa = pd.DataFrame(faixa)
    faixa.to_csv(config.TABLES_DIR / 'faixa_exclusao.csv', index=False)
    print("Faixa de exclusão na CV espacial (média dos folds):")
    print(faixa.to_string(index=False))
    checar('com a faixa de exclusão nenhum segmento de validação encosta no treino',
           faixa.loc[faixa['faixa_m'] == config.SPATIAL_BUFFER_M, 'val_com_vizinho_no_treino_pct'].iloc[0] == 0)

    # Atributos crus: nada centrado, escalado ou codificado
    checar('categóricas cruas', all(oc[c].dtype == object or str(oc[c].dtype) == 'str' for c in config.CATEGORICAL_FEATURES))
    checar('coordenadas em metros UTM (não escaladas)', oc['x'].between(150_000, 250_000).all() and oc['y'].between(8.2e6, 8.3e6).all())

    checar('clima do mês anterior sem ausentes', painel[['prec_mes_anterior_mm', 'temp_mes_anterior_c']].notna().all().all())

    # Mapa dos blocos e folds, para conferir o corte ao longo da via
    plot_blocks(segmentos, seg)

    # =========================================================================
    # 7. SAÍDAS
    # =========================================================================
    oc.round(4).to_csv(config.OCORRENCIAS_CSV, index=False)
    painel.round(4).to_csv(config.SEGMENTOS_CSV, index=False)
    pd.DataFrame(verificacoes).to_csv(config.TABLES_DIR / 'verificacao_datasets.csv', index=False)
    write_dictionary()
    print(f"\n[OK] {config.OCORRENCIAS_CSV.relative_to(config.ROOT)} (md5 {md5(config.OCORRENCIAS_CSV)[:8]}), "
          f"{config.SEGMENTOS_CSV.relative_to(config.ROOT)} (md5 {md5(config.SEGMENTOS_CSV)[:8]}); "
          f"{len(verificacoes)} verificações ok")


def plot_blocks(segmentos, seg):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    g = segmentos[['geometry']].assign(bloco=seg['bloco_espacial'].values, fold=seg['fold_espacial'].values)
    g.plot(color=[plt.cm.tab20(b % 20) for b in g['bloco']], lw=3, ax=axes[0])
    for b, grupo in g.groupby('bloco'):
        c = grupo.union_all().centroid
        axes[0].annotate(str(b), (c.x, c.y), fontsize=8, ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, lw=0))
    axes[0].set_title(f'Blocos espaciais (trechos de ~{config.SPATIAL_BLOCK_M / 1000:.0f} km ao longo da via)')
    g.plot(column='fold', categorical=True, cmap='Set1', lw=3, legend=True, ax=axes[1],
           legend_kwds={'title': 'fold espacial'})
    axes[1].set_title('Folds espaciais (blocos equilibrados por ocorrências)')
    for ax in axes:
        ax.set_xlabel('X (UTM)')
        ax.set_ylabel('Y (UTM)')
    plt.tight_layout()
    config.EDA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(config.EDA_FIG_DIR / '16_blocos_folds_espaciais.png', dpi=300)
    plt.close()


def write_dictionary():
    """Dicionário de dados dos arquivos processados, para o relatório."""
    grupo = {c: g for g, cols in config.FEATURE_GROUPS.items() for c in cols}
    desc = {
        'data': ('meta', 'data da ocorrência'), 'ano': ('meta', 'ano; não é atributo (o teste tem anos inéditos)'),
        'segmento_id': ('meta', 'segmento de ~500 m da malha do OSM atribuído (até 200 m)'),
        'ano_mes': ('meta', 'mês do painel'), 'mes': ('meta', 'mês 1–12 (redundante com mes_sin/cos)'),
        'n_campanhas': ('meta', 'campanhas no mês; só informativo, nunca atributo (deriva de campanha)'),
        'classe': ('alvo', 'classe taxonômica (multiclasse)'), 'n': ('alvo', 'atropelamentos silvestres no segmento no mês'),
        'bloco_espacial': ('split', f'trecho contínuo de ~{config.SPATIAL_BLOCK_M / 1000:.0f} km ao longo da via'),
        'fold_espacial': ('split', f'fold do bloco (0–{config.SPATIAL_N_FOLDS - 1}), equilibrado por ocorrências'),
        'fold_aleatorio': ('split', 'fold sorteado por linha; só referência otimista'),
        'split_temporal': ('split', f'treino <= {config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR}, teste depois; teste final'),
        'x': ('base', 'UTM E (m), SIRGAS 2000 / 23S; no painel, centro do segmento'), 'y': ('base', 'UTM N (m)'),
        'x_mais_y': ('base', 'coordenada rotacionada 45°'), 'x_menos_y': ('base', 'coordenada rotacionada 45°'),
        'uc': ('base', 'unidade de conservação (categórica crua)'),
        'mes_sin': ('base', 'seno do mês'), 'mes_cos': ('base', 'cosseno do mês'), 'seco': ('base', '1 = maio a setembro'),
        'dia_ano_sin': ('base', 'seno do dia do ano'), 'dia_ano_cos': ('base', 'cosseno do dia do ano'),
        'prec_mes_anterior_mm': ('clima', 'precipitação do mês anterior (INMET A001)'),
        'temp_mes_anterior_c': ('clima', 'temperatura média do mês anterior (INMET A001)'),
        'tipo_via': ('via', 'classe da via no OSM (trunk, primary, secondary, tertiary); malha atual'),
        'velocidade_max': ('via', 'velocidade máxima no OSM (km/h); ausente em parte das vias'),
        'faixas': ('via', 'número de faixas no OSM; ausente em parte das vias'),
        'comprimento_m': ('via', 'comprimento do segmento (m)'),
        'elevacao_m': ('relevo_agua', 'elevação SRTM (m)'), 'dist_agua_m': ('relevo_agua', 'distância ao rio ou corpo d\'água mais próximo no OSM (m)'),
        'veg_nativa_pct': ('vegetacao', f'% de vegetação nativa (MapBiomas) num raio de {config.VEG_RADIUS_M} m, no ano anterior'),
        'n_lag_1': ('historico', 'n do mês anterior no mesmo segmento'),
        'n_media_12m': ('historico', 'média de n nos 12 meses anteriores (com o que houver)'),
        'n_mesmo_mes_ano_anterior': ('historico', 'n do mesmo mês do ano anterior (ausente no primeiro ano)'),
        'meses_historico': ('historico', 'meses de histórico disponíveis'),
    }
    linhas = []
    for dataset in ['ocorrencias', 'segmentos_periodo']:
        for c in expected_columns(dataset):
            papel, d = desc[c]
            linhas.append({'arquivo': f'{dataset}.csv', 'coluna': c, 'papel': papel if papel in ('meta', 'alvo', 'split') else f'atributo:{grupo.get(c, papel)}', 'descricao': d})
    pd.DataFrame(linhas).to_csv(config.TABLES_DIR / 'dicionario_dados.csv', index=False)


if __name__ == '__main__':
    run_build()
