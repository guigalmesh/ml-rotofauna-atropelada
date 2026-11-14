"""
Etapa 9, uso do modelo para hotspots (pergunta 9 da discussão final): aplica
os modelos de hotspot já treinados (HGB e MLP, salvos pelo 05_binario) ao
período de teste (2014–2015), ordena segmentos pelo risco previsto e mede
quanto dos atropelamentos observados os trechos de maior risco concentram,
contra o histórico do segmento no treino e contra o sorteio. Gera o mapa de
risco previsto × atropelamentos observados. Não treina nem ajusta nada.
"""
import geopandas as gpd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.mlp import load_mlp, predict_mlp, to_dense

UC_NOMES = {'jbb_fal_ibge': 'JBB / FAL / IBGE', 'esecae': 'ESECAE', 'pnb': 'PNB'}
ROTULOS = {'hgb': 'HGB', 'mlp': 'MLP', 'historico_segmento': 'Histórico do segmento', 'ideal': 'Ideal (observado)'}


def scores(df: pd.DataFrame) -> pd.DataFrame:
    """Risco previsto de cada segmento × mês do teste, por modelo."""
    treino = df['split_temporal'] == 'treino'
    teste = df[~treino].copy()
    hgb = joblib.load(config.MODELS_DIR / 'binario_hgb.joblib')
    teste['hgb'] = hgb['pipe'].predict_proba(teste[hgb['features']])[:, 1]
    model, pre, art = load_mlp(config.MODELS_DIR / 'binario_mlp.joblib')
    teste['mlp'] = predict_mlp(model, to_dense(pre.transform(teste[art['features']])), art['task'])[1]
    # Referência sem modelo: fração de meses com atropelamento no segmento, no treino
    y_treino = (df.loc[treino, 'n'] >= config.HOTSPOT_MIN_N).astype(int)
    teste['historico_segmento'] = teste['segmento_id'].map(y_treino.groupby(df.loc[treino, 'segmento_id']).mean())
    return teste


def capture_curve(score: np.ndarray, n: np.ndarray, seed: int = config.SEED):
    """Fração dos atropelamentos acumulada ao inspecionar as unidades da maior para a menor nota (empates sorteados)."""
    ordem = np.random.default_rng(seed).permutation(len(score))
    ordem = ordem[np.argsort(-score[ordem], kind='stable')]
    return np.arange(1, len(score) + 1) / len(score), np.cumsum(n[ordem]) / n.sum()


def capture_table(unidades: pd.DataFrame, nivel: str) -> pd.DataFrame:
    """Captura nas frações de config e precisão (unidades com n >= 1 entre as inspecionadas)."""
    n = unidades['n'].values
    rows = []
    for modelo in ['hgb', 'mlp', 'historico_segmento', 'ideal']:
        score = n.astype(float) if modelo == 'ideal' else unidades[modelo].values
        frac, capt = capture_curve(score, n)
        ordem = np.random.default_rng(config.SEED).permutation(len(n))
        ordem = ordem[np.argsort(-score[ordem], kind='stable')]
        for f in config.HOTSPOT_TOP_FRACTIONS:
            k = int(round(f * len(n)))
            rows.append({'nivel': nivel, 'modelo': modelo, 'fracao_inspecionada': f, 'unidades': k,
                         'atropelamentos_capturados': capt[k - 1], 'precisao': (n[ordem[:k]] >= 1).mean(),
                         'ganho_sobre_sorteio': capt[k - 1] / f})
    return pd.DataFrame(rows)


def plot_capture(teste: pd.DataFrame, por_segmento: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, unidades, titulo in [(axes[0], teste, f'Segmento × mês ({len(teste)} unidades)'),
                                 (axes[1], por_segmento, f'Segmento, risco médio no período ({len(por_segmento)} segmentos)')]:
        n = unidades['n'].values
        for modelo, estilo in [('hgb', '-'), ('mlp', '-'), ('historico_segmento', '--')]:
            frac, capt = capture_curve(unidades[modelo].values, n)
            ax.plot(frac * 100, capt * 100, estilo, label=ROTULOS[modelo])
        frac, capt = capture_curve(n.astype(float), n)
        ax.plot(frac * 100, capt * 100, ':', color='gray', label=ROTULOS['ideal'])
        ax.plot([0, 100], [0, 100], color='black', lw=0.8, label='Sorteio')
        ax.axvline(config.HOTSPOT_MAP_FRACTION * 100, color='gray', lw=0.6)
        ax.set(title=titulo, xlabel='% das unidades inspecionadas, da maior para a menor nota',
               ylabel='% dos atropelamentos do teste', xlim=(0, 100), ylim=(0, 100))
        ax.grid(alpha=0.3)
    axes[0].legend(loc='lower right')
    fig.suptitle('Hotspots: concentração dos atropelamentos de 2014–2015 nos trechos de maior risco previsto')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(geo: gpd.GeoDataFrame, path):
    """Uma linha por UC: risco previsto (HGB) à esquerda, atropelamentos observados à direita; contorno preto = top previsto."""
    ucs = [u for u in UC_NOMES if u in set(geo['uc'])]
    fig, axes = plt.subplots(len(ucs), 2, figsize=(13, 4.3 * len(ucs)), constrained_layout=True)
    vmax_risco, vmax_n = geo['hgb'].max(), geo['n'].max()
    for linha, uc in zip(axes, ucs):
        sub = geo[geo['uc'] == uc]
        top = sub[sub['top_previsto']]
        for ax, coluna, cmap, vmax, rotulo in [(linha[0], 'hgb', 'YlOrRd', vmax_risco, 'Risco previsto médio (HGB)'),
                                               (linha[1], 'n', 'YlOrRd', vmax_n, 'Atropelamentos observados')]:
            top.plot(ax=ax, color='black', linewidth=7, zorder=1)
            sub.plot(ax=ax, column=coluna, cmap=cmap, vmin=0, vmax=vmax, linewidth=4, zorder=2, legend=True,
                     legend_kwds={'label': rotulo, 'shrink': 0.7})
            ax.set_title(f'{UC_NOMES[uc]}: {rotulo[0].lower()}{rotulo[1:]}')
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(''); ax.set_ylabel('')
            # Barra de escala de 2 km (coordenadas em metros, UTM 23S)
            x0, y0 = sub.total_bounds[0], sub.total_bounds[1] - 0.06 * (sub.total_bounds[3] - sub.total_bounds[1])
            ax.plot([x0, x0 + 2000], [y0, y0], color='black', lw=2, zorder=3)
            ax.annotate('2 km', (x0 + 1000, y0), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
            ax.set_facecolor('#eef2f5')
            ax.margins(0.08)
    fig.suptitle(f'Teste temporal (jan/2014 a mar/2015). Contorno preto: os {config.HOTSPOT_MAP_FRACTION:.0%} dos segmentos com maior risco previsto',
                 )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    config.HOTSPOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.SEGMENTOS_CSV)
    teste = scores(df)
    print(f"Teste: {len(teste)} segmento × mês, {teste['ano_mes'].min()} a {teste['ano_mes'].max()}, "
          f"{int(teste['n'].sum())} atropelamentos, {(teste['n'] >= 1).mean():.1%} das unidades com n >= 1")

    por_segmento = teste.groupby('segmento_id').agg(
        uc=('uc', 'first'), n=('n', 'sum'), meses_com_atropelamento=('n', lambda s: int((s >= 1).sum())),
        hgb=('hgb', 'mean'), mlp=('mlp', 'mean'), historico_segmento=('historico_segmento', 'first')).reset_index()
    k = int(round(config.HOTSPOT_MAP_FRACTION * len(por_segmento)))
    por_segmento['posicao_previsto'] = por_segmento['hgb'].rank(ascending=False, method='first').astype(int)
    por_segmento['posicao_observado'] = por_segmento['n'].rank(ascending=False, method='first').astype(int)
    por_segmento['top_previsto'] = por_segmento['posicao_previsto'] <= k
    por_segmento['top_observado'] = por_segmento['posicao_observado'] <= k

    captura = pd.concat([capture_table(teste, 'segmento_mes'), capture_table(por_segmento, 'segmento')], ignore_index=True)
    captura.to_csv(config.TABLES_DIR / 'hotspots_captura.csv', index=False, float_format='%.4f')
    print(f"\nCaptura dos atropelamentos do teste:\n{captura.to_string(index=False, float_format=lambda v: f'{v:.3f}')}")

    corr = pd.DataFrame([{'modelo': m, 'spearman_risco_x_observado': spearmanr(por_segmento[m], por_segmento['n'])[0]}
                         for m in ['hgb', 'mlp', 'historico_segmento']])
    corr.to_csv(config.TABLES_DIR / 'hotspots_correlacao.csv', index=False, float_format='%.4f')
    print(f"\nCorrelação de Spearman entre o risco por segmento e o total observado:\n{corr.to_string(index=False)}")
    acerto = (por_segmento['top_previsto'] & por_segmento['top_observado']).sum()
    print(f"\nDos {k} segmentos do topo previsto (HGB), {acerto} estão entre os {k} com mais atropelamentos observados")

    segmentos = gpd.read_file(config.EXTERNAL_PROCESSED_DIR / 'segmentos_osm.gpkg')[['segmento_id', 'ref', 'x_centro', 'y_centro', 'geometry']]
    geo = segmentos.merge(por_segmento, on='segmento_id', how='inner')
    ranking = geo.drop(columns='geometry').sort_values('posicao_previsto')
    ranking.to_csv(config.TABLES_DIR / 'hotspots_previstos.csv', index=False, float_format='%.4f')
    print(f"\nDez segmentos de maior risco previsto:\n"
          f"{ranking[['posicao_previsto', 'segmento_id', 'ref', 'uc', 'hgb', 'mlp', 'n', 'posicao_observado']].head(10).to_string(index=False)}")

    plot_capture(teste, por_segmento, config.HOTSPOT_FIG_DIR / 'captura.png')
    plot_map(geo, config.HOTSPOT_FIG_DIR / 'mapa_risco_previsto_x_observado.png')
    print(f"\n[OK] tabelas hotspots_*.csv em {config.TABLES_DIR.relative_to(config.ROOT)}/; figuras em {config.HOTSPOT_FIG_DIR.relative_to(config.ROOT)}/")


if __name__ == '__main__':
    run()
