"""
Splits dos datasets processados (Etapa 4).

Blocos espaciais são trechos contínuos de ~SPATIAL_BLOCK_M ao longo de cada
circuito da malha; cada bloco cai num fold espacial. Na validação espacial,
os segmentos de treino a até SPATIAL_BUFFER_M do bloco de validação saem do
treino daquela rodada (faixa de exclusão, calculada aqui e não gravada nos
CSVs). O split temporal (treino ≤ TEMPORAL_SPLIT_LAST_TRAIN_YEAR) é o teste
final; o fold aleatório é só referência otimista.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse.csgraph import connected_components

from src import config


# =============================================================================
# BLOCOS ESPACIAIS AO LONGO DA VIA
# =============================================================================
def segment_distance_matrix(segmentos: gpd.GeoDataFrame) -> np.ndarray:
    """Distância (m) entre as geometrias de todos os pares de segmentos."""
    geoms = segmentos.geometry.values
    return np.array([geoms.distance(g) for g in geoms])


def _ordered_walk(adj: np.ndarray, nodes: np.ndarray) -> list[int]:
    """
    Ordem de percurso de um componente: busca em profundidade a partir de
    uma ponta (nó de menor grau; empate pelo menor índice), de modo que
    segmentos consecutivos na ordem sejam vizinhos na via.
    """
    graus = adj[np.ix_(nodes, nodes)].sum(axis=1)
    inicio = nodes[np.lexsort((nodes, graus))[0]]
    visitado, ordem, pilha = set(), [], [inicio]
    while pilha:
        atual = pilha.pop()
        if atual in visitado:
            continue
        visitado.add(atual)
        ordem.append(atual)
        vizinhos = [v for v in nodes if adj[atual, v] and v not in visitado]
        pilha.extend(sorted(vizinhos, reverse=True))
    return ordem


def _attach_small_components(adj: np.ndarray, dist: np.ndarray, comprimento: np.ndarray,
                             min_len_m: float, max_dist_m: float) -> np.ndarray:
    """
    Circuitos pequenos (< min_len_m) ganham uma aresta até o segmento mais
    próximo de outro circuito (se a <= max_dist_m), para serem percorridos e
    cortados junto com ele. Repete até não restar circuito pequeno ligável.
    """
    adj = adj.copy()
    for _ in range(len(adj)):
        _, comp = connected_components(adj, directed=False)
        ligou = False
        for c in np.unique(comp):
            nodes = np.where(comp == c)[0]
            if comprimento[nodes].sum() >= min_len_m:
                continue
            outros = np.where(comp != c)[0]
            if len(outros) == 0:
                continue
            d = dist[np.ix_(nodes, outros)]
            i, j = np.unravel_index(d.argmin(), d.shape)
            if d[i, j] <= max_dist_m:
                adj[nodes[i], outros[j]] = adj[outros[j], nodes[i]] = True
                ligou = True
                break
        if not ligou:
            return adj
    return adj


def spatial_blocks(segmentos: gpd.GeoDataFrame, dist: np.ndarray, block_m: float,
                   gap_m: float = config.SEGMENT_MAX_GAP_M) -> pd.Series:
    """
    bloco_espacial por segmento: cada circuito (componente de segmentos a
    <= gap_m, com os circuitos pequenos ligados ao vizinho mais próximo) é
    percorrido ao longo da via e cortado em k = round(L / block_m) trechos de
    comprimento parecido (L = comprimento do circuito), para não sobrar toco.
    """
    adj = dist <= gap_m
    np.fill_diagonal(adj, False)
    comprimento = segmentos['comprimento_m'].values
    adj = _attach_small_components(adj, dist, comprimento, block_m / 2, block_m)
    _, comp = connected_components(adj, directed=False)
    bloco = np.full(len(segmentos), -1)
    proximo = 0
    for c in np.unique(comp):
        nodes = np.where(comp == c)[0]
        total = comprimento[nodes].sum()
        k = max(1, int(round(total / block_m)))
        alvo = total / k
        acumulado = 0.0
        for node in _ordered_walk(adj, nodes):
            # Bloco pelo ponto médio do segmento na posição acumulada ao longo
            # do circuito, para o erro de arredondamento não cair todo no último
            meio = acumulado + comprimento[node] / 2
            bloco[node] = proximo + min(k - 1, int(meio // alvo))
            acumulado += comprimento[node]
        proximo += k
    # Renumera na ordem de aparição, para ids estáveis de 0 a K-1
    ids = {b: i for i, b in enumerate(pd.unique(bloco))}
    return pd.Series([ids[b] for b in bloco], index=segmentos.index, name='bloco_espacial')


def balanced_block_folds(bloco: pd.Series, peso: pd.Series, n_folds: int, seed: int) -> pd.Series:
    """
    Fold de cada bloco: blocos embaralhados com a semente e distribuídos de
    forma gulosa, do mais pesado para o mais leve, sempre ao fold com menor
    peso acumulado (peso = ocorrências do bloco), para equilibrar os folds.
    """
    peso_bloco = peso.groupby(bloco).sum()
    rng = np.random.default_rng(seed)
    ordem = peso_bloco.sample(frac=1, random_state=int(rng.integers(1 << 31))).sort_values(ascending=False, kind='stable')
    carga = np.zeros(n_folds)
    fold = {}
    for b, w in ordem.items():
        k = int(carga.argmin())
        fold[b] = k
        carga[k] += w
    return bloco.map(fold).rename('fold_espacial')


# =============================================================================
# FAIXA DE EXCLUSÃO E GERADORES DE SPLIT
# =============================================================================
def spatial_cv_splits(segmento_id: pd.Series, fold_espacial: pd.Series, segmentos: gpd.GeoDataFrame,
                      dist: np.ndarray, buffer_m: float):
    """
    Gera (treino, validação) como máscaras booleanas sobre as linhas, uma
    rodada por fold presente nas linhas. As linhas do fold k validam; das
    demais, saem do treino as que estão em segmentos a <= buffer_m de qualquer
    segmento do fold k. Funciona com qualquer subconjunto de linhas (por
    exemplo, só o treino temporal, ou ocorrências que não cobrem todos os
    segmentos): o fold de cada segmento vem das próprias linhas.
    """
    pos = pd.Series(np.arange(len(segmentos)), index=segmentos['segmento_id'].values)
    seg_id = np.asarray(segmento_id)
    fold = np.asarray(fold_espacial)
    linha_seg = pos.reindex(seg_id).values
    for k in np.sort(np.unique(fold)):
        val = fold == k
        val_segs = np.unique(linha_seg[val])
        perto = dist[:, val_segs].min(axis=1) <= buffer_m
        treino = ~val & ~perto[linha_seg]
        yield treino, val


def temporal_split(ano: pd.Series) -> pd.Series:
    return np.where(ano <= config.TEMPORAL_SPLIT_LAST_TRAIN_YEAR, 'treino', 'teste')


def random_folds(n: int, n_folds: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_folds, size=n)
