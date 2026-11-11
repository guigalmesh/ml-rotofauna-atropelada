"""
Pré-processamento, modelos clássicos e o laço de avaliação por split (Etapa 5).

Todo pré-processamento (imputação pela mediana, StandardScaler nos lineares,
one-hot das categóricas) fica dentro de um Pipeline e é ajustado só no treino
de cada rodada. Os splits vêm das colunas gravadas na Etapa 4; a faixa de
exclusão espacial é aplicada aqui, em cada rodada.
"""
import time

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.splits import segment_distance_matrix, spatial_cv_splits

LINEAR_MODELS = {'logistica', 'linear', 'poisson'}
TREE_MODELS = {'random_forest', 'hgb'}


# =============================================================================
# ATRIBUTOS E PRÉ-PROCESSAMENTO
# =============================================================================
def feature_set(nome: str, dataset: str, include_history: bool = False) -> list[str]:
    """Colunas de um conjunto da ablação para o dataset ('ocorrencias' ou 'segmentos_periodo')."""
    grupos = list(config.ABLATION_SETS[nome])
    if include_history and dataset == 'segmentos_periodo':
        grupos.append(config.HISTORY_GROUP)
    cols = [c for g in grupos for c in config.FEATURE_GROUPS[g]]
    if dataset == 'ocorrencias':
        cols += config.OCCURRENCE_EXTRA_FEATURES
    elif 'via' in grupos:
        cols += config.PANEL_EXTRA_FEATURES
    return cols


def make_preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    cat = [c for c in features if c in config.CATEGORICAL_FEATURES]
    num = [c for c in features if c not in cat]
    passos = [('imputacao', SimpleImputer(strategy='median'))]
    if scale:
        passos.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('num', Pipeline(passos), num),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
    ])


def classical_models(problema: str, seed: int = config.SEED) -> dict:
    """Modelos da Etapa 5 por problema: referências ingênuas, linear e árvores."""
    if problema == 'multiclasse':
        return {
            'majoritaria': DummyClassifier(strategy='most_frequent'),
            'estratificada': DummyClassifier(strategy='stratified', random_state=seed),
            'logistica': LogisticRegression(max_iter=5000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced',
                                                    random_state=seed, n_jobs=-1),
            'hgb': HistGradientBoostingClassifier(class_weight='balanced', random_state=seed),
        }
    if problema == 'binario':
        return {
            'prevalencia': DummyClassifier(strategy='prior'),
            'logistica': LogisticRegression(max_iter=5000, class_weight='balanced'),
            'random_forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight='balanced',
                                                    random_state=seed, n_jobs=-1),
            'hgb': HistGradientBoostingClassifier(class_weight='balanced', random_state=seed),
        }
    if problema == 'regressao':
        return {
            'media': DummyRegressor(strategy='mean'),
            'linear': LinearRegression(),
            'poisson': PoissonRegressor(alpha=1e-4, max_iter=2000),
            'random_forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1),
            'hgb': HistGradientBoostingRegressor(loss='poisson', random_state=seed),
        }
    raise ValueError(problema)


def build_pipeline(nome: str, estimator, features: list[str]) -> Pipeline:
    """Scaler só nos lineares (nas árvores não muda nada); imputação e one-hot sempre."""
    return Pipeline([('pre', make_preprocessor(features, scale=nome in LINEAR_MODELS)), ('modelo', estimator)])


# =============================================================================
# SPLITS
# =============================================================================
class SplitContext:
    """Geometria dos segmentos para a faixa de exclusão espacial."""

    def __init__(self):
        self.segmentos = gpd.read_file(config.EXTERNAL_PROCESSED_DIR / 'segmentos_osm.gpkg')
        self.dist = segment_distance_matrix(self.segmentos)


def iter_splits(df: pd.DataFrame, split: str, ctx: SplitContext, buffer_m: float = config.SPATIAL_BUFFER_M):
    """
    Rodadas (nome, máscara de treino, máscara de avaliação) de um split:
      aleatorio: 5 folds sorteados por linha, dentro do treino temporal
      espacial:  5 folds por bloco com faixa de exclusão, dentro do treino temporal
      temporal:  uma rodada, treino <= ano de corte e teste depois (teste final)
    """
    treino_temporal = (df['split_temporal'] == 'treino').values
    if split == 'temporal':
        yield 'teste', treino_temporal, ~treino_temporal
    elif split == 'aleatorio':
        folds = df['fold_aleatorio'].values
        for k in np.sort(np.unique(folds[treino_temporal])):
            val = treino_temporal & (folds == k)
            yield f'fold{k}', treino_temporal & ~val, val
    elif split == 'espacial':
        sub = df[treino_temporal]
        idx = np.where(treino_temporal)[0]
        for k, (tr, val) in enumerate(spatial_cv_splits(sub['segmento_id'], sub['fold_espacial'], ctx.segmentos, ctx.dist, buffer_m)):
            m_tr, m_val = np.zeros(len(df), bool), np.zeros(len(df), bool)
            m_tr[idx[tr]] = True
            m_val[idx[val]] = True
            yield f'fold{k}', m_tr, m_val
    else:
        raise ValueError(split)


# =============================================================================
# LAÇO DE AVALIAÇÃO
# =============================================================================
def predict(pipe: Pipeline, X: pd.DataFrame, problema: str):
    """Previsão e, na classificação, a probabilidade (da classe 1 no binário)."""
    y_pred = pipe.predict(X)
    if problema == 'regressao':
        return y_pred, None
    proba = pipe.predict_proba(X)
    # Binário (duas colunas): probabilidade da classe 1; multiclasse: matriz inteira
    return y_pred, proba[:, 1] if proba.shape[1] == 2 else proba


def evaluate_models(df: pd.DataFrame, y: pd.Series, problema: str, dataset: str, modelos: dict,
                    conjuntos: list[str], splits: list[str], ctx: SplitContext, metric_fn,
                    keep_predictions: tuple[str, str] | None = None) -> tuple[list[dict], dict]:
    """
    Treina cada modelo × conjunto de atributos × rodada de cada split e devolve
    as linhas de resultado (formato longo) e, se pedido, as previsões da rodada
    (split, nome) para cada modelo do conjunto 'completo'.
    """
    rows, predicoes = [], {}
    for conjunto in conjuntos:
        for split in splits:
            feats = feature_set(conjunto, dataset, include_history=(split != 'espacial'))
            for nome, est in modelos.items():
                for rodada, m_tr, m_val in iter_splits(df, split, ctx):
                    pipe = build_pipeline(nome, est, feats)
                    t0 = time.perf_counter()
                    pipe.fit(df.loc[m_tr, feats], y[m_tr])
                    tempo = time.perf_counter() - t0
                    y_pred, y_proba = predict(pipe, df.loc[m_val, feats], problema)
                    metricas = metric_fn(y[m_val], y_pred, y_proba)
                    for metrica, valor in metricas.items():
                        rows.append({'problema': problema, 'modelo': nome, 'conjunto_atributos': conjunto, 'split': split,
                                     'fold': rodada, 'metrica': metrica, 'valor': float(valor), 'tempo_s': round(tempo, 3)})
                    if keep_predictions and (split, rodada) == keep_predictions and conjunto == 'completo':
                        predicoes[nome] = {'mask': m_val, 'y_pred': y_pred, 'y_proba': y_proba, 'pipe': pipe, 'features': feats}
                print(f"  {conjunto:14s} {split:9s} {nome:14s} ok")
    return rows, predicoes


def importance_by_permutation(pipe: Pipeline, X: pd.DataFrame, y, scoring: str) -> pd.DataFrame:
    """Importância por permuta nos atributos originais (o one-hot fica dentro do pipeline)."""
    r = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=config.PERMUTATION_REPEATS,
                               random_state=config.SEED, n_jobs=-1)
    return (pd.DataFrame({'atributo': X.columns, 'importancia_media': r.importances_mean, 'importancia_desvio': r.importances_std})
              .sort_values('importancia_media', ascending=False).reset_index(drop=True))
