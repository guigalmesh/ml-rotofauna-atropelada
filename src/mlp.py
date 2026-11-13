"""
MLP em PyTorch para os três problemas (Etapa 6), com a função de treino já
parametrizada para o Optuna (Etapa 7) e o estudo de regularização (Etapa 8).

Perdas: CrossEntropy com pesos de classe (multiclasse), BCEWithLogits com
pos_weight (binário) e Poisson NLL com saída em log-taxa (regressão de
contagem). O early stopping usa uma validação interna tirada do treino com o
mesmo critério da rodada (bloco espacial, último ano ou sorteio); o conjunto
avaliado nunca participa do treino nem do early stopping.
"""
import copy
import time

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score, mean_poisson_deviance, roc_auc_score

from src import config
from src.evaluation import (
    HIGHER_IS_BETTER, MAIN_METRIC, classification_metrics, comparison_table,
    plot_comparison, plot_learning_curves, regression_metrics, save_results,
)
from src.models import feature_set, iter_splits, make_preprocessor
from src.splits import spatial_cv_splits

ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh}
LOG_RATE_LIMIT = 10.0   # limita a log-taxa da regressão: exp(10) já é muito além de qualquer n


# =============================================================================
# REDE E TREINO
# =============================================================================
class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden=(64, 32), activation='relu', dropout=0.0):
        super().__init__()
        camadas, anterior = [], n_in
        for h in hidden:
            camadas += [nn.Linear(anterior, h), ACTIVATIONS[activation]()]
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            anterior = h
        camadas.append(nn.Linear(anterior, n_out))
        self.net = nn.Sequential(*camadas)

    def forward(self, x):
        return self.net(x)


def hidden_layers(n_layers: int, n_units: int) -> tuple:
    """Camadas em funil: a primeira com n_units e cada seguinte com a metade (mínimo 8)."""
    return tuple(max(8, n_units // 2 ** i) for i in range(n_layers))


def params_from_search(p: dict) -> dict:
    """Converte os hiperparâmetros do espaço de busca (n_layers, n_units, ...) nos argumentos de train_mlp."""
    p = dict(p)
    return {'hidden': hidden_layers(p.pop('n_layers'), p.pop('n_units')), **p}


def _criterion(task: str, y_tr: np.ndarray, n_out: int = 1):
    """Perda de cada problema, com os pesos calculados só no treino."""
    if task == 'multiclasse':
        # minlength e o piso de 1 cobrem o caso de uma classe ausente no treino da rodada
        contagem = np.maximum(np.bincount(y_tr, minlength=n_out), 1)
        pesos = len(y_tr) / (len(contagem) * contagem)
        return nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32))
    if task == 'binario':
        pos = max(y_tr.sum(), 1)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y_tr) - pos) / pos, dtype=torch.float32))
    return nn.PoissonNLLLoss(log_input=True, full=False)


def _targets(task: str, y: np.ndarray) -> torch.Tensor:
    return torch.tensor(y, dtype=torch.long if task == 'multiclasse' else torch.float32)


def _forward(model: nn.Module, X: torch.Tensor, task: str) -> torch.Tensor:
    out = model(X)
    if task == 'multiclasse':
        return out
    out = out.squeeze(1)
    return out.clamp(-LOG_RATE_LIMIT, LOG_RATE_LIMIT) if task == 'regressao' else out


def predict_mlp(model: nn.Module, X: np.ndarray, task: str):
    """(previsão, probabilidade): classe e probabilidades, ou a taxa prevista na regressão."""
    model.eval()
    with torch.no_grad():
        out = _forward(model, torch.tensor(X, dtype=torch.float32), task)
        if task == 'multiclasse':
            proba = torch.softmax(out, dim=1).numpy()
            return proba.argmax(axis=1), proba
        if task == 'binario':
            proba = np.nan_to_num(torch.sigmoid(out).numpy(), nan=0.5)
            return (proba >= 0.5).astype(int), proba
        return np.nan_to_num(torch.exp(out).numpy(), nan=0.0, posinf=np.exp(LOG_RATE_LIMIT)), None


def main_metric(task: str, y_true: np.ndarray, y_pred, y_proba) -> float:
    """Métrica principal do problema (F1 macro, ROC-AUC ou desvio de Poisson)."""
    if task == 'multiclasse':
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    if task == 'binario':
        return roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan
    return mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, task: str, *,
              seed: int, hidden=(64, 32), activation='relu', dropout=0.0, weight_decay=0.0, optimizer='adam',
              lr=1e-3, batch_size=64, max_epochs=300, patience=20, early_stopping=True, n_out=None):
    """
    Treina a MLP e devolve (modelo, histórico por época, melhor época). O
    histórico traz a loss e a métrica principal no treino e na validação,
    medidas em modo de avaliação. Com early_stopping, para após `patience`
    épocas sem melhora da loss de validação e restaura os melhores pesos;
    sem ele, treina até max_epochs e fica com os pesos finais.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(config.TORCH_THREADS)
    n_out = n_out or (int(max(y_tr.max(), y_val.max())) + 1 if task == 'multiclasse' else 1)
    model = MLP(X_tr.shape[1], n_out, hidden, activation, dropout)
    criterio = _criterion(task, y_tr, n_out)
    otimizadores = {
        'adam': lambda p: torch.optim.Adam(p, lr=lr, weight_decay=weight_decay),
        'rmsprop': lambda p: torch.optim.RMSprop(p, lr=lr, weight_decay=weight_decay),
        'sgd': lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=weight_decay),
    }
    otimizador = otimizadores[optimizer](model.parameters())
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), _targets(task, y_tr)
    Xv, yv = torch.tensor(X_val, dtype=torch.float32), _targets(task, y_val)
    gerador = torch.Generator().manual_seed(seed)

    historico, melhor_loss, melhor_estado, melhor_epoca, sem_melhora = [], np.inf, None, 0, 0
    for epoca in range(1, max_epochs + 1):
        model.train()
        ordem = torch.randperm(len(Xt), generator=gerador)
        for i in range(0, len(Xt), batch_size):
            lote = ordem[i:i + batch_size]
            otimizador.zero_grad()
            loss = criterio(_forward(model, Xt[lote], task), yt[lote])
            if not torch.isfinite(loss):
                break
            loss.backward()
            otimizador.step()
        model.eval()
        with torch.no_grad():
            loss_tr = criterio(_forward(model, Xt, task), yt).item()
            loss_val = criterio(_forward(model, Xv, task), yv).item()
        linha = {'epoca': epoca, 'loss_treino': loss_tr, 'loss_val': loss_val}
        for nome, X, y in [('treino', X_tr, y_tr), ('val', X_val, y_val)]:
            y_pred, y_proba = predict_mlp(model, X, task)
            linha[f'metrica_{nome}'] = main_metric(task, y, y_pred, y_proba)
        historico.append(linha)
        if not np.isfinite(loss_val):
            break
        if loss_val < melhor_loss - 1e-6:
            melhor_loss, melhor_epoca, sem_melhora = loss_val, epoca, 0
            melhor_estado = copy.deepcopy(model.state_dict())
        else:
            sem_melhora += 1
            if early_stopping and sem_melhora >= patience:
                break
    if early_stopping and melhor_estado is not None:
        model.load_state_dict(melhor_estado)
    else:
        melhor_epoca = len(historico)
    return model, pd.DataFrame(historico), melhor_epoca


# =============================================================================
# VALIDAÇÃO INTERNA DO EARLY STOPPING
# =============================================================================
def inner_validation_mask(df: pd.DataFrame, m_tr: np.ndarray, split: str, ctx, seed: int):
    """
    Divide o treino da rodada em (ajuste, validação interna) com o mesmo
    critério do split: temporal → último ano do treino; espacial → um fold de
    blocos, com faixa de exclusão; aleatório → fração sorteada.
    """
    idx = np.where(m_tr)[0]
    m_fit, m_val = np.zeros(len(df), bool), np.zeros(len(df), bool)
    if split == 'temporal':
        ultimo = df.loc[m_tr, 'ano'].max()
        m_val[idx] = (df['ano'].values[idx] == ultimo)
        m_fit[idx] = ~m_val[idx]
    elif split == 'espacial':
        sub = df.iloc[idx]
        rodadas = list(spatial_cv_splits(sub['segmento_id'], sub['fold_espacial'], ctx.segmentos, ctx.dist, config.SPATIAL_BUFFER_M))
        tr, val = rodadas[seed % len(rodadas)]
        m_fit[idx[tr]] = True
        m_val[idx[val]] = True
    else:
        sorteio = np.random.default_rng(seed).random(len(idx)) < config.MLP_INNER_VAL_FRACTION
        m_val[idx[sorteio]] = True
        m_fit[idx[~sorteio]] = True
    return m_fit, m_val


# =============================================================================
# LAÇO DE AVALIAÇÃO
# =============================================================================
def to_dense(X) -> np.ndarray:
    return np.asarray(X.todense() if hasattr(X, 'todense') else X, dtype=np.float32)


def evaluate_mlp(df: pd.DataFrame, y: np.ndarray, task: str, problema: str, dataset: str, conjuntos: list[str],
                 splits: list[str], ctx, *, scale: bool = True, nome: str = 'mlp', params: dict | None = None,
                 classes: list | None = None):
    """
    Treina a MLP em cada conjunto × split × rodada × semente e devolve as
    linhas de resultado, os históricos das rodadas do teste temporal no
    conjunto 'completo' e o artefato (modelo + pré-processador) da semente com
    menor loss de validação interna nesse mesmo cenário.
    """
    params = {**config.MLP_DEFAULTS, **(params or {})}
    metric_fn = regression_metrics if task == 'regressao' else classification_metrics
    rows, historicos, melhor = [], [], None
    for conjunto in conjuntos:
        for split in splits:
            feats = feature_set(conjunto, dataset, include_history=(split != 'espacial'))
            sementes = config.MLP_SEEDS_TEST if split == 'temporal' else config.MLP_SEEDS_CV
            for rodada, m_tr, m_aval in iter_splits(df, split, ctx):
                for seed in sementes:
                    m_fit, m_val = inner_validation_mask(df, m_tr, split, ctx, seed)
                    assert not (m_aval & (m_fit | m_val)).any(), 'conjunto avaliado vazou para o treino'
                    assert not (m_fit & m_val).any() and m_fit.any() and m_val.any()
                    pre = make_preprocessor(feats, scale=scale).fit(df.loc[m_fit, feats])
                    X_fit, X_val, X_aval = (to_dense(pre.transform(df.loc[m, feats])) for m in (m_fit, m_val, m_aval))
                    t0 = time.perf_counter()
                    model, hist, epoca = train_mlp(X_fit, y[m_fit], X_val, y[m_val], task, seed=seed,
                                                   n_out=len(classes) if classes else None, **params)
                    tempo = time.perf_counter() - t0
                    y_pred, y_proba = predict_mlp(model, X_aval, task)
                    if task == 'regressao':
                        metricas = metric_fn(y[m_aval], y_pred)
                    else:
                        metricas = metric_fn(y[m_aval], y_pred, y_proba if task == 'binario' else None)
                    for metrica, valor in metricas.items():
                        rows.append({'problema': problema, 'modelo': nome, 'conjunto_atributos': conjunto, 'split': split,
                                     'fold': rodada, 'metrica': metrica, 'valor': float(valor), 'tempo_s': round(tempo, 3),
                                     'semente': seed, 'epocas': epoca})
                    principal = metricas[MAIN_METRIC[task]]
                    print(f"  {nome:15s} {conjunto:9s} {split:9s} {rodada:6s} semente {seed}: parou na época {epoca:3d} "
                          f"de {len(hist):3d} | {MAIN_METRIC[task]} = {principal:.3f} | {tempo:5.1f}s", flush=True)
                    if split == 'temporal' and conjunto == 'completo':
                        historicos.append(hist.assign(semente=seed))
                        loss_val = hist['loss_val'].min()
                        if melhor is None or loss_val < melhor['loss_val']:
                            melhor = {'loss_val': loss_val, 'state_dict': copy.deepcopy(model.state_dict()), 'pre': pre,
                                      'features': feats, 'params': params, 'semente': seed, 'classes': classes, 'task': task,
                                      'n_in': X_fit.shape[1]}
    return rows, historicos, melhor


def run_mlp_section(df: pd.DataFrame, y: np.ndarray, task: str, problema: str, dataset: str, ctx, classes: list | None = None):
    """
    Seção da MLP dos scripts 04–06: treina nos conjuntos base e completo nos
    três splits, repete o completo sem StandardScaler (sensibilidade ao
    pré-processamento), e grava resultados, curvas, comparação com os
    clássicos e o modelo do teste temporal.
    """
    metrica = MAIN_METRIC[task]
    print(f"\nMLP {config.MLP_DEFAULTS['hidden']}, {config.MLP_DEFAULTS['activation']}, Adam lr {config.MLP_DEFAULTS['lr']}, "
          f"batch {config.MLP_DEFAULTS['batch_size']}, até {config.MLP_DEFAULTS['max_epochs']} épocas, "
          f"paciência {config.MLP_DEFAULTS['patience']}")
    rows, historicos, melhor = evaluate_mlp(df, y, task, problema, dataset, config.MLP_FEATURE_SETS, config.SPLITS, ctx, classes=classes)
    print("\nSensibilidade: MLP sem StandardScaler (conjunto completo)")
    rows_sem, _, _ = evaluate_mlp(df, y, task, problema, dataset, ['completo'], ['espacial', 'temporal'], ctx,
                                  scale=False, nome='mlp_sem_scaler', classes=classes)
    results = save_results(rows + rows_sem, problema)

    curvas = pd.concat(historicos, ignore_index=True)
    curvas.to_csv(config.TABLES_DIR / f'curvas_mlp_{problema}.csv', index=False)
    plot_learning_curves(curvas, metrica, f'MLP {problema}: treino × validação interna (teste temporal, conjunto completo)',
                         config.MODEL_FIG_DIR / f'{problema}_mlp_curvas.png')
    comparativo = comparison_table(results, problema, metrica)
    comparativo.to_csv(config.TABLES_DIR / f'comparativo_{problema}.csv')
    plot_comparison(results, problema, metrica, config.MODEL_FIG_DIR / f'{problema}_mlp_vs_classicos.png')
    print(f"\nComparativo no conjunto completo ({metrica}; {'maior' if HIGHER_IS_BETTER[metrica] else 'menor'} é melhor):")
    print(comparativo.to_string())

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor, config.MODELS_DIR / f'{problema}_mlp.joblib')
    print(f"\n[OK] MLP: resultados em {config.RESULTS_CSV.relative_to(config.ROOT)}; "
          f"modelo (semente {melhor['semente']}) em {(config.MODELS_DIR / f'{problema}_mlp.joblib').relative_to(config.ROOT)}")
    return results


def load_mlp(path):
    """Recarrega o artefato salvo: devolve (modelo, pré-processador, artefato)."""
    art = joblib.load(path)
    n_out = len(art['classes']) if art['classes'] else 1
    p = art['params']
    model = MLP(art['n_in'], n_out, p['hidden'], p['activation'], p['dropout'])
    model.load_state_dict(art['state_dict'])
    return model.eval(), art['pre'], art
