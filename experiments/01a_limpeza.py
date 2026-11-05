from src import config
from src.preprocessing import build_interim


def run_cleaning():
    print("=" * 75)
    print("LIMPEZA: RAW → INTERIM")
    print("=" * 75)
    df, decisoes, qualidade, campanhas = build_interim(config.RAW_CSV)

    # Verificações da base limpa
    assert (df['classe'] != 'nao_identificado').all()
    assert (df['uc'] != 'nao_identificado').all()
    proibidas = set(config.PROHIBITED_FEATURES) - set(config.AUX_COLUMNS)
    assert not proibidas & set(df.columns), f"Colunas proibidas no interim: {proibidas & set(df.columns)}"
    assert decisoes['linhas_depois'].iloc[-1] == len(df)
    assert set(df['campanha']) <= set(campanhas['campanha'])

    config.INTERIM_CSV.parent.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.INTERIM_CSV, index=False)
    decisoes.to_csv(config.DECISOES_LIMPEZA_CSV, index=False)
    qualidade.to_csv(config.QUALIDADE_ORIGINAL_CSV)
    campanhas.to_csv(config.CAMPANHAS_CSV, index=False)

    print(decisoes[['etapa', 'linhas_antes', 'linhas_depois']].to_string(index=False))
    print()
    for _, d in decisoes.iterrows():
        print(f"- {d['etapa']}: {d['motivo']}")
    print(f"\nInterim: {len(df)} linhas x {df.shape[1]} colunas → {config.INTERIM_CSV.relative_to(config.ROOT)}")
    print(f"Decisões → {config.DECISOES_LIMPEZA_CSV.relative_to(config.ROOT)}")
    print(f"Qualidade da base original → {config.QUALIDADE_ORIGINAL_CSV.relative_to(config.ROOT)}")
    print(f"Campanhas: {len(campanhas)} → {config.CAMPANHAS_CSV.relative_to(config.ROOT)}")


if __name__ == '__main__':
    run_cleaning()
