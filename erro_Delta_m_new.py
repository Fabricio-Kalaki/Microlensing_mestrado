import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcula_e_Delta_m(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:

    #Localiza o CSV de trabalho
    if csv_input is None:
        csv_files = sorted(OUTPUT_DIR.glob("*_predict.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo '_predict.csv' encontrado em {OUTPUT_DIR}")
        csv_path = csv_files[0]
    else:
        csv_path = OUTPUT_DIR / csv_input
        if not str(csv_path).endswith(".csv"):
            csv_path = csv_path.with_suffix(".csv")

    df = pd.read_csv(csv_path)

    # Converte colunas numéricas
    for col in ['A (mag)', 'FL/FF', 'e_FL/FF', 'e_A (mag)']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Remove coluna antiga se existir
    if 'e_Delta_m (mag)' in df.columns:
        df.drop(columns=['e_Delta_m (mag)'], inplace=True)

    # Cálculo do erro
    M = df['A (mag)']
    fls = df['FL/FF']
    sigma_fls = df['e_FL/FF']
    sigma_M = df['e_A (mag)']
    ln10 = np.log(10)

    term1 = ((-5*(M - 1)) / (2*ln10 * (fls + 1) * (fls + M)))**2 * sigma_fls**2
    term2 = (5 / (2*ln10 * (fls + M)))**2 * sigma_M**2
    e_delta = np.sqrt(term1 + term2)

    # Insere coluna após Delta_m
    insert_idx = df.columns.get_loc('Delta_m (mag)') + 1 if 'Delta_m (mag)' in df.columns else len(df.columns)
    df.insert(loc=insert_idx, column='e_Delta_m (mag)', value=e_delta)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com e_Delta_m salvo em: {csv_path}")

    return str(csv_path)
