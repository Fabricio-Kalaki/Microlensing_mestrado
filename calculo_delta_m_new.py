import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def delta_m_calc(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:

    # Localiza arquivo CSV
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

    # Verifica colunas obrigatórias
    for col in ['FL/FF', 'A (mag)']:
        if col not in df.columns:
            raise KeyError(f"A coluna '{col}' não foi encontrada. Execute o passo correspondente antes.")

    # Remove Delta_m antiga se existir
    if 'Delta_m (mag)' in df.columns:
        df.drop(columns=['Delta_m (mag)'], inplace=True)

    # Calcula Delta_m
    df['Delta_m (mag)'] = 2.5 * np.log10((df['FL/FF'] + df['A (mag)']) / (df['FL/FF'] + 1))

    # Insere Delta_m após e_A (mag) se existir, senão após A (mag)
    insert_after = 'e_A (mag)' if 'e_A (mag)' in df.columns else 'A (mag)'
    values_dm = df.pop('Delta_m (mag)')
    df.insert(df.columns.get_loc(insert_after)+1, 'Delta_m (mag)', values_dm)

    # Garante que FL/FF permaneça logo após Delta_m
    if 'Delta_m (mag)' in df.columns and 'FL/FF' in df.columns:
        idx_flff = df.columns.get_loc('Delta_m (mag)') + 1
        values_flff = df.pop('FL/FF')
        df.insert(idx_flff, 'FL/FF', values_flff)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com delta_m_calc salvo em: {csv_path}")

    return str(csv_path)
