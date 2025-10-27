import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcula_e_A(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a incerteza em A ('e_A (mag)') a partir de S e e_S,
    e atualiza o mesmo CSV (_predict.csv).
    """

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

    # Insere coluna e_A se não existir
    if 'e_A (mag)' not in df.columns:
        pos = df.columns.get_loc('A (mag)') + 1
        df.insert(pos, 'e_A (mag)', np.nan)

    # Garante que S e e_S são numéricos
    df['S (mas)'] = pd.to_numeric(df['S (mas)'], errors='coerce').fillna(0)
    df['e_S (mas)'] = pd.to_numeric(df['e_S (mas)'], errors='coerce').fillna(0)

    # Calcula erro de A linha a linha (propagação de incerteza)
    for i, row in df.iterrows():
        S = row['S (mas)']
        e_S = row['e_S (mas)']
        if S > 0 and e_S >= 0:
            sigmaA = np.sqrt(((-8 / (S**2 * (S**2 + 4)**1.5))**2) * e_S**2)
            df.at[i, 'e_A (mag)'] = sigmaA

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com e_A (mag) salvo em: {csv_path}")

    return str(csv_path)
