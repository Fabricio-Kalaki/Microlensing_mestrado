import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcula_e_S(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a incerteza em S ('e_S (mas)') e insere no CSV atualizado (_predict.csv).
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

    # Insere coluna e_S se não existir
    if 'e_S (mas)' not in df.columns:
        pos = df.columns.get_loc('S (mas)') + 1
        df.insert(pos, 'e_S (mas)', np.nan)

    # Calcula erro de S linha a linha
    for i in df.index:
        sigma_dmin   = df.at[i, 'e_dmin (mas)']
        dmin         = df.at[i, 'dmin_robusto (mas)']
        sigma_thetaE = df.at[i, 'e_RE (mas)']
        thetaE       = df.at[i, 'RE (mas)']

        if pd.isna(dmin) or pd.isna(thetaE) or dmin == 0 or thetaE == 0:
            continue

        sigma_s = np.sqrt((sigma_dmin/thetaE)**2 + ((dmin * sigma_thetaE)/(thetaE**2))**2)
        df.at[i, 'e_S (mas)'] = sigma_s

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com erro em S salvo em: {csv_path}")

    return str(csv_path)
