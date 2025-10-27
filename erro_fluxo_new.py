import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcula_e_FL_FF(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a incerteza em FL/FF ('e_FL/FF') a partir de magnitudes e atualiza o mesmo CSV (_predict.csv).
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

    # Verifica se FL/FF já existe
    if 'FL/FF' not in df.columns:
        raise ValueError(f"A coluna 'FL/FF' não existe no CSV {csv_path}. Execute calcula_fluxo primeiro.")

    # Garante que colunas são numéricas
    df['FL/FF'] = pd.to_numeric(df['FL/FF'], errors='coerce').fillna(0)
    df['Fonte_e_Gmag (mag)'] = pd.to_numeric(df['Fonte_e_Gmag (mag)'], errors='coerce').fillna(0)
    df['Lente_e_Gmag (mag)'] = pd.to_numeric(df['Lente_e_Gmag (mag)'], errors='coerce').fillna(0)

    # Calcula erro de FL/FF
    e_flff = (df['FL/FF'] * np.log(10) / 2.5) * np.sqrt(
        df['Fonte_e_Gmag (mag)']**2 + df['Lente_e_Gmag (mag)']**2
    )

    # Remove coluna antiga se existir e insere nova logo após FL/FF
    if 'e_FL/FF' in df.columns:
        df.drop(columns=['e_FL/FF'], inplace=True)

    insert_idx = df.columns.get_loc('FL/FF') + 1
    df.insert(loc=insert_idx, column='e_FL/FF', value=e_flff)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com e_FL/FF salvo em: {csv_path}")

    return str(csv_path)
