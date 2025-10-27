import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcula_A_mag(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a amplificação A (mag) a partir de S e atualiza o mesmo CSV (_predict.csv).
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

    # Garante que S é numérico
    df['S (mas)'] = pd.to_numeric(df['S (mas)'], errors='coerce').fillna(0)

    # Função de amplificação
    def amplificacao(s: float) -> float:
        return (s**2 + 2) / (s * np.sqrt(s**2 + 4)) if s != 0 else 0.0

    df['A (mag)'] = df['S (mas)'].apply(amplificacao)

    # Insere coluna A (mag) após e_S (mas) se existir, senão após S (mas)
    if 'e_S (mas)' in df.columns:
        insert_idx = df.columns.get_loc('e_S (mas)') + 1
    else:
        insert_idx = df.columns.get_loc('S (mas)') + 1
    values = df.pop('A (mag)')
    df.insert(insert_idx, 'A (mag)', values)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com A (mag) salvo em: {csv_path}")

    return str(csv_path)
