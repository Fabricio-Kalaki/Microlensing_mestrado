import pandas as pd
import numpy as np
from config import OUTPUT_DIR

def calcular_velocidades_relativa(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula velocidades relativas entre pares de estrelas no CSV de entrada (_predict.csv).
    """

    # Localiza o CSV no diretório de saída
    if csv_input is None:
        csv_files = sorted(OUTPUT_DIR.glob("*_predict.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo '_predict.csv' encontrado em {OUTPUT_DIR}")
        csv_path = csv_files[0]
    else:
        csv_path = OUTPUT_DIR / csv_input
        if not str(csv_path).endswith(".csv"):
            csv_path = csv_path.with_suffix(".csv")

    # --- Leitura ---
    df = pd.read_csv(csv_path)

    # Calcula velocidades relativas (mas/yr)
    df['Vrel_RA (mas/yr)'] = df['Fonte_pmRA (mas/yr)'] - df['Lente_pmRA (mas/yr)']
    df['Vrel_DE (mas/yr)'] = df['Fonte_pmDE (mas/yr)'] - df['Lente_pmDE (mas/yr)']
    df['Vrel_t (mas/yr)']  = np.sqrt(df['Vrel_RA (mas/yr)']**2 + df['Vrel_DE (mas/yr)']**2)

    # Reorganiza colunas após 'Distancia_Fonte (parsec)'
    cols = list(df.columns)
    insert_pos = cols.index('Distancia_Fonte (parsec)') + 1
    for col in ['Vrel_RA (mas/yr)', 'Vrel_DE (mas/yr)', 'Vrel_t (mas/yr)']:
        cols.insert(insert_pos, cols.pop(cols.index(col)))
        insert_pos += 1
    df = df[cols]

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com velocidades relativas salvo em: {csv_path}")

    return str(csv_path)
