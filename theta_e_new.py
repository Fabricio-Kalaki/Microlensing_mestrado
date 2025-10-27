import pandas as pd
import numpy as np
import astropy.units as u
from astropy.constants import G, c
from config import OUTPUT_DIR

def calcula_theta(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula θ_E (Einstein radius) para pares de estrelas no CSV de entrada (_predict.csv).
    Garante que 'RE (mas)' exista e que 'e_RE (mas)' (se presente) fique imediatamente após ela.
    """

    # Localiza o CSV de trabalho
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

    # Converte colunas relevantes
    df['Lente_Plx (mas)'] = pd.to_numeric(df['Lente_Plx (mas)'], errors='coerce').fillna(0)
    df['Fonte_Plx (mas)'] = pd.to_numeric(df['Fonte_Plx (mas)'], errors='coerce').fillna(0)
    df['Lente_Mass-Flame (Msun)'] = pd.to_numeric(df['Lente_Mass-Flame (Msun)'], errors='coerce').fillna(0)

    # Calcula θ_E para cada linha
    theta_vals = []
    for _, row in df.iterrows():
        plx_lens = row['Lente_Plx (mas)'] * u.mas
        plx_source = row['Fonte_Plx (mas)'] * u.mas
        M = row['Lente_Mass-Flame (Msun)'] * u.M_sun

        term = plx_lens if plx_source == 0 * u.mas else (plx_lens - plx_source)
        theta_rad = np.sqrt((4 * G * M / c**2) * term / u.au)
        theta_mas = theta_rad.to(u.mas, equivalencies=u.dimensionless_angles())
        theta_vals.append(theta_mas.value)

    # Insere coluna após 'Vrel_t (mas/yr)'
    insert_idx = df.columns.get_loc('Vrel_t (mas/yr)') + 1
    df.insert(insert_idx, 'RE (mas)', theta_vals)
 
    # Salva CSV
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com Einstein θ_E salvo em: {csv_path}")

    return str(csv_path)
