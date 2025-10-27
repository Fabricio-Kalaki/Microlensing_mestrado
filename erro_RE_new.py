import numpy as np
import pandas as pd
from astropy.constants import G, c
from astropy import units as u
from config import OUTPUT_DIR

def calcula_e_RE(
    csv_input: str = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a incerteza em RE ('e_RE (mas)') usando paralaxes e massa.
    Garante que a coluna 'e_RE (mas)' fique logo após 'RE (mas)'.
    """

    # Se nenhum CSV for especificado, pega o primeiro *_predict.csv
    if csv_input is None:
        csv_files = sorted(OUTPUT_DIR.glob("*_predict.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo '_predict.csv' encontrado em {OUTPUT_DIR}")
        csv_path = csv_files[0]
    else:
        csv_path = OUTPUT_DIR / csv_input

    df = pd.read_csv(csv_path)

    # Insere coluna e_RE se não existir
    if 'e_RE (mas)' not in df.columns:
        idx = df.columns.get_loc('RE (mas)') + 1
        df.insert(idx, 'e_RE (mas)', np.nan)

    df.replace('--', 0, inplace=True)

    # Converte colunas relevantes para numérico
    cols_to_num = [
        'Lente_Mass-Flame (Msun)',
        'Lente_Plx (mas)', 'Fonte_Plx (mas)',
        'Lente_e_Plx (mas)', 'Fonte_e_Plx (mas)',
        'RE (mas)'
    ]
    df[cols_to_num] = df[cols_to_num].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Calcula sigma θ_E
    def calculate_sigma_thetaE(row):
        M = row['Lente_Mass-Flame (Msun)'] * u.M_sun
        plx_lens = row['Lente_Plx (mas)'] * u.mas
        plx_source = row['Fonte_Plx (mas)'] * u.mas
        e_plx_lens = row['Lente_e_Plx (mas)'] * u.mas
        e_plx_source = row['Fonte_e_Plx (mas)'] * u.mas

        if min(plx_lens.value, plx_source.value, e_plx_lens.value, e_plx_source.value) < 0:
            return np.nan

        delta_plx = plx_lens - plx_source
        denom = np.sqrt((4 * G * M / c**2) * delta_plx)
        term1 = ((2 * G * M) / (c**2 * denom))**2 * e_plx_lens**2
        term2 = ((2 * G * M) / (c**2 * denom))**2 * e_plx_source**2
        sigma2 = term1 + term2
        sigma_thetaE = np.sqrt(sigma2 / u.au)
        return sigma_thetaE.to(u.mas, equivalencies=u.dimensionless_angles()).value

    df['e_RE (mas)'] = df.apply(calculate_sigma_thetaE, axis=1)

    # Reorganiza a coluna e_RE após RE
    colunas = df.columns.tolist()
    if 'e_RE (mas)' in colunas and 'RE (mas)' in colunas:
        colunas.remove('e_RE (mas)')
        re_index = colunas.index('RE (mas)')
        colunas.insert(re_index + 1, 'e_RE (mas)')
        df = df[colunas]

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com erro em RE salvo em: {csv_path}")

    return str(csv_path)
