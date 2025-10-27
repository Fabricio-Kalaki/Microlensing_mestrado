import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from tqdm import tqdm
from config import OUTPUT_DIR


# ---------- Funções vetorizadas ---------
def compute_sigma_dt_vect(dra, ddec, dpra, dpdec, sigma_dra, sigma_ddec, sigma_dpra, sigma_dpdec):
    dp2 = dpra**2 + dpdec**2

    dtd_dra   = dpra / dp2
    dtd_ddec  = dpdec / dp2
    dtd_dpra  = -((dra * dpra**2) + (2*dpdec*ddec*dpra - dra*dpdec**2)) / (dp2**2)
    dtd_dpdec = -((ddec * dpdec**2) + (2*dra*dpra*dpdec - ddec*dpra**2)) / (dp2**2)

    sigma_dt2 = (dtd_dra**2) * sigma_dra**2 + \
                (dtd_ddec**2) * sigma_ddec**2 + \
                (dtd_dpra**2) * sigma_dpra**2 + \
                (dtd_dpdec**2) * sigma_dpdec**2

    sigma_dt = np.sqrt(sigma_dt2).to(u.yr)
    return sigma_dt


def compute_sigma_dmin_vect(dra, ddec, dpra, dpdec,
                            sigma_dra, sigma_ddec, sigma_dpra, sigma_dpdec,
                            dt, sigma_dt):
    S1 = dra - dpra*dt
    S2 = ddec - dpdec*dt
    den = np.sqrt(S1**2 + S2**2)

    ddmin_ddra   = S1 / den
    ddmin_dddec  = S2 / den
    ddmin_dpra   = dt * (dt*dpra - dra) / den
    ddmin_dpdec  = dt * (dt*dpdec - ddec) / den
    ddmin_dt     = ((dpra**2 + dpdec**2)*dt - dra*dpra - ddec*dpdec) / den

    sigma_dmin2 = (ddmin_ddra**2 * sigma_dra**2 +
                   ddmin_dddec**2 * sigma_ddec**2 +
                   ddmin_dpra**2 * sigma_dpra**2 +
                   ddmin_dpdec**2 * sigma_dpdec**2 +
                   ddmin_dt**2 * sigma_dt**2)

    sigma_dmin = np.sqrt(sigma_dmin2)
    return sigma_dmin


# ---------- Função principal ----------
def calcula_incerteza(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula incertezas e aplica filtro para pares de estrelas no CSV de entrada (_predict.csv).
    """

    #Identificação do CSV correto dentro do OUTPUT_DIR
    if csv_input is None:
        csv_files = sorted(OUTPUT_DIR.glob("*_predict.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo '_predict.csv' encontrado em {OUTPUT_DIR}")
        csv_path = csv_files[0]
    else:
        csv_path = OUTPUT_DIR / csv_input
        if not str(csv_path).endswith(".csv"):
            csv_path = csv_path.with_suffix(".csv")

    # --- Leitura do CSV ---
    df = pd.read_csv(csv_path, index_col='Indice')

    # Remove colunas antigas se existirem
    insert_pos = df.columns.get_loc('dmin_robusto (mas)') + 1
    for col in ['e_dt (yr)', 'e_dmin (mas)']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Adiciona colunas zeradas
    df.insert(insert_pos, 'e_dt (yr)', 0.0)
    df.insert(insert_pos + 1, 'e_dmin (mas)', 0.0)

    # --- Coordenadas ---
    lens = SkyCoord(ra=df["Lente_RA_ICRS (deg)"].values*u.deg,
                    dec=df["Lente_DE_ICRS (deg)"].values*u.deg)
    src = SkyCoord(ra=df["Fonte_RA_ICRS (deg)"].values*u.deg,
                   dec=df["Fonte_DE_ICRS (deg)"].values*u.deg)

    dra_ang, ddec_ang = lens.spherical_offsets_to(src)
    dra  = dra_ang.to(u.mas)
    ddec = ddec_ang.to(u.mas)

    dpra  = (df["Lente_pmRA (mas/yr)"].values - df["Fonte_pmRA (mas/yr)"].values) * (u.mas/u.yr)
    dpdec = (df["Lente_pmDE (mas/yr)"].values - df["Fonte_pmDE (mas/yr)"].values) * (u.mas/u.yr)

    sigma_dpra  = np.sqrt(df["Lente_e_pmRA (mas/yr)"].values**2 + df["Fonte_e_pmRA (mas/yr)"].values**2) * (u.mas/u.yr)
    sigma_dpdec = np.sqrt(df["Lente_e_pmDE (mas/yr)"].values**2 + df["Fonte_e_pmDE (mas/yr)"].values**2) * (u.mas/u.yr)

    cos_l = np.cos(lens.dec.to(u.rad).value)
    cos_s = np.cos(src.dec.to(u.rad).value)
    sigma_dra  = np.sqrt((df["Lente_e_RA_ICRS (mas)"].values * cos_l)**2 +
                         (df["Fonte_e_RA_ICRS (mas)"].values * cos_s)**2) * u.mas
    sigma_ddec = np.sqrt(df["Lente_e_DE_ICRS (mas)"].values**2 + df["Fonte_e_DE_ICRS (mas)"].values**2) * u.mas

    dt = df["dt (yr)"].values * u.yr

    sigma_dt = compute_sigma_dt_vect(dra, ddec, dpra, dpdec,
                                     sigma_dra, sigma_ddec, sigma_dpra, sigma_dpdec)
    sigma_dmin = compute_sigma_dmin_vect(dra, ddec, dpra, dpdec,
                                         sigma_dra, sigma_ddec, sigma_dpra, sigma_dpdec,
                                         dt, sigma_dt)

    df['e_dt (yr)'] = sigma_dt.value
    df['e_dmin (mas)'] = sigma_dmin.value

    df = df[df['e_dt (yr)'] <= 1].sort_values(by='e_dt (yr)')

    if save_csv:
        df.to_csv(csv_path)
        print(f"Tabela atualizada com incertezas e filtro salva em: {csv_path}")

    return str(csv_path)
