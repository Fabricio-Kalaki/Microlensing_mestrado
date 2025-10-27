import os
import numpy as np
import pandas as pd
from astropy.table import Table
from tqdm import tqdm
from max_aprox_altair import compare_batch_one_to_many
from config import INPUT_DIR, OUTPUT_DIR


def compare_loop_healpix(
    tbl=None, 
    healpix_n: int = 6,   # ordem padrão
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Inicia o pipeline: lê um arquivo CSV do diretório INPUT_DIR (ou um arquivo/table específico)
    e gera o primeiro arquivo de saída no OUTPUT_DIR com o sufixo '_predict.csv'.
    O nome segue o padrão: gaia_healpix{n}_{idx}_predict.csv
    """

    #Se nenhum arquivo for informado, pega o primeiro CSV em INPUT_DIR
    if tbl is None:
        csv_files = sorted(INPUT_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {INPUT_DIR}")
        tbl = csv_files[0]
        print(f"Usando arquivo de entrada: {tbl.name}")

    # Se o índice Healpix não foi passado, tenta extrair do nome do arquivo
    if healpix_idx is None:
        try:
            healpix_idx = int(''.join(filter(str.isdigit, tbl.stem.split('_')[-1])))
        except Exception:
            healpix_idx = 0

    # Converter entrada para pandas.DataFrame
    if isinstance(tbl, str) or isinstance(tbl, os.PathLike):
        tbl = pd.read_csv(tbl)
    elif isinstance(tbl, Table):
        tbl = tbl.to_pandas()

    # ----------------------------------------------------------------------
    # ⚙️ Lógica original
    # ----------------------------------------------------------------------

    rename_map = {
        'source_id': 'Source',
        'ra': 'RA_ICRS',
        'dec': 'DEC_ICRS',
        'ra_error': 'e_RA_ICRS',
        'dec_error': 'e_DE_ICRS',
        'pmra': 'pmRA',
        'pmra_error': 'e_pmRA',
        'pmdec': 'pmDE',
        'pmdec_error': 'e_pmDE',
        'parallax': 'Plx',
        'parallax_error': 'e_Plx',
        'radial_velocity': 'RV',
        'radial_velocity_error': 'e_VR',
        'phot_g_mean_mag': 'Gmag',
        'phot_g_mean_mag_error': 'e_Gmag',
        'mass_flame': 'Mass_Flame',
    }
    for old, new in rename_map.items():
        if old in tbl.columns and old != new:
            tbl.rename(columns={old: new}, inplace=True)

    source_id = tbl['Source'].astype(str).to_numpy()
    ra_deg = tbl['RA_ICRS'].to_numpy(dtype=float)
    dec_deg = tbl['DEC_ICRS'].to_numpy(dtype=float)
    pmra = tbl['pmRA'].to_numpy(dtype=float)
    pmdec = tbl['pmDE'].to_numpy(dtype=float)
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    plx = tbl['Plx'].to_numpy(dtype=float)
    e_ra = tbl['e_RA_ICRS'].to_numpy(dtype=float)
    e_dec = tbl['e_DE_ICRS'].to_numpy(dtype=float)
    e_pmra = tbl['e_pmRA'].to_numpy(dtype=float)
    e_pmdec = tbl['e_pmDE'].to_numpy(dtype=float)
    e_plx = tbl['e_Plx'].to_numpy(dtype=float)
    gmag = tbl['Gmag'].to_numpy(dtype=float)
    e_gmag = tbl['e_Gmag'].to_numpy(dtype=float)
    rv = tbl['RV'].to_numpy(dtype=float)
    e_rv = tbl['e_VR'].to_numpy(dtype=float)
    mass_flame = tbl['Mass_Flame'].to_numpy(dtype=float)

    cols = [
        "Estrela 1", "Estrela 2",
        "dt (yr)", "dmin (mas)",
        "Lente_RA_ICRS (deg)", "Lente_e_RA_ICRS (mas)",
        "Lente_DE_ICRS (deg)", "Lente_e_DE_ICRS (mas)",
        "Lente_pmRA (mas/yr)", "Lente_e_pmRA (mas/yr)",
        "Lente_pmDE (mas/yr)", "Lente_e_pmDE (mas/yr)",
        "Lente_Plx (mas)", "Lente_e_Plx (mas)",
        "Lente_RV (km/s)", "Lente_e_RV (km/s)",
        "Lente_Gmag (mag)", "Lente_e_Gmag (mag)", "Lente_Mass-Flame (Msun)",
        "Fonte_RA_ICRS (deg)", "Fonte_e_RA_ICRS (mas)",
        "Fonte_DE_ICRS (deg)", "Fonte_e_DE_ICRS (mas)",
        "Fonte_pmRA (mas/yr)", "Fonte_e_pmRA (mas/yr)",
        "Fonte_pmDE (mas/yr)", "Fonte_e_pmDE (mas/yr)",
        "Fonte_Plx (mas)", "Fonte_e_Plx (mas)",
        "Fonte_RV (km/s)", "Fonte_e_RV (km/s)",
        "Fonte_Gmag (mag)", "Fonte_e_Gmag (mag)", "Fonte_Mass-Flame (Msun)"
    ]

    rows = []
    N = ra_deg.size
    DT_MAX = 50.0
    DMIN_MAX_ARCSEC = 10.0

    for i in tqdm(range(N-1), desc="Comparando estrelas", unit="estrela", total=N-1, mininterval=0.3):
        js = np.arange(i+1, N)
        dt_years, dmin_arcsec = compare_batch_one_to_many(
            ra_rad[i], dec_rad[i], pmra[i], pmdec[i],
            ra_rad[js], dec_rad[js], pmra[js], pmdec[js],
            clip_years=DT_MAX
        )

        if dt_years.size == 0:
            continue

        keep = (np.abs(dt_years) < DT_MAX) & (dmin_arcsec < DMIN_MAX_ARCSEC)
        if not np.any(keep):
            continue

        js_keep = js[keep]
        dt_keep = dt_years[keep]
        dmin_mas = dmin_arcsec[keep] * 1e3
        t_jyear = 2016.0 + dt_keep

        cond_i_better = (
            (plx[i] >= plx[js_keep]) |
            ((plx[i] > 0) & (plx[js_keep] == 0)) |
            ((plx[i] > 0) & (plx[js_keep] < 0))
        )
        lens_idx = np.where(cond_i_better, i, js_keep)
        src_idx = np.where(cond_i_better, js_keep, i)

        for L, S, tval, dval in zip(lens_idx, src_idx, t_jyear, dmin_mas):
            rows.append((
                source_id[L], source_id[S],
                tval, dval,
                ra_deg[L], e_ra[L],
                dec_deg[L], e_dec[L],
                pmra[L], e_pmra[L],
                pmdec[L], e_pmdec[L],
                plx[L], e_plx[L],
                rv[L], e_rv[L],
                gmag[L], e_gmag[L], mass_flame[L],
                ra_deg[S], e_ra[S],
                dec_deg[S], e_dec[S],
                pmra[S], e_pmra[S],
                pmdec[S], e_pmdec[S],
                plx[S], e_plx[S],
                rv[S], e_rv[S],
                gmag[S], e_gmag[S], mass_flame[S],
            ))

    # ----------------------------------------------------------------------
    # Salvar arquivo com nome padrão no OUTPUT_DIR
    # ----------------------------------------------------------------------
    base_name = f"gaia_healpix{healpix_n}_{healpix_idx}"
    output_path = OUTPUT_DIR / f"{base_name}_predict.csv"

    if save_csv:
        if rows:
            df = pd.DataFrame.from_records(rows, columns=cols)
            num_cols = cols[2:]
            df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            df["Estrela 1"] = df["Estrela 1"].astype(str)
            df["Estrela 2"] = df["Estrela 2"].astype(str)
            df["Indice"] = df["Estrela 1"] + "-" + df["Estrela 2"]
            df.set_index("Indice", inplace=True)
            df.to_csv(output_path)
        else:
            pd.DataFrame(columns=cols + ["Indice"]).set_index("Indice").to_csv(output_path)

        print(f"CSV salvo com nome padronizado: {output_path}")

    return str(output_path)
