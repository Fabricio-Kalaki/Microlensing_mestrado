import pandas as pd
from astropy import units as u
from config import OUTPUT_DIR


def calcular_distancia(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Converte paralaxes em distâncias para pares de estrelas no CSV (_predict.csv).
    """

    #Localiza o CSV atualizado no OUTPUT_DIR
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

    df['Lente_Plx (mas)'] = pd.to_numeric(df['Lente_Plx (mas)'], errors='coerce').fillna(0).clip(lower=0)
    df['Fonte_Plx (mas)'] = pd.to_numeric(df['Fonte_Plx (mas)'], errors='coerce').fillna(0).clip(lower=0)

    def parallax_to_distance(p_mas: float) -> float:
        p_arcsec = p_mas * u.mas.to(u.arcsec)
        return 0.0 if p_arcsec == 0 else round(1.0 / p_arcsec, 3)

    dist_lente = [parallax_to_distance(p) for p in df['Lente_Plx (mas)']]
    dist_fonte = [parallax_to_distance(p) for p in df['Fonte_Plx (mas)']]

    insert_idx = df.columns.get_loc('Fonte_Mass-Flame (Msun)') + 1
    df.insert(insert_idx,     'Distancia_Lente (parsec)', dist_lente)
    df.insert(insert_idx + 1, 'Distancia_Fonte (parsec)', dist_fonte)

    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com distâncias salvo em: {csv_path}")

    return str(csv_path)
