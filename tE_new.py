import pandas as pd
from config import OUTPUT_DIR

def te_calc(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula tE (tempo de cruzamento) em anos e dias a partir de RE e Vrel_t.
    """

    #Localiza o CSV de trabalho no diretório de saída
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

    # Garante que as colunas são numéricas
    df['RE (mas)'] = pd.to_numeric(df['RE (mas)'], errors='coerce').fillna(0)
    df['Vrel_t (mas/yr)'] = pd.to_numeric(df['Vrel_t (mas/yr)'], errors='coerce').fillna(0)

    # Calcula tE (yr)
    te_yr = df['RE (mas)'] / df['Vrel_t (mas/yr)']
    if 'tE (yr)' in df.columns:
        df.drop(columns=['tE (yr)'], inplace=True)
    idx_yr = df.columns.get_loc('RE (mas)') + 1
    df.insert(idx_yr, 'tE (yr)', te_yr)

    # Calcula tE (days)
    te_days = te_yr * 365.25
    if 'tE (days)' in df.columns:
        df.drop(columns=['tE (days)'], inplace=True)
    idx_days = df.columns.get_loc('tE (yr)') + 1
    df.insert(idx_days, 'tE (days)', te_days)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com tempo de cruzamento salvo em: {csv_path}")

    return str(csv_path)
