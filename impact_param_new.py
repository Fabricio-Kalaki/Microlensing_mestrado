import pandas as pd
from config import OUTPUT_DIR

def calcula_S(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula S = dmin / RE e insere no CSV atualizado (_predict.csv).
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

    # Verifica colunas obrigatórias
    required_cols = ['dmin_robusto (mas)', 'RE (mas)']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colunas obrigatórias ausentes em {csv_path}: {missing_cols}")

    # Calcula S
    df['S (mas)'] = df['dmin_robusto (mas)'] / df['RE (mas)']

    # Posiciona coluna após tE (days), se existir
    if 'tE (days)' in df.columns:
        pos = df.columns.get_loc('tE (days)') + 1
    else:
        pos = len(df.columns)
    s_col = df.pop('S (mas)')
    df.insert(pos, 'S (mas)', s_col)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com S salvo em: {csv_path}")

    return str(csv_path)
