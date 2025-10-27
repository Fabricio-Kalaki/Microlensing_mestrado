import pandas as pd
from config import OUTPUT_DIR

def calcula_fluxo(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Calcula a razão de fluxos FL/FF a partir de magnitudes e atualiza o mesmo CSV (_predict.csv).
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

    # Remove coluna antiga se existir
    if 'FL/FF' in df.columns:
        df.drop(columns=['FL/FF'], inplace=True)

    # Calcula FL/FF = 10 ** ((Fonte_Gmag - Lente_Gmag) / 2.5)
    df['FL/FF'] = 10 ** ((df['Fonte_Gmag (mag)'] - df['Lente_Gmag (mag)']) / 2.5)

    # Reorganiza coluna para após 'e_A (mag)' ou 'A (mag)'
    insert_after = 'e_A (mag)' if 'e_A (mag)' in df.columns else 'A (mag)'
    col = df.pop('FL/FF')
    df.insert(df.columns.get_loc(insert_after) + 1, 'FL/FF', col)

    # Salva CSV atualizado
    if save_csv:
        df.to_csv(csv_path, index=False)
        print(f"CSV atualizado com FL/FF salvo em: {csv_path}")

    return str(csv_path)
