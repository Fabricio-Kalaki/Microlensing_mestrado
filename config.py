from pathlib import Path

# BASE_DIR aponta para o diretório raiz do projeto (onde está este arquivo config.py)
BASE_DIR = Path(__file__).parent

# Diretório de saída (resultados do pipeline)
OUTPUT_DIR = BASE_DIR / "Prediction_table"

# Diretório de entrada (arquivos CSV do Gaia)
INPUT_DIR = BASE_DIR / "Gaia_catalog"

# Garante que as pastas existam
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
