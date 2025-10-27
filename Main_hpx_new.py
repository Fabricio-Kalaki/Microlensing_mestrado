import json
import re
import csv
from pathlib import Path
from tqdm import tqdm

# ===== Imports do pipeline (seus) =====
from config import INPUT_DIR, OUTPUT_DIR
from compare_loop_healpix_altair_new import compare_loop_healpix
from robust_propagation_altair_new import propagate_robust
from Incertezas_new import calcula_incerteza
from Dist_calc_new import calcular_distancia
from Velocidade_relativa_new import calcular_velocidades_relativa
from theta_e_new import calcula_theta
from erro_RE_new import calcula_e_RE
from tE_new import te_calc
from impact_param_new import calcula_S
from erro_S_new import calcula_e_S
from calculo_A_new import calcula_A_mag
from erro_A_new import calcula_e_A
from calculo_fluxo_new import calcula_fluxo
from erro_fluxo_new import calcula_e_FL_FF
from calculo_delta_m_new import delta_m_calc
from erro_Delta_m_new import calcula_e_Delta_m

# -------------------- CONFIG --------------------
CHECKPOINT_PATH = Path("/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint.json")
PROGRESS_PATH   = Path("/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress.json")

# CSV de whitelist (contendo apenas pixels < 500000)
WHITELIST_CSV = Path("Healpix6_pixels_abaixo_500000.csv")
WHITELIST_COL = "healpix_6"

# -------------------- HELPERS --------------------
def last_number_from_name(p: Path) -> int:
    m = re.search(r'(\d+)(?=\D*$)', p.stem)
    return int(m.group(1)) if m else -1

def listar_csvs(input_dir: Path):
    return sorted(Path(input_dir).glob("*.csv"), key=last_number_from_name)

def save_checkpoint(arquivo_atual: str, etapa_atual: int):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"arquivo_atual": arquivo_atual, "etapa_atual": etapa_atual}, f, indent=2)

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {}

def load_progress() -> set:
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as f:
                data = json.load(f)
            return set(int(x) for x in data.get("pixels_concluidos", []))
        except Exception:
            return set()
    return set()

def save_progress(pixels_done: set):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"pixels_concluidos": sorted(list(pixels_done))}, f, indent=2)

def load_whitelist_pixels(csv_path: Path, col_name: str) -> set[int]:
    """
    Lê a whitelist (coluna healpix_6) e retorna um set de pixels permitidos.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Whitelist não encontrada: {csv_path.resolve()}")
    pixels = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or col_name not in reader.fieldnames:
            raise ValueError(f"Coluna '{col_name}' não encontrada na whitelist; colunas: {reader.fieldnames}")
        for row in reader:
            v = (row.get(col_name) or "").strip()
            if not v:
                continue
            try:
                pixels.add(int(v))
            except ValueError:
                continue
    if not pixels:
        raise ValueError("Whitelist vazia após leitura.")
    return pixels

# -------------------- Pipeline principal --------------------
def main():
    # 0) Carrega whitelist (contém apenas pixels < 500000)
    allowed_pixels = load_whitelist_pixels(WHITELIST_CSV, WHITELIST_COL)
    print(f"Whitelist carregada: {len(allowed_pixels)} pixels (ex.: {sorted(list(allowed_pixels))[:10]} ...)")

    # 1) Lista e FILTRA os CSVs de entrada pelos pixels permitidos
    all_inputs = listar_csvs(Path(INPUT_DIR))
    if not all_inputs:
        print(f"Nenhum arquivo CSV encontrado em {INPUT_DIR}")
        return

    arquivos_input = []
    ignorados = 0
    for p in all_inputs:
        pix = last_number_from_name(p)
        # só entra se o número do pixel estiver na whitelist
        if pix in allowed_pixels:
            arquivos_input.append(p)
        else:
            ignorados += 1

    if not arquivos_input:
        print("Nenhum arquivo corresponde à whitelist. Verifique nomes e a coluna healpix_6 do CSV.")
        return

    print(f"Arquivos a processar após filtro: {len(arquivos_input)} (ignorados: {ignorados})")

    # Etapas (apenas para contar quando finaliza um pixel)
    etapas = [
        ("1) Compare Loop Healpix", compare_loop_healpix),
        ("2) Propagate Robust", propagate_robust),
        ("3) Incertezas e Filtra", calcula_incerteza),
        ("4) Distância", calcular_distancia),
        ("5) Velocidades Rel.", calcular_velocidades_relativa),
        ("6) Theta", calcula_theta),
        ("7) Erro RE", calcula_e_RE),
        ("8) tE Calc", te_calc),
        ("9) Impact Param S", calcula_S),
        ("10) Erro S", calcula_e_S),
        ("11) A Mag", calcula_A_mag),
        ("12) Erro A", calcula_e_A),
        ("13) Fluxo", calcula_fluxo),
        ("14) Erro Fluxo", calcula_e_FL_FF),
        ("15) Delta m", delta_m_calc),
        ("16) Erro Delta m", calcula_e_Delta_m),
    ]
    TOTAL_ETAPAS = len(etapas)

    # Retomada
    checkpoint = load_checkpoint()
    ultimo_arquivo = checkpoint.get("arquivo_atual")
    ultima_etapa   = checkpoint.get("etapa_atual", 0)

    pixels_done = load_progress()

    for arquivo in arquivos_input:
        nome = arquivo.stem
        pixel = last_number_from_name(arquivo)

        # Se já concluído em execução anterior, pula
        if pixel in pixels_done:
            continue

        # Se retomando e já passamos desse arquivo, pula
        if ultimo_arquivo and nome < ultimo_arquivo:
            continue

        print(f"\n=== Iniciando pipeline para: {nome} (pixel {pixel}) ===")
        csv_path = None

        for i, (etapa_nome, func) in enumerate(etapas, start=1):
            if ultimo_arquivo == nome and i <= ultima_etapa:
                continue

            print(f"\nExecutando etapa {i}/{TOTAL_ETAPAS}: {etapa_nome}...")

            try:
                if func.__name__ == "compare_loop_healpix":
                    csv_path = func(tbl=arquivo)
                else:
                    csv_path = func(csv_input=csv_path)

                # Atualiza checkpoint (o viewer usa para marcar o pixel atual)
                save_checkpoint(nome, i)

            except Exception as e:
                print(f"Erro na etapa {i} ({etapa_nome}): {e}")
                print("Pipeline pausado. Corrija o erro e rode novamente para retomar.")
                save_checkpoint(nome, i)
                return

        print(f"Pipeline finalizado para: {nome}")

        # Marca pixel como concluído (o viewer passa a plotar bolinha permanente)
        pixels_done.add(pixel)
        save_progress(pixels_done)
        save_checkpoint(nome, TOTAL_ETAPAS)

    print("\nTodos os arquivos foram processados com sucesso!")
    # Opcional: limpar checkpoint ao final
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

if __name__ == "__main__":
    main()
