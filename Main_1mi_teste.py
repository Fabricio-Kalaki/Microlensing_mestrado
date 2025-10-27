# ===== Limite de threads BLAS/NumPy =====
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import re
from pathlib import Path
from multiprocessing import Pool, get_context, cpu_count
import pandas as pd

# ===== Imports do pipeline =====
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
# Lê TODOS os pixels da tabela 1 milhão (na ordem do CSV)
WHITELIST_CSV = Path("Healpix6_pixels_1mi.csv")
WHITELIST_COL = "healpix_6"   # coluna com o índice do pixel
LIMIT_N = None                # None = todos; defina um int para limitar opcionalmente

# Checkpoints/Progresso/Logs (por pixel, como no 700k paralelo)
CHECKPOINT_DIR = Path("/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoints_1mi")
PROGRESS_PATH  = Path("/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_1mi.json")
LOG_DIR        = Path(OUTPUT_DIR) / "logs_pipeline_1mi"

# Paralelismo
DEFAULT_PROCS = max(1, (cpu_count() or 2) - 1)
CHUNKSIZE = 1  # bom para balancear quando os tempos variam por pixel

# -------------------- HELPERS --------------------
def last_number_from_name(p: Path) -> int:
    m = re.search(r'(\d+)(?=\D*$)', p.stem)
    return int(m.group(1)) if m else -1

def listar_csvs(input_dir: Path):
    return sorted(Path(input_dir).glob("*.csv"), key=last_number_from_name)

def per_pixel_ckpt_path(pixel: int) -> Path:
    return CHECKPOINT_DIR / f"ckpt_pixel_{pixel}.json"

def save_progress(pixels_done: set[int]):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"pixels_concluidos": sorted(list(pixels_done))}, f, indent=2)

def load_progress() -> set[int]:
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH, "r") as f:
                data = json.load(f)
            return set(int(x) for x in data.get("pixels_concluidos", []))
        except Exception:
            return set()
    return set()

def load_pixels(csv_path: Path, col_name: str, limit_n: int | None = None) -> list[int]:
    """
    Lê a coluna 'col_name' do CSV (na ordem do arquivo), detectando automaticamente o separador.
    Retorna todos os pixels (ou os primeiros 'limit_n' se definido).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Tabela/whitelist não encontrada: {csv_path.resolve()}")

    df = pd.read_csv(csv_path, sep=None, engine="python")  # autodetecta vírgula/;/\t/|
    cols = {c.lower(): c for c in df.columns}
    if col_name not in df.columns:
        alt = cols.get(col_name.lower())
        if alt:
            df.rename(columns={alt: col_name}, inplace=True)
        else:
            raise ValueError(f"Coluna '{col_name}' não encontrada; colunas: {list(df.columns)}")

    vals = (
        df[col_name].astype(str).str.strip()
          .str.extract(r'(\d+)')[0].dropna().astype(int).tolist()
    )
    if limit_n is not None:
        vals = vals[:limit_n]
    if not vals:
        raise ValueError("Nenhum pixel válido encontrado na tabela.")
    return vals

# -------------------- PIPELINE (etapas) --------------------
def build_etapas():
    return [
        ("1) Compare Loop Healpix", compare_loop_healpix),
        ("2) Propagate Robust",    propagate_robust),
        ("3) Incertezas e Filtra", calcula_incerteza),
        ("4) Distância",           calcular_distancia),
        ("5) Velocidades Rel.",    calcular_velocidades_relativa),
        ("6) Theta",               calcula_theta),
        ("7) Erro RE",             calcula_e_RE),
        ("8) tE Calc",             te_calc),
        ("9) Impact Param S",      calcula_S),
        ("10) Erro S",             calcula_e_S),
        ("11) A Mag",              calcula_A_mag),
        ("12) Erro A",             calcula_e_A),
        ("13) Fluxo",              calcula_fluxo),
        ("14) Erro Fluxo",         calcula_e_FL_FF),
        ("15) Delta m",            delta_m_calc),
        ("16) Erro Delta m",       calcula_e_Delta_m),
    ]

# -------------------- Worker: um arquivo/pixel --------------------
def run_one_file(arquivo: Path) -> dict:
    """Executa o pipeline completo para UM arquivo/pixel (com checkpoint e log por pixel)."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    nome = arquivo.stem
    pixel = last_number_from_name(arquivo)
    log_path = LOG_DIR / f"pipeline_{nome}.log"
    ckpt_path = per_pixel_ckpt_path(pixel)

    # Retomada por pixel
    ultima_etapa = 0
    if ckpt_path.exists():
        try:
            with open(ckpt_path, "r") as f:
                data = json.load(f)
            ultima_etapa = int(data.get("etapa_atual", 0))
        except Exception:
            ultima_etapa = 0

    etapas = build_etapas()
    total = len(etapas)

    csv_path = None
    try:
        with open(log_path, "a") as log:
            print(f"=== Iniciando pipeline para: {nome} (pixel {pixel}) ===", file=log)
            for i, (etapa_nome, func) in enumerate(etapas, start=1):
                if i <= ultima_etapa:
                    continue
                print(f"Executando etapa {i}/{total}: {etapa_nome}...", file=log)
                if func.__name__ == "compare_loop_healpix":
                    csv_path = func(tbl=arquivo)   # 1ª etapa recebe o Path do CSV Gaia
                else:
                    csv_path = func(csv_input=csv_path)
                # checkpoint por pixel
                with open(ckpt_path, "w") as f:
                    json.dump({"arquivo": nome, "pixel": pixel, "etapa_atual": i}, f, indent=2)

            print(f"Pipeline finalizado para: {nome}", file=log)

        # sucesso: remove ckpt do pixel
        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass

        return {"pixel": pixel, "nome": nome, "ok": True, "error": None}

    except Exception as e:
        with open(log_path, "a") as log:
            print(f"Erro no pixel {pixel} ({nome}): {e}", file=log)
        return {"pixel": pixel, "nome": nome, "ok": False, "error": str(e)}

# -------------------- Orquestrador paralelo --------------------
def main(n_procs: int | None = None):
    n_procs = int(n_procs) if n_procs is not None else DEFAULT_PROCS

    # 0) Pixels na ordem da whitelist 1mi
    selected_pixels_ordered = load_pixels(WHITELIST_CSV, WHITELIST_COL, LIMIT_N)
    selected_pixels_set = set(selected_pixels_ordered)
    print(
        f"Selecionados {len(selected_pixels_ordered)} pixels (ordem da tabela). "
        f"Ex.: {selected_pixels_ordered[:10]}"
    )

    # 1) CSVs de entrada
    all_inputs = listar_csvs(Path(INPUT_DIR))
    if not all_inputs:
        print(f"Nenhum arquivo CSV encontrado em {INPUT_DIR}")
        return

    # 2) pixel -> arquivo do catálogo
    pixel_to_file: dict[int, Path] = {}
    for p in all_inputs:
        pix = last_number_from_name(p)
        if pix in selected_pixels_set:
            pixel_to_file[pix] = p

    # 3) arquivos na ordem da whitelist
    arquivos_input: list[Path] = []
    faltando: list[int] = []
    for pix in selected_pixels_ordered:
        if pix in pixel_to_file:
            arquivos_input.append(pixel_to_file[pix])
        else:
            faltando.append(pix)

    if faltando:
        print(
            f"Aviso: {len(faltando)} pixels da tabela não têm CSV correspondente em {INPUT_DIR}: "
            f"{faltando[:15]}{' ...' if len(faltando) > 15 else ''}"
        )
    if not arquivos_input:
        print("Nenhum arquivo da Gaia_catalog corresponde aos pixels selecionados da tabela.")
        return

    # 4) Progresso global
    done = load_progress()
    arquivos_input = [a for a in arquivos_input if last_number_from_name(a) not in done]
    if not arquivos_input:
        print("Nada a processar: todos os arquivos já marcados como concluídos.")
        return

    print(
        f"Arquivos a processar agora: {len(arquivos_input)} "
        f"(execução paralela com {n_procs} processos, chunksize={CHUNKSIZE})."
    )

    # 5) Execução paralela (spawn é estável no Jupyter)
    with get_context("spawn").Pool(processes=n_procs) as pool:
        for result in pool.imap_unordered(run_one_file, arquivos_input, chunksize=CHUNKSIZE):
            pixel = result["pixel"]
            nome  = result["nome"]
            if result["ok"]:
                done.add(pixel)
                save_progress(done)
                print(f"Concluído: {nome} (pixel {pixel}). Progresso total: {len(done)}")
            else:
                print(
                    f"Erro em {nome} (pixel {pixel}): {result['error']}. "
                    f"Consulte o log em {LOG_DIR}/pipeline_{nome}.log. Continuação com outros pixels..."
                )

    print("\nExecução paralela finalizada.")
    print(f"Pixels concluídos: {len(done)}. Registros em {PROGRESS_PATH}.")

if __name__ == "__main__":
    # Ex.: python pipeline_1mi.py --procs 1
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--procs", type=int, default=None, help="CPUs lógicas a usar (processos)")
    args = p.parse_args()
    main(n_procs=args.procs)

