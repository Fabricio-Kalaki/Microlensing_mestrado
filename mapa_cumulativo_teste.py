# mapa_multifontes_minimo.py
import json, time, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import healpy as hp
import matplotlib.pyplot as plt

# ================== CONFIG DO MAPA ==================
NSIDE = 64
NEST  = True
COORD = ['G']
CMAP  = "inferno"
NORM  = "log"

DOT_MS_DONE = 3   # bolinha para concluídos
DOT_MS_CURR = 6   # bolinha para pixel atual

# ========= FONTES (NOVAS) =========
# Mantém o "principal" e adiciona os 6 mains novos (500k..1mi), com arquivos próprios
FONTES = [
    {
        "nome": "Main Principal",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint.json",
        "cor": "green",
    },
    {
        "nome": "500k",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_500k.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_500k.json",
        "cor": "deepskyblue",
    },
    {
        "nome": "600k",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_600k.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_600k.json",
        "cor": "cyan",
    },
    {
        "nome": "700k",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_700k.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_700k.json",
        "cor": "red",
    },
    {
        "nome": "800k",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_800k.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_800k.json",
        "cor": "yellow",
    },
    {
        "nome": "900k",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_900k.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_900k.json",
        "cor": "magenta",
    },
    {
        "nome": "1mi",
        "progress": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_progress_1mi.json",
        "checkpoint": "/srv/jupyterhub/shared/fabricio/Microlensing_mestrado/pipeline_checkpoint_1mi.json",
        "cor": "orange",
    },
]

# ============== detecção de Jupyter ==============
try:
    from IPython.display import display
    IN_JUPYTER = True
except Exception:
    display = None
    IN_JUPYTER = False

# ========================== HELPERS ==========================
def last_number_from_name(name: str) -> int:
    import re
    m = re.search(r'(\d+)(?=\D*$)', Path(name).stem)
    return int(m.group(1)) if m else -1

def _json_progress_pixels(progress_path: Path) -> set[int]:
    if not progress_path or not progress_path.exists():
        return set()
    try:
        with open(progress_path, "r") as f:
            data = json.load(f)
        return set(int(x) for x in data.get("pixels_concluidos", []))
    except Exception:
        return set()

def _json_checkpoint_pixel(ck_path: Path) -> int | None:
    if not ck_path or not ck_path.exists():
        return None
    try:
        with open(ck_path, "r") as f:
            d = json.load(f)
    except Exception:
        return None
    arq = d.get("arquivo_atual")
    if not arq:
        return None
    return last_number_from_name(arq)

def carregar_mapa_base(nome_arquivo: str, nside: int) -> np.ndarray:
    num_pixels = hp.nside2npix(nside)
    # Observação: a base deve ter colunas: healpix_6, n_objetos
    df = pd.read_csv(Path(nome_arquivo), dtype={'healpix_6': 'int64', 'n_objetos': 'float64'})
    base = np.full(num_pixels, hp.UNSEEN, dtype=float)
    base[df['healpix_6'].values] = df['n_objetos'].values
    return base

# ========================== DESENHO ==========================
def draw_into_figure(fig, base_map: np.ndarray, fontes: list[dict]):
    """Reusa a MESMA figura; desenha base e, por fonte, pontos concluídos e pixel atual."""
    fig.clf()

    vals = base_map[base_map != hp.UNSEEN]
    vmin = np.nanmin(vals) if vals.size else None
    vmax = np.nanmax(vals) if vals.size else None

    hp.disable_warnings()
    hp.mollview(
        base_map,
        nest=NEST,
        title=f"Mapa base + Progresso (múltiplas fontes) — NSIDE={NSIDE}",
        unit="Contagem de Estrelas",
        cmap=CMAP,
        norm=NORM,
        cbar=True,
        coord=COORD,
        min=vmin, max=vmax,
        fig=fig.number,
    )

    # concluídos e pixel atual por fonte
    import matplotlib.lines as mlines
    proxies, labels = [], []

    for fonte in fontes:
        cor = fonte.get("cor", "white")
        prog = Path(fonte["progress"]) if fonte.get("progress") else None
        ckpt = Path(fonte["checkpoint"]) if fonte.get("checkpoint") else None

        pixels_done = _json_progress_pixels(prog) if prog else set()
        pixel_atual = _json_checkpoint_pixel(ckpt) if ckpt else None

        if pixels_done:
            arr = np.fromiter(pixels_done, dtype=int)
            thetas, phis = hp.pix2ang(NSIDE, arr, nest=NEST)
            hp.projplot(thetas, phis, '.', alpha=0.8, ms=DOT_MS_DONE, color=cor)

        if pixel_atual is not None and pixel_atual >= 0:
            theta, phi = hp.pix2ang(NSIDE, pixel_atual, nest=NEST)
            hp.projplot(theta, phi, 'o', color=cor, ms=DOT_MS_CURR)

        proxies.append(mlines.Line2D([], [], color=cor, marker='o', linestyle='None', markersize=6))
        labels.append(fonte.get("nome", "job"))

    if proxies:
        plt.legend(proxies, labels, loc='lower left', fontsize=8, frameon=True)

# ========================== MAIN ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="CSV base com colunas healpix_6,n_objetos")
    parser.add_argument("--intervalo", type=float, default=3.0, help="segundos entre atualizações")
    args = parser.parse_args()

    # Carrega mapa base uma vez
    try:
        base_map = carregar_mapa_base(args.base, NSIDE)
    except Exception as e:
        print(f"Erro ao carregar mapa base: {e}")
        return

    # Uma figura e (em Jupyter) um display handle para atualizar o MESMO output
    plt.ioff()
    fig = plt.figure(num="Progresso HEALPix", figsize=(10, 6))
    handle = display(fig, display_id=True) if IN_JUPYTER else None

    # Estado para detectar mudanças (qualquer fonte)
    last_sig = None
    while True:
        sig_parts = []
        for f in FONTES:
            for p in (f.get("progress"), f.get("checkpoint")):
                if p:
                    P = Path(p)
                    sig_parts.append((p, P.stat().st_mtime if P.exists() else None))
        sig = tuple(sig_parts)

        if sig != last_sig:
            draw_into_figure(fig, base_map, FONTES)
            if IN_JUPYTER and handle is not None:
                handle.update(fig)
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)
            last_sig = sig

        time.sleep(args.intervalo)

if __name__ == "__main__":
    main()
