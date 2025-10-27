# -*- coding: utf-8 -*-
"""
@author: Fabrício Santos Kalaki e Altair Ramos Gomes Júnior
"""

import numpy as np
import pandas as pd
import astropy.constants as const
import astropy.units as u
import spiceypy as spice
from astropy.time import Time
import os
from tqdm import tqdm
from astropy.coordinates import SkyCoord, SphericalRepresentation
from config import OUTPUT_DIR

PI = np.pi
DEG2RAD = PI / 180.0
MAS2RAD = DEG2RAD / 3.6e6                 # 1 mas in rad
JULIAN_YEAR_DAYS = 365.25
DAY2YEAR = 1.0 / JULIAN_YEAR_DAYS
SECONDS_PER_YEAR = JULIAN_YEAR_DAYS * 86400.0
AU_KM = 149_597_870.7                     # IAU 2012, km
C_KM_S = 299_792.458
C_KM_YR = C_KM_S * SECONDS_PER_YEAR
TAU_A = AU_KM / C_KM_YR                   # A / c  (years)
KM_S_to_AU_YR = SECONDS_PER_YEAR / AU_KM  # km/s -> AU/yr

# Carrega os kernels SPICE
spice.furnsh('/srv/jupyterhub/shared/data/kernels/de440.bsp')

def bar_to_geo(pos_bar1, pos_bar2, time):
    dt = time - Time(2000, format='jyear', scale='tdb')
    position1 = spice.spkpos('399', dt.sec, 'J2000', 'NONE', '0')[0]
    position1 = SkyCoord(*position1.T * u.km, representation_type='cartesian')
    if np.any(pos_bar1.distance.unit.is_unity()) or np.any(np.isnan(pos_bar1.distance)):
        topo1 = pos_bar1
    else:
        topo1 = pos_bar1.cartesian - position1.cartesian
        topo1 = topo1.represent_as(SphericalRepresentation)
        topo1 = SkyCoord(topo1.lon, topo1.lat, topo1.distance)
    if np.any(pos_bar2.distance.unit.is_unity()) or np.any(np.isnan(pos_bar2.distance)):
        topo2 = pos_bar2
    else:
        topo2 = pos_bar2.cartesian - position1.cartesian
        topo2 = topo2.represent_as(SphericalRepresentation)
        topo2 = SkyCoord(topo2.lon, topo2.lat, topo2.distance)
    return topo1, topo2


def spatial_motion(ra, dec, pmra, pmdec, parallax=0, rad_vel=0,  dt=0, *,
    return_skycoord: bool = False):
    """Applies spatial motion to star coordinate.

    Parameters
    ----------
    ra `int`, `float`
        Right Ascension of the star at t=0 epoch, in deg.

    dec : `int`, `float`
        Declination of the star at t=0 epoch, in deg.

    pmra : `int`, `float`
        Proper Motion in RA of the star at t=0 epoch, in mas/year.

    pmdec : `int`, `float`
        Proper Motion in DEC of the star at t=0 epoch, in mas/year.

    parallax : `int`, `float`
        Parallax of the star at t=0 epoch, in mas.

    rad_vel : `int`, `float`
        Radial Velocity of the star at t=0 epoch, in km/s.

    dt : `int`, `float`
        Variation of time from catalogue epoch, in days.

    cov_matrix : `2D-array`
        6x6 covariance matrix.
    """

    # ---------- convert/prepare inputs with broadcasting ----------
    ra0   = np.asanyarray(ra) * DEG2RAD
    dec0  = np.asanyarray(dec) * DEG2RAD
    pmra0 = np.asanyarray(pmra) * MAS2RAD
    pmdec0= np.asanyarray(pmdec) * MAS2RAD
    par0  = np.asanyarray(parallax) * MAS2RAD
    rv0   = np.asanyarray(rad_vel) * KM_S_to_AU_YR   # AU/yr
    dt    = np.asanyarray(dt) * DAY2YEAR            # yr

    # This ensures all arrays share a common shape via broadcasting
    ra0, dec0, pmra0, pmdec0, par0, rv0, dt = np.broadcast_arrays(
        ra0, dec0, pmra0, pmdec0, par0, rv0, dt
    )
    S = ra0.shape  # final shape

    # treat non-positive parallax as "unknown distance"
    par_pos = par0 > 0.0
    # avoid division by zero downstream; use tiny positive for math
    par_safe = np.where(par_pos, par0, 1e-16)

    sra, cra = np.sin(ra0), np.cos(ra0)
    sde, cde = np.sin(dec0), np.cos(dec0)

    # local triad (S,3)
    p0 = np.stack((-sra,  cra,  np.zeros_like(sra)), axis=-1)
    q0 = np.stack((-sde*cra, -sde*sra, cde), axis=-1)
    r0 = np.stack(( cde*cra,  cde*sra, sde), axis=-1)

    # distances / light times
    b0 = AU_KM / par_safe                # km
    tau_0 = b0 / C_KM_YR                 # yr
    vec_b0 = r0 * b0[..., None]          # (S,3)

    # proper-motion vector on tangent plane (rad/yr along p0,q0)
    vec_mi0 = p0 * pmra0[..., None] + q0 * pmdec0[..., None]   # (S,3)
    mi0_sq = pmra0*pmra0 + pmdec0*pmdec0
    mi0 = np.sqrt(mi0_sq)

    # radial "proper motion" term
    mi_r0 = rv0 / b0  # ~ 1/yr in consistent units

    # apparent space velocity
    v0 = b0[..., None]*(r0*mi_r0[..., None] + vec_mi0)         # (S,3) km/yr
    v0_sq = np.sum(v0*v0, axis=-1)
    v0_norm = np.sqrt(v0_sq)

    # |vec_b0 + v0*dt|^2
    vb = vec_b0 + v0 * dt[..., None]
    vb2 = np.sum(vb*vb, axis=-1)

    # |v0 x b0|^2 = |v0|^2|b0|^2 - (v0·b0)^2
    v0_dot_b0 = np.sum(v0*vec_b0, axis=-1)
    b0_sq = np.sum(vec_b0*vec_b0, axis=-1)
    cross2 = v0_sq * b0_sq - v0_dot_b0*v0_dot_b0

    # time/distance/velocity scale factors
    f_T_num = (dt + 2.0*tau_0)
    f_T_den = tau_0 + (1.0 - v0_norm/C_KM_YR)*dt + np.sqrt(vb2 + (2.0*dt/(C_KM_YR**2 * tau_0))*cross2)/C_KM_YR
    f_T = f_T_num / f_T_den
    dtfT = dt * f_T

    f_D = np.sqrt(1.0 + 2.0*mi_r0*dtfT + (mi0_sq + mi_r0*mi_r0)*(dtfT*dtfT))
    f_V = 1.0 + (TAU_A/par_safe)*(mi_r0*(f_D - 1.0) + f_D*(mi0_sq + mi_r0*mi_r0)*dtfT)

    # new direction
    vec_u = (r0*(1.0 + mi_r0[..., None]*dtfT[..., None]) + vec_mi0*dtfT[..., None]) * f_D[..., None]  # (S,3)

    z = vec_u[..., 2]
    dec_new = np.arcsin(z)
    cdec_new = np.cos(dec_new)
    # guard against tiny numerical drift putting cos(dec)=0 then dividing; use arctan2(y,x) equivalence:
    x = vec_u[..., 0] / np.where(cdec_new != 0.0, cdec_new, 1.0)
    y = vec_u[..., 1] / np.where(cdec_new != 0.0, cdec_new, 1.0)
    ra_new = np.arctan2(y, x)

    # parallax & distance
    par_new = par_safe * f_D
    dist_km = AU_KM / par_new
    # where parallax was non-positive originally, mark distance as NaN (unknown)
    dist_km = np.where(par_pos, dist_km, np.nan)

    if not return_skycoord:
        return ra_new, dec_new, dist_km

    # Build a SkyCoord, attaching distance only where known
    # Split into two masks to avoid constructing two full SkyCoords unnecessarily
    if par_pos.any():
        # With distance for known-parallax elements
        sc_with = SkyCoord(ra=np.where(par_pos, ra_new, 0.0)*u.rad,
                           dec=np.where(par_pos, dec_new, 0.0)*u.rad,
                           distance=np.where(par_pos, dist_km, 1.0)*u.km)
        # Without distance for unknown-parallax elements
        sc_wo = SkyCoord(ra=np.where(~par_pos, ra_new, 0.0)*u.rad,
                         dec=np.where(~par_pos, dec_new, 0.0)*u.rad)
        # Merge fields back (SkyCoord supports masked stacking via frame attributes)
        # We combine by taking the spherical components from one or the other:
        ra_all  = ra_new*u.rad
        dec_all = dec_new*u.rad
        # Distances: quantity with NaN for unknown
        dist_all = np.where(par_pos, dist_km, np.nan) * u.km
        return SkyCoord(ra=ra_all, dec=dec_all, distance=dist_all)
    else:
        return SkyCoord(ra=ra_new*u.rad, dec=dec_new*u.rad)

def propagate_robust(
    csv_input: str = None,
    healpix_n: int = None,
    healpix_idx: int = None,
    save_csv: bool = True
) -> str:
    """
    Propaga posições robustas para pares de estrelas.

    csv_input: caminho do CSV de entrada (geralmente _predict.csv)
    healpix_n, healpix_idx: apenas para compatibilidade, não obrigatórios
    save_csv: salva arquivo atualizado no mesmo CSV
    """

    # --- Ajuste de caminho de entrada (sem mudar lógica) ---
    if csv_input is None:
        # Se nenhum arquivo for informado, pega o primeiro *_predict.csv no OUTPUT_DIR
        csv_files = sorted(OUTPUT_DIR.glob("*_predict.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo '_predict.csv' encontrado em {OUTPUT_DIR}")
        csv_path = csv_files[0]
    else:
        # Se for apenas nome, monta caminho dentro de OUTPUT_DIR
        csv_path = OUTPUT_DIR / csv_input
        if not str(csv_path).endswith(".csv"):
            csv_path = csv_path.with_suffix(".csv")

    # --- Leitura do CSV (sem alteração) ---
    df = pd.read_csv(csv_path, index_col='Indice')

    for col in ['dt_robusto (yr)', 'dmin_robusto (mas)']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    df.insert(
        loc=df.columns.get_loc('dt (yr)') + 1,
        column='dt_robusto (yr)',
        value=0.0)
    
    df.insert(
        loc=df.columns.get_loc('dmin (mas)') + 1,
        column='dmin_robusto (mas)',
        value=0.0)

    for indice, linha in tqdm(df.iterrows(),
                          total=len(df),
                          desc="Propagando posições para todos os pares"):

        # parâmetros da LENTE
        ra1 = linha['Lente_RA_ICRS (deg)']# * u.deg
        dec1 = linha['Lente_DE_ICRS (deg)']# * u.deg
        pmra1 = linha['Lente_pmRA (mas/yr)']# * u.mas/u.year
        pmdec1 = linha['Lente_pmDE (mas/yr)']# * u.mas/u.year
        plx1 = linha['Lente_Plx (mas)']#  * u.mas
        radvel1 = linha['Lente_RV (km/s)']# * u.km/u.s

        # parâmetros da FONTE
        ra2 = linha['Fonte_RA_ICRS (deg)']# * u.deg
        dec2 = linha['Fonte_DE_ICRS (deg)']# * u.deg
        pmra2 = linha['Fonte_pmRA (mas/yr)']# * u.mas/u.year
        pmdec2 = linha['Fonte_pmDE (mas/yr)']# * u.mas/u.year
        plx2 = linha['Fonte_Plx (mas)']# * u.mas
        radvel2 = linha['Fonte_RV (km/s)']# * u.km/u.s

        # Tempo
        dt_value = linha['dt (yr)']
        t0  = Time(dt_value, format='jyear')
        t_cat = Time(2016.0, format='jyear')

        tt = t0 + np.arange(-300, 300, 0.2) * u.day
        delta_jd = (tt - t_cat).jd

        pos1 = spatial_motion(
            ra=ra1, dec=dec1,
            pmra=pmra1, pmdec=pmdec1,
            parallax=plx1, rad_vel=radvel1,
            dt=delta_jd, return_skycoord=True)

        pos2 = spatial_motion(
            ra=ra2, dec=dec2,
            pmra=pmra2, pmdec=pmdec2,
            parallax=plx2, rad_vel=radvel2,
            dt=delta_jd, return_skycoord=True)

        pos1_geo, pos2_geo = bar_to_geo(pos1, pos2, tt)
        dist = pos1_geo.separation(pos2_geo)
        imin = dist.argmin()
        dmin = dist[imin].mas
        tmin = tt[imin].jyear

        df.loc[indice, 'dmin_robusto (mas)'] = dmin
        df.loc[indice, 'dt_robusto (yr)'] = tmin

    if save_csv:
        df.to_csv(csv_path)
        print(f"Tabela atualizada com propagação salva em: {csv_path}")

    return str(csv_path)