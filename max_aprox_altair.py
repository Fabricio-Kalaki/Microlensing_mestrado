import numpy as np

# --- constants ---
RAD2MAS = (180.0/np.pi) * 3600.0 * 1e3     # mas per rad
MAS2ARCSEC = 1.0/1e3                       # arcsec per mas
EPS = 1e-18                                # for numerical safety


def _wrap_dra(ra2, ra1):
    """RA difference wrapped to [-pi, pi]."""
    return np.arctan2(np.sin(ra2 - ra1), np.cos(ra2 - ra1))


def compare_batch_one_to_many(
    ra1_rad, dec1_rad, pmra1_masyr, pmdec1_masyr,
    ra2_rad, dec2_rad, pmra2_masyr, pmdec2_masyr,
    clip_years=None
):
    """
    Fast vectorized compare(): one star (1) vs many stars (2).

    Inputs
    ------
    ra*_rad, dec*_rad : float or ndarray
        Positions in radians (ICRS).
    pmra*_masyr, pmdec*_masyr : float or ndarray
        Proper motions in mas/yr (pmRA already includes cos(dec) convention).

    Returns
    -------
    dt_years : ndarray, shape broadcast to ra2
    dmin_arcsec : ndarray, same shape
    """
    # 1) tangent-plane offsets in mas (relative to star 1 at its dec)
    #dra = _wrap_dra(ra2_rad, ra1_rad)
    dx_mas = ((ra2_rad - ra1_rad) * np.cos(dec1_rad)) * RAD2MAS
    dy_mas = (dec2_rad - dec1_rad)     * RAD2MAS

    # 2) relative PM in mas/yr
    dpmx = pmra1_masyr  - pmra2_masyr
    dpmy = pmdec1_masyr - pmdec2_masyr
    v2 = dpmx*dpmx + dpmy*dpmy

    # 3) best-approach time (years)
    num = dx_mas*dpmx + dy_mas*dpmy
    dt_years = num / (v2 + EPS)

    #if clip_years is not None:
    #    dt_years = np.clip(dt_years, -clip_years, clip_years)

    # 4) minimum distance at t* (mas)
    rx = dx_mas - dpmx * dt_years
    ry = dy_mas - dpmy * dt_years
    dmin_mas = np.sqrt(rx*rx + ry*ry)

    # If v2 ≈ 0, take distance at t=0 (no relative motion)
    still_mask = v2 < EPS
    if np.any(still_mask):
        dmin_mas = np.where(still_mask, np.nan, dmin_mas)
        # set dt=0 for these to avoid huge numbers
        dt_years = np.where(still_mask, np.nan, dt_years)

    return dt_years, dmin_mas * MAS2ARCSEC


def compare_batch_pairs(
    ra_rad, dec_rad, pmra_masyr, pmdec_masyr,
    idx_i, idx_j, clip_years=None
):
    """
    Same as above but for explicit index pairs (arrays of i,j with same length).
    Useful if you prebuild candidate pairs (e.g., by HEALPix/neighbor search).
    """
    ra1   = ra_rad[idx_i]
    dec1  = dec_rad[idx_i]
    ra2   = ra_rad[idx_j]
    dec2  = dec_rad[idx_j]
    pmra1 = pmra_masyr[idx_i]
    pmdec1= pmdec_masyr[idx_i]
    pmra2 = pmra_masyr[idx_j]
    pmdec2= pmdec_masyr[idx_j]

    return compare_batch_one_to_many(
        ra1, dec1, pmra1, pmdec1,
        ra2, dec2, pmra2, pmdec2,
        clip_years=clip_years
    )
