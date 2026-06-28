#!/usr/bin/env python3
"""
trace.py — Source de vérité de la trace continue HRP Gavarnie→Loudenvielle.

Assemble bout à bout les 3 fichiers GPX sources en UNE seule séquence de points
continue (cirque de Gavarnie → HRP → descente Loudenvielle), avec un km cumulé
monotone de 0 à ~100 km. Cette trace unique sert à la visualisation, au profil
et au découpage en étapes.

Utilisation directe :
    python trace.py          # (ré)génère output/trace_complete.gpx
"""

import math
from pathlib import Path

import gpxpy
import gpxpy.gpx
import numpy as np
from scipy.spatial import cKDTree

# Fichiers sources
GPX_APPROACH     = "gpx/pont-d-espagne-refuge-des-oulettes-de-gaube.gpx"
GPX_CIRQUE       = "gpx/cirque-de-gavarnie.gpx"
GPX_HRP          = ["gpx/hrp-68-91.gpx", "gpx/hrp-91-117.gpx"]
GPX_LOUDENVIELLE = "gpx/loudenvielle-soula.gpx"

# Points clés de découpe (lat, lon)
OULETTES_COORDS = (42.79291, -0.14143)   # Refuge des Oulettes de Gaube — raccord approche ↔ HRP
GAVARNIE_COORDS = (42.7378, -0.0191)     # village de Gavarnie sur l'HRP
PAILHA_COORDS   = (42.7196061, 0.0005074)  # Chalet du Pailha — jonction cirque ↔ HRP
SOULA_COORDS    = (42.72316, 0.41952)    # jonction La Soula (HRP ↔ descente)

# Au-delà de 6 km dans loudenvielle-soula.gpx on est déjà sur le retour ;
# on cherche La Soula dans la première moitié de la boucle.
LOUDENVIELLE_SEARCH_KM = 6.0

OUTPUT_GPX = Path("output/trace_complete.gpx")


# ---------------------------------------------------------------------------
# Utilitaires géodésiques
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_km_array(pts: list) -> list:
    km = [0.0]
    for i in range(1, len(pts)):
        km.append(km[-1] + haversine_m(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1]) / 1000.0)
    return km


def load_gpx(path: str | Path) -> list[tuple[float, float, float]]:
    with open(path, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    return [
        (pt.latitude, pt.longitude, pt.elevation or 0.0)
        for track in gpx.tracks
        for seg in track.segments
        for pt in seg.points
    ]


def find_nearest_idx(pts: list, lat: float, lon: float) -> int:
    cos_lat = math.cos(math.radians(lat))
    arr = np.array([[p[0], p[1] * cos_lat] for p in pts])
    _, idx = cKDTree(arr).query([[lat, lon * cos_lat]])
    return int(idx[0])


# ---------------------------------------------------------------------------
# Assemblage de la trace continue
# ---------------------------------------------------------------------------

def build_continuous_trace() -> dict:
    """
    Retourne un dict décrivant la trace continue :
      pts      : list[(lat, lon, ele)] — UNE séquence continue
      km       : list[float]           — km cumulé monotone (0 → total)
      bounds   : list[(nom, i_start, i_end)] — frontières des portions
      total_km : float

    Ordre : Pont d'Espagne → (approche) Oulettes de Gaube → (HRP, Vignemale :
            Bayssellance, Ossoue) → Gavarnie → tour du cirque → Pailha → (HRP)
            Espuguettes → … → La Soula → descente Loudenvielle.

    Subtilités :
    - Départ au Pont d'Espagne (parking, 1475 m) ; l'approche jusqu'au Refuge des
      Oulettes de Gaube vient du GPX dédié (8.7 km, +674 m), pas de l'HRP.
    - cirque-de-gavarnie.gpx est une BOUCLE fermée ; on n'en garde que Gavarnie→
      Pailha (la boucle frôle le Chalet du Pailha à 9 m) et on jette le retour.
    Raccords : approche[Oulettes]→HRP[Oulettes], HRP[Gavarnie]→cirque[0] ≈ 100 m,
    cirque[Pailha]→HRP[Pailha] ≈ 120 m, HRP[La Soula]→descente ≈ 23 m.
    """
    hrp = []
    for p in GPX_HRP:
        hrp.extend(load_gpx(p))
    oulettes_idx = find_nearest_idx(hrp, *OULETTES_COORDS)
    gav_idx      = find_nearest_idx(hrp, *GAVARNIE_COORDS)
    pailha_idx   = find_nearest_idx(hrp, *PAILHA_COORDS)
    soula_idx    = find_nearest_idx(hrp, *SOULA_COORDS)

    # --- Portion 1 : approche Pont d'Espagne → Oulettes de Gaube ---
    approach = load_gpx(GPX_APPROACH)

    # --- Portion 2 : HRP Oulettes → Gavarnie (Vignemale : Bayssellance, Ossoue) ---
    hrp_vignemale = hrp[oulettes_idx : gav_idx + 1]

    # --- Portion 3 : cirque de Gavarnie, coupé au Pailha ---
    cirque = load_gpx(GPX_CIRQUE)
    cirque_pailha_idx = find_nearest_idx(cirque, *PAILHA_COORDS)
    cirque_trim = cirque[: cirque_pailha_idx + 1]

    # --- Portion 4 : HRP du Pailha à La Soula (par Espuguettes) ---
    hrp_main = hrp[pailha_idx : soula_idx + 1]

    # --- Portion 5 : descente La Soula → Loudenvielle ---
    lou_loop = load_gpx(GPX_LOUDENVIELLE)
    lou_km = compute_km_array(lou_loop)
    search_end = next(i for i, k in enumerate(lou_km) if k >= LOUDENVIELLE_SEARCH_KM)
    soula_lou_idx = find_nearest_idx(lou_loop[: search_end + 1], *SOULA_COORDS)
    descente = list(reversed(lou_loop[: soula_lou_idx + 1]))

    # --- Concaténation en UNE séquence continue ---
    pts = approach + hrp_vignemale + cirque_trim + hrp_main + descente
    km = compute_km_array(pts)

    n1 = len(approach)
    n2 = len(hrp_vignemale)
    n3 = len(cirque_trim)
    n4 = len(hrp_main)
    n5 = len(descente)
    bounds = [
        ("Approche Pont d'Espagne→Oulettes", 0,                   n1 - 1),
        ("HRP Oulettes→Gavarnie (Vignemale)", n1,                 n1 + n2 - 1),
        ("Cirque Gavarnie→Pailha",            n1 + n2,            n1 + n2 + n3 - 1),
        ("HRP Pailha→La Soula",               n1 + n2 + n3,       n1 + n2 + n3 + n4 - 1),
        ("Descente Loudenvielle",             n1 + n2 + n3 + n4,  n1 + n2 + n3 + n4 + n5 - 1),
    ]

    return {
        "pts": pts,
        "km": km,
        "bounds": bounds,
        "total_km": km[-1],
    }


def save_gpx(pts: list, path: Path = OUTPUT_GPX) -> None:
    """Écrit la trace continue dans un fichier GPX mono-track mono-segment."""
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name="HRP Gavarnie → Loudenvielle")
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for lat, lon, ele in pts:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, elevation=ele))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())


def main() -> None:
    from rich.console import Console
    console = Console()

    console.print("[bold blue]Assemblage de la trace continue…[/bold blue]")
    trace = build_continuous_trace()
    pts, km, bounds = trace["pts"], trace["km"], trace["bounds"]

    for nom, i0, i1 in bounds:
        seg_km = km[i1] - km[i0]
        console.print(f"  {nom:28s} : {i1 - i0 + 1:6d} pts | km {km[i0]:6.1f} → {km[i1]:6.1f}  ({seg_km:.1f} km)")

    # Détection de sauts éventuels aux raccords
    max_gap = 0.0
    for i in range(1, len(pts)):
        d = haversine_m(pts[i-1][0], pts[i-1][1], pts[i][0], pts[i][1])
        max_gap = max(max_gap, d)
    console.print(f"  [dim]Plus grand saut entre 2 points consécutifs : {max_gap:.0f} m[/dim]")

    console.print(f"  [bold]Total : {trace['total_km']:.1f} km, {len(pts):,} points[/bold]")

    save_gpx(pts)
    console.print(f"[bold green]✓ {OUTPUT_GPX} écrit[/bold green]")


if __name__ == "__main__":
    main()
