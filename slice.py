#!/usr/bin/env python3
"""
slice.py — Découpage de la trace continue HRP Gavarnie→Loudenvielle en étapes.

La trace de référence est construite par trace.py (cirque→Pailha→HRP→La Soula→
descente Loudenvielle, ~92 km, km 0 = Gavarnie).

Modes d'usage :
    python slice.py                                  # itinéraire recommandé par défaut
    python slice.py --km 28 52 76 104 125 150
    python slice.py --refuges "Cabane d'Estaubé" "Cabane des Aguilous" ... "Loudenvielle"

Km-effort (charge) = distance + D+/100 + D-/300.

Produit dans output/ : J1_…_….gpx, J2_…, … + tableau récap.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import gpxpy.gpx
import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

from trace import build_continuous_trace, haversine_m

console = Console()

REFUGES_JSON = Path("output/refuges.json")
ITINERARY_JSON = Path("itinerary.json")
OUTPUT_DIR = Path("output")

SMOOTH_WINDOW = 9         # pts pour lissage élévation (anti-bruit GPS léger)
HYSTERESIS_M = 8          # mètre min pour compter gain/perte (D+ réaliste)
MAX_LOADED_OK = 36        # seuil km-effort par étape (profil sportif)
SNAP_SLEEP_M = 500        # distance max refuge↔trace pour servir de point de coupe

# Types OSM considérés comme refuges gardés
GUARDED_TYPES = {"alpine_hut", "refuge", "auberge"}

# Itinéraire recommandé par défaut — départ Pont d'Espagne (parking, Cauterets).
# On bivouaque À CÔTÉ des refuges gardés (pas de nuit en dur, pas de réservation).
DEFAULT_ITINERARY = [
    "Pont d'Espagne",                # départ (parking)
    "Refuge des Oulettes de Gaube",  # bivouac près gardé
    "Refuge de Bayssellance",        # bivouac près gardé — plus haut des Pyrénées
    "Refuge des Espuguettes",        # bivouac près gardé (via Gavarnie + cirque)
    "Auberge de la Munia",           # secteur Héas
    "Cabaña de Barrosa",             # cabane / bivouac
    "Refugio de Barannetas",         # abri / bivouac
    "Refugio de Viadós",             # bivouac près gardé
    "Refuge de la Soula",            # bivouac près gardé
    "Loudenvielle",
]


# ---------------------------------------------------------------------------
# Stats D+/D- + km-effort
# ---------------------------------------------------------------------------

def smooth_elevation(elevations: list[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    arr = np.array(elevations, dtype=float)
    if len(arr) < window:
        return arr
    return uniform_filter1d(arr, size=window, mode="nearest")


def compute_dplus_dminus(ele_smooth: np.ndarray, hysteresis: float = HYSTERESIS_M) -> tuple[float, float]:
    """D+/D- avec filtre hystérèse pour éviter le bruit."""
    dplus = dminus = 0.0
    ref = ele_smooth[0]
    for e in ele_smooth[1:]:
        diff = e - ref
        if diff > hysteresis:
            dplus += diff
            ref = e
        elif diff < -hysteresis:
            dminus += abs(diff)
            ref = e
    return dplus, dminus


def loaded_km(dist_km: float, dplus: float, dminus: float) -> float:
    """Km-effort = distance + 1 km / 100 m D+ + 1 km / 300 m D-."""
    return dist_km + dplus / 100.0 + dminus / 300.0


def slugify(name: str) -> str:
    name = name.strip()
    for src, dst in [("é","e"),("è","e"),("ê","e"),("à","a"),("â","a"),("á","a"),
                     ("î","i"),("ï","i"),("ô","o"),("ó","o"),("ù","u"),("û","u"),
                     ("ñ","n"),("ç","c"),(" ","_"),("/","_"),("-","_"),("'","")]:
        name = name.replace(src, dst)
    return re.sub(r"[^A-Za-z0-9_]", "", name)


# ---------------------------------------------------------------------------
# Résolution des points de coupure
# ---------------------------------------------------------------------------

def km_to_idx(km_target: float, km_array: list) -> int:
    return int(np.argmin(np.abs(np.array(km_array) - km_target)))


def build_snapper(pts: list):
    cos_lat = math.cos(math.radians(sum(p[0] for p in pts) / len(pts)))
    tree = cKDTree(np.array([[p[0], p[1] * cos_lat] for p in pts]))
    return tree, cos_lat


def resolve_refuges(names: list[str], pts: list, km: list) -> list[tuple[int, str]]:
    """Retourne [(idx, nom)] pour chaque nom. 'Gavarnie'=début, 'Loudenvielle'=fin."""
    refuges_db: dict[str, dict] = {}
    if REFUGES_JSON.exists():
        with open(REFUGES_JSON, encoding="utf-8") as f:
            for r in json.load(f):
                refuges_db[r["nom"].lower()] = r

    tree, cos_lat = build_snapper(pts)
    out: list[tuple[int, str]] = []
    for pos, name in enumerate(names):
        lower = name.lower()
        if pos == 0:
            # Le premier point est toujours le départ = début de la trace (trailhead)
            out.append((0, name))
            continue
        if lower == "loudenvielle":
            out.append((len(pts) - 1, "Loudenvielle"))
            continue
        matched = next((r for k, r in refuges_db.items() if lower in k or k in lower), None)
        if matched:
            _, idx = tree.query([[matched["lat"], matched["lon"] * cos_lat]])
            out.append((int(idx[0]), matched["nom"]))
        else:
            console.print(f"[yellow]  Refuge inconnu : '{name}' — ignoré[/yellow]")
    return out


# ---------------------------------------------------------------------------
# Liste des refuges candidats (points de coupe possibles)
# ---------------------------------------------------------------------------

def candidate_stops(pts: list, km: list) -> list[dict]:
    """Refuges à <500 m de la trace, triés par km, avec km-effort cumulé."""
    if not REFUGES_JSON.exists():
        return []
    refs = json.load(open(REFUGES_JSON, encoding="utf-8"))
    tree, cos_lat = build_snapper(pts)

    ele_smooth = smooth_elevation([p[2] for p in pts])
    for r in refs:
        _, idx = tree.query([[r["lat"], r["lon"] * cos_lat]])
        i = int(idx[0])
        r["i"] = i
        r["km_cont"] = km[i]
        r["dist_cont"] = haversine_m(r["lat"], r["lon"], pts[i][0], pts[i][1])
        dplus, dminus = compute_dplus_dminus(ele_smooth[: i + 1])
        r["charge"] = loaded_km(km[i], dplus, dminus)

    on = [r for r in refs if r["dist_cont"] <= SNAP_SLEEP_M and r["km_cont"] > 0.3]
    on.sort(key=lambda r: r["km_cont"])
    return on


def load_itinerary() -> list[str]:
    if ITINERARY_JSON.exists():
        with open(ITINERARY_JSON, encoding="utf-8") as f:
            return [e["name"] for e in json.load(f)]
    return DEFAULT_ITINERARY


def load_guarded_names() -> set[str]:
    names: set[str] = set()
    if REFUGES_JSON.exists():
        for r in json.load(open(REFUGES_JSON, encoding="utf-8")):
            if r["type"] in GUARDED_TYPES:
                names.add(r["nom"])
    return names


def compute_stages(pts: list, km: list, breaks: list[tuple[int, str]],
                   guarded_names: set[str]) -> list[dict]:
    """Calcule les stats par étape. Retourne une liste de dicts (sans I/O)."""
    stages: list[dict] = []
    for j in range(len(breaks) - 1):
        i0, nom0 = breaks[j]
        i1, nom1 = breaks[j + 1]
        seg_pts = pts[i0 : i1 + 1]
        seg_km = km[i0 : i1 + 1]
        dist = seg_km[-1] - seg_km[0]
        ele_sm = smooth_elevation([p[2] for p in seg_pts])
        dplus, dminus = compute_dplus_dminus(ele_sm)
        stages.append({
            "day": j + 1, "start": nom0, "end": nom1,
            "i0": i0, "i1": i1, "pts": seg_pts,
            "dist": dist, "dplus": dplus, "dminus": dminus,
            "charge": loaded_km(dist, dplus, dminus),
            "ele_start": seg_pts[0][2], "ele_end": seg_pts[-1][2],
            "guarded": nom1 in guarded_names,
            "is_end": nom1.lower() == "loudenvielle",
        })
    return stages


# ---------------------------------------------------------------------------
# Export GPX
# ---------------------------------------------------------------------------

def export_stage_gpx(stage_pts: list, stage_name: str, output_path: Path) -> None:
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=stage_name)
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    for lat, lon, ele in stage_pts:
        seg.points.append(gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon, elevation=ele))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(gpx.to_xml())


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Découpe la trace HRP en étapes journalières.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--km", nargs="+", type=float, metavar="KM",
                       help="Points de coupure en km le long de la trace.")
    group.add_argument("--refuges", nargs="+", metavar="NOM",
                       help='Noms des refuges (dans l\'ordre). "Gavarnie"=début, "Loudenvielle"=fin.')
    parser.add_argument("--list", action="store_true",
                        help="Liste les refuges candidats (km, km-effort, gardé) et sort.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Construction de la trace continue…[/bold blue]")
    tr = build_continuous_trace()
    pts, km = tr["pts"], tr["km"]
    total_km = tr["total_km"]
    ele_full = smooth_elevation([p[2] for p in pts])
    dpt, dmt = compute_dplus_dminus(ele_full)
    console.print(
        f"  {len(pts):,} pts | {total_km:.1f} km | D+ {dpt:.0f} m | D- {dmt:.0f} m | "
        f"charge totale {loaded_km(total_km, dpt, dmt):.1f} km-effort"
    )

    # --- Mode liste ---
    if args.list:
        stops = candidate_stops(pts, km)
        t = Table(title="Refuges candidats", header_style="bold cyan", show_lines=False)
        t.add_column("km", justify="right")
        t.add_column("charge", justify="right")
        t.add_column("alt", justify="right")
        t.add_column("dist", justify="right")
        t.add_column("gardé", justify="center")
        t.add_column("nom", min_width=30)
        for r in stops:
            g = "[green]G[/green]" if r["type"] in GUARDED_TYPES else "·"
            ele = f"{r['ele']:.0f}" if r.get("ele") else "?"
            t.add_row(f"{r['km_cont']:.1f}", f"{r['charge']:.1f}", ele,
                      f"{r['dist_cont']:.0f} m", g, r["nom"])
        console.print(t)
        return

    # --- Résolution des points de coupure ---
    if args.km:
        breaks = [(km_to_idx(k, km), f"km{k:.0f}") for k in args.km]
        breaks[0] = (0, "Gavarnie")
    else:
        if args.refuges:
            names = args.refuges
        else:
            names = load_itinerary()
            src = ITINERARY_JSON if ITINERARY_JSON.exists() else "DEFAULT_ITINERARY"
            console.print(f"[dim]  (itinéraire depuis {src})[/dim]")
        breaks = resolve_refuges(names, pts, km)

    # Dédup + ordre croissant
    clean: list[tuple[int, str]] = [breaks[0]]
    for idx, nom in breaks[1:]:
        if idx > clean[-1][0]:
            clean.append((idx, nom))
    breaks = clean
    if len(breaks) < 2:
        console.print("[red]Au moins 2 points de coupure requis.[/red]")
        sys.exit(1)

    # Refuges gardés (pour annoter la fin de chaque étape)
    guarded_names = load_guarded_names()
    stages = compute_stages(pts, km, breaks, guarded_names)

    # --- Tableau récap ---
    table = Table(title="Étapes — HRP Gavarnie → Loudenvielle", show_lines=True,
                  header_style="bold cyan")
    table.add_column("Étape", style="bold", min_width=34, no_wrap=False)
    table.add_column("Dist.", justify="right")
    table.add_column("D+", justify="right", style="yellow")
    table.add_column("D-", justify="right", style="cyan")
    table.add_column("Charge", justify="right", style="bold")
    table.add_column("Alt.", justify="right")
    table.add_column("Couchage", justify="center")

    console.print(f"\n[bold blue]Génération de {len(breaks) - 1} étape(s)…[/bold blue]")

    tot_dist = tot_dp = tot_dm = tot_charge = 0.0
    for s in stages:
        tot_dist += s["dist"]; tot_dp += s["dplus"]; tot_dm += s["dminus"]; tot_charge += s["charge"]

        over = s["charge"] > MAX_LOADED_OK
        charge_str = f"[red]{s['charge']:.1f}[/red]" if over else f"{s['charge']:.1f}"
        couchage = "[green]⛺ près gardé[/green]" if s["guarded"] else (
            "fin" if s["is_end"] else "cabane/biv")

        fname = f"J{s['day']}_{slugify(s['start'])}_{slugify(s['end'])}.gpx"
        export_stage_gpx(s["pts"], f"J{s['day']} {s['start']} → {s['end']}", OUTPUT_DIR / fname)

        table.add_row(
            f"J{s['day']}  {s['start']} → {s['end']}",
            f"{s['dist']:.1f} km",
            f"{s['dplus']:.0f} m",
            f"{s['dminus']:.0f} m",
            f"{charge_str} km",
            f"{s['ele_start']:.0f}→{s['ele_end']:.0f}",
            couchage,
        )
        if over:
            console.print(f"[yellow]  ⚠  J{s['day']} ({s['start']} → {s['end']}) : {s['charge']:.1f} km-effort > {MAX_LOADED_OK}[/yellow]")

    table.add_row("[bold]TOTAL[/bold]", f"[bold]{tot_dist:.1f} km[/bold]",
                  f"[bold]{tot_dp:.0f} m[/bold]", f"[bold]{tot_dm:.0f} m[/bold]",
                  f"[bold]{tot_charge:.1f} km[/bold]", "", "")
    console.print(table)
    console.print(f"\n[bold green]✓ GPX exportés dans {OUTPUT_DIR}/[/bold green]")


if __name__ == "__main__":
    main()
