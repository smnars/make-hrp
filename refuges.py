#!/usr/bin/env python3
"""
refuges.py — Détection automatique des refuges sur la trace HRP Gavarnie→Loudenvielle

Utilisation :
    python refuges.py

Produit : output/refuges.json
"""

import json
import math
import time
from pathlib import Path

import gpxpy
import numpy as np
import requests
from rich.console import Console
from rich.table import Table
from scipy.spatial import cKDTree

console = Console()

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
]
MAX_DIST_M = 1500       # seuil de proximité à la trace
DEDUP_DIST_M = 300      # seuil de dédoublonnage waypoints connus / OSM
BBOX_MARGIN = 0.05      # marge en degrés pour la bounding box Overpass

# Mots dans le nom OSM indiquant une entrée parasite (granges, fermes…)
NAME_BLACKLIST = ["borda", "bordas", "vivac", "cabañas", "prau", "solana"]
# Entrées sans nom et sans altitude : probablement peu utiles
REQUIRE_NAME_OR_ELE = True

OVERPASS_HEADERS = {
    "User-Agent": "hrp-gpx-tool/1.0 (trek planning, contact: github)",
    "Accept": "application/json",
}

GPX_FILES = [
    Path("gpx/hrp-68-91.gpx"),
    Path("gpx/hrp-91-117.gpx"),
]
OUTPUT_FILE = Path("output/refuges.json")

WAYPOINTS_CONNUS = [
    {"nom": "Gavarnie village",       "lat": 42.7378, "lon": -0.0191, "ele": 1365, "type": "village"},
    {"nom": "Auberge du Maillet",     "lat": 42.736,  "lon":  0.086,  "ele": 1837, "type": "auberge"},
    {"nom": "Refuge des Espuguettes", "lat": 42.7225, "lon":  0.0098, "ele": 2027, "type": "refuge"},
    {"nom": "Lacs de Barroude bivouac","lat": 42.752,  "lon":  0.145,  "ele": 2350, "type": "bivouac"},
    {"nom": "Refuge de Viados",       "lat": 42.6607, "lon":  0.3772, "ele": 1760, "type": "refuge"},
    {"nom": "Refuge de La Soula",     "lat": 42.660,  "lon":  0.384,  "ele": 1690, "type": "refuge"},
    {"nom": "Cabane Aygues-Tortes",   "lat": 42.661,  "lon":  0.430,  "ele": 2100, "type": "cabane"},
    {"nom": "Refuge du Portillon",    "lat": 42.701,  "lon":  0.496,  "ele": 2571, "type": "refuge"},
]


# ---------------------------------------------------------------------------
# Utilitaires géodésiques
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance en mètres entre deux points (lat/lon en degrés)."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Chargement de la trace
# ---------------------------------------------------------------------------

def load_hrp_trace() -> list[tuple[float, float, float]]:
    """Charge et concatène les deux fichiers HRP (gap 36 m à la jonction → concat directe)."""
    points: list[tuple[float, float, float]] = []
    for gpx_path in GPX_FILES:
        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        for track in gpx.tracks:
            for segment in track.segments:
                for pt in segment.points:
                    points.append((pt.latitude, pt.longitude, pt.elevation or 0.0))
    return points


def compute_km_array(points: list[tuple[float, float, float]]) -> list[float]:
    """Tableau cumulatif de distances (km) depuis le début de la trace."""
    km = [0.0]
    for i in range(1, len(points)):
        d = haversine_m(points[i - 1][0], points[i - 1][1], points[i][0], points[i][1])
        km.append(km[-1] + d / 1000.0)
    return km


# ---------------------------------------------------------------------------
# KD-Tree pour snap géométrique rapide
# ---------------------------------------------------------------------------

def build_kdtree(points: list[tuple[float, float, float]]) -> tuple[cKDTree, float]:
    """
    Construit un cKDTree sur coordonnées lat/lon pondérées par cos(lat_moy).
    Retourne (tree, cos_lat) — cos_lat sert à convertir les requêtes.
    """
    lat_mean = math.radians(sum(p[0] for p in points) / len(points))
    cos_lat = math.cos(lat_mean)
    arr = np.array([[p[0], p[1] * cos_lat] for p in points], dtype=np.float64)
    return cKDTree(arr), cos_lat


def snap_to_trace(
    lat: float,
    lon: float,
    tree: cKDTree,
    cos_lat: float,
    km_array: list[float],
    points: list[tuple[float, float, float]],
) -> tuple[float, float]:
    """
    Retourne (km_sur_trace, distance_m) du point de la trace le plus proche.
    Distance euclidienne projetée pour la recherche, haversine pour la mesure finale.
    """
    query = np.array([[lat, lon * cos_lat]])
    _dist_proj, idx = tree.query(query, k=1)
    nearest = points[idx[0]]
    dist_m = haversine_m(lat, lon, nearest[0], nearest[1])
    return km_array[idx[0]], dist_m


# ---------------------------------------------------------------------------
# Requête Overpass
# ---------------------------------------------------------------------------

def fetch_overpass(bbox_str: str, max_retries: int = 3) -> dict | None:
    """Interroge l'API Overpass ; essaie plusieurs miroirs avec retry (délai croissant)."""
    query = f"""
[out:json][timeout:30];
(
  node["tourism"="alpine_hut"]{bbox_str};
  node["tourism"="wilderness_hut"]{bbox_str};
  node["amenity"="shelter"]{bbox_str};
  way["tourism"="alpine_hut"]{bbox_str};
  way["tourism"="wilderness_hut"]{bbox_str};
  way["amenity"="shelter"]{bbox_str};
  relation["tourism"="alpine_hut"]{bbox_str};
  relation["tourism"="wilderness_hut"]{bbox_str};
);
out center;
"""
    for server in OVERPASS_SERVERS:
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    server,
                    data={"data": query},
                    headers=OVERPASS_HEADERS,
                    timeout=50,
                )
                resp.raise_for_status()
                console.print(f"  [dim]Overpass OK via {server}[/dim]")
                return resp.json()
            except requests.RequestException as exc:
                delay = 5 * (attempt + 1)
                if attempt < max_retries - 1:
                    console.print(
                        f"[yellow]  {server} tentative {attempt + 1}/{max_retries} : {exc}"
                        f" — retry dans {delay} s…[/yellow]"
                    )
                    time.sleep(delay)
                else:
                    console.print(f"[yellow]  {server} échoué ({exc})[/yellow]")
    console.print("[red]Tous les serveurs Overpass sont inaccessibles[/red]")
    return None


# ---------------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1. Chargement de la trace HRP
    console.print("[bold blue]Chargement de la trace HRP…[/bold blue]")
    points = load_hrp_trace()
    km_array = compute_km_array(points)
    total_km = km_array[-1]
    console.print(f"  {len(points):,} points — longueur totale : {total_km:.1f} km")

    tree, cos_lat = build_kdtree(points)

    # 2. Bounding box avec marge
    lats = [p[0] for p in points]
    lons = [p[1] for p in points]
    south = min(lats) - BBOX_MARGIN
    west  = min(lons) - BBOX_MARGIN
    north = max(lats) + BBOX_MARGIN
    east  = max(lons) + BBOX_MARGIN
    # OverpassQL bbox syntax: (south,west,north,east) avec parenthèses rondes
    bbox_str = f"({south:.4f},{west:.4f},{north:.4f},{east:.4f})"
    console.print(f"  BBox (avec marge {BBOX_MARGIN}°) : {bbox_str}")

    # 3. Requête Overpass
    console.print("\n[bold blue]Interrogation Overpass API…[/bold blue]")
    data = fetch_overpass(bbox_str)

    refuges: list[dict] = []

    if data:
        elements = data.get("elements", [])
        console.print(f"  {len(elements)} éléments OSM dans la zone")

        for el in elements:
            if el["type"] in ("way", "relation"):
                lat = el["center"]["lat"]
                lon = el["center"]["lon"]
            else:
                lat = el["lat"]
                lon = el["lon"]

            tags = el.get("tags", {})
            nom = tags.get("name") or tags.get("ref") or f"Abri OSM #{el['id']}"
            ele_raw = tags.get("ele")
            try:
                ele = float(ele_raw) if ele_raw else None
            except (ValueError, TypeError):
                ele = None

            # Filtre : exclure granges/fermes aragonaises et abris sans intérêt
            nom_lower = nom.lower()
            if any(w in nom_lower for w in NAME_BLACKLIST):
                continue
            if REQUIRE_NAME_OR_ELE and nom.startswith("Abri OSM #") and ele is None:
                continue

            km_trace, dist_m = snap_to_trace(lat, lon, tree, cos_lat, km_array, points)

            if dist_m <= MAX_DIST_M:
                osm_type = tags.get("tourism") or tags.get("amenity") or "unknown"
                refuges.append({
                    "nom": nom,
                    "lat": lat,
                    "lon": lon,
                    "ele": ele,
                    "km_trace": round(km_trace, 2),
                    "distance_trace_m": round(dist_m),
                    "source": "osm",
                    "type": osm_type,
                })

        console.print(
            f"  [green]{len(refuges)} refuges/abris à moins de {MAX_DIST_M} m de la trace[/green]"
        )
    else:
        console.print("[yellow]Pas de données Overpass — passage aux waypoints connus uniquement[/yellow]")

    # 4. Fusion avec WAYPOINTS_CONNUS
    console.print(f"\n[bold blue]Fusion avec {len(WAYPOINTS_CONNUS)} waypoints connus…[/bold blue]")
    added = 0
    for wp in WAYPOINTS_CONNUS:
        km_trace, dist_m = snap_to_trace(wp["lat"], wp["lon"], tree, cos_lat, km_array, points)

        # Dédoublonnage : un refuge OSM < 300 m → ignorer le waypoint connu
        is_duplicate = any(
            haversine_m(wp["lat"], wp["lon"], r["lat"], r["lon"]) < DEDUP_DIST_M
            for r in refuges
        )
        if is_duplicate:
            console.print(f"  [dim]{wp['nom']} — doublon OSM, ignoré[/dim]")
            continue

        refuges.append({
            "nom": wp["nom"],
            "lat": wp["lat"],
            "lon": wp["lon"],
            "ele": wp.get("ele"),
            "km_trace": round(km_trace, 2),
            "distance_trace_m": round(dist_m),
            "source": "known",
            "type": wp.get("type", "refuge"),
        })
        added += 1

    console.print(f"  {added} waypoints connus ajoutés (sur {len(WAYPOINTS_CONNUS)})")

    # 5. Dedup par nom identique : garder le plus proche de la trace
    def normalize_name(n: str) -> str:
        return n.lower().strip()

    seen: dict[str, int] = {}  # nom normalisé → index dans la liste
    deduped: list[dict] = []
    for r in refuges:
        key = normalize_name(r["nom"])
        if key in seen:
            existing = deduped[seen[key]]
            if r["distance_trace_m"] < existing["distance_trace_m"]:
                deduped[seen[key]] = r
        else:
            seen[key] = len(deduped)
            deduped.append(r)
    removed = len(refuges) - len(deduped)
    if removed:
        console.print(f"  [dim]{removed} doublon(s) nom identique supprimé(s)[/dim]")
    refuges = deduped

    # 8. Tri par km croissant
    refuges.sort(key=lambda r: r["km_trace"])

    # 9. Sauvegarde JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(refuges, f, ensure_ascii=False, indent=2)
    console.print(f"\n[bold green]✓ {OUTPUT_FILE} — {len(refuges)} entrées sauvegardées[/bold green]")

    # 10. Tableau récapitulatif
    table = Table(
        title="Refuges sur la trace HRP",
        show_lines=False,
        header_style="bold cyan",
    )
    table.add_column("Nom", style="cyan", min_width=30)
    table.add_column("km",        justify="right", style="white")
    table.add_column("Altitude",  justify="right", style="yellow")
    table.add_column("Dist. trace", justify="right", style="magenta")
    table.add_column("Type",      style="dim")
    table.add_column("Source",    style="dim")

    for r in refuges:
        ele_str = f"{r['ele']:.0f} m" if r["ele"] is not None else "—"
        source_color = "green" if r["source"] == "known" else "blue"
        table.add_row(
            r["nom"],
            f"{r['km_trace']:.1f}",
            ele_str,
            f"{r['distance_trace_m']} m",
            r["type"],
            f"[{source_color}]{r['source']}[/{source_color}]",
        )

    console.print(table)


if __name__ == "__main__":
    main()
