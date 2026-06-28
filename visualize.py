#!/usr/bin/env python3
"""
visualize.py — Carte interactive HRP Gavarnie→Loudenvielle

Génère output/map.html avec :
  - Traces HRP, Loudenvielle, Cirque de Gavarnie
  - Marqueurs kilométriques toutes les 5 km
  - Marqueurs refuges depuis output/refuges.json
  - Couches fond de carte switchables (OSM, IGN France, IGN España)
  - Profil altimétrique Plotly embarqué sous la carte

Utilisation :
    python visualize.py
"""

import json
from pathlib import Path

import folium
import numpy as np
import plotly.graph_objects as go
from folium.plugins import MiniMap
from rich.console import Console
from scipy.spatial import cKDTree

from trace import build_continuous_trace, haversine_m
from slice import load_itinerary

console = Console()

OUTPUT_MAP = Path("output/map.html")
REFUGES_JSON = Path("output/refuges.json")

# Couleur de la trace continue
COLOR_TRACE = "#e63946"   # rouge

# Distance max à la trace continue pour considérer un refuge "sur l'itinéraire"
REFUGE_ON_TRACE_M = 800

# Couleurs refuges par type
REFUGE_COLORS = {
    "alpine_hut":     "#e76f51",
    "wilderness_hut": "#f4a261",
    "shelter":        "#a8dadc",
    "refuge":         "#e76f51",
    "auberge":        "#f4a261",
    "bivouac":        "#a8dadc",
    "cabane":         "#f4a261",
    "village":        "#2a9d8f",
    "unknown":        "#999999",
}


# ---------------------------------------------------------------------------
# Positionnement des refuges sur la trace continue
# ---------------------------------------------------------------------------

def snap_refuges(trace: dict, refuges: list) -> list:
    """
    Ajoute à chaque refuge :
      km_cont   : km sur la trace continue (point le plus proche)
      dist_cont : distance (m) à la trace continue
    """
    pts = trace["pts"]
    km  = trace["km"]
    cos_lat = np.cos(np.radians(np.mean([p[0] for p in pts])))
    tree = cKDTree(np.array([[p[0], p[1] * cos_lat] for p in pts]))

    for r in refuges:
        _, idx = tree.query([[r["lat"], r["lon"] * cos_lat]])
        i = int(idx[0])
        r["km_cont"] = km[i]
        r["dist_cont"] = haversine_m(r["lat"], r["lon"], pts[i][0], pts[i][1])
    return refuges


def resolve_itinerary(refuges: list, itinerary: list[str], trace: dict) -> list[dict]:
    """
    Associe chaque nom de l'itinéraire à des coords (match refuge par sous-chaîne ;
    départ = début de trace, Loudenvielle = fin de trace).
    Retourne [{role, label, nom, lat, lon}], role ∈ {depart, nuit, arrivee}.
    """
    pts = trace["pts"]
    stops: list[dict] = []
    night = 0
    for pos, name in enumerate(itinerary):
        lower = name.lower()
        if pos == 0:
            stops.append({"role": "depart", "label": "Départ", "nom": name,
                          "lat": pts[0][0], "lon": pts[0][1]})
            continue
        if lower == "loudenvielle":
            stops.append({"role": "arrivee", "label": "Fin", "nom": "Loudenvielle",
                          "lat": pts[-1][0], "lon": pts[-1][1]})
            continue
        matched = next(
            (r for r in refuges if lower in r["nom"].lower() or r["nom"].lower() in lower),
            None,
        )
        if not matched:
            console.print(f"[yellow]  Étape introuvable pour la carte : '{name}'[/yellow]")
            continue
        night += 1
        stops.append({"role": "nuit", "label": f"J{night}", "nom": matched["nom"],
                      "lat": matched["lat"], "lon": matched["lon"]})
    return stops


def add_overnight_markers(m: folium.Map, refuges: list, trace: dict) -> None:
    """Pose des pastilles numérotées (départ / J1…Jn / arrivée) aux lieux de bivouac."""
    stops = resolve_itinerary(refuges, load_itinerary(), trace)
    group = folium.FeatureGroup(name="Bivouacs (étapes)", show=True)

    role_color = {"depart": "#2a9d8f", "nuit": "#1d3557", "arrivee": "#e63946"}
    for s in stops:
        if s["lat"] is None:
            continue
        color = role_color[s["role"]]
        html = (
            f'<div style="background:{color};color:#fff;border:2px solid #fff;'
            f'border-radius:50%;width:30px;height:30px;line-height:28px;'
            f'text-align:center;font-weight:bold;font-size:12px;font-family:sans-serif;'
            f'box-shadow:0 0 4px rgba(0,0,0,.5);">{s["label"]}</div>'
        )
        folium.Marker(
            location=(s["lat"], s["lon"]),
            icon=folium.DivIcon(html=html, icon_size=(30, 30), icon_anchor=(15, 15)),
            tooltip=f"{s['label']} — bivouac près de {s['nom']}" if s["role"] == "nuit"
                    else f"{s['label']} — {s['nom']}",
            z_index_offset=1000,
        ).add_to(group)

    group.add_to(m)


# ---------------------------------------------------------------------------
# Profil altimétrique Plotly
# ---------------------------------------------------------------------------

def make_elevation_profile(trace: dict, refuges: list) -> str:
    """Génère le HTML du profil altimétrique le long de la trace continue."""
    all_pts = trace["pts"]
    all_km  = trace["km"]
    all_ele = [p[2] for p in all_pts]
    total_km = all_km[-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_km,
        y=all_ele,
        mode="lines",
        line=dict(color=COLOR_TRACE, width=2),
        fill="tozeroy",
        fillcolor="rgba(230,57,70,0.15)",
        name="Altitude",
        hovertemplate="<b>km %.1f depuis Gavarnie</b><br>%{y:.0f} m<extra></extra>",
    ))

    # Annotations : seulement les refuges réellement sur l'itinéraire
    for r in refuges:
        if r.get("ele") and r.get("dist_cont", 1e9) <= REFUGE_ON_TRACE_M:
            fig.add_vline(
                x=r["km_cont"],
                line_width=1,
                line_dash="dot",
                line_color="#666",
                annotation_text=r["nom"],
                annotation_position="top",
                annotation_font_size=9,
                annotation_textangle=-45,
            )

    fig.update_layout(
        height=280,
        margin=dict(l=60, r=20, t=30, b=60),
        xaxis=dict(title="km depuis Gavarnie", gridcolor="#eee"),
        yaxis=dict(title="Altitude (m)", gridcolor="#eee"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        hovermode="x unified",
    )

    import plotly.io as pio
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Carte Folium
# ---------------------------------------------------------------------------

def build_map(trace: dict, refuges: list) -> folium.Map:
    full_pts = trace["pts"]
    full_km  = trace["km"]

    lats = [p[0] for p in full_pts]
    lons = [p[1] for p in full_pts]
    center = (sum(lats) / len(lats), sum(lons) / len(lons))

    m = folium.Map(location=center, zoom_start=10, tiles=None)

    # ---- Fonds de carte ----
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr='&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>',
        name="OpenStreetMap",
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://wxs.ign.fr/essentiels/geoportail/wmts"
            "?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&LAYER=GEOGRAPHICALGRIDSYSTEMS.PLANIGNV2"
            "&STYLE=normal&FORMAT=image/png"
            "&TILEMATRIXSET=PM&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
        ),
        attr='&copy; <a href="https://www.ign.fr/">IGN France</a>',
        name="IGN France (Plan IGN)",
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles=(
            "https://www.ign.es/wmts/mapa-raster"
            "?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
            "&LAYER=MTN25&STYLE=default&FORMAT=image/jpeg"
            "&TILEMATRIXSET=GoogleMapsCompatible&TILEMATRIX={z}&TILEROW={y}&TILECOL={x}"
        ),
        attr='&copy; <a href="https://www.ign.es/">IGN España</a>',
        name="IGN España (MTN25)",
        show=False,
    ).add_to(m)

    # ---- Trace continue unique (Gavarnie → Loudenvielle) ----
    folium.PolyLine(
        [(p[0], p[1]) for p in full_pts],
        color=COLOR_TRACE,
        weight=3,
        opacity=0.9,
        tooltip="Trace HRP Gavarnie → Loudenvielle (100 km)",
    ).add_to(m)

    # ---- Marqueurs kilométriques toutes les 5 km ----
    km_step = 5
    next_km = km_step
    km_group = folium.FeatureGroup(name="Marqueurs km (toutes les 5 km)", show=True)

    for i, k in enumerate(full_km):
        if k >= next_km:
            lat, lon, ele = full_pts[i]
            popup_html = (
                f"<b>km {k:.1f} depuis Gavarnie</b><br>"
                f"Altitude : {ele:.0f} m<br>"
                f"Lat : {lat:.5f}<br>"
                f"Lon : {lon:.5f}"
            )
            folium.CircleMarker(
                location=(lat, lon),
                radius=4,
                color="#333",
                fill=True,
                fill_color="#fff",
                fill_opacity=1.0,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"km {next_km}",
            ).add_to(km_group)
            next_km += km_step

    km_group.add_to(m)

    # ---- Marqueurs refuges ----
    refuge_group = folium.FeatureGroup(name="Refuges / Abris", show=True)

    for r in refuges:
        rtype = r.get("type", "unknown")
        color = REFUGE_COLORS.get(rtype, REFUGE_COLORS["unknown"])
        ele_str = f"{r['ele']:.0f} m" if r.get("ele") is not None else "?"
        source_label = "OSM" if r["source"] == "osm" else "connu"
        km_cont = r.get("km_cont", 0.0)
        dist_cont = r.get("dist_cont", 0.0)
        on_trace = dist_cont <= REFUGE_ON_TRACE_M
        km_line = (
            f"km : {km_cont:.1f}<br>"
            if on_trace
            else "<i>hors itinéraire</i><br>"
        )

        popup_html = (
            f"<b>{r['nom']}</b><br>"
            f"Type : {rtype}<br>"
            f"Altitude : {ele_str}<br>"
            f"{km_line}"
            f"Distance trace : {dist_cont:.0f} m<br>"
            f"Source : {source_label}"
        )

        icon_name = "home" if rtype in ("alpine_hut", "refuge", "auberge") else "tent"
        if rtype == "village":
            icon_name = "info-sign"

        tooltip = (
            f"{r['nom']} — km {km_cont:.1f}"
            if on_trace
            else f"{r['nom']} (hors itinéraire)"
        )
        folium.Marker(
            location=(r["lat"], r["lon"]),
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=tooltip,
            icon=folium.Icon(color="white", icon_color=color, icon=icon_name, prefix="glyphicon"),
        ).add_to(refuge_group)

    refuge_group.add_to(m)

    # ---- Pastilles des bivouacs (J1…Jn) ----
    add_overnight_markers(m, refuges, trace)

    folium.LayerControl(collapsed=False).add_to(m)
    MiniMap(toggle_display=True).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Injection du profil Plotly dans le HTML Folium
# ---------------------------------------------------------------------------

def inject_profile(map_html: str, profile_html: str) -> str:
    """Insère le profil altimétrique sous la carte dans le HTML Folium."""
    profile_block = f"""
</div>
<div style="
  width: 100%;
  background: white;
  padding: 12px 20px;
  box-sizing: border-box;
  border-top: 2px solid #ddd;
  font-family: sans-serif;
">
  <h3 style="margin:0 0 8px 0; font-size:14px; color:#333;">
    Profil altimétrique HRP Gavarnie → Loudenvielle
  </h3>
  {profile_html}
</div>
<div style="display:none">
"""
    # Remplace la balise fermante </div> juste avant </body>
    # On cherche le pattern typique de Folium : `</div>\n</body>`
    inject_marker = "</div>\n</body>"
    replacement = profile_block + "\n</body>"
    if inject_marker in map_html:
        return map_html.replace(inject_marker, replacement, 1)
    # Fallback : injection avant </body>
    return map_html.replace("</body>", profile_block + "\n</body>", 1)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_MAP.parent.mkdir(parents=True, exist_ok=True)

    console.print("[bold blue]Assemblage de la trace continue…[/bold blue]")
    trace = build_continuous_trace()
    for nom, i0, i1 in trace["bounds"]:
        console.print(
            f"  {nom:28s} : km {trace['km'][i0]:6.1f} → {trace['km'][i1]:6.1f}"
        )
    console.print(f"  [bold]Total : {trace['total_km']:.1f} km, {len(trace['pts']):,} points[/bold]")

    refuges = []
    if REFUGES_JSON.exists():
        with open(REFUGES_JSON, encoding="utf-8") as f:
            refuges = json.load(f)
        snap_refuges(trace, refuges)
        on = sum(1 for r in refuges if r["dist_cont"] <= REFUGE_ON_TRACE_M)
        console.print(f"  {len(refuges)} refuges chargés ({on} sur l'itinéraire)")
    else:
        console.print(f"[yellow]  {REFUGES_JSON} absent — lancez refuges.py d'abord[/yellow]")

    console.print("\n[bold blue]Construction de la carte…[/bold blue]")
    m = build_map(trace, refuges)

    console.print("[bold blue]Génération du profil altimétrique…[/bold blue]")
    profile_html = make_elevation_profile(trace, refuges)

    console.print("[bold blue]Assemblage du HTML…[/bold blue]")
    map_html_raw = m.get_root().render()
    final_html = inject_profile(map_html_raw, profile_html)

    with open(OUTPUT_MAP, "w", encoding="utf-8") as f:
        f.write(final_html)

    console.print(f"\n[bold green]✓ {OUTPUT_MAP} généré[/bold green]")
    console.print(f"  Ouvrez dans un navigateur : file://{OUTPUT_MAP.resolve()}")


if __name__ == "__main__":
    main()
