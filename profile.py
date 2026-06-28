#!/usr/bin/env python3
"""
profile.py — Profil altimétrique scientifique (PNG) de la trace continue.

Génère output/profil_hrp.png à partir de trace.py et de l'itinéraire de slice.py.
Style sobre/scientifique : grille fine, ticks mineurs, annotations des bivouacs
colorées selon le type (refuge gardé / cabane-bivouac).

Utilisation :
    python profile.py
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from trace import build_continuous_trace
from slice import (
    smooth_elevation, load_itinerary,
    resolve_refuges, load_guarded_names, compute_stages,
)

OUTPUT_PNG = Path("output/profil_hrp.png")

# Palette sobre
COL_LINE   = "#1b3a4b"   # ligne de crête
COL_FILL   = "#9fb8c6"   # remplissage sous la courbe
COL_GUARD  = "#2a7f3f"   # bivouac près refuge gardé (vert)
COL_CABANE = "#b06a2c"   # bivouac cabane / sauvage (orange)
COL_GRID   = "#d8d8d8"


def main() -> None:
    tr = build_continuous_trace()
    pts, km = tr["pts"], tr["km"]
    km = np.array(km)
    ele = smooth_elevation([p[2] for p in pts])

    # Étapes / bivouacs
    breaks = resolve_refuges(load_itinerary(), pts, list(km))
    clean = [breaks[0]]
    for idx, nom in breaks[1:]:
        if idx > clean[-1][0]:
            clean.append((idx, nom))
    breaks = clean
    guarded = load_guarded_names()
    stages = compute_stages(pts, list(km), breaks, guarded)

    # rcParams "scientifique"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
    })

    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)

    ymin = float(ele.min()) - 80
    ymax = float(ele.max())

    # ---- Ombrage alterné des étapes ----
    for j, s in enumerate(stages):
        x0, x1 = km[s["i0"]], km[s["i1"]]
        if j % 2 == 0:
            ax.axvspan(x0, x1, color="#000000", alpha=0.025, lw=0)

    # ---- Courbe + remplissage ----
    ax.fill_between(km, ele, ymin, color=COL_FILL, alpha=0.45, lw=0, zorder=2)
    ax.plot(km, ele, color=COL_LINE, lw=1.1, zorder=3)

    # ---- Marqueurs de bivouac (fin de chaque étape sauf l'arrivée) ----
    for s in stages:
        if s["is_end"]:
            continue
        x = km[s["i1"]]
        y = ele[s["i1"]]
        color = COL_GUARD if s["guarded"] else COL_CABANE
        ax.axvline(x, color=color, lw=0.8, ls="--", alpha=0.7, zorder=4)
        ax.plot(x, y, "o", ms=5, mfc=color, mec="white", mew=0.8, zorder=6)
        ax.annotate(
            f"J{s['day']}  {s['end']}",
            xy=(x, ymax), xytext=(x, ymax + 60),
            rotation=90, ha="center", va="bottom",
            fontsize=7, color=color, fontweight="bold",
            annotation_clip=False, zorder=6,
        )

    # ---- Départ / Arrivée ----
    for x, lab in [(km[0], "Départ\nPont d'Espagne"), (km[-1], "Arrivée\nLoudenvielle")]:
        ax.plot(x, ele[0 if x == km[0] else -1], "s", ms=6,
                mfc="#333333", mec="white", mew=0.8, zorder=6)
        ax.annotate(lab, xy=(x, ymax), xytext=(x, ymax + 60),
                    rotation=90, ha="center", va="bottom",
                    fontsize=7, color="#333333", fontweight="bold",
                    annotation_clip=False, zorder=6)

    # ---- Axes / grille ----
    ax.set_xlim(km[0], km[-1])
    ax.set_ylim(ymin, ymax + 320)
    ax.set_xlabel("Distance (km)", fontsize=10)
    ax.set_ylabel("Altitude (m)", fontsize=10)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(250))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", color=COL_GRID, lw=0.5, zorder=0)
    ax.grid(which="minor", color=COL_GRID, lw=0.3, alpha=0.6, zorder=0)
    ax.tick_params(which="both", direction="out", length=4, width=0.8)
    ax.tick_params(which="minor", length=2)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---- Titre + stats ----
    tot_dist = sum(s["dist"] for s in stages)
    tot_dp = sum(s["dplus"] for s in stages)
    tot_dm = sum(s["dminus"] for s in stages)
    n_guard = sum(1 for s in stages if s["guarded"])
    ax.set_title("Profil altimétrique — HRP Pont d'Espagne (Cauterets) → Loudenvielle",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    stats = (f"{tot_dist:.0f} km  ·  +{tot_dp:.0f} m / −{tot_dm:.0f} m  ·  "
             f"{len(stages)} jours  ·  {n_guard} bivouacs près d'un refuge gardé")
    ax.text(0.0, 1.015, stats, transform=ax.transAxes, fontsize=8.5,
            color="#555555", ha="left", va="bottom")

    # Légende manuelle
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", mfc=COL_GUARD, mec="white",
               ms=7, label="bivouac près refuge gardé"),
        Line2D([0], [0], marker="o", color="w", mfc=COL_CABANE, mec="white",
               ms=7, label="bivouac cabane / sauvage"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.5, frameon=True,
              framealpha=0.9, edgecolor="#cccccc")

    fig.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"✓ {OUTPUT_PNG} ({OUTPUT_PNG.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
