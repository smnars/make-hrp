#!/usr/bin/env python3
"""
report.py — Rapport résumé partageable de l'itinéraire (Markdown).

Génère output/rapport_hrp.md à partir de la trace continue (trace.py) et de
l'itinéraire par défaut (slice.py). À montrer pour validation.

Utilisation :
    python report.py
"""

from datetime import date
from pathlib import Path

from rich.console import Console

from trace import build_continuous_trace
from slice import (
    MAX_LOADED_OK,
    load_itinerary, resolve_refuges, load_guarded_names, compute_stages,
)

console = Console()
OUTPUT_MD = Path("output/rapport_hrp.md")


def main() -> None:
    tr = build_continuous_trace()
    pts, km = tr["pts"], tr["km"]
    breaks = resolve_refuges(load_itinerary(), pts, km)
    # dédup + ordre croissant (comme slice.py)
    clean = [breaks[0]]
    for idx, nom in breaks[1:]:
        if idx > clean[-1][0]:
            clean.append((idx, nom))
    breaks = clean

    guarded = load_guarded_names()
    stages = compute_stages(pts, km, breaks, guarded)

    tot_dist = sum(s["dist"] for s in stages)
    tot_dp = sum(s["dplus"] for s in stages)
    tot_dm = sum(s["dminus"] for s in stages)
    tot_charge = sum(s["charge"] for s in stages)
    n_guarded = sum(1 for s in stages if s["guarded"])
    depart = stages[0]["start"]
    arrivee = stages[-1]["end"]

    L: list[str] = []
    L.append("# HRP — Pont d'Espagne (Cauterets) → Loudenvielle")
    L.append("")
    L.append(f"*Itinéraire généré le {date.today().isoformat()} — trek en bivouac*")
    L.append("")
    L.append("## Vue d'ensemble")
    L.append("")
    L.append(f"- **Départ :** {depart} (parking au-dessus de Cauterets)")
    L.append(f"- **Arrivée :** {arrivee}")
    L.append(f"- **Distance :** {tot_dist:.0f} km")
    L.append(f"- **Dénivelé :** +{tot_dp:.0f} m / −{tot_dm:.0f} m")
    L.append(f"- **Durée :** {len(stages)} jours")
    L.append(f"- **Bivouacs à côté d'un refuge gardé :** {n_guarded} / {len(stages) - 1}")
    L.append(f"- **Charge totale :** {tot_charge:.0f} km-effort *(km + D+/100 + D−/300)*")
    L.append("")
    L.append("> On dort en **bivouac** tout du long — on s'installe *à côté* des refuges/cabanes "
             "(pas de nuit en dur, pas de réservation). Les refuges gardés servent de point d'eau / ravito.")
    L.append("")
    L.append("## Étapes")
    L.append("")
    L.append("| Jour | Étape | Dist | D+ | D− | Charge | Bivouac |")
    L.append("|---|---|---:|---:|---:|---:|---|")
    for s in stages:
        couchage = "⛺ près refuge gardé" if s["guarded"] else ("arrivée" if s["is_end"] else "cabane / bivouac")
        flag = " ⚠" if s["charge"] > MAX_LOADED_OK else ""
        L.append(
            f"| J{s['day']} | {s['start']} → {s['end']} | {s['dist']:.1f} km | "
            f"+{s['dplus']:.0f} | −{s['dminus']:.0f} | {s['charge']:.1f}{flag} | {couchage} |"
        )
    L.append(f"| | **TOTAL** | **{tot_dist:.0f} km** | **+{tot_dp:.0f}** | **−{tot_dm:.0f}** | **{tot_charge:.0f}** | |")
    L.append("")
    L.append("## Bivouacs")
    L.append("")
    for s in stages[:-1]:
        kind = "à côté du refuge gardé" if s["guarded"] else "cabane non gardée / bivouac"
        L.append(f"- **Nuit {s['day']} — {s['end']}** ({kind})")
    L.append("")
    L.append("## Points à valider")
    L.append("")
    L.append("- **Bivouac** : autorisé dans le Parc national des Pyrénées du coucher au lever du soleil, "
             "à plus d'1 h de marche des limites/routes. Côté espagnol (Posets-Maladeta), règles similaires en altitude.")
    L.append("- Section espagnole **Munia → Viadós** : aucun refuge gardé, prévoir autonomie eau + nourriture (cabañas + bivouac).")
    L.append("- Les 2 grosses journées (J3 ~38, J5 ~37 km-effort) peuvent être coupées par un bivouac intermédiaire si besoin.")
    L.append("- J1/J2 sont courtes (approche Oulettes + Bayssellance) : fusionnables en une journée Pont d'Espagne → Bayssellance (~26 km-effort).")
    L.append("")
    L.append("## Fichiers")
    L.append("")
    L.append("- **Carte interactive** : `output/map.html` (trace + profil altimétrique + pastilles des bivouacs)")
    L.append("- **Profil altimétrique** : `output/profil_hrp.png`")
    L.append("- **Traces GPX par étape** : `output/J1_*.gpx` … (à importer dans Visorando, OsmAnd, Organic Maps…)")
    L.append("- **Trace complète** : `output/trace_complete.gpx`")
    L.append("")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(L), encoding="utf-8")
    console.print(f"[bold green]✓ {OUTPUT_MD} généré[/bold green] ({len(stages)} étapes, {tot_dist:.0f} km, +{tot_dp:.0f} m)")


if __name__ == "__main__":
    main()
