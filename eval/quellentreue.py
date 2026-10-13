#!/usr/bin/env python3
"""
Quellentreue aus der manuell bewerteten Tabelle (Vorlage: eval/quellentreue_vorlage.csv).

Je Zeile eine Aussage aus der Antwort des Sprachmodells, die dort zitierte
Fundstelle und das Urteil in der Spalte gedeckt:
    ja         die Fundstelle trägt die Aussage vollständig
    teilweise  nur ein Teil der Aussage steht in der Fundstelle
    nein       die Fundstelle trägt die Aussage nicht
Leere Urteile gelten als noch nicht bewertet und zählen nicht mit.

Ausgewiesen werden, gesamt und je Frage:
    streng      Anteil "ja" an den bewerteten Aussagen
    großzügig   Anteil "ja" oder "teilweise"

Aufruf:
    python eval/quellentreue.py --bewertung eval/quellentreue.csv
"""
import argparse
import csv
import io
import json
import sys
from pathlib import Path

SPALTEN = ("frage_id", "aussage", "zitierte_fundstelle", "gedeckt", "anmerkung")
URTEILE = ("ja", "teilweise", "nein")


def lade_bewertung(pfad: Path) -> list[dict]:
    """Bewertete Aussagen lesen (Semikolon oder Komma, UTF-8 mit oder ohne BOM)."""
    text = pfad.read_text(encoding="utf-8-sig")
    kopf = text.split("\n", 1)[0]
    trenner = ";" if kopf.count(";") >= kopf.count(",") else ","
    zeilen = list(csv.DictReader(io.StringIO(text, newline=""), delimiter=trenner))
    fehlend = [s for s in SPALTEN if zeilen and s not in zeilen[0]]
    if not zeilen or fehlend:
        raise ValueError(f"{pfad.name}: keine Aussagen oder Spalten fehlen: {', '.join(fehlend)}")
    for nr, zeile in enumerate(zeilen, start=2):
        urteil = (zeile.get("gedeckt") or "").strip().lower()
        if urteil and urteil not in URTEILE:
            raise ValueError(f"{pfad.name}, Zeile {nr}: gedeckt='{zeile['gedeckt']}', erlaubt: {', '.join(URTEILE)}")
        zeile["gedeckt"] = urteil
    return zeilen


def anteile(urteile: list[str]) -> dict:
    """Zählung und Anteile für eine Menge von Urteilen (leere ausgenommen)."""
    bewertet = [u for u in urteile if u]
    n = len(bewertet)
    zaehlung = {u: bewertet.count(u) for u in URTEILE}
    return {
        "aussagen": n,
        "unbewertet": len(urteile) - n,
        **zaehlung,
        "streng": round(zaehlung["ja"] / n, 4) if n else None,
        "grosszuegig": round((zaehlung["ja"] + zaehlung["teilweise"]) / n, 4) if n else None,
    }


def werte_aus(zeilen: list[dict]) -> dict:
    """Quellentreue gesamt und je Frage."""
    je_frage: dict[str, list[str]] = {}
    for zeile in zeilen:
        je_frage.setdefault(zeile["frage_id"], []).append(zeile["gedeckt"])
    return {
        "gesamt": anteile([z["gedeckt"] for z in zeilen]),
        "je_frage": {fid: anteile(u) for fid, u in je_frage.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quellentreue aus der bewerteten Aussagentabelle")
    parser.add_argument("--bewertung", type=Path, required=True, help="ausgefüllte Vorlage (CSV)")
    parser.add_argument("--json", type=Path, default=None, help="Ergebnis zusätzlich als JSON speichern")
    args = parser.parse_args(argv)

    try:
        ergebnis = werte_aus(lade_bewertung(args.bewertung))
    except ValueError as e:
        print(f"Fehler in der Bewertung: {e}", file=sys.stderr)
        return 2

    g = ergebnis["gesamt"]
    print(f"{g['aussagen']} bewertete Aussagen ({g['unbewertet']} unbewertet) aus "
          f"{len(ergebnis['je_frage'])} Fragen: ja {g['ja']}, teilweise {g['teilweise']}, nein {g['nein']}")
    if g["aussagen"]:
        print(f"Quellentreue streng (ja): {g['streng']:.3f}")
        print(f"Quellentreue großzügig (ja oder teilweise): {g['grosszuegig']:.3f}")
    if args.json:
        args.json.write_text(json.dumps(ergebnis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Ergebnis: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
