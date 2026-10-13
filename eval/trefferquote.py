#!/usr/bin/env python3
"""
Trefferquote der Suche an einem Fragenkatalog: Treffer@k und MRR.

Für jede Frage ruft das Skript dieselbe Suchfunktion auf wie das MCP-Werkzeug
search_papers (retrieval.search.search_from_settings, Einstellungen aus
config/settings.yaml) und sucht die Soll-Fundstelle in der Trefferliste.

Fragenkatalog (JSONL, eine Frage je Zeile):
    {"id": "F01", "frage": "...", "soll_doi": "10.xxxx/...", "soll_abschnitt": "results"}
statt soll_abschnitt auch "soll_chunk_id": "<doi>___<n>"; ohne beides zählt jede
Fundstelle aus dem Paper.

Treffer: gleiche DOI (Groß-/Kleinschreibung egal) und, falls angegeben, gleicher
Abschnittstyp (section_type) bzw. gleiche Chunk-ID.
    Treffer@k  Anteil der Fragen mit Soll-Fundstelle unter den ersten k Treffern
    MRR        Mittel über 1/Rang der ersten passenden Fundstelle; 0, wenn sie
               nicht unter den ersten top_k Treffern liegt

Aufruf (aus dem Projektordner, Qdrant mit der Wissensbasis muss laufen):
    python eval/trefferquote.py --fragen eval/fragen.jsonl
    python eval/trefferquote.py --fragen eval/fragen.jsonl --alpha 0     # nur BM25
Ausgabe in --ausgabe (Standard eval/ergebnisse/): trefferquote_<zeit>.csv mit
einer Zeile je Frage und trefferquote_<zeit>.json mit der Zusammenfassung.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJEKT = Path(__file__).resolve().parent.parent
if str(PROJEKT) not in sys.path:
    sys.path.insert(0, str(PROJEKT))

PFLICHTFELDER = ("id", "frage", "soll_doi")


def lade_fragen(pfad: Path) -> list[dict]:
    """Fragenkatalog aus JSONL lesen; leere Zeilen werden übersprungen."""
    fragen = []
    for nr, zeile in enumerate(pfad.read_text(encoding="utf-8").splitlines(), start=1):
        if not zeile.strip():
            continue
        try:
            frage = json.loads(zeile)
        except json.JSONDecodeError as e:
            raise ValueError(f"{pfad.name}, Zeile {nr}: kein gültiges JSON ({e})") from e
        fehlend = [f for f in PFLICHTFELDER if not frage.get(f)]
        if fehlend:
            raise ValueError(f"{pfad.name}, Zeile {nr}: Feld(er) fehlen: {', '.join(fehlend)}")
        fragen.append(frage)
    ids = [f["id"] for f in fragen]
    doppelt = sorted({i for i in ids if ids.count(i) > 1})
    if doppelt:
        raise ValueError(f"{pfad.name}: doppelte id: {', '.join(doppelt)}")
    return fragen


def soll_fundstelle(frage: dict) -> str:
    """Soll-Fundstelle als Text für die Ergebnistabelle."""
    if frage.get("soll_chunk_id"):
        return frage["soll_chunk_id"]
    return frage.get("soll_abschnitt") or "(ganzes Paper)"


def ist_soll(treffer, frage: dict) -> bool:
    """Passt ein SearchResult zur Soll-Fundstelle der Frage?"""
    chunk = treffer.chunk
    if chunk.doi.lower() != frage["soll_doi"].lower():
        return False
    if frage.get("soll_chunk_id"):
        return chunk.chunk_id.lower() == frage["soll_chunk_id"].lower()
    if frage.get("soll_abschnitt"):
        return chunk.section_type == frage["soll_abschnitt"]
    return True


def bewerte(fragen: list[dict], suche, k: int = 5, top_k: int = 10) -> tuple[list[dict], dict]:
    """
    Stellt jede Frage an suche(frage, top_k) und bewertet die Trefferliste.

    Gibt die Zeilen je Frage und die Kennzahlen (Treffer@k, MRR) zurück.
    """
    zeilen = []
    for frage in fragen:
        treffer = suche(frage["frage"], top_k)
        rang = next((i for i, t in enumerate(treffer, start=1) if ist_soll(t, frage)), None)
        erster = treffer[0].chunk if treffer else None
        zeilen.append({
            "id": frage["id"],
            "frage": frage["frage"],
            "soll_doi": frage["soll_doi"],
            "soll_fundstelle": soll_fundstelle(frage),
            "rang": rang if rang is not None else "",
            f"treffer_at_{k}": int(rang is not None and rang <= k),
            "reziproker_rang": round(1.0 / rang, 4) if rang else 0.0,
            "rang1_doi": erster.doi if erster else "",
            "rang1_abschnitt": erster.section_type if erster else "",
        })

    n = len(zeilen)
    kennzahlen = {
        "fragen": n,
        "k": k,
        "top_k": top_k,
        f"treffer_at_{k}": round(sum(z[f"treffer_at_{k}"] for z in zeilen) / n, 4) if n else 0.0,
        "mrr": round(sum(1.0 / z["rang"] for z in zeilen if z["rang"]) / n, 4) if n else 0.0,
        "ohne_treffer": sum(1 for z in zeilen if not z["rang"]),
    }
    return zeilen, kennzahlen


def schreibe_csv(zeilen: list[dict], pfad: Path) -> None:
    """Eine Zeile je Frage; Semikolon und UTF-8 mit BOM, damit Excel die Datei direkt liest."""
    with open(pfad, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(zeilen[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(zeilen)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treffer@k und MRR der Suche an einem Fragenkatalog")
    parser.add_argument("--fragen", type=Path, required=True, help="Fragenkatalog (JSONL)")
    parser.add_argument("--ausgabe", type=Path, default=PROJEKT / "eval" / "ergebnisse",
                        help="Ordner für CSV und Zusammenfassung (Standard: eval/ergebnisse)")
    parser.add_argument("--k", type=int, default=5, help="Treffer@k (Standard: 5)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Länge der Trefferliste, zugleich Grenze für MRR (Standard: 10)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="hybrid_alpha für diesen Lauf (Standard: Wert aus config/settings.yaml)")
    args = parser.parse_args(argv)

    if args.k > args.top_k:
        parser.error("--k darf nicht größer als --top-k sein")

    from config.loader import settings
    from retrieval import search as suchmodul

    try:
        fragen = lade_fragen(args.fragen)
    except ValueError as e:
        print(f"Fehler im Fragenkatalog: {e}", file=sys.stderr)
        return 2
    if not fragen:
        print("Der Fragenkatalog ist leer.", file=sys.stderr)
        return 2

    alpha = settings.search.hybrid_alpha if args.alpha is None else args.alpha

    def suche(frage: str, top_k: int):
        # dieselbe Funktion und dieselben Einstellungen wie das Werkzeug search_papers
        return suchmodul.search_from_settings(frage, top_k=top_k, hybrid_alpha=alpha)

    zeilen, kennzahlen = bewerte(fragen, suche, k=args.k, top_k=args.top_k)
    kennzahlen.update({
        "hybrid_alpha": alpha,
        "embedding_modell": settings.embedding.model,
        "collection": settings.qdrant.collection,
        "fragenkatalog": args.fragen.name,
        "zeitpunkt": datetime.now().isoformat(timespec="seconds"),
    })

    args.ausgabe.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_pfad = args.ausgabe / f"trefferquote_{stempel}.csv"
    json_pfad = args.ausgabe / f"trefferquote_{stempel}.json"
    schreibe_csv(zeilen, csv_pfad)
    json_pfad.write_text(json.dumps(kennzahlen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{kennzahlen['fragen']} Fragen, hybrid_alpha={alpha}")
    print(f"Treffer@{args.k}: {kennzahlen[f'treffer_at_{args.k}']:.3f}")
    print(f"MRR (bis Rang {args.top_k}): {kennzahlen['mrr']:.3f}")
    print(f"ohne Treffer: {kennzahlen['ohne_treffer']}")
    print(f"Je Frage: {csv_pfad}")
    print(f"Zusammenfassung: {json_pfad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
