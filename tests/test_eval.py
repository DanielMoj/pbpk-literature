"""Auswertung (eval/): Trefferquote auf dem synthetischen Korpus, Kennzahlen, Quellentreue."""
import csv
import json
from types import SimpleNamespace

import pytest

from hilfen import KORPUS, PROJEKT, lade_skript

trefferquote = lade_skript("eval/trefferquote.py")
quellentreue = lade_skript("eval/quellentreue.py")

BEISPIELFRAGEN = PROJEKT / "eval" / "fragen.beispiel.jsonl"


def lies_csv(pfad):
    with open(pfad, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def lauf(tmp_path, *argumente):
    """trefferquote.main mit Ausgabe nach tmp_path; gibt (Zeilen, Zusammenfassung) zurück."""
    code = trefferquote.main(["--fragen", str(BEISPIELFRAGEN), "--ausgabe", str(tmp_path), *argumente])
    assert code == 0
    (csv_datei,) = tmp_path.glob("trefferquote_*.csv")
    (json_datei,) = tmp_path.glob("trefferquote_*.json")
    return lies_csv(csv_datei), json.loads(json_datei.read_text(encoding="utf-8"))


def test_beispielfragen_passen_zum_synthetischen_korpus():
    fragen = trefferquote.lade_fragen(BEISPIELFRAGEN)
    assert [f["id"] for f in fragen] == ["B01", "B02", "B03"]
    dois = {json.loads(p.read_text(encoding="utf-8"))["metadata"]["original_metadata"]["doi"]
            for p in KORPUS.glob("*_extracted.json")}
    for f in fragen:
        assert f["soll_doi"] in dois
        assert f.get("soll_abschnitt") or f.get("soll_chunk_id")


def test_trefferquote_laeuft_auf_dem_synthetischen_korpus(korpus, tmp_path, capsys):
    zeilen, zusammenfassung = lauf(tmp_path)

    assert [z["id"] for z in zeilen] == ["B01", "B02", "B03"]
    assert list(zeilen[0]) == [
        "id", "frage", "soll_doi", "soll_fundstelle", "rang", "treffer_at_5",
        "reziproker_rang", "rang1_doi", "rang1_abschnitt",
    ]
    assert zusammenfassung["fragen"] == 3
    assert zusammenfassung["hybrid_alpha"] == 0.7       # aus config/settings.yaml
    assert zusammenfassung["embedding_modell"] == "BAAI/bge-large-en-v1.5"
    # Zusammenfassung und Zeilen passen zueinander
    treffer = [int(z["treffer_at_5"]) for z in zeilen]
    raenge = [int(z["rang"]) for z in zeilen if z["rang"]]
    assert zusammenfassung["treffer_at_5"] == pytest.approx(sum(treffer) / 3, abs=1e-4)
    assert zusammenfassung["mrr"] == pytest.approx(sum(1 / r for r in raenge) / 3, abs=1e-4)
    assert zusammenfassung["ohne_treffer"] == 3 - len(raenge)
    assert "Treffer@5" in capsys.readouterr().out


def test_ergebnis_haengt_von_hybrid_alpha_ab(korpus, tmp_path):
    """B03 fragt nach "kidney" und "Beispielin": nur BM25 findet die Stelle sicher (Rang 1)."""
    nur_bm25, zusammenfassung = lauf(tmp_path / "bm25", "--alpha", "0")
    assert zusammenfassung["hybrid_alpha"] == 0.0
    b03 = next(z for z in nur_bm25 if z["id"] == "B03")
    assert (b03["rang"], b03["treffer_at_5"], b03["rang1_doi"]) == ("1", "1", "10.5555/synth.003")

    nur_dicht, _ = lauf(tmp_path / "dicht", "--alpha", "1")
    b03_dicht = next(z for z in nur_dicht if z["id"] == "B03")
    assert b03_dicht["treffer_at_5"] == "0"   # die dichte Attrappe kennt "Beispielin" nicht


def test_kennzahlen_mit_festen_raengen():
    """Treffer@5 und MRR mit einer Attrappe der Suche, deren Ränge feststehen."""
    def treffer(doi, abschnitt, nr):
        return SimpleNamespace(chunk=SimpleNamespace(doi=doi, section_type=abschnitt, chunk_id=f"{doi}___{nr}"))

    andere = [treffer("10.1/andere", "methods", i) for i in range(10)]
    listen = {
        "rang 1": [treffer("10.1/a", "results", 0)] + andere[:9],
        "rang 3": andere[:2] + [treffer("10.1/B", "methods", 4)] + andere[2:9],
        "rang 7": andere[:6] + [treffer("10.1/c", "results", 2)] + andere[6:9],
        "fehlt": andere,
    }
    fragen = [
        {"id": "1", "frage": "rang 1", "soll_doi": "10.1/A", "soll_abschnitt": "results"},
        {"id": "2", "frage": "rang 3", "soll_doi": "10.1/b", "soll_chunk_id": "10.1/B___4"},
        {"id": "3", "frage": "rang 7", "soll_doi": "10.1/c"},
        {"id": "4", "frage": "fehlt", "soll_doi": "10.1/d", "soll_abschnitt": "results"},
    ]
    aufrufe = []

    def suche(frage, top_k):
        aufrufe.append(top_k)
        return listen[frage][:top_k]

    zeilen, kennzahlen = trefferquote.bewerte(fragen, suche, k=5, top_k=10)
    assert [z["rang"] for z in zeilen] == [1, 3, 7, ""]
    assert [z["treffer_at_5"] for z in zeilen] == [1, 1, 0, 0]
    assert kennzahlen["treffer_at_5"] == 0.5
    assert kennzahlen["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 7 + 0) / 4, abs=1e-4)
    assert kennzahlen["ohne_treffer"] == 1
    assert aufrufe == [10, 10, 10, 10]

    # falscher Abschnitt zählt nicht
    falsch = [{"id": "5", "frage": "rang 1", "soll_doi": "10.1/a", "soll_abschnitt": "methods"}]
    assert trefferquote.bewerte(falsch, suche)[0][0]["rang"] == ""


def test_dieselbe_suchfunktion_wie_search_papers(korpus, monkeypatch, tmp_path):
    from mcp_server.tools.search_papers import search_papers
    from retrieval import search as suchmodul

    original = suchmodul.search_from_settings
    aufrufer = []

    def beobachtet(query, **kwargs):
        aufrufer.append(query)
        return original(query, **kwargs)

    monkeypatch.setattr(suchmodul, "search_from_settings", beobachtet)
    search_papers("kidney elimination", top_k=3)
    lauf(tmp_path)
    assert aufrufer[0] == "kidney elimination"
    assert len(aufrufer) == 1 + 3  # Werkzeug + drei Beispielfragen


def test_fehlerhafter_fragenkatalog(tmp_path, capsys):
    datei = tmp_path / "fragen.jsonl"
    datei.write_text('{"id": "X1", "frage": "ohne Soll-DOI"}\n', encoding="utf-8")
    assert trefferquote.main(["--fragen", str(datei), "--ausgabe", str(tmp_path)]) == 2
    assert "soll_doi" in capsys.readouterr().err
    assert not list(tmp_path.glob("trefferquote_*"))


# ---------------------------------------------------------------------------
# Quellentreue
# ---------------------------------------------------------------------------

def test_vorlage_hat_die_spalten_der_bewertung():
    kopf = (PROJEKT / "eval" / "quellentreue_vorlage.csv").read_text(encoding="utf-8").strip()
    assert kopf.split(";") == list(quellentreue.SPALTEN)


def test_quellentreue_streng_und_grosszuegig(tmp_path, capsys):
    datei = tmp_path / "bewertung.csv"
    datei.write_text(
        "﻿frage_id;aussage;zitierte_fundstelle;gedeckt;anmerkung\n"
        "B01;Die AUC steigt auf das 2,4-Fache.;10.5555/synth.001, results;ja;\n"
        'B01;"Die Halbwertszeit wird kürzer;\nmehrzeilig";10.5555/synth.001, results;nein;falsche Richtung\n'
        "B02;Entscheidend ist das 90%-Konfidenzintervall.;10.5555/synth.002, methods;Ja;\n"
        "B02;TOST nutzt zweimal fünf Prozent.;10.5555/synth.002, methods;teilweise;\n"
        "B03;Noch offen.;10.5555/synth.003, results;;\n",
        encoding="utf-8",
    )
    ergebnis = quellentreue.werte_aus(quellentreue.lade_bewertung(datei))
    assert ergebnis["gesamt"] == {
        "aussagen": 4, "unbewertet": 1, "ja": 2, "teilweise": 1, "nein": 1,
        "streng": 0.5, "grosszuegig": 0.75,
    }
    assert ergebnis["je_frage"]["B01"]["streng"] == 0.5
    assert ergebnis["je_frage"]["B02"]["grosszuegig"] == 1.0
    assert ergebnis["je_frage"]["B03"]["streng"] is None

    assert quellentreue.main(["--bewertung", str(datei), "--json", str(tmp_path / "q.json")]) == 0
    assert "streng (ja): 0.500" in capsys.readouterr().out
    assert json.loads((tmp_path / "q.json").read_text(encoding="utf-8"))["gesamt"]["ja"] == 2


def test_quellentreue_ungueltiges_urteil(tmp_path):
    datei = tmp_path / "bewertung.csv"
    datei.write_text("frage_id,aussage,zitierte_fundstelle,gedeckt,anmerkung\nB01,x,y,vielleicht,\n",
                     encoding="utf-8")
    with pytest.raises(ValueError, match="vielleicht"):
        quellentreue.lade_bewertung(datei)
    assert quellentreue.main(["--bewertung", str(datei)]) == 2
