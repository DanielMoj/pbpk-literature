"""
Zerlegung entlang der Abschnitte (ingestion/chunk_papers.py) am synthetischen
Paper 10.5555/synth.001 im konsolidierten JSON-Format.

Aufbau des Papers (Wörter): Abstract 70 (aus den GROBID-Metadaten),
1 Introduction 79, 2 Methods 18, 2.1 Model structure 93 (mit Aufzählung),
3 Results 332 in drei Absätzen (110/111/111), 4 Discussion 78, 5 Conclusion 14,
References, Tabelle 1 mit Inhalt (51 Wörter), Tabelle 2 ohne Inhalt.
Grenzen gelten für Wörter × subword_factor (Standard 1.3, min 50, max 350).
"""
import copy
import json

import pytest

from ingestion import chunk_papers
from ingestion.chunk_papers import _estimate_tokens, chunk_directory, chunk_document

from hilfen import KORPUS

DOI = "10.5555/synth.001"


@pytest.fixture
def paper() -> dict:
    return json.loads((KORPUS / "10.5555_synth.001_extracted.json").read_text(encoding="utf-8"))


def absaetze(paper: dict, titel: str) -> list[str]:
    sektion = next(s for s in paper["content"]["sections"] if s["title"] == titel)
    return [e["text"] for e in sektion["elements"] if e["type"] == "paragraph"]


def test_metadaten_aus_sidecar_und_grobid(paper):
    meta = chunk_document(paper)[0].metadata
    assert meta.doi == DOI                                   # Sidecar (CrossRef)
    assert meta.journal == "Journal of Synthetic Test Data"  # Sidecar
    assert meta.year == 2024
    assert meta.title.startswith("A minimal PBPK model")     # GROBID
    assert meta.authors[0].startswith("Erika Mustermann")    # GROBID, mit Affiliation
    assert meta.abstract.startswith("Fiktivol is a fictional compound")


def test_chunks_folgen_den_abschnitten(paper):
    chunks = chunk_document(paper)
    assert [c.section_type for c in chunks] == [
        "abstract", "introduction", "methods", "results", "results", "discussion", "table",
    ]
    assert [c.chunk_id for c in chunks] == [f"{DOI}___{i}" for i in range(7)]
    assert all(c.doi == DOI for c in chunks)


def test_kurzer_abschnitt_wird_mit_dem_naechsten_zusammengelegt(paper):
    """2 Methods (18 Wörter ≈ 23 Tokens < 50) geht im folgenden 2.1 Model structure auf."""
    methods = chunk_document(paper)[2]
    assert methods.section_type == "methods"
    assert methods.text.startswith("All data in this example are simulated.")
    assert "The model consists of a central plasma compartment" in methods.text


def test_aufzaehlungen_werden_uebernommen(paper):
    methods = chunk_document(paper)[2]
    assert "- Healthy renal function: glomerular filtration rate 100 mL/min" in methods.text
    assert "- Severe renal impairment: 20 mL/min" in methods.text


def test_kurzer_letzter_abschnitt_wird_angehaengt(paper):
    """5 Conclusion (14 Wörter) hat keinen Nachfolger und wird an die Diskussion gehängt."""
    diskussion = chunk_document(paper)[5]
    assert diskussion.section_type == "discussion"
    assert diskussion.text.startswith("The example shows how a PBPK model")
    assert diskussion.text.endswith("in severe renal impairment.")


def test_langer_abschnitt_wird_an_absaetzen_geteilt(paper):
    """3 Results: 332 Wörter ≈ 431 Tokens > 350; Absatz 1+2 (287) passen, Absatz 3 beginnt neu."""
    p1, p2, p3 = absaetze(paper, "3 Results")
    teil1, teil2 = chunk_document(paper)[3:5]
    assert teil1.text == p1 + "\n\n" + p2
    assert teil2.text == p3


def test_hoechstlaenge_eingehalten(paper):
    for chunk in chunk_document(paper):
        if chunk.section_type != "table":  # Tabellen werden nicht geteilt
            assert _estimate_tokens(chunk.text, 1.3) <= 350


def test_literatur_und_tabellen_ohne_inhalt_werden_ausgelassen(paper):
    chunks = chunk_document(paper)
    texte = " ".join(c.text for c in chunks)
    assert "A fictional reference used only for testing" not in texte  # References
    assert "Physiological parameters of the minimal model" not in texte  # Tabelle 2, nur GROBID
    assert all(c.section_type != "references" for c in chunks)


def test_tabelle_mit_inhalt_wird_zeilenweise_gerendert(paper):
    tabelle = chunk_document(paper)[-1]
    zeilen = tabelle.text.split("\n")
    assert zeilen[0] == "Table 1. Simulated exposure of Fiktivol by renal function group"
    assert "Group | GFR (mL/min) | AUC ratio | Half-life (h)" in zeilen
    assert "Severe impairment | 20 | 2.4 | 21" in zeilen


def test_subword_faktor_steuert_die_teilung(paper):
    """Mit Faktor 1.0 zählen die 332 Wörter als 332 Tokens ≤ 350: Results bleibt ganz."""
    chunks = chunk_document(paper, subword_factor=1.0)
    typen = [c.section_type for c in chunks]
    assert typen == ["abstract", "introduction", "methods", "results", "discussion", "table"]
    assert chunks[3].text == "\n\n".join(absaetze(paper, "3 Results"))


def test_subword_faktor_steuert_das_zusammenlegen():
    """40 Wörter: mit 1.3 sind es 52 Tokens (eigener Chunk), mit 1.0 nur 40 (wird angehängt)."""
    doc = {
        "metadata": {"doi": "10.5555/flach", "title": "Flaches Schema"},
        "sections": [
            {"heading": "Introduction", "text": " ".join(["word"] * 60)},
            {"heading": "Results", "text": " ".join(["result"] * 40)},
        ],
    }
    assert [c.section_type for c in chunk_document(doc, subword_factor=1.3)] == ["introduction", "results"]
    assert [c.section_type for c in chunk_document(doc, subword_factor=1.0)] == ["introduction"]


def test_ohne_faktor_gilt_der_standard_und_nicht_der_letzte_aufruf(paper):
    chunk_document(paper, subword_factor=1.0)
    assert len(chunk_document(paper)) == 7
    assert chunk_papers._SUBWORD_FACTOR == 1.3


def test_schaetzung_der_tokens():
    assert _estimate_tokens("eins zwei drei vier", 1.0) == 4
    assert _estimate_tokens("eins zwei drei vier", 1.3) == 5   # int(5.2)
    assert _estimate_tokens("", 1.3) == 0


def test_mindestlaenge_null_legt_nichts_zusammen(paper):
    typen = [c.section_type for c in chunk_document(paper, min_chunk_length=0)]
    assert typen == [
        "abstract", "introduction", "methods", "methods", "results", "results",
        "discussion", "conclusion", "table",
    ]


def test_abstract_wird_nie_zusammengelegt(paper):
    kurz = copy.deepcopy(paper)
    kurz["metadata"]["document_metadata"]["abstract"] = "A very short abstract."
    chunks = chunk_document(kurz)
    assert chunks[0].section_type == "abstract"
    assert chunks[0].text == "A very short abstract."
    assert chunks[1].section_type == "introduction"


def test_absatz_ohne_absatzgrenzen_wird_an_saetzen_geteilt():
    saetze = " ".join(f"Sentence number {i} describes one simulated finding." for i in range(80))
    doc = {"metadata": {"doi": "10.5555/saetze"}, "sections": [{"heading": "Results", "text": saetze}]}
    chunks = chunk_document(doc, subword_factor=1.3)
    assert len(chunks) > 1
    assert all(_estimate_tokens(c.text, 1.3) <= 350 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)          # Schnitt an Satzgrenzen
    assert " ".join(c.text for c in chunks) == saetze


def test_text_ohne_satzzeichen_wird_an_wortgrenzen_geteilt():
    text = " ".join(f"w{i}" for i in range(1000))
    doc = {"metadata": {"doi": "10.5555/woerter"}, "sections": [{"heading": "Methods", "text": text}]}
    chunks = chunk_document(doc, subword_factor=1.3)
    assert len(chunks) == 4                                   # je 269 Wörter (350 / 1.3)
    assert all(_estimate_tokens(c.text, 1.3) <= 350 for c in chunks)
    assert " ".join(c.text for c in chunks) == text


def test_ganzes_korpus():
    chunks = chunk_directory(KORPUS, subword_factor=1.3)
    assert {c.doi for c in chunks} == {f"10.5555/synth.00{i}" for i in range(1, 5)}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
