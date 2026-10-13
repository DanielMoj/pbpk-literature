"""
MCP-Werkzeuge als Funktionen gegen das synthetische Korpus in Qdrant im Speicher:
Ausgabeformat, Reihenfolge, Autorensuche, Annotationen und Lückenübersicht.
"""
import json

import pytest

from ingestion.chunk_papers import chunk_document
from ingestion.embed_and_store import embed_and_store
from mcp_server.tools.annotate_paper import annotate_paper
from mcp_server.tools.find_similar import find_similar
from mcp_server.tools.get_gap_landscape import get_gap_landscape
from mcp_server.tools.get_paper import get_paper
from mcp_server.tools.search_by_author import search_by_author
from mcp_server.tools.search_papers import search_papers


def test_search_papers_liefert_fundstellen_mit_quelle(korpus):
    antwort = json.loads(search_papers("kidney elimination", top_k=3))
    assert antwort["total"] == 3
    erster = antwort["results"][0]
    assert set(erster) == {
        "doi", "title", "authors", "year", "journal", "section_type",
        "snippet", "score", "gap_status", "tags", "relevant_for",
    }
    assert erster["doi"].startswith("10.5555/synth.")
    assert erster["snippet"]


def test_search_papers_mit_filtern(korpus):
    antwort = json.loads(search_papers("dose", year_min=2025, top_k=10))
    assert antwort["total"] > 0
    assert {r["doi"] for r in antwort["results"]} == {"10.5555/synth.004"}


def test_get_paper_in_dokumentreihenfolge(korpus):
    """Mehr als zehn Chunks: ___10 gehört hinter ___9, nicht hinter ___1."""
    abschnitte = [
        {"heading": f"Part {i}", "text": f"Abschnitt {i}. " + " ".join(["text"] * 60)}
        for i in range(12)
    ]
    doc = {"metadata": {"doi": "10.5555/lang", "title": "Langes Paper", "year": 2020}, "sections": abschnitte}
    embed_and_store(chunk_document(doc, subword_factor=1.3), collection="papers")

    paper = json.loads(get_paper("10.5555/lang"))
    assert paper["total_sections"] == 12
    assert [s["text"].split(".")[0] for s in paper["sections"]] == [f"Abschnitt {i}" for i in range(12)]


def test_get_paper_unbekannte_doi(korpus):
    assert "not found" in json.loads(get_paper("10.5555/gibt-es-nicht"))["error"]


def test_get_paper_liefert_metadaten_und_abschnitte(korpus):
    paper = json.loads(get_paper("10.5555/synth.002"))
    assert paper["title"].startswith("Two one-sided tests")
    assert paper["year"] == 2021
    assert [s["section_type"] for s in paper["sections"]] == [
        "abstract", "introduction", "methods", "results", "discussion",
    ]
    assert paper["annotation"]["gap_status"] == "UNASSESSED"


@pytest.mark.parametrize("name, erwartet", [
    ("Mustermann", ["10.5555/synth.004", "10.5555/synth.001", "10.5555/synth.002"]),  # Jahr absteigend
    ("Erika Mustermann", ["10.5555/synth.001", "10.5555/synth.002"]),
    ("Doe", ["10.5555/synth.004", "10.5555/synth.001", "10.5555/synth.003"]),
    ("Niemand", []),
])
def test_search_by_author_findet_nachnamen_in_grobid_angaben(korpus, name, erwartet):
    """Die Autorenangaben enthalten die Affiliation ("Erika Mustermann, Beispiel-Universität, ...")."""
    antwort = json.loads(search_by_author(name))
    assert [r["doi"] for r in antwort["results"]] == erwartet
    assert all(r["snippet"] for r in antwort["results"])


def test_annotate_paper_ergaenzt_listen_und_ueberschreibt_status(korpus):
    erst = json.loads(annotate_paper("10.5555/synth.002", gap_status="REAL_GAP",
                                     tags="TOST, equivalence", relevant_for="P4", note="erste Notiz"))
    assert erst["annotation"] == {
        "gap_status": "REAL_GAP", "tags": ["TOST", "equivalence"],
        "relevant_for": ["P4"], "note": "erste Notiz",
    }
    dann = json.loads(annotate_paper("10.5555/synth.002", gap_status="CROWDED", tags="TOST,variability"))
    assert dann["annotation"]["gap_status"] == "CROWDED"
    assert dann["annotation"]["tags"] == ["TOST", "equivalence", "variability"]
    assert dann["annotation"]["note"] == "erste Notiz"

    # gilt für alle Chunks des Papers
    paper = json.loads(get_paper("10.5555/synth.002"))
    assert paper["annotation"]["tags"] == ["TOST", "equivalence", "variability"]
    treffer = json.loads(search_papers("bioequivalence", tags="variability", top_k=20))
    assert treffer["total"] == 5
    assert {r["doi"] for r in treffer["results"]} == {"10.5555/synth.002"}


@pytest.mark.parametrize("argumente, fehler", [
    ({"doi": "10.5555/synth.002", "gap_status": "UNKLAR"}, "Invalid gap_status"),
    ({"doi": "10.5555/synth.002"}, "No annotation fields"),
    ({"doi": "10.5555/gibt-es-nicht", "note": "x"}, "not found"),
])
def test_annotate_paper_fehler(korpus, argumente, fehler):
    assert fehler in json.loads(annotate_paper(**argumente))["error"]


def test_gap_landscape_laesst_unbewertete_weg_auch_mit_thema(korpus):
    annotate_paper("10.5555/synth.002", gap_status="REAL_GAP", relevant_for="P4")

    ohne_thema = json.loads(get_gap_landscape())
    assert ohne_thema["summary"] == {"REAL_GAP": 1, "total_annotated": 1}
    assert ohne_thema["groups"]["REAL_GAP"][0]["doi"] == "10.5555/synth.002"

    mit_thema = json.loads(get_gap_landscape(topic="bioequivalence kidney dose"))
    assert set(mit_thema["groups"]) == {"REAL_GAP"}

    unbewertet = json.loads(get_gap_landscape(gap_status="UNASSESSED"))
    assert {e["doi"] for e in unbewertet["groups"]["UNASSESSED"]} == {
        "10.5555/synth.001", "10.5555/synth.003", "10.5555/synth.004",
    }
    assert json.loads(get_gap_landscape(relevant_for="P9"))["summary"] == {"total_annotated": 0}


def test_find_similar_schliesst_das_ankerpaper_aus(korpus):
    antwort = json.loads(find_similar("10.5555/synth.001", top_k=3))
    dois = [r["doi"] for r in antwort["results"]]
    assert antwort["anchor_section"] == "abstract"
    assert "10.5555/synth.001" not in dois
    assert len(dois) == len(set(dois)) == 3
    assert json.loads(find_similar("10.5555/gibt-es-nicht"))["results"] == []
