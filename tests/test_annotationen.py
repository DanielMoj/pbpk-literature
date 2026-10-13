"""
Annotationen sichern und zurückschreiben, gegen Qdrant im Speicher:
export_annotations.py → Neuaufbau (wie setup_qdrant.py --recreate und reingest)
→ import_annotations.py, Zuordnung über die DOI.
"""
import json

import pytest
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from ingestion.chunk_papers import chunk_directory
from ingestion.embed_and_store import embed_and_store
from mcp_server.tools.annotate_paper import annotate_paper
from mcp_server.tools.get_paper import get_paper

from hilfen import DIM, KORPUS, lade_skript

export = lade_skript("scripts/export_annotations.py")
importer = lade_skript("scripts/import_annotations.py")
setup_qdrant = lade_skript("scripts/setup_qdrant.py")

ANNOTIERT = ("10.5555/synth.002", "10.5555/synth.004")


def neu_aufbauen(client):
    """Wie setup_qdrant.py --recreate und python -m ingestion.reingest."""
    setup_qdrant.create_collection(client, "papers", recreate=True, dense_size=DIM)
    embed_and_store(chunk_directory(KORPUS, subword_factor=1.3), collection="papers")


def annotation(doi: str) -> dict:
    return json.loads(get_paper(doi))["annotation"]


def chunks_mit(client, key: str, wert: str) -> int:
    return client.count("papers", count_filter=Filter(must=[
        FieldCondition(key=key, match=MatchValue(value=wert))
    ])).count


@pytest.fixture
def annotiert(korpus):
    annotate_paper("10.5555/synth.002", gap_status="REAL_GAP", tags="TOST,equivalence",
                   relevant_for="P4", note="Beispielnotiz")
    annotate_paper("10.5555/synth.004", tags="paediatric")  # Tag ohne Lückenstatus
    return korpus


def test_export_neuaufbau_import_stellt_alles_wieder_her(annotiert, tmp_path, capsys):
    vorher = {doi: annotation(doi) for doi in ANNOTIERT}
    papers = export.fetch_annotated_papers(annotiert, "papers", include_unassessed=True)
    export.export_json(papers, tmp_path / "annotations.json")

    neu_aufbauen(annotiert)
    assert annotation("10.5555/synth.002")["gap_status"] == "UNASSESSED"  # Einspielen setzt zurück
    assert annotation("10.5555/synth.004")["tags"] == []

    assert importer.main(["--input", str(tmp_path / "annotations.json")]) == 0
    for doi in ANNOTIERT:
        assert annotation(doi) == vorher[doi]
    # auf allen Chunks des Papers, nicht nur auf dem ersten
    assert chunks_mit(annotiert, "doi", "10.5555/synth.002") == chunks_mit(annotiert, "tags", "TOST") == 5
    assert chunks_mit(annotiert, "tags", "paediatric") == chunks_mit(annotiert, "doi", "10.5555/synth.004")
    # die übrigen Paper bleiben unbewertet
    assert annotation("10.5555/synth.001")["gap_status"] == "UNASSESSED"
    assert "4 von 4 Papern zugeordnet" in capsys.readouterr().out


def test_export_ohne_all_erfasst_nur_paper_mit_luckenstatus(annotiert):
    """Darum steht --all im README: synth.004 hat nur einen Tag und fehlte sonst in der Sicherung."""
    dois = {p["doi"] for p in export.fetch_annotated_papers(annotiert, "papers")}
    assert dois == {"10.5555/synth.002"}


def test_import_aus_der_csv_des_exports(annotiert, tmp_path):
    vorher = {doi: annotation(doi) for doi in ANNOTIERT}
    papers = export.fetch_annotated_papers(annotiert, "papers", include_unassessed=True)
    export.export_csv(papers, tmp_path / "annotations.csv")

    neu_aufbauen(annotiert)
    assert importer.main(["--input", str(tmp_path / "annotations.csv")]) == 0
    for doi in ANNOTIERT:
        assert annotation(doi) == vorher[doi]


def test_trockenlauf_fehlende_doi_und_schreibweise(korpus, tmp_path):
    datei = tmp_path / "annotations.json"
    datei.write_text(json.dumps([
        {"doi": "10.5555/SYNTH.002", "gap_status": "CROWDED", "tags": ["x"], "relevant_for": [], "note": ""},
        {"doi": "10.5555/gibt-es-nicht", "gap_status": "REAL_GAP", "tags": [], "relevant_for": [], "note": ""},
    ]), encoding="utf-8")
    eintraege = importer.load_annotations(datei)

    trocken = importer.import_annotations(korpus, "papers", eintraege, dry_run=True)
    assert trocken == {"entries": 2, "papers": 1, "chunks": 5,
                       "missing": ["10.5555/gibt-es-nicht"], "dry_run": True}
    assert annotation("10.5555/synth.002")["gap_status"] == "UNASSESSED"  # nichts geschrieben

    importer.import_annotations(korpus, "papers", eintraege)
    assert annotation("10.5555/synth.002")["gap_status"] == "CROWDED"     # DOI-Schreibweise egal


@pytest.mark.parametrize("inhalt, fehler", [
    ('[{"doi": "10.5555/synth.002", "gap_status": "UNKLAR"}]', "UNKLAR"),
    ('[{"gap_status": "REAL_GAP"}]', "keine DOI"),
    ('{"doi": "10.5555/synth.002"}', "Liste"),
])
def test_ungueltige_exportdatei(tmp_path, inhalt, fehler):
    datei = tmp_path / "annotations.json"
    datei.write_text(inhalt, encoding="utf-8")
    with pytest.raises(ValueError, match=fehler):
        importer.load_annotations(datei)
    assert importer.main(["--input", str(datei)]) == 2
