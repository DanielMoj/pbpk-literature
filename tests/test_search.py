"""
Hybride Suche (retrieval/search.py): Wirkung von hybrid_alpha, Filter, Fusion.

Anfrage "kidney elimination" auf dem synthetischen Korpus:
- Die dichte Attrappe kennt kidney = renal und bevorzugt das Paper zur
  Niereninsuffizienz (10.5555/synth.001), das das Wort "kidney" nie benutzt.
- BM25 findet nur wörtliche Treffer; ganz oben steht der Ergebnisabschnitt von
  10.5555/synth.003, der als Einziger "kidney" enthält (zweimal).
Die Erwartungen werden unabhängig von search() aus direkten Qdrant-Abfragen
der beiden Ranglisten berechnet.
"""
import pytest
from qdrant_client.http.models import ScoredPoint

from retrieval import search as suche

from hilfen import bm25_vektor, dichter_vektor, woerter

ANFRAGE = "kidney elimination"
TIEFE = 30  # Kandidaten je Rangliste, wie search() bei top_k=10


def rangliste(client, using: str, vektor) -> list[str]:
    punkte = client.query_points("papers", query=vektor, using=using, limit=TIEFE,
                                 with_payload=True).points
    return [p.payload["chunk_id"] for p in punkte]


def dicht(client, anfrage=ANFRAGE):
    return rangliste(client, "dense", dichter_vektor(anfrage))


def bm25(client, anfrage=ANFRAGE):
    return rangliste(client, "bm25", bm25_vektor(anfrage, anfrage=True))


def ids(ergebnisse) -> list[str]:
    return [r.chunk.chunk_id for r in ergebnisse]


def test_alpha_1_ist_nur_dicht(korpus, attrappen):
    ergebnis = suche.search(ANFRAGE, top_k=10, hybrid_alpha=1.0)
    assert ids(ergebnis) == dicht(korpus)[:10]
    assert attrappen.bm25 == 0  # BM25 wird gar nicht abgefragt


def test_alpha_0_ist_nur_bm25(korpus, attrappen):
    ergebnis = suche.search(ANFRAGE, top_k=10, hybrid_alpha=0.0)
    assert ids(ergebnis) == bm25(korpus)[:10]
    assert attrappen.dicht == 0  # das Embedding-Modell wird nicht gebraucht
    for r in ergebnis:  # nur wörtliche Treffer
        assert {"kidney", "elimination"} & set(woerter(r.chunk.text))


def test_alpha_entscheidet_ueber_den_ersten_treffer(korpus):
    erster_dicht = suche.search(ANFRAGE, top_k=5, hybrid_alpha=1.0)[0].chunk
    erster_bm25 = suche.search(ANFRAGE, top_k=5, hybrid_alpha=0.0)[0].chunk
    assert erster_dicht.doi == "10.5555/synth.001"           # semantisch: renal ~ kidney
    assert "kidney" not in erster_dicht.text.lower()
    assert erster_bm25.chunk_id == "10.5555/synth.003___3"   # lexikalisch: "kidney"
    assert "kidney" in erster_bm25.text.lower()


@pytest.mark.parametrize("alpha", [0.2, 0.5, 0.7, 0.9])
def test_zwischenwerte_sind_gewichtete_rrf(korpus, alpha):
    """Erwartung nachgerechnet: alpha / (60 + Rang dicht) + (1 - alpha) / (60 + Rang BM25)."""
    erwartet: dict[str, float] = {}
    for gewicht, liste in ((alpha, dicht(korpus)), (1 - alpha, bm25(korpus))):
        for rang, chunk_id in enumerate(liste, start=1):
            erwartet[chunk_id] = erwartet.get(chunk_id, 0.0) + gewicht / (60 + rang)

    ergebnis = suche.search(ANFRAGE, top_k=10, hybrid_alpha=alpha)
    assert len(ergebnis) == 10
    for r in ergebnis:
        assert r.score == pytest.approx(erwartet[r.chunk.chunk_id])
    punkte = [r.score for r in ergebnis]
    assert punkte == sorted(punkte, reverse=True)
    assert punkte[-1] >= sorted(erwartet.values(), reverse=True)[9] - 1e-12


def test_beide_ranglisten_fliessen_ein(korpus):
    """Bei 0.5 steht der BM25-Spitzenreiter unter den ersten fünf, obwohl er dicht weit hinten liegt."""
    assert dicht(korpus).index("10.5555/synth.003___3") >= 5
    ergebnis = ids(suche.search(ANFRAGE, top_k=5, hybrid_alpha=0.5))
    assert "10.5555/synth.003___3" in ergebnis
    assert ergebnis[0] == "10.5555/synth.001___1"  # in beiden Listen weit oben


def test_filter_gilt_fuer_beide_ranglisten(korpus):
    ergebnis = suche.search(ANFRAGE, top_k=10, hybrid_alpha=0.5, year_min=2024)
    assert ergebnis
    assert all(r.chunk.metadata.year >= 2024 for r in ergebnis)
    assert "10.5555/synth.003___3" not in ids(ergebnis)  # 2023, obwohl BM25-Spitze

    nur_methoden = suche.search(ANFRAGE, top_k=10, hybrid_alpha=0.5, section_type="methods")
    assert {r.chunk.section_type for r in nur_methoden} == {"methods"}


def test_einstellung_hybrid_alpha_wirkt(korpus, monkeypatch):
    from config.loader import settings
    monkeypatch.setattr(settings.search, "hybrid_alpha", 0.0)
    assert suche.search_from_settings(ANFRAGE, top_k=3)[0].chunk.chunk_id == "10.5555/synth.003___3"
    monkeypatch.setattr(settings.search, "hybrid_alpha", 1.0)
    assert suche.search_from_settings(ANFRAGE, top_k=3)[0].chunk.doi == "10.5555/synth.001"


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_ungueltiges_alpha(korpus, alpha):
    with pytest.raises(ValueError, match="hybrid_alpha"):
        suche.search(ANFRAGE, hybrid_alpha=alpha)


def test_leere_anfrage_sucht_nur_nach_metadaten(korpus, attrappen):
    assert suche.search("   ", top_k=5) == []  # ohne Filter: nichts
    ergebnis = suche.search("", top_k=50, doi="10.5555/synth.002")
    assert {r.chunk.doi for r in ergebnis} == {"10.5555/synth.002"}
    assert len(ergebnis) == 5
    assert attrappen.dicht == attrappen.bm25 == 0


def test_treffer_enthalten_fundstelle_und_ausschnitt(korpus):
    r = suche.search(ANFRAGE, top_k=1, hybrid_alpha=0.0)[0]
    assert r.chunk.doi == "10.5555/synth.003"
    assert r.chunk.section_type == "results"
    assert "kidney" in r.snippet  # Absatz mit den meisten Anfragewörtern


def test_rrf_formel_und_gleichstand():
    def p(i):
        return ScoredPoint(id=i, version=0, score=0.0, payload={})

    fusion = suche._weighted_rrf([(0.7, [p(1), p(2)]), (0.3, [p(2), p(3)])], limit=10)
    punkte = {punkt.id: wert for wert, punkt in fusion}
    assert punkte[1] == pytest.approx(0.7 / 61)
    assert punkte[2] == pytest.approx(0.7 / 62 + 0.3 / 61)
    assert punkte[3] == pytest.approx(0.3 / 62)
    assert [punkt.id for _, punkt in fusion] == [2, 1, 3]

    # Gleichstand: Reihenfolge des ersten Auftretens (dicht vor BM25)
    gleich = suche._weighted_rrf([(0.5, [p(7)]), (0.5, [p(8)])], limit=10)
    assert [punkt.id for _, punkt in gleich] == [7, 8]
    assert suche._weighted_rrf([], limit=5) == []
