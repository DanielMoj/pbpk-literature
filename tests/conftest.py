"""
Fixtures: Attrappen statt Modellen und das synthetische Korpus in Qdrant im Speicher.
Die Attrappen selbst stehen in hilfen.py.
"""
import pytest
from qdrant_client import QdrantClient

from hilfen import DIM, KORPUS, bm25_vektor, dichter_vektor, lade_skript


@pytest.fixture(autouse=True)
def kein_qdrant_server(monkeypatch):
    """Sicherung: Kein Test erreicht einen laufenden Qdrant-Server (etwa die echte Wissensbasis)."""
    def gesperrt(*args, **kwargs):
        raise RuntimeError("In Tests ist nur Qdrant im Speicher erlaubt (Fixture korpus)")

    monkeypatch.setattr("retrieval.client.QdrantClient", gesperrt)


class Zaehler:
    """Zählt Aufrufe der Anfrage-Encoder (zeigt, welche Rangliste abgefragt wurde)."""

    def __init__(self):
        self.dicht = 0
        self.bm25 = 0


@pytest.fixture
def attrappen(monkeypatch) -> Zaehler:
    """Ersetzt bge-large und fastembed in Ingestion und Suche durch die Attrappen."""
    import ingestion.embed_and_store as einspielen
    import retrieval.search as suche

    zaehler = Zaehler()

    def anfrage_dicht(query, *args, **kwargs):
        zaehler.dicht += 1
        return dichter_vektor(query)

    def anfrage_bm25(query):
        zaehler.bm25 += 1
        return bm25_vektor(query, anfrage=True)

    monkeypatch.setattr(einspielen, "_encode_dense",
                        lambda texts, *a, **k: [dichter_vektor(t) for t in texts])
    monkeypatch.setattr(einspielen, "_encode_sparse",
                        lambda texts: [bm25_vektor(t) for t in texts])
    monkeypatch.setattr(suche, "_encode_query_dense", anfrage_dicht)
    monkeypatch.setattr(suche, "_encode_query_sparse", anfrage_bm25)
    return zaehler


@pytest.fixture
def korpus(attrappen):
    """
    Synthetisches Korpus in Qdrant im Speicher, eingespielt über dieselben
    Funktionen wie im Betrieb: setup_qdrant.create_collection,
    chunk_papers.chunk_directory und embed_and_store.embed_and_store.
    """
    from ingestion.chunk_papers import chunk_directory
    from ingestion.embed_and_store import embed_and_store
    from retrieval.client import use_client

    setup_qdrant = lade_skript("scripts/setup_qdrant.py")
    client = QdrantClient(":memory:")
    setup_qdrant.create_collection(client, "papers", dense_size=DIM)
    use_client(client)
    try:
        chunks = chunk_directory(KORPUS, subword_factor=1.3)
        embed_and_store(chunks, collection="papers")
        yield client
    finally:
        use_client(None)
        client.close()
