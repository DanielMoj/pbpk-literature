"""Konfiguration (config/loader.py): Standardwerte, mitgelieferte settings.yaml, device: auto."""
import sys
from types import SimpleNamespace

import pytest

from config.loader import PROJECT_ROOT, Settings, load_settings, resolve_device


def test_mitgelieferte_einstellungen():
    s = load_settings()
    assert (s.qdrant.host, s.qdrant.port, s.qdrant.collection) == ("127.0.0.1", 6333, "papers")
    assert s.embedding.model == "BAAI/bge-large-en-v1.5"
    assert s.embedding.device == "auto"
    assert (s.chunking.min_chunk_length, s.chunking.max_chunk_length, s.chunking.subword_factor) == (50, 350, 1.3)
    assert (s.search.default_top_k, s.search.hybrid_alpha) == (10, 0.7)
    assert (s.mcp.transport, s.mcp.sse_port) == ("stdio", 8080)
    assert PROJECT_ROOT.joinpath("config", "settings.yaml").exists()


def test_fehlende_werte_bekommen_standardwerte(tmp_path):
    datei = tmp_path / "settings.yaml"
    datei.write_text("qdrant:\n  port: 7000\nsearch:\n  hybrid_alpha: 0.2\nmcp:\n", encoding="utf-8")
    s = load_settings(datei)
    assert s.qdrant.port == 7000
    assert s.qdrant.host == "127.0.0.1"          # fehlender Schlüssel
    assert s.search.hybrid_alpha == 0.2
    assert s.search.default_top_k == 10
    assert s.embedding.device == "auto"          # fehlender Abschnitt
    assert s.chunking.subword_factor == 1.3
    assert s.mcp.transport == "stdio"            # leerer Abschnitt


def test_leere_datei_ergibt_nur_standardwerte(tmp_path):
    datei = tmp_path / "settings.yaml"
    datei.write_text("", encoding="utf-8")
    assert load_settings(datei) == Settings()


def test_unbekannter_schluessel_faellt_auf(tmp_path):
    datei = tmp_path / "settings.yaml"
    datei.write_text("search:\n  hybrid_alfa: 0.5\n", encoding="utf-8")
    with pytest.raises(TypeError, match="hybrid_alfa"):
        load_settings(datei)


def torch_attrappe(cuda: bool):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))


@pytest.mark.parametrize("cuda, erwartet", [(True, "cuda"), (False, "cpu")])
def test_auto_waehlt_cuda_nur_wenn_vorhanden(monkeypatch, cuda, erwartet):
    monkeypatch.setitem(sys.modules, "torch", torch_attrappe(cuda))
    assert resolve_device("auto") == erwartet


@pytest.mark.parametrize("geraet", ["cpu", "cuda", "cuda:1"])
def test_feste_geraete_bleiben(monkeypatch, geraet):
    monkeypatch.setitem(sys.modules, "torch", torch_attrappe(False))
    assert resolve_device(geraet) == geraet


@pytest.mark.parametrize("modulname", ["retrieval.search", "ingestion.embed_and_store"])
def test_modell_wird_mit_aufgeloestem_geraet_geladen(monkeypatch, modulname):
    """Suche und Ingestion laden SentenceTransformer ohne GPU auf der CPU."""
    import importlib

    modul = importlib.import_module(modulname)
    geladen = {}

    class ModellAttrappe:
        def __init__(self, name, device):
            geladen.update(name=name, device=device)

    monkeypatch.setitem(sys.modules, "torch", torch_attrappe(False))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=ModellAttrappe))
    monkeypatch.setattr(modul, "_dense_model", None)

    modul._get_dense_model("BAAI/bge-large-en-v1.5", "auto")
    assert geladen == {"name": "BAAI/bge-large-en-v1.5", "device": "cpu"}
