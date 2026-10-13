"""
Testhilfen: Pfade, Attrappen für Embedding und BM25, Laden von Skripten.

Die Tests laufen ohne Netz, GPU, Docker und Modell-Download. Statt bge-large
und fastembed rechnen zwei deterministische Attrappen:

- dichter_vektor: zählt je "Konzept" die Wörter, die mit einem Stamm aus
  KONZEPTE beginnen (kidney und renal landen im selben Konzept). Sie verhält
  sich damit wie ein semantisches Modell, das Synonyme kennt.
- bm25_vektor: Termhäufigkeiten der Wörter ohne Stoppwörter; Qdrant ergänzt
  die IDF. Sie findet nur Wörter, die wörtlich vorkommen.
"""
import importlib.util
import re
import sys
import zlib
from pathlib import Path

from qdrant_client.http.models import SparseVector

PROJEKT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
KORPUS = FIXTURES / "korpus"

KONZEPTE = {
    "niere": ("kidney", "renal", "glomerul", "nephr", "creatinin", "filtration"),
    "leber": ("liver", "hepat", "cyp3a4"),
    "elimination": ("clearance", "elimination", "eliminat", "cleared", "excret"),
    "kinder": ("child", "paediatr", "pediatr", "infant"),
    "aequivalenz": ("equivalen", "bioequivalen", "tost", "sided"),
    "simulation": ("simulat", "virtual", "monte"),
    "wechselwirkung": ("interaction", "inhibit", "comedication"),
    "dosis": ("dose", "dosing", "regimen"),
}
DIM = len(KONZEPTE) + 1  # plus konstante Komponente, damit kein Vektor null ist

STOPPWOERTER = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "we",
    "were", "what", "which", "why", "with",
}


def woerter(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def dichter_vektor(text: str) -> list[float]:
    """Deterministischer Ersatz für bge-large: Konzepthäufigkeiten plus Konstante."""
    tokens = woerter(text)
    vektor = [
        float(sum(1 for t in tokens if t.startswith(staemme)))
        for staemme in KONZEPTE.values()
    ]
    return vektor + [0.3]


def bm25_vektor(text: str, anfrage: bool = False) -> SparseVector:
    """Deterministischer Ersatz für fastembed/BM25: Termhäufigkeiten, Anfrage mit Gewicht 1."""
    zaehler: dict[int, float] = {}
    for t in woerter(text):
        if t in STOPPWOERTER:
            continue
        index = zlib.crc32(t.encode()) & 0x7FFFFFFF
        zaehler[index] = 1.0 if anfrage else zaehler.get(index, 0.0) + 1.0
    indizes = sorted(zaehler)
    return SparseVector(indices=indizes, values=[zaehler[i] for i in indizes])


def lade_skript(relativer_pfad: str):
    """Lädt ein Skript (scripts/, eval/) über seinen Pfad als Modul, ohne Paketnamen."""
    pfad = PROJEKT / relativer_pfad
    name = "skript_" + pfad.stem
    spec = importlib.util.spec_from_file_location(name, pfad)
    modul = importlib.util.module_from_spec(spec)
    sys.modules[name] = modul
    spec.loader.exec_module(modul)
    return modul
