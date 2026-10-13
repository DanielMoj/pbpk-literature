# pbpk-literature: RAG-Wissensbasis mit MCP-Server für die Recherche in Fachliteratur

**English summary.** A local retrieval-augmented knowledge base for scientific papers, built for PBPK and pharmacometrics literature. PDFs are parsed into sections (GROBID; table contents and OCR optionally via Google Document AI), chunked along the sections, embedded (BAAI/bge-large-en-v1.5) and stored in Qdrant together with BM25 sparse vectors. Search is hybrid: a dense and a BM25 ranking are fused by weighted reciprocal rank fusion (`hybrid_alpha`). A Model Context Protocol (MCP) server written in Python exposes search and annotation tools to an LLM client: the server returns passages with their sources, the language model in the client writes the answer. Offline tests (pytest, synthetic corpus, in-memory Qdrant) cover chunking, hybrid search and the server; `eval/` prepares a study of hit rate and source fidelity. The repository contains code only, no PDFs and no extracted texts.

---

## Prinzip

```
eigene PDFs → Triage → GROBID (Sektionen, Tabellenpositionen, Figuren)
                     → Google Document AI (optional: Tabelleninhalte, OCR für Scans)
            → Konsolidierung (JSON je Paper) → Chunks entlang der Sektionen
            → Embeddings (bge-large) + BM25 → Qdrant
                                                 ↑
                    Sprachmodell im Client  ←  MCP-Server (Python)
```

Der MCP-Server liefert Fundstellen mit Quelle (DOI, Abschnitt, Textausschnitt); die Antwort formuliert das angebundene Sprachmodell, etwa Claude. So lässt sich jede Aussage der Antwort an einer Fundstelle prüfen. Wie oft die zitierten Fundstellen die Aussagen tatsächlich tragen, misst die vorbereitete [Auswertung](#auswertung).

## MCP-Werkzeuge

| Werkzeug | Funktion |
|---|---|
| `search_papers` | hybride Suche (dichte Embeddings und BM25, gewichtet mit `hybrid_alpha`) mit Metadatenfiltern (Zeitschrift, Jahr, Tags, Lückenstatus, Relevanz) |
| `get_paper` | alle Abschnitte eines Papers in Dokumentreihenfolge und seine Annotationen, zu einer DOI |
| `find_similar` | nächste Nachbarn im Embedding-Raum, ausgehend von einem Abschnitt (Standard: Abstract) |
| `search_by_author` | Suche nach Autorinnen und Autoren (Textabgleich in den gespeicherten Autorenangaben, z. B. Nachname) |
| `annotate_paper` | Tags, Relevanz, Notiz und Lückenstatus setzen |
| `get_gap_landscape` | Übersicht der annotierten Paper nach Lückenstatus, optional zu einem Thema |

## Hybride Suche

`search_papers` fragt Qdrant zweimal mit demselben Metadatenfilter ab, einmal mit dem Embedding der Anfrage, einmal mit ihrem BM25-Vektor, und fügt beide Ranglisten mit gewichteter Reciprocal Rank Fusion zusammen:

```
score = hybrid_alpha / (60 + Rang dicht) + (1 − hybrid_alpha) / (60 + Rang BM25)
```

Verfahren und Konstante 60 stammen aus der Originalarbeit zur Reciprocal Rank Fusion, die die Ranglisten ungewichtet summiert: Cormack, Clarke und Büttcher (2009), SIGIR '09, S. 758–759, doi:10.1145/1571941.1572114. Die Gewichtung mit `hybrid_alpha` ist eine Erweiterung.

`hybrid_alpha` steht in `config/settings.yaml` (Standard 0.7). Bei 1 zählt nur die dichte Rangliste, bei 0 nur BM25; eine Rangliste mit Gewicht 0 wird gar nicht abgefragt, bei 0 wird also auch kein Embedding-Modell geladen. Ist fastembed nicht verfügbar, meldet die Suche das im Log und es zählt nur die dichte Rangliste (bei `hybrid_alpha` 0 bleibt das Ergebnis dann leer). Die BM25-Gewichte der Abschnitte erzeugt fastembed (`Qdrant/bm25`), die IDF berechnet Qdrant.

## Aufbau

```
config/        Einstellungen (Qdrant, Embedding-Modell, Chunking, Suche, MCP-Transport), Zeitschriftenliste
extraction/    PDF → strukturiertes JSON (Triage, GROBID, Document AI, Bilder, Konsolidierung)
ingestion/     Chunking, Embeddings, Ingest in Qdrant, DOI-Listen und Metadaten
retrieval/     hybride Suche, Datenmodelle, Qdrant-Client
mcp_server/    MCP-Server und Werkzeuge
scripts/       Qdrant-Collection anlegen, Annotationen sichern und zurückschreiben, Demo
eval/          Auswertung: Trefferquote und Quellentreue
tests/         pytest, synthetisches Korpus in tests/fixtures/
```

Die Extraktion ist in [`EXTRACTION_ARCHITECTURE_DETAILED.md`](EXTRACTION_ARCHITECTURE_DETAILED.md) beschrieben.

## Installation

Voraussetzungen:

- Python: getestet mit 3.11.17, 3.12.15 und 3.14.4 unter Windows 11 (Paketversionen unter [Teststand](#teststand)).
- Qdrant als Docker-Container: `docker-compose.yaml` startet `qdrant/qdrant:v1.19.2`, passend zu qdrant-client 1.19 (getestet 1.19.1; der Client verlangt höchstens eine Minor-Version Abstand zum Server). Die Tests nutzen Qdrant im Speicher; zusätzlich ist die Fassung gegen einen Container in dieser Version geprüft (siehe [Teststand](#teststand)).
- Für die Extraktion ein laufender [GROBID](https://github.com/kermitt2/grobid)-Server (`http://localhost:8070`); nicht Teil der Tests.

```bash
docker compose up -d                      # Qdrant, nur unter 127.0.0.1 erreichbar
pip install -r requirements.txt
python scripts/setup_qdrant.py            # Collection anlegen (dicht + BM25 mit IDF)
```

Das Embedding-Gerät steht in `config/settings.yaml` auf `device: auto`: CUDA, wenn PyTorch eine GPU findet, sonst CPU. Eine GPU ist nicht nötig. Ob pip eine PyTorch-Fassung mit CUDA installiert, hängt vom System ab; unter Windows liefert PyPI nur die CPU-Fassung, die CUDA-Fassung gibt es über [pytorch.org](https://pytorch.org).

Google Document AI ist optional, kostenpflichtig und standardmäßig aus (`google.enabled: false` in `extraction/config.yaml`). Wer es nutzen will, installiert `google-cloud-documentai`, setzt `enabled: true`, trägt die Kennungen des eigenen Google-Cloud-Projekts ein und gibt die Zugangsdaten über `GOOGLE_APPLICATION_CREDENTIALS` an. Ohne Document AI gehen Tabellen nicht in die Suche ein: Die Zerlegung liest Tabelleninhalte nur aus den Zellen von Document AI, nicht aus dem Tabellen-XML, das GROBID mitliefert.

## Benutzung

1. Eigene, rechtmäßig beschaffte PDFs ablegen: je Paper ein Unterordner in `data/pdfs/` mit `paper.pdf` und `metadata.json`. Ohne `metadata.json` überspringt die Extraktion das Paper. Die Datei liefert DOI, Zeitschrift und Jahr; Titel, Autorinnen und Autoren sowie das Abstract übernimmt die Pipeline bevorzugt aus GROBID:

   ```json
   {"doi": "10.1234/beispiel.2024.001", "title": "…", "authors": ["Mustermann E"], "journal": "…", "year": 2024}
   ```

2. Extraktion: `cd extraction && python main_pipeline.py`
3. Ingest: `python -m ingestion.reingest --input data/extracted/`
4. MCP-Server im Client eintragen, etwa in Claude Code:

```json
{ "mcpServers": { "pbpk-literature": { "command": "python", "args": ["mcp_server/server.py"] } } }
```

Standardmäßig läuft der Server über `stdio`. Mit `--transport sse` oder `--transport streamable-http` ist er über HTTP erreichbar; er lauscht dann auf `127.0.0.1`, Port aus `mcp.sse_port` (Standard 8080) oder `--port`. Der Schutz gegen DNS-Rebinding lässt dann nur Host-Header von `localhost`, `127.0.0.1` und `[::1]` zu, außer `PBPK_MCP_ALLOW_ANY_HOST=1` ist gesetzt. Das ist nur hinter einem anderweitig abgesicherten Zugang sinnvoll. Beim Start lädt der Server weder das Embedding-Modell noch verbindet er sich mit Qdrant; beides geschieht beim ersten Werkzeugaufruf. Unter Windows importiert er beim stdio-Transport aber schon die Bibliotheken (numpy, torch, fastembed), was einige Sekunden dauert: Der stdio-Transport liest stdin dort blockierend in einem eigenen Thread, und ein Werkzeug, das eine kompilierte Bibliothek erst beim ersten Aufruf lädt, bliebe dabei hängen.

Weitere Werkzeuge: `python -m ingestion.fetch_metadata` holt DOI-Listen für die Zeitschriften aus `config/journals.yaml`, `python -m ingestion.enrich_doi_lists` ergänzt Abstracts.

### Wissensbasis neu aufbauen, Annotationen behalten

Neu aufbauen muss man eine Wissensbasis aus einer früheren Fassung: Ältere Collections haben keine BM25-Vektoren oder keine IDF, und Bestände älterer Qdrant-Versionen lassen sich nicht direkt unter 1.19 öffnen, weil Qdrant beim Upgrade keine Minor-Version überspringen kann ([Qdrant-Dokumentation](https://qdrant.tech/documentation/upgrades/)); `docker-compose.yaml` legt deshalb ein eigenes Volume an. Das Einspielen setzt die Annotationen (Lückenstatus, Tags, Relevanz, Notiz) zurück. Sie werden vorher gesichert und danach über die DOI wieder zugeordnet:

```bash
python scripts/export_annotations.py --all --format json        # 1. sichern nach data/annotations_<Zeit>.json
python scripts/setup_qdrant.py --recreate                        # 2. Collection neu anlegen (löscht alle Daten)
python -m ingestion.reingest --input data/extracted/             # 3. neu einspielen
python scripts/import_annotations.py --input data/annotations_<Zeit>.json --dry-run   # 4. prüfen
python scripts/import_annotations.py --input data/annotations_<Zeit>.json             #    und zurückschreiben
```

`--all` gehört zur Sicherung: Ohne die Option erfasst der Export nur Paper mit Lückenstatus, Tags oder Notizen ohne Status fehlten dann. Der Import ordnet über die DOI zu (Groß- und Kleinschreibung egal), überschreibt die vier Felder auf allen Chunks eines Papers mit den gesicherten Werten und listet DOIs auf, die nicht mehr in der Collection stehen. Er liest auch die CSV-Datei des Exports.

## Beispiel

Erzeugt mit `python scripts/demo_synthetisch.py --frage "Why was kidney impairment not considered for Beispielin?" --top-k 2`. Das Skript spielt das synthetische Testkorpus (`tests/fixtures/korpus`, vier erfundene Paper) mit den echten Modellen in Qdrant im Speicher ein und ruft `search_papers` auf; es braucht weder Docker noch GROBID, lädt aber beim ersten Lauf das Embedding-Modell (rund 1,3 GB). Antwort, beim zweiten Treffer gekürzt auf DOI, Abschnitt, Ausschnitt und Score:

```json
{
  "results": [
    {
      "doi": "10.5555/synth.003",
      "title": "Hepatic impairment and CYP3A4 inhibition: simulated exposure of the fictional drug Beispielin",
      "authors": ["John Doe, Example Institute of Pharmacometrics", "Jane Doe, Example Institute of Pharmacometrics"],
      "year": 2023,
      "journal": "Journal of Synthetic Test Data",
      "section_type": "results",
      "snippet": "Unlike the liver, the kidney contributes less than five percent to the elimination of Beispielin, so kidney impairment was not considered further in this simulation.",
      "score": 0.0164,
      "gap_status": "UNASSESSED",
      "tags": [],
      "relevant_for": []
    },
    {
      "doi": "10.5555/synth.003",
      "section_type": "abstract",
      "snippet": "Beispielin is a fictional drug that is metabolised almost completely by CYP3A4 in the liver. We simulated its exposure in patients with hepatic impairment and during co-administration of a strong CYP3A4 inhibitor. […]",
      "score": 0.0161
    }
  ],
  "total": 2
}
```

Die Antwort an die Nutzerin oder den Nutzer formuliert das Sprachmodell im Client aus diesen Fundstellen.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Die Tests laufen ohne Netz, GPU, Docker und Modell-Download. Statt bge-large und fastembed rechnen zwei deterministische Attrappen (`tests/hilfen.py`), Qdrant läuft im Speicher (`QdrantClient(":memory:")`), Testdaten sind vier selbst geschriebene, erfundene Paper im konsolidierten JSON-Format (`tests/fixtures/korpus`). Eine Sicherung in `tests/conftest.py` verhindert, dass ein Test einen laufenden Qdrant-Server erreicht.

Geprüft wird die Logik:

- Zerlegung: Chunks entlang der Abschnitte, Zusammenlegen kurzer Abschnitte, Teilen langer an Absatz-, Satz- und Wortgrenzen, Mindest- und Höchstlänge, Subword-Faktor, Tabellen, Literaturliste, Metadaten aus GROBID und Sidecar.
- Hybride Suche: Wirkung von `hybrid_alpha` (0 nur BM25, 1 nur dicht) gegen direkt in Qdrant abgefragte Ranglisten, nachgerechnete gewichtete Fusion, Filter auf beiden Ranglisten.
- MCP-Server: die sechs Werkzeuge und ihre Eingabeschemata, Start über stdio mit `initialize` und `list_tools`, ein Werkzeugaufruf über stdio, der kompilierte Bibliotheken nachlädt (Vorladen unter Windows), Start ohne Modell und ohne Qdrant, Host-Header-Schutz (fremder Host abgelehnt, mit `PBPK_MCP_ALLOW_ANY_HOST=1` zugelassen), die Werkzeuge selbst gegen das Korpus im Speicher.
- Annotationen: Sicherung, Neuaufbau und Import über die DOI (JSON und CSV, Trockenlauf, fehlende DOIs).
- Konfiguration (Standardwerte, `device: auto`) und die Skripte in `eval/`.

Nicht geprüft wird die Güte der Treffer an echter Literatur; dafür ist die [Auswertung](#auswertung) vorbereitet. Ebenfalls nicht abgedeckt: die Extraktion mit GROBID und Document AI, die echten Modelle (nur über das Demo-Skript) und ein Qdrant-Server statt des Speichermodus (einmal von Hand geprüft, siehe Teststand).

## Auswertung

Der Ordner [`eval/`](eval/README.md) enthält das Werkzeug für eine Vorstudie zur Messmethodik: Trefferquote (Treffer@5 und MRR) an einem Fragenkatalog mit Soll-Fundstellen, berechnet mit derselben Suchfunktion wie `search_papers`, und eine Vorlage zur manuellen Bewertung der Quellentreue. Die Vorstudie ist in Vorbereitung, der Fragenkatalog folgt; Ergebnisse enthält das Repository noch nicht.

## Übertragbarkeit

Extraktion, Zerlegung, hybride Suche und MCP-Server enthalten kein PBPK-spezifisches Wissen. Das Fachgebiet steckt in der Literaturauswahl:

- `config/journals.yaml` legt fest, für welche Zeitschriften (ISSN, ab Jahrgang, optional Schlagwörter) `python -m ingestion.fetch_metadata` DOI-Listen holt; gelesen wird die Datei nur dort. Einträge ohne Schlagwörter fragt das Skript bei CrossRef ab, das Zeitschriften aller Fächer kennt; Einträge mit Schlagwörtern bei EuropePMC, das nur die Lebenswissenschaften abdeckt. Für Literatur zu Controlling, Prognose oder Logistik trägt man die Zeitschriften daher ohne Schlagwörter ein.
- Was durchsucht wird, bestimmen die PDFs in `data/pdfs/` und ihre `metadata.json`.

Grenzen: Embedding-Modell und BM25 sind auf englische Texte ausgelegt (bge-large-en-v1.5; fastembed-BM25 mit englischen Stoppwörtern und Stammformen). Das Modell lässt sich in `config/settings.yaml` tauschen; ein Modell mit anderer Vektorgröße braucht eine entsprechend angelegte Collection (`create_collection(..., dense_size=…)` in `scripts/setup_qdrant.py`, Standard 1024), die BM25-Sprache ist im Code fest. Die Zuordnung der Abschnittsüberschriften kennt die üblichen Teile empirischer Artikel; andere Überschriften laufen als `other` mit. Die Beschreibungen der MCP-Werkzeuge, die das Sprachmodell liest, nennen PBPK-Beispiele (`mcp_server/tools/`).

## Grenzen und nicht enthalten

- Keine PDFs, keine extrahierten Volltexte, keine Vektordatenbank: Die Wissensbasis wird aus der eigenen Literatursammlung aufgebaut.
- Kein Werkzeug zum Herunterladen von Volltexten.
- Tabellen gehen nur mit Document AI in die Suche ein (siehe Installation); Tabellen-Chunks werden nicht nach Länge geteilt.
- Geprüft wurde das System bisher im Eigeneinsatz an PBPK-Literatur und mit den Tests oben; eine systematische Auswertung an einem festen Fragenkatalog steht aus.

## Teststand

13.10.2026, Windows 11, je eine frisch angelegte virtuelle Umgebung mit Python 3.11.17, 3.12.15 und 3.14.4 (`pip install -r requirements-dev.txt`): `python -m compileall -q .` und der Import aller Module ohne Fehler, `pytest -q` → **99 passed** (14 bis 27 s; die stdio-Tests starten den Server mit Vorladen). Der MCP-Server startete über stdio ohne laufendes Qdrant; `initialize` und `list_tools` lieferten die sechs Werkzeuge. Gegen einen Container `qdrant/qdrant:v1.19.2` (Python 3.12, synthetisches Korpus mit den echten Modellen eingespielt) liefen alle sechs Werkzeuge direkt und über MCP/stdio, die Autorensuche mit Nachnamen und vollem Namen sowie der Neuaufbau mit Export und Import der Annotationen, ohne Versionswarnung des Clients. Das Demo-Skript lief unter Python 3.12 und 3.14 mit den echten Modellen auf der CPU (`device: auto`, PyTorch ohne CUDA, BM25 über fastembed). Getestete Versionen: mcp 1.30.0, qdrant-client 1.19.1, sentence-transformers 6.1.0, fastembed 0.9.0, torch 2.14.1 (CPU), PyMuPDF 1.28.2, pytest 9.1.1.

## Entstehung

Der Code ist mit agentengestützter Codeerzeugung entstanden. Eigener Anteil: fachliche Spezifikation, Aufbau der Pipeline und der Wissensbasis, Wahl von Chunking, Embedding und hybrider Suche, Anbindung über MCP sowie Prüfung und Freigabe.

## Zitieren

Bitte über die DOI der jeweiligen Version zitieren (Zenodo); die Angaben stehen in [`CITATION.cff`](CITATION.cff).

## Lizenz

Eigener Code: [MIT](LICENSE).

### Drittkomponenten und Lizenzen

Das Repository enthält keinen fremden Code und keine Modelle; die Komponenten werden installiert, als eigener Dienst betrieben oder beim ersten Lauf geladen.

| Komponente | Rolle | Lizenz (nachgesehen in) |
|---|---|---|
| [PyMuPDF](https://github.com/pymupdf/PyMuPDF) | PDF lesen, Figuren (nur `extraction/`) | AGPL-3.0 oder kommerzielle Lizenz von Artifex (README des Projekts, Paketmetadaten) |
| [GROBID](https://github.com/kermitt2/grobid) | Struktur der PDFs, eigener Server | Apache-2.0 (LICENSE im Repository) |
| [Qdrant](https://github.com/qdrant/qdrant) und qdrant-client | Vektordatenbank (Docker-Image) und Python-Client | Apache-2.0 (LICENSE im Repository, Paketmetadaten) |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | Embedding-Modell | MIT (Modellkarte) |
| fastembed und [Qdrant/bm25](https://huggingface.co/Qdrant/bm25) | BM25-Vektoren (Stoppwörter, Stammformen) | Apache-2.0 (Paketmetadaten, Modellseite) |
| sentence-transformers | rechnet das Embedding-Modell | Apache-2.0 (Paketmetadaten) |
| mcp (MCP Python SDK) | MCP-Server | MIT (Paketmetadaten) |

PyMuPDF steht unter der AGPL-3.0. MCP-Server und Suche importieren es nicht. Wer die Extraktion zusammen mit PyMuPDF weitergibt oder eine veränderte Fassung als Dienst über ein Netzwerk anbietet, muss die Bedingungen der AGPL-3.0 einhalten (Quelltext für die Empfänger beziehungsweise, nach § 13, für die Nutzer des Dienstes); alternativ bietet Artifex eine kommerzielle Lizenz an.
