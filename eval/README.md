# Auswertung: Trefferquote und Quellentreue

**Stand: Vorstudie in Vorbereitung, der Fragenkatalog folgt.** Dieser Ordner enthält das Werkzeug, noch keine Ergebnisse.

Die Vorstudie zur Messmethodik soll an 30 bis 50 Fachfragen mit Soll-Fundstelle zeigen, wie oft die Soll-Fundstelle unter den ersten fünf Treffern liegt (Trefferquote) und welcher Anteil der Aussagen in den Antworten durch die zitierten Quellen gedeckt ist (Quellentreue).

## Messgrößen

### Trefferquote der Suche

Für jede Frage ruft `trefferquote.py` dieselbe Suchfunktion auf wie das MCP-Werkzeug `search_papers` (`retrieval.search.search_from_settings`, Einstellungen aus `config/settings.yaml`, ohne Metadatenfilter) und sucht in der Trefferliste die erste passende Fundstelle. Passend heißt: gleiche DOI und, falls angegeben, gleicher Abschnittstyp (`soll_abschnitt`, etwa `results`) oder gleiche Chunk-ID (`soll_chunk_id`).

Mit N Fragen und rang_i als Rang der ersten passenden Fundstelle von Frage i:

```
Treffer@5 = (1/N) · Σ 1[rang_i ≤ 5]
MRR       = (1/N) · Σ 1/rang_i        (0, wenn die Fundstelle nicht unter den ersten 10 Treffern liegt)
```

`k` (Standard 5) und die Länge der Trefferliste (`--top-k`, Standard 10 wie bei `search_papers`) lassen sich einstellen. Mit `--alpha` läuft dieselbe Frageliste mit einem anderen `hybrid_alpha`, etwa 0 (nur BM25) und 1 (nur Embeddings) zum Vergleich mit der hybriden Suche.

### Quellentreue der Antworten

Die Antworten formuliert das Sprachmodell im MCP-Client aus den Fundstellen. Jede Antwort wird von Hand in einzelne Aussagen zerlegt; je Aussage wird die zitierte Fundstelle nachgelesen und bewertet:

| `gedeckt` | Bedeutung |
|---|---|
| `ja` | die zitierte Fundstelle trägt die Aussage vollständig |
| `teilweise` | nur ein Teil der Aussage steht in der Fundstelle |
| `nein` | die Fundstelle trägt die Aussage nicht, oder es ist keine angegeben |

`quellentreue.py` weist den Anteil gedeckter Aussagen gesamt und je Frage zweifach aus: streng (nur `ja`) und großzügig (`ja` oder `teilweise`). Leere Urteile gelten als noch nicht bewertet und zählen nicht mit.

## Dateien

| Datei | Inhalt |
|---|---|
| `fragen.beispiel.jsonl` | Format des Fragenkatalogs: je Zeile `id`, `frage`, `soll_doi` und `soll_abschnitt` oder `soll_chunk_id`. Die drei Zeilen beziehen sich auf das synthetische Testkorpus (`tests/fixtures/korpus`), nicht auf echte Literatur. |
| `trefferquote.py` | Treffer@k und MRR; schreibt je Frage eine CSV-Zeile und eine Zusammenfassung als JSON (mit `hybrid_alpha`, Modell, Collection, Zeitpunkt) |
| `quellentreue_vorlage.csv` | Vorlage für die manuelle Bewertung: `frage_id`, `aussage`, `zitierte_fundstelle`, `gedeckt`, `anmerkung` |
| `quellentreue.py` | wertet die ausgefüllte Vorlage aus |

CSV-Dateien nutzen das Semikolon als Trennzeichen und UTF-8, damit Excel sie direkt öffnet; `quellentreue.py` liest auch Komma-getrennte Dateien.

## Ablauf

1. **Fragenkatalog festlegen.** `eval/fragen.jsonl` nach dem Muster von `fragen.beispiel.jsonl` anlegen, die Soll-Fundstelle je Frage vor dem ersten Lauf bestimmen. Die Chunk-ID hat die Form `<doi>___<n>`; `n` ist die Position des Abschnitts in der Ausgabe von `get_paper`, gezählt ab 0.
2. **Trefferquote messen** (Qdrant mit der Wissensbasis muss laufen):

   ```bash
   python eval/trefferquote.py --fragen eval/fragen.jsonl
   python eval/trefferquote.py --fragen eval/fragen.jsonl --alpha 0    # Vergleich: nur BM25
   python eval/trefferquote.py --fragen eval/fragen.jsonl --alpha 1    # Vergleich: nur Embeddings
   ```

   Die Ausgabe landet in `eval/ergebnisse/` (`trefferquote_<Zeitpunkt>.csv` und `.json`).
3. **Antworten erzeugen.** Jede Frage im MCP-Client (etwa Claude mit diesem Server) stellen und die Antwort mit ihren Quellenangaben sichern.
4. **Quellentreue bewerten.** `quellentreue_vorlage.csv` kopieren, je Aussage eine Zeile ausfüllen, dann:

   ```bash
   python eval/quellentreue.py --bewertung eval/quellentreue.csv --json eval/ergebnisse/quellentreue.json
   ```

`tests/test_eval.py` lässt `trefferquote.py` auf dem synthetischen Korpus laufen, mit Attrappen statt der echten Modelle; die Zahlen daraus prüfen nur die Rechnung und sind keine Ergebnisse.
