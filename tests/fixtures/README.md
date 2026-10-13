# Synthetisches Testkorpus

`korpus/` enthält vier erfundene Paper im konsolidierten JSON-Format, wie es `extraction/improved_consolidator.py` schreibt (erzeugt mit dem Konsolidierer der Pipeline). Alle Texte sind für die Tests selbst geschrieben; Wirkstoffe (Fiktivol, Beispielin), Zeitschrift, Verlag sowie Autorinnen und Autoren (Platzhalternamen) sind erfunden, die DOIs nutzen das Test-Präfix 10.5555. Zu keinem Paper gibt es ein PDF.

| Datei | Thema | Prüft vor allem |
|---|---|---|
| `10.5555_synth.001_extracted.json` | PBPK-Modell, Niereninsuffizienz | Zerlegung: kurzer Abschnitt vorne angefügt, langer an Absätzen geteilt, kurzes Fazit an die Diskussion gehängt, Aufzählung, Tabelle mit und ohne Inhalt, Literaturliste |
| `10.5555_synth.002_extracted.json` | Bioäquivalenz mit TOST | Suche, Annotationen, Autorensuche |
| `10.5555_synth.003_extracted.json` | Leberinsuffizienz, CYP3A4-Hemmung | einziger Abschnitt mit dem Wort "kidney": wörtlicher Treffer für BM25 |
| `10.5555_synth.004_extracted.json` | Dosiswahl bei Kindern | Filter nach Jahr, ähnliche Paper |
