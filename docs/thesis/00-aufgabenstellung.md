# Aufgabenstellung

> Migriert aus Notion am 2026-10-28 (Seite `db99243cf1924e77bd2bb49a0b864f87`, zuletzt bearbeitet 2026-07-23). Text unverändert übernommen.

**BHT Berlin · FB III · Luxemburger Str. 10 · 13353 Berlin**

**Betreuer:** Prof. Dr. Ronny Schomacker

## Aufgabenstellung für die Masterarbeit von Silas Pignotti

**Titel:** *Urban-kontextsensitives Downscaling satellitenbasierter Oberflächentemperaturen mit Deep Learning: Eine Ablationsstudie für Berlin*

### Einleitung

Der anthropogene Klimawandel erhöht die thermische Belastung in hochverdichteten urbanen Räumen. In Berlin begünstigen Versiegelung, geringe Verdunstung, eingeschränkte Durchlüftung sowie komplexe Gebäude- und Vegetationsstrukturen die Ausbildung ausgeprägter Surface Urban Heat Islands. Für kleinräumige stadtklimatische Analysen werden daher hochaufgelöste Informationen zur Land Surface Temperature benötigt.

Thermale Landsat-8/9-Daten ermöglichen eine flächendeckende Ableitung der Oberflächentemperatur, sind für Analysen auf Block-, Straßen- oder Freiflächenebene jedoch räumlich zu grob. Sentinel-2 liefert hochaufgelöste spektrale Proxy-Informationen, aber keine direkte thermale Messung. Deep-Learning-basierte Downscaling-Verfahren können die grob aufgelösten thermalen Landsat-Messungen mit hochaufgelösten Sentinel-2-Daten sowie urbanen und meteorologischen Kontextinformationen kombinieren. Das Modell muss dabei die Einflüsse von Bebauung, Versiegelung, Vegetation, Verschattung und Wetterbedingungen abbilden und zugleich sicherstellen, dass die erzeugten 10-m-Muster bei der Rückaggregation mit der gemessenen Landsat-LST übereinstimmen. Da unabhängige thermale Referenzdaten in 10 m Auflösung fehlen, muss die Ergebnisqualität zusätzlich anhand räumlicher Strukturmetriken und ergänzender Referenzdaten kritisch bewertet werden.

### Ziel

Ziel der Masterarbeit ist die Entwicklung, Anwendung und kritische Evaluation eines Deep-Learning-basierten Ansatzes zum Downscaling satellitenbasierter Land Surface Temperature für Berlin. Landsat-basierte LST wird dabei mithilfe von Sentinel-2-Daten und urbanen Kontextinformationen auf eine Zielauflösung von 10 m übertragen.

Eine kontrollierte Ablationsstudie quantifiziert den Beitrag spektraler und morphologischer Eingangsgrößen, von Verschattung und Sonnengeometrie sowie meteorologischer Kontextvariablen zur Modellgüte. Ein ausbleibender Beitrag einzelner Gruppen gilt ebenfalls als valides Ergebnis. Ergänzend wird untersucht, ob thermalspezifische Trainingsstrategien und Bewertungsmetriken die physikalische Plausibilität und räumliche Qualität verbessern. Das Modell wird anschließend zur Erzeugung und stadtklimatischen Interpretation einer hochaufgelösten LST-Zeitreihe für Berlin eingesetzt. Ausgewählte Klimaanpassungsmaßnahmen können bei geeigneter Datenlage exemplarisch untersucht werden.

### Teilaufgaben

1. **Theoretische Fundierung & State-of-the-Art:** Systematische Literaturrecherche zu Surface Urban Heat Islands, satellitenbasierter Land Surface Temperature, thermaler Fernerkundung, Deep-Learning-basiertem Downscaling und urbanen Kontextfaktoren. Zu untersuchen sind insbesondere die Grenzen des direkten 100-m-auf-10-m-Downscalings, Validierungsprobleme bei fehlenden Referenzdaten und geeignete Metriken für thermische Muster.
2. **Datenbeschaffung, Homogenisierung & Preprocessing:** Aufbau einer reproduzierbaren Geodaten- und Fernerkundungspipeline für Berlin. Landsat-8/9-LST, Sentinel-2-Reflektanzdaten, urbane Morphologie-, Vegetations- und Versiegelungsdaten sowie meteorologische Kontextvariablen sind hinsichtlich Projektion, Auflösung, zeitlicher Passung, Wolkenmaskierung, Co-Registrierung und Qualität zu harmonisieren.
3. **Feature Engineering & urbane Kontextmodellierung:** Ableitung und fachliche Begründung spektraler Indizes, Vegetations-, Wasser- und Versiegelungsmaße, morphologischer Kennwerte, Verschattungs- und Dichteindikatoren, Sonnengeometrie sowie meteorologischer Variablen. Die Prädiktoren sind statistisch zu prüfen und als konsistente Datenbasis für Training, Validierung und Anwendung bereitzustellen.
4. **Modellentwicklung, Baseline-Vergleich & Ablationsstudie:** Implementierung eines Deep-Learning-basierten Downscaling-Modells und einer klassischen Random-Forest-Baseline. Als Backbone dient durchgängig ein fester, deterministischer 2D-U-Net. Über die kumulativen Ablationsstufen variieren ausschließlich die Eingangskanäle und in der letzten Stufe die Verlustfunktion, nicht die Netzarchitektur. Zusätzlich ist zu prüfen, ob thermalspezifische Verlustfunktionen, Aggregationsrestriktionen oder räumliche Regularisierungen die physikalische Plausibilität und Mustertreue verbessern.
5. **Validierung, Unsicherheitsanalyse & räumliche Mustertreue:** Entwicklung eines mehrstufigen Validierungskonzepts aus klassischen Fehlermaßen, Aggregationskonsistenz zur Landsat-Ausgangsauflösung, räumlichen Strukturmetriken und, soweit verfügbar, unabhängigen Referenz- oder Stationsdaten. Unsicherheiten durch fehlende 10-m-Thermalreferenzen, räumliche Autokorrelation, zeitliche Übertragbarkeit und Proxy-Variablen sind kritisch zu diskutieren.
6. **Anwendung Berlin & stadtklimatische Interpretation:** Generierung einer hochaufgelösten LST-Zeitreihe für Berlin und Analyse thermischer Belastungsmuster auf geeigneten räumlichen Bezugsebenen. Die Ergebnisse sind mit stadtklimatisch relevanten Raumstrukturen zu verknüpfen und hinsichtlich Hotspots sowie der Effekte von Vegetation, Wasser und Versiegelung zu interpretieren.
7. **Retrospektive Bewertung klimaadaptiver Maßnahmen:** Exemplarische Untersuchung ausgewählter Maßnahmen wie Dachbegrünung, Stadtbaumpflanzungen oder Entsiegelung. Soweit geeignete Vorher-Nachher-Daten vorliegen, ist ein Difference-in-Differences-Ansatz oder ein vergleichbares quasi-experimentelles Design anzuwenden. Ziel ist eine methodisch transparente Abschätzung thermischer Effekte, keine umfassende kausale Bewertung.
8. **Methodendiskussion, Reproduzierbarkeit & Einordnung:** Kritische Bewertung des Ansatzes hinsichtlich Modellgüte, Skalierbarkeit, Interpretierbarkeit, Datenabhängigkeit und Nutzbarkeit für die stadtklimatische Planung. Herauszuarbeiten sind insbesondere die Grenzen des Deep-Learning-Ansatzes, Risiken einer Überinterpretation synthetischer hochaufgelöster LST-Produkte und Anforderungen an eine belastbare operative Anwendung.

### Abgrenzung

Die Arbeit behandelt ausschließlich satellitenbasierte Oberflächentemperatur und damit Surface Urban Heat Islands. Atmosphärische Urban Heat Islands, human-biometeorologischer Komfort und mikroklimatische Strahlungsmodelle sind nicht Gegenstand der Arbeit. Entwickelt wird keine grundsätzlich neue Deep-Learning-Architektur, sondern ein bestehender Modellansatz wird unter Einbezug urbaner Kontextinformationen systematisch evaluiert. Temporale Sequenzmodelle wie 3D-Faltungen, ConvLSTM oder temporale Attention sind ausgeschlossen. Zeitliche Information geht ausschließlich über Sonnengeometrie und antezedente Witterung in das Modell ein. Ein Multi-Skalen-Vergleich der Einflussfaktoren bleibt dem Ausblick vorbehalten.

### Abgabe

Die Ausarbeitung erfolgt gemäß den wissenschaftlichen Richtlinien der Berliner Hochschule für Technik. Neben der gedruckten und digitalen Masterarbeit sind folgende digitale Produkte strukturiert und dokumentiert abzugeben:

- **Programmcode & Workflows:** kommentierte Skripte, Notebooks oder Pipeline-Beschreibungen zur reproduzierbaren Datenaufbereitung, Modellierung, Validierung und Visualisierung.
- **Modell- und Ergebnisdaten:** verwendete Eingangsdaten beziehungsweise deren reproduzierbare Referenzierung, Feature-Layer, modellierte LST-Raster, Validierungsergebnisse und Metadaten.
- **Modellartefakte:** Modellkonfigurationen, Hyperparameter, trainierte Modellstände oder reproduzierbare Trainingsprotokolle.
- **Kartographische Ergebnisprodukte:** Karten zu LST, Hotspots, Unsicherheiten, Ablationsergebnissen und gegebenenfalls Maßnahmenvergleichen.
- **Methodische Dokumentation:** technische Dokumentation der Datenquellen, Verarbeitungsschritte, Modellvarianten, Validierungsmetriken und bekannten Einschränkungen.

---

Berlin, 24. Juli 2026

Prof. Dr. Ronny Schomacker
