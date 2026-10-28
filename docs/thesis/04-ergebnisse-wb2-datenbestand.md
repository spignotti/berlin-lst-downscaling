# WB2 Ergebnisse: Datenbestand und Pipelines

> Migriert aus Notion am 2026-10-28 (Seite `0515103db5684a4c8a93e5adbffe48df`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

> **Historischer Stand (Juli/August 2026), abgelöst durch features v6 / ARD v10, siehe [Datenbasis & Feature Engineering](03-methodik/datenbasis-feature-engineering.md).**

> 📊 Konsolidierter Abschlussstand von WB2: Primärdaten-ARD, Sekundärdaten, Feature Engineering, QA sowie der publizierte Split-, Eligibility- und Scaler-Vertrag für WB3.

## Ergebnisstand

> Dieser Snapshot (27.08.2026) beschreibt den V3-Datenbestand (28 Kanäle, training/v1). Der gültige Datenstand (features/v6, ARD v10, training/v3) steht in [04 · Ergebnisse & Analysen](04-ergebnisse.md).

**Snapshot:** 27.08.2026

**Datenzeitraum:** 2017 bis 2026

**Training und Evaluation:** 2017 bis 2025

**Inferenz-Reserve:** 2026

**Status:** WB2 vollständig abgeschlossen

## ARD-Pipeline

Die satellitenseitige ARD-Pipeline ist vollständig verarbeitet. Alle Produkte liegen als reproduzierbare COGs mit STAC-Metadaten, Provenance und Abschlussmarkern im GCS-Bucket vor.

| Produkt | Anzahl | Auflösung | Rolle |
| --- | --- | --- | --- |
| Landsat 8/9 C2 L2 | 345 Anker | 100 m | LST-Target und zeitlicher Produktanker |
| Sentinel-2 L2A | 158 eindeutige Szenen | 10 m | Spektrale Guidance |
| ECOSTRESS L2T | 6 Granules | 70 m | Unabhängige thermische Validierung |

**Abschluss:** 509 von 509 Manifest-Einträgen stehen auf `done`.

### Zeitraum und Rollen

- 324 Landsat-Anker aus 2017 bis 2025 bilden den Trainings- und Evaluationsbestand.
- 21 Landsat-Anker aus 2026 sind als reine Inferenz-Reserve vorgesehen.
- Für die spätere 10-m-LST-Zeitreihe bleibt Landsat der thermische Anker. Sentinel-2 und Sekundärdaten liefern die räumliche und physikalische Guidance.

### Kopplung

- 345 Landsat-Sentinel-2-Kopplungen wurden erzeugt.
- Die 158 eindeutigen Sentinel-2-Szenen werden im Mittel 2,18-mal und maximal 5-mal verwendet.
- Die Wiederverwendung ist fachlich beabsichtigt. Bei späteren Datensplits müssen Kopplungen derselben Sentinel-2-Szene gemeinsam gruppiert werden, damit keine identischen S2-Pixel über Training und Evaluation verteilt werden.
- Die Clear-Fraction-Schwelle beträgt 5 Prozent. Der Median der gekoppelten Szenen liegt bei 93,45 Prozent gemeinsamer klarer Fläche.

## Sekundärdaten-Pipeline

### Statische und quasi-statische Quellprodukte

| Quelle | Vintage | Ergebnis |
| --- | --- | --- |
| Versiegelung | 2016 und 2021 | 10-m-Produkte vollständig publiziert |
| Vegetationshöhe | 2020 | 10 m, Mittelwert und Maximum vollständig publiziert |
| DGM | 2021 | 1 m auf 10 m aufbereitet und vollständig publiziert |
| LoD2-Morphologie | 2024 | Höhenstatistiken und BCR vollständig publiziert |

**Status:** 5 von 5 vorgesehenen Quellprodukten vollständig publiziert.

### Abgeleitete Geometrie

Folgende Produktgruppen sind vollständig vorhanden:

- Gebäude-DSM
- Vegetations-DSM
- kombiniertes DSM
- Gebäude-Horizontwürfel mit 36 Azimutbändern
- Vegetations-Horizontwürfel mit 36 Azimutbändern
- Sky View Factor

**Status:** 6 von 6 abgeleiteten Geometrieprodukten vollständig publiziert.

### Dynamische Produkte

Pro Landsat-Anker werden fünf dynamische Modellkanäle in drei COG-Produkten erzeugt:

- ERA5-Land: Lufttemperatur zum Szenenzeitpunkt, Solarstrahlung zum Szenenzeitpunkt und 72-Stunden-Mittel der Solarstrahlung
- Gebäudeschatten
- Vegetationsschatten

Day of Year sowie Sonnenazimut und Sonnenhöhe werden aus dem Aufnahmezeitpunkt abgeleitet und als Metadaten geführt.

**Abschluss:** Die dynamische Pipeline C ist vollständig abgeschlossen. Alle 324 Trainings-Szenen (2017 bis 2025) und alle 21 Inferenz-Szenen (2026) wurden erfolgreich verarbeitet. Insgesamt 1.035 COGs (972 Training + 63 Inferenz) mit STAC, Provenance und Veröffentlichungs-Markern.

**ERA5-Inzident:** Ein CDS-Parameterfehler (area-Reihenfolge N,S,W,E statt N,W,S,E) verursachte OOM-Events durch nahe-globale Downloads. Behoben durch korrekte Parameterreihenfolge, pro-Szene-isolierten Runner und Cache-Bereinigung. Fünf fehlgeschlagene Szenen aus September 2019 wurden bei Wiederholung erfolgreich verarbeitet.

## Zeitliche Semantik der Sekundärdaten

- Wenn mehrere vergleichbare Vintages existieren, wird das neueste nicht-zukünftige Vintage verwendet.
- Wenn nur ein belastbares Vintage existiert, wird es als bestverfügbare statische Approximation für alle Szenen verwendet.
- Die aktuelle Geometrie-Policy nutzt LoD2 2024, DGM 2021 und Vegetationshöhe 2020 retrospektiv. Das ist für die Ablationsfrage vertretbar, bildet aber keine historischen Strukturänderungen ab.
- Für die spätere DiD-Auswertung werden Änderungen der statischen Geometrie deshalb nicht als zeitlich beobachtete Intervention interpretiert.

## Feature Engineering und Trainingsfreigabe

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Abgelöst durch features/v6 und training/v3, siehe [04 · Ergebnisse & Analysen](04-ergebnisse.md).

- Feature-Release V3 umfasst 324 von 324 erfolgreiche 28-Kanal-Stacks; Stage-2-QA `cc00406a` meldet 0 Findings.
- Der temporale Vertrag ist umgesetzt: 2017 bis 2023 Training, 2024 Validation, 2025 Test und 2026 Inferenz.
- Das Release enthält 5.520.165 valide 100-m-Zellbeobachtungen: 4.057.392 Training, 696.942 Validation und 765.831 Test.
- Jede zugelassene 100-m-Zelle besitzt ein gültiges Landsat-Target und vollständigen Support durch 100 von 100 10-m-Featurepixeln. Das entspricht 552.016.500 gültigen 10-m-Featurepixelbeobachtungen über alle Splits.
- 30 Szenen sind mit `no_eligible_cells` ausgeschlossen; die 21 Szenen aus 2026 bleiben als `inference_deferred` außerhalb von Training, Modellselektion und Skalierung.
- Der Scaler basiert ausschließlich auf 4.057.392 Trainingszellen beziehungsweise 405.739.200 Werten je Kanal. Kontinuierliche Kanäle verwenden Z-Score, Niederschlag `log1p` plus Z-Score, Schatten bleibt binär.
- Der unabhängige Audit bestätigt: kein Split-Leakage über `s2_scene_id`, korrekte strikte Eligibility, deterministische Zell-IDs und Mittelpunktkoordinaten sowie erfolgreichen Readback aller Artefakte.
- Fachlicher Vertrag: WB2c-4 Splits, Normalisierung und Skalierung (Notion-Archiv). Technischer Abschlussbeleg: WB2c-4 Release Evidence (2026-08-27) (Notion-Archiv).

## Reproduzierbarkeit und Qualität

- Einheitliches CRS EPSG:25833 und verschachtelte 10-m-, 70-m- und 100-m-Gitter.
- COG-Validierung auf CRS, Shape, Origin und NoData.
- Manifest-, Config- und Ledger-basierter Ablauf mit atomaren Writes und idempotentem Resume.
- DWD dient ausschließlich als externe Plausibilisierung der ERA5-Land-Temperatur, nicht als Modellkanal. Validierung abgeschlossen (23.07.): 1.508 Matches über 345 Anker, Bias −0,03 °C, MAE 0,77 °C, RMSE 0,98 °C. ERA5-Land-Temperaturen sind plausibel.

> ✅ **WB2 vollständig abgeschlossen.** Primär- und Sekundärdaten, Feature-Release V3, QA sowie Split-, Eligibility- und Scaler-Artefakte sind reproduzierbar publiziert und für WB3 freigegeben. Patch-Geometrie, Sampler und Trainings-I/O gehören ab jetzt zu WB3.

## Quellen

- Finaler WB2-Datenbericht — Pipeline-Abschluss 22.07.2026 (Notion-Archiv)
- [WB2 ARD-Pipeline](03-methodik/wb2-ard-pipeline.md)
- [WB2 Sekundärdaten](03-methodik/wb2-sekundaerdaten.md)
- [WB2 Feature Engineering & Trainingsdatenaufbereitung](03-methodik/wb2-feature-engineering.md)
- WB2c-4 As-built: Feature-Stack-, Masken-, Metadaten- und QA-Vertrag (V3) (Notion-Archiv)
- WB2c-4 Release Evidence (2026-08-27) (Notion-Archiv)

## Nachtrag 08.10.2026 – wissenschaftlich relevante Fortschreibung des WB2-Snapshots

Der obige Abschlussbericht ist ein **Snapshot vom 27.08.2026**. Inzwischen enthält der normative [Repository-Datenvertrag](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md) wichtige zusätzliche bzw. präzisierte Aussagen. **Bei Abweichungen gilt dieser Nachtrag**; der ältere Text bleibt als historischer Bericht erhalten.

### Zeitliche Zuordnung der Stadtmorphologie

Die frühere Aussage „LoD2 2024 retrospektiv für alle Jahre“ ist **nicht mehr der vollständige Stand**. Es wurden historische Bestände hinzugefügt und ein Carry-forward-Mapping der LoD-Morphologie dokumentiert:

| Szenenjahre | zugeordnetes Gebäude-Morphologie-Vintage |
| --- | --- |
| 2017–2020 | 2017 (aus LoD1-Footprints 2017, mit LoD2-Dachgeometrie 2021 gefiltert) |
| 2021 | 2021 |
| 2022–2023 | 2022 |
| 2024–2026 | 2024 |

- Die 2017er Rekonstruktion verwendet einen Gebäude-Bestandsfilter auf 2017-Footprints, aber Detailgeometrie aus 2021. Somit **kein vollständig zeitgenössisches 2017er LoD2-Messprodukt**.
- DGM 2021 und Vegetationshöhe 2020 bleiben über die Jahre als Approximation; die Versiegelung nutzt nach Szenenjahr 2016 bzw. 2021.
- **Interpretationsgrenze:** Es ist zwar zeitlich differenzierteres Gebäudeinventar integriert, aber keine vollständige jährliche Erfassung der baulichen oder vegetativen Veränderung. Eine Maßnahmenkausalität kann daraus nicht abgeleitet werden.
- Technische Vintage-IDs, Veröffentlichungsmarker und konkrete Pfadregeln bleiben ausschließlich im [GitHub-Contract](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md).

### V3-Semantik der Modellprädiktoren

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Der aktuelle 22-Kanal-Stack (features/v6) ist in [Datenbasis & Feature Engineering](03-methodik/datenbasis-feature-engineering.md) beschrieben.

Der aktive V3-Stack enthält 28 Kanäle: sechs Sentinel-2-Reflektanzbänder, vier spektrale Indizes/Proxies, acht Morphologiekanäle, **acht ERA5-Land-Kanäle** und zwei binäre Schattenkanäle. Die ältere Beschreibung von nur drei meteorologischen Modellkanälen stammt aus einem früheren Dynamik-Snapshot. Die Sonnengeometrie und Day-of-Year werden als Metadaten geführt und steuern gegebenenfalls abgeleitete Schatten, bilden aber **keine zusätzlichen eigenständigen V3-Kanäle**.

- Die „Albedo“ ist ein lineares sechsbandiges kurzwelliges Reflektanz-Proxy, **keine** physikalisch BRDF-korrigierte Albedo.
- Gebäude-DSM, Vegetations-DSM und kombinierter DSM dienen als interne Ableitungsprodukte für SVF/Horizonte/Schatten, **nicht** als unmittelbare Modellkanäle.
- V3 korrigiert eine wissenschaftlich bedeutsame NoData-Semantik: **Quellabdeckung ohne Gebäude bedeutet Gebäudehöhe/BCR = 0; fehlende Quellabdeckung bedeutet NaN.** Nicht alle anderen verfügbaren Kanäle werden gelöscht, wenn ein einzelner Kanal fehlt. Das maskenbasierte vollständige Feature-Vektor-Kriterium bleibt hiervon getrennt. Die V2-Regel verwarf massiv zu viele gültige Featurepixel; das ist ein **Versions-/Messmodell-Unterschied**, kein natürliches Berliner Verteilungsmuster. [Historische V3-Korrekturevidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/phase-2-preparation.md); aktueller Vertrag: [data-sources-and-contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md).

### Qualitäts- und Evidenzgrenzen

- Der DWD-Abgleich mit Bias −0,03 °C, MAE 0,77 °C und RMSE 0,98 °C bezieht sich laut [historischer Phase-1-Dokumentation](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/phase-1-delivery.md) auf die **vor dem Neuaufbau verwendeten skalaren ERA5-Produkte**. Er ist **kein unabhängiger Verifikationsnachweis der aktuellen acht räumlichen ERA5-Kanäle**; die frühere pauschale Plausibilisierung darf daher nicht so zitiert werden.
- „Reproduzierbar publiziert“ bedeutet valide, deterministisch verknüpfte Datensätze und prüfbare Provenance/QA, nicht identische Bytes bei vollständigem Neuaufbau aus leerem Bucket. Live-Quellabfragen und Laufzeitmetadaten verhindern einen pauschalen Byte-Identitätsanspruch.
- Aktuelles kanonisches GCS-Root nach dem Kontenwechsel im September: `gs://berlin-lst-training-data/`. Alte Cloud-Pfade in früheren Notes sind historische Belege. [Transferabschluss](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/gcs-inventory-and-transfer.md).
- 2026 bleibt Inferenz-Reserve: 21 Anker sind in Manifest/ARD erfasst, wurden für das Trainingsrelease jedoch **nicht** als V3-Trainingsfeatures materialisiert. Daraus folgt noch **keine** bereits vorliegende 2026er 10-m-LST-Zeitreihe.

### Verbleib der Originale

Mess- und Datenfreigabeprotokolle in GitHub/GCS bleiben unverändert; hier wurden nur für Methodik und Ausarbeitung notwendige Aussagen fortgeschrieben. Quantitative Modellbefunde der Baselines und Ablation: [WB3 · Ergebnistabellen – Baselines und Ablation](04-ergebnisse-wb3-ergebnistabellen.md). Die wissenschaftliche Interpretation ist getrennt dokumentiert in [WB3 · Diskussion der Ablationsbefunde](05-diskussion-wb3-ablationsbefunde.md).

### Release-Prüfung: Stichprobe und Sentinel-2-Kopplung (27.08.2026)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Die Prüfung bezieht sich auf das Release training/v1; für training/v3 siehe [04 · Ergebnisse & Analysen](04-ergebnisse.md).

Das unabhängige WB2c-4-Acceptance-Audit bestätigte **345 Manifestzeilen**, davon 21 reine Inferenz-Reserve, 324 Kandidaten 2017–2025 und **30 Szenen ohne eine einzige eligible Zelle**. **294 Szenen** wurden mit gültigen Trainings-, Validierungs- oder Testzellen publiziert. Unter den drei Nicht-Inferenz-Splits liegen **149 eindeutige Sentinel-2-Szenen-IDs** vor; es wurde **keine** Überschneidung einer `s2_scene_id` über diese Splits festgestellt. Auch Eligibility, Cell-ID-Determinismus, Train-Only-Scaler und Artefaktlesbarkeit wurden als PASS protokolliert. Diese technische Leakage-Prüfung ersetzt keinen räumlichen Holdout.

[Original-Acceptance-Audit](https://app.notion.com/p/3c9356451f668182ac88cfd2350b9007) (historische Pfadangaben; kanonischer GCS-Bucket seit September: `gs://berlin-lst-training-data/`).
