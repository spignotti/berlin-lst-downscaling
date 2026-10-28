# WB2 ARD-Pipeline

> Migriert aus Notion am 2026-10-28 (Seite `101d47201582443da9e2157ecf696194`, zuletzt bearbeitet 2026-07-15). Text unverändert übernommen.

> 🛰️ Stand 2026-07-08. Lebendes Ergebnis-Dokument für WB2. Datenakquise über Planetary Computer STAC + odc-stac (Landsat/Sentinel-2) und earthaccess/CMR (ECOSTRESS). Google Earth Engine ist abgelöst. Zeitliche Kopplung Landsat zu Sentinel-2 steht. Keine Code-Details, die liegen im Repo.

## Datenakquise

- Landsat/Sentinel-2: Planetary Computer STAC via odc-stac (stac\_load mit URL-Re-Signing, Retry mit exponentiellem Backoff).
- ECOSTRESS: earthaccess/CMR, L2T-COGs direkt gelesen.
- Kanonisches Raster: 10 m / 70 m / 100 m nested, EPSG:25833, gemeinsamer Origin 369190/5838410.

## Szenen-Selektion & Kopplung (abgeschlossen)

- Landsat-Anker 2017 bis 2025, Mai bis September: 873 STAC-Treffer, 519 nach Pixel-QA-Filter (clear\_frac >= 0.05).
- Sentinel-2-Kopplung: 474 gekoppelt (91 % Rate), 45 Anker verwaist (kein S2-Kandidat >= 0.05 clear\_frac).
- Mittlerer clear\_frac 0.740 (Range 0.054 bis 1.000).
- ECOSTRESS-Matches: 12 Granules, alle aus 2018 (CMR, Fenster von plus/minus 2 h).
- Manifest: data/ard/manifest.parquet, 1005 Zeilen über alle drei Quellen.
- s2cloudless nicht integriert: SCL-Maskierung reicht für Berliner Sommerszenen (Mai bis September), keine zusätzliche Abhängigkeit.

> ⚠️ ECOSTRESS liegt nur für 2018 vor (12 Granules). Die Multi-Source-Validierung (WB5) braucht dafür später eine andere Strategie. Die Sentinel-2-zu-Landsat-Kopplung selbst funktioniert über den vollen Zeitraum.

## ARD-Datenvolumen

| Quelle | Szenen | COG pro Szene | Gesamt |
| --- | --- | --- | --- |
| Landsat (100 m, 1 Band) | 474 | \~595 KB | \~285 MB |
| Sentinel-2 (10 m, 4 Bänder) | 474 | \~215 MB | \~102 GB |
| ECOSTRESS (70 m, 1 Band) | 12 | \~984 KB | \~12 MB |
| Total | 960 |  | \~102 GB |

- Sentinel-2 dominiert mit 99,7 % des Volumens (4 Bänder, 10 m, float32).
- Laufzeit sequenziell \~4,5 h (Landsat \~20 min, Sentinel-2 \~4,2 h, ECOSTRESS \~6 min), gemessen aus dem Smoke-Test.
- GCS-Speicherkosten: Standard \~2 \$/Monat, Archive \~0,41 \$/Monat nach 90 Tagen. Mit den 600 Euro GCP-Credits vernachlässigbar.
- Kompression: Deflate + Predictor 2, 512x512-Kacheln, Overviews 2/4/8/16 (COG-Standard, keine Optimierung nötig).

## Pipeline-Status

- Modus: Manifest-driven (build\_[manifest.py](http://manifest.py), dann manifest.parquet, dann run\_[ard.py](http://ard.py)).
- Schema v5 (nach dem Grid-Change).
- COG-Validierung nach jedem Schreibvorgang: CRS, Bandzahl, Grid-Alignment, Nicht-all-NaN-Prüfung.
- Idempotenz und Resume: jede Szene im Ledger getrackt (exporting, dann done oder failed), Prozess jederzeit abbrech- und neustartbar, bereits fertige Szenen mit Schema v5 werden übersprungen.
- Lokaler Smoke-Test erfolgreich (3 von 3 Szenen, cog\_validated, total\_failed 0).

## Nächste Schritte

- ARD-Voll-Run (474 gekoppelte Paare, \~102 GB, \~4,5 h) in den nächsten Tagen. Volumen unkritisch, Run kann über Nacht laufen.
- Danach Sekundärdaten-Pipeline (Stage 2 bis 4: Morphologie, SVF, Verschattung, Meteorologie).
- Anschließend DL-Training mit Ablation.

## Landsat 8/9 Grundlagen

- **Rolle:** Landsat 8/9 liefern ausschließlich das LST-Label für Training und Inferenz. Landsat-Surface-Reflectance wird nicht als Prädiktor verwendet.
- **Produkt:** Collection 2 Level-2 Surface Temperature, primär Band 10, in Kelvin. Atmosphärenkorrektur und ASTER-GED-Emissivität sind bereits Bestandteil des Produkts; kein eigenes LST-Retrieval.
- **Räumliche Information:** TIRS besitzt etwa 100 m native Auflösung. Eine 30-m-Auslieferung enthält keine zusätzliche thermale Detailinformation; das Projekt hält das Target deshalb auf dem kanonischen 100-m-Grid.
- **Zeitliche Abdeckung:** Landsat 8 seit 2013, Landsat 9 seit 2021. Kombiniert ergibt sich seit 2022 ein nomineller 8-Tage-Revisit. Projektscope bleibt Mai bis September 2017 bis 2025.
- **Überflug:** Vormittagsüberflug über Berlin um etwa 10:12 Uhr. Die relativ konstante Aufnahmezeit verbessert die Vergleichbarkeit, bildet aber nicht das tägliche LST-Maximum ab.
- **Qualität:** Wolken, Wolkenschatten, Cirrus und Sättigung werden pixelweise über die Collection-2-QA-Bänder behandelt. Bewölkte Pixel werden maskiert, nicht rekonstruiert.
- **Aktueller Bestand:** 474 gekoppelte Landsat-Sentinel-2-Paare nach Szenenselektion. Der ARD-Voll-Run schreibt die Landsat-Targets auf das kanonische 100-m-Grid.

**Primärquellen:** [NASA Landsat 8](https://science.nasa.gov/mission/landsat-8/), [NASA Landsat 9](https://science.nasa.gov/mission/landsat-9/), [USGS Collection 2 Surface Temperature](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-temperature).

## Legacy: archivierte OpenCode-Notes

Frühere OpenCode-Notes zum Stand der Datenpipeline. Nach der Konsolidierung in dieses Dokument archiviert, weiterhin über die Notes-DB mit Archiv-Filter auffindbar.

- Plan A+B abgeschlossen — Manifest + Canonical Grid + Validation (Notion-Archiv) (Primärquelle für Selektion und Kopplung oben)
- ARD Datenvolumen — 960 Szenen, ~102 GB, ~4.5h Laufzeit (Notion-Archiv) (Primärquelle für Datenvolumen oben)
- Architektur-Dokumentation & Cloud-Readiness-Review (berlin-lst-downscaling) (Notion-Archiv) (Schema v4, lokaler Stand, Code-Details gehören ins Repo/[PROJECT.md](http://PROJECT.md))
- Volumen-Scan Szenen-Selektion 2017-2025 (Juli 2026) (Notion-Archiv) (Schätzung, durch gemessenes ARD-Datenvolumen ersetzt)
- ARD Pipeline Audit — Cloud-Masking Feinjustierung + Pipeline-Verifikation (2026-06-28) (Notion-Archiv) (GEE-Ära, Bug-Funde in gee\_[scenes.py](http://scenes.py), obsolet)
- ARD Pipeline Audit + ECOSTRESS Reprojection Fix (2026-06-28) (Notion-Archiv) (GEE-Ära, vor dem Pivot)
- Feasibility Check: Secondary Data Sources (Notion-Archiv) (Sekundärdaten-Machbarkeit, Inhalt im Project Memory, wieder relevant erst für Stage 2 bis 4)
- Coupling Analysis: Landsat ↔ Sentinel-2 (Δt Window) (Notion-Archiv) (Δt-Fenster plus/minus 3 bestätigt, in Kopplungsergebnis aufgegangen)
- ECOSTRESS Data Availability — Updated Findings (2026-06-24) (Notion-Archiv) (durch tatsächliches Kopplungsergebnis ersetzt: 12 Granules 2018)
- Forschung: Datenquellen-Recherche verdichtet (2026-06-23) (Notion-Archiv) (in Research Results eingeflossen, überholt)
- Research Results — Data Availability & Additional Data Sources (Notion-Archiv) (GEE-Ära, Zeitfenster und Median-Empfehlung überholt)
