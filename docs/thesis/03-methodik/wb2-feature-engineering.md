# WB2 Feature Engineering & Trainingsdatenaufbereitung

> Migriert aus Notion am 2026-10-28 (Seite `44fa223915bf49a39b303ba79a340254`, zuletzt bearbeitet 2026-08-27). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

> **Historischer Stand (Juli/August 2026), abgelöst durch features v6 / ARD v10, siehe [Datenbasis & Feature Engineering](datenbasis-feature-engineering.md).**

> 🧩 **WB2 Feature Engineering und Trainingsdatenaufbereitung abgeschlossen:** Feature-Release V3, QA, temporale Splits, strikte 100-m-Eligibility, stabile Zellidentitäten und train-only Scaler sind reproduzierbar publiziert und für WB3 freigegeben.

## Status

- Feature-Ableitung und szenenbezogene Stacks: abgeschlossen.
- Morphologie-Korrektur und V3-Release: abgeschlossen.
- QA Stufe 2: bestanden.
- WB2c-4 Planung und technische Artefakte: abgeschlossen.
- Übergabe an WB3: freigegeben am 27.08.2026.

## Rolle in WB2

1. **WB2 Grundlagen & Festlegungen:** datensatzübergreifender Scope, Raster, Zeitraum und methodische Grundentscheidungen.
2. **WB2 ARD-Pipeline:** szenenweise Primärdaten, Masken und zeitliche Kopplung.
3. **WB2 Sekundärdaten:** statische, quasi-statische und dynamische Zusatzdaten.
4. **WB2 Feature Engineering & Trainingsdatenaufbereitung:** abgeleitete Features, szenenbezogene Zusammenführung, QA und leakage-freier Übergabevertrag an WB3.

## Erfolgreich abgeschlossen

### Feature-Release V3

- 324 von 324 Feature-Stacks erfolgreich, 0 fehlgeschlagen.
- 28 Kanäle in fester Reihenfolge.
- Stage-2-QA-Run `cc00406a` mit `0 findings` und `ok: true`.
- 21 Landsat-Anker aus 2026 als reine Inferenz-Reserve.
- V1 und V2 retiriert.
- Verbindlicher Bericht: WB2c-4 As-built: Feature-Stack-, Masken-, Metadaten- und QA-Vertrag (V3) (Notion-Archiv).

### Fachlicher Kanalvertrag

- Spektral: B02, B03, B04, B08, B11 und B12.
- Indizes und Albedo: NDVI, NDWI nach McFeeters, NDBI und S2-Broadband-Albedo.
- Morphologie: Gebäudehöhe Mittelwert, Standardabweichung und Maximum, Gebäudeabdeckungsgrad, Vegetationshöhe Mittelwert und Maximum, Versiegelung und SVF.
- Meteorologie: Szenen-T2m, Szenen-SSRD, 72-h-SSRD-Mittel, VPD, 10-m-Windgeschwindigkeit und Niederschlag für 0 bis 24, 24 bis 48 und 48 bis 72 Stunden.
- Verschattung: Gebäude- und Vegetationsschatten getrennt.

### Korrigierte Semantik

- `0` bedeutet bekannte Fläche ohne Gebäude beziehungsweise ohne Vegetationsobjekt.
- `NaN` bedeutet fehlende Datenabdeckung.
- Verfügbarkeit wird kanalweise behandelt; die fehlerhafte Alles-oder-Nichts-Semantik von V2 ist entfernt.
- `feature_valid` beschreibt die Verfügbarkeit des vollständigen Inputvektors, enthält aber nicht die Landsat-Target-Gültigkeit.

## WB2c-4 Ergebnis

Der fachliche Vertrag steht in WB2c-4 Splits, Normalisierung und Skalierung (Notion-Archiv); der technische Abschlussbeleg in WB2c-4 Release Evidence (2026-08-27) (Notion-Archiv).

### Publizierter Release

- Release: `gs://berlin-lst-data/training/v1`, basierend ausschließlich auf Feature-Release V3 mit Config-Hash `d9eb25995b2f4911`.
- Manifest: 345 Szenen, davon 240 Training, 41 Validation, 43 Test und 21 Inferenz.
- Status: 294 publizierte Szenen; 30 Szenen mit `no_eligible_cells` ausgeschlossen; 21 Inferenzszenen als `inference_deferred` geführt.
- Split: 2017 bis 2023 Training, 2024 Validation, 2025 Test und 2026 Inferenz.
- Kein Wert von `s2_scene_id` berührt mehrere nicht-inferenzielle Splits; 149 eindeutige Sentinel-2-Szenen wurden leakage-frei gruppiert.

### Valide Zellen und Pixel

- Gesamt: 5.520.165 valide 100-m-Zellbeobachtungen.
- Training: 4.057.392 Zellen.
- Validation: 696.942 Zellen.
- Test: 765.831 Zellen.
- Jede Zelle erfüllt exakt: gültiges Landsat-Target und 100 von 100 gültige 10-m-Featurepixel.
- Daraus folgen 552.016.500 gültige 10-m-Featurepixelbeobachtungen insgesamt und 405.739.200 im Trainingssplit, jeweils pro Kanal.
- Die Werte sind Beobachtungen über Szenen, keine eindeutigen räumlichen Orte. Dieselbe Rasterzelle kann zu mehreren Aufnahmezeitpunkten vorkommen.

### Scaler und Identitäten

- Scaler ausschließlich aus dem Trainingssplit und den trainingsfähigen Zellen.
- 23 kontinuierliche Kanäle: Z-Score; drei Niederschlagskanäle: `log1p` plus Z-Score; zwei Schattenkanäle: unverändert 0/1.
- `cell_id`, Rasterzeile, Rasterspalte und Mittelpunktkoordinaten sind deterministisch aus dem kanonischen EPSG:25833-Grid reproduzierbar.
- Der unabhängige Audit meldet 0 Zell-ID- und 0 Koordinatenabweichungen sowie erfolgreichen Readback aller publizierten Artefakte.

### Publizierte Artefakte

- Szenenmanifest: `gs://berlin-lst-data/training/v1/manifest.parquet`
- Zellindex: `gs://berlin-lst-data/training/v1/cells.parquet`
- Scaler: `gs://berlin-lst-data/training/v1/scaler.json`
- QA-Report: `gs://berlin-lst-data/training/v1/qa/training/93315218/report.json`
- Release-Marker: `gs://berlin-lst-data/training/v1/complete.json`
- V3-COGs bleiben unveränderte Source of Truth.

## Bewusst nach WB3 und WB5 vertagt

- Pseudo-Pair- und Patch-Geometrie.
- Patchgröße und Mindestanteil gültiger Zellen pro Patch.
- Regel `no_valid_patch` nach Festlegung der Patches.
- Sampler, Masked Loss und Batch-Aufbau.
- COG-gegen-Zarr-Benchmark und endgültige Trainingsschnittstelle.
- Spatial Holdout, Puffer und kombinierter räumlich-zeitlicher Stresstest.
- Echte rollierende Temporal Cross-Validation.

## Nächste Schritte

1. WB3a Training Unit und Patch-Vertrag planen (Notion-Archiv)
2. WB3a Trainings-I/O benchmarken: COG vs. Zarr (Notion-Archiv)
3. WB5 Spatial Holdout und Robustheitsläufe planen (Notion-Archiv)

## Übergabevertrag an WB3

Der WB2-Bestand bleibt rasterbasiert. Raster- und Tensorwerte liegen in COG und später optional Zarr, Metadaten und Indizes in Parquet. Trainingspatches werden nicht vorab materialisiert; der spätere PyTorch-/TorchGeo-DataLoader liest räumliche Fenster on the fly.

## Referenzen

- Notion-Seite dcd52e71 (Notion-Archiv)
- [WB2 ARD-Pipeline](wb2-ard-pipeline.md)
- [WB2 Sekundärdaten](wb2-sekundaerdaten.md)
- WB2c-4 Splits, Normalisierung und Skalierung (Notion-Archiv)
- WB2c-4 As-built: Feature-Stack-, Masken-, Metadaten- und QA-Vertrag (V3) (Notion-Archiv)
- WB2c-4 Release Evidence (2026-08-27) (Notion-Archiv)
- Notion-Seite d4879e15 (Notion-Archiv)
- Notion-Seite d70d4ed1 (Notion-Archiv)
