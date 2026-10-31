# 03 · Methodik – aktueller Stand

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668138a32ae0c279cc18f4`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

## Datenaufbau und Trainingskohorte

- **Zeitraum:** Training 2017–2023, Validation 2024, Test 2025; 2026 nur Inferenz.
- **Thermisches Target:** Landsat 8/9 LST auf dem kanonischen 100-m-Raster, gebildet als Flächenmittel der 30-m-`lwir11`-Werte; QA bitweise auf dem nativen 30-m-Raster ausgewertet.
- **Eingangsgrößen:** `features/v6` (Trainingsfreigabe `training/v3`) mit 22 möglichen 10-m-Kanälen: Sentinel-2-Spektralkanäle (6), Indizes (4), urbane Morphologie (6; Sichtfaktor SVF mit Radius 100 m), ERA5-Land-Meteorologie (4, als Szenenwert) und Gebäude-/Vegetationsschatten (2). Grundlage ist ARD v10 mit 286 von 290 Paarungen; zwei Sentinel-2-Szenen sind durch das Coverage-Gate ausgeschlossen. Der Patch-Index v3 umfasst 8.078 Patches (Train 5.523, Validation 1.047, Test 1.508; [QA](../../results/patch-index-v3/patch_index_qa.json)). `features/v5` (28 Kanäle), `training/v1` und `training/v2` sind nur noch historisch.
- **Datenfreigabe:** Strikte Trainingseignung für vollständige 10-m-Featurevektoren in allen 100 Unterzellen eines gültigen 100-m-Thermalziels; Scaler ausschließlich auf Trainingdaten angepasst.

## Modell und Ablation

Fester 2D-U-Net-Backbone mit physikalischem LST-Prior als separatem Input und nullinitialisiertem Korrektur-Head (Kelvin-Residual). Training anhand synthetischer Pseudo-Pairs (1.000-m-Prior, 100-m-Supervision); Ausgabeauflösung 10 m, Trainingsloss und primäre Validierungsmetrik nach Rückaggregation auf 100 m.

Aktuelles Ablationsdesign auf `training/v3` (`features/v6`, 22 Kanäle):

- Eingangsvarianten: (1) volle Optik (alle 22 Kanäle), (2) ohne NDVI, (3) ohne Rot, NIR und Indizes.
- Optionaler TESSERA-Arm: nur als Code vorhanden, nicht gelaufen; die handgebauten Features bleiben das Hauptmodell.
- v6-Ablationsmatrix eingefroren (9 Arme × Seeds 0, 1, 2 = 27 Läufe, 40 Epochen, Cosine-Lernrate, ohne Early Stopping; Test 2025 einmal auf dem besten Validation-2024-Checkpoint); offen nur Loss-Varianten und Neuauflage der Random-Forest-Baseline auf `training/v3` ([Protokoll](modell-ablationsprotokoll.md)).
- Ergebnisse auf `training/v3`: ausstehend.

### Historische Ablationsleiter (training/v1)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Abgelöst durch das Ablationsdesign auf `training/v3` (siehe oben und [Modell & Ablationsprotokoll](modell-ablationsprotokoll.md)).

| Stufe | Aktive Modellkanäle | Variation |
| --- | --- | --- |
| 1 | 10 | Spektralblock, masked L1 |
| 2 | 18 | • Morphologie |
| 3 | 20 | • Gebäude-/Vegetationsschatten |
| 4 | 28 | • ERA5-Land |
| 5 | 28 | Gleiche Inputs wie 4; masked L1 plus SSIM-/Gradientenstrafen |

Die historischen Vollruns nutzten die fixierte Residualrepräsentation, dieselbe Patchgeometrie, den temporalen Split und Seed 0.

## Validierung und Berlin-Anwendung

**Bereits umgesetzt (historisch, `training/v1`):** Full-Split-Vergleich mit naiver Prior-Expansion, 2024-Validation-basierte Checkpointwahl und 2025-Testauswertung der damaligen Ablationsruns; die Random-Forest-Baseline war auf `training/v1` abgeschlossen. Auf `training/v3` ist die Evaluation (inklusive Random-Forest- und naiver Baseline) ausstehend.

**Noch nicht abgeschlossen:** unabhängige thermale Vergleichsdaten, räumlicher Holdout, strukturbezogene 10-m-Plausibilitätsprüfung und die vollständige 10-m-LST-Zeitreihe für Berlin. Zusätzliche Untersuchungen sind für die Einordnung der räumlichen Aussagekraft erforderlich.

**Technische Evidenz:** [Daten- und Release-Contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md) · [Modell- und Ablations-Contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-stage-configs.md) · [Features v6](../../features-v6.md) · [Republish v6](../../republish-v6.md) · [W&B-Projekt](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling).

## Methodische Teilbereiche

- [Datenbasis & Feature Engineering](datenbasis-feature-engineering.md)

- [Modell & Ablationsprotokoll](modell-ablationsprotokoll.md)

- [Evaluation & Unsicherheitsanalyse](evaluation-unsicherheit.md)

- [Berlin-Anwendung & geplante Auswertungen](berlin-anwendung.md)
