# Modell & Ablationsprotokoll

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66816db85df2e3ae6ddd09`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

**Umgesetzter Methodenstand, Stand 28.10.2026.**

## Modell und Trainingsziel

Fester 2D-U-Net-Backbone mit thermischem Prior und nullinitialisiertem residualem Korrektur-Head. Output ist eine 10-m-LST-Vorhersage. Trainiert wird auf Pseudo-Pairs: Landsat wird als grober 1.000-m-Prior degradiert, das 100-m-Landsat-Target ist die Trainingsaufsicht. Aktuelle Datengrundlage ist `training/v3` (`features/v6`, 22 Kanäle); das Target ist das Flächenmittel der 30-m-`lwir11`-Werte. Das Residual wird in Kelvin gelernt. Die historischen Stage-1–5-Läufe nutzten `training/v1` mit Nearest-Neighbour-Resampling. Der Vorhersagefehler wird nach exakter Aggregation der 10-m-Prediction in 10×10-Blöcken auf 100-m-Zellen ermittelt.

## Studien-Design

Aktuelles Design auf `training/v3`:

- Naive Baseline: physikalischen Grobprior blockweise expandieren (identische Kohorte/Maske).
- Fester 2D-U-Net-Backbone mit Kelvin-Residual (Prior als separater Input, nullinitialisierter Korrektur-Head).
- Drei Eingangsvarianten: (1) volle Optik (alle 22 Kanäle), (2) ohne NDVI, (3) ohne Rot, NIR und Indizes.
- Optionaler TESSERA-Arm: nur als Code vorhanden, nicht gelaufen; ersetzt die 6 Morphologie- und 2 Schattenkanäle durch jährliche TESSERA-Embeddings. Die handgebauten Features bleiben das Hauptmodell.
- v6-Ablationsmatrix eingefroren (Entscheidung 31.10.2026): 9 Arme × Seeds 0, 1, 2 = 27 Läufe, je 40 Epochen, Cosine-Lernratenplan mit 2 Warmup-Epochen (T_max = 40, eta_min = 0), ohne Early Stopping, mit vorgebautem Patch-Cache. Offen sind nur noch die Loss-Varianten und die Neuauflage der Random-Forest-Baseline auf `training/v3`.
- Modellauswahl und Testauswertung (Entscheidung 31.10.2026): Die Modellauswahl nutzt ausschließlich Validation 2024. Die finale Testauswertung 2025 wird pro Lauf genau einmal auf dem besten Validation-2024-Checkpoint gerechnet (`validation/mae_100m` auf dem 100-m-Raster), nicht auf der letzten Epoche.
- Begründung Cosine-Lernratenplan: Ein Bestätigungslauf mit **einem Seed** (Seed 0, 40 Epochen, nur Validation, Test unberührt) erreichte eine beste Validation-MAE von 0,3801 K in Epoche 39 von 40, gegenüber 0,3846 K in Epoche 59 von 60 beim Pilotlauf mit konstanter Lernrate (dort 0,391 K nach 40 Epochen). Ab Epoche 30 bleibt die Cosine-Kurve im Bereich 0,380–0,396 K, die des Pilotlaufs im Bereich 0,391–0,461 K.
- Ergebnisse auf `training/v3`: **ausstehend**.

### Historisches Studiendesign (training/v1)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Abgelöst durch das Studiendesign auf `training/v3` (siehe oben).

- Naive-Baseline: physikalischen Grobprior blockweise expandieren (identische Kohorte/Maske).
- Stage 1: 10 spektrale/Indexkanäle, masked L1.
- Stage 2: + 8 Morphologiekanäle (18), masked L1.
- Stage 3: + Gebäude- und Vegetationsschatten (20), masked L1.
- Stage 4: + 8 ERA5-Land-Kanäle (28), masked L1.
- Stage 5: 28 Kanäle wie Stage 4; masked L1 plus SSIM-/Gradienten-Strafanteile.
- Zusätzliche isolierte Kontrollläufe: Spektral + Schatten und Spektral + ERA5. Sie erlauben begrenzte Aussagen über Redundanz, **kein** vollständiges faktorielles Wirkungsdesign.
- Backbone, Residual-Prior, Datensplit, Patchgeometrie und Selektion bleiben bei den dokumentierten Vollruns eingefroren. **Ein Seed; keine Unsicherheitsintervalle aus wiederholten Initialisierungen.**

## Methodische Aussagegrenzen und Nachweise

[Tensor- und Pseudo-Pair-Contract](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/pseudo-pair-tensor-contract.md) · [Ablationskonfigurationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-stage-configs.md) · [Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md).

Die Fixierung der Modell- und Loss-Konfigurationen ist in GitHub nachgewiesen. Die Random-Forest-Baseline wurde auf `training/v1` abgeschlossen (09.10.2026; historisch, siehe [Ergebnistabellen WB3](../04-ergebnisse-wb3-ergebnistabellen.md)); auf `training/v3` ist sie **ausstehend**.

Weiterführend: [Features v6](../../features-v6.md) · [TESSERA-Arm](../../tessera-arm.md).

- [Pseudo-Pair- und Tensorvertrag (WB3)](pseudo-pair-tensorvertrag.md)
