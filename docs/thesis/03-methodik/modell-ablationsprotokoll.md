# Modell & Ablationsprotokoll

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66816db85df2e3ae6ddd09`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.

**Umgesetzter Methodenstand, Stand 08.10.2026.**

## Modell und Trainingsziel

Fester 2D-U-Net-Backbone mit thermischem Prior und nullinitialisiertem residualem Korrektur-Head. Output ist eine 10-m-LST-Vorhersage. Trainiert wird auf Pseudo-Pairs: Landsat wird als grober 1.000-m-Prior degradiert, das 100-m-Landsat-Target ist die Trainingsaufsicht. In `training/v2` ist dieses Target das Flächenmittel der 30-m-`lwir11`-Werte; die dokumentierten Stage-1–5-Läufe nutzen `training/v1` mit Nearest-Neighbour-Resampling. Der Vorhersagefehler wird nach exakter Aggregation der 10-m-Prediction in 10×10-Blöcken auf 100-m-Zellen ermittelt.

## Studien-Design

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

Die Fixierung der Modell- und Loss-Konfigurationen ist in GitHub nachgewiesen. **Die Random-Forest-Baseline (#59) ist noch nicht als abgeschlossener Vergleich dokumentiert.**

- [Pseudo-Pair- und Tensorvertrag (WB3)](pseudo-pair-tensorvertrag.md)
