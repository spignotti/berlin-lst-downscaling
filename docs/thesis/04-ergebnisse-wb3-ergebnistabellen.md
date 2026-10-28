# WB3 · Ergebnistabellen – Baselines und Ablation

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66814a8b35e21918019713`, zuletzt bearbeitet 2026-10-09). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Alle Tabellen dieser Seite beruhen auf training/v1 und sind historisch; Ergebnisse auf training/v3 stehen aus, siehe [04 · Ergebnisse & Analysen](04-ergebnisse.md).

**Wissenschaftliche Diskussion:** [WB3 · Diskussion der Ablationsbefunde](05-diskussion-wb3-ablationsbefunde.md). Die Grenzen der quantitativen Befunde werden dort näher eingeordnet.

**Ergebnisstand 08.10.2026:** Die quantitativen Baseline-, Ablations- und Isolationsbefunde werden innerhalb des festgelegten Vergleichsprotokolls berichtet. Experimentelle Evidenz: GitHub, W&B und GCS. Modell- und Ablationsprotokoll: [Modell & Ablationsprotokoll](03-methodik/modell-ablationsprotokoll.md).

## Kennzahl, Kohorte und Vergleichbarkeit

- **Primärmetrik:** gepoolte zellgewichtete maskierte MAE in K auf gültigen 100-m-Zellen. Die 10-m-Modellvorhersage wird über genau 10×10-Pixel auf 100 m aggregiert. Dies ist **keine** unabhängige 10-m-Genauigkeitsmessung.
- **Splits:** Train 2017–2023, Validation 2024, Test 2025; gleiche Residual-/Backbone-/Patch-Basis für Ablation, Seed 0.
- **Basiskohorte:** Validation: 1.342 angefragt, 1.341 ausgewertet, 1 ausgeschlossen, 341.588 gültige Zellen. Test: 1.803/1.803 ausgewertet, 459.564 gültige Zellen. Identische Patch-Zulassung für naiven Vergleich und Stage 1. Weitere Stufen dokumentieren dieselbe zugelassene Test-/Validation-Kohorte.
- **Protokoll:** Checkpoint nach Validation ausgewählt; Test 2025 jeweils nach Freeze pro Run einmal ausgewertet. Kein zwischen Stufen stattfindendes Lernraten-/Architektur-Retuning.

## A. Historischer Ausgangs-Probe und Residual-Recovery

> **Historisch (training/v1).** Die Probe ist durch die Full-Split-Läufe und den korrigierten Datenstand abgelöst, siehe [04 · Ergebnisse & Analysen](04-ergebnisse.md).

| Probe | Validation MAE (K) | Referenz auf identischer begrenzter Kohorte |
| --- | --- | --- |
| Absoluter Kelvin-Output: ursprünglicher Stage-1-Probe, NO-GO | 294.84 | Naiver Prior 1.5261 |
| Kelvin-Residual-Recovery, bester Checkpoint, GO | 1.1929 | Naiver Prior 1.5261 |

Diese **begrenzten** Probe-Kohorten sind **nicht** mit den nachfolgenden Full-Split-Zahlen als derselbe Test zu verrechnen. [Probe-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-probe-results.md) · [Residual-Recovery-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-debug-results.md).

## B. Full-Split-Baseline und Stage 1

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Neuberechnung auf training/v3: [04 · Ergebnisse & Analysen](04-ergebnisse.md).

| Modell | Validation 2024 MAE (K) | Test 2025 MAE (K) |
| --- | --- | --- |
| Naive blockweise Expansion des Grobpriors | 1.61010 | 1.39523 |
| Stage 1: spektral/Indizes, residual (10 Kanäle) | **0.50285** | **0.53257** |

Stage 1: 20 Epochen, bester Validierungscheckpoint Epoche 19. Originale [Baseline](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/baseline-full-results.md), [Stage-1-Full-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-full-results.md) und [Stage-1-W&B-Run](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/30e5g89u).

## C. Kumulative Ablation

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Neuberechnung auf training/v3: [04 · Ergebnisse & Analysen](04-ergebnisse.md).

| Stufe | Hinzukommende Featuregruppe / Änderung | Modellkanäle\* | Loss | Validation 2024 MAE (K) | Test 2025 MAE (K) |
| --- | --- | --- | --- | --- | --- |
| 1 | Spektral / Indizes | 10 | masked L1 | 0.50285 | 0.53257 |
| 2 | • Morphologie | 18 | masked L1 | 0.51167 | 0.51366 |
| 3 | • Gebäude-/Vegetationsschatten | 20 | masked L1 | 0.50601 | 0.49991 |
| 4 | • acht ERA5-Land-Kanäle | 28 | masked L1 | 0.49962 | 0.47535 |
| 5 | gleiche Inputs wie Stage 4; thermal-aware Ziel | 28 | masked L1 + SSIM + Gradienten-Terme | **0.47624** | **0.45931** |

*Separater LST-Prior nicht mitgezählt.*

**Deskriptive Beobachtungen:** Im 2025-Test nimmt die MAE entlang der kumulativen Stufen ab. Auf Validation ist der Verlauf bis Stufe 4 nicht monoton. Stufe 5 verändert das Trainingziel bei gleichen Featurekanälen, nicht die Kontextfeatures. Der Vergleich unterschiedlicher Ablationsstufen wird unter den statistischen und räumlichen Einschränkungen der Ein-Seed-Studie bewertet.

Quellen: [Ablation-Full-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md) · [Stage-Konfigurationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-stage-configs.md); W&B [Stage 2](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/s09hx959), [Stage 3](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/nfzywjnt), [Stage 4](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/fnqrmqnf), [Stage 5](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/ivt0b55f).

## D. Nicht-kumulative Isolationsläufe

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Neuberechnung auf training/v3: [04 · Ergebnisse & Analysen](04-ergebnisse.md).

| Variante (zusätzlich zu Spektral) | Kanäle | Validation 2024 MAE (K) | Test 2025 MAE (K) |
| --- | --- | --- | --- |
| • Schatten, ohne Morphologie/ERA5 | 12 | 0.50417 | 0.52797 |
| • ERA5, ohne Morphologie/Schatten | 18 | **0.48504** | **0.47023** |
| Referenz: kumulativ Stage 2 (+ Morphologie) | 18 | 0.51167 | 0.51366 |
| Referenz: kumulativ Stage 4 (+ alle Featurefamilien) | 28 | 0.49962 | 0.47535 |

Hier wurde keine zusätzliche Leave-one-out-Konfiguration mit Meteorologie plus einzelnen anderen Familien getestet. Quelle [Ablation-Full-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md); [W&B Shadows](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/w1piuu0f), [W&B ERA5](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling/runs/wi4uf6o3).

## E. Random-Forest-Baseline (Abschluss 09.10.2026)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Neuberechnung auf training/v3: [04 · Ergebnisse & Analysen](04-ergebnisse.md).

Vier feste Random-Forest-Profile mit identischen Seed-0-Trainingszellen und kumulativen Feature-Sets der U-Net-Stufen 1–4. Train 2017–2023; Validation 2024; Test 2025. Der RF trainiert auf 500.000 ausgewählten gültigen 100-m-Zellen (aus 1.741.542 Kandidaten), verwendet aggregierte Features und den physikalischen Prior; bei der 10-m-Inferenz werden pixelweise Residuen geschätzt und für die MAE exakt auf 100 m zurückaggregiert. Ein RF für Stufe 5 entfällt, da dort nur der U-Net-Loss geändert wird.

| Feature-Set | RF Validation MAE (K) | RF Test MAE (K) | U-Net Test MAE (K) |
| --- | --- | --- | --- |
| 1: Spektral/Indizes (10 Kanäle) | 1.28859 | 1.27036 | 0.53257 |
| 2: + Morphologie (18) | 1.26061 | 1.23274 | 0.51366 |
| 3: + Schatten (20) | 1.25810 | 1.25927 | 0.49991 |
| 4: + ERA5 (28) | **1.18268** | **1.18341** | 0.47535 |

**Befund:** Alle vier RF-Varianten übertreffen die naive Prior-Expansion (Test 1.39523 K), bleiben auf derselben Auswertungskohorte aber deutlich hinter den zugehörigen U-Net-Stufen. Das vollständige RF-Feature-Set reduziert die Test-MAE gegenüber naiv um 0.21182 K (15,2 %). Morphologie verbessert den RF-Test gegenüber Stage 1 leicht; mit zusätzlichem Schatten steigt der Testfehler von 1.23274 auf 1.25927 K, und ERA5 senkt ihn auf 1.18341 K. Dies sind deskriptive Unterschiede eines einzigen Seeds, keine isolierten oder statistisch signifikanten Feature-Effekte.

**Vergleichsgrenzen:** Die Bewertungszellen und Ausschlussregeln sind mit der naiven Baseline abgeglichen; RF-Zellstichprobe, Trainingsrepräsentation und Modellkapazität unterscheiden sich vom patchbasierten U-Net. Der RF setzt außerdem den Transfer aggregierter Trainingsfeatures auf native 10-m-Inferenz voraus. Es handelt sich um eine temporale Pseudo-Pair-Rekonstruktion bei wiederkehrenden Standorten, nicht um einen Nachweis unabhängiger 10-m-Genauigkeit oder räumlicher Generalisierung.

**Evidenz:** [Validierter Ergebnisbericht](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/random-forest-full-results.md) · [Methodenvertrag](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/random-forest-baseline.md) · [Validatorprotokoll](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/results/random-forest-full-20261008T143403Z-5A7B6B4D/validation.txt) · [Issue #59](https://github.com/spignotti/berlin-lst-downscaling/issues/59). Vier Modelle auf vollständigen Splits geprüft: Validation 1.341 Patches/341.588 Zellen; Test 1.803 Patches/459.564 Zellen; zehn Real-Patches neu berechnet, zwei Native-Probes repliziert, 500.000 Trainingszellen auditiert. Offline-W&B-Run `pbzqtb54` lokal gesichert; Stand 09.10. noch nicht ins Online-W&B-Projekt synchronisiert.

## Nachweis und Grenzen des Datenstands

Dieser Abschnitt bezieht sich auf den Stand training/v1.

Die Werte und Zählungen stammen aus eingefrorenen, validierten Ergebnisberichten, nicht aus einer hier duplizierten operativen W&B-Run-Tabelle. **Ein validierter Random-Forest-Vergleich ist in Abschnitt E ergänzt.** Ergebnisse für Spatial Holdout, unabhängige 10-m-LST-Wahrheit oder die fertige Berlin-Zeitreihe liegen in dieser Übersicht weiterhin nicht vor.

**Interpretation und Forschungsgrenzen:** [05 · Diskussion & Limitationen](05-diskussion.md). Technische Evidenz: [W&B-Projekt](https://wandb.ai/pignottisilas-berliner-hochschule-f-r-technik/berlin-lst-downscaling) und GitHub-Links oben.
