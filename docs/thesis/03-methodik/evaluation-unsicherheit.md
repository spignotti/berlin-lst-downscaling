# Evaluation & Unsicherheitsanalyse

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66815b9901dc208dc2518d`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

**Evaluationsstand 28.10.2026:** Die bisherige quantitative Modellbewertung (historisch, `training/v1`) umfasst die Skalentreue auf 100 m und den zeitlichen Split 2017–2025. Die Evaluation auf `training/v3` ist ausstehend; die ergänzende unabhängige und räumliche Validierung ist noch nicht abgeschlossen.

## Bereits in der Ablation umgesetzt

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Die Evaluation auf `training/v3` ist ausstehend.

- Checkpoint-Selection auf 2024-Validation, Vergleich mehrerer Runs mit 2025-Test nach Checkpoint-Freeze (je Run einmal).
- Primärmetrik: maskierte, zellgewichtete MAE auf **100 m**, nach exaktem 10×10-Mittel über die 10-m-Ausgabe. Sekundärmetriken nur entsprechend ihrer jeweiligen dokumentierten Rolle.
- Vergleich der naiven prior-expansion Baseline mit Stage 1 und allen Ablationsstufen auf definiertem Kohortenvertrag.

## Vorgesehene zusätzliche Evaluation

- Vergleich mit ergänzenden thermalen Referenzen wie ECOSTRESS unter Berücksichtigung räumlicher Auflösung, Überflugzeit und Sensorunsicherheit.
- Räumlich unabhängiger Holdout zur Ergänzung des bereits getesteten zeitlichen Holds-out und Analyse räumlicher Robustheit.
- Räumliche Strukturqualität, Temperaturkontraste, lokale thermische Plausibilität und Unsicherheit ohne native 10-m-Ground-Truth.
- Quantifizierung bzw. kritische Einordnung der Ein-Seed-Beschränkung und der Mehrfachauswertung derselben 2025-Testkohorte.

## Wolkenmaskierung und Selektionsrisiken

In Abstimmung mit der Betreuung wird eine mögliche **oberflächentypabhängige Verzerrung der Wolkenklassifikation** als Selektionsrisiko berücksichtigt. Nicht erkannte Cirrus- oder Wolkenreste können thermische Landsat-Beobachtungen beeinflussen; bei systematisch unterschiedlicher Maskierungsqualität zwischen Stadtstrukturen könnte die gültige Trainingsstichprobe selektiv zusammengesetzt sein. Der frühere Cloud-Masking-Audit vom 28.06.2026 belegt entsprechende technische Maskenprüfungen in einem historischen GEE-Pipeline-Stand, ist aber **kein Nachweis** eines vorhandenen Bias im aktuellen Datenstand (ARD v10, `training/v3`). Eine eigenständige statistische Untersuchung solcher Restverzerrungen ist bislang nicht als Ergebnis dokumentiert.

## Emissivitätskopplung des Landsat-Targets

Eine Read-only-Diagnose ([Emissivitätsdiagnose](../../emissivity-diagnostic.md)) auf neun Landsat-8/9-Szenen (6 Sommer, 3 Winter, 2022–2024) vergleicht (a) die pixelweise Inversion mit der Produktemissivität, (b) die statische ASTER-GED-v3-Emissivität (AG100.003, Mittel 2000–2008, an TIRS B10 angepasst nach Malakar 2018, ohne NDVI-Anpassung) und (c) eine einheitliche Szenenmedian-Emissivität als Obergrenze.

- Differenz (a)−(b) über 8,48 Mio. klare Pixel: Mittel −0,07 K, Median +0,04 K, p05/p95 −0,86 / +0,57 K. Unter (c) betragen p05/p95 −1,04 / +1,17 K (etwa 1,5-fach breiter).
- Die Differenz (a)−(b) steigt mit dem Versiegelungsgrad von −0,21 K (0–10 % versiegelt) auf +0,14 K (90–100 %).
- Die sommerlichen Steigungen der Landsat-NDVI–LST-Beziehung sind mit Produktemissivität 4–17 % steiler als mit statischer GED-Emissivität (Median etwa 7 %); die Korrelationen ändern sich um höchstens 0,02.
- Die Größenordnung liegt unterhalb der Produktunsicherheit (ST_QA, Mittel 2,2–5,2 K).

Folgerung: Ein eigener Target-Arm (z. B. LST mit statischer Emissivität neu berechnet) wird nicht in die Ablation aufgenommen. Die Emissivitätskopplung bleibt als Limitation der Verwendung von C2 ST_B10 als Target dokumentiert. Einschränkungen der Diagnose: GED-Zeitraum 2000–2008 gegenüber Szenen 2022–2024 (Landbedeckungswandel nicht trennbar); Landsat 9 nutzt die Landsat-8-Koeffizienten nach Malakar (dokumentierte Näherung); das Sanity-Residuum (a) gegenüber ST_B10 beträgt im Mittel 0,11–0,17 K, das Ziel < 0,1 K wird nicht erreicht (einheitliche Konstante).

## Aussagegrenze

Die bisherige 100-m-Aggregations-MAE misst **keine direkt beobachtete 10-m-LST-Genauigkeit**. Aufgrund räumlich wiederkehrender Patchanker ist auch eine unabhängige räumliche Transferfähigkeit noch nicht nachgewiesen.

[Aufgabenstellung](../00-aufgabenstellung.md) · [Stage-1-Limitationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-full-results.md) · [Ablations-Limitationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md).
