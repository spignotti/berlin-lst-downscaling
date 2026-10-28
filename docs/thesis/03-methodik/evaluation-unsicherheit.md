# Evaluation & Unsicherheitsanalyse

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66815b9901dc208dc2518d`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.

**Evaluationsstand 08.10.2026:** Die bisherige quantitative Modellbewertung umfasst die Skalentreue auf 100 m und den zeitlichen Split 2017–2025. Die ergänzende unabhängige und räumliche Validierung ist noch nicht abgeschlossen.

## Bereits in der Ablation umgesetzt

- Checkpoint-Selection auf 2024-Validation, Vergleich mehrerer Runs mit 2025-Test nach Checkpoint-Freeze (je Run einmal).
- Primärmetrik: maskierte, zellgewichtete MAE auf **100 m**, nach exaktem 10×10-Mittel über die 10-m-Ausgabe. Sekundärmetriken nur entsprechend ihrer jeweiligen dokumentierten Rolle.
- Vergleich der naiven prior-expansion Baseline mit Stage 1 und allen Ablationsstufen auf definiertem Kohortenvertrag.

## Vorgesehene zusätzliche Evaluation

- Vergleich mit ergänzenden thermalen Referenzen wie ECOSTRESS unter Berücksichtigung räumlicher Auflösung, Überflugzeit und Sensorunsicherheit.
- Räumlich unabhängiger Holdout zur Ergänzung des bereits getesteten zeitlichen Holds-out und Analyse räumlicher Robustheit.
- Räumliche Strukturqualität, Temperaturkontraste, lokale thermische Plausibilität und Unsicherheit ohne native 10-m-Ground-Truth.
- Quantifizierung bzw. kritische Einordnung der Ein-Seed-Beschränkung und der Mehrfachauswertung derselben 2025-Testkohorte.

## Wolkenmaskierung und Selektionsrisiken

In der [Betreuungsbesprechung Ende Juli 2026](https://app.notion.com/p/f10d9184bd4a4c87be630abbea8475f8) wurde eine mögliche **oberflächentypabhängige Verzerrung der Wolkenklassifikation** diskutiert. Nicht erkannte Cirrus- oder Wolkenreste können thermische Landsat-Beobachtungen beeinflussen; bei systematisch unterschiedlicher Maskierungsqualität zwischen Stadtstrukturen könnte die gültige Trainingsstichprobe selektiv zusammengesetzt sein. Der frühere [Cloud-Masking-Audit vom 28.06.2026](https://app.notion.com/p/38d356451f66818bb446f40023498682) belegt entsprechende technische Maskenprüfungen in einem historischen GEE-Pipeline-Stand, ist aber **kein Nachweis** eines vorhandenen Bias im heutigen publizierten V3-Datensatz. Eine eigenständige statistische Untersuchung solcher Restverzerrungen ist bislang nicht als Ergebnis dokumentiert.

## Aussagegrenze

Die bisherige 100-m-Aggregations-MAE misst **keine direkt beobachtete 10-m-LST-Genauigkeit**. Aufgrund räumlich wiederkehrender Patchanker ist auch eine unabhängige räumliche Transferfähigkeit noch nicht nachgewiesen.

[Aufgabenstellung](../00-aufgabenstellung.md) · [Stage-1-Limitationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-full-results.md) · [Ablations-Limitationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md).
