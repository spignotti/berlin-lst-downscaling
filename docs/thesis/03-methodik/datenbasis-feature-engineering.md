# Datenbasis & Feature Engineering

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668118953ae616cf7ecfc8`, zuletzt bearbeitet 2026-10-22). Text unverändert übernommen.

**Datenstand: 22.10.2026 (training/v2).** Für Berlin liegt ein publiziertes, räumlich harmonisiertes Multi-Source-Feature-Release mit thermischen Landsat-Zielwerten und hochaufgelösten Kontextprädiktoren vor.

## Umgesetzt

- Landsat 8/9 als thermischer Anker, Sentinel-2 als hochaufgelöste spektrale Proxyquelle, Berlin-Geodaten für Morphologie/Verschattung, ERA5-Land für atmosphärische Prädiktoren.
- Nach Szenen gematchte Daten; train 2017–2023, val 2024, test 2025, 2026 Inferenzreserve.
- 10-m-kanonisches Grid (EPSG:25833), 100-m-Thermalziel als Flächenmittel des 30-m-Landsat-Produkts; Feature-Release v5 (Vorgänger V3) mit 28 aktiven Kanälen. Vier Featurefamilien: Spektral/Index (10), Morphologie (8), Schatten (2), ERA5 (8).
- Train-Only-Scaler und eligibility bei 100 % validem 10-m-Feature-Support innerhalb jeder gültigen 100-m-Zelle. LoD-Vintage-Mapping und NoData-/Abdeckungssemantik sind methodisch relevant.

## Trainingsfreigabe und Split-Integrität

Der WB2c-4-Trainingsrelease vom 27.08.2026 (`training/v1`, Grundlage der dokumentierten Stage-1–5-Läufe; abgelöst durch `training/v2`, siehe unten) enthält **345 Ankerzeilen**, davon 324 aus dem Zeitraum 2017–2025 und 21 als Inferenz-Reserve für 2026. Unter den 324 Trainings-/Evaluationskandidaten besitzen **30 keine gültigen 100-m-Zellen** nach der strikten Eligibility-Regel. Damit sind **294 Szenen** für die Modellstichprobe freigegeben. Die tatsächlich publizierten Zellen verteilen sich auf **4.057.392 Train**, **696.942 Validation** und **765.831 Test**. Der unabhängige Release-Check bestätigte die deterministische Zell-ID-Bildung, die Train-Only-Skalierung und **keine Sentinel-2-Szenen-ID, die in mehreren nicht-Inferenz-Splits vorkommt** (149 eindeutige S2-Szenen im geprüften Bestand). Das ist eine geprüfte formale Split-Eigenschaft; eine räumlich unabhängige Validierung ist daraus nicht abzuleiten.

**Evidenz:** [WB2c-4 Release Evidence, 27.08.2026](https://app.notion.com/p/3c9356451f668182ac88cfd2350b9007), normativer [Repository-Datenvertrag](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md). Die im Augustbericht aufgeführten früheren GCS-Pfade wurden im September in den kanonischen Bucket `gs://berlin-lst-training-data/` übertragen.

## Datenstand training/v2 und EDA v2

**Stand 22.10.2026.** Die Trainingsfreigabe `training/v2` ist die Datengrundlage für das finale Training. Sie baut auf Feature-Release v5, dem korrigierten Versiegelungsprodukt `imperviousness-v3` und dem Landsat-ARD mit Flächenmittel-Target auf. `training/v1` und Feature-Release V3 bleiben unverändert erhalten. Die dokumentierten Stage-1–5-Läufe, die Isolationsläufe und die Random-Forest-Baseline beruhen auf `training/v1`.

### Datenstände

| Artefakt | Kennung | Umfang | Verweis |
| --- | --- | --- | --- |
| `training/v2` | Run `bae22596`, `training_policy_hash` `08c2ec4d5a257c7e`, Feature-Config-Hash `2d2086f9d36e48fe` | `features/v5` mit 324/324 Feature-Stacks; 345 Manifest-Szenen, davon 283 publiziert und 62 ausgeschlossen; 4.728.053 eligible Szene×Zelle-Zeilen (Train 3.489.565, Validation 608.091, Test 630.397); 69.870 eindeutige eligible Zellen | `gs://berlin-lst-training-data/training/v2` · [PR #79](https://github.com/spignotti/berlin-lst-downscaling/pull/79) (`76b26275`) |
| `training/patch-index/v2` | Run `1a21270a` | Publizierte Schwelle 244/256 eligible Zellen: 6.619 Patches in 166 Szenen (Train 4.606, Validation 895, Test 1.118). Nachgezählt, nicht publiziert: 230/256 → 8.678 Patches in 184 Szenen; 200/256 → 12.076 in 199; 128/256 → 18.131 in 233. Zum Vergleich `patch-index/v1`: 10.018 in 182 | `gs://berlin-lst-training-data/training/patch-index/v2` · [PR #79](https://github.com/spignotti/berlin-lst-downscaling/pull/79) |
| EDA v2 | Run `training-eda-v2-20261021T165939Z-280B1A35`, Scan-Commit `d454cf9e` | 18 Parquet-Tabellen und `summary.json` über `training/v2`, `features/v5`, `patch-index/v2` und `evaluation/stadtstruktur/v1`; 322 gescannte Szenen; Validator bestanden | `gs://berlin-lst-training-data/evaluation/eda/v2/` · [docs/eda-v1.md](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/eda-v1.md) · [PR #78](https://github.com/spignotti/berlin-lst-downscaling/pull/78) (`dfe0af9b`) |

Ausschlussgründe der 62 Szenen: `no_eligible_cells` (39), `inference_deferred` (21, alle 2026) und `usgs_st_implausible_20220827` (2).

### Datenkorrekturen in training/v2

| Bereich | Regel | Verweis |
| --- | --- | --- |
| Versiegelung | Umweltatlas-01.02-Klassenraster 2016/2021 werden über eine Code-Tabelle in Prozent übersetzt: 0 → 0 %, 5–95 → Codewert, 100/102 → 100 %, 110 (Gleisschotter) → 100 %. Code 103 innerhalb von Stadtstruktur-2024-Wasser (`typ` 100) → 0 %; Code 103 auf Straßenland → Abschnittswert `vg` aus WFS-Layer `ua_versiegelung_2021:versieg2021_str`; Code 103 in anderen Blöcken und Code 101 → flächengleiches Mittel der Spenderpixel im Block (Rückfall: 100-m-Ring bzw. nächster Block ≤ 100 m). Kein eigenes Bezirksmittel; nicht erhobene Straßenabschnitte tragen den Bezirkswert bereits im Feld `vg`. Nicht zuordenbare Pixel bleiben NaN. 0 unbekannte Codes; Code 101 in beiden Vintages vollständig zugeordnet. Das 2016-Raster trägt EPSG:25833 nur als WKT ohne EPSG-Code. | `static/sources/imperviousness-v3` (Config-Hash 2016 `826b245461e8`, 2021 `02be2b1ab273`) · [Datenvertrag](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md) |
| Landsat-QA | Schnee (QA\_PIXEL Bit 5) gilt nicht mehr als klar. Wolkenkonfidenz (Bits 8–9) ≥ 2 gilt als bewölkt, auch ohne gesetztes Wolkenbit. Wasser (Bit 7) bleibt klar. | [PR #79](https://github.com/spignotti/berlin-lst-downscaling/pull/79) |
| Landsat-Target | LST auf dem kanonischen 100-m-Raster ist das Flächenmittel der gelieferten 30-m-`lwir11`-Werte (`resampling=average`), nicht mehr Nearest Neighbour. QA wird bitweise auf dem nativen 30-m-Raster ausgewertet; eine 100-m-Zelle erhält ein Bit, sobald ein überlappendes 30-m-Pixel es trägt (`native_bit_or`). Die Cloud-Dilation bleibt bei 2 Pixeln auf 100 m (200 m Puffer). | `ard/full/2017-2026-qa-avg-20261019`, Schema-Hash `7|st=average|qa=native_bit_or` |
| Szenen-Denylist | `LC09_L2SP_193023_20220827_02_T2` und `LC09_L2SP_193024_20220827_02_T2` sind wegen physikalisch unplausibel kalter LST ausgeschlossen (`usgs_st_implausible_20220827`), ohne Maske, Zellen oder Scaler-Beitrag. Das LST-Minimum der Freigabe liegt danach bei 278,885 K; kein Szenenmittel liegt unter 280 K. | `training_policy_hash` enthält die Denylist |

### Befunde der EDA v2

Alle Werte beziehen sich auf 100-m-Zellen der Freigabe `training/v2`; sie beschreiben das Trainingsziel, keine 10-m-Struktur.

- **Szenen und Saisonalität:** 283 publizierte Szenen (Train 209, Validation 36, Test 38); je Jahr zwischen 12 (2021) und 50 (2023), 2017 mit 25. Keine Winterszene (DJF). Publiziert sind 164 JJA-, 53 MAM- und 66 SON-Szenen; JJA stellt 2.476.968 der 4.728.053 eligible Zeilen. Beobachtungsgewichtete Mittel: JJA 308,1 K, MAM 302,2 K, SON 299,0 K.
- **Abdeckung:** 69.870 von 89.066 AOI-Zellen (78,4 %) sind mindestens einmal eligible. 2021 ist die Abdeckungslücke mit 46.015 Zellen (51,7 %). Der Übergang von target-gültigen zu eligible Zellen scheitert überwiegend an unvollständigem 10-m-Feature-Support (12.520.645 target-gültige Szene×Zellen ohne vollen Support gegenüber 4.728.053 eligible).
- **Patches:** Bei 244/256 liegen 6.619 Patches in 166 Szenen. 117 publizierte Szenen liefern keinen Patch (Train 94, Validation 7, Test 16); damit tragen nur 115 von 209 Train-Szenen, 29 von 36 Validierungs- und 22 von 38 Testszenen zum Patchtraining bzw. zur Patch-Evaluation bei.
- **LST-Verteilung:** Beobachtungsgewichtet Mittel 304,64 K, Standardabweichung 6,50 K, q05–q95 293,76–315,17 K, Bereich 278,885–336,923 K. Zwischen-Szenen-Standardabweichung 6,16 K gegenüber mittlerer Innerhalb-Szenen-Standardabweichung 2,98 K. Splitmittel: Train 304,80 K, Validation 305,53 K, Test 302,93 K.
- **Heterogenität im 1-km-Block:** Mittlere Innerhalb-Block-Standardabweichung 1,48 K (p95 3,17 K), mittlere Spannweite 6,13 K; mittlere absolute 4-Nachbar-Differenz 0,755 K. Mittlere Block-Abdeckung mit eligible Zellen 0,588; 32.978 von 80.477 Szene×Block-Zeilen liegen unter 0,5.
- **Features:** Die drei ERA5-Niederschlagskanäle sind innerhalb einer Szene praktisch konstant (Innerhalb/Zwischen-Verhältnis \< 1 %) und stark schief. `building_height_mean` und `building_height_max` korrelieren auf 100 m mit r = 0,9998; `building_height_std` ist auf 100 m im Median 0. Szenenübergreifend korrelieren `vpd_scene` (0,73) und `t2m_scene` (0,73) am stärksten mit LST, innerhalb einer Szene `ndbi` (0,71), `imperviousness` (0,69) und `ndvi` (−0,60). Kein Kanal wird dadurch entfernt.
- **Stadtstruktur (Proposal B, Vintage 2024, zellflächengewichtete Szenenanomalie):** Wasser −6,12 K, Wald −5,10 K, Park/Friedhof −1,26 K, niedrige Wohnbebauung −0,24 K, geschlossene Blockbebauung +2,34 K, Gewerbe/Industrie +3,34 K. Die Vorzeichen bleiben in allen Vintages und im Testjahr 2025 stabil; Landwirtschaft wechselt das Vorzeichen zwischen Jahren. Versiegelte Plätze, Camping und Baustellen sind in allen Vintages zu klein für eigenständige Strata. Wasser ist nach der Korrektur mit 6,4 % versiegelt (vorher 86,5 %); der Rest stammt aus Nicht-103-Codes innerhalb der Wasserpolygone.

**Aussagegrenze:** Das Testjahr 2025 ist in der EDA beschrieben und damit kein unberührter Holdout mehr für deskriptive Aussagen. 2026 ist nur Inventar. Eine eligible Zelle liegt außerhalb des AOI-Gitters. Die Ursache ungültiger 10-m-Subpixel ist nicht gespeichert.

## Datenbezogene Aussagegrenzen

Der Bandname `s2_broadband_albedo` bezeichnet ein kurzwelliges Reflektanz-Proxy, kein physikalisch vollständig korrigiertes Albedo-Produkt. Statische Geometrie-/Vegetationsvintages sind keine lückenlose historische Beobachtung aller Gebäudeveränderungen. ECOSTRESS gehört zur ergänzenden thermalen Validierung, nicht zum Training.

## Technische Evidenz

[GitHub: Data sources and contracts](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/data-sources-and-contracts.md) (exakte Kanäle, Vintages, Grids, QA/Provenance). Die historischen WB2-Unterlagen dokumentieren die Entwicklung des Datenbestands; die normative aktuelle Feature- und Release-Spezifikation steht im Repository.

## Deskriptive Datencharakteristik

Die deskriptive Verteilungsanalyse von Target, Prädiktoren, Abdeckung und Stadtstruktur liegt als EDA v2 über `training/v2` vor (Abschnitt „Datenstand training/v2 und EDA v2“, [docs/eda-v1.md](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/eda-v1.md)). Abbildungen für den Ergebnisteil werden aus den gespeicherten Tabellen unter `gs://berlin-lst-training-data/evaluation/eda/v2/tables/` erzeugt.

- [WB2 ARD-Pipeline](wb2-ard-pipeline.md)

- [WB2 Sekundärdaten](wb2-sekundaerdaten.md)

- [WB2 Feature Engineering & Trainingsdatenaufbereitung](wb2-feature-engineering.md)

## Entscheidung 15.10.2026 – Stadtstruktur-Jahreszuordnung für die Evaluation

Technische Schicht [GitHub #70](https://github.com/spignotti/berlin-lst-downscaling/issues/70), Release `evaluation/stadtstruktur/v1` mit originalen 2020–2024-Umweltatlas-Klassen/100-m-Flächenanteilen, ist verfügbar. Analog zum dokumentierten Morphologie-**Carry-forward** je Datenstand: **2017–2020 → 2020, 2021 → 2021, 2022 → 2022, 2023 → 2023, 2024–2025 → 2024**. Für 2017–2019 ist dies ausdrücklich **Carry-back** auf ältesten Stand 2020 und eine zeitliche Approximation, keine beobachtete frühere Strukturklasse. Zuordnung durch technische Folgeumsetzung versionieren und testen; bestehende 28-Feature-Pipeline nicht verändern. **Nicht klassifizierte Flächen** (insb. ISU5-Straßenraum) bleiben unklassifiziert und werden **nur aus stadtstrukturstratifizierten Analysen** ausgeschlossen, nicht aus der allgemeinen LST-Trainings-/Testkohorte. Gemischte 100-m-Zellen als Flächenanteile behalten. Noch offen: wenige wissenschaftlich begründete, verständliche Strukturgruppen festlegen und Crosswalk freigeben, dann EDA konkretisieren.

## Prüfauftrag 15.10.2026 – wissenschaftliche Konsolidierung der Stadtstrukturklassen

[GitHub Issue #74](https://github.com/spignotti/berlin-lst-downscaling/issues/74) erstellt: Klasse-für-Klasse-Analyse der im Release `stadtstruktur-v1` vorhandenen Original-06.08-Typen und amtlichen 06.07-Stadtstrukturgruppen mit **flächengewichteten** Flächenanteilen, Zell-/Mischungsstatistik, Seltenheit, Zeitständen und transparenten Crosswalk-Vorschlägen. Bestehende `report.json` enthält keine ausreichend detaillierte Verteilung pro Klasse; deshalb noch keine wissenschaftliche Freigabe. **Amtliche 06.07-Klassifikation als erste externe Referenz**, vorhandene private 10-Gruppen-Variante B ausdrücklich vorläufig (insb. heterogene Grün-/Offenflächen und Sondernutzungen); kein LCZ-System behaupten, wenn nicht eigenständig fachlich validiert. Freigabe nach sachlicher Eignung, Klassen-Support und zeitlicher Stabilität, *vor* EDA; nicht nach Modellfehlern/Testmetrik optimieren. Code und Analysestufe separat, keine Veränderung der 28 Featurekanäle. Die zuvor beschlossene temporale Proxyzuordnung und Ausschluss nicht klassifizierter Zellen **nur aus Strukturstrata** gelten weiterhin.

## Wissenschaftliche Freigabe 16.10.2026 – Stadtstrukturklassifikation für EDA und räumliche Evaluation

**Proposal B aus** [**GitHub PR #75**](https://github.com/spignotti/berlin-lst-downscaling/pull/75) **ist fachlich freigegeben** als primäre, zweckgebundene analytische Gruppierung. **Nicht** verwechseln mit dem früher veröffentlichten `candidate B` aus `stadtstruktur-v1`; dieser ist für die thermische Gruppierung **verworfen**, weil `vegetated_open` unzulässig unterschiedliche offene Flächen vereint (entspricht im ausgewerteten Bestand der großen amtlichen Strukturklasse 18, 2024 ca. 325,76 km² bzw. 41 % der klassifizierten Fläche). Die **amtliche Berliner Umweltatlas-06.07-****`ststrnr`****-Klassifikation bleibt die externe, veröffentlichte Referenz**. Wir geben weder LCZ noch ein allgemeingültiges thermisches Klassifikationssystem vor.

**Begründung:** Proposal B trennt thermisch plausibel verschiedene Oberflächentypen, insbesondere Wald, Park/Friedhof, Landwirtschaft, Kleingartenanlagen, Brache und versiegelte Stadtplätze, und hält weitere bauliche Gruppen auseinander. Die Gruppierung wurde vor Betrachtung von Modellfehlern auf Grundlage von offiziellen 06.08-Originaltypen, fachlicher Semantik und statistischem Flächensupport vorgeschlagen. 2024 sind ca. 790,82 km² auf Zellen mit Mittelpunkt innerhalb Berlins klassifiziert; ca. 97,30 km² Straßen-/sonstige nicht kartierte Fläche bleiben unklassifiziert. Elf von zwölf regulären Vergleichsgruppen überschreiten das vorab definierte Flächen-/Zellsupport-Kriterium; versiegelte Stadtplätze (\~0,47 km²) sind für robuste eigenständige Aussagen zu klein. Camping ist eine kleine Grenzgruppe; Baustellen und Wasser sind Ausschlusskategorien für reguläre Strukturvergleiche; öffentliche/Sondernutzungen heterogen und vorsichtig zu interpretieren. Kleine Kategorien nicht nachträglich in andere Gruppen mischen oder anhand von MAE optimieren.

**Flächengewichtung statt Mehrheitsklasse:** 2024 sind ca. 58 % der abgedeckten 100-m-Zellen aus mehreren 06.08-Typen und ca. 53 % aus mehreren Proposal-B-Gruppen zusammengesetzt; auf 1 km sind 964/1.008 Blöcken gemischt. Relevante Ergebnisse nach tatsächlichen Flächenanteilen je 100-m-Zelle auswerten, offizielle 06.07-Klassen als Vergleich/Sensitivität, keine künstliche eindeutige Klassenzuschreibung; Support pro geografischem Block und Szene bei späterer Inferenz prüfen, nicht Flächenpixel als unabhängige Samples zählen.

**Zeitliche Referenz:** 2017–2020→2020 (2017–2019 ausdrücklich Carry-back), 2021→2021, 2022→2022, 2023→2023, 2024–2025→2024 (Carry-forward); 2026-Anker unzugeordnet. Projekt-eigene 28 Trainingsfeatures und allgemeine Training-/Testkohorten unverändert. Nicht klassifizierte Flächen nur aus stadtstrukturstratifizierten Auswertungen ausschließen. Quellen-/Skalenunsicherheiten und zeitliche Näherung in der Thesis benennen.

**Evidenz:** [PR #75](https://github.com/spignotti/berlin-lst-downscaling/pull/75), Report `docs/results/stadtstruktur-class-review/report.json` auf Branch `analysis/stadtstruktur-class-review`, archivierte volle Tabellen `gs://berlin-lst-training-data/evaluation/stadtstruktur/class-review-v1/`, Quellrelease `stadtstruktur-v1` unverändert. Die formale Freigabe hier in Notion ist erfolgt; **technische Approval-Metadaten/der PR sind vor dem Merge noch auf** **`awaiting_scientific_review`** **und müssen beim Abschluss konsistent mit dieser Entscheidung umgesetzt werden.** User übernimmt Merge/PR-Abschluss selbst. Danach EDA-Issue auf dieser Grundlage spezifizieren, EDA und stadtstratifizierte Modellevaluation bleiben ausstehend.
