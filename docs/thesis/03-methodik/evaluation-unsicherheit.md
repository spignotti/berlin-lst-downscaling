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

## Paired scene bootstrap (ablation evaluation)

The ablation compares 9 arms with 3 seeds each. The comparison of two arms uses a paired scene cluster bootstrap implemented in `berlin_lst_downscaling.analysis.scene_bootstrap` with the CLI `scripts/analyze_scene_bootstrap.py`.

**Estimand.** Per run and scene, patch sums are pooled (sum of absolute errors divided by the sum of valid cells for MAE; sum of SSIM values divided by the sum of SSIM windows for SSIM), never averaged as patch means. Per arm, the three seed runs are averaged per scene. The primary estimate is the equal-scene-weight mean of the paired difference d_s = A - B over scenes.

**Resampling.** The unit is the scene. Scenes are resampled with replacement (B = 10 000, fixed `numpy.random.default_rng` seed, scene ids sorted) and the 95 % interval is the percentile interval of the replicate means (endpoints are the 2.5 % and 97.5 % quantiles from NumPy `np.quantile(..., method="linear")`). Patches are not the unit because neighbouring patches within a scene share atmosphere and surface state and are spatially autocorrelated, so treating them as independent would understate uncertainty. Seeds are never resampled.

**Seed summaries.** Per-seed pooled overall metrics (mean, SD with ddof = 1, per-seed difference, sign consistency such as "3/3 same sign") are reported as diagnostics only. They differ from the primary estimand, which weights scenes equally. No seed-level significance test is performed: with n = 3 the smallest two-sided sign-flip p-value is 0.25.

**Mismatch handling.** By default all six runs must have identical scene sets and identical per-scene patch sets; otherwise the tool raises and names the affected runs and scenes. `--allow-scene-mismatch` drops whole scenes that are not present with identical patch sets in all runs (never an intersection of patches) and records every dropped scene with its reason and affected runs. If no scene remains, the tool fails. Differing `valid_cells` or `ssim_windows` for the same patch, or conflicting scene dates, always raise.

**SSIM.** Scenes without SSIM windows have a null SSIM; if a retained scene lacks support for the selected metric the tool raises. Direction: MAE is lower-is-better (A better if d < 0), SSIM is higher-is-better (A better if d > 0); ties are reported separately.

**Multiplicity.** `holm_adjust` applies the Holm correction to a declared family of comparisons against one reference arm. The module does not produce bootstrap p-values yet; this is optional future work.

**Validation first.** The analyzer is validation-only: the CLI accepts only `--split validation` and rejects test-labelled exports. The tool only reads explicit local export files and never launches an evaluation.

**Export schema (schema_version 1).** One Parquet file per run and split, columns: `schema_version` int32, `arm` string, `seed` int64 (nullable), `run_id` string, `checkpoint` string (nullable), `patch_id` string, `scene_id` string, `scene_date` string (nullable, YYYY-MM-DD), `split` string, `abs_error_sum` float64, `valid_cells` int64, `ssim_sum` float64, `ssim_windows` int64, `filled_feature_pixels` int64 (nullable). The footer key `patch_export` holds JSON with `schema_version`, `split`, `arm`, `seed`, `run_id`, `checkpoint`. File names: `eval_validation_patches.parquet`, `eval_test_patches.parquet`, and `eval_stage{n}_<split>_patches.parquet` for the random-forest runs.

**Reporting table template.**

| Arm A | Arm B | Metric | Mean A (SD) | Mean B (SD) | Paired diff A - B | 95 % CI | Scenes A better / tie / B better | Seed sign consistency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `<arm>` | `<arm>` | `<metric>` | `<value (sd)>` | `<value (sd)>` | `<value>` | `[<low>, <high>]` | `<n / n / n>` | `<k/3 same sign>` |

**Example (validation only).**

```bash
uv run python scripts/analyze_scene_bootstrap.py \
  --arm-a <arm_a>/seed0/eval_validation_patches.parquet <arm_a>/seed1/eval_validation_patches.parquet <arm_a>/seed2/eval_validation_patches.parquet \
  --arm-b <arm_b>/seed0/eval_validation_patches.parquet <arm_b>/seed1/eval_validation_patches.parquet <arm_b>/seed2/eval_validation_patches.parquet \
  --output-prefix <out_dir>/<arm_a>_vs_<arm_b>
```
