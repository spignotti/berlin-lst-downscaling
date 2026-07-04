# Szenen-Selektion — Volumen-Scan

**Datum:** 2026-07-04 20:43
**Zeitraum:** 2017/2018/2019/2020/2021/2022/2023/2024/2025 | Monate [5, 6, 7, 8, 9]

## Counts

- Landsat-Anker total: **220**
  - Gekoppelt (geschätzt): **143**  (Rate ≈ 65%)
  - Verworfen (geschätzt): **77**
- S2-Kandidaten (Summe): **2640**
- ECOSTRESS-Granules: **249**

## Volumen-Schätzung (GB)

| Quelle | Szenen | GB/Szene | Gesamt GB |
|--------|--------|-----------|-----------|
| Landsat | 220 | 0.05 | 11.0 |
| Sentinel-2 | 2640 | 0.2 | 528.0 |
| ECOSTRESS | 249 | 0.005 | 1.25 |
| **Total** | | | **540.2** |

## Anmerkungen

- S2-Kandidaten sind Summe über alle Anker-Fenster (Über­schätzung bei Overlap)
- ECOSTRESS ohne Pixel-Load geschätzt; nur CMR-Metadaten
- coupling_rate = geschätzt aus λ=0.1 und median Δt=1d → Score ≈ clear_frac − 0.033
- Min. clear_frac-Schwelle = 0.3