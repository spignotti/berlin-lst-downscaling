# Stadtstruktur evaluation layer

Umweltatlas maps 06.07 (Stadtstruktur) and 06.08 (Flächentypen
differenziert) on the ISU5 1:5,000 block map. The layer supports
descriptive statistics and stratified evaluation. It is not a V3 feature
channel, and it does not change the frozen 28-band contract.

Scientific class groups are not frozen. Notion holds that decision:
[Datenbasis & Feature Engineering](https://app.notion.com/p/3f3356451f668118953ae616cf7ecfc8).
The page does not yet record an approved consolidation. Candidate
groupings in the release stay `proposal_awaiting_scientific_approval`.

## Sources

Publisher: Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin.
Licence: Datenlizenz Deutschland – Zero – Version 2.0
(`dl-de/zero-2.0`, <https://www.govdata.de/dl-de/zero-2.0>).

| Vintage | Reference date | WFS | Open-data record |
|--------:|----------------|-----|------------------|
| 2020 | 2020-12-31 | `ua_stadtstruktur_2020:ua_stadtstruktur_diff_2020` | [2020 record](https://daten.berlin.de/datensaetze/stadtstruktur-2020-umweltatlas-wfs-cca244ae) |
| 2021 | 2021-12-31 | `ua_stadtstruktur:b_stadtstruktur_differenziert_2021` | [2021–2024 record](https://daten.berlin.de/datensaetze/stadtstruktur-umweltatlas-ab-2021-wfs-8065af3b) |
| 2022 | 2022-12-31 | `ua_stadtstruktur:b_stadtstruktur_differenziert_2022` | same |
| 2023 | 2023-12-31 | `ua_stadtstruktur:b_stadtstruktur_differenziert_2023` | same |
| 2024 | 2024-12-31 | `ua_stadtstruktur:b_stadtstruktur_differenziert_2024` | same |

The differentiated layer carries both the 06.08 code `typ` and the 06.07
code `ststrnr`, on the ISU5 key `schluessel`. The 06.07-only layer
repeats those block parts and is not stored separately. There is no open
vintage for 2017–2019 or for 2025. This release does not assign those
scene years to a neighbouring vintage.

ISU5 street polygons are a separate dataset. Uncovered area inside the
Berlin boundary is street land or another unmapped gap. It is not given
a Stadtstruktur class.

## Contract

- CRS: EPSG:25833. Source polygons stay in the publisher's block geometry.
- 100 m shares use the canonical grid from `common/grid.py`. Each row is
  the planar area of one `typ` in one cell. Mixed cells keep every `typ`.
- A cell is inside the AOI when its centre lies in
  `data/boundaries/berlin_landesgrenze.geojson`. The baked
  `aoi_100m.tif` is not on that lattice: its origin is about
  (369800.83, 5837459.27) and its pixel size is about 99.97 m by 100.10 m,
  against the canonical origin (369190, 5838410) and exact 100 m cells.
- 1,000 m blocks are 10 by 10 cells on that same origin. Edge blocks can
  contain fewer than 100 cells. `share_of_nominal_block` uses 1 km².
- `cells.parquet` columns `majority_*` are diagnostics. They are not the
  evaluation class.
- Candidate A is the publisher's 06.07 code, with null left null on water.
- Candidate B is a coarser thermal-surface proposal. Original codes stay
  recoverable. Neither candidate is applied as a cell label. Type 93
  (`Flughafen`) occurs only in the 2020 vintage and is proposed under
  transport. The 2022 and 2023 vintages each contain one or two polygons
  whose `typklar` disagrees with the modal label of that `typ`. The source
  polygons keep the original string. Share rows use the modal label.
  `ststrnr` stays unique for those types.

Release root:

```
gs://berlin-lst-training-data/evaluation/stadtstruktur/v1/
```

## Commands

Publish and upload:

```
uv run python -m berlin_lst_downscaling.data.stadtstruktur.pipeline \
  --release-dir /tmp/stadtstruktur-v1 \
  --output-root gs://berlin-lst-training-data/evaluation/stadtstruktur/v1
```

Read the local release back:

```
uv run python scripts/validators/validate_stadtstruktur.py \
  --release-dir /tmp/stadtstruktur-v1
```

The compact report and the unapproved grouping file are also in
`docs/results/stadtstruktur-v1/`.

## Release v1, accessed 2026-10-14

The WFS feature counts match the service `numberMatched` values: 26,378
(2020), 26,397 (2021), 26,422 (2022), 26,431 (2023), and 26,613 (2024).
Planar polygon area is 793.46 km² in 2020 and 793.37 km² in 2024. The
dissolved area matches that total to numerical noise, so the block parts
do not overlap. The Berlin boundary is 890.67 km². About 97.3 km² inside
it, 10.9 percent, has no Stadtstruktur polygon. That gap is the ISU5
street land and any other unmapped area. It is left unclassified. Source
area outside the boundary is 0.034 km² in 2020 and none in 2021–2024.

Official `flalle` and planar geometry area differ by a median of about
0.052 percent. A 20-polygon cell-tiling check stays within 1e-7 m² of an
independent grid-envelope intersection. A separate 1 m raster sample stays
within 0.3 percent of the polygon area.

Each vintage covers 90,447 canonical 100 m cells, of which 89,066 have
their centre inside the boundary. About 52,000 of the covered cells, and
56 percent of the covered area, contain more than one `typ`. A single
majority label would hide that mixture. The share tables are the
evaluation product. Class groups are not frozen.
