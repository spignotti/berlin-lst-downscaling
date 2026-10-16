# Stadtstruktur evaluation layer

Umweltatlas maps 06.07 (Stadtstruktur) and 06.08 (Flächentypen
differenziert) on the ISU5 1:5,000 block map. The layer supports
descriptive statistics and stratified evaluation. It is not a V3 feature
channel, and it does not change the frozen 28-band contract.

The evaluation grouping was decided on 2026-10-16: proposal B from the
class review, the split in `class_review.py` (`PROPOSAL_B`). Official
06.07 `ststrnr` stays the external reference. The coarser candidate B
stored in v1, which pools open land as `vegetated_open`, was not
selected. Proposal B is the grouping for later stratified description.
It is not a V3 feature and it is not written as a cell label. The v1
files keep `proposal_awaiting_scientific_approval` because that release
was not rewritten. The class-review report keeps
`awaiting_scientific_review` as the record of the check before this
decision.

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
evaluation product. The v1 files do not store a chosen class label.
The decision is recorded at the top of this document.

## Class review, accessed 2026-10-15

Issue #74. This review reads v1 and does not rewrite it. Group membership
was fixed before the areas were computed. The published report still
says `awaiting_scientific_review`. That field is the state of the check.
On 2026-10-16 proposal B was adopted for later stratified description.
The report file is evidence and stays as published.

Machine-readable tables, including the per-class rows:

`gs://berlin-lst-training-data/evaluation/stadtstruktur/class-review-v1/`

Compact copy: `docs/results/stadtstruktur-class-review/report.json`.

```
uv run python scripts/operators/review_stadtstruktur_classes.py \
  --release-dir /tmp/stadtstruktur-v1 \
  --manifest /tmp/stadtstruktur-manifest/manifest.parquet \
  --output-dir /tmp/stadtstruktur-class-review-v1

uv run python scripts/validators/validate_stadtstruktur_class_review.py \
  --review-dir /tmp/stadtstruktur-class-review-v1
```

`--release-dir` is a local copy of the v1 prefix. The published manifest
is the v3 anchor list, 345 anchors.

Class area is the sum of `shares.parquet` `area_m2` on cells with
`inside_aoi`. In 2024 that is 790.82 km². The street and other unmapped
gap stays out of every class share: 97.30 km² in the v1 coverage, about
10.9 percent of Berlin. Cells whose centre is outside the boundary hold
the remaining share area. Their exclusion makes the classified total
2.55 km² smaller than the geometric source-inside area. Polygon counts
are block parts that intersect the boundary, once per `schluessel`.
Effective cells are `area_m2 / 10000`. A sparse label is under 1.0 km²
or under 50 effective cells. It does not move a type into another group.

Scene years for a later stratified description, not as area weights:
2017–2019 carry back to vintage 2020 (28, 33, and 31 anchors), 2020 uses
2020 (31), 2021–2023 use the same year (18, 42, 57), and 2025 carries
forward to 2024 (43 anchors, with 41 in 2024 itself). Twenty-one anchors
in 2026 are listed and left unassigned. Anchor counts are not multiplied
by cell counts.

`schluessel` matches across vintages. From 2020 to 2024, 99.1 percent of
the earlier keys and 98.3 percent of the later keys match, so the key is
usable as a panel. Of the matched 2024 polygon area, 98.7 percent keeps
its `typ`. The changed area is 10.3 km².

Three groupings are in the crosswalk, plus the published candidate:

- A is official `ststrnr`. Its largest 2024 class is code 18, open and
  special land, 325.76 km², 41 percent of classified area. That class is
  the same area as published candidate B's `vegetated_open`.
- Published candidate B stays in the file as a comparison. It is not the
  working grouping. Wald, Park, Landwirtschaft, Brache, and Stadtplatz
  each have 2024 area (169.47, 34.51, 34.85, 25.26, and 0.47 km²) and
  were pooled there. The areas named here are the 2024 `typ` areas,
  not the proposal-B group totals.
- B splits those surfaces and keeps boundary types in their own groups:
  shed yards, heterogeneous infill, Kerngebiet, utilities, camping, and
  uncovered sports. Water and construction are exclusion bins. Public
  uses stay one heterogeneous bin. Eleven of twelve comparison groups
  clear the support floor. Stadtplatz is the one that does not, at
  0.47 km², and it stays its own group. Camping is a sparse boundary
  group. Construction is a sparse exclusion group. Airport (`typ` 93)
  is 4.44 km² in 2020 and absent afterwards.
- C pools B back into seven bins, including one open-land bin, so the
  loss of the split is visible. C is a sensitivity.

In 2024, 58 percent of covered inside cells contain more than one `typ`,
53 percent more than one proposal-B group, and 51 percent more than one
`ststrnr`. At 1 km, 964 of 1,008 blocks that meet those cells are mixed
under proposal B. A majority label would still hide the mixture.
Mapped transport touches 8,730 cells and dominates 80 percent of the
covered cell in only 671 of them. Forest dominates at that threshold in
15,166 of 22,766 touched cells.

Proposal B is the decided grouping for a later description. Official
06.07 stays the external reference. The choice did not use a model error.
Stadtplatz, camping, and construction stay in the groups fixed before
the area tables, including where those groups are sparse.
