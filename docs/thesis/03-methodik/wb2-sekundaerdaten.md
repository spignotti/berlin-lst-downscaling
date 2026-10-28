# WB2 Sekundärdaten

> Migriert aus Notion am 2026-10-28 (Seite `61879b5f5bc641e88bc7eb56724972bc`, zuletzt bearbeitet 2026-07-15). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet; Abschnitte zu Tooling und archivierten Arbeitsnotizen entfernt.

> **Historischer Stand (Juli/August 2026), abgelöst durch features v6 / ARD v10, siehe [Datenbasis & Feature Engineering](datenbasis-feature-engineering.md).**

> 🧱 Lebendes Ergebnisdokument für Beschaffung, Aufbereitung und zeitliche Zuordnung der Sekundärdaten in den Ablationsstufen 2 bis 4. Implementierungsdetails bleiben im Repository.

## Status

Research und Cloud-Workflow am 14.07.2026 konsolidiert. Frühere unvollständige Produkte unter `_raw/secondary/` und `ard/static/` wurden verworfen; die Sekundärpipeline wird sauber neu aufgebaut. Die Implementierung erfolgt datensatzweise mit lokalen Fixtures, Cloud-Smoke-Test und anschließendem Voll-Run.

## Verbindliche Korrekturen aus dem Review

- **ERA5-Land bleibt die primäre meteorologische Eingangsquelle.** DWD ist Validierung und Sanity-Check, nicht Teil der zwölf Modellkanäle. Die Einordnung von ERA5-Temperatur als bloßes Backup und DWD als dynamischer Modellinput wird nicht übernommen.
- **Zeitliche Semantik und Download-Kadenz sind getrennt.** ERA5-Land ist dynamisch, wird aber gebündelt nach Jahr und Monat heruntergeladen. Pro Szene werden daraus die benötigten Features abgeleitet.
- **LoD2-Vintages propagieren.** Wenn mehrere LoD2-Stände genutzt werden, müssen Gebäudeprodukte, DSM, Horizon, SVF und Schatten konsistent demselben Vintage-Mapping folgen.
- **DGM-Kacheln werden über die vollständige Projekt-AOI bestimmt.** Eine pauschale Begrenzung auf „urban Berlin“ ist unzulässig, wenn dadurch Terrain- oder Vegetationskontext im AOI fehlt.
- **xarray-spatial ist der bevorzugte SVF-Kandidat, aber noch keine ungeprüfte Festlegung.** API, numerische Plausibilität, Laufzeit und Dependency-Kompatibilität werden in einem Smoke-Test gegen eine kleine Referenz geprüft.
- **Die Linux-VM bestimmt die Produktionsumgebung.** Ein numba-Pin darf nicht allein aus einem lokalen macOS-x86\_64-Wheel-Problem abgeleitet werden.
- **Die 20-GB-VM-Disk ist ein realer Engpass.** DGM und LoD2 werden kachelweise verarbeitet und bereinigt oder die Disk wird bewusst vergrößert. Diese Entscheidung fällt im gemeinsamen Pipeline-Foundation-Task.

## Datenklassen

| Klasse | Quelle oder Produkt | Download | Verarbeitung |
| --- | --- | --- | --- |
| Statisch | DGM 1 m | Einmal je AOI-Kachel | Einmal |
| Quasi-statisch | Vegetationshöhe 2020 | Ein Snapshot | Einmal |
| Quasi-statisch | Versiegelung 2016 und 2021 | Einmal je Vintage | Einmal je Vintage |
| Quasi-statisch | LoD2 | Einmal je nutzbarem Vintage | Einmal je Vintage |
| Abgeleitet quasi-statisch | Gebäudehöhe, Höhen-Std, BCR, DSM, Horizon, SVF | Kein eigener Download | Je Geometrie-Vintage |
| Dynamisch | ERA5-Land t2m und SSRD | Gebündelt je Jahr und Monat | Je Szene |
| Dynamisch abgeleitet | Gebäude- und Vegetationsschatten | Kein eigener Download | Je Szene |
| Dynamisch abgeleitet | Day of Year und Sonnenstand | Kein Download | Je Szene |
| Validierung | DWD-Stationsdaten | Gebündelt über wetterdienst | Sanity-Check, kein Modellkanal |

## Quellen und bevorzugte Werkzeuge

| Quelle | Zugriff | Bevorzugter Stack | Ziel |
| --- | --- | --- | --- |
| Vegetationshöhe 2020 | Geoportal Berlin ATOM, einzelnes Archiv | requests, rasterio/GDAL | Vegetationshöhe 10 m und Vegetations-DSM |
| Versiegelung 2016/2021 | Geoportal Berlin ATOM | requests, rasterio/GDAL | Versiegelungsgrad 10 m je Vintage |
| LoD2 CityGML | Geoportal Berlin ATOM, gekachelt | Streaming-XML-Parser, Shapely, Rasterio | Gebäudehöhe, Höhen-Std, BCR, Gebäude-DSM |
| DGM 1 m | Geoportal Berlin ATOM, XYZ-ZIPs | GDAL XYZ, gdalbuildvrt, gdalwarp; Fallback gdal\_grid | Boden-DEM 10 m |
| ERA5-Land | CDS API | cdsapi, GRIB, cfgrib/xarray | t2m, SSRD und 3-Tage-Mittel je Szene |
| DWD | DWD Open Data | wetterdienst | Validierung und Sanity-Check |
| SVF | Aus DSM | xarray-spatial nach Smoke-Test | SVF 10 m je Vintage |
| Schatten | Aus Horizon und Sonnenstand | Custom numpy/numba | Gebäude- und Vegetationsschatten je Szene |

## Gestaffelter Ablauf

1. Gemeinsamen Secondary-Foundation-Layer für Config, Pfade, Ledger, Downloads, COG-Vertrag und QA festlegen.
2. Vegetationshöhe und Versiegelung als einfache statische Quellen implementieren und Cloudpfad validieren.
3. ERA5-Land gebündelt abrufen und die Szenen-Zuordnung prüfen.
4. LoD2 und DGM vollständig für die Projekt-AOI verarbeiten.
5. DSM und getrennte Gebäude-/Vegetationshorizonte erzeugen.
6. SVF gegen Referenzfälle validieren.
7. Gebäude- und Vegetationsschatten je Szene erzeugen.
8. Dynamische Meteorologie und DoY je Szene ausgeben.
9. Gesamten Secondary-Stack mit einem statischen Ausschnitt und einer Szene smoke-testen, anschließend Voll-Run.

## Bucket-Layout

```txt
_raw/secondary/{source}/{vintage_or_period}/
_staging/secondary/{source}/{run_id}/
ard/static/morphology/{source}/{vintage}/
ard/static/dsm/{vintage}/
ard/static/horizon/{component}/{vintage}/
ard/static/svf/{vintage}/
ard/dynamic/shadow/{year}/{scene_id}/
ard/dynamic/meteorology/{year}/{scene_id}/
qa/secondary/{run_id}/
```

## Qualitätsregeln

- Alle Rasterprodukte müssen exakt dem kanonischen Grid entsprechen.
- Kontinuierliche Größen werden sinnvoll aggregiert; kategoriale und binäre Layer nutzen nearest neighbor.
- Jeder Voll-Run wird gegen Soll-Kacheln, Soll-Vintages und Soll-Szenen geprüft.
- Kein Produkt gilt als fertig, wenn die räumliche Abdeckung nur formal, aber nicht inhaltlich plausibel ist.
- SSRD wird entsprechend der verifizierten ERA5-Land-Akkumulationssemantik in W/m² umgerechnet.
- SVF und Schatten werden getrennt für Gebäude und Vegetation plausibilisiert.

## Konsolidierungsquellen

- Sekundärdaten Cloud-Workflow (Notion-Archiv)
- Sekundärdaten Research Findings (Notion-Archiv)
