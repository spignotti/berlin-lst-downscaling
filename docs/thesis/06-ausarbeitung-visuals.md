# Visuals & Präsentationen

> Migriert aus Notion am 2026-10-28 (Seite `b99a5ea6ae9d4d7c8f91b978d81a3d4a`, zuletzt bearbeitet 2026-07-21). Text unverändert übernommen.

## Zweck

Arbeitsbereich für wiederverwendbare Präsentationskonzepte, technische Diagramme, Infografiken und Datenvisualisierungen der Masterarbeit. Inhaltliche Aussagen bleiben an Project Memory, WB1–WB5 und den jeweiligen Ergebnisdokumenten gebunden.

## Visueller Grundstil

- Editorial und minimal: weißer Grund, Anthrazit als Textfarbe, Blau als Standardakzent.
- Eine Kernaussage pro Folie oder Grafik. Großzügiger Weißraum, klare Hierarchie, keine dekorativen Elemente ohne Funktion.
- Karten, Daten und methodische Logik stehen vor generischen Illustrationen.
- Konsistente Typografie, Achsen, Legenden, Kartenrahmen und Quellenzeilen.
- Semantik: Grün = valide oder erledigt, Orange = offen, Rot = Risiko oder Fehler.

## Präsentationskonzept v0.1

**Zielgruppe:** Betreuung und fachlich interessierte Personen.

**Erzählung:** Problem und Forschungslücke → Daten und methodischer Ansatz → aktueller Umsetzungsstand → nächste Entscheidungen und Schritte.

### Standardstruktur für Status- oder Betreuungsdeck

1. Ausgangslage, Ziel und Forschungsfrage
2. Forschungslücke und wissenschaftlicher Beitrag
3. Datenbasis und räumlich-zeitliche Auflösung
4. End-to-End-Ablauf von Quellen bis Trainingsdaten
5. Ablationsdesign und Hypothesen
6. Aktueller Daten- und Pipeline-Status
7. Validierungslogik und bekannte Limitationen
8. Nächste Schritte und Entscheidungsbedarf

## Diagramm- und Infografik-Backlog

- [x] **End-to-End-Pipeline:** Landsat, Sentinel-2, Sekundärdaten und ECOSTRESS bis zum 10-m-LST-Produkt.
- [x] **Phasenmodell:** WB2 Daten-Pipeline → WB3 Training und Ablation → WB5 Validierung → WB4 Analyse und Implikationen → Schreiben.
- [x] **Ablationsleiter:** Spektral → +Morphologie → +Verschattung und Sonnengeometrie → +Meteorologie → +Thermal-aware Loss.
- [ ] **Pseudo-Pair-Prinzip:** Training auf künstlich vergröberten Skalen, Anwendung auf 100 m → 10 m.
- [ ] **Validierungsdreieck:** Scale Consistency, SSIM und temporale CV, ergänzt durch ECOSTRESS für 2018.

## Technische Diagramme v0.1

### End-to-End-Pipeline

```mermaid
flowchart LR
    LS["Landsat 8/9<br>100 m LST-Label"]
    S2["Sentinel-2<br>10 m Spektralfeatures"]
    SEC["Sekundärdaten<br>Morphologie, Schatten, ERA5-Land"]
    ECO["ECOSTRESS<br>70 m Validierung, 2018"]
    ARD["ARD und szenenbezogener Datenwürfel<br>EPSG:25833"]
    FE["Feature Engineering<br>QA, Indizes, Splits, Normalisierung"]
    TRAIN["Training und Ablation<br>fester 2D-U-Net + RF-Baseline"]
    PROD["10 m LST-Produkt<br>Berlin, Warmsaison"]
    VAL["Validierung<br>SSIM, Scale Consistency, temporale CV"]

    LS --> ARD
    S2 --> ARD
    SEC --> ARD
    ARD --> FE --> TRAIN --> PROD --> VAL
    ECO --> VAL
```

### Phasenmodell

```mermaid
flowchart LR
    WB2["WB2<br>Daten-Pipeline und Preprocessing"] --> WB3["WB3<br>Training und fünf Ablationsstufen"]
    WB3 --> WB5["WB5<br>Validierung und Ergebnisqualität"]
    WB5 --> WB4["WB4<br>Analyse und Implikationen"]
    WB4 --> WRITE["Thesis<br>Methodik, Ergebnisse, Diskussion"]
    WB2 -. "Datenvertrag" .-> WB5
    WB3 -. "10 m LST-Produkte" .-> WB4
```

### Ablationsleiter

```mermaid
flowchart LR
    S1["1. Spektral<br>Sentinel-2-Indizes"] --> S2["2. + Morphologie<br>LoD2, Vegetationshöhe, Versiegelung, SVF"]
    S2 --> S3["3. + Verschattung<br>Gebäude, Vegetation, Sonnengeometrie"]
    S3 --> S4["4. + Meteorologie<br>ERA5-Land, antezedente Witterung, DoY"]
    S4 --> S5["5. + Thermal-aware Loss"]
    S1 -. "gleicher deterministischer 2D-U-Net-Backbone" .-> S5
```

## Arbeitsmodus

- Mermaid dient als versionierbare technische Quelle für Notion und Repository.
- Präsentations- und Thesis-Grafiken werden als eigenständige, lesbare Visuals umgesetzt.
- Jedes Visual erhält Titel, Kernaussage, Daten- oder Quellenzeile und Versionsstand.
- Gestaltung wird anhand konkreter Artefakte iterativ weiterentwickelt.
