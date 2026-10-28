# Design

> Migriert aus Notion am 2026-10-28 (Seite `c017d294d9484210b3f78fe1e1d23a6b`, zuletzt bearbeitet 2026-07-23). Text unverändert übernommen.

Design-Sprache der Masterarbeit. Verbindlich für alle visuellen Artefakte: Decks, Diagramme, Infografiken, Datenvisualisierungen. Löst den Abschnitt Visueller Grundstil in [Visuals & Präsentationen](06-ausarbeitung-visuals.md) ab. Deckstruktur, Diagramm-Backlog und Mermaid-Quellen bleiben dort.

## Kontext & Zielgruppe

- Thema: urbanes LST-Downscaling Berlin, U-Net, TorchGeo, GCP.
- Zielgruppe: Betreuung und fachlich interessierte Personen.
- Haltung: editorial und minimal. Eine Kernaussage pro Folie oder Grafik. Karten, Daten und methodische Logik vor generischen Illustrationen.

## Farben

Rollen und exakte Werte. Quelle: Radix Colors (Slate, Indigo, Green, Orange, Red), auf Deck-Kontrast getrimmt.

| Rolle | Hex | Verwendung |
| --- | --- | --- |
| Text primär | #14181F | Fließtext, Überschriften |
| Text sekundär | #5B6370 | Labels, Captions, Achsen |
| Canvas | #FFFFFF | Folienhintergrund |
| Fläche weich | #F6F7F9 | seltene Neutralfläche, Tabellen-Zebra |
| Border | #DFE3E8 | Trennlinien, Rahmen |
| Akzent solid | #3358D4 | Betonung, ein dominantes Element, Linien |
| Akzent dunkel | #273EA5 | Hover, Text auf Tint |
| Akzent-Tint | #E1E9FF | nur semantische Statusfläche |
| Grün valide | #2F9E63 | valide, erledigt |
| Orange offen | #C86A1F | offen, in Arbeit |
| Rot Risiko | #D93A3F | Risiko, Fehler, Limitation |

Regeln:

- Farbe trägt Bedeutung oder ist neutral. Grün, Orange und Rot nur semantisch.
- Akzent solid statt blasser Tints. Tint nur für Statusflächen.
- 60-30-10: rund 60 % Canvas, 30 % Neutral und Text, 10 % Akzent.
- Kontrast mindestens WCAG AA.

Diagramm-Paletten:

- Kategorisch (sparsam, maximal 4 bis 5): #3358D4, #14181F, #5B6370, semantische Farben nur bei semantischer Bedeutung.
- Sequenziell: perzeptuell uniform und farbenblind-sicher (viridis oder cividis).
- LST und thermisch: magma oder inferno, konsistente Skala und Legende über alle Karten.

## Typografie

- Sans (UI, Folien, Fließtext): Inter, alternativ System-Sans.
- Mono (Code, Werte, Tabellen): JetBrains Mono.
- Skala (Deck 1280x720): Eyebrow-Label 14, Folientitel 40 bis 44, Untertitel 24, Body 18 bis 20, Caption 13 bis 14.
- Gewichte: Titel 600, Body 400, Labels 500 in Versalien mit leichtem Letter-Spacing.

## Layout

- Canvas: 16:9, 1280x720.
- Grid: 12 Spalten, großzügige Außenränder ab 64 px, konsistente Rinnen.
- Spacing-Scale: 4, 8, 12, 16, 24, 32, 48, 64.
- Radius: 8 px, sparsam. Struktur über Linien und Weißraum statt Boxen.
- Volle Fläche nutzen, dominantes Element nahe der optischen Mitte.

## Bildsprache

- Echte Fach-Visuals vor Deko: Berlin-LST-Karten, Plots, Pipeline- und Phasendiagramme.
- Jedes Visual: Titel, Kernaussage, Daten- oder Quellenzeile, Versionsstand.
- Konsistente Achsen, Legenden, Kartenrahmen, Nordpfeil und Maßstab bei Karten.
- Icons: Lucide, ein Stroke-Weight um 1.75 px, Größe 20 bis 24 px, Farbe Anthrazit oder Akzent, funktional statt dekorativ.

## Anti-Slop-Bannliste

Global:

- Keine dekorativen Farbflächen, keine Regenbogen-Kategorien.
- Keine leeren Deko-Elemente oder Platzhalter-Kästen.
- Keine Card-Wall, Layout variieren.
- Keine blassen Tint-Flächen als Hauptfarbe.
- Icons nicht auf jede Karte.

Projektspezifisch (aus dem Betreuungsdeck-Teardown):

- Kein leerer Gradient-Kasten auf der Titelfolie.
- Keine Regenbogen-Timeline, keine off-palette Brauntöne.
- Keine gequetschten Diagrammspalten, keine erzwungenen Wortumbrüche.
- Kein leeres unteres Folien-Drittel.

## Artefakt-Defaults

- Deck-Format: HTML, Export als PDF. PPTX nur bei editierbarer Übergabe.
- Deck-Narrativ: Problem und Forschungslücke, dann Daten und Methode, dann Umsetzungsstand, dann nächste Entscheidungen.
- Standardstruktur Status- und Betreuungsdeck:
  1. Ausgangslage, Ziel, Forschungsfrage
  2. Forschungslücke und Beitrag
  3. Datenbasis und Auflösung
  4. End-to-End-Ablauf
  5. Ablationsdesign und Hypothesen
  6. Daten- und Pipeline-Status
  7. Validierungslogik und Limitationen
  8. Nächste Schritte und Entscheidungsbedarf
- Pro Folie: Eyebrow-Label, eine Kernaussage, Takeaway-Fußzeile, Seitenzahl.

## Änderungslog

- v0.1 (2026-07-22): Design-Sprache aus Visuals & Präsentationen extrahiert und formalisiert. Palette neu gesetzt: Indigo-Akzent #3358D4 solid, semantische Sekundärfarben, Diagramm-Paletten definiert. Ersetzt die willkürliche Ausgangsfarbwahl.
