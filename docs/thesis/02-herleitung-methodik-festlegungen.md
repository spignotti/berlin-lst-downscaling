# Methodik-Festlegungen: U-Net-Backbone, Ablationsdesign und Scope

> Migriert aus Notion am 2026-10-28 (Seite `381831dfd31c4560b8ef2c32564e93f6`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.

Festgehaltene Methodik- und Scope-Entscheidungen aus den Arbeitssessions (Stand 03.06.2026). Ergänzt den explorativen Literaturscan um konkrete Festlegungen: was wir machen und was wir bewusst nicht machen. Speist direkt WB3 (Modellentwicklung und Ablation).

## Modell-Setup

- Fester, deterministischer 2D-U-Net-Backbone über alle Ablationsstufen hinweg. Die Architektur bleibt konstant, getauscht werden nur die Input-Kanäle (und in Stufe 5 das Trainingsziel bzw. der Loss).
- Keine eigene Architektur-Entwicklung. Bausteine aus der Literatur werden auf Konfigurationsebene übernommen (Feature-Engineering der Inputs), nicht als neue Netz-Architektur.
- Referenz für den Backbone: MRU-Net-Stil (U-Net mit Residual-Mapping, strided conv statt Pooling, transpose conv im Decoder).
- Diffusionsmodelle (PGDM, DiffScaler, LDM) sind nicht der Hauptweg. Höchstens optionaler, separater Vergleich am Ende, falls Zeit bleibt.

## Ablationsdesign

- Kumulativ, vorwärts-schrittweise: Spektral → + Morphologie und Oberflächengeometrie → + Verschattung und Sonnengeometrie → + Meteorologie → + Loss.
- Morphologie und Oberflächengeometrie umfasst Gebäudehöhe, Sky View Factor und Dichte aus LoD2, ergänzt um Vegetationshöhe aus einem Canopy Height Model (CHM) und das digitale Geländemodell (DEM, auch zur Normierung der Gebäudehöhe als DSM minus DEM). Fraktionale Vegetationsbedeckung bleibt im spektralen Block, da sie aus dem NDVI abgeleitet ist.
- Verschattung und Sonnengeometrie nutzt Schattenmasken aus LoD2 und CHM mit dem Sonnenstand zum Überflug (gegen 10:00 bis 10:30). Die Stufe ist physikalisch motiviert durch die feste Überflugzeit und bringt eine kontrollierte, aus Datum und Geometrie abgeleitete Zeitabhängigkeit, ohne Sequenzmodellierung.
- **Historischer Entscheidungsstand (03.06.2026):** Die konkrete Form der Schatten-Features war hier noch offen; inzwischen sind im V3-Stack zwei getrennte binäre Schattenkanäle für Gebäude und Vegetation implementiert (siehe aktuellen Repository-Datenvertrag). Kandidaten aus dem neuen Korpus: binäre Schattenmaske, fraktionaler Schattenanteil je Aggregationsbuffer (Xu et al. 2025: Schattenindex korreliert über den Großteil des Tages stärker mit Tg, Ta und Tmrt als das SVF, optimaler Radius 15 bis 20 m) sowie getrennte Kanäle für Gebäude- und Baumschatten (Lindberg & Grimmond 2011 und Park et al. 2021 begründen separate LoD2- und CHM-Masken, Park misst -1.09 bis 2.12 Prozent LST je Prozent Baumkronenvolumen). Porosität poröser Baumschatten gegen massive Gebäudeschatten ggf. unterscheiden (Park, Limitation). Entscheidung erst nach Sichtung der Detailzahlen im Notebook NB2.
- **Redundanz-Hypothese Verschattung gegen Morphologie und SVF:** Die Verschattungs-Stufe und das SVF dürfen nicht zweimal dasselbe testen. Das SVF integriert die Himmelssichtbarkeit über die Hemisphäre und korreliert stark mit dem Schlagschatten; eine Landsat-Studie leitet das SVF sogar direkt aus Schattendetektion ab (R² 0.78 gegen LiDAR). Genau deshalb wird der Schatten als richtungs- und zeitabhängiger, pro Szene aus dem Sonnenstand berechneter Wurf modelliert und nicht als statisches Jahresmittel, das auf das SVF zurückfiele. Geprüft wird, ob dieser dynamische Schatten einen Beitrag über das statische SVF plus Gebäude- und Vegetationshöhe hinaus liefert oder dazu redundant ist. Beide Ausgänge sind valide Befunde.
- **Schlanke, vergleichbare Stufen:** Damit die eigentlich interessante Aussage (ist die Sonneneinstrahlung redundant zur Morphologie) interpretierbar bleibt, wird die Morphologie-Stufe bewusst kompakt gehalten und nicht in viele Einzel-Layer zersplittert. Jede Stufe fügt eine klar umrissene, inhaltlich zusammengehörige Feature-Gruppe hinzu, sodass der Sprung zwischen den Stufen vergleichbar bleibt und der Beitrag sauber einer Gruppe zugeordnet werden kann. Auf die Kollinearität zwischen Schatten, SVF und Höhe ist bei der RF-Baseline und der SHAP-Interpretation zu achten.
- Ziel ist das Screening des **Beitrags je Layer** (welche Features verbessern das Downscaling), nicht die optimale Integration jedes einzelnen Features.
- Faire, bewusst einfache Integration jedes Layers, damit die Stufen vergleichbar bleiben. Keine pro-Layer-Optimierung der Einbindung.
- Optional Leave-one-out auf der vollen Konfiguration, um Einzelbeiträge zu prüfen. Kein vollständiges faktorielles Design (Kombinations-Explosion wird vermieden).
- Nach Abschluss der kumulativen Hauptablation und mit finaler Config: gezielte Zusatzruns (z. B. Baseline + eine Featuregruppe, Leave-one-out) nur bei klarer Hypothese aus den Ergebnissen (Redundanz, Interaktionen, unklare Einzelbeiträge). Kein faktorielles Design, keine Kombinationssuche ohne Ergebnisbezug.
- Stufe 5 (Loss) ist orthogonal: gleiche Modellarchitektur und Inputs wie Stufe 4; das ursprüngliche masked-L1-Ziel wird um SSIM- und Gradienten-Terme ergänzt. Der Checkpoint wird weiterhin nach maskierter 100-m-MAE gewählt (Implementierungsstand 08.10.2026).

## Forschungslogik

- Erwartung: Das Modell wird mit sinnvollen Layern tendenziell genauer, aber eine monotone Verbesserung ist nicht garantiert.
- Null-Ergebnisse sind valide Befunde. Ein Layer, der nichts beiträgt, ist ein Ergebnis, kein Misserfolg.
- Framing als Beitrags-Studie (welche Faktoren tragen bei), nicht als Beleg für einen Skalen-Mechanismus.
- Die Layer, die sich als nützlich erweisen, werden später vertieft.

## Umgang mit der Zeit

- Die ursprünglich geplante eigene Temporal-Stufe entfällt. Aggregierte oder statische Werte der letzten Tage und Wochen sind antezedente Witterung und damit inhaltlich Teil der Meteorologie. Eine getrennte Temporal-Stufe würde fast dasselbe Signal testen und wäre im Nulleffekt nicht interpretierbar.
- Zeitliche Information läuft im umgesetzten V3-Featurestack über die acht ERA5-Land-Kanäle (einschließlich Witterung zum Aufnahmezeitpunkt und vorangegangener 72 Stunden); die Verschattungs-Stufe integriert den Szenen-Sonnenstand in die berechneten Schattenmasken. Day-of-Year und Sonnengeometrie werden laut Datenvertrag als Metadaten geführt, nicht als eigenständige V3-Modellkanäle.
- Keine echte 3D- oder Sequenz-Verarbeitung (kein 3D-CNN, kein ConvLSTM). Der Backbone bleibt 2D.

## Abgrenzung (bewusst nicht Teil der Arbeit)

- **Skalenabhängigkeit der Einflussfaktoren:** Wir prüfen nicht, ob nicht-spektrale Features beim Sprung 100m→10m stärker wirken als bei gröberen Skalierungen (z. B. 1km→100m). Der Multi-Scale-Vergleich gehört in den Ausblick.
- **Temporale Sequenzmodellierung:** Es gibt keine eigene Temporal-Stufe und keine 3D- bzw. Sequenz-Verarbeitung (3D-CNN, ConvLSTM, temporale Attention). Zeitliche Information läuft über die Meteorologie. Ob eine Sequenzverarbeitung Mehrwert brächte, bleibt eigene Forschungsfrage und Ausblick.
- Keine Architektur-Suche und keine Optimierung des Input-Kanal-Designs.

## Bezug zur Code Review

- Repos werden nach Feature-Engineering-Techniken durchsucht, nicht nach Architektur. Welches Repo für welchen Layer relevant ist, steht in der Code-Review-Task: Code-Review: Beispielrepo recherchieren — Masterarbeit (Notion-Archiv).

## Benennung und Deliverable

- Das Endprodukt ist eine konkrete Konfiguration plus finales Setup, keine neue Architektur. Ein Name benennt das Produkt bzw. Setup, nicht eine erfundene Netz-Klasse.
- Geplantes Deliverable: veröffentlichtes Modell (Hugging Face) zur Erzeugung neuer 10m-LST-Zeitreihen für Berlin.

## Noch in die Aufgabenstellung übernehmen

- [x] "Beitrag testen, nicht Optimalintegration" plus Future-Work-Abgrenzung
- [x] Contribution-Framing statt Skalen-Mechanismus
- [x] Null-Ergebnisse sind valide
- [x] Fester 2D-U-Net-Backbone, Diffusion optional und separat

## Aktualisierung 08.10.2026: aus GitHub-Evidenz übernommene Entscheidungen

Dieser Abschnitt dokumentiert **den realisierten Forschungsstand** und ersetzt im Konfliktfall den frühen Planungsstand vom 03.06.2026. Normative technische Interfaces bleiben im Repository; wissenschaftliche Ergebnisse/Interpretationen in [Notion-Seite 3f335645](https://app.notion.com/p/3f3356451f66812bb823fd5ecd7f6fa6) (Notion-Archiv).

### Kelvin-Residual statt absoluter Initialvorhersage

- Der erste begrenzte absolute Output-Probe erzielte auf Validation 294,84 K MAE und verfehlte damit den gleichartig ausgewerteten naiven Prior (1,5261 K). Die Diagnose führte dies in der Ausgangssituation auf die uninitialisierte absolute Kelvin-Skala (Prediction nahe 0 K) zurück. Dies ist ein Befund zum getesteten Setup, nicht zur Unmöglichkeit absoluter LST-Modelle.
- Methodische Festlegung danach: **Vorhersage als additive Kelvin-Korrektur zum aus Grob-LST rekonstruierten Prior**; nullinitialisierter Korrektur-Head. Der begrenzte Residual-Recovery-Run erzielte 1,1929 K Validierungs-MAE und erhielt ein vorregistriertes GO. Daraufhin wurde die Repräsentation für Stage 1–5 fixiert. [Probe- und Debug-Evidenz](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-debug-results.md).

### Konstante Vergleichsbedingungen

- Fester 2D-U-Net-Backbone, Tiefe 4, Breite 32, Seed 0, Lernrate 1e-3, zunächst masked L1. Die Stage-1-Vollversion und Ablationsstufen 2–5 verwenden denselben Residual-Prior, denselben temporalen Split, dieselbe Patch-Geometrie und dieselbe Checkpoint-Metrik; keine Zwischen-Retunes der Architektur oder Lernrate.
- Hauptablation: Stage 1 Spektral (10 Modellkanäle), Stage 2 + Morphologie (18), Stage 3 + Gebäudeschatten und Vegetationsschatten (20), Stage 4 + acht ERA5-Kanäle (28), Stage 5 identischer Input mit masked L1 + SSIM-/Gradientenstrafen (28). Schatten- und ERA5-Isolationen sind **zusätzliche Nicht-Leiter-Runs**, keine Ablationsstufen. [Ablationskonfigurationen](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-stage-configs.md).
- Pseudo-Pair-Methode weiterhin entscheidend: Die feine Prognose wird nur nach exaktem 10×10-Mittel auf nativen 100-m-Landsat-Zellen bewertet. Der Training-Prior ist eine Degradation derselben Landsat-Szene auf 1.000 m, nicht ein extern unabhängig gemessener Zustand. Der 2025-Test dient einmal je Lauf der Bewertung nach 2024-Validation-Selektion. Technischer Vertrag: [Pseudo-Pair und Tensor](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/pseudo-pair-tensor-contract.md).
- Die resultierenden MAE-Werte beantworten eine **scale-consistency**-Frage auf 100 m. Sie belegen ausdrücklich **nicht** native 10-m-Genauigkeit oder unabhängige räumliche Generalisierung. Die Wiederholung räumlicher Fensteranker in verschiedenen Jahren und die Ein-Seed-Studie begrenzen die Interpretation.
- **Reproduzierbarkeit:** W&B dokumentiert operative Runs/Metriken, GCS die publizierte Evidenz samt Checkpoint, GitHub die eingefrorenen nachprüfbaren Protokolle; Notion führt ausschließlich die wissenschaftliche Begründung und Interpretationen.

### Abgrenzung zur späteren Thesis-Diskussion

Der Ergebnisblock [Notion-Seite 3f335645](https://app.notion.com/p/3f3356451f66812bb823fd5ecd7f6fa6) (Notion-Archiv) unterscheidet Beobachtung und Interpretation. Insbesondere ist die gegenüber dem Naive-Prior große Stage-1-Verbesserung **nicht** der Beitrag der zusätzlichen urbanen Kontextvariablen. Für diese zählt ausschließlich der inkrementelle Vergleich gegen Stage 1, unter den dargelegten Einschränkungen.
