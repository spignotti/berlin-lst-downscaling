# 02 · Herleitung & Forschungsentscheidungen

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668124be48d7c9ca37548d`, zuletzt bearbeitet 2026-10-23). Text unverändert übernommen.

Teil 02 leitet die Methodik der Arbeit aus dem geprüften Forschungsstand und den Befunden der Projektarbeit ab. Die Argumentation folgt der Kette Anforderungen aus der Literatur → Designkonsequenzen → Forschungsfragen und Entscheidungen → offene Punkte und Limitationen. Technische Details der Umsetzung stehen in [03 · Methodik – aktueller Stand](03-methodik/README.md), die Quellensynthesen in [01 · Literatur & Forschungsstand](01-literatur/README.md) (Literatur I–VI). Maßgebend für den zulässigen Umfang ist die [Aufgabenstellung](00-aufgabenstellung.md).

> 📌 **Stand 23.10.2026.** Maßgeblich ist der Code-Stand `main` (f1a57dc) mit Feature-Release features/v6 (22 Kanäle). Modellkennzahlen früherer Datenstände sind wegen der korrigierten Sentinel-2-Reflektanz und der korrigierten LoD2-Gebäudedaten nicht mehr gültig und werden auf dem neuen Datenstand vollständig neu berechnet; Teil 02 enthält deshalb keine Ergebnisse. Statusangaben: **festgelegt** · **vorläufig** (Parameter gesetzt, Prüfung ausstehend) · **offen** (Entscheidung ausstehend) · **Limitation**.

## 1 Problem

Oberflächennahe städtische Wärmeinseln (SUHI) sind in Berlin räumlich stark heterogen. Zusammenhänge zwischen Landoberflächentemperatur (LST) und Landnutzung, Versiegelung, Gebäudestruktur und Vegetation sind für Berlin belegt (Dugord et al. 2014; Vulova & Kleinschmit 2019; Dong et al. 2022; Klopfer 2023). Die Effekte variieren nach Saison, Tageszeit und Aggregationsebene; die Befunde beruhen überwiegend auf Blöcken, Stadtstrukturtypen oder 30–100-m-Rastern. Sie belegen Assoziationen, aber keinen Zusatznutzen einzelner hochaufgelöster Merkmale in einem Downscaling-Modell.

Landsat 8/9 misst thermal nativ mit etwa 100 m; das Collection-2-Produkt liefert diese Information auf ein 30-m-Raster resampelt (USGS). Für Fragestellungen auf Block- und Straßenraumebene ist das zu grob. Sentinel-2 (10 m) und Berliner Geodaten bilden feine Oberflächenstrukturen ab, messen aber keine Temperatur. Downscaling verbindet beide Ebenen. Verfahren unterscheiden sich erheblich darin, welche Skalenannahmen sie treffen und auf welcher Referenzskala eine Verbesserung nachgewiesen wird (Li et al. 2019; Tang et al. 2021; Jamaluddin et al. 2021; Andriambololonaharisoamalala et al. 2025).

## 2 Anforderungen aus der Literatur

| Nr. | Anforderung | Evidenz | Designkonsequenz |
| --- | --- | --- | --- |
| A1 | Skalensprung um Faktor 10 im heterogenen Stadtraum | Grob kalibrierte Beziehungen gelten auf feineren Einheiten nur eingeschränkt (Li et al. 2019; Tang et al. 2021); räumlicher Kontext verbessert die Rekonstruktion (Tang et al. 2021; Nguyen et al. 2022) | Faltendes Netz mit Patch-Kontext; feine Führung aus einer einheitlichen 10-m-Quelle |
| A2 | Keine unabhängige thermale 10-m-Referenz | 10-m-Produkte werden auf Landsat-Testdaten, indirekt oder auf gröberer Skala bewertet (Jamaluddin et al. 2021; Bouaziz et al. 2025); synthetische Testgüte überträgt sich nicht automatisch auf absolut feinere Skalen (Nguyen et al. 2022) | Training auf Pseudo-Paaren; Bewertung auf der beobachteten 100-m-Skala; 10-m-Aussagen nur als Plausibilitätseinordnung |
| A3 | Physikalische Bindung an die beobachtete Skala | Ein Mittelwertzwang sichert nur die Aggregationsbedingung, nicht die Subpixelstruktur (Mollière et al. 2023; Literatur III); feinere Ausgaberaster sind keine native Sensorinformation (Xue et al. 2020; USGS) | Aggregationsverlust auf 100 m; dokumentierte Zielbildung und Maskierung; Prüfung der Aggregationsart |
| A4 | Isolierbarer Beitrag von Kontextgruppen und Trainingsziel | Einflüsse von 2D/3D-Morphologie, Vegetation, SVF, Verschattung und Witterung sind belegt (u. a. Scarano & Mancini 2017; Park et al. 2021; Luo et al. 2026; Su et al. 2026; Li, H. et al. 2018); eine kontrollierte Isolation dieser Faktoren fehlt studienübergreifend | Festgehaltener Backbone; Variation nur der Eingangsgruppen bzw. des Trainingsziels |
| A5 | Zeitliche und räumliche Übertragbarkeit | Validierung muss am Vorhersageziel und an der Abhängigkeitsstruktur ausgerichtet sein (Roberts et al. 2017); räumliche Validierung zeigt oft deutlich geringere Güte als zufällige (Ploton et al. 2020) | Jahresweiser Split; zusätzlicher räumlicher Holdout mit über alle Termine gruppierten Orten |
| A6 | Vergleich mit Referenzverfahren auf identischer Kohorte | Kein universell bester Downscaling-Algorithmus; RF übertrifft in einzelnen Vergleichen neuronale Netze (Li et al. 2019); Fehlerwerte verschiedener Studien sind wegen unterschiedlicher Referenzskalen nicht vergleichbar (Literatur II) | Naive Prior-Expansion und Random Forest auf identischer Maske und Metrik |

## 3 Forschungslücke und Beitrag

Pauschale Lückenbehauptungen sind durch den Forschungsstand nicht gedeckt. Neuronale Downscaling-Modelle haben bereits 3D-Stadtinformation wie den Sky-View-Factor genutzt (Gkolemi et al. 2023), 10-m-LST wurde aus Landsat und Sentinel-2 geschätzt (Jamaluddin et al. 2021), Morphologieparameter verbessern RF-Downscaling (Li, X. et al. 2022), und meteorologische Antriebe sind als Eingang selten, aber nicht neu (Wu et al. 2022; Kustura et al. 2025). Diffusions- und Foundation-Model-Ansätze für LST-Super-Resolution existieren (Zhang, H. et al. 2026; Chen et al. 2026).

Belastbar ist eine enge, bedingte Lücke: Es fehlt eine kontrollierte Bestimmung des Zusatznutzens stadtkontextueller Prädiktorgruppen und eines strukturgewichteten Trainingsziels bei festgehaltenem Modell, expliziter Maskierung, nachvollziehbaren Referenzverfahren und offen benannten Grenzen von Skalenwechsel und Validierung.

Der Beitrag der Arbeit ist eine reproduzierbare, bedingte Ablationsstudie eines festgehaltenen residualen 2D-U-Net für Berlin mit naiver Prior-Expansion und Random Forest als Referenzen, detaillierten urbanen und meteorologischen Featuregruppen, einem Vergleichsarm mit gelernten Geodaten-Embeddings und einer Bewertung auf der beobachteten 100-m-Skala. Die Arbeit beansprucht keine neue Netzarchitektur, keine universelle Überlegenheit von Deep Learning und keine unabhängig nachgewiesene 10-m-Temperaturgenauigkeit.

## 4 Forschungsfrage, Teilfragen und Hypothesen

**Leitfrage.** Welchen Beitrag leisten urbane Kontextinformationen und thermalsensitives Training zum Downscaling satellitenbasierter Land Surface Temperature für Berlin, und wie lässt sich die Qualität erzeugter 10-m-Muster ohne unabhängige thermale 10-m-Ground-Truth bewerten?

| Teilfrage | Inhalt | Bezug |
| --- | --- | --- |
| F1 | Rekonstruiert das residuale U-Net die 100-m-LST besser als die naive Prior-Expansion und der Random Forest? | A6 |
| F2 | Welchen inkrementellen Beitrag leisten Morphologie, Verschattung und Meteorologie gegenüber einem spektralen Anker? | A4 |
| F3 | Tragen Morphologie und Verschattung bei bereits vorhandener meteorologischer Information zusätzlich bei, und sind sie zueinander redundant? | A4 |
| F4 | Verändert ein strukturgewichtetes Trainingsziel die Rekonstruktion bei identischen Eingängen? | A4 |
| F5 | Ersetzen oder ergänzen gelernte 10-m-Embeddings ohne Thermalinformation die handgebauten Kontextgruppen? | A4, A6 |
| F6 | Bleiben die Befunde auf räumlich getrennten Gebieten und über verschiedene Initialisierungen bestehen? | A5 |
| F7 | Mit welchen struktur- und stadtstrukturbezogenen Kriterien lässt sich die Qualität der 10-m-Muster ohne feine Referenz einordnen, und wo liegen die Grenzen dieser Einordnung? | A2, A3 |

**Hypothesen.**

- **H1:** Versiegelung, Gebäudestruktur, Vegetationsstruktur und zeitabhängige Verschattung enthalten über spektrale Merkmale hinaus Information für die LST-Rekonstruktion (Literatur I; Su et al. 2026; Li, X. et al. 2022).
- **H2:** Szenenbezogen berechneter Schatten liefert einen Beitrag über statischen Sky-View-Factor und Höhenmerkmale hinaus, oder er ist dazu redundant.
- **H3:** Szenenbezogene Witterung moduliert die Amplitude der innerstädtischen LST-Kontraste. Da der Grobprior das Szenenniveau bereits trägt, kann Meteorologie im residualen Design nur über die Wechselwirkung mit Oberflächenmerkmalen beitragen (vgl. Li, H. et al. 2018 zum Einfluss von Vorregen auf die LST-Versiegelungs-Beziehung).
- **H4:** Struktur- und Gradiententerme verändern das 100-m-Fehlerbild, ohne automatisch eine höhere Treue auf 10 m herzustellen (Dai et al. 2025; Literatur V).
- **H5:** Gelernte Embeddings aus Sentinel-1/-2 erreichen den Beitrag der handgebauten Kontextgruppen ganz oder teilweise; ein Zusatznutzen über die Handfeatures hinaus ist offen.

Eine monotone Verbesserung mit jeder Gruppe wird nicht vorausgesetzt. Nullbeiträge und Redundanzen sind gleichwertige Befunde. Die Arbeit ist als Beitragsstudie angelegt, nicht als Nachweis eines Skalenmechanismus.

## 5 Herleitung der Designentscheidungen

### 5.1 Untersuchungsgebiet, Zeitraum und Zielgröße

- **Berlin als Fallstudie (festgelegt).** Berlin vereint ausgeprägte Kontraste zwischen bebauten, grünen und wassergeprägten Flächen mit frei verfügbaren Umweltatlas-, LoD2- und Versiegelungsdaten (Dugord et al. 2014; Klopfer 2023). Daraus folgt die Eignung als datenreiche, morphologisch vielfältige Fallstudie, keine Repräsentativität für andere Städte. Bezugssystem EPSG:25833, 10-m-Raster exakt in das 100-m-Raster verschachtelt; diese technischen Festlegungen sind projektspezifisch.
- **Warmsaison Mai–September (festgelegt).** Dieselbe Abgrenzung nutzt eine Berliner SUHI-Analyse (Li, H. et al. 2018). Sommerliche Tages-LST ist räumlich stärker gespreizt als im Winter, und Prädiktorbeziehungen ändern sich saisonal bis zum Vorzeichenwechsel (Dong et al. 2022; Vulova & Kleinschmit 2019). Die Saisonwahl begründet den Scope, keine garantierte Stabilität der Beziehungen innerhalb der Saison. Zeitraum 2017–2025 mit Ziel, 2026 als Inferenzjahr.
- **Ziel: Landsat 8/9 Collection 2 Level 2 Surface Temperature auf 100 m (festgelegt).** Das 100-m-Ziel ist das Flächenmittel des 30-m-Produkts. Die native Thermalauflösung liegt bei 100 m (USGS; Xue et al. 2020); eine offizielle Empfehlung zur Reaggregation existiert nicht, und das Flächenmittel stellt die Sensor-PSF nicht wieder her. Die Zielbildung ist damit ein eigenes, reproduzierbares Rasterdesign.
- **QA-Maskierung (festgelegt).** Schnee gilt nicht als klar (QA\_PIXEL Bit 5), Wolken-Konfidenz ≥ 2 (medium/high, Bits 8–9) gilt als bewölkt. Die Bitsemantik ist durch USGS dokumentiert, die Schwelle ist eine eigene Filterentscheidung. Restkontamination durch Wolken und Wolkenschatten ist im ST-Produkt dokumentiert, aber nicht quantifiziert (USGS).
- **Szenenausschluss (festgelegt).** Die Aufnahmen vom 27.08.2022 sind wegen physikalisch unplausibel kalter LST ausgeschlossen. Ein Temperatur- oder Hitzefilter wird nicht angewendet; Hitzetag-Auswahlen wie bei Klopfer (2023) sind eine zulässige Abgrenzung, aber kein Gütekriterium für Downscaling. Landsat-Szenen mit weniger als 5 % Überdeckung der AOI werden ausgeschlossen, da sie nur Randstreifen beitragen.

### 5.2 Spektraldomäne und Datenqualität

- **Sentinel-2 als einzige Spektraldomäne (festgelegt).** Landsat liefert ausschließlich das thermische Ziel. Ein Wechsel zwischen Landsat-Reflektanz im Training und Sentinel-2 in der Anwendung würde einen Sensorbruch erzeugen.
- **Radiometrische Harmonisierung (festgelegt).** Ab Processing Baseline 04.00 tragen Sentinel-2-L2A-Daten einen BOA-Offset von −1000 DN. Die Diagnose aller Szenen zeigt eine scharfe Trennung: 2017–2021 Baselines 02.12/03.00 ohne Offset, ab 2022 Baselines 04.00–05.11 mit Offset. Der Offset wird je Szene vor der Skalierung korrigiert. Ohne Korrektur lägen Validierungs- und Testjahr vollständig, das Training nur teilweise im verschobenen Regime.
- **Räumliche Vollständigkeit (festgelegt).** Alle MGRS-Kacheln eines Aufnahmetags werden geladen, je Kachel harmonisiert und auf das Projektraster mosaikiert; eine Abdeckungsprüfung schließt fehlende Sentinel-2-Abdeckung innerhalb der AOI aus. Selektion und Datenaufbereitung verwenden dieselben Sentinel-2-Aufnahmen. Ohne Mosaik fehlte der Osten Berlins systematisch in allen Splits.
- **Zulässigkeit (festgelegt).** Eine 100-m-Zelle ist nur zulässig, wenn der Landsat-Wert und alle 100 zugehörigen 10-m-Featurepixel gültig sind. Eingänge werden nicht imputiert, die Skalierung wird nur aus dem Trainingszeitraum geschätzt. Zulässigkeitsquoten beziehen sich auf Zielzellen innerhalb der AOI.
- **Wolkenmaske in dichter Bebauung (offen).** Sen2Cor sieht Gebäudeschatten in SCL 2 und Wolkenschatten in SCL 3 vor (Louis et al. 2022); eine quantitative Verwechslungsrate in Städten ist nicht belegt. Die strikte Zulässigkeitsregel kann häufig maskierte Bereiche systematisch ausschließen. Gültigkeitsraten werden deshalb nach Stadtstruktur und Schattenlage ausgewertet.

### 5.3 Eingangsfeatures (features/v6, 22 Kanäle)

| Gruppe | Kanäle | Evidenz | Status |
| --- | --- | --- | --- |
| Spektral (10) | B02, B03, B04, B08, B11, B12, NDVI, NDWI (McFeeters), NDBI, Breitband-Albedo | Jamaluddin et al. 2021; Yuan, W. et al. 2025; Zusatzinformation von Indizes gegenüber Reflexion gering (Zhang, H. et al. 2026) | festgelegt |
| Morphologie (6) | Gebäudehöhe (Mittel), Überbauungsgrad, Vegetationshöhe (Mittel, flächengewichtet), Kronenbedeckungsgrad, Versiegelungsgrad, Sky-View-Factor | Klopfer 2023; Luo et al. 2026; Su et al. 2026; Li, X. et al. 2022; Scarano & Mancini 2017; Gkolemi et al. 2023 | festgelegt; Kronenschwelle und SVF-Radius vorläufig |
| Verschattung (2) | Gebäude- und Vegetationsschatten, binär je Szene aus dem Sonnenstand zum Überflug | Lindberg & Grimmond 2011; Park et al. 2021; Xu et al. 2025 | festgelegt |
| Meteorologie (4) | Lufttemperatur, Globalstrahlung, Windgeschwindigkeit (je Szene), Niederschlagssumme 0–72 h | Wu et al. 2022; Kustura et al. 2025; Li, H. et al. 2018; Muñoz-Sabater et al. 2021 | festgelegt; Kopplung als Ablationsschalter |

Herleitung der Änderungen gegenüber dem früheren 28-Kanal-Stand:

- **Gebäudedaten.** LoD2-Gebäude werden einschließlich Gebäudeteilen (BuildingPart) gelesen, Überbauungsanteile flächenexakt berechnet. Damit gehen auch kleine Gebäude und Gebäude ein, deren Höhe nur an Gebäudeteilen hinterlegt ist. Gebäudehöhe, Überbauung, Gebäude-DSM, Schatten und SVF beruhen auf dieser korrigierten Grundlage.
- **Maximum- und Streuungskanäle entfallen.** Mittlere und maximale Gebäudehöhe korrelieren im eigenen Datensatz mit r = 0,9998. Ein eigenständiger Beitrag maximaler Höhen ist in der Literatur nicht isoliert getestet (Luo et al. 2026 nutzt das Maximum nur in einem zusammengesetzten Indikator), Höhenstreuung ist nur auf Blockebene belegt. Die maximale Höhe bleibt intern für das Gebäude-DSM erhalten.
- **Kronenbedeckungsgrad statt maximaler Vegetationshöhe.** Baumbedeckung ist der dominante Prädiktor intraurbaner LST-Unterschiede, Baumhöhe zeigt eine kleinere zusätzliche Beziehung (Su et al. 2026, auf 100-m-Einheiten). Die Vegetationshöhe wird flächengewichtet gemittelt, der Kronenanteil als Anteil der Pixel über einer Höhenschwelle gebildet.
- **Vegetationsdatensatz.** Die Berliner Vegetationshöhen 2020 bleiben Quelle. Globale Produkte sind nicht nachweislich überlegen: Meta/WRI berichtet einen globalen MAE von 2,8 m ohne Berliner Validierung (Tolan et al. 2024), das ETH-Produkt setzt bebaute Flächen auf NoData (Lang et al. 2023). Der Stand 2020 wird auf alle Jahre fortgeschrieben (Limitation).
- **Versiegelung.** Wasserflächen sind aus der amtlichen Versiegelungskartierung ausgenommen; Wasser = 0 % ist daher eine eigene Kodierung über die Wasserpolygone der Stadtstrukturkarte. Straßenland trägt die Abschnittswerte der Erhebung 2014 und ist eine flächenhafte Schätzung, keine lokal gemessene 10-m-Versiegelung (Umweltatlas Berlin 2021).
- **Meteorologie reduziert und szenenkonstant.** ERA5-Land hat eine native Auflösung von etwa 9 km und keine explizite Stadtdarstellung (Muñoz-Sabater et al. 2021); urbane Stationsvergleiche zeigen systematische Biases und abgeschwächte Stadtmuster (Zou et al. 2022; Yu et al. 2026). Räumliche Variation innerhalb Berlins wäre daher Interpolation, kein Stadtsignal. Zudem können räumlich glatte Kovariaten als implizite Ortskodierung wirken (Ploton et al. 2020). Standard ist deshalb ein Szenenwert je Kanal, die bilineare Kopplung eine Ablationsvariante. VPD entfällt, da aus Lufttemperatur abgeleitet und eng an sie gekoppelt; die drei 24-h-Niederschlagsfenster sind innerhalb einer Szene praktisch konstant (EDA v2) und werden zu einer 72-h-Summe zusammengefasst; die antezedente Strahlung entfällt.
- **Keine weiteren Kontextgruppen.** Local Climate Zones und Stadtstrukturtypen sind aus denselben Größen abgeleitet und dienen als Evaluationsstrata; als Eingang würden sie die Stratifizierung zirkulär machen. Distanzen zu Wasser- und Grünflächen sind im Patch-Kontext über NDVI und NDWI sichtbar. Anthropogene Wärme, Bodenfeuchte und Bevölkerungsdichte sind thematisch plausibel, ihr Zusatznutzen auf 10–100 m ist nicht belegt. Tageszeit und Sonnengeometrie wirken nur über die Schattenmasken.

### 5.4 Modell und Trainingsprinzip

- **Fester 2D-U-Net mit Patch-Kontext (festgelegt).** U-Net-artige Encoder-Decoder sind für LST-Super-Resolution etabliert (Nguyen et al. 2022; Beber et al. 2025; Zhang, H. et al. 2026). Ein kontrollierter Vergleich mit RF oder MLP bei identischer Datenbasis, der den Backbone isoliert, liegt nicht vor. Die Konstanthaltung dient der Ablationsfrage: Eine Architektursuche würde Feature- und Architektureffekte vermischen. Ein Patch umfasst 160 × 160 Pixel auf 10 m, entsprechend 16 × 16 Zielzellen.
- **Residuale Kelvin-Korrektur zum Grobprior (festgelegt).** Additives Residuallernen gegenüber dem interpolierten Grobfeld ist direkt belegt und wird mit vereinfachtem Lernen bzw. schnellerer Konvergenz begründet (Nguyen et al. 2022; Zhang, H. et al. 2026). Der nullinitialisierte Korrektur-Head und die Überlegenheit gegenüber absolutem Lernen sind eigene Befunde: Eine absolute Kelvin-Ausgabe erwies sich unter begrenztem Budget als ungeeigneter Startpunkt, die residuale Variante bestand einen vorab festgelegten Recovery-Test.
- **Pseudo-Paare 1.000 → 100 m (festgelegt).** Das 100-m-Ziel wird in exakten 10 × 10-Blöcken zu einem 1.000-m-Prior gemittelt und expandiert; die 10-m-Ausgabe wird exakt auf 100 m aggregiert, erst dort wird der Fehler berechnet. In der Anwendung ersetzt die beobachtete 100-m-LST den Prior. Synthetische Degradationspaare sind verbreitet (Tang et al. 2021; Nguyen et al. 2022; Dai et al. 2025).
- **Skalenübertragung 100 → 10 m (Limitation).** Ein analoger Transfer (4 km → 1 km trainiert, 1 km → 250 m angewendet) erreichte gegen eine externe ASTER-Referenz nur das Niveau von ATPRK und bikubischer Interpolation; die Autoren bewerten die Skaleninvarianz als unzureichend (Nguyen et al. 2022). Eine bessere Regression auf Zielskala ist kein hinreichender Transfernachweis (Li, N. et al. 2022). Die 10-m-Ausgabe ist daher ein experimenteller Transfer mit eigenem Validierungsbedarf.
- **Aggregationsverlust und Aggregationsart (festgelegt; Prüfung offen).** Der Fehler wird nur auf gültigen 100-m-Zellen berechnet. Das arithmetische Kelvin-Mittel nähert die radiometrische Mischung an; ob eine T⁴-Aggregation auf Berliner Daten relevant abweicht, wird geprüft (Literatur III). Ein harter Constraint, der die Ausgabe exakt auf die Beobachtung projiziert, ist nur als Sensitivitätstest am finalen Produkt vorgesehen und kein Genauigkeitsnachweis.
- **Trainingsziel (festgelegt; Gewichte vorläufig).** Basisziel ist die maskierte L1 auf 100 m. Die thermalsensitive Variante ergänzt bei identischen Eingängen 0,1·(1 − SSIM) und 0,1·Gradientenfehler auf 100 m mit 7 × 7-Fenster; nur vollständig gültige Fenster gehen ein, Patches ohne gültiges SSIM-Fenster werden aus dem SSIM-Mittel ausgeschlossen und bleiben im L1-Term. Kombinierte Pixel- und Strukturverluste sind als Prinzip belegt, die Unterschiede zwischen Gewichtungen sind dort klein und skalenabhängig (Dai et al. 2025). Für Gewichte, Fenstergröße und Gradiententerm gibt es keinen Beleg. Verglichen wird das zusammengesetzte Ziel mit der L1, keine Einzelterme.

### 5.5 Datenaufteilung, Patchauswahl und Übertragbarkeit

- **Zeitlicher Split (festgelegt).** Training 2017–2023, Validierung 2024, Test 2025, Inferenz 2026. Sentinel-2-Szenen erscheinen nur in einem Split. Modellauswahl ausschließlich über 2024.
- **Patchschwelle (festgelegt).** Validierungs- und Test-Patches benötigen 244 von 256 zulässigen Zielzellen. Für das Training werden die Schwellen 244, 230 und 200 auf der Validierung 2024 verglichen; die gelockerten Schwellen erhöhen die Zahl nutzbarer Patches und Szenen, beziehen aber unvollständigere Patches ein.
- **Räumlicher Holdout (festgelegt; Parameter offen).** Jahresweise Trennung verhindert nicht, dass dieselben Stadtflächen in Training und Test wiederkehren. Ein räumlicher Holdout mit festen Ortsgruppen über alle Termine ergänzt daher den zeitlichen Test. Blockgröße und Puffer werden aus empirischen Autokorrelationsreichweiten der Trainingsdaten abgeleitet und über mehrere Größen geprüft (Valavi et al. 2019; Roberts et al. 2017); der Puffer muss mindestens den Patch-Kontext abdecken, die Skalierung wird nur auf der Trainingsfläche geschätzt. Übertragbare Parameter aus der Literatur existieren nicht; die Puffer von Ploton et al. (2020) beziehen sich auf großräumige Biomassemodelle. Ein einzelner Holdout ist keine räumliche Kreuzvalidierung.
- **Unsicherheit (festgelegt).** Seed 0 ist die reproduzierbare Basis; vorab bestimmte Endvarianten werden mit drei Seeds gerechnet. Die Evaluationsunsicherheit wird über räumlich blockierte Bootstrap-Intervalle geschätzt. Drei Seeds tragen keine starken Signifikanzaussagen.

### 5.6 Referenzverfahren und Vergleichsarm

- **Naive Prior-Expansion (festgelegt)** als Untergrenze, die nur die grobe Struktur trägt.
- **Random Forest (festgelegt)** als nichtlineare klassische Referenz auf aggregierten 100-m-Zellen mit Residualziel, ausgewertet in denselben Featureprofilen (Li et al. 2019; Tang et al. 2021; Li, X. et al. 2022). Der Vergleich zeigt, ob das U-Net eine praktikable klassische Referenz übertrifft. Er ist kein fairer Architekturvergleich, da sich Repräsentation, Kapazität, Optimierung und Kontextzugriff unterscheiden.
- **TESSERA-Embeddings als Vergleichsarm (festgelegt).** Moderne Foundation-Model-Ansätze für LST-Super-Resolution sind bisher als Preprint mit synthetischer Degradation belegt (Chen et al. 2026); für AlphaEarth, Clay oder SatMAE liegen keine geprüften LST-Benchmarks vor. Ein Architekturwechsel würde die Ablationslogik brechen. Geprüft wird deshalb der Austausch einer Eingangsgruppe im selben U-Net: TESSERA-Jahresembeddings (10 m, 128 Kanäle, nur Sentinel-1/-2, CC0) werden auf EPSG:25833 reprojiziert und über eine lernbare 1 × 1-Faltung ohne vorgeschaltete PCA eingespeist. Training 2017–2023, Auswertung auf 2024 und 2025, auf der Schnittmenge der zulässigen Zellen beider Arme. features/v6 hängt nicht von TESSERA ab. AlphaEarth-Embeddings scheiden aus, da sie laut Produktdokumentation Landsat-Thermalbänder als Eingang nutzen und damit Zielinformation tragen. Die Jahresembeddings enthalten Sentinel-2-Information des gesamten Jahres einschließlich der Eingangsszene; für 2026 liegen keine Embeddings vor, der Arm ist daher Vergleich und nicht Teil des Inferenzprodukts.

### 5.7 Evaluation ohne 10-m-Referenz

- **Primärmetrik (festgelegt):** maskierte MAE in Kelvin auf gültigen 100-m-Zellen nach Aggregation der 10-m-Ausgabe. Sie misst Skalentreue, keine 10-m-Genauigkeit. SSIM wird sekundär auf ungeklemmten Vorhersagen berichtet; der expandierte Prior erreicht bereits hohe SSIM-Werte, die Metrik trennt daher schwach.
- **Struktur- und Residualdiagnostik (festgelegt; Operationalisierung offen):** Bias, Moran's I der Residuen (vgl. Klopfer 2023), Gradienten- und Kantenvergleiche sowie Hotspot-Überlappung auf identischen Referenzzellen. Referenzfreie Strukturmetriken belegen keine wahre 10-m-LST; größere Schärfe oder geringe Residuen-Autokorrelation bedeuten keine nachgewiesene bessere Temperaturgenauigkeit, und optische Kanten dürfen nicht als thermische Wahrheit gelten.
- **Stratifizierung (festgelegt):** nach Berliner Stadtstrukturtypen, Gruppierung Vorschlag B, mit flächengewichteten Anteilen je 100-m-Zelle.
- **Externe Referenz (offen):** ECOSTRESS ist nicht Teil der Kernvalidierung. Zeitlich mit Landsat kollokierte Aufnahmen über Berlin sind selten, und das Produkt ist mit etwa 70 m selbst fehlerbehaftet und keine 10-m-Referenz (Hulley et al. 2022; Qi et al. 2026). Externe Thermaldaten, auch kommerzielle Archivdaten, bleiben eine Option für eine Plausibilitätsprüfung.

## 6 Belegstatus der Parameter

Die Literatur stützt mehrere Entscheidungen nur im Prinzip. Die folgende Übersicht trennt Evidenz und eigene Festlegung und nennt den Umgang in der Arbeit.

| Parameter | Durch Literatur gestützt | Nicht belegt | Umgang |
| --- | --- | --- | --- |
| Loss-Gewichte 0,1/0,1, 7 × 7-SSIM, Gradiententerm | Kombination aus L1 und SSIM als Prinzip (Dai et al. 2025) | Gewichte, Fenster, Gradiententerm; SSIM auf grob aggregierten Rastern mit wenigen Pixeln je Fenster | Eigene Festlegung, dokumentiert; Audit der Termbeiträge; Gewichte als Ablationskandidat |
| Niederschlagssumme 0–72 h | Vorregen verändert die LST-Oberflächen-Beziehung (Li, H. et al. 2018: 12 h); Vorfeuchte als Auswahlkriterium (Klopfer 2023) | Fensterlänge 72 h; Wind- und Strahlungseffekte auf Tageskontraste | Physikalisch motivierte Feature-Hypothese; Beitrag über die Meteorologiestufe der Ablation |
| Meteorologische Kopplung | Grobe Forcings werden in der Literatur bilinear oder zellweise zugeordnet (Kustura et al. 2025; Yu et al. 2026); ERA5-Land ohne Stadtdarstellung (Muñoz-Sabater et al. 2021) | Optimale Kopplung; Shortcut durch glatte Kovariaten im eigenen Netz | Szenenwert als Standard, bilinear als Ablationsschalter |
| Skalentransfer 1.000 → 100 auf 100 → 10 m | Analoger Transfer mit externer Prüfung (Nguyen et al. 2022) | Nachweis für den konkreten Transfer gegen eine echte Thermalreferenz nahe 10 m | Limitation; Strukturdiagnostik; externe Referenz optional |
| 100-m-Ziel als Flächenmittel | Native Thermalauflösung 100 m (USGS; Xue et al. 2020) | PSF-erhaltende Reaggregationsregel | Eigenes Rasterdesign, dokumentiert |
| Wolken-Konfidenz ≥ 2, Schnee nicht klar | Bitsemantik (USGS) | Optimale Schwelle; Restwolken-Bias | Eigene Filterentscheidung; Maskenverlust dokumentiert |
| SVF-Suchradius 30 m auf 10-m-DSM | 30 m als untersuchter Radius (Scarano & Mancini 2017); Sensitivitätsmethodik (Dirksen et al. 2019) | Optimum für Berlin; auf 10-m-DSM gehen Straßendetails verloren (Dirksen et al. 2019) | Vorläufig; Radiusdiagnose vor dem Republish; DSM-Auflösung als Limitation |
| Kronenschwelle 2,5 m | Bedeckung dominanter als Höhe (Su et al. 2026, 100 m) | Schwellenwert; Vergleich bei ≤ 30 m | Vorläufig; eigene Festlegung, Prüfung der Verteilung, sonst Limitation |
| Maximum- und Streuungskanäle | Höhenstreuung auf Blockebene (Luo et al. 2026) | Zusatznutzen auf ≤ 50 m bzw. 10 m | Entfernt; Redundanz im eigenen Datensatz |
| Kumulative Ablationsleiter | Feature-Kombinationsstudien als Präzedenz (Zhang, H. et al. 2026) | Exakte Leiter; reihenfolgeunabhängige Interpretation | Bedingte Interpretation; Isolationsläufe; die Ablation selbst ist der Beleg |
| Holdout-Blockgröße und Puffer | Ableitung aus Autokorrelationsreichweiten (Valavi et al. 2019; Roberts et al. 2017) | Berliner Parametersatz | Empirische Bestimmung im eigenen Datensatz |
| Nullinitialisierter Residual-Head | Residualprinzip (Nguyen et al. 2022; Zhang, H. et al. 2026) | Kontrollierter Residual-gegen-absolut-Vergleich | Eigenes Versuchsergebnis (Recovery-Test) |

## 7 Ablationsdimensionen

Das finale Ablationsdesign wird auf dem neuen Datenstand festgelegt. Kandidaten sind:

- kumulative Leiter Spektral → Morphologie → Verschattung → Meteorologie mit bedingter 2 × 2-Matrix aus Morphologie und Verschattung bei fester Meteorologie;
- thermalsensitives Trainingsziel gegenüber L1 bei identischen Eingängen, gegebenenfalls mit Variation der Gewichte;
- Kopplung der Meteorologie: Szenenwert gegenüber bilinear;
- Train-Patchschwelle 244, 230 oder 200;
- TESSERA-Arm gegenüber den handgebauten Kontextgruppen;
- räumlicher Holdout für vorab gewählte Varianten;
- gegebenenfalls eine Mindestschwelle für die Sentinel-2-Klarheit je Szene.

Eine kumulative Leiter misst den bedingten Zugewinn zur bisherigen Featuremenge. Reihenfolge und Interaktionen verhindern die Interpretation als isolierten Gesamtwert einer Gruppe; das thermalsensitive Ziel ist zudem eine Änderung des Trainingsziels, keine Featuregruppe. Für unabhängige Gruppenbeiträge sind gezielte Isolations- oder Entfernungsläufe nötig.

## 8 Abgrenzung

- Untersucht wird die Landoberflächentemperatur der Warmsaison, nicht die Lufttemperatur und kein Jahresgang.
- Keine neue Netzarchitektur und keine Architektursuche. Diffusionsmodelle höchstens als separater optionaler Vergleich.
- Keine Sequenzmodelle (3D-CNN, ConvLSTM, temporale Attention). Zeit geht nur über die Sonnengeometrie der Schattenmasken und die szenenbezogene Witterung ein. Eine Untersuchung der Skalenabhängigkeit der Einflussfaktoren gehört in den Ausblick.
- Eine kausale Bewertung von Klimaanpassungsmaßnahmen ist nicht Voraussetzung. Teilaufgabe 7 kann nach Training und Zeitreihe exemplarisch ergänzt werden. Quasi-experimentelle Designs liegen vor (McConnell et al. 2022: Kontrollflächen im Abstand von 90–270 m, Year Fixed Effects; Lin et al. 2026); deren Effekte sind nicht auf Berlin übertragbar, und Identifikation, Paralleltrends und Spillover wären eigens zu prüfen.
- Deliverable ist ein veröffentlichtes Modell mit Setup für 10-m-LST-Zeitreihen Berlins in der Warmsaison.

## 9 Offene Punkte und Limitationen

**Offene Entscheidungen vor Training und Ablation**

- Emissivitätskopplung zwischen LST-Ziel und optischen Prädiktoren: Prüfen: Emissivitätskopplung zwischen LST-Ziel und optischen Prädiktoren (Notion-Archiv)
- Termbeiträge und Gewichte des thermalsensitiven Ziels: Stage-5-Loss auditieren: Termbeiträge, SSIM-Support und Run-Lock (Notion-Archiv)
- Arithmetische gegenüber T⁴-Aggregation: Research-Spot: Radiometrische LST-Aggregation und Pseudo-Pair-Skalenwechsel (Notion-Archiv)
- SVF-Radius und Kronenschwelle vor dem Republish der Features
- Parameter des räumlichen Holdouts und finales Ablationsdesign

**Offen nach dem Training**

- 100-m-Rückkonsistenz bei 10-m-Inferenz, nachgelagerte Projektion oder Limitation: Methodencheck: 100-m-Konsistenz bei 10-m-Inferenz (Notion-Archiv)
- Räumliche Wirkung der Featuregruppen nach Stadtstruktur: Räumliche Wirkung aller Featuregruppen analysieren (Notion-Archiv)
- Umfang von Teilaufgabe 7 und Verfügbarkeit einer externen Thermalreferenz

**Limitationen**

- Keine unabhängige Validierung der 10-m-Feinstruktur; die 100-m-MAE belegt Skalentreue.
- Unvalidierter Skalentransfer des Pseudo-Pair-Lernens auf 100 → 10 m.
- Das Testjahr 2025 wurde in früheren Ablationen und in der EDA gesichtet und ist kein unberührter Holdout; die Modellauswahl erfolgt ausschließlich über 2024.
- Einzelner räumlicher Holdout statt räumlicher Kreuzvalidierung; drei Seeds nur für Endvarianten.
- Vegetationshöhen 2020 für alle Jahre fortgeschrieben; LoD2-Stände für frühe Jahre aus späteren Beständen abgeleitet.
- SVF auf 10-m-DSM; Versiegelung im Straßenraum als Abschnittsschätzung; Wasser = 0 % als eigene Kodierung.
- Mögliche Selektionseffekte durch die strikte Zulässigkeitsregel und die Sentinel-2-Wolkenmaske in dichter Bebauung.
- Meteorologie beschreibt groben Szenenkontext; innerstädtische Witterungsgradienten werden nicht aufgelöst.
- Unvollständige Interpretierbarkeit der kumulativen Leiter wegen Reihenfolge und Interaktionen.

## Zugehörige Festlegungen

- [Methodik-Festlegungen: U-Net-Backbone, Ablationsdesign und Scope](02-herleitung-methodik-festlegungen.md)
