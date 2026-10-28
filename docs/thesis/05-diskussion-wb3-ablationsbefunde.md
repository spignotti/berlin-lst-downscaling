# WB3 · Diskussion der Ablationsbefunde

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f66814e9b25deacb0f788cc`, zuletzt bearbeitet 2026-10-09). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Die gesamte Seite bezieht sich auf die Ablation auf training/v1. Die aktuellen Evidenzgrenzen stehen in [05 · Diskussion & Limitationen](05-diskussion.md).

**Zugehörige Ergebnistabellen:** [WB3 · Ergebnistabellen – Baselines und Ablation](04-ergebnisse-wb3-ergebnistabellen.md). Die Beurteilung bezieht sich auf diese veröffentlichten MAE-Vergleiche.

**Bewertungsmaßstab:** maskierte MAE nach 10×10-Rückaggregation auf gültige 100-m-Zellen; kein direkter 10-m-Genauigkeitsnachweis.

**Stand 08.10.2026:** Wissenschaftliche Interpretation der kumulativen Ablation, der Isolationsläufe und ihrer Aussagegrenzen. Technische Primärevidenz: [Ablation](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/ablation-full-results.md), [Stage 1](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/stage1-full-results.md).

## 1. Welche Verbesserung zählt als Beitrag urbaner Kontextvariablen?

**Gesicherter Befund:** Schon der residuale Stage-1-Spektralansatz übertrifft die naive Prior-Expansion auf demselben Bewertungssatz deutlich. Diese Differenz belegt das Lernen gegenüber der naiven Referenz, **nicht** den Zusatznutzen urbaner Morphologie, Verschattung oder Meteorologie, da sie in Stage 1 fehlen.

**Interpretation:** Der inkrementelle Zusatzbeitrag der Kontextfeatures ergibt sich ausschließlich aus den **Stufenunterschieden** gegenüber Stage 1; der kumulative Vergleich hält viele Designaspekte fest, ersetzt aber kein vollständig kontrolliertes faktorielles Experiment.

## 2. Morphologie, Schatten und meteorologische Prädiktoren

**Beobachtungen:** Beim einzigen dokumentierten Seed verbessert sich die Test-MAE über die kumulative Ablationsleiter; Validation stagniert bzw. schwankt bis Stage 4. Der ERA5-Schritt ist der größte inkrementelle Feature-Schritt der 2025-Testwerte; die Änderung von Stage 4 zu Stage 5 ist ein **Loss-Vergleich bei gleichen Eingängen**.

**Interpretationshypothesen (noch nicht belegt):**

- ERA5 kann zusätzliche wettersituative Information tragen, die ein nur spektraler Ansatz nicht bereitstellt. Die beobachtete Verbesserung erlaubt jedoch keine kausale Aussage über einzelne ERA5-Kanäle.
- Kleine Unterschiede zwischen Stage 2 und 3 könnten begrenzten eigenständigen Schattenbeitrag, Redundanz mit Morphologie oder Lern-/Seed-Varianz bedeuten. Ein eindeutiger Schluss ist mit der vorliegenden Leiter nicht möglich.
- Der Stage-5-Gewinn in aggregierter MAE spricht für die Nützlichkeit des modifizierten Loss im getesteten Setup, **nicht automatisch** für realistischere mikroskalige Temperaturkontraste.

## 3. Was leisten die Isolationsruns?

Die isolierte ERA5-Erweiterung liegt im 2025-Test zahlenmäßig leicht unter dem vollständigen Stage-4-Stack, die isolierte Schatten-Erweiterung verändert Stage 1 nur wenig. **Das ist kein Nachweis**, dass Morphologie „schadet“ oder Schatten grundsätzlich nutzlos sind. Zwischen Isolation und kumulativer Leiter ändern sich die Kombinationen der Prädiktoren; Interaktionen, Redundanzen und stochastische Effekte sind nicht systematisch aufgelöst.

**Offene Forschungsfrage:** Ein gezielter Leave-one-out-Vergleich bei konstanter meteorologischer Eingangsgruppe könnte den zusätzlichen Beitrag der Morphologie und Verschattung besser isolieren. Ein solcher Vergleich wurde bislang nicht durchgeführt.

## 4. Evidenz- und Generalisierungsgrenzen

- **Messgröße:** Der 10-m-Output wird für die Fehlerbewertung auf 100 m zurückaggregiert. Gute Scale-Consistency impliziert nicht die native thermische 10-m-Korrektheit oder plausible Subpixelmuster.
- **Pseudo-Pairs:** Der Prior entsteht aus Degradation des Landsat-Targets derselben Szene; dieses Lernexperiment ist nicht gleichbedeutend mit Prüfung gegenüber unabhängig gemessener feiner LST.
- **Temporaler Holdout:** Train 2017–2023, Validation 2024, Test 2025, aber wiederkehrende räumliche Patchpositionen. Kein Spatial-Holdout und damit kein belastbarer Beleg für räumliche Transferfähigkeit.
- **Eine Initialisierung:** Ein Seed pro Ablationsvariante, keine Varianz-/Signifikanzabschätzung; kleine Differenzen sind entsprechend vorsichtig zu lesen.
- **Testzugriff in der Studie:** Zwar wird je Run der 2025-Test nach Checkpointwahl einmal ausgewertet; der Test wird über mehrere Stufen und Isolationsläufe wiederholt betrachtet. Kein vollständig unberührter finaler Test für nachgelagerte Forschungsentscheidungen ist hier nachgewiesen.
- **Sensor-/Referenzunsicherheit:** Unabhängige hochauflösende thermale Referenz ist bislang nicht in diesem Ablationsvergleich eingesetzt. Landsat-, ECOSTRESS- und andere Quellen sind im Validierungskapitel fachlich gesondert einzuordnen.

## 5. Wissenschaftliche Folgefragen

1. Was können ECOSTRESS, räumliche Strukturmetriken und räumlicher Holdout **zusätzlich** zur 100-m-Skalentreue belegen?
2. Wie stabil sind die beobachteten Inkremente über Seeds, Szenen, Stadtstrukturen und Wetterlagen? Was davon passt noch in den Scope?
3. (historisch, training/v1) Der Random-Forest-Vergleich ist seit 09.10.2026 abgeschlossen: bestes RF-Test-MAE 1.18341 K, U-Net mit gleichen Featurefamilien (Stage 4) 0.47535 K. Wie weit erlaubt der Vergleich unterschiedlicher Lernrepräsentationen und Modellkapazitäten eine methodische Aussage über den Vorteil räumlicher DL-Modelle? [Validierter RF-Bericht](https://github.com/spignotti/berlin-lst-downscaling/blob/main/docs/random-forest-full-results.md).
4. Wie lassen sich die Ergebnisse in der Berliner stadtklimatischen Anwendung beschreiben, ohne LST mit Lufttemperatur oder FITNAH-Modellkarten mit thermischer Ground Truth gleichzusetzen?
5. Wo bleiben kausale oder quasi-experimentelle Maßnahmenbewertungen optional statt notwendige Schlussfolgerung?

## Stand der wissenschaftlichen Evidenz

Die bisherigen Beobachtungen stützen Aussagen zur **aggregierten Modellgüte innerhalb der dokumentierten Ablation**. Ein belastbarer Nachweis physikalisch korrekter 10-m-Temperaturdetails, räumlicher Generalisierbarkeit oder kausaler Wirkungen einzelner urbaner Prädiktoren liegt noch nicht vor.
