# 05 · Diskussion & Limitationen

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668168be3ec75744ea3716`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.
> Überarbeitet am 2026-10-28: Datenstand features/v6 / ARD v10 / training/v3; ältere Inhalte als historisch gekennzeichnet.

## Einordnung der historischen Ablationsbefunde (training/v1)

> **Historischer Stand – ungültig nach Datenkorrektur (training/v1).** Die Ablation wird auf training/v3 neu berechnet, siehe [04 · Ergebnisse & Analysen](04-ergebnisse.md).

**Spektraler Ausgangspunkt:** Bereits Stage 1 verbessert die aggregierte MAE gegenüber der naiven Prior-Expansion deutlich. Dieser Unterschied ist dem vollständigen residualen spektralen Modellansatz zuzuordnen, nicht einem später hinzugefügten urbanen Featureblock.

**Urbane Prädiktoren:** In der kumulativen Ablation sinkt die 2025-Test-MAE beim Hinzufügen der Kontextgruppen. Die Validierungs-MAE ist jedoch bis Stage 4 nicht monoton. Morphologie und Schatten werden in der kumulativen Leiter nur zusammen mit den vorherigen Featuregruppen bewertet. Der reine Schatten-Isolationslauf zeigt nur einen kleinen Testunterschied gegenüber Stage 1.

**Meteorologie:** Das isolierte ERA5-Modell erreicht auf dem 2025-Test eine etwas niedrigere aggregierte MAE als die vollständig kombinierte Stage 4. Das zeigt, dass ein größerer Featureumfang in den vorliegenden Daten nicht in jedem Vergleich zu einer weiteren Verbesserung führt; die zugrunde liegenden Interaktionen sind nicht isoliert bestimmt.

**Thermalsensitives Training:** Stage 5 verbessert bei unveränderten Eingangsfeatures die dokumentierte 100-m-MAE. Ob dadurch realistischere kleinräumige Temperaturmuster entstehen, bleibt offen.

## Wissenschaftliche Einschränkungen

Die folgenden Einschränkungen wurden für die Ablationsleiter auf training/v1 formuliert und gelten weiter, soweit das Design unverändert ist.

- Nur ein Seed pro dokumentierter Modellvariante; keine robuste Varianzschätzung kleiner Stufendifferenzen.
- Zeitlicher Holdout mit wiederkehrenden räumlichen Patchankern; keine belegte unabhängige räumliche Generalisierung.
- Verlustfunktion und Bewertungsmetrik prüfen Aggregationskonsistenz zur 100-m-Landsat-Skala, nicht thermische 10-m-Ground-Truth.
- Die 2025-Testkohorte wurde über mehrere Ablations- und Isolationsläufe hinweg zur Ergebnisbetrachtung herangezogen.
- Der Grobprior im Pseudo-Pair-Training wird aus der Landsat-Beobachtung derselben Szene abgeleitet.

## Emissivitätskopplung des Landsat-Targets

Eine schreibgeschützte Diagnostik an 9 Landsat-8/9-Szenen (6 Sommer, 3 Winter, 2022–2024) vergleicht die Inversion mit der Produktemissivität (a) mit einer statischen ASTER-GED-v3-Emissivität (b) ([Emissivitätsdiagnostik](../emissivity-diagnostic.md)). Über 8,48 Mio. klare Pixel liegt die Differenz (a)−(b) im Mittel bei −0,07 K (Median +0,04 K; p05/p95 −0,86 / +0,57 K). Die Differenz ist damit ein Effekt zweiter Ordnung neben der Produktunsicherheit ST_QA von im Mittel 2,2–5,2 K. Sie ist jedoch nicht zufällig verteilt: Mit zunehmender Versiegelung steigt (a)−(b) systematisch von −0,21 K (0–10 % versiegelt) auf +0,14 K (90–100 %).

Die sommerlichen Landsat-NDVI–LST-Steigungen sind mit der Produktemissivität um 4–17 % steiler als mit der statischen Emissivität (Median ca. 7 %); die Korrelationen ändern sich um höchstens 0,02. Für die Interpretation NDVI-basierter Eingabearme ist zu berücksichtigen, dass ein Teil der NDVI–LST-Beziehung im Target aus der Emissivitätskopplung stammen kann; der Einfluss dieser Kopplung auf die Ergebnisse der NDVI-basierten Eingabearme ist nicht quantifiziert.

Eine eigene Target-Variante (etwa LST mit statischer Emissivität neu berechnet) wird nicht in die Ablation aufgenommen. Die Emissivitätskopplung des C2-Targets ST_B10 bleibt damit eine dokumentierte Limitation. Vorbehalte der Diagnostik: Die GED-Daten (2000–2008) liegen vor den Szenen (2022–2024), Landnutzungswandel ist nicht trennbar; für Landsat 9 werden die Landsat-8-Koeffizienten verwendet.

## Aktuelle Evidenzgrenzen

Stand: training/v3 (features/v6, ARD v10).

- **Bewertungsskala:** Die Bewertung erfolgt auf 100-m-Pseudo-Paaren (Prior 1.000 m, Supervision 100 m); die 10-m-Ausgabe wird nach exakter 10×10-Aggregation bewertet. Es gibt keine unabhängige thermische 10-m-Referenz.
- **Zeitlicher Split:** Train 2017–2023, Validation 2024, Test 2025; 2026 dient nur der Inferenz. Der Test 2025 darf nicht in Checkpoint-, Modell-, Hyperparameter- oder Schwellenwahl eingehen.
- **Datenausschluss:** Das ARD-v10-Gate schließt 4 Paarungen aus (286 von 290 verbleiben).
- **EDA v3:** Die EDA beschreibt auch das Jahr 2025 deskriptiv ([summary.json](../results/eda-v3/summary.json), `data_gaps`); für deskriptive Aussagen ist 2025 damit kein unberührter Holdout (wie schon bei EDA v2). Modellentscheidungen dürfen sich nicht auf diese Deskription stützen.
- **Seeds, Varianzschätzung und Baselines:** offen, abhängig von der finalen Ablationsmatrix.
- **Emissivität:** siehe oben; keine Target-Variante im Vergleich.

## Vertiefende Diskussion

- [WB3 · Diskussion der Ablationsbefunde](05-diskussion-wb3-ablationsbefunde.md)
