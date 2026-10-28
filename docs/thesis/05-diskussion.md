# 05 · Diskussion & Limitationen

> Migriert aus Notion am 2026-10-28 (Seite `3f3356451f668168be3ec75744ea3716`, zuletzt bearbeitet 2026-10-08). Text unverändert übernommen.

## Einordnung der vorliegenden Ablationsbefunde

**Spektraler Ausgangspunkt:** Bereits Stage 1 verbessert die aggregierte MAE gegenüber der naiven Prior-Expansion deutlich. Dieser Unterschied ist dem vollständigen residualen spektralen Modellansatz zuzuordnen, nicht einem später hinzugefügten urbanen Featureblock.

**Urbane Prädiktoren:** In der kumulativen Ablation sinkt die 2025-Test-MAE beim Hinzufügen der Kontextgruppen. Die Validierungs-MAE ist jedoch bis Stage 4 nicht monoton. Morphologie und Schatten werden in der kumulativen Leiter nur zusammen mit den vorherigen Featuregruppen bewertet. Der reine Schatten-Isolationslauf zeigt nur einen kleinen Testunterschied gegenüber Stage 1.

**Meteorologie:** Das isolierte ERA5-Modell erreicht auf dem 2025-Test eine etwas niedrigere aggregierte MAE als die vollständig kombinierte Stage 4. Das zeigt, dass ein größerer Featureumfang in den vorliegenden Daten nicht in jedem Vergleich zu einer weiteren Verbesserung führt; die zugrunde liegenden Interaktionen sind nicht isoliert bestimmt.

**Thermalsensitives Training:** Stage 5 verbessert bei unveränderten Eingangsfeatures die dokumentierte 100-m-MAE. Ob dadurch realistischere kleinräumige Temperaturmuster entstehen, bleibt offen.

## Wissenschaftliche Einschränkungen

- Nur ein Seed pro dokumentierter Modellvariante; keine robuste Varianzschätzung kleiner Stufendifferenzen.
- Zeitlicher Holdout mit wiederkehrenden räumlichen Patchankern; keine belegte unabhängige räumliche Generalisierung.
- Verlustfunktion und Bewertungsmetrik prüfen Aggregationskonsistenz zur 100-m-Landsat-Skala, nicht thermische 10-m-Ground-Truth.
- Die 2025-Testkohorte wurde über mehrere Ablations- und Isolationsläufe hinweg zur Ergebnisbetrachtung herangezogen.
- Der Grobprior im Pseudo-Pair-Training wird aus der Landsat-Beobachtung derselben Szene abgeleitet.

## Vertiefende Diskussion

- [WB3 · Diskussion der Ablationsbefunde](05-diskussion-wb3-ablationsbefunde.md)
