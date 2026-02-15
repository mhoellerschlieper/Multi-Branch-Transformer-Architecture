#catastrophical forgetting

## Einordnung und begriffliche Praezisierung

Im vorliegenden Quelltext wird **catastrophic forgetting** nicht als formal garantiert verhindertes Phaenomen behandelt, sondern als **operational adressiertes Risiko** im Rahmen eines kontinuierlichen Trainingssettings, in dem neue Trainingsdaten online hinzukommen, Teilmodelle (Branches) selektiv trainiert werden und sich die Modelltopologie potenziell erweitert; die Vermeidung bzw Reduktion katastrophalen Vergessens erfolgt dabei ueber mehrere, im Code explizit erkennbare Mechanismen, die zusammen primaer darauf abzielen, (a) die Verteilung der Trainingssignale zu stabilisieren, (b) die Abweichung der Funktion nach strukturellen Aenderungen zu begrenzen und (c) die Leistung auf frueheren Daten systematisch zu ueberwachen.

## 1) Experience Replay als zentraler Anti Forgetting Mechanismus

### 1.1 Replay Buffer als persistente Gedachtnisstruktur
Der unmittelbarste und explizit als Vergessensbremse deklarierte Baustein ist der **experience replay buffer** `replay_buffer_ascii`, der Token Sequenzen aus dem bereits gesehenen Datenstrom speichert:

- Datenstruktur: `v_rows: Vec<Vec<usize>>`
- Kapazitaet: `i_capacity` (im Code typischerweise 5000, gekappt auf max 50000)
- Sampling: zufaellig, deterministisch seedbar (`StdRng`)

Die Funktion `push_row` nimmt jede frische Trainingszeile auf, truncate defensiv auf `MAX_SEQ_LEN` und implementiert bei voller Kapazitaet eine zufaellige Ueberschreibung (random replacement), wodurch ein Bias hin zu sehr alten Beispielen reduziert wird, waehrend zugleich ein reprasentativer Querschnitt historischer Beispiele erhalten bleibt.

### 1.2 Replay Nutzung in der Trainingsschleife
In `train_with_progress_continuous_learning_online_ascii` wird nach jedem frischen Schritt:

1. `rb_replay.push_row(v_row);`
2. mit Wahrscheinlichkeit `d_replay_p` ein Replay Schritt ausgeloest, wobei
3. `rb_replay.sample_row()` ein historisches Beispiel liefert, das erneut trainiert wird.

Damit entsteht ein Mischtraining aus **aktuellen** und **historischen** Beispielen, was in der Literatur als Standardstrategie gilt, um die Gewichtsupdates nicht ausschliesslich auf neue Daten zu konzentrieren und dadurch alte Loesungen zu ueberschreiben (vgl. Goodfellow et al., 2013; Rolnick et al., 2019).

### 1.3 Phasenabhaengige Replay Ramp (Stabilisierung)
Die Replay Intensitaet ist nicht konstant, sondern wird ueber `phase_strategy_config_ascii::replay_p_at_step` ramped:

- `d_replay_p_start`
- `d_replay_p_max`
- `i_replay_ramp_steps`

Diese Ramp Logik reduziert in fruehen Schritten die Interferenz und erhoeht Replay erst, wenn ein stabileres Training erreicht ist; dadurch wird verhindert, dass Replay den frischen Fit in einer sehr instabilen Warmup Phase uebermaessig dominiert, waehrend spaeter der Regularisierungseffekt gegen Vergessen staerker greift.

### 1.4 Replay Effekt Messung (Delta Loss)
Ein weiterer Hinweis auf eine bewusste Anti Forgetting Intention ist `replay_metrics_ascii`, insbesondere:

- `update_delta_loss(d_loss_fresh, d_loss_replay)`
- Statistik ueber `delta_loss = loss_replay - loss_fresh`

Das ist kein direkter Vergessensbeweis, aber eine **Messprozedur**, die anzeigt, ob Replay Beispiele systematisch schlechter oder besser laufen als frische Beispiele, was in der Praxis ein Proxy fuer Drift bzw Vergessen sein kann.

## 2) Retention Evaluation auf festen Kontrollmengen

### 2.1 Fixierte Kontrollsets als Diagnostik gegen Vergessen
Der Code erzeugt zwei Kontrollmengen aus dem initial tokenisierten Datensatz:

- `v_control_old` (take first slice)
- `v_control_new` (take last slice)

Diese werden periodisch ausgewertet durch `eval_control_set_loss_ascii`, die forward only im eval mode (`set_training(false)`) eine mittlere Cross Entropy Loss berechnet.

### 2.2 Retention Delta als Drift Indikator
Die Struktur `retention_metrics_ascii` speichert:

- `d_loss_control_old`, `d_loss_control_new`
- `d_retention_delta_old`, `d_retention_delta_new` relativ zu Baselines `d_old0`, `d_new0`

Damit wird Vergessen nicht verhindert, aber **fruehzeitig detektiert**; in einem operativen System ist dies relevant, weil es erlaubt, Replay Parameter, Lernraten oder weitere Schutzmechanismen adaptiv nachzuregulieren, auch wenn diese Adaptive Steuerung im vorliegenden Ausschnitt noch nicht als closed loop implementiert ist.

## 3) Selektives Branch Training und Fairness Diagnostik als indirekte Stabilisierung

### 3.1 Branch Masking und Mindestaktivitaet
Im Kontext `ParallelBlockGroup` wird pro Schritt eine Availability Mask gesampelt (`sample_availability_mask`) mit:

- Teilnahme Wahrscheinlichkeiten `v_branch_participation_p`
- Mindestanzahl aktiver Branches `i_min_active_branches`

Durch Mindestaktivitaet wird verhindert, dass eine zu aggressive Sparsity dazu fuehrt, dass bestimmte Branches ueber lange Zeit keine Gradienten erhalten, was funktional einer Form von interner Spezialisierung mit anschliessender Erosion nicht genutzter Pfade entsprechen kann; zwar ist dies primaer als Continuous Learning Mechanismus motiviert, aber es wirkt auch gegen interne Degradation, die in komplexen MoE oder Multi Branch Architekturen einen Vergessenscharakter haben kann.

### 3.2 Inverse Participation Scaling fuer unverzerrte Gradienten
Die Option `b_scale_by_inverse_participation` berechnet pro Branch einen Faktor `1/p_i` (geclamped), der beim Backward pro aktivem Branch angewendet wird (`backward_with_availability_mask`).

Die Intuition lautet: Wenn Branches nur mit Wahrscheinlichkeit `p_i` aktiv sind, dann sollen die selten aktiven Branches im Erwartungswert nicht systematisch weniger Update Masse bekommen; die Skalierung stellt eine **unverzerrte Schaetzung** der Gradienten im Sinne des Sampling Designs her, was Trainingsinstabilitaet und unerwuenschte Dominanz einzelner Branches reduziert, und damit indirekt eine Form von Vergessensdynamik durch Untertraining bestimmter Teilpfade mindert (vgl. theoretisch verwandt: Robbins & Monro, 1951; in praxisnahen Settings: Shazeer et al., 2017).

### 3.3 Fairness Metriken gegen Branch Starvation
`branch_fairness_metrics_ascii` zaehlt Selektionen und berechnet u a Gini und Top1 Share; diese Diagnostik ist zwar nicht selbst eine Regularisierung, aber sie adressiert ein typisches Failure Pattern: Wenn eine Routing Logik oder EMA Auswahl immer denselben Branch bevorzugt, koennen andere Branches faktisch "vergessen", weil sie nicht mehr trainiert werden, und das Modell verliert dadurch Diversitaet, was im Zeitverlauf zu Leistungsabfall auf Teilbereichen alter Daten fuehrt.

## 4) Konservative Gewichts Injektion bei Expansion als Schutz gegen Funktionsspruenge

### 4.1 Expansion ist ein struktureller Eingriff mit Vergessensrisiko
Das Hinzufuegen neuer Branches kann aehnliche Effekte wie Fine Tuning auf neue Daten ausloesen: neue Kapazitaet wird geschaffen, aber ohne Schutz kann die Aggregation die Funktion abrupt veraendern.

### 4.2 Conservative Injection reduziert die sofortige Dominanz neuer Branches
Die Methode `add_branches_with_conservative_injection_ascii` skaliert alte Branch Gewichte mit `(1-eta)` und verteilt `eta` auf neue Branches, gefolgt von Normalisierung.

Das reduziert die Wahrscheinlichkeit, dass neue Branches sofort einen zu grossen Anteil am Output erhalten, wodurch die bisherige Funktion erhalten bleibt und neue Kapazitaet graduell lernen kann, ohne alte Kompetenzen zu ueberschreiben; dies ist konzeptionell eine Form von **funktionaler Kontinuitaetsregularisierung** durch Aggregationsgewichte.

## 5) Drift Proxy um Expansion: Logit Distanz als Kontinuitaetsmessung

Wenn Expansion stattfindet, berechnet der Code vor und nach dem Expand:

- `logits_distance_l2_ascii`
- `logits_cosine_distance_ascii`

und sammelt diese in `drift_metrics_ascii`.

Diese Drift Metriken sind kein direkter Forgetting Test, aber sie begrenzen bzw quantifizieren Funktionsspruenge bei Strukturwachstum, was praktisch relevant ist, da grosse Spruenge oft mit ploetzlichem Leistungseinbruch auf frueheren Aufgaben einhergehen; die Kombination aus konservativer Injektion und Drift Monitoring bildet damit einen weiteren Anti Forgetting Schutzrahmen.

## 6) Was der Code nicht implementiert (und daher nicht als "verhindert" gelten kann)

Aus Expertensicht ist wichtig, dass der Code keine der folgenden klassischen Anti Forgetting Verfahren als harte Garantien implementiert:

- keine explizite Regularisierung a la EWC oder SI,
- keine distillation gegen ein eingefrorenes Teacher Modell als Nebenverlust,
- keine strikt getrennte Memory Rehearsal Strategie mit stratified sampling,
- keine explizite Lernratensteuerung basierend auf Retention Metriken (nur Messung).

Daher ist die korrekte Beschreibung: Der Code **reduziert** catastrophic forgetting mit Replay, konservativer Expansion und Retention Monitoring, anstatt es formal zu verhindern.

## Zusammenfassung der wirksamen Mechanismen im Quelltext

1. **Experience replay** (Buffer + probabilistisches Replay Training) ist der primaere aktive Mechanismus gegen Vergessen.  
2. **Retention Evaluation** auf fixierten Kontrollmengen detektiert Vergessensdrift.  
3. **Masking mit Mindestaktivitaet**, **inverse participation scaling** und **Fairness Diagnostik** stabilisieren die Verteilung der Updates ueber Branches und reduzieren Branch Starvation.  
4. **Konservative Gewichts Injektion** bei Expansion und **Logit Drift Messung** begrenzen Funktionsspruenge, die als Vergessensereignis wirken koennen.

## Literatur (APA)

```text
Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). 
An empirical investigation of catastrophic forgetting in gradient-based neural networks. 
arXiv preprint arXiv:1312.6211.

Robbins, H., & Monro, S. (1951). A stochastic approximation method. 
The Annals of Mathematical Statistics, 22(3), 400-407.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). 
Experience replay for continual learning. Advances in Neural Information Processing Systems, 32.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). 
Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. 
arXiv preprint arXiv:1701.06538.
```
