## Implementierung von Continuous Learning

### 1. Begrifflicher Rahmen und Abgrenzung im Codekontext
Im MBT wird *Continuous Learning* als ein Trainingsmodus operationalisiert, der (a) fortlaufend neue Trainingsdaten waehrend des laufenden Trainings akzeptiert, (b) die Modellaktualisierung ohne harte Trainingsstopps ermoeglicht, (c) eine partielle Aktivierung und selektive Aktualisierung von Modellteilen (Branches) zulaesst und (d) die Modellkapazitaet bei Bedarf autonom erweitern kann, waehrend parallel dazu Diagnostik, Fortschrittsereignisse und Snapshot-Mechanismen die Einbettung in einen produktionsnahen Betrieb (Training neben Serving) stützen.

Technisch ist das Konzept im Code nicht als einzelnes Modul realisiert, sondern als Zusammenspiel mehrerer Bausteine, die in den Trainingsfunktionen der `Llm`-Struktur, in der Multi-Branch-Architektur (`ParallelBlockGroup`) sowie in den Online-Ingestion- und Metrikpfaden zusammengefuehrt werden.

---

### 2. Online-Trainingsdatenaufnahme (Streaming Ingestion) als Kernvoraussetzung

#### 2.1 Event-basierter Datenkanal
Continuous Learning setzt im Code an der Faehigkeit an, neue Daten *waehrend* eines laufenden Trainings einzuschleusen. Dies geschieht ueber den Typ:

- `TrainingDataEventAscii` mit Events:
  - `add_training_file_json_array { s_path }`
  - `add_training_rows { v_rows }`
  - `shutdown`

Diese Events werden ueber einen `Receiver<TrainingDataEventAscii>` (mpsc) in die Trainingsschleife injiziert.

#### 2.2 Nicht-blockierendes Draining in Trainingsschleifen
Die Funktion

- `drain_training_data_events_non_blocking_ascii(...)`

liest in einer Schleife `try_recv()`-basiert alle aktuell verfuegbaren Events aus dem Kanal, was zwei Eigenschaften sicherstellt:

1. **Keine Blockierung** der Trainingsschleife (wichtig fuer Latenz, Cancel und Progress-Events).
2. **Amortisierter Dateneinfluss**: Neue Daten werden in Batches aufgenommen, nicht nur an Epochengrenzen.

#### 2.3 Validierung, Budgetierung und Tokenisierung
Neue Rohdaten werden ueber

- `append_tokenized_rows_ascii(...)`

in `v_tokenized_data: Vec<Vec<usize>>` ueberfuehrt, wobei mehrere Schutzmechanismen implementiert sind:

- Gesamtbudget: `i_max_total_rows`
- Pro-Event-Limit: `i_max_rows_per_event`
- Rejection leerer Zeilen, zu kurzer Tokenfolgen (`len() < 2`)
- Truncation auf `MAX_SEQ_LEN`
- Robustheit gegen Parse-Fehler und Tokenizer-Fehler, mit Counter-Tracking

Damit implementiert der Code Continuous Learning als *append-only* Trainingsdatenstrom mit stabilen Kapazitaetsgrenzen und Fehlerisolierung.

---

### 3. Partielle Branch-Verfuegbarkeit (Masking) und Unbiased Updates

#### 3.1 ContinuousLearningConfig als Steuerstruktur
Die Struktur

- `ContinuousLearningConfig`

definiert die Parameter, die das *stochastische Teilnehmen* einzelner Branches pro Trainingsschritt bestimmen:

- `v_branch_participation_p: Vec<f32>` (Teilnahmewahrscheinlichkeiten pro Branch)
- `i_min_active_branches: usize` (Untergrenze fuer aktive Branches)
- `b_scale_by_inverse_participation: bool` (unverzerrte Skalierung)
- `u64_mask_seed: u64` (Determinismus)

Die Methode `validate()` stellt sicher, dass p-Werte in `(0, 1]` liegen und die Maskendimension konsistent ist.

#### 3.2 Mask Sampling pro Schritt
Die Maske wird durch

- `sample_availability_mask(rng, v_p, i_min_active)`

erzeugt. Zunaechst wird fuer jeden Branch Bernoulli-sampled (`u < p_i`), anschliessend wird garantiert, dass mindestens `i_min_active_branches` aktiv sind (durch zufaelliges Aktivieren bislang inaktiver Indizes). Dieser Mechanismus ist fuer Continuous Learning relevant, weil er eine *partielle, variable Substruktur* des Netzes pro Schritt trainierbar macht und damit unterschiedliche Betriebszustaende (z.B. partieller Ausfall, Kapazitaetsregulierung, exploratives Routing) modelliert.

#### 3.3 Forward und Backward mit Availability Mask
Die `ParallelBlockGroup` stellt maskierte Varianten bereit:

- `forward_with_availability_mask(a_input, v_active_mask)`
- `backward_with_availability_mask(a_grads, d_lr, v_active_mask, opt_inv_participation_scale)`

Wesentlich ist dabei:

- Im Forward werden nur aktive Branches ausgefuehrt (Rayon-parallel), danach wird ueber aktive Outputs gemittelt.
- Im Backward werden nur aktive Branches rueckwaerts gerechnet; optional wird pro Branch ein Skalierungsfaktor angewandt.

#### 3.4 Inverse Participation Scaling (Unbiasedness im Erwartungswert)
Wenn `b_scale_by_inverse_participation` aktiv ist, wird pro Branch ein Faktor `1/p_i` (geclamped) berechnet und im Backward auf die Gradienten angewandt, bevor `br.backward(...)` aufgerufen wird. Dadurch wird erreicht, dass Branches, die selten aktiv sind, im Erwartungswert eine vergleichbare Update-Masse erhalten wie haeufig aktive Branches, was die *systematische Unteroptimierung* seltener Pfade reduziert und damit eine typische Instabilitaetsquelle von partiell aktivierten Architekturen adressiert (Robbins & Monro, 1951; Shazeer et al., 2017).

---

### 4. Selektives Branch-Training mit EMA-gestuetzter Routing-Logik

#### 4.1 Phase-orientierte Aktivierung der EMA-Selektion
Die Struktur

- `phase_strategy_config_ascii`

steuert, ob und wann eine EMA-basierte Branch-Selektion aktiv ist:

- `b_enable_ema_branch_selection`
- `i_ema_warmup_steps`
- Methode: `ema_is_active(i_total_steps)`

Damit wird ein zweistufiger Mechanismus implementiert: Zunaechst kann eine explorative oder breit verteilte Teilnahme (Maske) erfolgen, spaeter wird selektiver geroutet.

#### 4.2 EMA-State pro Branch
Die Struktur

- `branch_loss_ema_state_ascii`

haelt pro Branch eine EMA des beobachteten Loss (`v_ema_loss: Vec<Option<f32>>`) und aktualisiert diese ueber `update_ema(i_branch, d_loss)`.

#### 4.3 Auswahl des Branches mit minimaler EMA-Loss
Die Funktion

- `select_branch_min_ema_loss_ascii(...)`

evaluiert fuer jede *verfuegbare* Branch (durch Maske gefiltert) einen Loss auf dem aktuellen Beispiel (Forward nur durch diese Branch via Single-Branch-Maske), aktualisiert die EMA und waehlt anschliessend die Branch mit minimalem EMA-Score.

Das Resultat ist ein *deterministisches, datengetriebenes* Routing innerhalb der zulaessigen Teilmenge. Operational ist dies Continuous Learning, weil der Lernprozess nicht nur neue Daten akzeptiert, sondern auch intern adaptiv entscheidet, *welcher Teil des Modells* bevorzugt trainiert wird, basierend auf fortlaufend aktualisierten, zeitlich geglaetteten Performanzsignalen.

---

### 5. Experience Replay als Stabilitaets- und Retentionskomponente im Continuous Learning

#### 5.1 Replay Buffer als fortlaufendes Gedaechtnis
Der Typ

- `replay_buffer_ascii`

speichert Token-Sequenzen bis zur Kapazitaet (typisch 5000) und erlaubt zufaelliges Sampling. Bei voller Kapazitaet ueberschreibt er zufaellige Slots, was eine einfache, aber robuste Reservoir-aehnliche Erhaltung eines gemischten historischen Pools bewirkt (Rolnick et al., 2019).

#### 5.2 Phase-gesteuerte Replay-Wahrscheinlichkeit (Ramp)
Die Replay-Wahrscheinlichkeit wird ueber

- `phase_strategy_config_ascii::replay_p_at_step(i_total_steps)`

zwischen `d_replay_p_start` und `d_replay_p_max` ueber `i_replay_ramp_steps` hochgeregelt. Der Code mischt damit Fresh-Updates und Replay-Updates adaptiv ueber die Zeit, was in einem Streaming-Setting eine zentrale Operationalisierung von Continuous Learning darstellt, weil das Modell nicht nur neue Daten nachzieht, sondern zugleich die *zeitliche Kohärenz* der Parameterentwicklung absichert.

---

### 6. Autonome Kapazitaetserweiterung (Width Expansion) als Continuous Learning im strukturellen Sinn

#### 6.1 Triggerpunkt: periodische Expansion Checks
In `train_with_progress_continuous_learning_online_ascii` wird in festen Intervallen (Konfiguration `i_expand_check_every_steps`) geprueft, ob Expansion sinnvoll ist, sofern `b_enable_autonomous_expansion` aktiv ist.

#### 6.2 Diagnostikmetriken als Entscheidungskriterium
Die `ParallelBlockGroup` berechnet ueber

- `compute_metrics_from_inputs(v_inputs)`

Metriken wie `path_starvation_index`, `top1_share`, `effective_num_paths`. In

- `try_autonomous_expand_first_pg_ascii(...)`

werden conservative Triggerregeln genutzt (z.B. Starvation oder Collapse Indikatoren), um Expansion auszulösen.

#### 6.3 Konservative Gewichtsinjektion
Expansion fuegt einen neuen Branch (hier: `TransformerSequence` aus zwei `TransformerBlock`) hinzu und ruft:

- `add_branches_with_conservative_injection_ascii(v_new_branches, d_eta_injection)`

auf. Dabei werden alte Aggregationsgewichte mit `(1 - eta)` skaliert und neue Branches teilen sich `eta`, wodurch ein abrupter Funktionssprung reduziert wird. Dies ist Continuous Learning in einer weitergehenden Bedeutung: Das System passt nicht nur Parameter, sondern auch die *Kapazitaetsstruktur* online an.

#### 6.4 Drift-Proxy zur Funktionskontinuitaet
Um Expansion nicht blind zu vollziehen, berechnet der Code vor/nach Expansion Logit-Distanzen (`L2`, Cosine) und sammelt diese in `drift_metrics_ascii`. Damit entsteht ein Feedbacksignal, das zumindest als Monitoring der funktionalen Kontinuitaet dient.

---

### 7. Produktiver Betrieb: Cancel, Progress-Events und Snapshotting

#### 7.1 Cooperative Cancel
Nahezu alle zentralen Schleifen pruefen

- `b_cancel.load(Ordering::SeqCst)`

und terminieren kooperativ. Continuous Learning in Produktionsumgebungen erfordert genau dieses Muster, weil Training nicht exklusiv laufen darf.

#### 7.2 Fortschritts- und Metrikereignisse
Der Code emittiert ueber `Sender<TrainingProgressEventAscii>` regelmaessig Events, die neben Loss auch umfangreiche Streaming- und Branch-Metriken enthalten (Ingestion-Raten, Coverage, Mask-Statistiken, Replay-Anteile, Retention-Losses, Fairness, Drift). Diese Telemetrie ist keine Lernlogik im engen Sinn, stellt aber die Voraussetzung dar, Continuous Learning *operativ zu steuern*.

#### 7.3 Snapshots fuer Serving Updates
Ueber `export_parameters_snapshot()` und periodisches Senden auf `tx_snapshot` kann ein Serving-Modell Parameterupdates erhalten, ohne dass Training blockiert. Das ist eine typische Continuous-Learning-Architekturentscheidung: Training und Serving koexistieren, verbunden durch stueckweise, versionierbare Parameterstroeme.

---

## Zusammenfassende Bewertung der Implementierung
Continuous Learning wird im Code als mehrschichtige Pipeline implementiert, die aus Online-Datenaufnahme, partieller Modellteilnahme (Masking), erwartungstreuer Update-Skalierung, adaptiver Branch-Selektion (EMA), Experience Replay, autonomer Kapazitaetserweiterung, Drift- und Retentionsmonitoring sowie produktionsnaher Steuerung (Cancel, Progress, Snapshots) besteht. Der Ansatz ist damit nicht lediglich ein "Online-Training", sondern eine Kombination aus *Datenstrom-Inkrement*, *struktureller und funktionaler Adaptivitaet* und *Telemetry-gestuetzter Betriebssicherheit*.

---

## Literatur (APA)
```text
Robbins, H., & Monro, S. (1951). A stochastic approximation method. The Annals of Mathematical Statistics, 22(3), 400-407.

Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., & Wayne, G. (2019). Experience replay for continual learning. Advances in Neural Information Processing Systems, 32.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.
```
