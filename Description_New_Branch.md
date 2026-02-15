## Zeitpunkt und Ausloeser fuer das Hinzufuegen eines neuen Branches

Im obigen Code kommt ein neuer Branch **ausschliesslich** im Rahmen der **autonomen Expansion** des ersten `ParallelBlockGroup` hinzu, und zwar dann, wenn die Trainingsschleife regelmaessig eine Expansion prueft, bestimmte Diagnostikmetriken einen Engpass anzeigen und die konfigurierten Sicherheitsgrenzen eine Erweiterung erlauben.

### 1. Wo im Code wird die Expansion angestossen

Der Eintrittspunkt ist die Funktion:

- `Llm::train_with_progress_continuous_learning_online_ascii(...)`

Innerhalb der inneren Trainingsschleife (pro Zeile bzw. pro Schritt) wird in festem Intervall geprueft:

- `cfg_phase.b_enable_autonomous_expansion == true`
- `cfg_phase.i_expand_check_every_steps > 0`
- und konkret: `(i_total_steps % cfg_phase.i_expand_check_every_steps) == 0`

Erst **an diesen Checkpoints** ist eine Expansion ueberhaupt moeglich.

### 2. Welche Voraussetzungen muessen erfuellt sein (Guards)

Wenn der Intervall-Trigger greift, wird zusaetzlich geprueft, ob es diagnostisch verwertbare Inputs gibt, denn die Entscheidung basiert auf Metriken, die aus Aktivierungen vor der `ParallelBlockGroup` gewonnen werden:

- `collect_parallel_block_group_inputs_for_diagnostics(...)` oder
- bei der erweiterten Variante im Online-Loop: es werden diagnostische Inputs aus Prompts erzeugt und an `try_autonomous_expand_first_pg_ascii(...)` uebergeben.

Falls diese Inputs leer sind, findet **keine** Expansion statt.

Zusaetzlich gilt eine harte Obergrenze:

- Wenn `pg_num_branches >= cfg_phase.i_max_total_branches`, dann wird **keine** Expansion ausgefuehrt (Rueckgabe `Ok(false)`).

### 3. Die eigentliche Entscheidung: Wann wird wirklich ein neuer Branch hinzugefuegt

Die konkrete Entscheidung erfolgt in:

- `Llm::try_autonomous_expand_first_pg_ascii(cfg_phase, v_diag_inputs)`

Dort wird zunaechst ueber:

- `pg.compute_metrics_from_inputs(v_diag_inputs)?`

ein `ParallelBlockGroupMetrics` Objekt berechnet. Danach werden drei Trigger-Bedingungen ausgewertet:

- `b_starved`: `m.d_path_starvation_index > 0.60`
- `b_collapsed`: `m.d_top1_share > 0.70`
- `b_low_eff`: `m.d_effective_num_paths < 2.0`

Nur wenn **mindestens eine** dieser Bedingungen wahr ist:

- `if !(b_starved || b_collapsed || b_low_eff) { return Ok(false); }`

kommt es zur Expansion.

### 4. Was genau wird hinzugefuegt und wie

Wenn die Triggerbedingungen erfuellt sind, wird **genau ein** neuer Branch erzeugt, und zwar als `TransformerSequence` aus zwei `TransformerBlock` Instanzen:

- `tb1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)`
- `tb2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM)`
- `seq = TransformerSequence::new(vec![tb1, tb2])`
- `Box::new(seq)`

Anschliessend wird dieser Branch der ersten `ParallelBlockGroup` hinzugefuegt durch:

- `pg.add_branches_with_conservative_injection_ascii(vec![b_new_branch], cfg_phase.d_eta_injection)?;`

Dabei werden die bestehenden Branch-Gewichte konservativ skaliert und der neue Branch erhaelt einen Anteil gemaess `d_eta_injection` (mit Clamping auf `[0.0, 0.5]`), was der Code in `add_branches_with_conservative_injection_ascii(...)` implementiert.

### 5. Praezise Kurzantwort

Ein neuer Branch wird hinzugefuegt, **wenn**:

1. autonomes Expandieren aktiviert ist (`b_enable_autonomous_expansion`),
2. der Trainingsschritt ein Vielfaches von `i_expand_check_every_steps` ist,
3. ein `ParallelBlockGroup` existiert und noch nicht die Maximalzahl an Branches erreicht ist,
4. die Diagnostikmetriken eine Ueberlastung bzw. Kollaps anzeigen, konkret mindestens eine der Schwellwertbedingungen gilt:
   - `path_starvation_index > 0.60` oder
   - `top1_share > 0.70` oder
   - `effective_num_paths < 2.0`,
5. und dann wird per `add_branches_with_conservative_injection_ascii(...)` genau ein neuer `TransformerSequence` Branch (2 Bloecke) injiziert.

### Literaturhinweis (APA)

Keine externen Quellen werden im Code referenziert; die Aussagen ergeben sich unmittelbar aus dem bereitgestellten Quelltext und dessen Expansionslogik (Schlieper, 2026).
