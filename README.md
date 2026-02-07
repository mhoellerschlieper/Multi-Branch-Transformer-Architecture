1. Projektüberblick
Dieses Repository implementiert und dokumentiert eine Multi-Branch-Transformer-Architektur (MBT) in Rust, wobei die Parallelität nicht primär als externe Ausführungsstrategie (z. B. reine Pipeline-Partitionierung entlang der Tiefe), sondern als architektonisch explizite Breitenstruktur innerhalb einzelner Layer modelliert wird: Pro Layer werden mehrere Transformer-Blöcke bzw. Block-Sequenzen gleichzeitig ausgeführt, deren Pfadausgaben anschließend durch eine Aggregationsstufe zu einer gemeinsamen Layerrepräsentation fusionieren.

Die technische Motivation ergibt sich aus praktischen Inferenz- und Betriebsengpässen großer Sprachmodelle, die in realen Systemen häufig weniger durch reine FLOP-Komplexität als vielmehr durch Speicherbedarf, Speicherbandbreite, KV-Cache-Management, sowie Kommunikations- und Synchronisationskosten in verteilten Umgebungen bestimmt werden; MBT positioniert die Layerbreite hierbei zugleich als Partitionierungs- und Orchestrierungseinheit für heterogene Infrastruktur, einschließlich Peer-to-Peer-(P2P)-Topologien.

Ergänzend adressiert das Projekt drei systemische Zielklassen, die im MBT-Ansatz als konstitutiv betrachtet werden:

Ausfallsicherheit (Fault Tolerance) durch renormalisierte Aggregation bei partieller Pfadverfügbarkeit,
Continuous Learning trotz asynchroner, zeitvariabler Teilnahme von Pfaden,
Continuous Expandable Width als kontrollierte, laufende Erweiterbarkeit der Layerbreite durch konservative Gewichtsinjektion und graduelle Regewichtung.
2. Architekturkonzept: Multi-Branch Transformer (MBT)
2.1 Grundprinzip: Parallelität als Layerbreite
In einem MBT-Layer existiert eine Menge paralleler Pfade 

{TBl,1​,…,TBl,K​}, die denselben Layer-Input verarbeiten und Pfadausgaben erzeugen; die Layerausgabe ergibt sich aus einer Aggregation über Gewichte 
αi(l)​ (nichtnegativ, normiert):

h(l+1)=∑i=1K​αi(l)​zi(l)​.

Damit entsteht ein System, in dem gleichzeitige Ausführung nicht als Sonderfall der Infrastruktur, sondern als Bestandteil des Modellgraphen definiert ist, wodurch eine explizite Kopplung zwischen Modellstruktur und Verteilbarkeit möglich wird.

2.2 Aggregation als zentrale Orchestrierungs- und Robustheitskomponente
Die Aggregationsstufe fungiert nicht nur als numerischer Operator, sondern als systemkritische Komponente, weil sie zugleich folgende Funktionen ermöglicht:

Fusion paralleler Pfadausgaben,
Pfadgewichtung (statisch oder adaptiv),
Maskierung und Renormalisierung bei Pfadausfall,
Governance-Regeln gegen Pfad-Verarmung und Gewichtskollaps,
potenziell robuste Aggregation gegen byzantinische Ausreißerpfade (z. B. trimmed mean / median-of-means als mögliche Erweiterung).
3. Verteilte Ausführung im P2P-Setting (Konzept)
Das Repository orientiert sich an der Prämisse, dass MBT insbesondere in P2P- und heterogenen Netzwerken eine natürliche Zuordnung erlaubt: ein Pfad (Branch) entspricht einer klaren Partitionseinheit, die von einer Node übernommen und ausgeführt werden kann, während ein Aggregationsknoten oder ein verteiltes Aggregationsprotokoll die Ergebnisse fusioniert.

Im Unterschied zu rein sequenzieller Block-Verteilung entlang der Tiefe (die vor allem Speicher pro Node reduziert, jedoch typischerweise keinen parallelen Token-Speedup liefert), kann MBT potenziell die kritische Pfadlänge reduzieren, sofern Straggler- und Netzwerk-Overheads operativ kontrolliert werden (z. B. über Quorum/Timeout-Regeln).

4. Trainingslogik: Tiefe vs. Breite
4.1 Tiefentraining (Depth)
Die Tiefe bleibt als sequenzielle Layerkette kompatibel mit klassischer Backpropagation; das Projekt übernimmt daher das übliche Prinzip schichtweiser Fehlerpropagation.

4.2 Breitentraining (Width)
Das Breitentraining erfordert zusätzlich Mechanismen, um parallele Pfade stabil und kapazitätserhaltend zu optimieren, insbesondere um:

Pfad-Verarmung (untertrainierte Pfade) zu verhindern,
Dominanz einzelner Pfade (Weight-Collapse) zu vermeiden,
die Abhängigkeit von singulären Bewertungsregeln zu reduzieren.
Ein etabliertes Leitprinzip besteht in einer adaptiven Gewichtung der Pfade mit Mindestbeteiligung, sodass jeder Pfad nichttriviale Gradientenexposition erhält und im Ausfallfall nicht bloß „kalte Redundanz“ darstellt.

5. Ausfallsicherheit (Fault Tolerance)
MBT modelliert Ausfallrobustheit über eine Maskierungsvariable 

mi(l)​∈{0,1} und eine renormalisierte Gewichtung:
α~i(l)​=∑j=1K​mj(l)​αj(l)​mi(l)​αi(l)​​(sofern Nenner>0),h~(l+1)=∑i=1K​α~i(l)​zi(l)​.

Damit bleibt die Layerfunktion wohldefiniert, solange pro Layer mindestens ein Pfad verfügbar ist; die Systemqualität degradiert kontrolliert in Abhängigkeit der entfernten Gewichtmasse, sofern Governance-Regeln (Mindestbeteiligung, Normkontrolle, Quorum/Timeout) implementiert sind.

6. Continuous Learning und Continuous Expandable Width
6.1 Continuous Learning (asynchron, partiell)
Continuous Learning wird als Training unter zeitvariabler Menge aktiver Pfade modelliert; Updates sind damit auch bei partieller Teilnahme formulierbar, sofern Teilnahmeasymmetrien kompensiert werden (z. B. durch pfadspezifische Skalierung der Lernrate) und Vergessen durch geeignete Regularisierung kontrolliert wird.

6.2 Continuous Expandable Width (kontinuierliche Breiten-Erweiterung)
Neue Pfade werden im laufenden Betrieb als zusätzliche Branches integriert, wobei eine konservative Gewichtsinjektion den Funktionssprung begrenzt; ein typisches Schema lautet mit Injektionsrate 
β∈(0,1):

bestehende Pfade: 
αi′​=(1−β)αi​
neue Pfade (uniform): 
αj′​=β/M
Die operative Zielsetzung besteht darin, neue Rechenressourcen unmittelbar in zusätzliche Modellkapazität zu überführen, ohne einen disruptiven Neustart der Gesamtarchitektur zu erzwingen.

7. Implementationsumfang (Rust): Komponenten und Module
Das Repository folgt einem „self-contained“-Ansatz und integriert die Kernbestandteile eines kompakten LLM-Stacks, wobei die MBT-Architektur als Erweiterungs- und Strukturprinzip über den Transformer-Layer gelegt wird.

Typischerweise umfasst die Codebasis folgende Funktionsbereiche (die konkrete Dateistruktur kann projektspezifisch abweichen):

Core-Layer/Model: Embeddings, Self-Attention, Feed-Forward, LayerNorm, Transformer-Block, Output Projection, sowie MBT-spezifische Branch- und Aggregationslogik.
Tokenizer: Byte Pair Encoding (BPE) mit deterministischer Trainingskonfiguration.
Training: autoregressives Next-Token-Training (Pretraining und Instruction-Tuning als Varianten desselben Loops), Optimizer (z. B. Adam) und numerische Stabilitätsmechanismen (z. B. Gradient Clipping).
Inference: greedy decoding sowie optional temperature / top-k / top-p (abhängig vom Stand des Implementationsumfangs).
Checkpoints: robustes Speichern/Laden inkl. Tokenizer- und Modellparametern in konsistentem Format, mit Fokus auf Reproduzierbarkeit und Formkompatibilität.
8. Checkpoints und Determinismus: „Load with Rebuild“
Ein zentraler Kompatibilitätsaspekt ergibt sich aus der Abhängigkeit der Output-Projektion von der Vokabulargröße (Shape: 
[demb​,∣V∣]); daher wird ein Verfahren eingesetzt, das beim Laden:

Checkpoint validiert (Version/Magic),
Tokenizer rekonstruiert,
Modell anhand des im Checkpoint gespeicherten Vokabulars neu aufbaut,
Parameter vektorbasiert einspielt.
Dieses Vorgehen reduziert Shape-Mismatch-Risiken und unterstützt A/B-Vergleiche durch reproduzierbare Tokenizer-Konfigurationen.

9. Build, Run und Nutzung (CLI-orientiert)
9.1 Voraussetzungen
Rust stable toolchain
Cargo
9.2 Build
cargo build --release

9.3 Run
cargo run --release

Je nach Implementationsstand stellt eine CLI typischerweise Funktionen bereit, die Training, Speichern, Laden sowie Prompt-basierte Inferenz unterstützen.

10. Sicherheit und Robustheit (Systemperspektive)
Für verteilte MBT-Setups gilt Sicherheit nicht als optionaler Zusatz, sondern als systemische Voraussetzung, da Branches als Angriffspunkte fungieren können (Byzantinische Outputs, Poisoning, Straggling/DoS). Wesentliche Schutzklassen umfassen:

Integrität von Modellartefakten (Hashing, Signaturen, Versionierung),
Quorum-/Timeout-Policies gegen Tail-Latency und Straggling,
Norm- und Gewichtskontrollen in der Aggregation,
optional: Governance- und Audit-Schicht (z. B. Blockchain-basierte Registry/Attestation in erweiterten Zielarchitekturen).
Die On-Chain-Ausführung des Modells ist dabei nicht zwingend, während eine Chain als manipulationsresistenter Ordnungsrahmen für Identität, Artefakt-Hashes, Update-Freigaben und Sanktionen dienen kann.

11. Abgrenzung zu MoE / Switch / „Multi-Path“
MBT unterscheidet sich konzeptionell und technisch von sparsely-gated MoE-Ansätzen und Switch-Transformern, die Breite primär als tokenweises Routing auf wenige aktivierte Expert*innen realisieren, während MBT Breite als gleichzeitig aktive Pfade pro Layer mit expliziter Aggregation definiert; zugleich unterscheidet sich MBT von „Multi-Path“-Interpretationen residualer Netze, weil Mehrpfadigkeit nicht bloß analytisch, sondern strukturell und orchestrationstauglich implementiert ist (Shazeer et al., 2017; Fedus et al., 2022; Veit et al., 2016).

12. Roadmap (technisch, erweiterungsorientiert)
Je nach aktuellem Stand sind typische nächste Schritte:

Effiziente Inferenz: KV-Cache, Batching, Maskierung, Mixed Precision.
Verteilungsruntime: Branch-Discovery, Scheduling, Quorum-basierte Aggregation, Straggler-Management.
Robuste Aggregation: Ausreißerresistenz, Reputations-/Trust-Gewichte, Mindestbeteiligungs-Governance.
Continuous Learning Governance: Update-Validierung, Rollback, Poisoning-Detektion.
Testbarkeit: Unit-Tests (Tokenizer, Softmax-Stabilität, Checkpoint-Roundtrip), deterministische Golden-Tests.
13. Lizenz und Kontakt
Lizenzinformationen sind dem Repository zu entnehmen.
Kontakt und Projektbezug (gemäß den bereitgestellten Materialien): ms...@expchat.ai sowie die verlinkten GitHub-Projekte zur Rust-basierten LLM- und verteilten GPT-Node-Implementierung.


Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538.

Veit, A., Wilber, M. J., & Belongie, S. (2016). Residual networks behave like ensembles of relatively shallow networks. In Advances in Neural Information Processing Systems.
