# Exkurs: Experimentelle Evaluation von Large Language Models zur Produktenklassifikation

## 1. Motivation und Forschungsfrage
Die vorangegangenen Kapitel haben gezeigt, dass regelbasierte Ansätze (Regex, Keyword-Matching) bei der Klassifikation von unstrukturierten Produktdaten an natürliche Grenzen stoßen. Insbesondere die semantische Unterscheidung zwischen einem "PV-Modul" und dessen "Zubehör" (z.B. "Anschlusskabel für PV-Modul") erfordert oft ein Kontextverständnis, das über starre Regelsätze hinausgeht.

Vor diesem Hintergrund untersucht dieser Exkurs, ob moderne Large Language Models (LLMs) eine robuste Alternative darstellen. Die zentrale Forschungsfrage lautet:
> *"Kann ein LLM-basierter Ansatz die regelbasierte Klassifikation in Präzision und Recall übertreffen, und ist dieser Ansatz ökonomisch sowie technisch für Datensätze >70.000 Zeilen skalierbar?"*

## 2. Methodik und Systemdesign

### 2.1 Modellwahl und Architektur
Für das Experiment wurde primär **OpenAI GPT-4o-mini** gewählt. Die Entscheidung basierte auf einem initialen Preis-Leistungs-Vergleich: Da die Aufgabe "Extraktion und Klassifikation" weniger Reasoning-Kapazität erfordert als komplexe Textgenese, bieten "Mini"-Modelle den optimalen Trade-off zwischen Kosten (~$0.15/1M Tokens) und Geschwindigkeit. Als Vergleichsgröße wurde im späteren Verlauf die **GLM-4.5 Serie (ZhipuAI)** evaluiert, um alternative Provider-Strategien zu testen.

Als **State-of-the-Art Referenz** wurde zudem die neue **GPT-5 Serie** (gpt-5.2 und gpt-5-mini) evaluiert, die seit Januar 2026 verfügbar ist. Diese Modelle erfordern spezielle technische Anpassungen:

| Konfiguration | Anforderung |
| :--- | :--- |
| **temperature** | Muss entfernt werden (analog zu O1-Reasoning-Modellen) |
| **Client-Timeout** | Erhöhung auf 20 Minuten (Deep Reasoning Latency) |
| **Parallelisierung** | Max. 2-5 Worker (Ressourcenintensiv) |

Der technische Aufbau folgt einer **Batch-Verarbeitung**:
1.  **Input**: Unstrukturierte Produktbeschreibung (Titel, Attribute).
2.  **Verarbeitung**: Bündelung von **50 Produkten pro Request** (Batching), um HTTP-Overhead zu minimieren und den Kontext effizient zu nutzen.
3.  **Output**: Ein striktes JSON-Schema, das neben der Klasse (`is_pv_module`: Boolean) auch `confidence`, `reasoning` und technische Attribute wie `power_watts` liefert.

### 2.2 Prompt Engineering und Fehlerkorrektur (Iterativer Prozess)
Ein zentrales Erkenntnis dieses Experiments war, dass "Out-of-the-Box" Prompting nicht ausreicht. Der Optimierungsprozess verlief in zwei Phasen:

*   **Phase 1 (Naive Definition)**: Der initiale Prompt definierte lediglich positive Kriterien ("Ist es ein PV-Modul?"). Dies führte zu einer hohen Anzahl an **False Positives** (Präzision < 90%). Zubehörteile wie "Montageset für PV-Anlage" oder "Wechselrichter für Balkonkraftwerk" wurden fälschlich als Module klassifiziert, da das LLM stark auf das Keyword "PV-Anlage" reagierte.
*   **Phase 2 (Semantic Negative Constraints)**: Zur Lösung wurde das Konzept der "Semantischen Ausschlussregeln" eingeführt. Der System-Prompt wurde um explizite Anweisungen erweitert, was *kein* Modul ist (z.B. *"Zubehör & Dienstleistungen MUSS FALSE SEIN, auch wenn 'PV-Anlage' im Text steht"*). Dieser Schritt war entscheidend, um die Ambiguität der Sprache aufzulösen.

### 2.3 Technische Implementierung
Die Integration wurde als robuste Python-Anwendung realsiert, die eine hohe Typsicherheit und Fehlerresistenz gewährleistet.
*   **SDK-Abstraktion**: Der Zugriff auf die Modelle erfolgt über die offiziellen SDKs (`openai` und `zhipuai`). Eine benutzerdefinierte `LLMClient`-Klasse abstrahiert die Provider-Logik, sodass nahtlos zwischen GPT-4o-mini und GLM-4.5 gewechselt werden kann.
*   **Validierung mit Pydantic**: Um die strikte Einhaltung des JSON-Schemas sicherzustellen, wurde die Bibliothek **Pydantic** eingesetzt. Das erwartete Ausgabeformat (Klasse `ClassificationResult`) wird als Datenmodell definiert. Dies garantiert, dass die API-Antworten valide Datentypen enthalten (z.B. `power_watts` als Integer, `is_pv_module` als Boolean) und filtert "halluzinierte" oder defekte Antworten automatisch heraus.
*   **Concurrency Control**: Die parallele Verarbeitung wird über Pythons `ThreadPoolExecutor` gesteuert, was eine feingranulare Kontrolle der Worker-Anzahl ermöglicht, um Rate Limits dynamisch zu vermeiden.

## 3. Empirische Ergebnisse

### 3.1 Klassifikationsgüte (Ground Truth Vergleich)
Die finale Evaluation auf einem validierten Testdatensatz (n=1.000) zeigte, dass der optimierte Ansatz (Phase 2) die menschliche Genauigkeit erreicht:

| Metrik | Ergebnis | Interpretation |
| :--- | :--- | :--- |
| **Recall** | **100.00%** | Es wurden *alle* echten PV-Module identifiziert. Keine "False Negatives". |
| **Precision** | **100.00%** | Es gab *keine* Falschmeldungen. Zubehör wurde zu 100% korrekt ausgefiltert. |
| **F1-Score** | **100.00%** | Perfekte Balance zwischen Vollständigkeit und Genauigkeit. |

Besonders hervorzuheben ist die qualitative Analyse der vermeintlichen Grenzfälle: Produkte wie "Netzersatzanlagen" oder "Hybrid-Wechselrichter", die für Laien (und einfache Regex) schwer von Modulen zu unterscheiden sind, wurden vom LLM dank der semantischen Regeln korrekt als "Nicht-Modul" erkannt.

### 3.2 Skalierbarkeit und Limits (Stress-Tests)
Ein wesentlicher Teil der Untersuchung widmete sich der technischen Skalierbarkeit für den Zieldatensatz von 70.000 Zeilen.

#### 3.2.1 Batch Size Limits (Basis-Modelle)
*   Tests mit Batch-Größen von 10, 50, 100, 200 und 500 zeigten einen klaren "Breaking Point".
*   Bis **50 Produkte/Request** blieb die Präzision stabil bei 100%.
*   Ab **200 Produkten/Request** (ca. 30k Tokens) traten bei Modellen wie `glm-4-plus` **JSON-Schema-Fehler** (Halluzinationen) auf.
*   *Empfehlung*: Eine Batch-Size von 50 stellt das lokale Optimum zwischen Durchsatz und Stabilität dar.

#### 3.2.2 GPT-5 Serie – State-of-the-Art Benchmarking (Januar 2026)
In einer "Extreme Benchmarking"-Reihe wurden die neuen GPT-5 Modelle intensiv getestet:

**GPT-5.2 (Flagship Reasoning Model)**:

| Batch Size | Ergebnis | Anmerkungen |
| :--- | :--- | :--- |
| 10 | ✅ Erfolg | Stabil, aber langsam (~30s/Batch) |
| 50 | ✅ Erfolg | Stabil |
| 100 | ✅ Erfolg | **SOTA Performance** – Doppelte Batch-Kapazität vs. GPT-4o-mini |
| 150 | ✅ Erfolg | **Maximum Stable Batch Size** |
| 200 | ⚠️ Degradiert | ~12% Datenverlust durch Output-Truncation |
| 500 | ❌ Fehlschlag | Massiver Datenverlust (98%), "Garbage Output" |

**GPT-5-Mini**:

| Batch Size | Ergebnis |
| :--- | :--- |
| 10 | ✅ Erfolg |
| 50 | ⚠️ Instabil (Timeouts) |
| 100+ | ❌ Fehlschlag |

*Fazit*: `gpt-5-mini` ist für Batch-Processing ungeeignet (Max: Batch 10-50).

#### 3.2.3 Token-Limit Experiment
**Hypothese**: Kann eine Erhöhung von `max_output_tokens` (auf 16k) die Truncation bei Batch 200/500 beheben?

**Ergebnis**:
*   **Technisch**: Ja, der Prozess läuft durch.
*   **Qualitativ**: Nein. Der Output war "Garbage" (Halluzinationen, Wiederholung des Inputs).

**Erklärung**: Das Limit ist nicht nur syntaktisch (Token-Count), sondern **semantisch (Attention-Span)** bedingt. Das Modell "vergisst" seine Aufgabe bei zu viel Kontext. Diese Erkenntnis ist fundamental für das Verständnis der Skalierungsgrenzen von LLMs.

#### 3.2.4 API Rate Limits
*   Bei hoher Parallelisierung (>20 Workers) stieß `gpt-4o-mini` im Tier-1-Status an **Rate Limits** (429 Errors).
*   Die Lösung lag in einer adaptiven Drosselung auf **5-10 parallele Worker** oder dem Wechsel auf Provider mit höheren Concurrency-Limits (z.B. ZhipuAI/GLM-4.5-Air, welches im Stress-Test bei 20 Workers stabil blieb).

#### 3.2.5 Modellempfehlungen

| Modell | Max Batch Size | Empfehlung |
| :--- | :--- | :--- |
| gpt-4o-mini | 50 | **Preis-Leistungs-Sieger** für Produktion |
| glm-4.5-air | 50 | Alternative mit höheren Concurrency-Limits |
| gpt-5.2 | 150 | **State-of-the-Art Referenz** für maximale Qualität |
| gpt-5-mini | 10-50 | Nicht empfohlen für Batching |

### 3.3 Kostenanalyse
Die Wirtschaftlichkeit ist für den Einsatz in der Bachelorarbeit gegeben:
*   Die Kosten für den verwendeten Testlauf (1.000 Zeilen) lagen bei **~$0.11**.
*   Hochgerechnet auf den Gesamtdatensatz (70.000 Zeilen) ergeben sich Kosten von ca. **$7.50 (€6.90)** bei Nutzung von `gpt-4o-mini`.
*   Alternative Modelle wie `glm-4.5-flash` könnten diese Kosten theoretisch auf nahe Null senken, zeigten im Test jedoch instabileres Verhalten bei hoher Last.

#### 3.3.1 Effizienz-Vergleich: Small vs. Large Batches (GPT-5.2)
Ein kontrollierter Test mit 500 Items (identischer Datensatz) ergab:

| Strategie | Zeit | Kosten | Qualität |
| :--- | :--- | :--- | :--- |
| **Small Batches** (10×50) | 6 min | $1.70 | 100% |
| **Optimal** (1×150) | ~3 min | ~$1.27 | 100% |
| **Large Batch** (1×500) | 2.3 min | $1.07 | 0% (Garbage) |

**Fazit**: Größere Batches sind schneller und billiger, aber nur bis **Batch 150 stabil**. Der vermeintliche Kostenvorteil bei Batch 500 ist illusorisch, da der Output unbrauchbar ist. `gpt-5.2` ist zwar teurer pro Token als `gpt-4o-mini`, bietet aber durch die höhere Batch-Kapazität (150 vs. 50) eine **Verdopplung der Effizienz pro Request**.

## 4. Fazit
Das Experiment bestätigt die Hypothese: LLMs können die regelbasierte Klassifikation nicht nur ersetzen, sondern qualitativ übertreffen (100% F1-Score in der Stichprobe). Durch gezieltes Prompt Engineering ("Negative Constraints") und technisches Batching ist der Ansatz sowohl präzise als auch ökonomisch skalierbar.

**Kernerkenntnisse aus dem GPT-5 Benchmarking**: Die neueste GPT-5 Serie ermöglicht Batch-Größen bis 150 Items/Request, was die Effizienz gegenüber GPT-4o-mini (max. 50) verdoppelt. Höhere Werte führen jedoch zu Qualitätsverlust durch semantische Attention-Limits – ein fundamentales Ergebnis, das die physischen Grenzen aktueller Transformer-Architekturen aufzeigt.

**Produktionsempfehlung**: Für die finale Datenbereinigung dieser Arbeit wird `gpt-4o-mini` als kosteneffiziente Standardmethode gewählt. Für zukünftige Projekte mit höheren Qualitätsanforderungen stellt `gpt-5.2` mit Batch-Size 150 die technische Referenz dar.
