# Exkurs: Experimentelle Evaluation von Large Language Models zur Produktenklassifikation

## 1. Motivation und Forschungsfrage
Die vorangegangenen Kapitel haben gezeigt, dass regelbasierte Ansätze (Regex, Keyword-Matching) bei der Klassifikation von unstrukturierten Produktdaten an natürliche Grenzen stoßen. Insbesondere die semantische Unterscheidung zwischen einem "PV-Modul" und dessen "Zubehör" (z.B. "Anschlusskabel für PV-Modul") erfordert oft ein Kontextverständnis, das über starre Regelsätze hinausgeht.

Vor diesem Hintergrund untersucht dieser Exkurs, ob moderne Large Language Models (LLMs) eine robuste Alternative darstellen. Die zentrale Forschungsfrage lautet:
> *"Kann ein LLM-basierter Ansatz die regelbasierte Klassifikation in Präzision und Recall übertreffen, und ist dieser Ansatz ökonomisch sowie technisch für Datensätze >70.000 Zeilen skalierbar?"*

## 2. Methodik und Systemdesign

### 2.1 Modellwahl und Architektur
Für das Experiment wurde primär **OpenAI GPT-4o-mini** gewählt. Die Entscheidung basierte auf einem initialen Preis-Leistungs-Vergleich: Da die Aufgabe "Extraktion und Klassifikation" weniger Reasoning-Kapazität erfordert als komplexe Textgenese, bieten "Mini"-Modelle den optimalen Trade-off zwischen Kosten (~$0.15/1M Tokens) und Geschwindigkeit. Als Vergleichsgröße wurde im späteren Verlauf die **GLM-4.5 Serie (ZhipuAI)** evaluiert, um alternative Provider-Strategien zu testen.

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

1.  **Batch Size Limits**:
    *   Tests mit Batch-Größen von 10, 50, 100, 200 und 500 zeigten einen klaren "Breaking Point".
    *   Bis **50 Produkte/Request** blieb die Präzision stabil bei 100%.
    *   Ab **200 Produkten/Request** (ca. 30k Tokens) traten bei Modellen wie `glm-4-plus` **JSON-Schema-Fehler** (Halluzinationen) auf, und die API verweigerte teilweise die Antwort (Rate Limits).
    *   *Empfehlung*: Eine Batch-Size von 50 stellt das lokale Optimum zwischen Durchsatz und Stabilität dar.

2.  **API Rate Limits**:
    *   Bei hoher Parallelisierung (>20 Workers) stieß `gpt-4o-mini` im Tier-1-Status an **Rate Limits** (429 Errors).
    *   Die Lösung lag in einer adaptiven Drosselung auf **5-10 parallele Worker** oder dem Wechsel auf Provider mit höheren Concurrency-Limits (z.B. ZhipuAI/GLM-4.5-Air, welches im Stress-Test bei 20 Workers stabil blieb).

### 3.3 Kostenanalyse
Die Wirtschaftlichkeit ist für den Einsatz in der Bachelorarbeit gegeben:
*   Die Kosten für den verwendeten Testlauf (1.000 Zeilen) lagen bei **~$0.11**.
*   Hochgerechnet auf den Gesamtdatensatz (70.000 Zeilen) ergeben sich Kosten von ca. **$7.50 (€6.90)** bei Nutzung von `gpt-4o-mini`.
*   Alternative Modelle wie `glm-4.5-flash` könnten diese Kosten theoretisch auf nahe Null senken, zeigten im Test jedoch instabileres Verhalten bei hoher Last.

## 4. Fazit
Das Experiment bestätigt die Hypothese: LLMs können die regelbasierte Klassifikation nicht nur ersetzen, sondern qualitativ übertreffen (100% F1-Score in der Stichprobe). Durch gezieltes Prompt Engineering ("Negative Constraints") und technisches Batching (50 Items/Request) ist der Ansatz sowohl präzise als auch ökonomisch skalierbar. Für die finale Datenbereinigung dieser Arbeit wird daher dieser LLM-basierte Ansatz als Standardmethode gewählt.
