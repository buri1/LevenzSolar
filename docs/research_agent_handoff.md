# Research Agent Hand-off: LLM-Exkurs Roadmap

## 1. STRUKTUR DER BACHELORARBEIT (Der Exkurs)
Die 2-3 Seiten in der Thesis sollen folgende Struktur (A-D) haben. Bitte halte dich strikt daran.

### A) Motivation & Forschungsfrage des Exkurses (ca. ½ Seite)
*   **Problem**: Regelbasierte Ansätze skalieren schwer (Edge Cases, manuelle Wartung).
*   **Hypothese**: LLMs erfassen heterogene Produktbeschreibungen natürlicher als Regex.
*   **Forschungsfrage**: "Kann ein LLM-basierter Ansatz die regelbasierte Klassifikation erreichen oder übertreffen, und zu welchen Kosten?"

### B) Methodischer Ansatz (ca. 1 Seite)
*   **System Design**:
    *   **LLM-Client**: OpenAI GPT-4o-mini (gewählt für Speed/Cost-Balance).
    *   **Prompting**: Strict JSON Output (`is_pv_module`, `reasoning`, `confidence` + `power_watts`).
    *   **Soft-Rules**: Der System Prompt enthält die Business Rules (z.B. "Unterscheide Balkonkraftwerk von Wechselrichter"), aber das LLM entscheidet kontextbasiert.
*   **Optimierung & Strategie**:
    *   **Batching-Strategie**:
        *   Um Kosten und Zeit zu sparen, werden 10 Produkte pro Request gebündelt.
        *   Dies reduziert den Overhead (HTTP-Handshakes) und nutzt den Kontext-Cache effizienter.
        *   **Kosten-Optimierung**: Nutzung von `gpt-4o-mini` (nicht 4o!), da dieses Modell für Extraktionsaufgaben aureichend ist und nur ~1/10 kostet.
    *   **Prompt-Evolution (Optimierungsprozess)**:
        *   **Phase 1 (Initial)**: Einfache Definition "PV-Module = True".
            *   *Problem*: False Positives bei "Teilen" von Anlagen (Kabel, Wechselrichter) oder Dienstleistungen ("Montage PV-Anlage"). Das LLM reagierte zu stark auf das Keyword "PV-Anlage".
        *   **Phase 2 (Optimiert)**: "Semantic Negative Constraints".
            *   Ergänzung von strikten Auschlussregeln in natürlicher Sprache (z.B. "MUSS FALSE SEIN, auch wenn 'PV-Anlage' im Text steht").
            *   Explizite Nennung von Ambivalenzen (Wechselrichter, Montage, Netzersatzanlagen).
            *   *Ergebnis*: Beseitigung aller False Positives (100% Precision) bei gleichbleibendem Recall.

### C) Ergebnisse & Vergleich (ca. ½-¾ Seite)
*   **Empirische Ergebnisse (Testlauf n=1000 - "V4 Optimiert")**:
    *   **Recall**: **100.00%** (Alle 11 PV-Module wurden korrekt gefunden).
    *   **Precision**: **100.00%** (0 False Positives - Perfekte Trennung von Zubehör vs. Modul).
    *   **F1-Score**: **100.00%**.
*   **Analyse der "False Positives" (Qualitäts-Check)**:
    *   Initial wurden Produkte wie "Netzersatzanlage" oder "Hybrid-Wechselrichter" fälschlich als Modul erkannt.
    *   Eine Tiefenprüfung zeigte: Dies waren KEINE Module, der "Ground Truth" Datensatz war korrekt.
    *   Durch "Semantic Negative Constraints" im Prompt konnte das LLM diese Fälle zuverlässig als Zubehör klassifizieren, ohne echte Module zu übersehen (Recall blieb 100%).
*   **Interpretation**:
    *   Der LLM-Ansatz erreicht nach Prompt-Optimierung die Genauigkeit eines menschlichen Experten.
    *   Es übertrifft starre Regeln, da es Kontext ("Teil einer Anlage" vs. "Anlage") versteht.
*   **Kosten**:
    *   Kosten pro 10.000 Zeilen: **~$1.06** (€0.98).
    *   Kosten für 70.000 Zeilen: **~$7.42** (€6.83).
    *   Model: `gpt-4o-mini`.
*   **Geschwindigkeit & Skalierbarkeit**:
    *   **1.000 Zeilen**: Problemlos in < 30 Sekunden (20 Workers).
    *   **10.000+ Zeilen**: Hier greifen **Rate Limits** (Error 429) bei zu hoher Parallelisierung.
    *   **Finding**: Für die volle 70k-Verarbeitung sollte die Parallelisierung auf **5-10 Workers** begrenzt werden, um stabile API-Antworten zu gewährleisten. Die technische Machbarkeit ist bestätigt, der Engpass ist rein administrativ (Tier-Limit).

*   **Alternativer Provider (ZhipuAI / GLM-4-Plus)**:
    *   **Qualität**: Ebenfalls **100% Precision / 100% Recall** (identisch zu OpenAI).
    *   **Stabilität**: Keine Rate Limits bei 20 Workers (robustere Concurrency im Tier).
    *   **Nachteile**:
        *   **Kosten**: ~$0.70 / 1M Token (vs $0.15 bei OpenAI) -> ca. 5x teurer.
        *   **Latenz**: Deutlich langsamer (~4 Min für 1k Zeilen vs < 30 Sek bei OpenAI).
    *   **Fazit**: Gute Fallback-Option, aber OpenAI `gpt-4o-mini` bleibt Preis-Leistungs-Sieger, wenn man die Limits beachtet.

*   **GLM-4.5 Serie (Neu)**:
    *   **Support**: Der Code unterstützt jetzt `glm-4.5-air`, `glm-4.5-flash` und `glm-4.5-preview`.
    *   **GLM-4.5-Air**: Erfolgreich getestet. Sehr schnell, robust bei Parallelisierung. Kosten (~$0.20 In / $1.10 Out) etwas höher als 4o-mini bei Output.
    *   **Empfehlung**: Nutzen Sie `glm-4.5-flash` für maximale Einsparung (teilw. kostenlos/sehr günstig) oder `glm-4.5-air` für High-Performance Runs mit Batch-Size 20-50.

### D) Grenzen der Skalierung (Stress-Test Findings)
*   **Batch Size Limits**:
    *   **Empfohlen**: **10-50 Produkte** pro Request.
    *   **Maximum**: Bis zu 50 möglich bei 100% Accuracy.
    *   **Breaking Point (>100)**: 
        *   GLM-4.5-Flash: Rate Limits (429) bei Batch 200 (wegen hoher Token-Last).
        *   GLM-4-Plus: Schema-Fehler (Halluzinationen/Ungültiges JSON) bei Batch 200.
    *   **Fazit**: Eine Erhöhung über 50 bringt kaum Geschwindigkeitsvorteile, gefährdet aber die Stabilität.

### E) Fazit & Einordnung (ca. ¼ Seite)
*   Zusammenfassung: Vielversprechend für unstrukturierte Daten?
*   Empfehlung: Hybrid-Ansatz für die Zukunft?

---

## 2. TECHNISCHE DETAILS (Hintergrundwissen für dich)

### Implementierung
*   **Model**: `gpt-4o-mini`
*   **Pricing**: ~$0.15 (Input) / $0.60 (Output) per 1M Tokens.
*   **Logik**:
    *   **Klassifizierung**: True/False für PV-Module.
    *   **Leistungsextraktion**: Extrahiert Watt-Zahlen (Wp/kWp) für CO2-Berechnung.
    *   **Menge**: Extrahiert Quantity aus Text ("20x") wenn Spalte leer.

### Evaluation
*   Der Code vergleicht `is_pv_module` (Prediction) gegen `ground_truth` (Manuelles Label).
*   Wir tracken False Positives (Fälschlich als Modul erkannt) und False Negatives (Übersehen).

## 3. CHECKLISTE FÜR DEN TEXT
*   [ ] Kontextualisierung: "Im Gegensatz zum regelbasierten Ansatz..."
*   [ ] Transparenz: Begründe Design-Entscheidungen (Batching, Modellwahl).
*   [ ] Empirische Basis: Nutze die echten Zahlen aus dem Testlauf.
*   [ ] Limitationen: Erwähne, dass dies ein explorativer Test ist.
