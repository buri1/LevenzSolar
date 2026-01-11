# Exkurs: LLM-basierte PV-Modul Klassifizierung

## Motivation und Hintergrund

Neben dem regelbasierten Klassifizierungsansatz wurde ein alternativer Ansatz auf Basis von Large Language Models (LLMs) untersucht. Die Hypothese war, dass LLMs durch ihr implizites Wissen über Produktkategorien und technische Terminologie eine robuste Klassifizierung ohne explizite Regelformulierung ermöglichen könnten.

## Methodik

### Technische Implementierung

Die Implementierung nutzt die OpenAI API mit dem Modell `gpt-4o-mini`, welches für Klassifizierungsaufgaben mit großen Datenmengen optimiert ist. Der Ansatz basiert auf drei Kernkomponenten:

1. **Batch-Processing:** Anstatt jeden Datensatz einzeln zu verarbeiten, werden mehrere Produkte (typischerweise 10-30) in einer Anfrage gebündelt. Dies reduziert den API-Overhead und ermöglicht eine effizientere Token-Nutzung.

2. **Strukturiertes Prompting:** Ein sorgfältig entwickelter System-Prompt definiert die Klassifizierungskriterien und das erwartete Ausgabeformat (JSON).

3. **Leistungsextraktion:** Zusätzlich zur binären Klassifizierung extrahiert das System die elektrische Leistung (in Watt) aus Produktbeschreibungen, um eine Berechnung der CO₂-Einsparungen zu ermöglichen.

### System-Prompt Design

Der verwendete System-Prompt folgt einem strukturierten Aufbau:

```
KLASSIFIZIERUNG (is_pv_module)
TRUE - Echte PV-Module & Kraftwerke:
  - Einzelmodule (Trina, Jinko, Aiko, JA Solar, etc.)
  - Sets & Systeme (Balkonkraftwerke, PV-Sets)
  - Technische Signale: Wp, kWp, N-Type, TOPCon, etc.

FALSE - Zubehör & Dienstleistungen:
  - Wechselrichter, Speicher, Smart Meter
  - Montage-Material, Services

LEISTUNGSEXTRAKTION (nur bei is_pv_module=true)
  - Modul-Leistung: "450W", "450Wp" → 200-600W
  - Anlagen-Leistung: "14,5 kWp" → 14500W
```

Die Kriterien wurden bewusst kompakt gehalten, um Token zu sparen und gleichzeitig das semantische Verständnis des Modells zu nutzen – im Gegensatz zum regelbasierten Ansatz, der explizite Pattern-Matching-Regeln benötigt.

### Parallelisierung und Skalierung

Für die Verarbeitung großer Datensätze wurde eine parallele Verarbeitung implementiert. Unter Berücksichtigung der API-Rate-Limits (500 Requests/Minute für Standard-Tier) können mehrere Batches gleichzeitig verarbeitet werden, was die Gesamtverarbeitungszeit erheblich reduziert.

## Ergebnisse

### Klassifizierungsgenauigkeit

Die Evaluation gegen den annotierten Testdatensatz (N=154 Einträge) ergab folgende Metriken:

| Metrik | Wert |
|--------|------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |

Die Confusion Matrix zeigt:
- True Positives: 29 (korrekt als PV-Modul erkannt)
- True Negatives: 125 (korrekt als Nicht-PV-Modul erkannt)
- False Positives: 0
- False Negatives: 0

### Leistungsextraktion

Von den 29 korrekt klassifizierten PV-Modulen konnte bei 28 (96,6%) erfolgreich eine Leistungsangabe extrahiert werden. Die Extraktion erfolgte aus verschiedenen Quellen:
- Explizite Watt-Angaben (z.B. "450Wp"): 85%
- Produktcodes (z.B. "TSM-445NEG9R.28"): 10%
- kWp-Angaben bei Anlagen: 5%

### Kostenanalyse

Die API-Kosten wurden systematisch erfasst:

| Parameter | Wert |
|-----------|------|
| Testdatensatz | 154 Zeilen |
| Batch-Size | 10 |
| Modell | gpt-4o-mini |
| Kosten gesamt | ~$0.02 |
| Kosten pro Zeile | ~$0.00013 |

**Hochrechnung auf Produktionsdaten:**
| Datensatz | Geschätzte Kosten |
|-----------|-------------------|
| 1.000 Zeilen | ~$0.13 |
| 10.000 Zeilen | ~$1.30 |
| 70.000 Zeilen | ~$9.10 |

Die Verarbeitungszeit betrug ca. 2 Sekunden pro Batch bei sequenzieller Verarbeitung. Mit paralleler Verarbeitung (5 Worker) reduziert sich die Gesamtzeit für 70.000 Zeilen auf geschätzte 1,5-2 Stunden.

## Vergleich mit dem regelbasierten Ansatz

| Aspekt | Regelbasiert | LLM-basiert |
|--------|--------------|-------------|
| Entwicklungsaufwand | Hoch (Regelformulierung) | Mittel (Prompt Engineering) |
| Anpassbarkeit | Explizite Regeländerungen | Prompt-Modifikation |
| Erklärbarkeit | Hoch (deterministische Regeln) | Mittel (Reasoning im Output) |
| Wartbarkeit | Komplex bei neuen Produkten | Einfacher durch Generalisierung |
| Kosten | Einmalig (Entwicklung) | Laufend (API-Kosten) |
| Geschwindigkeit | Sehr schnell | Langsamer (API-Latenz) |
| Offline-Fähigkeit | Ja | Nein |

## Diskussion

Der LLM-basierte Ansatz erreichte auf dem Testdatensatz eine perfekte Klassifizierungsgenauigkeit. Dies lässt sich auf mehrere Faktoren zurückführen:

1. **Semantisches Verständnis:** Das Modell kann kontextbezogene Entscheidungen treffen und versteht implizit, dass ein "Balkonkraftwerk" trotz enthaltener Wechselrichter primär ein stromerzeugender Artikel ist.

2. **Robustheit gegenüber Variationen:** Unterschiedliche Schreibweisen (z.B. "450W", "450 Watt", "450Wp") werden ohne explizite Regeln korrekt interpretiert.

3. **Generalisierung:** Neue, unbekannte Produktnamen können anhand technischer Merkmale klassifiziert werden.

Ein kritischer Aspekt ist die Abhängigkeit von externen API-Diensten und die damit verbundenen laufenden Kosten. Für einen produktiven Einsatz auf dem gesamten Datensatz (~70.000 Einträge) entstehen Kosten von ca. 10€ – was im Vergleich zum Entwicklungsaufwand des regelbasierten Ansatzes moderat ist.

## Fazit

Der LLM-basierte Ansatz bietet eine valide Alternative zum regelbasierten Verfahren. Die Stärken liegen in der schnellen Implementierung und der Robustheit gegenüber Variationen in den Eingabedaten. Für einen produktiven Einsatz empfiehlt sich eine Kombination beider Ansätze: Der regelbasierte Ansatz für die Massendaten (kostengünstig, schnell) und der LLM-Ansatz für Grenzfälle oder als Validierungsinstrument.
