# ☀️ LEVENZ SOLAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT--5-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Automatisierte Klassifizierung von Handwerker-Rechnungen zur Nachhaltigkeitsanalyse.**

Dieses Projekt wurde entwickelt, um Produktdaten aus der Datenbank einer Handwerkersoftware zu analysieren. Ziel ist es, **Solarmodule und PV-Systeme** präzise von reinem Zubehör oder Dienstleistungen zu unterscheiden und die **elektrische Leistung (kWp)** für CO₂-Berechnungen zu extrahieren.

---

## 🚀 Funktionen

*   **KI-gestützte Klassifizierung**: Nutzt OpenAI GPT-5-mini, um Produktnamen und -beschreibungen zu verstehen.
*   **Leistungsextraktion**: Extrahiert automatisch Watt/kWp-Angaben für CO₂-Berechnungen.
*   **Kostentracking**: Detaillierter Bericht über API-Kosten pro Zeile und Hochrechnungen.
*   **Parallele Verarbeitung**: Skalierbar für große Datensätze (70k+ Zeilen).
*   **Transparente Entscheidungen**: Jede Klassifizierung enthält eine Wahrscheinlichkeit (`Confidence`) und eine Begründung (`Reasoning`).
*   **Evaluations-Tools**: Precision, Recall, F1-Score mit Confusion Matrix.

## 📊 Ergebnisse

| Metrik | Wert |
|--------|------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1-Score | 100% |

**Kosten-Prognose (gpt-5-mini):**
| Datensatz | Kosten |
|-----------|--------|
| 1,000 Zeilen | ~€0.16 |
| 70,000 Zeilen | ~€11.39 |

---

## 📋 Kriterien der Klassifizierung

### ✅ IST ein PV-Modul / System
*   **Einzelmodule**: Glas-Glas, Glas-Folie, Full Black (z.B. Trina, Jinko, Meyer Burger).
*   **Komplettsysteme**: Balkonkraftwerke, Mini-Solaranlagen (inkl. Modulen).
*   **Technische Indikatoren**: Angabe von Watt-Peak (Wp, kWp), Zelltechnologien (TOPCon, HJT).

### ❌ IST KEIN PV-Modul (Zubehör/Service)
*   **Elektronik**: Reine Wechselrichter, Batteriespeicher (ohne Module), Smart Meter.
*   **Infrastruktur**: Dachhaken, Montageschienen, Kabel, Stecker.
*   **Dienstleistungen**: Montage, Anmeldung, Gerüstbau.

---

## 🛠️ Installation

### 1. Repository klonen
```bash
git clone https://github.com/burakisme/LevenzSolar.git
cd LevenzSolar
```

### 2. Abhängigkeiten installieren
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Umgebungsvariablen
Erstelle eine `.env` Datei:
```env
OPENAI_API_KEY=sk-dein_key_hier
```

---

## 💻 Verwendung

### Klassifizierung starten
```bash
# Standard (alle Zeilen, Batch-Size 10)
python main.py

# Schneller Test (10 Zeilen)
python main.py --limit 10

# Mit Parallelisierung (schneller)
python main.py --batch-size 20 --parallel 5

# Andere Input-Datei
python main.py --input data/meine_daten.csv --output data/ergebnis.csv
```

### CLI Optionen
| Option | Default | Beschreibung |
|--------|---------|--------------|
| `--batch-size` | 10 | Produkte pro API-Anfrage |
| `--parallel` | 1 | Parallele API-Worker |
| `--model` | gpt-5-mini | OpenAI Modell |
| `--limit` | - | Max. Zeilen (für Tests) |
| `--input` | data/Testdaten... | Input CSV |
| `--output` | data/output.csv | Output CSV |

### Evaluation
```bash
python evaluate.py
```

---

## 📂 Projektstruktur

```plaintext
LevenzSolar/
├── data/
│   ├── output.csv              # Ergebnisse der KI
│   └── evaluation_errors.csv   # Fehleranalyse
├── docs/
│   └── bachelorarbeit_exkurs.md  # Dokumentation für Thesis
├── src/
│   ├── llm_client.py           # OpenAI API + Kostentracking
│   ├── models.py               # Datenmodelle (Pydantic)
│   └── processor.py            # CSV Verarbeitung
├── main.py                     # Hauptprogramm
├── evaluate.py                 # Qualitätsprüfung
├── .env                        # API Keys (nicht im Git)
└── requirements.txt            # Python Abhängigkeiten
```

---

## 📄 Dokumentation

Für die Bachelorarbeit siehe: [`docs/bachelorarbeit_exkurs.md`](docs/bachelorarbeit_exkurs.md)

---

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte erstelle einen Pull Request oder eröffne ein Issue.