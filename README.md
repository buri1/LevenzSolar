# ☀️ LEVENZ SOLAR

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenAI](https://img.shields.io/badge/AI-OpenAI%20GPT-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Automatisierte Klassifizierung von Handwerker-Rechnungen zur Nachhaltigkeitsanalyse.**

Dieses Projekt wurde entwickelt, um Produktdaten aus der Datenbank einer Handwerkersoftware zu analysieren. Ziel ist es, **Solarmodule und PV-Systeme** präzise von reinem Zubehör oder Dienstleistungen zu unterscheiden. Dies ermöglicht eine detaillierte Auswertung im Hinblick auf Nachhaltigkeitskennzahlen.

---

## 🚀 Funktionen

*   **KI-gestützte Klassifizierung**: Nutzt moderne Large Language Models (OpenAI GPT), um Produktnamen und -beschreibungen zu verstehen.
*   **Präzise Unterscheidung**: Trennt eigentliche Stromerzeuger (PV-Module, Balkonkraftwerke) von Peripherie (Wechselrichter, Kabel, Montage).
*   **Transparente Entscheidungen**: Jede Klassifizierung enthält eine Wahrscheinlichkeit (`Confidence`) und eine Begründung (`Reasoning`).
*   **Evaluations-Tools**: Integrierte Skripte zum Abgleich der Ergebnisse mit "Ground Truth"-Daten für Qualitätskontrollen.

## 📋 Kriterien der Klassifizierung

Das System unterscheidet nach folgenden strengen Kriterien:

### ✅ IST ein PV-Modul / System
*   **Einzelmodule**: Glas-Glas, Glas-Folie, Full Black (z.B. Trina, Jinko, Meyer Burger).
*   **Komplettsysteme**: Balkonkraftwerke, Mini-Solaranlagen (inkl. Modulen).
*   **Technische Indikatoren**: Angabe von Watt-Peak (Wp, kWp), Zelltechnologien (TOPCon, HJT).

### ❌ IST KEIN PV-Modul (Zubehör/Service)
*   **Elektronik**: Reine Wechselrichter, Batteriespeicher (ohne Module), Smart Meter.
*   **Infrastruktur**: Dachhaken, Montageschienen, Kabel, Stecker.
*   **Dienstleistungen**: Montage, Anmeldung, Gerüstbau.
*   **Fremdgewerke**: Sanitär, allgemeine Elektroinstallation.

---

## 🛠️ Installation

### 1. Repository klonen
```bash
git clone https://github.com/your-username/levenz-solar.git
cd levenz-solar
```

### 2. Abhängigkeiten installieren
Es wird empfohlen, ein virtuelles Environment zu nutzen.
```bash
pip install -r requirements.txt
```

### 3. Umgebungsvariablen
Erstelle eine `.env` Datei im Hauptverzeichnis (siehe `.env.example`) und füge deinen OpenAI API Key hinzu:

```env
OPENAI_API_KEY=sk- dein_key_hier
```

---

## 💻 Verwendung

### Datensatz vorbereiten
Lege deine Eingabedaten als CSV-Datei unter `data/input.csv` ab. Die Datei muss mindestens folgende Spalten enthalten:
*   `product_id`: Eindeutige ID
*   `product_name`: Name/Beschreibung des Produkts

### Klassifizierung starten
Führe das Hauptskript aus, um die Produkte zu analysieren:

```bash
python main.py
```
Das Ergebnis wird in `data/output.csv` gespeichert.

### Evaluation (Optional)
Wenn du Testdaten mit bekannten Lösungen hast (`data/Testdaten Mit Loesung CSV.csv` oder ähnlich), kannst du die Qualität der KI überprüfen:

```bash
python evaluate.py
```
Dies gibt eine Genauigkeitsstatistik aus und speichert Abweichungen in `data/evaluation_errors.csv`.

---

## 📂 Projektstruktur

```plaintext
levenz-solar/
├── data/
│   ├── input.csv             # Deine Eingabedaten
│   ├── output.csv            # Ergebnisse der KI
│   └── evaluation_errors.csv # Fehleranalyse (generiert)
├── src/
│   ├── llm_client.py         # Logik für OpenAI API & Prompting
│   └── models.py             # Datenmodelle (Pydantic)
├── main.py                   # Hauptprogramm
├── evaluate.py               # Skript zur Qualitätsprüfung
├── .env                      # API Keys (nicht im Git)
└── requirements.txt          # Python Abhängigkeiten
```

---

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte erstelle einen Pull Request oder eröffne ein Issue für Verbesserungsvorschläge.

## 📄 Lizenz

Bachelor Thesis Project.
