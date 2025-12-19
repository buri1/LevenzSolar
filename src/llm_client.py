import os
import json
import time
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI, APIError, RateLimitError
from dotenv import load_dotenv
from .models import ClassificationResult

load_dotenv()

class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Checking if it's set in the environment, if not, user needs to set it.
            # We won't raise here strictly to allow instantiation, but methods will fail or we raise if we want strictness.
            # PRD: "API key setup".
            pass 
        
        # If key is missing, OpenAI client might raise error later or now.
        # Let's allow it to be None and check in calls or just let OpenAI lib handle it.
        # However, for reproducible script, checking here is good.
        if not self.api_key:
             print("Warning: OPENAI_API_KEY not found in environment variables.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5-mini"
       # self.temperature = 0.3

    def classify_batch(self, batch: List[Dict[str, Any]]) -> List[ClassificationResult]:
        # Prepare content: Keep showing all fields as context is helpful, but rely on semantic understanding
        products_text = ""
        for item in batch:
            # Convert item dict to a string representation, filtering out None/NaN
            item_str = ", ".join([f"{k}: {v}" for k, v in item.items() if pd.notnull(v)])
            products_text += f"- {item_str}\n"
        
        system_prompt = (
            "Du bist ein technischer Experte für Photovoltaik-Komponenten. Deine Aufgabe ist es, Produkte präzise zu klassifizieren.\n\n"
            "ZIEL: Identifiziere Artikel, deren Hauptzweck die Erzeugung von Solarstrom ist ('is_pv_module': true).\n\n"
            "KRITERIEN FÜR 'TRUE' (Echte PV-Module & Kraftwerke):\n"
            "1. Einzelmodule: Alle Glas-Glas, Glas-Folie oder Full Black Module (z.B. Trina Vertex, Jinko Tiger, Aiko Neostar, Meyer Burger, Solar Fabrik, JA Solar, Longi, Canadian Solar).\n"
            "2. Sets & Systeme: \"Balkonkraftwerke\", \"PV-Sets\" oder \"Mini-Solaranlagen\". Auch wenn diese Wechselrichter/Speicher enthalten, ist das Gesamtsystem als Stromerzeuger (Modul-Einheit) zu werten.\n"
            "3. Technische Signale: Wp (Watt Peak), kWp, N-Type, TOPCon, ABC-Technologie, HJT, bifazial, monokristallin, Halfcut, Shingled.\n"
            "4. Schwellenwert: Wenn eine Leistungsangabe in Wp/kWp vorhanden ist und es sich nicht um reines Zubehör handelt, ist es ein PV-Modul/System.\n\n"
            "KRITERIEN FÜR 'FALSE' (Zubehör & Dienstleistung):\n"
            "1. Rein elektrische Komponenten: Wechselrichter (Inverter), Batteriespeicher (ohne Module), Smart Meter, DTU, Optimierer (BRC/Tigo).\n"
            "2. Montage & Infrastruktur: Dachhaken, Schienen, Kabel, Stecker, Schrauben, Ballastierung.\n"
            "3. Services: Montageleistungen, Anmeldung beim Netzbetreiber, Gerüstbau, Lieferpauschalen.\n"
            "4. Fremdgewerke: Heizungssanitär (Wandscheiben, Rohre), allgemeine Elektrotechnik (FI-Schalter).\n\n"
            "AUSGABE-FORMAT:\n"
            "Antworte AUSSCHLIESSLICH als valides JSON-Objekt mit einem Array 'results'.\n"
            "{\n"
            "  \"results\": [\n"
            "    {\n"
            "      \"product_id\": \"String\",\n"
            "      \"product_name\": \"String\",\n"
            "      \"is_pv_module\": Boolean,\n"
            "      \"Confidence\": Float (0.0-1.0),\n"
            "      \"Reasoning\": \"Kurze technische Begründung (z.B. 'Balkonkraftwerk-Set inkl. Module' oder 'Reine Dienstleistung Anmeldung')\"\n"
            "    }\n"
            "  ]\n"
            "}"
        )

        user_prompt = f"Analysiere folgende Produkte und gib das JSON zurück:\n{products_text}"

        retries = 3
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    #temperature=self.temperature
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from API")

                data = json.loads(content)
                results_data = data.get("results", [])
                
                parsed_results = []
                for item in results_data:
                    # Validate with Pydantic
                    try:
                        res = ClassificationResult(**item)
                        parsed_results.append(res)
                    except Exception as e:
                        print(f"⚠️ Validation error for item {item.get('product_id', 'unknown')}: {e}")
                        
                return parsed_results

            except (RateLimitError, APIError) as e:
                if attempt == retries - 1:
                    raise e
                wait_time = 2 ** attempt
                print(f"API Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except json.JSONDecodeError as e:
                print(f"JSON Error: {e}. Retrying...")
                if attempt == retries - 1:
                    raise e
                time.sleep(1)
            except Exception as e:
                 print(f"Result processing error: {e}")
                 raise e
                 
        return []
