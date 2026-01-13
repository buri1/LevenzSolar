import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None

from .models import ClassificationResult

load_dotenv()

# OpenAI Pricing (USD per 1M tokens) - Updated Jan 2026
# Pricing per 1M tokens (USD approx.)
PRICING = {
    "gpt-5-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-5.2": {"input": 5.00 / 1_000_000, "output": 15.00 / 1_000_000}, # Hypothetical High-Performance
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    
    # ZhipuAI (approx exchange rate 1 RMB = 0.14 USD)
    "glm-4-plus": {"input": 0.70 / 1_000_000, "output": 0.70 / 1_000_000}, 
    "glm-4-air": {"input": 0.14 / 1_000_000, "output": 0.14 / 1_000_000},
    "glm-4-flash": {"input": 0.01 / 1_000_000, "output": 0.01 / 1_000_000},
    
    # GLM-4.5 Series (USD)
    "glm-4.5-preview": {"input": 3.00 / 1_000_000, "output": 3.00 / 1_000_000}, # Estimating high
    "glm-4.5-air": {"input": 0.20 / 1_000_000, "output": 1.10 / 1_000_000}, # Based on user text
    "glm-4.5-flash": {"input": 0.00 / 1_000_000, "output": 0.00 / 1_000_000}, # Free?
}

@dataclass
class UsageStats:
    """Track API usage and costs"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    rows_processed: int = 0
    batches_processed: int = 0
    errors: int = 0
    
    @property
    def cost_per_row(self) -> float:
        """Average cost per row in USD"""
        return self.total_cost_usd / self.rows_processed if self.rows_processed > 0 else 0
    
    @property
    def cost_per_row_eur(self) -> float:
        """Average cost per row in EUR (approx 0.92 rate)"""
        return self.cost_per_row * 0.92
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary dict for reporting"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_cost_eur": round(self.total_cost_usd * 0.92, 4),
            "rows_processed": self.rows_processed,
            "cost_per_row_usd": round(self.cost_per_row, 6),
            "cost_per_row_eur": round(self.cost_per_row_eur, 6),
            "batches_processed": self.batches_processed,
            "errors": self.errors,
        }
    
    def estimate_cost_for_rows(self, num_rows: int) -> Dict[str, float]:
        """Estimate cost for processing a given number of rows"""
        if self.rows_processed == 0:
            return {"estimated_usd": 0, "estimated_eur": 0}
        cost = self.cost_per_row * num_rows
        return {
            "estimated_usd": round(cost, 2),
            "estimated_eur": round(cost * 0.92, 2),
        }


# System prompt with power extraction rules
SYSTEM_PROMPT = """Du bist ein technischer Experte für Photovoltaik-Komponenten.

AUFGABE: Klassifiziere Produkte und extrahiere Leistungsdaten für CO2-Berechnungen.

## KLASSIFIZIERUNG (is_pv_module)

TRUE - Echte PV-Module & Kraftwerke:
- Einzelmodule: Glas-Glas, Glas-Folie, Full Black Module (Trina, Jinko, Aiko, JA Solar, Longi, Canadian Solar, Meyer Burger, Solar Fabrik, etc.)
- Sets & Systeme: "Balkonkraftwerke", "PV-Sets", "Mini-Solaranlagen" (auch wenn Wechselrichter/Speicher enthalten - das Gesamtsystem ist ein Stromerzeuger)
- Technische Signale: Wp (Watt Peak), kWp, N-Type, TOPCon, ABC-Technologie, HJT, bifazial, monokristallin, Halfcut, Shingled

FALSE - Zubehör & Dienstleistungen (MUSS FALSE SEIN, auch wenn "PV-Anlage" im Text steht!):
- Elektrische Komponenten: Wechselrichter (Inverter), Hybrid-Wechselrichter, Batteriespeicher (ohne Module), Smart Meter, DTU, Optimierer
- Montage & Infrastruktur: Dachhaken, Schienen, Kabel, Stecker, Schrauben, Ballastierung, Unterkonstruktion
- Services: "Montage", "Installation", "Anmeldung", "Gerüstbau", "Lieferung", "Versand", "Spedition"
- Unklare "Teile": Wenn nur "Teil einer PV-Anlage" steht, aber kein konkretes Modul/Set erkennbar ist -> FALSE

## LEISTUNGSEXTRAKTION (nur bei is_pv_module=true)

### MODUL-LEISTUNG (power_watts, power_source="module_wp"):
- Erkenne explizite Angaben: "450W", "450Wp", "450 Watt", "445 Wattpeak"
- Bereich: 200-600 Watt pro Modul
- Aus Produktcode wenn kein explizites W/Wp: "TSM-445NEG9R.28" → 445W, "JAM54D41" mit "445W" im Text

### ANLAGEN-LEISTUNG (power_source="anlage_kwp"):
- Bei "Balkonkraftwerk", "PV-Anlage", "Photovoltaikanlage": Suche kWp-Angabe
- "14,5 kWp" → power_watts=14500
- "2.0kWp" → power_watts=2000
- Anlagen haben typischerweise 300-2000W Gesamtleistung bei Balkonkraftwerken oder höher bei Großanlagen

### MENGE (quantity):
- Primär aus quantity-Spalte wenn > 1
- Aus Text: "20x Module", "15 Stück", "12 Stk", "5 pcs"
- IGNORIERE Maßangaben wie "1755×1038mm" 
- Bei Anlagen/Balkonkraftwerken: quantity = 1 (ist ein Paket)
- Falls unklar: quantity = null

### GESAMTLEISTUNG (total_power_watts):
- Berechne: power_watts * quantity (wenn beide vorhanden)
- Bei Anlagen (kWp): total_power_watts = power_watts (quantity ist bereits 1)

## AUSGABE-FORMAT
Antworte AUSSCHLIESSLICH als valides JSON-Objekt:
{
  "results": [
    {
      "product_id": "String",
      "product_name": "String (kurz, max 50 Zeichen)",
      "is_pv_module": Boolean,
      "Confidence": Float (0.0-1.0),
      "Reasoning": "Kurze Begründung (max 100 Zeichen)",
      "power_watts": Integer oder null,
      "quantity": Integer oder null,
      "total_power_watts": Integer oder null,
      "power_source": "module_wp" | "anlage_kwp" | "productcode" | null
    }
  ]
    }
  ]
}
WICHTIG: Erstelle für JEDES Eingabe-Produkt einen Eintrag im `results` Array, auch wenn `is_pv_module` false ist!"""


class LLMClient:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self.usage = UsageStats()
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            # Increase timeout for GPT-5 reasoning models (default is often 60s/600s)
            self.client = OpenAI(api_key=api_key, timeout=1200.0)
            print(f"🤖 Initialized OpenAI Client ({model}) with 20min timeout")
            
        elif provider == "zhipuai":
            if ZhipuAI is None:
                raise ImportError("zhipuai not installed. Run 'pip install zhipuai'")
            
            # Map user env var to expected key if needed
            api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("ZAI_API_KEY")
            if not api_key:
                raise ValueError("ZHIPUAI_API_KEY (or ZAI_API_KEY) not found")
                
            self.client = ZhipuAI(api_key=api_key)
            print(f"🤖 Initialized ZhipuAI Client ({model})")
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
    def _update_usage(self, response, rows_in_batch: int):
        """Update usage statistics from API response"""
        usage = response.usage
        self.usage.prompt_tokens += usage.prompt_tokens
        self.usage.completion_tokens += usage.completion_tokens
        self.usage.total_tokens += usage.total_tokens
        self.usage.rows_processed += rows_in_batch
        self.usage.batches_processed += 1
        
        # Calculate cost
        pricing = PRICING.get(self.model, PRICING["gpt-4o-mini"])
        cost = (usage.prompt_tokens * pricing["input"]) + \
               (usage.completion_tokens * pricing["output"])
        self.usage.total_cost_usd += cost

    def classify_batch(self, batch: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Classify a batch of products and extract power data"""
        # Prepare content: show all fields as context
        products_text = ""
        # Columns to exclude from the LLM input to prevent leakage
        excluded_cols = {'is_pv_module', 'Confidence', 'Reasoning', 'power_watts', 'total_power_watts', 'power_source'}
        
        for item in batch:
            # Convert item dict to a string representation, filtering out None/NaN and excluded columns
            item_data = {k: v for k, v in item.items() 
                         if pd.notnull(v) and k not in excluded_cols}
            
            item_str = ", ".join([f"{k}: {v}" for k, v in item_data.items()])
            products_text += f"- {item_str}\n"
        
        user_prompt = f"Analysiere folgende Produkte und gib das JSON zurück:\n{products_text}"

        retries = 3
        for attempt in range(retries):
            try:
                # Prepare arguments
                kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    "response_format": {"type": "json_object"},
                }
                
                # Reasoning models (o1, gpt-5) do not support temperature
                if "o1" in self.model or "gpt-5" in self.model:
                    pass # Temp not supported
                else:
                    kwargs["temperature"] = 0.1
                
                # GPT-5 Series: Set appropriate max_completion_tokens
                # Without this, batches may truncate or timeout
                if "gpt-5" in self.model:
                    if "gpt-5.2" in self.model:
                        kwargs["max_completion_tokens"] = 16384  # 16k for flagship
                    else:  # gpt-5-mini
                        kwargs["max_completion_tokens"] = 8192   # 8k for mini

                response = self.client.chat.completions.create(**kwargs)
                
                if hasattr(response, 'usage'):
                   # print(f"DEBUG USAGE: {response.usage}")
                   pass
                
                # Update usage stats
                self._update_usage(response, len(batch))
                
                content = response.choices[0].message.content
                if not content:
                    print(f"❌ Empty response content from API for batch")
                    raise ValueError("Empty response from API")
                
                # DEBUG: Print content preview
                # print(f"DEBUG Response: {content[:100]}...")

                data = json.loads(content)
                results_data = data.get("results", [])
                
                if not results_data:
                    print(f"⚠️ No 'results' found in JSON. Content: {content[:500]}...")
                
                parsed_results = []
                for item in results_data:
                    # Validate with Pydantic
                    try:
                        res = ClassificationResult(**item)
                        parsed_results.append(res)
                    except Exception as e:
                        print(f"⚠️ Validation error for item {item.get('product_id', 'unknown')}: {e}")
                        self.usage.errors += 1
                        
                return parsed_results

            except json.JSONDecodeError as e:
                print(f"JSON Error: {e}. Retrying...")
                self.usage.errors += 1
                if attempt == retries - 1:
                    raise e
                time.sleep(1)
            except Exception as e:
                # Retry logic with exponential backoff
                if attempt == retries - 1:
                    self.usage.errors += 1
                    raise e
                wait_time = 2 ** attempt
                print(f"API Error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                 
        return []
    
    def get_usage_report(self) -> str:
        """Get a formatted usage report"""
        stats = self.usage.get_summary()
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                     API USAGE REPORT                         ║
╠══════════════════════════════════════════════════════════════╣
║  Model:              {self.model:<39} ║
║  Rows Processed:     {stats['rows_processed']:<39} ║
║  Batches Processed:  {stats['batches_processed']:<39} ║
║  Errors:             {stats['errors']:<39} ║
╠══════════════════════════════════════════════════════════════╣
║  TOKENS                                                      ║
║  ├─ Prompt:          {stats['prompt_tokens']:<39} ║
║  ├─ Completion:      {stats['completion_tokens']:<39} ║
║  └─ Total:           {stats['total_tokens']:<39} ║
╠══════════════════════════════════════════════════════════════╣
║  COSTS                                                       ║
║  ├─ Total (USD):     ${stats['total_cost_usd']:<38} ║
║  ├─ Total (EUR):     €{stats['total_cost_eur']:<38} ║
║  ├─ Per Row (USD):   ${stats['cost_per_row_usd']:<38} ║
║  └─ Per Row (EUR):   €{stats['cost_per_row_eur']:<38} ║
╠══════════════════════════════════════════════════════════════╣
║  PROJECTIONS                                                 ║"""
        
        # Add projections for common dataset sizes
        for num_rows in [1000, 10000, 70000]:
            est = self.usage.estimate_cost_for_rows(num_rows)
            report += f"\n║  ├─ {num_rows:,} rows:       ${est['estimated_usd']:<10} (€{est['estimated_eur']}){'':>16} ║"
        
        report += """
╚══════════════════════════════════════════════════════════════╝"""
        return report
