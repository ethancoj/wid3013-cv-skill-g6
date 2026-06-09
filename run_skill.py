"""
run_skill.py
============
The OpenClaw entry point for the Document Restoration Skill.

How OpenClaw calls this:
  python run_skill.py <image_path> [category]

Examples:
  python run_skill.py samples/test_image.jpg
  python run_skill.py samples/test_image.jpg "Field Notes"

Output:
  - Prints a structured Telegram-ready response
  - Saves the HTML dashboard to tests/test_output.html
  - Saves the Golden Payload JSON to samples/output/payload.json
"""

import json
from dotenv import load_dotenv
load_dotenv()
import os
import sys
from jinja2 import Environment, FileSystemLoader
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemma-4-26b-a4b-it")

def extract_insights_with_llm(extracted_text):
    prompt = f"""You are a document analysis assistant for anthropology and social science students.

Given this OCR-extracted text from a historical document, provide a brief analysis covering:
1. Document Type (e.g. letter, newspaper, catalog, legal record)
2. Estimated Era/Period
3. Key Entities (people, places, organizations mentioned)
4. Historical Context (brief relevance to social science research)
5. Research Value (what a student could use this document for)

Keep each section to 1-2 sentences. Be factual based only on what the text contains.

Text:
{extracted_text[:3000]}"""

    try:
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=1000,
        ))
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ LLM insights failed: {e}")
        return ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_HTML = os.path.join(BASE_DIR, "tests", "test_output.html")
OUTPUT_DIR = os.path.join(BASE_DIR, "samples", "output")
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.insert(0, SRC_DIR)

def render_dashboard(payload):
    """Render the HTML dashboard from the Golden Payload."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("dashboard_template.html")

    html_output = template.render(
        doc_id=payload["metadata"]["doc_id"],
        timestamp=payload["metadata"]["timestamp"],
        filename=payload["metadata"]["filename"],
        health_score=payload["metadata"]["health_score"],
        width=payload["metadata"]["width"],
        height=payload["metadata"]["height"],
        proc_time=payload["metadata"]["processing_time_seconds"],
        status=payload["metadata"]["status"],
        img_before=payload["images"]["before_url"],
        img_after=payload["images"]["after_url"],
        extracted_text=payload["content"]["extracted_text"],
        annotated_url=payload["images"]["annotated_url"],
        layout_url=payload["images"]["layout_url"],
        table_headers=[],
        table_rows=[],
        overall_pct=payload["insights"]["confidence"]["overall_pct"],
        high_conf_pct=payload["insights"]["confidence"]["high_confidence_pct"],
        chart_data=payload["insights"]["confidence"]["distribution_chart_data"],
        audit_steps=payload["insights"]["audit_trail"],
        ai_insights=payload.get("ai_insights", ""),
    )

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_output)

def format_telegram_response(payload, category, dashboard_path):
    """Format the structured Telegram-ready output message."""
    status = payload["metadata"]["status"]
    status_emoji = "✅" if status == "success" else "❌"

    text_preview = payload["content"]["extracted_text"][:200].strip()
    if len(payload["content"]["extracted_text"]) > 200:
        text_preview += "..."

    audit_steps = payload["insights"]["audit_trail"]
    steps_text = "\n".join([f"  • {s['name']}" for s in audit_steps])

    response = f"""
{status_emoji} Document processed successfully

📄 File: {payload["metadata"]["filename"]}
🏥 Health Score: {payload["metadata"]["health_score"]}/100
🔍 Confidence: {payload["insights"]["confidence"]["overall_pct"]}%
⏱ Processing time: {payload["metadata"]["processing_time_seconds"]}s

📝 Extracted Text Preview:
{text_preview}

🔧 Processing steps applied:
{steps_text}

📊 Full dashboard saved to:
{dashboard_path}

📁 Saved under category: {category}
""".strip()

    return response

def main():
    # --- Parse arguments ---
    if len(sys.argv) < 2:
        print("Usage: python run_skill.py <image_path> [category]")
        print("Example: python run_skill.py samples/test_image.jpg 'Field Notes'")
        sys.exit(1)

    image_path = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "Uncategorized"

    # --- Validate image exists ---
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        print("Please make sure your image is in the samples/ folder.")
        sys.exit(1)

    # --- Load CV processing script (Script 4 is the final chosen script) ---
    try:
        from src.cv_processing_4 import process_document
    except ImportError:
        print("❌ Error: cv_processing_4.py not found in src/")
        print("Make sure your teammate has pushed their CV script.")
        sys.exit(1)

    # --- Run CV processing ---
    print(f"Processing image: {image_path}")
    print(f"Category: {category}")
    print("Running CV pipeline...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = process_document(image_path, OUTPUT_DIR)

    # --- LLM document insights ---
    print("Generating AI insights...")
    ai_insights = extract_insights_with_llm(payload["content"]["extracted_text"])
    payload["ai_insights"] = ai_insights
    payload["content"]["tables"] = []

    # --- Save Golden Payload JSON ---
    json_path = os.path.join(OUTPUT_DIR, "payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # --- Render dashboard ---
    render_dashboard(payload)

    # --- Save to vault (category folder) ---
    vault_dir = os.path.join(BASE_DIR, "workspace", "document_vault", category.replace(" ", "_"))
    os.makedirs(vault_dir, exist_ok=True)
    vault_json = os.path.join(vault_dir, f"{payload['metadata']['doc_id']}_payload.json")
    with open(vault_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # --- Print Telegram response ---
    response = format_telegram_response(payload, category, OUTPUT_HTML)
    print("\n" + "="*50)
    print("TELEGRAM RESPONSE:")
    print("="*50)
    print(response)

if __name__ == "__main__":
    main()