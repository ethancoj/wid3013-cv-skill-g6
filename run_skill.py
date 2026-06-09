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

# --- Section markers for parsing LLM response ---
SECTION_MARKERS = {
    "artifact_profile": "ARTIFACT PROFILE",
    "provenance": "PROVENANCE",
    "anthropological_matrix": "ANTHROPOLOGICAL MATRIX",
    "taxonomy": "TAXONOMY",
    "validity_bias": "VALIDITY AND BIAS",
    "executive_summary": "EXECUTIVE SUMMARY",
}


def extract_anthropological_analysis(extracted_text):
    """Send extracted OCR text to Gemma for anthropological analysis.
    
    Returns a dict with keys matching the 6 dashboard modules.
    Since Gemma outputs free-form text (not JSON), we ask for
    clearly labeled sections and parse them by header.
    """
    prompt = f"""You are a document analysis assistant for anthropology and social science students.
Analyze the following OCR-extracted text from a restored historical document.
Provide your analysis under EXACTLY these 6 section headers, in this order.
Write each header on its own line prefixed with "##".

## ARTIFACT PROFILE
Describe the physical format (bound book, loose letter, ledger, map, etc.), the likely substrate and medium (paper type, ink), and the apparent condition based on OCR quality. 1-2 sentences.

## PROVENANCE
State the estimated time period, geographic origin, author or creator (if identifiable), and the document genre (administrative, personal, legal, financial, religious, correspondence). 1-2 sentences.

## ANTHROPOLOGICAL MATRIX
Assess the author's positionality (outsider, insider, elite, marginalized), the intended audience (private, state, public, peers), and any dominant power dynamics visible in the text (colonialism, capitalism, religion, bureaucracy). Note any notable "silences" -- whose voices are absent. 2-4 sentences.

## TAXONOMY
List key locations, people, organizations, and thematic tags (kinship, ritual, economics, warfare, law, agriculture, medicine). Note any non-dominant-language or indigenous terms used. Use short bullet-style entries.

## VALIDITY AND BIAS
Assess proximity to events (first-hand, second-hand, hearsay), overt bias level (low to high), and whether the document is more valuable for understanding the author's culture or the subject's culture. 1-2 sentences.

## EXECUTIVE SUMMARY
Write a 2-sentence summary of what this document is and why it matters for social science research.

Be factual. Only analyze what the text contains. Do not invent information.

OCR Text:
{extracted_text[:3000]}"""

    try:
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=4000,
        ))
        raw = response.text.strip()
        return parse_sections(raw)
    except Exception as e:
        print(f"LLM anthropological analysis failed: {e}")
        return {}


def parse_sections(text):
    """Parse the LLM response into a dict keyed by module name.
    
    Looks for '## SECTION HEADER' lines and captures everything
    between them as that section's content.
    """
    result = {}
    lines = text.split("\n")
    current_key = None
    current_lines = []

    for line in lines:
        stripped = line.strip().lstrip("#").strip()
        matched_key = None
        for key, marker in SECTION_MARKERS.items():
            if stripped.upper().startswith(marker):
                matched_key = key
                break

        if matched_key:
            # Save previous section
            if current_key and current_lines:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = matched_key
            current_lines = []
        else:
            if current_key:
                current_lines.append(line)

    # Save last section
    if current_key and current_lines:
        result[current_key] = "\n".join(current_lines).strip()

    return result


# --- Paths ---
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
        overall_pct=payload["insights"]["confidence"]["overall_pct"],
        high_conf_pct=payload["insights"]["confidence"]["high_confidence_pct"],
        chart_data=payload["insights"]["confidence"]["distribution_chart_data"],
        audit_steps=payload["insights"]["audit_trail"],
        ai_analysis=payload.get("ai_analysis", {}),
    )

    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_output)


def format_telegram_response(payload, category, dashboard_path):
    """Format the structured Telegram-ready output message."""
    status = payload["metadata"]["status"]
    status_emoji = "\u2705" if status == "success" else "\u274c"

    text_preview = payload["content"]["extracted_text"][:200].strip()
    if len(payload["content"]["extracted_text"]) > 200:
        text_preview += "..."

    audit_steps = payload["insights"]["audit_trail"]
    steps_text = "\n".join([f"  \u2022 {s['name']}" for s in audit_steps])

    # Include executive summary if available
    ai = payload.get("ai_analysis", {})
    summary_text = ai.get("executive_summary", "")
    summary_block = f"\n\U0001f9e0 AI Summary:\n{summary_text}" if summary_text else ""

    response = f"""
{status_emoji} Document processed successfully

\U0001f4c4 File: {payload["metadata"]["filename"]}
\U0001f3e5 Health Score: {payload["metadata"]["health_score"]}/100
\U0001f50d Confidence: {payload["insights"]["confidence"]["overall_pct"]}%
\u23f1 Processing time: {payload["metadata"]["processing_time_seconds"]}s

\U0001f4dd Extracted Text Preview:
{text_preview}
{summary_block}

\U0001f527 Processing steps applied:
{steps_text}

\U0001f4ca Full dashboard saved to:
{dashboard_path}

\U0001f4c1 Saved under category: {category}
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
        print(f"Error: Image not found at {image_path}")
        print("Please make sure your image is in the samples/ folder.")
        sys.exit(1)

    # --- Load CV processing script ---
    try:
        from src.cv_processing_4 import process_document
    except ImportError:
        print("Error: cv_processing_4.py not found in src/")
        print("Make sure your teammate has pushed their CV script.")
        sys.exit(1)

    # --- Run CV processing ---
    print(f"Processing image: {image_path}")
    print(f"Category: {category}")
    print("Running CV pipeline...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = process_document(image_path, OUTPUT_DIR)

    # --- LLM anthropological analysis ---
    print("Generating anthropological analysis...")
    ai_analysis = extract_anthropological_analysis(payload["content"]["extracted_text"])
    payload["ai_analysis"] = ai_analysis
    payload["content"]["tables"] = []

    if ai_analysis:
        print(f"AI analysis complete - {len(ai_analysis)} modules populated:")
        for key in ai_analysis:
            print(f"   - {key}: {len(ai_analysis[key])} chars")
    else:
        print("AI analysis returned empty")

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