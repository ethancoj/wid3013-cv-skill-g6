"""
run_skill.py — OpenClaw entry point for the Document Restoration Skill.
"""

import json
import re
import base64
import webbrowser
from dotenv import load_dotenv
load_dotenv()
import os
import sys
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemma-4-26b-a4b-it")

SECTION_MARKERS = {
    "artifact_profile": "ARTIFACT PROFILE",
    "provenance": "PROVENANCE",
    "anthropological_matrix": "ANTHROPOLOGICAL MATRIX",
    "taxonomy": "TAXONOMY",
    "validity_bias": "VALIDITY AND BIAS",
    "executive_summary": "EXECUTIVE SUMMARY",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_HTML = os.path.join(BASE_DIR, "tests", "test_output.html")
OUTPUT_DIR = os.path.join(BASE_DIR, "samples", "output")
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)


def log(msg):
    print(msg, file=sys.stderr)


def checkbox_to_html(text):
    lines = text.split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- [x]") or stripped.startswith("- [X]"):
            label = stripped[5:].strip()
            html_lines.append(f'<label class="cb"><input type="checkbox" checked disabled> {label}</label>')
        elif stripped.startswith("- [ ]"):
            label = stripped[5:].strip()
            html_lines.append(f'<label class="cb"><input type="checkbox" disabled> {label}</label>')
        elif stripped.startswith("- "):
            html_lines.append(f'<div class="field-item">{stripped[2:]}</div>')
        elif ":" in stripped and not stripped.startswith("<"):
            parts = stripped.split(":", 1)
            key = parts[0].strip()
            val = parts[1].strip() if len(parts) > 1 else ""
            if val:
                html_lines.append(f'<div class="field-row"><span class="field-key">{key}:</span> <span class="field-val">{val}</span></div>')
            else:
                html_lines.append(f'<div class="field-heading">{key}</div>')
        elif stripped:
            html_lines.append(f'<div class="field-text">{stripped}</div>')
    return "\n".join(html_lines)


def extract_anthropological_analysis(extracted_text):
    prompt = f"""You are a document analysis assistant for anthropology students.
Analyze the OCR text below and fill in this template. Use [x] for checked and [ ] for unchecked boxes. Fill in bracketed fields with real values from the text.

## ARTIFACT PROFILE
Physical Format:
- [ ] Bound Book
- [ ] Loose Letter / Folio
- [ ] Ledger / Log
- [ ] Map / Diagram
- [ ] Other: [Specify]
Materiality: [substrate and medium]
Condition: [describe damage and what it suggests]
Legibility Score (1-5): [score]

## PROVENANCE
Temporal Origin: [year/decade/era]
Geographic Origin: [location]
Author: [name or Anonymous]
Role: [profession]
Document Genre:
- [ ] Administrative
- [ ] Personal Diary
- [ ] Legal
- [ ] Financial / Trade
- [ ] Religious
- [ ] Correspondence
- [ ] Other: [specify]

## ANTHROPOLOGICAL MATRIX
Positionality:
- [ ] Complete Outsider
- [ ] Assimilated Outsider
- [ ] Elite Insider
- [ ] Marginalized Insider
Intended Audience:
- [ ] Self (Private)
- [ ] Superiors / The State
- [ ] Public / Published
- [ ] Peers / Colleagues
Dominant Power Dynamic: [describe]
Notable Silences: [whose voices are absent]

## TAXONOMY
Locations: [list]
Key Figures: [list]
Thematic Tags:
- [ ] #Kinship
- [ ] #Ritual_Religion
- [ ] #Economics_Trade
- [ ] #Warfare_Conflict
- [ ] #Law_Justice
- [ ] #Agriculture_Environment
- [ ] #Medicine_Health
Indigenous Lexicon: [any non-dominant language terms used]

## VALIDITY AND BIAS
Proximity:
- [ ] First-hand witness
- [ ] Second-hand account
- [ ] Hearsay
- [ ] Post-hoc summary
Bias Level (1-5): [score]
Primary Value:
- [ ] Understanding Author's Culture
- [ ] Understanding Subject's Culture

## EXECUTIVE SUMMARY
Elevator Pitch: [2 sentences max about what this document is]
Key Takeaway: [1 sentence about why it matters]
Next Steps: [research suggestions]

Be factual. Only analyze what the text contains.

OCR Text:
{extracted_text[:3000]}"""

    try:
        response = model.generate_content(prompt, generation_config=genai.GenerationConfig(
            temperature=0,
            max_output_tokens=4000,
        ))
        raw = response.text.strip()
        sections = parse_sections(raw)
        for key in sections:
            sections[key] = checkbox_to_html(sections[key])
        return sections
    except Exception as e:
        log(f"LLM analysis failed: {e}")
        return {}


def parse_sections(text):
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
            if current_key and current_lines:
                result[current_key] = "\n".join(current_lines).strip()
            current_key = matched_key
            current_lines = []
        else:
            if current_key:
                current_lines.append(line)
    if current_key and current_lines:
        result[current_key] = "\n".join(current_lines).strip()
    return result


def img_to_data_uri(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".")
    return f"data:image/{ext};base64,{b64}"


def render_dashboard(payload):
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template = env.get_template("dashboard_template.html")
    html_output = template.render(
        doc_id=payload["metadata"]["doc_id"],
        timestamp=payload["metadata"]["timestamp"],
        filename=payload["metadata"]["filename"],
        health_score=payload["metadata"]["health_score"],
        extracted_text=payload["content"]["extracted_text"],
        overall_pct=payload["insights"]["confidence"]["overall_pct"],
        high_conf_pct=payload["insights"]["confidence"]["high_confidence_pct"],
        chart_data=payload["insights"]["confidence"]["distribution_chart_data"],
        audit_steps=payload["insights"]["audit_trail"],
        ai_analysis=payload.get("ai_analysis", {}),
        img_before=img_to_data_uri(payload["images"]["before_url"]),
        img_after=img_to_data_uri(payload["images"]["after_url"]),
        annotated_url=img_to_data_uri(payload["images"]["annotated_url"]),
    )
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_output)


def format_telegram_response(payload, category):
    text_preview = payload["content"]["extracted_text"][:200].strip()
    if len(payload["content"]["extracted_text"]) > 200:
        text_preview += "..."
    
    audit_steps = payload["insights"]["audit_trail"]
    steps_text = "\n".join([f"  * {s['name']}" for s in audit_steps])
    
    vault_dir = os.path.join("workspace", "document_vault", category.replace(" ", "_"))
    doc_id = payload["metadata"]["doc_id"]
    
    # Pull AI analysis sections (strip HTML tags for plain text)
    ai = payload.get("ai_analysis", {})
    
    import re
    def strip_html(text):
        return re.sub(r'<[^>]+>', '', text).strip() if text else "Not available"
    
    response = f"""[SUCCESS] Document processed successfully

[INFO] File: {payload["metadata"]["filename"]}
[INFO] Health Score: {payload["metadata"]["health_score"]}/100
[INFO] Confidence: {payload["insights"]["confidence"]["overall_pct"]}%

[INFO] Extracted Text Preview:
{text_preview}

[INFO] ANTHROPOLOGICAL ANALYSIS

[INFO] Artifact Profile:
{strip_html(ai.get("artifact_profile", ""))}

[INFO] Provenance:
{strip_html(ai.get("provenance", ""))}

[INFO] Anthropological Matrix:
{strip_html(ai.get("anthropological_matrix", ""))}

[INFO] Taxonomy:
{strip_html(ai.get("taxonomy", ""))}

[INFO] Validity & Bias:
{strip_html(ai.get("validity_bias", ""))}

[INFO] Executive Summary:
{strip_html(ai.get("executive_summary", ""))}

[INFO] Processing steps:
{steps_text}

[INFO] Saved under: {category}
[INFO] {vault_dir}/{doc_id}_payload.json""".strip()
    return response


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_skill.py <image_path> [category]")
        sys.exit(1)

    image_path = sys.argv[1]
    category = sys.argv[2] if len(sys.argv) > 2 else "Uncategorized"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    try:
        from cv_processing import process_any as process_document
    except ImportError:
        print("Error: cv_processing.py not found in src/")
        sys.exit(1)

    log(f"Processing image: {image_path}")
    log(f"Category: {category}")
    log("Running CV pipeline...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    payload = process_document(image_path, OUTPUT_DIR)

    # Wrap the CV payload into the expected OpenClaw schema
    wrapped_payload = {
        "metadata": {
            "doc_id": payload["filename"],
            "timestamp": datetime.now().isoformat(),
            "filename": payload["filename"],
            "health_score": payload["image_metrics"]["health_score"]
        },
        "content": {
            "extracted_text": payload["ocr"]["raw_text"],
            "tables": []
        },
        "insights": {
            "confidence": {
                "overall_pct": payload["ocr"]["overall_confidence_pct"],
                "high_confidence_pct": payload["ocr"]["high_confidence_pct"],
                "distribution_chart_data": payload["ocr"]["distribution"]
            },
            "audit_trail": payload["audit_trail"]
        },
        "images": {
            "before_url": payload["images"]["before_path"],
            "after_url": payload["images"]["after_path"],
            "annotated_url": payload["images"]["annotated_path"]
        },
        "ai_analysis": {}
    }

    log("Generating anthropological analysis...")
    ai_analysis = extract_anthropological_analysis(wrapped_payload["content"]["extracted_text"])
    wrapped_payload["ai_analysis"] = ai_analysis

    if ai_analysis:
        log(f"AI analysis complete - {len(ai_analysis)} modules populated")
    else:
        log("AI analysis returned empty")

    json_path = os.path.join(OUTPUT_DIR, "payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(wrapped_payload, f, indent=2)

    render_dashboard(wrapped_payload)

    # Auto-open dashboard in browser
    webbrowser.open(OUTPUT_HTML)

    vault_dir = os.path.join(BASE_DIR, "workspace", "document_vault", category.replace(" ", "_"))
    os.makedirs(vault_dir, exist_ok=True)
    vault_json = os.path.join(vault_dir, f"{wrapped_payload['metadata']['doc_id']}_payload.json")
    with open(vault_json, "w", encoding="utf-8") as f:
        json.dump(wrapped_payload, f, indent=2)

    response = format_telegram_response(wrapped_payload, category)
    print(response)


if __name__ == "__main__":
    main()