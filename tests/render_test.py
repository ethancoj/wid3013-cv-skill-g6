"""
render_test.py
Usage:
  python tests/render_test.py <image_path>

Examples:
  python tests/render_test.py samples/test_image.jpg
  python tests/render_test.py samples/old_faded.jpg
"""

import json
import os
import sys
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_PATH = os.path.join(BASE_DIR, "tests", "test_output.html")
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.insert(0, SRC_DIR)

import base64

def img_to_data_uri(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(path)[1].lstrip(".")
    return f"data:image/{ext};base64,{b64}"


def render_dashboard(payload, template_dir, output_path):
    env = Environment(loader=FileSystemLoader(template_dir))
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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Dashboard rendered successfully!")
    print(f"Open this in your browser: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/render_test.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Make sure your test image is in the samples/ folder")
        sys.exit(1)

    try:
        from cv_processing import process_document
    except ImportError:
        print("Error: cv_processing.py not found in src/")
        sys.exit(1)

    output_dir = os.path.join(BASE_DIR, "samples", "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing: {image_path}")
    payload = process_document(image_path, output_dir)

    # Save payload JSON
    json_path = os.path.join(output_dir, "payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Payload saved to: {json_path}")

    # Render dashboard
    render_dashboard(payload, TEMPLATE_DIR, OUTPUT_PATH)