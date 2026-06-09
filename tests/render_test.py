"""
render_test.py
Usage:
  python tests/render_test.py <image_path> [--script 1|2|3]

Examples:
  python tests/render_test.py samples/test_image.jpg
  python tests/render_test.py samples/test_image.jpg --script 1
  python tests/render_test.py samples/test_image.jpg --script 3
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

def get_script(script_num):
    if script_num == "1":
        import cv_processing as cv
        print("Using Script 1 (cv_processing.py) — Advanced pipeline with deskew + Sauvola")
    elif script_num == "3":
        import cv_processing_3 as cv
        print("Using Script 3 (cv_processing_3.py) — Full pipeline with resolution validation")
    else:
        import cv_processing_2 as cv
        print("Using Script 2 (cv_processing_2.py) — Standard pipeline (default)")
    return cv

def render_dashboard(payload, template_dir, output_path):
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("dashboard_template.html")

    tables = payload["content"].get("tables", [])
    table_headers = tables[0]["headers"] if tables else ["Field", "Value"]
    table_rows = tables[0]["rows"] if tables else []

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
        table_headers=table_headers,
        table_rows=table_rows,
        overall_pct=payload["insights"]["confidence"]["overall_pct"],
        high_conf_pct=payload["insights"]["confidence"]["high_confidence_pct"],
        chart_data=payload["insights"]["confidence"]["distribution_chart_data"],
        audit_steps=payload["insights"]["audit_trail"]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Dashboard rendered successfully!")
    print(f"Open this in your browser: {output_path}")

if __name__ == "__main__":
    # Parse arguments
    args = sys.argv[1:]
    if not args or args[0].startswith("--"):
        print("Usage: python tests/render_test.py <image_path> [--script 1|2|3]")
        sys.exit(1)

    image_path = args[0]
    script_num = "2"  # default
    if "--script" in args:
        idx = args.index("--script")
        if idx + 1 < len(args):
            script_num = args[idx + 1]

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Make sure your test image is in the samples/ folder")
        sys.exit(1)

    # Load the correct processing script
    cv = get_script(script_num)

    # Process the image
    output_dir = os.path.join(BASE_DIR, "samples", "output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing: {image_path}")
    payload = cv.process_document(image_path, output_dir)

    # Save payload JSON
    json_path = os.path.join(output_dir, "payload.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Payload saved to: {json_path}")

    # Render dashboard
    render_dashboard(payload, TEMPLATE_DIR, OUTPUT_PATH)