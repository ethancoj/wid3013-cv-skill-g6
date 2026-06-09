import json
import os
import sys
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_PATH = os.path.join(BASE_DIR, "tests", "test_output.html")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from cv_processing_2 import process_document

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
    print(f"Open this file in your browser: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/render_test.py <path_to_image>")
        print("Example: python tests/render_test.py samples/test_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    output_dir = os.path.join(BASE_DIR, "samples", "output")

    print(f"Processing image: {image_path}")
    payload = process_document(image_path, output_dir)

    json_path = os.path.join(output_dir, "payload.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Payload saved to: {json_path}")

    render_dashboard(payload, TEMPLATE_DIR, OUTPUT_PATH)