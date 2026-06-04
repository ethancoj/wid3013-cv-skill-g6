import json
import os
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYLOAD_PATH = os.path.join(BASE_DIR, "tests", "test_payload.json")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
OUTPUT_PATH = os.path.join(BASE_DIR, "tests", "test_output.html")

def render_dashboard(payload_path, template_dir, output_path):
    with open(payload_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("dashboard_template.html")

    html_output = template.render(
        doc_id=data["metadata"]["doc_id"],
        timestamp=data["metadata"]["timestamp"],
        filename=data["metadata"]["filename"],
        health_score=data["metadata"]["health_score"],
        width=data["metadata"]["width"],
        height=data["metadata"]["height"],
        proc_time=data["metadata"]["processing_time_seconds"],
        status=data["metadata"]["status"],
        img_before=data["images"]["before_url"],
        img_after=data["images"]["after_url"],
        extracted_text=data["content"]["extracted_text"],
        table_headers=data["content"]["tables"][0]["headers"],
        table_rows=data["content"]["tables"][0]["rows"],
        overall_pct=data["insights"]["confidence"]["overall_pct"],
        high_conf_pct=data["insights"]["confidence"]["high_confidence_pct"],
        chart_data=data["insights"]["confidence"]["distribution_chart_data"],
        audit_steps=data["insights"]["audit_trail"]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)

    print(f"Dashboard rendered successfully!")
    print(f"Open this file in your browser: {output_path}")

if __name__ == "__main__":
    render_dashboard(PAYLOAD_PATH, TEMPLATE_DIR, OUTPUT_PATH)
