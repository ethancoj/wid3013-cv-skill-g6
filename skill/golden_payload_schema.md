# Golden Payload JSON Schema & HTML Mapping

This document defines the "Golden Payload" JSON structure required to power the **Document Restoration Dashboard** and provides the mapping to the `dashboard_template.html`.

## 1. JSON Schema

The Python CV engine should output a single, unified JSON object following this structure:

```json
{
  "metadata": {
    "doc_id": "string",
    "timestamp": "ISO-8601 string",
    "filename": "string",
    "health_score": "integer (0-100)",
    "width": "integer",
    "height": "integer",
    "processing_time_seconds": "float",
    "status": "string (e.g., 'success', 'failed', 'processing')"
  },
  "images": {
    "before_url": "string (path/URL)",
    "after_url": "string (path/URL)",
    "annotated_url": "string (path/URL)",
    "layout_url": "string (path/URL)"
  },
  "content": {
    "extracted_text": "string (full text block)",
    "tables": [
      {
        "table_id": "string",
        "headers": ["string", "string"],
        "rows": [
          ["string", "string"],
          ["string", "string"]
        ]
      }
    ]
  },
  "insights": {
    "confidence": {
      "overall_pct": "integer",
      "high_confidence_pct": "integer",
      "distribution_chart_data": [
        {"label": "High", "value": 85},
        {"label": "Medium", "value": 10},
        {"label": "Low", "value": 5}
      ]
    },
    "audit_trail": [
      {
        "name": "string (e.g., 'Denoising')",
        "description": "string (e.g., 'Applied Gaussian blur to remove salt-and-pepper noise')"
      }
    ]
  }
}
```

---

## 2. HTML Template Mapping

The following table maps the JSON keys to the placeholders used in `dashboard_template.html`.

| JSON Path | HTML Template Placeholder | Purpose |
| :--- | :--- | :--- |
| `metadata.doc_id` | `{{ doc_id }}` | Unique ID in header |
| `metadata.timestamp` | `{{ timestamp }}` | Time in header |
| `metadata.health_score` | `{{ health_score }}` | Large green score in header |
| `images.before_url` | `{{ img_before }}` | Bottom layer of the slider |
| `images.after_url` | `{{ img_after }}` | Top layer of the slider |
| `content.extracted_text` | `{{ extracted_text }}` | Main text block |
| `content.tables[0].headers` | `{% for header in table_headers %}` | Table column titles |
| `content.tables[0].rows` | `{% for row in table_rows %}` | Table body content |
| `insights.confidence.high_confidence_pct`| `{{ high_conf_pct }}` | Confidence progress bar |
| `insights.confidence.distribution_chart_data`| `[Chart Data Placeholder]` | To be injected into a Chart.js script |
| `insights.audit_trail[i].name` | `{{ step.name }}` | Audit timeline item |
| `insights.audit_trail[i].description`| `{{ step.description }}` | Audit timeline detail |
| `metadata.width` / `height` | `{{ width }} / {{ height }}` | Technical metadata card |
| `metadata.processing_time_seconds` | `{{ proc_time }}` | Technical metadata card |

---

## 3. Implementation Example (Python)

Using `Jinja2` to merge the JSON payload with the HTML template:

```python
from jinja2 import Template
import json

# 1. Load your Golden Payload
with open('payload.json', 'r') as f:
    data = json.load(f)

# 2. Load the template
with open('dashboard_template.html', 'r') as f:
    template_content = f.read()

# 3. Render (Flattening the JSON for the template)
template = Template(template_content)
html_output = template.render(
    doc_id=data['metadata']['doc_id'],
    timestamp=data['metadata']['timestamp'],
    health_score=data['metadata']['health_score'],
    img_before=data['images']['before_url'],
    img_after=data['images']['after_url'],
    extracted_text=data['content']['extracted_text'],
    table_headers=data['content']['tables'][0]['headers'],
    table_rows=data['content']['tables'][0]['rows'],
    high_conf_pct=data['insights']['confidence']['high_confidence_pct'],
    audit_steps=data['insights']['audit_trail'],
    width=data['metadata']['width'],
    height=data['metadata']['height'],
    proc_time=data['metadata']['processing_time_seconds']
)

# 4. Save the final HTML
with open('report.html', 'w') as f:
    f.write(html_output)
```
