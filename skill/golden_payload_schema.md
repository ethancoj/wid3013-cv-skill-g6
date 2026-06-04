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

The following table maps every JSON key to the placeholder used in `dashboard_template.html`.

| JSON Path | HTML Template Placeholder | Purpose |
| :--- | :--- | :--- |
| `metadata.doc_id` | `{{ doc_id }}` | Unique ID in header and metadata card |
| `metadata.timestamp` | `{{ timestamp }}` | Processed-at time in header |
| `metadata.filename` | `{{ filename }}` | Filename in header `<h1>` and metadata card |
| `metadata.health_score` | `{{ health_score }}` | Large circular badge in header |
| `metadata.width` | `{{ width }}` | Dimensions field in metadata card |
| `metadata.height` | `{{ height }}` | Dimensions field in metadata card |
| `metadata.processing_time_seconds` | `{{ proc_time }}` | Processing time field in metadata card |
| `metadata.status` | `{{ status }}` | Status badge colour class and label in header |
| `images.before_url` | `{{ img_before }}` | Bottom (before) layer of the Before/After slider |
| `images.after_url` | `{{ img_after }}` | Top (after) layer of the Before/After slider |
| `images.layout_url` | `{{ layout_url }}` | Layout Map image card (Row 2 left) |
| `images.annotated_url` | `{{ annotated_url }}` | Annotated Image card (Row 2 right) |
| `content.extracted_text` | `{{ extracted_text }}` | Extracted text block (Row 3 right) |
| `content.tables[0].headers` | `{% for header in table_headers %}` | Structured data table column titles |
| `content.tables[0].rows` | `{% for row in table_rows %}` | Structured data table body rows |
| `insights.confidence.overall_pct` | `{{ overall_pct }}` | Overall confidence progress bar |
| `insights.confidence.high_confidence_pct` | `{{ high_conf_pct }}` | High-confidence text progress bar |
| `insights.confidence.distribution_chart_data` | `{{ chart_data \| tojson }}` | Injected into the Chart.js doughnut via a JSON `<script>` tag |
| `insights.audit_trail[i].name` | `{{ step.name }}` | Audit trail item title |
| `insights.audit_trail[i].description` | `{{ step.description }}` | Audit trail item detail |

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

# 3. Render — flatten every JSON key to a named template variable
template = Template(template_content)
html_output = template.render(
    # metadata.*
    doc_id      = data['metadata']['doc_id'],
    timestamp   = data['metadata']['timestamp'],
    filename    = data['metadata']['filename'],
    health_score= data['metadata']['health_score'],
    width       = data['metadata']['width'],
    height      = data['metadata']['height'],
    proc_time   = data['metadata']['processing_time_seconds'],
    status      = data['metadata']['status'],
    # images.*
    img_before    = data['images']['before_url'],
    img_after     = data['images']['after_url'],
    layout_url    = data['images']['layout_url'],
    annotated_url = data['images']['annotated_url'],
    # content.*
    extracted_text = data['content']['extracted_text'],
    table_headers  = data['content']['tables'][0]['headers'],
    table_rows     = data['content']['tables'][0]['rows'],
    # insights.confidence.*
    overall_pct  = data['insights']['confidence']['overall_pct'],
    high_conf_pct= data['insights']['confidence']['high_confidence_pct'],
    chart_data   = data['insights']['confidence']['distribution_chart_data'],
    # insights.audit_trail
    audit_steps  = data['insights']['audit_trail'],
)

# 4. Save the final HTML
with open('report.html', 'w') as f:
    f.write(html_output)
```