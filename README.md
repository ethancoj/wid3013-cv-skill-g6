# WID3013 CV Skill — Group 6
## Document Restoration for Social Science & Anthropology Students

An OpenClaw-powered Telegram bot that restores degraded historical documents and generates an interactive HTML dashboard with OCR extraction, confidence scoring, and AI-driven anthropological analysis.

Built for anthropology students working with fieldwork scans, archival materials, old letters, and historical records.

---

## What It Does

1. **Send** a document image to the Telegram bot
2. **CV pipeline** denoises, enhances, and extracts text via OCR (Tesseract)
3. **AI analysis** (Google Gemma) fills in 6 anthropological modules automatically
4. **Dashboard** renders as a self-contained HTML file with:
   - Before/after restoration slider
   - Annotated image with click-to-zoom
   - Editable OCR text (correct mistakes inline)
   - Confidence chart (Chart.js)
   - 6 anthropological analysis modules with checkboxes
   - Archival lineage fields (localStorage persisted)
   - Researcher's memo scratchpad

---

## Dashboard Modules

| Module | What It Covers |
|--------|---------------|
| Artifact Profile | Physical format, materiality, condition, legibility score |
| Provenance & Context | Time period, location, author, document genre |
| Anthropological Matrix | Author positionality, audience, power dynamics, silences |
| Taxonomy & Entities | Locations, people, thematic tags, indigenous terms |
| Validity & Bias | Source proximity, bias level, primary research value |
| Executive Summary | Elevator pitch, key takeaway, research next steps |

---

## Tech Stack

- **Bot Framework:** OpenClaw + Telegram
- **CV Processing:** OpenCV, Tesseract OCR
- **AI Provider:** Google Gemma (gemma-4-26b-a4b-it) via AI Studio
- **Dashboard:** Jinja2 HTML template, Chart.js
- **Language:** Python 3.13+

---

## Prerequisites

Before starting, make sure you have:

- [Python 3.13+](https://www.python.org/downloads/)
- [Node.js v22+](https://nodejs.org/)
- [Git](https://git-scm.com/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) — install to default path `C:\Program Files\Tesseract-OCR\`
- A Google AI Studio API key (free) from [aistudio.google.com](https://aistudio.google.com)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

---

## Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/ethancoj/wid3013-cv-skill-g6.git
cd wid3013-cv-skill-g6
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

Get your free API key from [aistudio.google.com](https://aistudio.google.com) — no credit card required.

### 5. Install and configure OpenClaw

```bash
npm install -g openclaw
openclaw plugins install @ytlailabs/ilmu-openclaw-plugin
openclaw onboard
```

During onboard:
- Select **Google** as the AI provider
- Paste your API key when prompted
- Configure **Telegram** as the channel and paste your bot token

### 6. Register the skill with OpenClaw

```bash
# Windows
copy skill\skill.md %USERPROFILE%\.openclaw\workspace\skills\skill.md

# macOS/Linux
cp skill/skill.md ~/.openclaw/workspace/skills/skill.md
```

### 7. Start the bot

```bash
openclaw tui
```

---

## Usage

### Via Telegram (Bot)

1. Open your Telegram bot in DM
2. Send a document image with the caption: **"restore this document"**
3. Wait for processing (CV pipeline + AI analysis, ~30-60 seconds)
4. The bot replies with a summary and a link to the dashboard
5. Open the dashboard link in your browser

### Via Command Line (Local Testing)

```bash
# Place a test image in samples/ folder, then:
python run_skill.py samples/your_image.jpg "Field Notes"

# Open the generated dashboard:
# tests/test_output.html (open in browser)
```

---

## Project Structure

```
wid3013-cv-skill-g6/
├── run_skill.py                  ← Main entry point
├── skill/
│   ├── skill.md                  ← OpenClaw skill config
│   ├── golden_payload_schema.md  ← JSON schema documentation
│   ├── document_vault_architecture.md
│   └── ...                       ← Design documentation
├── src/
│   └── cv_processing_4.py        ← CV processing script
├── templates/
│   └── dashboard_template.html   ← Jinja2 dashboard template
├── tests/
│   ├── test_output.html          ← Generated dashboard
│   └── test_payload.json         ← Dummy test data
├── samples/
│   └── output/                   ← Processed images + payload
├── workspace/
│   └── document_vault/           ← Categorized document storage
├── .env                          ← API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How It Works

```
User sends image in Telegram
        ↓
OpenClaw reads skill.md → detects trigger phrase
        ↓
Calls: python run_skill.py <image_path> <category>
        ↓
cv_processing_4.py runs:
  → Grayscale → Denoise → CLAHE → Binarize → OCR → Layout analysis
  → Outputs Golden Payload JSON
        ↓
Extracted text sent to Google Gemma LLM
  → Fills in 6 anthropological analysis modules
  → Returns checked/unchecked checkbox template
        ↓
Jinja2 merges payload + AI analysis into dashboard_template.html
  → Images embedded as base64 (self-contained HTML)
  → Checkboxes converted from markdown to real HTML elements
        ↓
Dashboard saved to tests/test_output.html
        ↓
Telegram response sent with summary + dashboard link
```

---

## Sample Documents for Testing

The skill works best with:
- Historical newspaper scans
- Old letters and correspondence
- Publisher catalogs and price lists
- Fieldwork notes and journal pages
- Ledgers and financial records
- Any degraded or faded document image (JPG, PNG)

---

## Limitations

- **OCR accuracy** drops on handwritten text, faded ink, or heavily damaged documents
- **Non-Latin scripts** (Jawi, Arabic, Chinese) may have lower accuracy depending on Tesseract language packs
- **Gemma LLM** occasionally truncates analysis if the document is very long — the dashboard gracefully hides incomplete modules
- **Gemma cannot output structured JSON** — we use a template-filling approach with section parsing instead
- **Bot works in Telegram DMs only** — group chat support requires additional OpenClaw configuration
- **Dashboard runs locally** — the localhost link works only on the machine running the bot

---

## Ethical Boundaries

- This skill only restores and organizes visible information
- It does **not** authenticate, date, or verify document legitimacy
- It does **not** identify individuals from photos or portraits
- All extracted text is presented as-is with confidence scores — users verify accuracy
- The skill must not be used to forge, alter, or misrepresent historical documents

---

## Team

| Member | Role |
|--------|------|
| Member 1 | Project Lead |
| Member 2 | CV Processing Lead |
| Member 3 | Design & Documentation |
| Member 4 (Ethan) | Tool & Integration Developer |
| Member 5 | Presentation Lead |

---

## License

This project was developed as part of the WID3013 course at Universiti Malaya. For academic use only.
