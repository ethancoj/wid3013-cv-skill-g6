# WID3013 CV Skill - Group 6
Document Restoration Skill for Social Science & Anthropology Students

## Setup Instructions

### Prerequisites
- Python 3.13+
- Node.js v22+
- Git

### 1. Clone the repo
git clone https://github.com/ethancoj/wid3013-cv-skill-g6.git
cd wid3013-cv-skill-g6

### 2. Python environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 3. Environment variables
Create a `.env` file in the root folder:
GOOGLE_API_KEY=your_key_here
im using gemini google/gemma-4-26b-a4b-it like amir

API key from https://aistudio.google.com

### 4. OpenClaw setup
npm install -g openclaw
openclaw onboard

When prompted, select Google as provider and paste your API key.

### 5. Run OpenClaw
openclaw tui

### Team
- Member 1: Domain & User Research
- Member 2: CV / Image Processing (src/)
- Member 3: Skill Workflow Design
- Member 4: Tool & Integration (templates/, skill/)
- Member 5: Evaluation & Presentation
