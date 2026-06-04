# Document Restoration Skill - Output & Visualization Design

## 1. Goal
To transform raw Computer Vision (CV) processing into "Insightful Outputs"—moving from mere pixel manipulation to structured, actionable knowledge.

## 2. Defining "Insightful"
An output is considered insightful if it provides one of the following:
* **Confidence Insights:** Quantifying the reliability of the recovered data (e.g., "This text is 85% likely to be correct").
* **Structural Insights:** Identifying the organization of the document (e.g., "This is a three-column newspaper layout").
* **Integrity Insights:** Documenting the changes made during restoration (e.g., "Denoising applied to remove grain").

## 3. Visual Requirements Mapping

| Required Visual | Implementation Strategy (The "How") | The "Insight" (The "Why") |
| :--- | :--- | :--- |
| **Before/After Enhancement** | A masking slider (left-to-right) over a single image container, showing the "after" with the "before" revealed underneath. | **Validation:** Visual proof that the CV techniques improved legibility without creating artifacts. |
| **Document Layout Visualization** | Bounding-box overlays identifying zones (Headers, Footers, Paragraphs, Tables, Signatures) using geometric/line detection (no heavy deep-learning models). | **Structure:** Allows immediate understanding of the document's skeleton. |
| **Annotated Image** | Enhanced image with callouts or highlights (e.g., highlighting a detected stain). | **Context:** Connects CV actions directly to specific areas of the image. |
| **Extracted Text Table** | A clean, digitalized Markdown/CSV table reconstructed from image-based tables. | **Data Utility:** Converts unstructured ink into structured, machine-readable data. |
| **Evidence Table** | A table showing: `[Original Crop]` $\rightarrow$ `[Enhanced Crop]` $\rightarrow$ `[Detected Text]` $\rightarrow$ `[Confidence %]`. | **Auditability:** The "Truth Table" for verifying OCR accuracy and spotting hallucinations. |
| **Chart Image** | Bar/Pie charts showing **Confidence Distribution** or **Character Frequency**. | **Quality Control:** A high-level metric of how much to trust the entire scan. |
| **Dashboard Image** | A composite "Summary Card" containing Title, Date, Preview, and a "Document Health Score." | **Efficiency:** Enables rapid skimming of large document batches. |
| **Visual Audit Report** | A comprehensive PDF/Canvas report summarizing the entire pipeline from raw file to final data. | **Provenance:** Provides a formal paper trail for historical or legal verification. |

## 4. The "Golden Payload" (Technical Strategy)
To enable these visuals, the CV engine should not just return an image, but a **Metadata Package (JSON)** containing:
1. **Image Paths:** (Paths to Before, After, Annotated, and Layout versions).
2. **Text Data:** (Strings + bounding box coordinates).
3. **Confidence Scores:** (Per-word or per-line reliability metrics).
4. **Processing Metadata:** (Filters used, processing time, image dimensions).
5. **Processing Status:** (e.g., `"success"`, `"failed"`, `"processing"`).

*This separation allows the "Math/CV" part to be decoupled from the "Visualization/Insight" part.*
