"""
Document Restoration + OCR Pipeline
====================================

For: WID3013 — Topic: Arts & Social Science (Anthropology & Sociology)
Use case: Restore old/degraded archival documents and extract metadata.

Pipeline stages (each one is a step in the audit_trail):
  1. Grayscale conversion
  2. Background normalization via morphological closing + division
     (removes uneven staining / vignette / yellowing)
  3. CLAHE contrast enhancement
  4. Non-local means denoising (removes speckle, preserves edges)
  5. Sauvola adaptive binarization (handles local lighting better than Otsu)
  6. Deskew via Hough on text contours
  7. Layout analysis (text-line detection)
  8. OCR via Tesseract with word-level confidences
  9. Metadata extraction (dates, persons, places, document type)
 10. Health score calculation

The script is structured so that in OpenClaw, after this runs, the LLM can
read the restored image directly with its native vision for high-accuracy
OCR. The `claude_vision_payload` field in the output gives a base64 image
ready to pass to the model. (See `# === HOOK ===` markers below.)
"""

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import json
import re
import os
import time
import hashlib
import base64
from datetime import datetime, timezone

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# --------------------------------------------------------------------------
# Restoration primitives
# --------------------------------------------------------------------------

def to_grayscale(img):
    """Step 1 — collapse to one channel; eliminates colour noise."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_background(gray, kernel=35):
    """
    Step 2 — the key restoration step. Estimate the page background by a
    large morphological closing (anything text-sized gets erased, leaving
    only the slow-varying paper/stain illumination). Divide the original
    by this estimate to flatten lighting and remove stains.
    """
    # Big elliptical kernel — must be larger than the thickest text stroke
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)
    # Heavy blur to make the background smooth
    bg = cv2.GaussianBlur(bg, (51, 51), 0)
    # Divide (safer with float; +1 prevents divide-by-zero)
    norm = (gray.astype(np.float32) / (bg.astype(np.float32) + 1.0)) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def enhance_contrast(gray, clip=2.5, tile=(8, 8)):
    """Step 3 — CLAHE: local histogram equalization, gentle clip limit."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(gray)


def denoise(gray):
    """Step 4 — non-local means; better than bilateral for grainy paper."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                    searchWindowSize=21)


def sauvola_threshold(gray, window=25, k=0.2, R=128):
    """
    Step 5 — Sauvola adaptive binarization. Uses local mean and stddev;
    much more robust than Otsu on documents with non-uniform background.
    Implemented manually since OpenCV doesn't ship Sauvola.
    Formula: T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1))
    """
    g = gray.astype(np.float32)
    # Local mean (box filter is fast)
    mean = cv2.boxFilter(g, ddepth=cv2.CV_32F, ksize=(window, window))
    sqmean = cv2.boxFilter(g * g, ddepth=cv2.CV_32F, ksize=(window, window))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0))
    T = mean * (1.0 + k * (std / R - 1.0))
    binary = (g > T).astype(np.uint8) * 255
    return binary


def remove_small_components(binary, min_area=15):
    """
    Step 5.5 — remove isolated dark blobs smaller than `min_area` pixels.
    These are residual speckle specks that survived denoising and would
    otherwise be interpreted as punctuation by OCR.
    """
    inv = 255 - binary
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = np.zeros_like(inv)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return 255 - out


def crop_to_content(binary, margin=30):
    """
    Step 5.6 — find the smallest box containing the dense text region and
    crop to it. Saves OCR from chewing through empty stained margins and
    also dramatically improves layout map quality.
    Returns (cropped_binary, (x_off, y_off, w, h)) so colour images can be
    cropped to match.
    """
    inv = 255 - binary
    # Smooth into a density map; threshold to find the "text mass"
    density = cv2.boxFilter(inv, ddepth=cv2.CV_32F, ksize=(80, 80))
    density = (density > 30).astype(np.uint8) * 255
    ys, xs = np.where(density > 0)
    if len(xs) < 100:
        return binary, (0, 0, binary.shape[1], binary.shape[0])
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, binary.shape[1])
    y1 = min(int(ys.max()) + margin, binary.shape[0])
    return binary[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)


def deskew(binary):
    """
    Step 6 — estimate skew angle from minAreaRect of all dark pixels,
    then rotate. Works because text is the dominant horizontal element.
    Returns rotated binary AND the angle (so we can rotate the colour
    image the same amount for the "after" preview).
    """
    # Get coords of dark pixels (text)
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 100:
        return binary, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect returns angle in (-90, 0]; normalize to small offsets
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Limit correction to plausible scan skews
    if abs(angle) > 15:
        return binary, 0.0
    h, w = binary.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# --------------------------------------------------------------------------
# Layout analysis & annotation
# --------------------------------------------------------------------------

def detect_text_lines_from_ocr(ocr_data, min_conf=30):
    """
    Derive line bounding boxes by grouping OCR words that share the same
    (block_num, par_num, line_num). This is dramatically more reliable than
    contour-based detection because only words Tesseract actually recognized
    contribute to a line — pure speckle is automatically excluded.
    Returns list of (x, y, w, h, n_words, mean_conf).
    """
    groups = {}
    n = len(ocr_data['text'])
    for i in range(n):
        txt = ocr_data['text'][i].strip()
        try:
            conf = float(ocr_data['conf'][i])
        except (ValueError, TypeError):
            conf = -1
        if not txt or conf < min_conf:
            continue
        key = (ocr_data['block_num'][i],
               ocr_data['par_num'][i],
               ocr_data['line_num'][i])
        x, y = ocr_data['left'][i], ocr_data['top'][i]
        w, h = ocr_data['width'][i], ocr_data['height'][i]
        if key not in groups:
            groups[key] = {"xs": [], "ys": [], "x2s": [], "y2s": [],
                           "confs": [], "n": 0}
        g = groups[key]
        g["xs"].append(x); g["ys"].append(y)
        g["x2s"].append(x + w); g["y2s"].append(y + h)
        g["confs"].append(conf); g["n"] += 1
    boxes = []
    for g in groups.values():
        if g["n"] < 1:
            continue
        x0, y0 = min(g["xs"]), min(g["ys"])
        x1, y1 = max(g["x2s"]), max(g["y2s"])
        boxes.append((x0, y0, x1 - x0, y1 - y0,
                      g["n"], float(np.mean(g["confs"]))))
    return sorted(boxes, key=lambda b: b[1])


def render_layout_map(shape, boxes):
    """Layout visualization with confidence-colored boxes and L# labels."""
    canvas = np.ones((shape[0], shape[1], 3), dtype=np.uint8) * 250
    for i, b in enumerate(boxes):
        x, y, w, h = b[0], b[1], b[2], b[3]
        conf = b[5] if len(b) >= 6 else 90
        if conf >= 80:
            col = (60, 170, 80)
        elif conf >= 50:
            col = (40, 170, 220)
        else:
            col = (60, 60, 220)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), col, 2)
        cv2.putText(canvas, f"L{i+1}", (x + 4, max(y + h - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    return canvas


def render_annotated(restored_bgr, ocr_data):
    """
    Annotate the restored image with per-word boxes colour-coded by
    OCR confidence (green = high, amber = medium, red = low).
    """
    out = restored_bgr.copy()
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    n = len(ocr_data['text'])
    for i in range(n):
        txt = ocr_data['text'][i].strip()
        try:
            conf = float(ocr_data['conf'][i])
        except (ValueError, TypeError):
            conf = -1
        if not txt or conf < 0:
            continue
        x, y = ocr_data['left'][i], ocr_data['top'][i]
        w, h = ocr_data['width'][i], ocr_data['height'][i]
        if conf >= 80:
            col = (60, 170, 80)
        elif conf >= 50:
            col = (40, 170, 220)
        else:
            col = (60, 60, 220)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 1)
    return out


# --------------------------------------------------------------------------
# OCR & metadata
# --------------------------------------------------------------------------

def run_ocr(binary):
    """Tesseract with word-level confidence. Returns text + conf metrics."""
    # PSM 6 = single uniform block of text — good for documents
    config = "--psm 6 --oem 3"
    data = pytesseract.image_to_data(binary, config=config,
                                     output_type=Output.DICT)
    raw_text = pytesseract.image_to_string(binary, config=config)

    # Build a CLEANED version of the text by grouping tokens by line and
    # keeping only those with confidence >= 40 — drops most speckle reads.
    lines = {}
    confs = []
    for i, tok in enumerate(data['text']):
        tok = tok.strip()
        try:
            c = float(data['conf'][i])
        except (ValueError, TypeError):
            c = -1
        if not tok or c < 0:
            continue
        confs.append(c)
        if c < 40:
            continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        lines.setdefault(key, []).append((data['left'][i], tok))
    clean_lines = []
    for key in sorted(lines.keys()):
        toks = sorted(lines[key], key=lambda t: t[0])
        clean_lines.append(" ".join(t for _, t in toks))
    clean_text = "\n".join(l for l in clean_lines if l.strip())

    if confs:
        overall = float(np.mean(confs))
        high_pct = 100.0 * sum(1 for c in confs if c >= 80) / len(confs)
        med_pct  = 100.0 * sum(1 for c in confs if 50 <= c < 80) / len(confs)
        low_pct  = 100.0 * sum(1 for c in confs if c < 50) / len(confs)
    else:
        overall = high_pct = med_pct = low_pct = 0.0
    return {
        "text": clean_text,       # filtered, line-grouped
        "raw_text": raw_text,     # original Tesseract output for reference
        "data": data,
        "overall_pct": round(overall),
        "high_pct": round(high_pct),
        "distribution": [
            {"label": "High",   "value": round(high_pct)},
            {"label": "Medium", "value": round(med_pct)},
            {"label": "Low",    "value": round(low_pct)},
        ],
    }


def extract_metadata(text):
    """
    Pull structured facts from OCR text. Designed for anthropological /
    sociological archival documents: dates, persons, places, document type.
    """
    found = {"dates": [], "persons": [], "places": [], "doc_type": None,
             "numeric_facts": []}

    # --- Dates: multiple formats ---
    # "14 March 1937", "March 14, 1937", "14/03/1937", "1937"
    months = (r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
              r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
              r"Nov(?:ember)?|Dec(?:ember)?")
    date_patterns = [
        rf"\b\d{{1,2}}\s+(?:{months})\s+\d{{4}}\b",
        rf"\b(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b(?:18|19|20)\d{2}\b",
    ]
    seen = set()
    for pat in date_patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            if m.lower() not in seen:
                seen.add(m.lower())
                found["dates"].append(m)

    # --- Persons: titles + capitalised names; Malay nasab patterns ---
    person_patterns = [
        r"\b(?:Dr|Prof|Mr|Mrs|Ms|Sir|Lady|Tuan|Encik|Cik|Puan|Hj|Haji)\."
        r"?\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b",
        r"\b[A-Z][a-z]+(?:\s+bin|\s+binti|\s+bt\.?|\s+b\.?)\s+[A-Z][a-z]+\b",
    ]
    seen = set()
    for pat in person_patterns:
        for m in re.findall(pat, text):
            key = re.sub(r"\s+", " ", m.strip()).lower()
            if key not in seen and len(key) > 4:
                seen.add(key)
                found["persons"].append(m.strip())

    # --- Places: explicit place markers ---
    place_patterns = [
        r"\b(?:Kampong|Kampung|Bandar|Daerah|Mukim|Negeri|Pulau|Sungai|"
        r"Bukit|Taman|Jalan)\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b",
        # Trailing "in/at/near <Capitalized place>"
        r"\b(?:in|at|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b",
    ]
    seen = set()
    for pat in place_patterns:
        for m in re.findall(pat, text):
            place = m if isinstance(m, str) else m[0]
            key = place.lower()
            if key not in seen and len(key) > 3:
                seen.add(key)
                found["places"].append(place.strip())

    # --- Document type: priority order; titles first then body ---
    type_priority = [
        ("field notes",       "Ethnographic Field Notes"),
        ("ethnographic",      "Ethnographic Field Notes"),
        ("census",            "Census Record"),
        ("manuscript",        "Manuscript"),
        ("minutes",           "Meeting Minutes"),
        ("survey",            "Survey Document"),
        ("report",            "Research Report"),
        ("interview",         "Interview Transcript"),
    ]
    head = text[:250].lower()
    body = text.lower()
    # Try title region first; fall back to whole body
    for kw, label in type_priority:
        if kw in head:
            found["doc_type"] = label
            break
    if not found["doc_type"]:
        for kw, label in type_priority:
            if kw in body:
                found["doc_type"] = label
                break

    # --- Numeric facts: lines containing multiple numbers (table-ish rows) ---
    for line in text.splitlines():
        nums = re.findall(r"\b\d+\b", line)
        if len(nums) >= 2 and len(line.strip()) < 80:
            found["numeric_facts"].append(line.strip())

    return found


def extract_table(ocr_data, text):
    """
    Detect a simple table in the OCR output: a header line whose next line
    contains the same number of numeric tokens. Returns one table or None.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines[:-1]):
        # Header candidate: 2-5 words, no digits, mostly TitleCase
        if re.search(r"\d", line):
            continue
        words = line.split()
        if not (2 <= len(words) <= 6):
            continue
        # Look at next 1-3 lines for a numeric row
        for j in range(i + 1, min(i + 4, len(lines))):
            nums = re.findall(r"\b\d+\b", lines[j])
            if len(nums) == len(words):
                # Heuristic header check — title-case-ish
                if sum(1 for w in words if w[0].isupper()) >= len(words) // 2:
                    return {
                        "table_id": "T1",
                        "headers": words,
                        "rows": [nums],
                    }
    return None


# --------------------------------------------------------------------------
# Health score
# --------------------------------------------------------------------------

def health_score(gray_input, ocr_overall_pct):
    """
    Combined input-quality + output-confidence score (0-100).
    Inputs considered:
      - Sharpness (Laplacian variance)
      - Contrast (stddev of pixel values)
      - OCR confidence (already in 0-100)
    """
    lap_var = cv2.Laplacian(gray_input, cv2.CV_64F).var()
    sharp = min(lap_var / 500.0, 1.0)          # 500+ counts as sharp
    contrast = min(float(gray_input.std()) / 64.0, 1.0)  # 64+ stddev is good
    ocr = ocr_overall_pct / 100.0
    score = (0.25 * sharp + 0.25 * contrast + 0.50 * ocr) * 100
    return int(round(score))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process_document(input_path, output_dir):
    t0 = time.perf_counter()
    os.makedirs(output_dir, exist_ok=True)

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(input_path)
    H, W = img_bgr.shape[:2]
    audit = []

    # === Restoration pipeline ===
    gray = to_grayscale(img_bgr)
    audit.append({"name": "Grayscale conversion",
                  "description": "Collapsed RGB to single channel to eliminate "
                                 "chromatic noise from paper aging."})

    norm = normalize_background(gray, kernel=35)
    audit.append({"name": "Background normalization",
                  "description": "Estimated paper background via large "
                                 "morphological closing (kernel=35) then divided "
                                 "the image by it — removes uneven staining, "
                                 "vignette, and yellowing in one step."})

    enhanced = enhance_contrast(norm, clip=2.5)
    audit.append({"name": "CLAHE contrast enhancement",
                  "description": "Contrast Limited Adaptive Histogram Equalization "
                                 "with clipLimit=2.5 to recover faded ink without "
                                 "amplifying noise."})

    denoised = denoise(enhanced)
    audit.append({"name": "Non-local means denoising",
                  "description": "Removed residual speckle while preserving "
                                 "letter edges (h=10, 7x7 patches, 21x21 search "
                                 "window)."})

    binary = sauvola_threshold(denoised, window=25, k=0.2)
    audit.append({"name": "Sauvola adaptive binarization",
                  "description": "Local-statistics thresholding (window=25, k=0.2). "
                                 "Handles non-uniform background better than Otsu."})

    n_before = (binary < 128).sum()
    binary = remove_small_components(binary, min_area=15)
    n_after = (binary < 128).sum()
    audit.append({"name": "Speckle removal",
                  "description": f"Removed isolated dark blobs <15 px² via "
                                 f"connected-component analysis "
                                 f"({n_before - n_after:,} noise pixels eliminated)."})

    binary, crop_box = crop_to_content(binary, margin=30)
    audit.append({"name": "Content-region crop",
                  "description": f"Cropped to text-dense region "
                                 f"{crop_box[2]}×{crop_box[3]} px "
                                 f"(offset {crop_box[0]}, {crop_box[1]}). "
                                 f"Eliminates blank stained margins."})

    binary, skew_angle = deskew(binary)
    audit.append({"name": "Deskew",
                  "description": f"Estimated skew via minAreaRect of text pixels "
                                 f"and rotated by {skew_angle:+.2f}°."})

    # Build a clean "after" RGB for the slider
    after_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # === OCR (must run before layout so we can use validated line groupings) ===
    ocr = run_ocr(binary)
    audit.append({"name": "OCR (Tesseract LSTM)",
                  "description": f"Extracted text with overall word confidence "
                                 f"{ocr['overall_pct']}%. "
                                 f"{ocr['high_pct']}% of words classified high-conf "
                                 f"(≥80)."})

    # === Layout analysis from OCR line groupings ===
    boxes = detect_text_lines_from_ocr(ocr['data'], min_conf=30)
    layout_img = render_layout_map(binary.shape, boxes)
    audit.append({"name": "Text-line layout analysis",
                  "description": f"Grouped {len(boxes)} text lines from OCR "
                                 f"block/paragraph/line indices; boxes colour-"
                                 f"coded by mean line confidence."})

    annotated = render_annotated(after_rgb, ocr['data'])

    # === Metadata extraction ===
    meta = extract_metadata(ocr['text'])
    audit.append({"name": "Metadata extraction",
                  "description": f"Regex-based extraction found "
                                 f"{len(meta['dates'])} date(s), "
                                 f"{len(meta['persons'])} person(s), "
                                 f"{len(meta['places'])} place(s). "
                                 f"Document type: {meta['doc_type'] or 'unknown'}."})

    table = extract_table(ocr['data'], ocr['text'])

    # === Save visual outputs ===
    base = os.path.splitext(os.path.basename(input_path))[0]
    paths = {
        "before":    f"{output_dir}/{base}_before.png",
        "after":     f"{output_dir}/{base}_after.png",
        "layout":    f"{output_dir}/{base}_layout.png",
        "annotated": f"{output_dir}/{base}_annotated.png",
    }
    cv2.imwrite(paths["before"],    img_bgr)
    cv2.imwrite(paths["after"],     after_rgb)
    cv2.imwrite(paths["layout"],    layout_img)
    cv2.imwrite(paths["annotated"], annotated)

    # === Build the golden-payload JSON ===
    doc_id = hashlib.md5(open(input_path, "rb").read()).hexdigest()[:12]
    proc_time = time.perf_counter() - t0
    hs = health_score(gray, ocr['overall_pct'])

    payload = {
        "metadata": {
            "doc_id": doc_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "filename": os.path.basename(input_path),
            "health_score": hs,
            "width": W,
            "height": H,
            "processing_time_seconds": round(proc_time, 2),
            "status": "success",
        },
        "images": {
            "before_url":    paths["before"],
            "after_url":     paths["after"],
            "annotated_url": paths["annotated"],
            "layout_url":    paths["layout"],
        },
        "content": {
            "extracted_text": ocr['text'],
            "tables": [table] if table else [],
        },
        "insights": {
            "confidence": {
                "overall_pct": ocr['overall_pct'],
                "high_confidence_pct": ocr['high_pct'],
                "distribution_chart_data": ocr['distribution'],
            },
            "audit_trail": audit,
        },
        # Extra fields not in the dashboard schema but useful for the LLM
        "_extras": {
            "extracted_entities": meta,
            "skew_corrected_deg": round(skew_angle, 2),
        },
    }

    # === HOOK === In OpenClaw, append the restored image as base64 so the
    # model can re-OCR it with native vision for higher accuracy than Tesseract.
    with open(paths["after"], "rb") as f:
        payload["_extras"]["claude_vision_payload"] = {
            "type": "image",
            "media_type": "image/png",
            "data_b64_truncated": base64.b64encode(f.read()).decode()[:60] + "...",
            "note": "Pass the full base64 of after_url to the LLM as a "
                    "vision message; the model's OCR will outperform Tesseract "
                    "on degraded text."
        }

    return payload


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/old_newspaper.jpg"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "output/script_1/old_newspaper"
    result = process_document(inp, outdir)
    # Save full payload
    with open(f"{outdir}/payload.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "_extras"},
                     indent=2)[:2000])
    print("\n--- entities ---")
    print(json.dumps(result["_extras"]["extracted_entities"], indent=2))