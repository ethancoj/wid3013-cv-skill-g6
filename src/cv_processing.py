"""
DOCUMENT PROCESSOR  —  document_processor.py
==============================================
WID3013 Practical CV Skill Assignment
Topic: Arts & Social Science — Archival Document Restoration

ROLE
─────
CV processing tool for OpenClaw / Hermes.
Accepts an image (JPG/PNG) or a PDF and returns a structured
tool response that OpenClaw reads to generate the final output.

This script does ONLY what computer vision is good at:
  ✓ Image restoration     (denoise → enhance → binarize)
  ✓ Super-resolution      (for low-res inputs)
  ✓ OCR                   (Tesseract LSTM + confidence metrics)
  ✓ Image quality score   (health score 0-100)
  ✓ Visual outputs        (before / after / layout / annotated)
  ✓ PDF handling          (scanned + digital pages)

It does NOT do:
  ✗ Entity extraction     — OpenClaw LLM reads the raw text
  ✗ Summarisation         — OpenClaw LLM's job
  ✗ Schema building       — OpenClaw LLM's job
  ✗ Dashboard generation  — OpenClaw LLM's job

ENTRY POINT
────────────
  from document_processor import process_any

  result = process_any("path/to/file.jpg", "output/folder")
  result = process_any("path/to/file.pdf", "output/folder")

  # result is a dict — passed directly as OpenClaw tool response

TOOL RESPONSE SCHEMA
─────────────────────
{
  "status":   "success" | "warning_low_res_recovered_<method>" | "failed: <reason>",
  "filename": str,

  "image_metrics": {
    "health_score":            int (0-100),
    "width":                   int,
    "height":                  int,
    "resolution_ok":           bool,
    "processing_time_seconds": float
  },

  "ocr": {
    "raw_text":               str,   <- full transcript, noise filtered
    "overall_confidence_pct": int,
    "high_confidence_pct":    int,
    "distribution": [
      {"label": "High",   "value": int},
      {"label": "Medium", "value": int},
      {"label": "Low",    "value": int}
    ]
  },

  "images": {
    "before_path":    str,  <- original input image
    "after_path":     str,  <- restored binary image
    "layout_path":    str,  <- paragraph layout map
    "annotated_path": str   <- confidence-coded word boxes
  },

  "restored_image_b64": str,  <- base64 PNG of after_path
                                  pass to LLM as vision input for
                                  high-accuracy OCR correction

  "audit_trail": [
    {"name": str, "description": str}, ...
  ]
}

PDF-only extra fields:
  "page_count":      int
  "pages_processed": int
  "scanned_pages":   int
  "digital_pages":   int
  "pages":           list  <- per-page breakdown (same schema per page)
"""

import os, cv2, time, base64, numpy as np, pytesseract

# PDF support — install with: pip install pymupdf
try:
    import fitz
    PDF_SUPPORTED = True
except ImportError:
    PDF_SUPPORTED = False

# Super-resolution — requires opencv-contrib-python
try:
    from cv2 import dnn_superres
    SR_SUPPORTED = True
except ImportError:
    SR_SUPPORTED = False


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Linux/Mac: usually not needed (Tesseract found automatically)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_LONG_EDGE     = 1500   # px — below this, SR upscale is triggered
MODELS_DIR        = "models"  # folder with ESPCN_x4.pb / FSRCNN-small_x4.pb
CONF_THRESHOLD    = 40     # OCR tokens below this % are treated as noise
MIN_DIGITAL_CHARS = 50     # PDF page needs > this many chars to be "digital"
PDF_RENDER_DPI    = 300    # DPI for rendering PDF pages to images


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _validate_resolution(img):
    """
    CV CONCEPT: Image quality assessment — spatial resolution check.

    Tesseract needs ~20-30 px per character height to distinguish letter
    shapes.  Below MIN_LONG_EDGE this is rarely met for typical documents.

    Returns (is_ok: bool, message: str)
    """
    h, w = img.shape[:2]
    if max(h, w) < MIN_LONG_EDGE:
        return False, (
            f"Low resolution ({w}x{h} px). "
            f"Minimum recommended: {MIN_LONG_EDGE} px. "
            f"Attempting super-resolution recovery."
        )
    return True, f"Resolution OK ({w}x{h} px)."


def _upscale_if_needed(img):
    """
    CV CONCEPT: Deep learning super-resolution — ESPCN (Efficient Sub-Pixel CNN).

    Only fires when image is below MIN_LONG_EDGE.
    ESPCN runs all convolutions at low resolution then shuffles pixels to
    the target size — much faster than models that upscale first.

    Tested on 500x246 newspaper: 22 words → 242 words extracted after SR.

    Model files (download once into MODELS_DIR):
      ESPCN_x4.pb        — https://github.com/fannymonori/TF-ESPCN
      FSRCNN-small_x4.pb — https://github.com/Saafke/FSRCNN_Tensorflow
    Falls back to bicubic interpolation if no model file found.

    Returns (upscaled_img, method_label, was_applied)
    """
    h, w = img.shape[:2]
    if max(h, w) >= MIN_LONG_EDGE:
        return img, "none", False

    if SR_SUPPORTED:
        for fname, algo, label in [
            ("ESPCN_x4.pb",        "espcn",  "ESPCN_DNN_4x"),
            ("FSRCNN-small_x4.pb", "fsrcnn", "FSRCNN_DNN_4x"),
        ]:
            path = os.path.join(MODELS_DIR, fname)
            if not os.path.exists(path):
                continue
            try:
                sr = dnn_superres.DnnSuperResImpl_create()
                sr.readModel(path)
                sr.setModel(algo, 4)
                return sr.upsample(img), label, True
            except Exception:
                continue

    # Bicubic fallback — always available
    up = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    return up, "bicubic_4x", True


def _health_score(gray, ocr_conf):
    """
    CV CONCEPT: Image quality assessment — three combined signals.

    1. Laplacian variance  — sharpness (measures edge energy)
       High variance = sharp text strokes. Low = blurry scan.
       Threshold: variance >= 400 is fully sharp.

    2. Pixel std deviation — contrast (tonal spread)
       High std = crisp dark ink on light paper.
       Low std = faded, ink and paper similar tone.
       Threshold: std >= 60 is high contrast.

    3. OCR confidence — direct extractability measure.
       Weighted most (40%) because it reflects the actual end goal.

    Returns int 0-100.
    """
    sharp = min(100, int(cv2.Laplacian(gray, cv2.CV_64F).var() / 400.0 * 100))
    cont  = min(100, int(float(np.std(gray)) / 60.0 * 100))
    return max(0, min(100, int(0.30 * sharp + 0.30 * cont + 0.40 * ocr_conf)))


def _restore_document(gray):
    """
    CV CONCEPT: Sequential preprocessing pipeline targeting three degradation types.

    STEP 1 — BILATERAL FILTER (denoising)
      Edge-aware smoothing: only averages pixels that are spatially close
      AND tonally similar. Removes paper grain without blurring ink edges.
      Parameters: d=9, sigmaColor=75, sigmaSpace=75.

    STEP 2 — CLAHE (contrast enhancement)
      Contrast Limited Adaptive Histogram Equalisation.
      Divides image into 8x8 tiles, equalises each independently.
      Recovers faded ink in dark areas without over-brightening clear areas.
      clipLimit=2.0 prevents noise amplification.

    STEP 3 — ADAPTIVE GAUSSIAN THRESHOLD (binarisation)
      Each pixel compared to its 15x15 Gaussian-weighted neighbourhood.
      Threshold T = local_mean - C (C=10).
      Stain-resistant: threshold rises inside stained areas so stain
      pixels are never misclassified as ink.

    Returns (binary_image, audit_entries)
    """
    audit = []

    # Step 1 — bilateral denoising
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    audit.append({
        "name": "Bilateral Denoising Filter",
        "description": (
            "Edge-aware bilateral filter (d=9, sigmaColor=75, sigmaSpace=75). "
            "Removes paper grain and scanner noise while preserving sharp "
            "ink-stroke boundaries that OCR depends on."
        )
    })

    # Step 2 — CLAHE contrast enhancement
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    audit.append({
        "name": "CLAHE Contrast Enhancement",
        "description": (
            "Contrast Limited Adaptive Histogram Equalisation "
            "(clipLimit=2.0, tile=8x8). Recovers faded ink strokes and "
            "compensates for uneven lighting across the document surface."
        )
    })

    # Step 3 — adaptive Gaussian binarisation
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=10
    )
    audit.append({
        "name": "Adaptive Gaussian Binarization",
        "description": (
            "Local Gaussian-weighted threshold per 15x15 neighbourhood (C=10). "
            "Isolates ink strokes from yellow staining and background gradients "
            "that would defeat global thresholding."
        )
    })

    return binary, audit


def _run_ocr(binary):
    """
    CV CONCEPT: Optical character recognition — Tesseract LSTM engine.

    PSM 6  = single uniform block of text (best for full-page documents).
    OEM 3  = LSTM + legacy fallback (best accuracy on archival fonts).

    NOISE FILTERING
      Tokens below CONF_THRESHOLD are scanner-border artefacts.
      Excluded from raw_text but kept in word-level data for the
      annotated image.

    TEXT REASSEMBLY
      Tokens grouped by (block, paragraph, line) and sorted left→right.
      Preserves reading order as proper line breaks.

    NOISE LINE STRIPPING
      Lines at the top/bottom where <35% of characters are alphanumeric
      are removed — always scanner border speckle, never real content.

    Returns dict: raw_text, data, overall_pct, high_pct, distribution
    """
    data = pytesseract.image_to_data(
        binary, config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT
    )
    all_confs = [float(c) for c in data["conf"] if float(c) >= 0]

    # Reassemble confident tokens preserving reading order
    groups = {}
    for i, tok in enumerate(data["text"]):
        tok = tok.strip()
        try:    conf = float(data["conf"][i])
        except: conf = -1.0
        if not tok or conf < CONF_THRESHOLD:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        groups.setdefault(key, []).append((data["left"][i], tok))

    lines = [
        " ".join(t for _, t in sorted(v))
        for _, v in sorted(groups.items())
    ]

    # Strip noise lines from top and bottom only
    def is_noise(line):
        alnum = sum(c.isalnum() for c in line)
        return len(line) > 0 and alnum / len(line) < 0.35

    while lines and is_noise(lines[0]):  lines.pop(0)
    while lines and is_noise(lines[-1]): lines.pop()

    raw_text = "\n".join(l for l in lines if l.strip())

    if all_confs:
        overall  = round(float(np.mean(all_confs)))
        high_pct = round(100 * sum(1 for c in all_confs if c >= 80)        / len(all_confs))
        med_pct  = round(100 * sum(1 for c in all_confs if 50 <= c < 80)   / len(all_confs))
        low_pct  = round(100 * sum(1 for c in all_confs if c < 50)         / len(all_confs))
    else:
        overall = high_pct = med_pct = low_pct = 0

    return {
        "raw_text":   raw_text,
        "data":       data,
        "overall_pct": overall,
        "high_pct":   high_pct,
        "distribution": [
            {"label": "High",   "value": high_pct},
            {"label": "Medium", "value": med_pct},
            {"label": "Low",    "value": low_pct},
        ]
    }


def _render_annotated(binary, ocr_data):
    """
    CV CONCEPT: Spatial confidence visualisation.

    Draws a bounding box around every OCR word, colour-coded by confidence:
      Green  (>=80) — high confidence, almost certainly correct
      Yellow (>=50) — medium confidence, probably correct
      Red    (<50)  — low confidence, verify manually

    Gives the researcher a spatial map of where the transcript is reliable.
    """
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for i, tok in enumerate(ocr_data["text"]):
        if not tok.strip(): continue
        try:    conf = float(ocr_data["conf"][i])
        except: continue
        if conf < 0: continue
        x, y = ocr_data["left"][i], ocr_data["top"][i]
        w, h = ocr_data["width"][i], ocr_data["height"][i]
        col  = (0,200,60) if conf >= 80 else (0,200,230) if conf >= 50 else (60,60,220)
        cv2.rectangle(out, (x, y), (x+w, y+h), col, 1)
    return out


def _render_layout(binary):
    """
    CV CONCEPT: Document layout analysis via morphological dilation.

    Horizontal kernel (15x5) progressively merges text blobs:
      characters → words → lines → paragraph blocks

    Contours on the dilated image mark the document's layout regions —
    title, paragraphs, tables, captions.
    """
    layout  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(cv2.bitwise_not(binary), kernel, iterations=2)
    for cnt in cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 15 and w < binary.shape[1] * 0.95:
            cv2.rectangle(layout, (x, y), (x+w, y+h), (180, 60, 60), 2)
    return layout


# ══════════════════════════════════════════════════════════════════════════════
# CORE: process a single image (numpy array)
# ══════════════════════════════════════════════════════════════════════════════

def _process_image_array(img_bgr, filename, output_dir, page_prefix=None):
    """
    Runs the full CV pipeline on a BGR numpy array.
    Used internally by both process_image() and process_pdf().

    page_prefix: used for PDF multi-page outputs (e.g. "doc_p001")
    Returns a tool response dict.
    """
    start    = time.time()
    prefix   = page_prefix or os.path.splitext(filename)[0]
    audit    = [{"name": "Greyscale Conversion",
                 "description": "Collapsed BGR to single-channel greyscale. "
                                "Colour is irrelevant for OCR."}]

    # Greyscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_bgr.shape[:2]

    # Resolution check
    res_ok, res_msg = _validate_resolution(img_bgr)
    audit.append({"name": "Resolution Validation", "description": res_msg})

    # Super-resolution if needed
    work      = img_bgr
    sr_method = "none"
    if not res_ok:
        work, sr_method, applied = _upscale_if_needed(img_bgr)
        if applied:
            h, w = work.shape[:2]
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            audit.append({
                "name": f"Super-Resolution ({sr_method})",
                "description": (
                    f"Input below {MIN_LONG_EDGE}px. "
                    f"Applied {sr_method} 4x upscale to give Tesseract "
                    f"sufficient pixel density per character."
                )
            })

    # Restoration
    binary, r_audit = _restore_document(gray)
    audit.extend(r_audit)

    # OCR
    ocr = _run_ocr(binary)
    audit.append({
        "name": "OCR — Tesseract LSTM (PSM 6, OEM 3)",
        "description": (
            f"{ocr['overall_pct']}% mean word confidence. "
            f"{ocr['high_pct']}% of tokens high-confidence (>=80%). "
            f"Tokens below {CONF_THRESHOLD}% dropped. "
            f"Noise lines stripped from top/bottom of output."
        )
    })

    # Health score (after OCR so we can include confidence)
    hs = _health_score(gray, ocr["overall_pct"])

    # Visual outputs
    annotated  = _render_annotated(binary, ocr["data"])
    layout_img = _render_layout(binary)
    after_img  = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    audit.append({"name": "Annotated Image",
                  "description": "Per-word bounding boxes: green >=80%, yellow 50-79%, red <50%."})
    audit.append({"name": "Layout Map",
                  "description": "Morphological dilation (kernel 15x5, 2 iterations) "
                                 "merged characters into paragraph blobs."})

    # Save images
    paths = {
        "before":    os.path.join(output_dir, f"{prefix}_before.png"),
        "after":     os.path.join(output_dir, f"{prefix}_after.png"),
        "annotated": os.path.join(output_dir, f"{prefix}_annotated.png"),
        "layout":    os.path.join(output_dir, f"{prefix}_layout.png"),
    }
    cv2.imwrite(paths["before"],    work)
    cv2.imwrite(paths["after"],     after_img)
    cv2.imwrite(paths["annotated"], annotated)
    cv2.imwrite(paths["layout"],    layout_img)

    with open(paths["after"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    status = "success" if res_ok else f"warning_low_res_recovered_{sr_method}"

    return {
        "status":   status,
        "filename": filename,
        "image_metrics": {
            "health_score":            hs,
            "width":                   w,
            "height":                  h,
            "resolution_ok":           res_ok,
            "processing_time_seconds": round(time.time() - start, 2)
        },
        "ocr": {
            "raw_text":               ocr["raw_text"],
            "overall_confidence_pct": ocr["overall_pct"],
            "high_confidence_pct":    ocr["high_pct"],
            "distribution":           ocr["distribution"]
        },
        "images": {
            "before_path":    paths["before"],
            "after_path":     paths["after"],
            "layout_path":    paths["layout"],
            "annotated_path": paths["annotated"]
        },
        "restored_image_b64": b64,
        "audit_trail": audit
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: process a single image file
# ══════════════════════════════════════════════════════════════════════════════

def process_image(input_path, output_dir):
    """
    Processes a single image file (JPG / PNG / BMP / TIFF).
    Returns a tool response dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(input_path)

    img = cv2.imread(input_path)
    if img is None:
        return {"status": f"failed: cannot read '{input_path}'",
                "filename": filename}

    return _process_image_array(img, filename, output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC: process a PDF file
# ══════════════════════════════════════════════════════════════════════════════

def process_pdf(input_path, output_dir, max_pages=None):
    """
    Processes a PDF file, handling both scanned and digital pages.

    SCANNED PAGE  — no embedded text, page is a rasterized image.
      Rendered at PDF_RENDER_DPI (300 DPI) and passed through the
      full CV restoration + OCR pipeline.

    DIGITAL PAGE  — embedded vector text (created by a word processor).
      Text extracted directly from the PDF — no image processing needed.
      Page still rendered to image for the layout/annotated visuals.

    Detection: page.get_text() returns '' for scanned, text for digital.
    A page needs > MIN_DIGITAL_CHARS characters to be classified as digital
    (avoids misclassifying scanned pages with tiny watermarks).

    Returns a tool response dict extended with:
      page_count, pages_processed, scanned_pages, digital_pages, pages[]
    """
    if not PDF_SUPPORTED:
        return {"status": "failed: PyMuPDF not installed. Run: pip install pymupdf",
                "filename": os.path.basename(input_path)}

    start    = time.time()
    filename = os.path.basename(input_path)
    base     = os.path.splitext(filename)[0]
    os.makedirs(output_dir, exist_ok=True)

    try:
        doc = fitz.open(input_path)
    except Exception as e:
        return {"status": f"failed: {e}", "filename": filename}

    total     = len(doc)
    limit     = min(total, max_pages) if max_pages else total
    pages     = []
    all_text  = []
    confs     = []
    healths   = []
    scan_n    = digital_n = 0

    print(f"PDF: {filename} — {total} page(s), processing {limit}")

    for n in range(1, limit + 1):
        page   = doc[n - 1]
        prefix = f"{base}_p{n:03d}"
        print(f"  Page {n}/{limit} — ", end="", flush=True)

        # Render page to image (needed for both scanned and digital)
        mat      = fitz.Matrix(PDF_RENDER_DPI / 72.0, PDF_RENDER_DPI / 72.0)
        pix      = page.get_pixmap(matrix=mat, alpha=False)
        img_rgb  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                       pix.height, pix.width, 3)
        img_bgr  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        embedded = page.get_text().strip()

        if len(embedded) >= MIN_DIGITAL_CHARS:
            # ── DIGITAL PAGE ─────────────────────────────────────────────────
            print("digital")
            digital_n += 1
            scale  = PDF_RENDER_DPI / 72.0
            blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
            raw_text = "\n\n".join(b[4].strip() for b in blocks if b[4].strip())

            # Layout map — draw text block boxes
            layout = img_bgr.copy()
            for b in blocks:
                if not b[4].strip(): continue
                x0, y0, x1, y1 = [int(c * scale) for c in b[:4]]
                cv2.rectangle(layout, (x0, y0), (x1, y1), (180, 60, 60), 2)

            # Annotated — green boxes (digital = fully reliable)
            annotated = img_bgr.copy()
            for b in blocks:
                if not b[4].strip(): continue
                x0, y0, x1, y1 = [int(c * scale) for c in b[:4]]
                cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 200, 60), 1)

            paths = {k: os.path.join(output_dir, f"{prefix}_{k}.png")
                     for k in ("before","after","layout","annotated")}
            cv2.imwrite(paths["before"],    img_bgr)
            cv2.imwrite(paths["after"],     img_bgr)
            cv2.imwrite(paths["layout"],    layout)
            cv2.imwrite(paths["annotated"], annotated)
            with open(paths["after"], "rb") as f:
                b64 = base64.b64encode(f.read()).decode()

            page_result = {
                "page": n, "page_type": "digital",
                "ocr": {
                    "raw_text":               raw_text,
                    "overall_confidence_pct": 100,
                    "high_confidence_pct":    100,
                    "distribution": [{"label":"High","value":100},
                                     {"label":"Medium","value":0},
                                     {"label":"Low","value":0}]
                },
                "image_metrics": {"health_score": 100,
                                   "resolution_ok": True},
                "images": {"before_path": paths["before"],
                           "after_path":  paths["after"],
                           "layout_path": paths["layout"],
                           "annotated_path": paths["annotated"]},
                "restored_image_b64": b64,
                "audit_trail": [{
                    "name": "Direct text extraction (digital PDF)",
                    "description": (
                        f"Page {n} has {len(raw_text)} chars of embedded text. "
                        f"Extracted from PDF content stream — no OCR needed."
                    )
                }]
            }
            confs.append(100)
            healths.append(100)

        else:
            # ── SCANNED PAGE — full CV pipeline ──────────────────────────────
            print("scanned")
            scan_n += 1
            page_result = _process_image_array(img_bgr, filename, output_dir,
                                               page_prefix=prefix)
            page_result["page"]      = n
            page_result["page_type"] = "scanned"
            confs.append(page_result["ocr"]["overall_confidence_pct"])
            healths.append(page_result["image_metrics"]["health_score"])

        pages.append(page_result)
        all_text.append(
            f"--- Page {n} ({page_result['page_type']}) ---\n"
            f"{page_result['ocr']['raw_text']}"
        )

    doc.close()

    avg_conf   = round(sum(confs)   / len(confs))   if confs   else 0
    avg_health = round(sum(healths) / len(healths)) if healths else 0
    first      = pages[0] if pages else {}

    return {
        "status":   "success",
        "filename": filename,
        "image_metrics": {
            "health_score":            avg_health,
            "width":                   0,
            "height":                  0,
            "resolution_ok":           True,
            "processing_time_seconds": round(time.time() - start, 2)
        },
        "ocr": {
            "raw_text":               "\n\n".join(all_text),
            "overall_confidence_pct": avg_conf,
            "high_confidence_pct":    avg_conf,
            "distribution":           first.get("ocr", {}).get("distribution", [])
        },
        "images":             first.get("images", {}),
        "restored_image_b64": first.get("restored_image_b64", ""),
        "audit_trail": [{
            "name": "PDF ingestion",
            "description": (
                f"Opened {filename} ({total} pages total, processed {limit}). "
                f"{scan_n} scanned (full CV pipeline) + "
                f"{digital_n} digital (direct text extraction)."
            )
        }],
        # PDF-specific
        "page_count":      total,
        "pages_processed": limit,
        "scanned_pages":   scan_n,
        "digital_pages":   digital_n,
        "pages":           pages
    }


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — call this from OpenClaw
# ══════════════════════════════════════════════════════════════════════════════

def process_any(input_path, output_dir):
    """
    Single function to call from the OpenClaw skill.
    Accepts any supported file and returns a tool response dict.

    Supported formats:
      Images : .jpg  .jpeg  .png  .bmp  .tif  .tiff
      PDF    : .pdf  (scanned and/or digital pages)
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".pdf":
        return process_pdf(input_path, output_dir)

    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        return process_image(input_path, output_dir)

    else:
        return {
            "status":   f"failed: unsupported file type '{ext}'",
            "filename": os.path.basename(input_path)
        }


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND LINE USAGE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <input_file> [output_dir]")
        print("  input_file  : path to JPG, PNG, or PDF")
        print("  output_dir  : folder for output images (default: output/)")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_dir  = sys.argv[2] if len(sys.argv) > 2 else "output"

    print(f"Processing: {input_path}")
    result = process_any(input_path, output_dir)

    # Save tool response JSON (b64 excluded for readability)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "tool_response.json")
    preview   = {k: v for k, v in result.items() if k != "restored_image_b64"}
    if "pages" in preview:
        preview["pages"] = [
            {k: v for k, v in pg.items() if k != "restored_image_b64"}
            for pg in preview["pages"]
        ]
    preview["restored_image_b64"] = f"<{len(result.get('restored_image_b64',''))} base64 chars>"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print(f"\nStatus  : {result['status']}")
    print(f"Health  : {result['image_metrics']['health_score']} / 100")
    print(f"OCR     : {result['ocr']['overall_confidence_pct']}%  "
          f"({len(result['ocr']['raw_text'])} chars)")
    print(f"Text    : {result['ocr']['raw_text'][:120].replace(chr(10), ' / ')}")
    if "page_count" in result:
        print(f"Pages   : {result['page_count']} total  "
              f"({result['scanned_pages']} scanned, "
              f"{result['digital_pages']} digital)")
    print(f"Saved   : {save_path}")