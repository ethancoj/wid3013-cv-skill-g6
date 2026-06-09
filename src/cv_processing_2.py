import os
import cv2
import time
import uuid
import json
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def calculate_health_score(original_gray):
    """
    Heuristically calculates a document 'health_score' (0-100) using OpenCV.
    - Sharpness is estimated using the Laplacian variance (higher is sharper).
    - Dynamic Contrast is estimated using the standard deviation of pixel values.
    """
    # 1. Estimate sharpness (Laplacian Variance)
    # A blurry or degraded document has lower variance.
    lap_var = cv2.Laplacian(original_gray, cv2.CV_64F).var()
    # Normalize sharpness: assume variance of 400+ is excellent (100)
    sharpness_score = min(100, int((lap_var / 400.0) * 100))
    
    # 2. Estimate dynamic range/contrast (Standard Deviation)
    # Low standard deviation indicates poor contrast (faint ink, uniform yellow staining).
    std_dev = np.std(original_gray)
    # Normalize contrast: standard deviation of 60+ is considered clear and high-contrast
    contrast_score = min(100, int((std_dev / 60.0) * 100))
    
    # Combine scores (60% weight on contrast, 40% on sharpness)
    health_score = int((0.6 * contrast_score) + (0.4 * sharpness_score))
    return max(0, min(100, health_score))

def detect_tables_heuristic(binary_image):
    """
    A heuristic approach to find tables by extracting vertical and horizontal gridlines.
    Note: Standard morphological operations struggle with complex or borderless tables.
    For production, deep learning models (like Table Transformer) should replace this logic.
    """
    # Invert the binarized image (lines must be white on black background)
    inverted = cv2.bitwise_not(binary_image)
    
    # Define structural kernels for detecting lines
    cols, rows = binary_image.shape
    horizontal_size = max(1, cols // 30)
    vertical_size = max(1, rows // 30)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    
    # Extract horizontal lines
    temp1 = cv2.erode(inverted, horizontal_kernel, iterations=1)
    horizontal_lines = cv2.dilate(temp1, horizontal_kernel, iterations=1)
    
    # Extract vertical lines
    temp2 = cv2.erode(inverted, vertical_kernel, iterations=1)
    vertical_lines = cv2.dilate(temp2, vertical_kernel, iterations=1)
    
    # Combine lines to find intersection junctions
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables_found = []
    table_count = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter table-like bounding boxes by area and aspect ratio
        if w > 80 and h > 40:
            table_count += 1
            tables_found.append({
                "table_id": f"tbl_{table_count}_{uuid.uuid4().hex[:6]}",
                "headers": ["Detected Region Left", "Detected Region Top", "Width", "Height"],
                "rows": [
                    [str(x), str(y), str(w), str(h)]
                ]
            })
            
    return tables_found

def process_document(input_path, output_dir):
    """
    Main processing pipeline. Cleans images, extracts text, generates annotated & layout maps,
    and returns a structured payload dictionary matching the Golden Payload Schema.
    """
    start_time = time.time()
    doc_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    filename = os.path.basename(input_path)
    
    # Initialize basic failed payload structure
    failed_payload = {
        "metadata": {
            "doc_id": doc_id,
            "timestamp": timestamp,
            "filename": filename,
            "health_score": 0,
            "width": 0,
            "height": 0,
            "processing_time_seconds": 0.0,
            "status": "failed"
        },
        "images": {
            "before_url": input_path,
            "after_url": "",
            "annotated_url": "",
            "layout_url": ""
        },
        "content": {
            "extracted_text": "",
            "tables": []
        },
        "insights": {
            "confidence": {
                "overall_pct": 0,
                "high_confidence_pct": 0,
                "distribution_chart_data": []
            },
            "audit_trail": []
        }
    }
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load Image
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Unable to read image at path: {input_path}")
            
        height, width, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate real dynamic health score metric before processing
        health_score = calculate_health_score(gray)
        
        # Initialize dynamic audit trail tracker
        audit_trail = [
            {"name": "Color Space Conversion", "description": "Converted the input BGR image to grayscale."}
        ]
        
        # 2. Preprocessing & Restoration
        # Denoising: Bilateral filter to smooth background paper textures while keeping text edges sharp
        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        audit_trail.append({
            "name": "Bilateral Denoising Filter", 
            "description": "Applied bilateral filter to remove paper grain and compression artifacts while preserving edge sharpness."
        })
        
        # Contrast Enhancement: CLAHE to even out light distributions and highlight faded text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        audit_trail.append({
            "name": "CLAHE Contrast Enhancement", 
            "description": "Evened out lighting and boosted faint ink strokes using Contrast Limited Adaptive Histogram Equalization."
        })
        
        # Adaptive Thresholding: Binarization to separate text strokes from background stains
        clean_bw = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 10
        )
        audit_trail.append({
            "name": "Adaptive Gaussian Binarization", 
            "description": "Converted enhanced image to crisp black-and-white to eliminate water stains and yellow aging gradients."
        })
        
        # Save Restored Image
        after_filename = f"restored_{filename}"
        after_path = os.path.join(output_dir, after_filename)
        cv2.imwrite(after_path, clean_bw)
        
        # 3. Perform OCR & extract confidence statistics using PyTesseract
        extracted_text = pytesseract.image_to_string(clean_bw).strip()
        ocr_data = pytesseract.image_to_data(clean_bw, output_type=pytesseract.Output.DICT)
        
        # Parse real word-level confidence metrics
        confidences = []
        for conf in ocr_data['conf']:
            # Pytesseract outputs -1 for blank spaces or non-character structures
            if conf != -1:
                confidences.append(conf)
                
        if confidences:
            overall_pct = int(np.mean(confidences))
            high_conf_count = sum(1 for c in confidences if c >= 80)
            med_conf_count = sum(1 for c in confidences if 50 <= c < 80)
            low_conf_count = sum(1 for c in confidences if c < 50)
            
            total_words = len(confidences)
            high_conf_pct = int((high_conf_count / total_words) * 100)
            med_conf_pct = int((med_conf_count / total_words) * 100)
            low_conf_pct = int((low_conf_count / total_words) * 100)
        else:
            overall_pct, high_conf_pct, med_conf_pct, low_conf_pct = 0, 0, 0, 0
            
        distribution_chart_data = [
            {"label": "High", "value": high_conf_pct},
            {"label": "Medium", "value": med_conf_pct},
            {"label": "Low", "value": low_conf_pct}
        ]
        
        # 4. Generate Visualizations
        # A. Annotated Image (individual word bounding boxes coloured by confidence)
        annotated_img = cv2.cvtColor(clean_bw, cv2.COLOR_GRAY2BGR)
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i].strip() != "" and ocr_data['conf'][i] != -1:
                x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                conf = ocr_data['conf'][i]
                
                # Colour-code boxes based on OCR confidence ranges
                color = (0, 255, 0) if conf >= 80 else (0, 255, 255) if conf >= 50 else (0, 0, 255)
                cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 1)
                
        annotated_filename = f"annotated_{filename}"
        annotated_path = os.path.join(output_dir, annotated_filename)
        cv2.imwrite(annotated_path, annotated_img)
        audit_trail.append({
            "name": "OCR Confidence Annotation", 
            "description": "Calculated spatial coordinates of each recognized word and color-coded bounding boxes by Tesseract accuracy ratings."
        })
        
        # B. Layout Map (Paragraph structures using morphological dilation)
        layout_img = cv2.cvtColor(clean_bw, cv2.COLOR_GRAY2BGR)
        # Dilate text to merge characters and words into cohesive block shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        dilated = cv2.dilate(cv2.bitwise_not(clean_bw), kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 30 and h > 15:  # Filter out tiny residual noise blocks
                cv2.rectangle(layout_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
        layout_filename = f"layout_{filename}"
        layout_path = os.path.join(output_dir, layout_filename)
        cv2.imwrite(layout_path, layout_img)
        audit_trail.append({
            "name": "Morphological Layout Segmentation", 
            "description": "Dilated character boundaries using morphological spacing kernels to isolate paragraphs, tables, and structural sections."
        })
        
        # 5. Extract Tables via Heuristic Check
        detected_tables = detect_tables_heuristic(clean_bw)
        
        # Complete success payload building
        processing_time_seconds = round(time.time() - start_time, 3)
        
        payload = {
            "metadata": {
                "doc_id": doc_id,
                "timestamp": timestamp,
                "filename": filename,
                "health_score": health_score,
                "width": width,
                "height": height,
                "processing_time_seconds": processing_time_seconds,
                "status": "success"
            },
            "images": {
                "before_url": input_path,
                "after_url": after_path,
                "annotated_url": annotated_path,
                "layout_url": layout_path
            },
            "content": {
                "extracted_text": extracted_text,
                "tables": detected_tables
            },
            "insights": {
                "confidence": {
                    "overall_pct": overall_pct,
                    "high_confidence_pct": high_conf_pct,
                    "distribution_chart_data": distribution_chart_data
                },
                "audit_trail": audit_trail
            }
        }
        return payload
        
    except Exception as e:
        # Fallback error handling if core processing crashes
        print(f"Error during processing: {str(e)}")
        failed_payload["metadata"]["processing_time_seconds"] = round(time.time() - start_time, 3)
        return failed_payload

# --- Example of usage ---
if __name__ == "__main__":
    # Define test directories
    input_image = "data/old_faded.jpg"  # <-- Place your test image filename here
    output_directory = "output/script_2/old_faded"  # <-- Desired output directory for results
    
    # Run the restoration script
    print("Initializing document processing pipeline...")
    result_payload = process_document(input_image, output_directory)
    
    # Save the output payload to JSON
    json_path = os.path.join(output_directory, "payload.json")
    with open(json_path, "w") as f:
        json.dump(result_payload, f, indent=2)
        
    print(f"Processing complete! Payload successfully generated and saved to {json_path}")