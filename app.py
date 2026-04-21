import io
import re
import shutil

import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance
try:
    import cv2
except ImportError:
    cv2 = None

st.set_page_config(page_title="OCR Demo", layout="centered")
st.title("OCR Demo: Upload image → extract fields")
st.write("Upload a PAN or RC image, then review and edit the extracted fields.")

ocr_model = None
ocr_engine = None
paddle_available = False
pytesseract_available = False

try:
    from paddleocr import PaddleOCR

    ocr_model = PaddleOCR(use_angle_cls=False, lang="en")
    ocr_engine = "paddle"
    paddle_available = True
except Exception:
    try:
        import pytesseract

        pytesseract_available = True
        ocr_engine = "tesseract"
    except Exception:
        ocr_engine = None


def extract_key_values(lines):
    values = {}
    for line in lines:
        if ":" in line:
            left, right = line.split(":", 1)
            values[left.strip().lower()] = right.strip()
    return values


def find_pan_name(lines, pan_index):
    for i in range(pan_index - 1, -1, -1):
        candidate = lines[i].strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if any(skip in lower for skip in ["name", "permanent account number", "income tax", "government", "india", "atr", "fen", "wt"]):
            continue
        if any(char.isdigit() for char in candidate):
            continue
        if len(candidate) < 3:
            continue
        return candidate
    return ""


def parse_pan_document(lines):
    parsed = {}
    for i, line in enumerate(lines):
        lower = line.lower()
        if "pan" in lower and any(x in lower for x in ["number", "no", "nfb"]):
            for l in range(max(0, i - 3), min(len(lines), i + 5)):
                candidate = lines[l].strip()
                match = re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", candidate)
                if match:
                    parsed["pan_number"] = match.group(0)
                    break
        if any(x in lower for x in ["atr", "name", "नाम"]):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 3 and not any(c.isdigit() for c in next_line):
                    if not any(skip in next_line.lower() for skip in ["father", "हिन्दी", "नाम", "atr", "fen", "wt"]):
                        if re.match(r"^[A-Z][A-Z\s]*[A-Z]$", next_line):
                            parsed["name"] = next_line
                            break
    return parsed


def normalize_reg_number(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())



def parse_rc_table(lines):
    parsed = {}
    
    for i, line in enumerate(lines):
        lower = line.lower()
        
        # Handle Regn. Number + Maker's Name on same header line
        if "regn" in lower and ("number" in lower or "no" in lower) and "maker" in lower:
            # Check next 1-2 lines for data
            for j in range(i + 1, min(i + 3, len(lines))):
                data_line = lines[j].strip()
                if not data_line:
                    continue
                
                # Extract registration number - support multiple formats: 9BG1547, CG09BG1547, etc.
                reg_match = re.search(r"(\d{0,2}[A-Z]{2}\d{4}|[A-Z]{2}\d{2}[A-Z]{2}\d{4})", data_line, re.IGNORECASE)
                if reg_match:
                    parsed["registration_number"] = normalize_reg_number(reg_match.group(1))
                    
                    # Extract maker name: look for known keywords after regn number
                    if "ashok" in data_line.lower():
                        maker_match = re.search(r"ASHOK\s+LEYLAND\s+LTD", data_line, re.IGNORECASE)
                        if maker_match:
                            parsed["maker_name"] = maker_match.group(0)
                    elif "tata" in data_line.lower():
                        maker_match = re.search(r"TATA\s+\w+(\s+\w+)?", data_line, re.IGNORECASE)
                        if maker_match:
                            parsed["maker_name"] = maker_match.group(0)
                    elif "maruti" in data_line.lower():
                        maker_match = re.search(r"MARUTI\s+\w+(\s+\w+)?", data_line, re.IGNORECASE)
                        if maker_match:
                            parsed["maker_name"] = maker_match.group(0)
                    break
        
        # Handle Model Name (usually on its own)
        if "model" in lower and ("name" in lower or "model" in lower) and "colour" not in lower and "body" not in lower:
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line not in ["—", "-", "–", "'", ""] and not any(c in next_line for c in ["Body", "Colour"]):
                    parsed["model_name"] = next_line
        
        # Handle Colour + Body Type on same header line
        if ("colour" in lower or "color" in lower) and "body" in lower and "type" in lower:
            # Check next 1-2 lines for data
            for j in range(i + 1, min(i + 3, len(lines))):
                data_line = lines[j].strip()
                if not data_line or len(data_line) < 3:
                    continue
                
                # Extract colour: first word (BROWN, WHITE, RED, etc.)
                colour_match = re.match(r"([A-Z]+)\s+", data_line)
                if colour_match:
                    colour = colour_match.group(1)
                    if colour not in ["TRK", "CG", "P", "S", "N"]:
                        parsed["colour"] = colour
                
                # Extract body type: everything after colour, before noise
                remaining = data_line[colour_match.end():].strip() if colour_match else data_line
                body_match = re.match(r"([A-Z\s\(\)]+?)(?:\s*[<>~]|$)", remaining)
                if body_match:
                    body_type = body_match.group(1).strip()
                    if body_type and body_type not in ["-", "—", "–"]:
                        parsed["body_type"] = body_type
                break
        
        # Handle Vehicle Class from header line or next line
        if "vehicle class" in lower:
            value = None
            if ":" in line:
                parts = line.split(":", 1)
                value = parts[1].strip()
            if not value and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 1 and not next_line.lower().startswith("model"):
                    value = next_line
            if value:
                parsed["vehicle_class"] = value
    
    return parsed if parsed else None


def preprocess_region(img):
    """Enhance contrast and denoise for better OCR"""
    if cv2 is None:
        pil_img = Image.fromarray(img)
        enhancer = ImageEnhance.Contrast(pil_img)
        return np.array(enhancer.enhance(1.5))
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    filtered = cv2.bilateralFilter(denoised, 9, 75, 75)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)


def run_ocr_on_region(image_region, region_name, ocr_engine, ocr_model):
    """Run OCR on a specific region"""
    lines = []
    try:
        if ocr_engine == "paddle":
            result = ocr_model.ocr(image_region, cls=False)
            lines = [line[1][0] for line in result[0]] if result else []
        elif ocr_engine == "tesseract":
            import pytesseract
            raw_text = pytesseract.image_to_string(image_region)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    except Exception as e:
        st.warning(f"Error processing {region_name}: {str(e)}")
    
    if lines:
        st.write(f"✓ {region_name}: {len(lines)} lines extracted")
    return lines


def extract_fields(text_lines):
    text = "\n".join(text_lines)
    values = extract_key_values(text_lines)
    rc_parsed = parse_rc_table(text_lines)
    if rc_parsed:
        return {
            "doc_type": "RC",
            "pan_name": "",
            "pan_number": "",
            "registration_number": rc_parsed.get("registration_number", ""),
            "vehicle_class": rc_parsed.get("vehicle_class", ""),
            "maker_name": rc_parsed.get("maker_name", ""),
            "model_name": rc_parsed.get("model_name", ""),
            "colour": rc_parsed.get("colour", ""),
            "body_type": rc_parsed.get("body_type", ""),
            "all_text": text,
        }

    pan_match = re.search(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", text)
    if pan_match:
        pan_parsed = parse_pan_document(text_lines)
        pan_number = pan_parsed.get("pan_number") or pan_match.group(0)
        pan_name = pan_parsed.get("name", "")
        if not pan_name:
            pan_line = next((i for i, line in enumerate(text_lines) if pan_match.group(0) in line), None)
            if pan_line is not None:
                pan_name = find_pan_name(text_lines, pan_line)
        return {
            "doc_type": "PAN",
            "pan_name": pan_name,
            "pan_number": pan_number,
            "registration_number": "",
            "vehicle_class": "",
            "maker_name": "",
            "model_name": "",
            "colour": "",
            "body_type": "",
            "all_text": text,
        }

    reg_match = None
    reg_pattern = re.compile(r"[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,2}\s*\d{4}", re.IGNORECASE)
    for line in text_lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = reg_pattern.search(stripped)
        if match:
            reg_match = normalize_reg_number(match.group(0))
            break

    reg_number = values.get("regn no", "") or values.get("regn no.", "") or values.get("registration no", "") or values.get("registration number", "") or values.get("regn", "") or reg_match or ""
    vehicle_class = values.get("vehicle class", "") or values.get("class", "") or values.get("vehicle type", "")
    maker_name = values.get("maker", "") or values.get("maker name", "") or values.get("manufacturer", "") or values.get("make", "")

    if not maker_name:
        for line in text_lines:
            lower = line.lower()
            if "maker" in lower and ":" not in line:
                maker_name = line.split("maker", 1)[-1].strip(" :-")
                if maker_name:
                    break
            if "manufacture" in lower and ":" not in line:
                maker_name = line.split("manufacture", 1)[-1].strip(" :-")
                if maker_name:
                    break

    if reg_number and not vehicle_class:
        for line in text_lines:
            if "vehicle class" in line.lower() or "class" in line.lower():
                parts = line.split(":", 1)
                if len(parts) > 1:
                    vehicle_class = parts[1].strip()
                    break

    doc_type = "PAN" if pan_match else "RC" if reg_number or any(k in text.lower() for k in ["regn", "registration", "maker", "vehicle class"]) else "UNKNOWN"

    return {
        "doc_type": doc_type,
        "pan_name": "",
        "pan_number": "",
        "registration_number": reg_number,
        "vehicle_class": vehicle_class,
        "maker_name": maker_name,
        "model_name": "",
        "colour": "",
        "body_type": "",
        "all_text": text,
    }


uploaded = st.file_uploader("Upload RC or PAN image", type=["png", "jpg", "jpeg", "webp", "tiff", "bmp"])

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(image, caption="Uploaded document", use_container_width=True)

    if not paddle_available and not pytesseract_available:
        st.warning("No OCR engine available. Install dependencies and rerun.")
        st.stop()
    
    if ocr_engine == "tesseract" and not shutil.which("tesseract"):
        st.warning("`pytesseract` installed but Tesseract binary missing. Install: `sudo apt-get install -y tesseract-ocr`")
        st.stop()

    st.info("🔍 Multi-pass extraction: full image + top region + contrast-enhanced...")
    image_np = np.array(image)
    
    # Define regions
    height = image_np.shape[0]
    width = image_np.shape[1]
    top_crop = int(height * 0.25)
    mid_start = top_crop
    mid_end = int(height * 0.65)
    
    top_region = image_np[:top_crop, :]
    mid_region = image_np[mid_start:mid_end, :]
    bottom_region = image_np[int(height * 0.65):, :]
    
    all_lines = []
    
    # Pass 1: Full image (baseline)
    st.write("📄 Pass 1: Full image OCR")
    lines_full = run_ocr_on_region(image_np, "Full image", ocr_engine, ocr_model)
    all_lines.extend(lines_full)
    
    # Pass 2: Top region (priority for Vehicle Class)
    st.write("📄 Pass 2: Top region (Vehicle Class)")
    lines_top = run_ocr_on_region(top_region, "Top region", ocr_engine, ocr_model)
    all_lines = lines_top + all_lines
    
    # Pass 3: Middle region (enhanced contrast)
    st.write("📄 Pass 3: Middle region (enhanced contrast)")
    mid_enhanced = preprocess_region(mid_region)
    lines_mid = run_ocr_on_region(mid_enhanced, "Middle region", ocr_engine, ocr_model)
    all_lines.extend(lines_mid)
    
    # Pass 4: Bottom region (enhanced contrast)
    st.write("📄 Pass 4: Bottom region (enhanced contrast)")
    bottom_enhanced = preprocess_region(bottom_region)
    lines_bottom = run_ocr_on_region(bottom_enhanced, "Bottom region", ocr_engine, ocr_model)
    all_lines.extend(lines_bottom)
    
    # Remove duplicates while preserving order
    seen = set()
    lines = []
    for line in all_lines:
        if line.strip() and line not in seen:
            lines.append(line)
            seen.add(line)
    
    st.success(f"✓ Total unique lines extracted: {len(lines)}")
    
    # Diagnostic: show raw top region extraction
    with st.expander("🔍 Raw OCR from TOP REGION:"):
        if lines_top:
            for i, line in enumerate(lines_top):
                st.write(f"{i}: {line}")
        else:
            st.write("(No text detected in top region)")
    
    fields = extract_fields(lines)

    st.info(f"Detected document type: {fields['doc_type']}")
    st.subheader("Extracted fields")

    if fields["doc_type"] == "PAN":
        pan_input = st.text_input("Name", value=fields["pan_name"])
        pan_number_input = st.text_input("PAN number", value=fields["pan_number"])
    elif fields["doc_type"] == "RC":
        reg_input = st.text_input("Registration number", value=fields["registration_number"])
        maker_input = st.text_input("Maker's name", value=fields["maker_name"])
        vehicle_class_input = st.text_input("Vehicle class", value=fields["vehicle_class"])
        model_input = st.text_input("Model name", value=fields["model_name"])
        colour_input = st.text_input("Colour", value=fields["colour"])
        body_input = st.text_input("Body type", value=fields["body_type"])
    else:
        st.warning("Unable to fully detect document type. Edit the fields below manually.")
        pan_input = st.text_input("Name", value=fields["pan_name"])
        pan_number_input = st.text_input("PAN number", value=fields["pan_number"])
        reg_input = st.text_input("Registration number", value=fields["registration_number"])
        maker_input = st.text_input("Maker's name", value=fields["maker_name"])
        vehicle_class_input = st.text_input("Vehicle class", value=fields["vehicle_class"])
        model_input = st.text_input("Model name", value=fields["model_name"])
        colour_input = st.text_input("Colour", value=fields["colour"])
        body_input = st.text_input("Body type", value=fields["body_type"])

    st.subheader("Raw OCR text")
    st.text_area("OCR output", value=fields["all_text"], height=240)

    st.markdown("---")
    st.write("If the auto-extracted values are incorrect, edit them above and copy the final values for your demo flow.")

    if not lines:
        st.error("No text was extracted. Try uploading a clearer image or a different document.")
