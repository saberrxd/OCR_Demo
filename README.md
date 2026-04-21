# OCR_Demo

A minimal demo to prove: upload an image, extract fields, and show editable output.

## What this demo does

- Upload a PAN or RC image
- Extract text using PaddleOCR or Tesseract OCR
- Detect PAN name, PAN number, and RC fields like registration number, vehicle class, maker's name, model name, colour, and body type
- Display extracted fields in an editable UI

## Run locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If Tesseract OCR is used as a fallback, install the system binary as well:
   ```bash
   sudo apt-get update && sudo apt-get install -y tesseract-ocr
   ```
2. Run the Streamlit app safely:
   ```bash
   python -m streamlit run app.py
   ```

   Or use the helper script:
   ```bash
   ./run.sh
   ```

## Notes

- This is intentionally small and focused: no backend, no WhatsApp, no scaling.
- If PaddleOCR is not installed, the app will prompt you to install dependencies.
