#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Use Python to run Streamlit reliably in environments where the streamlit command is unavailable.
python -m streamlit run app.py
