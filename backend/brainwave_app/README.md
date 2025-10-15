# BrainWave Analyzer — Streamlit & API Demo

This is a lightweight demo backend for the BrainWave Analyzer project. It contains:

- A Streamlit app demonstrating EEG->Image and Image->EEG workflows (`app.py`).
- A small FastAPI server providing two endpoints the frontend can call (`api.py`).
- A simple `models.py` with randomly initialized demo models (untrained).

Requirements
- See `requirements.txt`. TensorFlow is heavy; consider `tensorflow-cpu` if you don't need GPU support.

Run locally (PowerShell)
1. Create and activate a virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

3. Run the Streamlit app (separate terminal):

```powershell
streamlit run app.py
```

4. Run the FastAPI server (this is what the frontend expects on port 8000):

```powershell
# from this folder
uvicorn api:app --host 0.0.0.0 --port 8000
```

Notes
- The API endpoints are:
  - POST /api/eeg-to-image — JSON {"eeg": [..]} -> returns {"image_data_url": "data:image/png;base64,..."}
  - POST /api/image-to-eeg — form upload 'file' -> returns {"eeg": [..]}

If you'd like a Dockerfile, CI, or a tiny trained checkpoint for demo deterministic outputs, I can add those next.