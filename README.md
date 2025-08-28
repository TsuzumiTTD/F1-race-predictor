
A Streamlit app that predicts the next F1 race winner using historical finishes, qualifying, driver/track history, and optional weather enrichment.

## Features
- Single-driver win probability with readable feature contributions (percent impact ↑/↓).
- Full-grid win probabilities with an editable table.
- Weather chips (temp/rain/wet) and country flag for the selected circuit.
- “Tweak inputs” (grid adjustment, wet/hot) that update the main table.
- Load grid from Kaggle CSVs, upload a CSV, or fetch latest qualifying via Ergast.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/winner_app.py
