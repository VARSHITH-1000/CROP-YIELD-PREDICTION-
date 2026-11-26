# CropPredict — Crop Yield Prediction System

CropPredict is a crop yield prediction system for estimating and comparing agricultural yields. It uses the project's existing HistGradientBoosting, XGBoost, and Ridge stacking ensemble, serving its frontend through FastAPI and using the existing cleaned agricultural dataset and model artifacts.

## Project background and attribution

This working copy is based on the original **AgroSense** repository by **arsalen01**. The original implementation, machine-learning architecture, datasets, model artifacts, notebooks, analysis outputs, and original MIT license are retained. CropPredict branding, product-facing copy, documentation, and the comparison, local history, and model-information features described below are modifications in this working copy. These changes do not imply that the original implementation was independently developed here.

The original copyright and MIT license notice remain in [LICENSE](LICENSE).

## Existing project functionality

- Predicts crop yield in kg/ha for maize, rice, and wheat across the supported countries.
- Uses the existing HGB + XGBoost base models with a Ridge meta-learner.
- Uses historical dataset rows where available and iteratively forecasts beyond 2023.
- Includes the original cleaned agricultural data, notebooks, model artifacts, and analysis outputs.

The project's original README reported benchmark metrics of R² = 0.967, RMSE = 486 kg/ha, and MAPE = 6.4%. Those reported figures are reproduced here as project-reported metrics and have not been independently re-evaluated as part of these modifications.

## Added features in this working copy

### Compare two scenarios

Select a crop, country, and year for each of two scenarios. The app returns both predictions, their signed difference (Scenario A minus Scenario B), and a responsive bar chart. Both scenarios use the same validated prediction pipeline as the single-prediction endpoint.

### Local prediction history and CSV export

Successful single predictions and both results from successful comparisons are appended to `prediction_history.csv` beside `app.py`. The history contains only a UTC timestamp, country, crop, year, and predicted yield; it does not collect personal information. The page displays saved entries and provides a CSV export. If the file does not exist yet, history is shown as empty and export returns a header-only CSV.

### Model feature explanation

The explanation view reports the strongest available **global XGBoost component gain-importance** features. This is model-level information, not per-prediction ensemble attribution, SHAP output, or evidence of causation. No SHAP values are fabricated.

## Run locally

Python 3.10+ is recommended. Install the dependencies listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI's interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API

- `POST /predict` — predict one validated country/crop/year request.
- `POST /compare` — predict and compare two validated requests.
- `GET /history` — return successful prediction history (empty before the first prediction).
- `GET /history/export` — download the history as CSV.
- `POST /explain` — return the selected input context and global XGBoost component gain ranking.
- `GET /` — serve the application page.

The prediction inputs follow the existing supported values: years 1990–2100, 15 configured countries, and maize, rice, or wheat. Validation errors are returned by FastAPI as HTTP 422 responses.

## Data sources and included assets

The original project describes combining FAOSTAT agricultural production and fertilizer data with climate data. Existing dataset files, model artifacts, notebooks, generated images, and screenshots have been retained; these modifications do not regenerate or replace them.

## License

This project remains distributed under the MIT License in [LICENSE](LICENSE), including the original copyright notice.
