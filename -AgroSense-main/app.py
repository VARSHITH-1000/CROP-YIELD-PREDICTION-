import csv
import io
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from model_utils import FEATURE_NAMES, load_models, predict_yield


app = FastAPI(
    title="Crop Yield Prediction Using Machine Learning",
    description="Crop-yield estimates and iterative forecasts using the existing HGB, XGBoost, and Ridge ensemble.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

models = load_models()
HISTORY_PATH = Path(__file__).resolve().parent / "prediction_history.csv"
HISTORY_FIELDS = ["timestamp", "country", "crop", "year", "yield_kg_ha"]
HISTORY_LOCK = threading.Lock()

Country = Literal[
    "argentina", "australia", "brazil", "canada", "china", "france",
    "germany", "india", "indonesia", "japan", "mexico", "russia",
    "south africa", "uk", "usa",
]
Crop = Literal["maize", "rice", "wheat"]


class PredictRequest(BaseModel):
    year: int = Field(ge=1990, le=2100)
    country: Country
    crop: Crop


class CompareRequest(BaseModel):
    first: PredictRequest
    second: PredictRequest


def _record_predictions(results):
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "timestamp": timestamp,
            "country": request.country,
            "crop": request.crop,
            "year": request.year,
            "yield_kg_ha": result["yield"],
        }
        for request, result in results
    ]
    with HISTORY_LOCK:
        needs_header = not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0
        with HISTORY_PATH.open("a", newline="", encoding="utf-8") as history_file:
            writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerows(rows)


def _run_prediction(request: PredictRequest):
    result = predict_yield(
        year=request.year,
        country=request.country,
        crop=request.crop,
        models=models,
    )
    return result


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/styles.css")
def styles():
    return FileResponse("styles.css", media_type="text/css")


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = _run_prediction(req)
        _record_predictions([(req, result)])
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/compare")
def compare_predictions(req: CompareRequest):
    try:
        first_result = _run_prediction(req.first)
        second_result = _run_prediction(req.second)
        _record_predictions([
            (req.first, first_result),
            (req.second, second_result),
        ])
        difference = round(first_result["yield"] - second_result["yield"], 1)
        return {
            "first": {**req.first.model_dump(), **first_result},
            "second": {**req.second.model_dump(), **second_result},
            "difference_kg_ha": difference,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/history")
def prediction_history():
    if not HISTORY_PATH.exists():
        return {"history": []}
    try:
        with HISTORY_LOCK, HISTORY_PATH.open("r", newline="", encoding="utf-8") as history_file:
            rows = list(csv.DictReader(history_file))
        return {"history": rows}
    except (OSError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read prediction history: {exc}") from exc


@app.get("/history/export")
def export_prediction_history():
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=HISTORY_FIELDS)
    writer.writeheader()
    if HISTORY_PATH.exists():
        try:
            with HISTORY_LOCK, HISTORY_PATH.open("r", newline="", encoding="utf-8") as history_file:
                writer.writerows(csv.DictReader(history_file))
        except (OSError, csv.Error) as exc:
            raise HTTPException(status_code=500, detail=f"Could not export prediction history: {exc}") from exc
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="prediction_history.csv"'},
    )


@app.post("/explain")
def explain_prediction(req: PredictRequest):
    feature_gains = models["xgb"].get_score(importance_type="gain")
    importance = []
    for feature_name, gain in feature_gains.items():
        if feature_name.startswith("f") and feature_name[1:].isdigit():
            index = int(feature_name[1:])
            if index < len(FEATURE_NAMES):
                feature_name = FEATURE_NAMES[index]
        importance.append({"feature": feature_name, "gain": float(gain)})
    importance.sort(key=lambda item: item["gain"], reverse=True)

    return {
        "request": req.model_dump(),
        "method": "Global XGBoost component gain importance",
        "scope": "Global model-level ranking; not a per-prediction ensemble attribution.",
        "notice": (
            "Gain shows how the XGBoost component used features across its trees. "
            "It is not a SHAP value, does not describe this individual prediction, "
            "and does not establish causation."
        ),
        "top_features": importance[:8],
    }
