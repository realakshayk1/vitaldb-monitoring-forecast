# src/api/app.py
from __future__ import annotations

import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.config import TAU, SIGMA_MAX
from src.api.features import WindowStore
from src.api.inference import mc_predict

# IMPORTANT: import your CNN class from wherever it lives in your repo
from src.nn.model import TemporalCNN  # adjust if your path differs

MODEL_PATH = Path("models/nn/cnn_best.pt")

REQS = Counter("predict_requests_total", "Total prediction requests")
ALERTS = Counter("alerts_total", "Total alerts fired by policy")
LAT = Histogram("predict_latency_seconds", "Prediction latency in seconds")

app = FastAPI(title="VitalDB Monitoring Forecast API", version="0.1.0")

store: WindowStore | None = None
model = None
device = "cpu"


class PredictRequest(BaseModel):
    caseid: int
    t_end: int  # seconds from case start (same timebase as windows.parquet)


class PredictResponse(BaseModel):
    caseid: int
    t_end: int
    p_mean: float
    p_std: float
    alert: bool
    tau: float
    sigma_max: float


@app.on_event("startup")
def startup():
    global store, model, device

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model weights not found at {MODEL_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    store = WindowStore()

    # must match your trained model config
    model = TemporalCNN(in_channels=9).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print(f"✅ API started | device={device} | model={MODEL_PATH}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global store, model

    REQS.inc()
    t0 = time.time()

    if store is None or model is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        x_ct = store.get_sequence(req.caseid, req.t_end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    p_mean, p_std = mc_predict(model, x_ct, device=device)

    alert = (p_mean >= TAU) and (p_std <= SIGMA_MAX)
    if alert:
        ALERTS.inc()

    LAT.observe(time.time() - t0)

    return PredictResponse(
        caseid=req.caseid,
        t_end=req.t_end,
        p_mean=p_mean,
        p_std=p_std,
        alert=alert,
        tau=TAU,
        sigma_max=SIGMA_MAX,
    )
