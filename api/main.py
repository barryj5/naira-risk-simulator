import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from forecast import arima_forecast

app = FastAPI(title="Naira Risk Simulator")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast")
def forecast(days: int = 7):
    days = max(1, min(days, 30))
    return arima_forecast(days)