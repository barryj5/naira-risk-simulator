import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from forecast import arima_forecast, simulate_paths

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


@app.get("/simulate")
def run_simulation(method: str = "bootstrap", days: int = 10):
    if method not in ("bootstrap", "gan"):
        raise HTTPException(status_code=400, detail="method must be bootstrap or gan")
    try:
        return simulate_paths(method, days=days)
    except FileNotFoundError:
        raise HTTPException(status_code=503,
                            detail="GAN model not trained yet. Run: python src/gan.py")