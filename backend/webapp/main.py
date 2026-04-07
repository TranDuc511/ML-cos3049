# FastAPI app entry point
import sys
import os
from pathlib import Path

# Add the backend directory to sys.path so 'ai' module is found
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import predic, history, stats

app = FastAPI(title="Transaction Anomaly Detection API")

# Enable CORS for React frontend (running on default port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predic.router)
app.include_router(history.router)
app.include_router(stats.router)