"""
backend/app/main.py
===================
FastAPI Application Entry Point for the GIS Wetland Engine backend.
"""

import sys
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure package imports work regardless of working directory
BACKEND_DIR = Path(__file__).resolve().parent.parent
ENGINE_SRC = BACKEND_DIR / "Engine_module" / "src"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ENGINE_SRC) not in sys.path:
    sys.path.insert(0, str(ENGINE_SRC))

from app.routers import projects, datasets, runs, results, layers, reports
from app.schemas import APIErrorResponse, ErrorDetail

app = FastAPI(
    title="GIS Wetland Degradation & Depletion Engine API",
    description="Authoritative API backing the GIS Wetland degradation assessment pipeline.",
    version="1.0.0",
)

# Enable CORS for React frontend (Vite default dev server port 5173 / localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(projects.router)
app.include_router(datasets.router)
app.include_router(runs.router)
app.include_router(results.router)
app.include_router(layers.router)
app.include_router(reports.router)


@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "HEALTHY",
        "engine": "GIS Wetland Engine V1",
        "channels": ["NDVI", "GNDVI", "NDRE", "NIR", "RedEdge"],
        "classes": ["Background", "Severe Stress", "Moderate Stress", "Healthy Vegetation"]
    }


# Standardized error handler according to prompt specification
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content=APIErrorResponse(
            error=ErrorDetail(
                code="INTERNAL_SERVER_ERROR",
                message=str(exc),
                details={"path": str(request.url)}
            )
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
