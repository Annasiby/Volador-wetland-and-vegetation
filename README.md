# GIS Wetland Degradation & Depletion Engine

Integrated geospatial analysis application for Sentinel-2 temporal vegetation degradation, 4x4 transition analysis, Change Vector Analysis (CVA), NDWI surface-water monitoring, and cartographic PDF report generation.

---

## 1. System Architecture

```
React 19 Frontend (Vite + OpenLayers + ECharts)
       │
       ▼  HTTP / REST API (JSON + Tile/Raster Stream)
FastAPI Application Layer (backend/app)
       │
       ├── Project Service & Dataset Manager (Local FS Storage)
       ├── Async Background Run Manager (Worker Thread Pool)
       └── Tile & Layer Renderer (Rasterio / GeoTIFF Server)
       │
       ▼
GIS Core Engine (backend/Engine_module/src/gis_engine)
       ├── Ingestion & Input Validation
       ├── Spatial & Temporal Alignment
       ├── Preprocessing & Index Computation (NDVI, GNDVI, NDRE, NDWI)
       ├── ONNX Inference (5-Channel U-Net / ResNet34)
       ├── Change Analysis (4×4 Transition, CVA, Degradation, Recovery)
       ├── Hydrological Evidence & Wetland Masking
       ├── Metrics & Area Calculations
       ├── QA Validation (Input QA, Temporal QA, Model QA)
       └── PDF Report Generator (ReportLab / Matplotlib Cartography)
```

---

## 2. Quick Start

### Step 1: Generate Sample Dataset (Optional Out-of-the-Box Setup)
```bash
python backend/scripts/generate_sample_scene.py
python backend/scripts/create_onnx_model.py
```

### Step 2: Start the Backend FastAPI Server
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
The API server starts at `http://127.0.0.1:8000` (Swagger docs available at `http://127.0.0.1:8000/docs`).

### Step 3: Start the React Frontend Application
```bash
cd frontend
npm install
npm run dev
```
The web app opens at `http://localhost:5173`.

---

## 3. Running Automated Tests

### 1. Scientific GIS Engine Unit Tests (265 Tests)
```bash
cd backend/Engine_module
python -m pytest
```

### 2. FastAPI Application Server Integration Tests
```bash
python -m pytest backend/tests/test_fastapi_server.py
```

### 3. End-to-End Workflow & Scientific Regression Test Suite
```bash
python -m pytest backend/tests/test_e2e_full_integration.py
```

---

## 4. Scientific Principles

- **5-Channel Model Input**: Channels `0=NDVI`, `1=GNDVI`, `2=NDRE`, `3=NIR`, `4=RedEdge`.
- **4 Vegetation Classes**: `0=Background`, `1=Severe Stress`, `2=Moderate Stress`, `3=Healthy Vegetation`.
- **Authoritative Change Analysis**: 4×4 Transition Matrix, Change Vector Analysis (CVA), Vegetation Degradation/Recovery, and NDWI Surface Water extent.
- **Physical Wetland Depletion**: Claimed ONLY when independent temporal wetland spatial evidence is accepted via configuration.
