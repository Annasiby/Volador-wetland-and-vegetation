# GIS Engine Frontend --- UI Design & Implementation Plan

## 1. Purpose

This document defines the frontend UI design and the implementation
sequence for the GIS Wetland Degradation & Depletion Engine.

The frontend is a web application built first for desktop use. It is a
GIS analysis workstation, not a marketing dashboard.

The UI must expose the real engine workflow:

``` text
PROJECT
   ↓
DATASETS
   ↓
ANALYSIS CONFIGURATION
   ↓
RUN ENGINE
   ↓
MAP RESULTS
   ↓
METRICS
   ↓
QA / VALIDATION
   ↓
FINAL REPORT
```

The V1 engine is based on a two-date architecture:

``` text
T1 → T2
```

while the internal UI architecture should remain extensible to future
multi-date analysis.

------------------------------------------------------------------------

# 2. Frontend Technology Stack

## Core

  Technology        Purpose
  ----------------- ---------------------------
  React             UI framework
  TypeScript        Type safety
  Vite              Development/build tooling
  Tailwind CSS v4   Styling
  shadcn/ui         Reusable UI primitives
  Lucide React      Icons

## GIS / Visualization

  Technology       Purpose
  ---------------- --------------------------
  OpenLayers       Interactive GIS map
  GeoTIFF          Raster ingestion/display
  Apache ECharts   Charts and statistics

## Application State / Forms

  Technology        Purpose
  ----------------- -------------------------------
  TanStack Query    Server state and API requests
  React Hook Form   Forms
  Zod               Validation

------------------------------------------------------------------------

# 3. Design Philosophy

The interface should feel like a professional scientific GIS
workstation.

## Avoid

-   AI-dashboard visual language
-   Purple/blue gradients
-   Glassmorphism
-   Excessive cards
-   Large marketing hero sections
-   Fake metrics
-   Decorative charts without data
-   Unnecessary animation

## Prefer

-   Warm neutral background
-   White work surfaces
-   Dark charcoal typography
-   Restrained green accent
-   Thin borders
-   Clear hierarchy
-   Dense but readable information
-   GIS-first layout
-   Strong data provenance
-   Explicit validation states

------------------------------------------------------------------------

# 4. Visual System

## Colors

``` text
Background       #F7F7F5
Surface          #FFFFFF
Primary Text     #202020
Secondary Text   #6B6B67
Border           #E4E4DF

Primary Accent   #365C45
Warning          #A66A21
Danger           #9B3B32
Success          #3F6B4F
```

The accent should be used sparingly for active navigation, confirmed
states, primary actions, and important GIS selections.

------------------------------------------------------------------------

# 5. Application Shell

The application uses a persistent desktop workspace.

``` text
┌─────────────────────────────────────────────────────────────────────────┐
│ GIS ENGINE                         Project: Wetland Study    ● READY    │
├──────────────────┬──────────────────────────────────────────────────────┤
│                  │                                                      │
│ WORKSPACE        │  Current Workspace                                  │
│                  │                                                      │
│ Overview         │  ┌──────────────────────────────────────────────┐   │
│ Datasets         │  │                                              │   │
│ Analysis         │  │                  MAP VIEW                     │   │
│ Results          │  │                                              │   │
│ Validation       │  │                                              │   │
│ Report           │  │                                              │   │
│                  │  │                                              │   │
│                  │  └──────────────────────────────────────────────┘   │
│                  │                                                      │
│                  │  T1 Baseline                T2 Comparison           │
│                  │  ───────────                ───────────              │
│                  │                                                      │
│                  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│                  │  │Wetland │ │Affected│ │Change  │ │Quality │       │
│                  │  │Area    │ │Area    │ │Area    │ │Status  │       │
│                  │  │   —    │ │   —    │ │   —    │ │   —    │       │
│                  │  └────────┘ └────────┘ └────────┘ └────────┘       │
└──────────────────┴──────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# 6. Navigation

The sidebar is the primary workflow navigation.

``` text
WORKSPACE

Overview
Datasets
Analysis
Results
Validation
Report
```

## Navigation meaning

### Overview

Project status, current analysis state, high-level metrics, and map.

### Datasets

T1/T2 input management, metadata, validation, and dataset readiness.

### Analysis

Configuration of the analysis pipeline and execution.

### Results

Maps, transition statistics, CVA statistics, degradation/recovery
metrics, and area estimates.

### Validation

Input QA, spatial alignment, temporal QA, change QA, model validation,
and independent reference status.

### Report

Final report preview, evidence status, metadata, and PDF generation.

------------------------------------------------------------------------

# 7. Top Bar

The top bar contains:

``` text
GIS ENGINE
│
├── Current Project
├── Analysis Status
├── Run ID
└── Project Actions
```

Example:

``` text
GIS ENGINE      Wetland Study / 2026 Analysis

                                    ● READY
                                    Run: —
```

Possible analysis states:

``` text
READY
VALIDATING
RUNNING
COMPLETED
WARNING
FAILED
```

Do not use only color to communicate state. Always include text.

------------------------------------------------------------------------

# 8. Overview Screen

The Overview screen is the primary workspace.

## Sections

1.  Project header
2.  T1/T2 status
3.  Map viewport
4.  Key metrics
5.  Analysis status
6.  Evidence/quality summary

### Example

``` text
Wetland Change Analysis

T1
2025-08-20
Validated

T2
2026-08-20
Validated
```

Then:

``` text
┌──────────────────────────────────────────────────────────┐
│                                                          │
│                       MAP VIEW                           │
│                                                          │
│                No analysis loaded                        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Below:

``` text
Baseline Area      Affected Area      Change Area      QA
      —                  —                 —            —
```

No value is fabricated before the engine returns it.

------------------------------------------------------------------------

# 9. Dataset Screen

The dataset workflow is explicitly two-date.

``` text
DATASETS

T1 — Baseline
┌────────────────────────────────────────────┐
│ Dataset                                    │
│ Date: —                                    │
│ CRS: —                                     │
│ Resolution: —                              │
│ Bands: —                                   │
│ Validation: NOT RUN                        │
└────────────────────────────────────────────┘

T2 — Comparison
┌────────────────────────────────────────────┐
│ Dataset                                    │
│ Date: —                                    │
│ CRS: —                                     │
│ Resolution: —                              │
│ Bands: —                                   │
│ Validation: NOT RUN                        │
└────────────────────────────────────────────┘
```

## Required T1/T2 band inputs

``` text
green.tif
red.tif
rededge.tif
nir.tif
```

## Dataset validation

The UI should surface:

``` text
Files
Geometry
CRS
Resolution
Transform
NoData
Dimensions
Data validity
```

A dataset should have a clear state:

``` text
READY
WARNING
INVALID
NOT VALIDATED
```

------------------------------------------------------------------------

# 10. Dataset Metadata Panel

When a dataset is selected, show:

``` text
ACQUISITION
Date
Time / DOY if available

SPATIAL
CRS
Width
Height
Resolution
Extent
Transform

RASTER
NoData
Data type
Band availability

QUALITY
Input validation
```

Do not hide metadata that affects interpretation.

------------------------------------------------------------------------

# 11. Analysis Configuration Screen

The Analysis screen configures the actual engine.

``` text
ANALYSIS CONFIGURATION

Temporal
T1 → T2

Change Detection
☑ 4×4 Transition Analysis
☑ CVA

Wetland
☑ Wetland Mask

Metrics
☑ Degradation
☑ Recovery
☑ Area Statistics

Validation
☑ Input QA
☑ Temporal QA
☑ Change QA
```

Advanced parameters should be grouped separately.

------------------------------------------------------------------------

# 12. Configurable Parameters

These should be UI-editable but clearly marked as configurable rather
than scientifically locked constants.

``` text
CVA normalization
CVA threshold
CVA residual threshold
Severity weights
Minimum change area
Connected-component filtering
Phenological tolerance
Probability calibration
Degradation threshold
Recovery threshold
```

The UI should display:

``` text
Default
Current value
Units
Description
Source/config file
```

The frontend must never hard-code values that belong in engine
configuration.

------------------------------------------------------------------------

# 13. Locked Scientific Choices

The UI must not allow users to accidentally alter the core V1
architecture.

Locked V1 choices include:

``` text
Frozen U-Net / ResNet34 model
5-channel input
NDVI
GNDVI
NDRE
NIR
RedEdge

4 classes
Background
Severe
Moderate
Healthy

Two-date V1
T1 → T2

4×4 transition matrix
CVA
Wetland masking
```

If exposed in the UI, these should be displayed as:

``` text
LOCKED
```

rather than normal editable controls.

------------------------------------------------------------------------

# 14. Run Analysis Screen

The primary action:

``` text
RUN ANALYSIS
```

Before execution, display a confirmation summary:

``` text
PROJECT
Wetland Study

T1
2025-08-20

T2
2026-08-20

MODEL
U-Net / ResNet34
LOCKED

METHODS
4×4 Transition
CVA
Wetland Mask
Degradation
Recovery

VALIDATION
Input QA
Temporal QA
Change QA
```

Then:

``` text
[ RUN ANALYSIS ]
```

------------------------------------------------------------------------

# 15. Processing State

During execution, show actual engine stages.

``` text
ANALYSIS RUNNING

✓ Input validation
✓ Spatial alignment
✓ Preprocessing
✓ T1 inference
✓ T2 inference
→ Transition analysis
○ CVA
○ Wetland mask
○ Degradation
○ Recovery
○ Metrics
○ QA
○ Report preparation
```

The frontend should receive actual backend progress when available.

It must not simulate progress with a fake timer.

------------------------------------------------------------------------

# 16. Map Viewport

OpenLayers is the core map component.

The map should support:

``` text
Pan
Zoom
Fit extent
Layer visibility
Opacity
Legend
Identify / pixel inspection
Fullscreen
```

## Layer structure

``` text
BASE
├── Basemap / neutral background

INPUT
├── T1 raster
└── T2 raster

MODEL
├── T1 health
└── T2 health

CHANGE
├── Transition map
├── CVA magnitude
├── ΔNDVI
├── ΔGNDVI
└── ΔNDRE

ANALYSIS
├── Degradation
├── Recovery
├── Residual stress
└── Canopy loss

WETLAND
└── Wetland mask
```

Only layers actually produced by the engine should become available.

------------------------------------------------------------------------

# 17. Map Layer Panel

Example:

``` text
LAYERS

☑ Wetland Mask
☑ Degradation
☐ Recovery
☐ Transition
☐ CVA Magnitude
☐ T2 Health
☐ T1 Health
```

Each layer can expose:

``` text
Visibility
Opacity
Legend
```

The layer panel should distinguish:

``` text
OBSERVED
INFERRED
REFERENCE
```

where applicable.

------------------------------------------------------------------------

# 18. Map Legend

Legends must be data-specific.

For the four model classes:

``` text
HEALTH

Healthy
Moderate Stress
Severe Stress
Background
```

For transitions:

``` text
TRANSITION

H → H
H → M
H → S
H → B
M → H
M → M
M → S
M → B
S → H
S → M
S → S
S → B
B → H
B → M
B → S
B → B
```

Do not use ambiguous colors or unexplained categories.

------------------------------------------------------------------------

# 19. Results Screen

The Results screen converts raster outputs into interpretable
statistics.

## Main sections

``` text
1. Area
2. Transitions
3. Spectral Change
4. Degradation
5. Recovery
6. Wetland Extent
7. Data Quality
```

------------------------------------------------------------------------

# 20. Area Metrics

Show:

``` text
Baseline Area
Affected Area
Severity-Weighted Area
Normalized Degradation Ratio
Recovery Area
```

Example:

``` text
BASELINE AREA
—

AFFECTED AREA
—

SEVERITY-WEIGHTED AREA
—

NORMALIZED DEGRADATION
—
```

Units must be explicit.

------------------------------------------------------------------------

# 21. Transition Results

Use a table plus chart.

``` text
TRANSITION             AREA       %
Healthy → Moderate       —        —
Healthy → Severe         —        —
Moderate → Severe        —        —
Healthy → Background     —        —
Moderate → Healthy       —        —
Severe → Moderate        —        —
Severe → Healthy         —        —
```

Also expose the complete 4×4 matrix.

------------------------------------------------------------------------

# 22. Spectral Change

Show:

``` text
Mean ΔNDVI
Mean ΔGNDVI
Mean ΔNDRE

CVA Mean
CVA P95
```

Use ECharts for distribution/summary visualization when data are
available.

The UI must distinguish:

``` text
OBSERVED SPECTRAL CHANGE
```

from interpreted degradation.

------------------------------------------------------------------------

# 23. Degradation Results

Primary categories:

``` text
H → M
H → S
M → S
```

Canopy disappearance:

``` text
H → B
M → B
S → B
```

The UI must label vegetation-to-background results carefully.

Use:

> Vegetation-to-background transition

not automatically:

> Wetland depletion

------------------------------------------------------------------------

# 24. Recovery Results

Primary recovery transitions:

``` text
M → H
S → M
S → H
```

Display:

``` text
Recovery Area
Recovery Transitions
Recovery Percentage
```

Only show metrics returned by the engine.

------------------------------------------------------------------------

# 25. Wetland Extent Section

The UI must explicitly distinguish:

``` text
Wetland Vegetation Degradation
```

from:

``` text
Physical Wetland Depletion
```

If an independent wetland/hydrological layer exists:

``` text
WETLAND EXTENT

Status: AVAILABLE

Extent T1: —
Extent T2: —
Extent Change: —
Evidence Source: —
```

If it does not:

``` text
WETLAND EXTENT

Status: NOT AVAILABLE

Physical wetland depletion cannot be established
from vegetation evidence alone.
```

This distinction is a permanent product rule.

------------------------------------------------------------------------

# 26. Evidence Status

Every important result should be classified where applicable as:

``` text
OBSERVED
INFERRED
NOT AVAILABLE
NOT VALIDATED
```

Example:

``` text
Vegetation class transition
INFERRED

Spectral ΔNDVI
OBSERVED

Independent wetland extent
NOT AVAILABLE
```

This is preferable to presenting every output as equally certain.

------------------------------------------------------------------------

# 27. Validation Screen

The Validation screen is a first-class workflow step.

## Model QA

``` text
Input validity
Alignment
NoData
Inference
Class distribution
```

## Temporal QA

``` text
Date difference
Seasonal difference
Spatial overlap
Phenology warning
```

## Change QA

``` text
Tiny isolated patches
Edge artifacts
Unexpected transitions
CVA outliers
```

## Independent Validation

``` text
Reference data available
Reference samples
Olofsson estimation status
Independent wetland evidence
```

------------------------------------------------------------------------

# 28. Validation Status Design

Use explicit statuses:

``` text
PASS
WARNING
FAIL
NOT AVAILABLE
NOT VALIDATED
```

Example:

``` text
INPUT VALIDATION          PASS
SPATIAL ALIGNMENT        PASS
MODEL INFERENCE          PASS
PHENOLOGY                WARNING
INDEPENDENT VALIDATION   NOT AVAILABLE
```

Never hide warnings merely to make the dashboard look successful.

------------------------------------------------------------------------

# 29. Report Screen

The Report screen prepares the final publication-quality PDF.

Sections:

``` text
REPORT

Project Information
Analysis Period
Input Data
Methods
Results
Maps
Statistics
Validation
Evidence Status
Limitations
Metadata
```

Primary action:

``` text
GENERATE PDF REPORT
```

Secondary:

``` text
PREVIEW REPORT
```

------------------------------------------------------------------------

# 30. Report Readiness Gate

Before enabling final PDF generation, show:

``` text
REPORT READINESS

✓ Input validation
✓ T1/T2 analysis
✓ Transition analysis
✓ CVA
✓ Metrics
✓ QA

⚠ Independent validation unavailable
```

The report can still be generated when appropriate, but limitations must
be explicit.

Do not silently omit missing evidence.

------------------------------------------------------------------------

# 31. Final Report Content

The report UI should expose the same structure as the engine's final
report.

``` text
EXECUTIVE SUMMARY

DATA

T1
T2

CLASS AREAS

TRANSITIONS

SPECTRAL CHANGE

DEGRADATION

RECOVERY

WETLAND EXTENT

DATA QUALITY

VALIDATION

LIMITATIONS

METADATA
```

------------------------------------------------------------------------

# 32. Empty States

Empty states are important because the UI will often be used before an
analysis exists.

Good:

``` text
No analysis loaded

Run a T1/T2 analysis to populate
maps, metrics, and validation results.
```

Bad:

``` text
Affected Area: 43.2 ha
```

when the backend has not supplied a value.

------------------------------------------------------------------------

# 33. Error States

Errors should explain:

``` text
What failed
Why it failed
What can be done
```

Example:

``` text
SPATIAL ALIGNMENT FAILED

T1 resolution: 10 m
T2 resolution: 20 m

The datasets must be aligned to a
common analysis grid before comparison.

[Review Datasets]
```

Avoid generic:

``` text
Something went wrong.
```

------------------------------------------------------------------------

# 34. Responsive Strategy

V1 is desktop-first.

Target:

``` text
1440 × 900
1920 × 1080
```

The interface should still adapt to smaller desktop windows.

Mobile support is not a V1 priority.

GIS map interaction and analysis tables require desktop space.

------------------------------------------------------------------------

# 35. Component Architecture

Recommended structure:

``` text
src/
├── api/
│
├── app/
│
├── components/
│   ├── analysis/
│   ├── charts/
│   ├── layout/
│   │   ├── AppShell.tsx
│   │   ├── Sidebar.tsx
│   │   └── TopBar.tsx
│   │
│   ├── maps/
│   │   └── MapViewport.tsx
│   │
│   ├── qa/
│   ├── reports/
│   └── results/
│       └── MetricCard.tsx
│
├── features/
│   ├── analysis/
│   ├── dataset/
│   ├── project/
│   ├── report/
│   ├── results/
│   └── validation/
│
├── lib/
├── types/
├── App.tsx
├── App.css
├── index.css
└── main.tsx
```

------------------------------------------------------------------------

# 36. Implementation Sequence

Do not implement the entire frontend at once.

## Milestone A --- Application Shell

### Step A1

Create:

``` text
components/layout/AppShell.tsx
components/layout/Sidebar.tsx
components/layout/TopBar.tsx
```

### Step A2

Create:

``` text
components/results/MetricCard.tsx
components/maps/MapViewport.tsx
```

### Step A3

Replace the default Vite `App.tsx`.

### Step A4

Implement the visual system in:

``` text
index.css
App.css
```

### Step A5

Build and verify:

``` powershell
npm run build
```

### A Definition of Done

``` text
[ ] Sidebar works
[ ] Top bar works
[ ] Overview screen renders
[ ] Map viewport renders
[ ] Metric placeholders render
[ ] Navigation states work
[ ] Desktop layout works
[ ] npm run build passes
```

------------------------------------------------------------------------

# 37. Milestone B --- Dataset UI

Build:

``` text
T1 Dataset
T2 Dataset
Upload/select inputs
Metadata panel
Validation status
```

Definition of Done:

``` text
[ ] T1 can be selected
[ ] T2 can be selected
[ ] Band files visible
[ ] Metadata visible
[ ] Validation state visible
[ ] Invalid state is understandable
```

------------------------------------------------------------------------

# 38. Milestone C --- Analysis Configuration

Build:

``` text
Analysis configuration
Locked methods
Configurable parameters
Run confirmation
```

Definition of Done:

``` text
[ ] T1/T2 visible
[ ] Locked methods visible
[ ] Configurable values editable
[ ] Run summary correct
[ ] Run button connected to API layer
```

------------------------------------------------------------------------

# 39. Milestone D --- Real Map

Integrate:

``` text
OpenLayers
GeoTIFF
Engine raster outputs
```

First real layer:

``` text
T1 Health Map
```

Then:

``` text
T2 Health Map
Transition Map
CVA
Degradation
Recovery
Wetland Mask
```

Definition of Done:

``` text
[ ] GeoTIFF loads
[ ] CRS/extent correct
[ ] Layer visibility works
[ ] Opacity works
[ ] Legend works
[ ] Map fit works
```

------------------------------------------------------------------------

# 40. Milestone E --- Results

Connect:

``` text
statistics.json
transition_matrix.csv
report.json
```

Display:

``` text
Area metrics
Transition matrix
CVA statistics
Degradation
Recovery
Wetland extent
```

Definition of Done:

``` text
[ ] No fabricated values
[ ] Backend values displayed correctly
[ ] Units correct
[ ] Tables readable
[ ] Charts reflect actual data
```

------------------------------------------------------------------------

# 41. Milestone F --- Validation

Connect the engine QA output.

Display:

``` text
Input QA
Temporal QA
Change QA
Model QA
Independent validation
```

Definition of Done:

``` text
[ ] PASS/WARNING/FAIL states
[ ] Missing evidence visible
[ ] Warnings cannot be silently hidden
[ ] Validation details accessible
```

------------------------------------------------------------------------

# 42. Milestone G --- Report

Connect final report generation.

Workflow:

``` text
Analysis complete
      ↓
Validation complete
      ↓
Report readiness
      ↓
Preview
      ↓
Generate PDF
      ↓
Download / open generated report
```

Definition of Done:

``` text
[ ] Report readiness visible
[ ] Preview works
[ ] PDF generation works
[ ] Final report reflects actual engine results
[ ] Limitations are preserved
```

------------------------------------------------------------------------

# 43. Backend Integration Boundary

The frontend should communicate with the engine through an API layer.

Conceptually:

``` text
React UI
   ↓
API client
   ↓
Backend service
   ↓
GIS Engine
   ↓
Outputs
```

The frontend should not directly implement scientific calculations that
belong to the GIS engine.

The UI is responsible for:

``` text
Input
Configuration
Visualization
Interaction
Status
Results presentation
Report control
```

The engine is responsible for:

``` text
Raster processing
Model inference
Alignment
Transition analysis
CVA
Wetland analysis
Metrics
QA
Report data
```

------------------------------------------------------------------------

# 44. API-Oriented Frontend Features

Suggested API areas:

``` text
/projects
/datasets
/analysis
/runs
/results
/validation
/reports
```

Example conceptual flow:

``` text
POST /analysis/run

        ↓

run_id

        ↓

GET /runs/{run_id}

        ↓

status / progress

        ↓

GET /results/{run_id}

        ↓

maps + statistics + validation
```

Exact endpoint names should follow the backend contract rather than
being invented independently by the UI.

------------------------------------------------------------------------

# 45. Data Integrity Rules

The UI must never:

1.  Invent metrics.
2.  Infer wetland depletion from vegetation loss alone.
3.  Convert IoU into an area confidence interval.
4.  Hide unavailable independent validation.
5.  Pretend a warning is a pass.
6.  Change the frozen model configuration.
7.  silently substitute a different preprocessing pipeline.
8.  display a result before the backend has marked it available.

------------------------------------------------------------------------

# 46. UI Status Vocabulary

Use a consistent vocabulary across the application:

``` text
READY
RUNNING
COMPLETED
FAILED

PASS
WARNING
FAIL

AVAILABLE
NOT AVAILABLE
NOT VALIDATED

OBSERVED
INFERRED
REFERENCE
```

Avoid having multiple labels for the same state.

------------------------------------------------------------------------

# 47. Development Checkpoint Rules

After every milestone:

``` text
1. Run TypeScript/build
2. Run frontend
3. Manually inspect the UI
4. Verify no fake values
5. Verify navigation
6. Commit the milestone
```

Build command:

``` powershell
npm run build
```

Development server:

``` powershell
npm run dev
```

------------------------------------------------------------------------

# 48. First Build Target

The immediate target is **Milestone A**.

Create:

``` text
src/components/layout/AppShell.tsx
src/components/layout/Sidebar.tsx
src/components/layout/TopBar.tsx
src/components/results/MetricCard.tsx
src/components/maps/MapViewport.tsx
```

Then replace the temporary Vite screen with:

``` text
GIS ENGINE WORKSPACE
├── Sidebar
├── TopBar
├── Map Viewport
├── T1/T2 status
└── Metric strip
```

Do not connect the backend yet.

The first goal is a stable, professional UI shell.

------------------------------------------------------------------------

# 49. Definition of Frontend V1 Complete

The frontend is complete when a user can:

``` text
Create/select project
      ↓
Select T1/T2 datasets
      ↓
Review metadata
      ↓
Validate inputs
      ↓
Configure analysis
      ↓
Run engine
      ↓
Monitor execution
      ↓
View GIS maps
      ↓
Inspect transitions/CVA
      ↓
View degradation/recovery
      ↓
View wetland evidence
      ↓
Review QA
      ↓
Review report readiness
      ↓
Generate final PDF
```

The final UI should make the distinction between:

``` text
Vegetation Degradation
```

and:

``` text
Physical Wetland Depletion
```

impossible to miss.

------------------------------------------------------------------------

# 50. Immediate Next Action

Start with **Milestone A only**.

### Commands

``` powershell
New-Item -ItemType Directory -Force src\components\layout
New-Item -ItemType Directory -Force src\components\results
New-Item -ItemType Directory -Force src\components\maps

New-Item -ItemType File -Force src\components\layout\AppShell.tsx
New-Item -ItemType File -Force src\components\layout\Sidebar.tsx
New-Item -ItemType File -Force src\components\layout\TopBar.tsx
New-Item -ItemType File -Force src\components\results\MetricCard.tsx
New-Item -ItemType File -Force src\components\maps\MapViewport.tsx
```

Then implement the five components and update:

``` text
App.tsx
App.css
index.css
```

Finally:

``` powershell
npm run build
npm run dev
```

Do not start Dataset UI, API integration, or real GeoTIFF rendering
until Milestone A is visually and technically stable.
