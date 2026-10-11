/**
 * frontend/src/api/client.ts
 * TypeScript API client for communicating with the FastAPI backend server.
 */

const API_BASE = "http://127.0.0.1:8000/api";

export interface Project {
  id: string;
  name: string;
  description?: string;
  aoi_name?: string;
  created_at: string;
  t1_date?: string;
  t2_date?: string;
  status: string;
  datasets_ready: boolean;
  latest_run_id?: string;
  preparation_status?: "NOT_PREPARED" | "DRAFT" | "COMMITTED" | string;
  is_locked?: boolean;
  analysis_mode?: "vegetation" | "wetland" | string;
}

export interface ManualExclusion {
  id: string;
  geometry: Record<string, any>;
  category: string;
  comment: string;
}

export interface PreparationMetrics {
  source_pixels: number;
  inclusion_pixels: number;
  excluded_pixels: number;
  effective_pixels: number;
  source_ha?: number | null;
  inclusion_ha?: number | null;
  excluded_ha?: number | null;
  effective_ha?: number | null;
  area_basis: "PIXEL_COUNT_ONLY" | "RASTER_CRS_AND_TRANSFORM" | string;
}

export interface PreparationState {
  project_id: string;
  status: "NOT_PREPARED" | "DRAFT" | "COMMITTED";
  is_locked: boolean;
  committed_at?: string | null;
  updated_at?: string | null;
  inclusion?: Record<string, any> | null;
  manual_exclusions: ManualExclusion[];
  metrics: PreparationMetrics;
}

export interface PreviewInfo {
  width: number;
  height: number;
  extent: [number, number, number, number];
  crs: string | null;
  is_pixel_space: boolean;
}

export interface DatasetMetadata {
  date: string | null;
  crs: string | null;
  width: number | null;
  height: number | null;
  resolution: string | null;
  transform: string | null;
  nodata: number | null;
  dtype: string | null;
}

export interface ValidationResult {
  status: "PASS" | "WARNING" | "ERROR" | "NOT_VALIDATED";
  valid: boolean;
  messages: string[];
  metadata?: DatasetMetadata;
  dates_required?: boolean;
}

export interface RunStatus {
  run_id: string;
  project_id: string;
  status:
    | "QUEUED"
    | "VALIDATING"
    | "ALIGNING"
    | "PREPROCESSING"
    | "INFERENCE"
    | "CHANGE_ANALYSIS"
    | "HYDROLOGY"
    | "METRICS"
    | "VALIDATION"
    | "REPORTING"
    | "COMPLETED"
    | "FAILED";
  progress: number;
  current_step: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  message?: string;
  error?: string;
}

export interface MapLayer {
  id: string;
  name: string;
  type: string;
  url: string;
  format: string;
  bounds?: [number, number, number, number];
  crs: string;
  opacity: number;
  ready: boolean;
}
/** Area metrics produced by the GIS fusion engine. */
export interface AreaMetrics {
  total_area_ha?: number;
  vegetation_degradation_area_ha?: number;
  vegetation_recovery_area_ha?: number;
  water_loss_area_ha?: number;
  water_gain_area_ha?: number;
  hydro_ecological_degradation_area_ha?: number;
  hydro_ecological_recovery_area_ha?: number;
  // UI-consumed aliases (may or may not be present in a given engine version):
  degraded_area_ha?: number;
  total_change_area_ha?: number;
  [key: string]: number | string | boolean | null | undefined;
}

/** Degradation sub-metrics consumed by the UI. */
export interface DegradationMetrics {
  degraded_area_ha?: number;
  total_change_area_ha?: number;
  severity_weighted_degradation_ha?: number;
  normalized_degradation_ratio?: number;
  recovered_area_ha?: number;
  [key: string]: number | string | boolean | null | undefined;
}

/** CVA (Change Vector Analysis) statistics. */
export interface CvaMetrics {
  mean_magnitude?: number;
  p95_magnitude?: number;
  [key: string]: number | string | boolean | null | undefined;
}

/** Water change statistics. */
export interface WaterMetrics {
  net_water_change_ha?: number;
  [key: string]: number | string | boolean | null | undefined;
}

/** Transition matrix output. */
export interface TransitionMatrixMetrics {
  matrix_m2?: number[][];
  matrix_pixels?: number[][];
  [key: string]: number[][] | number | string | boolean | null | undefined;
}

/** QA flags within the validation block. */
export interface QaFlags {
  alignment?: string;
  temporal?: string;
  inference?: string;
  [key: string]: string | undefined;
}

export interface AnalysisResults {
  run_id: string;
  status: string;
  analysis_mode?: "vegetation" | "wetland" | string;
  temporal?: {
    t1_date?: string;
    t2_date?: string;
    interval_days?: number;
    date?: string;
    [key: string]: any;
  };
  vegetation?: {
    total_valid_pixels?: number;
    classes?: Record<string, { name?: string; count?: number; percentage?: number }>;
    [key: string]: any;
  };
  spectral_indices?: {
    ndvi?: { mean?: number; median?: number; std?: number; p95?: number; [key: string]: any };
    gndvi?: { mean?: number; median?: number; std?: number; p95?: number; [key: string]: any };
    ndre?: { mean?: number; median?: number; std?: number; p95?: number; [key: string]: any };
    [key: string]: any;
  };
  uncertainty?: {
    mean_confidence?: number;
    mean_confidence_t1?: number;
    [key: string]: any;
  };
  cva?: {
    mean_magnitude?: number;
    median_magnitude?: number;
    p95_magnitude?: number;
    mean_delta_ndvi?: number;
    mean_delta_gndvi?: number;
    mean_delta_ndre?: number;
    [key: string]: any;
  };
  hydrology?: {
    ndwi_threshold?: number;
    water_gain_area_ha?: number;
    water_loss_area_ha?: number;
    [key: string]: any;
  };
  degradation?: {
    pixel_count?: number;
    area_ha?: number;
    percentage?: number;
    [key: string]: any;
  };
  recovery?: {
    pixel_count?: number;
    area_ha?: number;
    percentage?: number;
    [key: string]: any;
  };
  change?: {
    category_summary?: Record<
      string,
      { count?: number; area_ha?: number; percentage?: number }
    >;
    [key: string]: any;
  };
  wetland?: {
    status?: string;
    source_type?: string;
    area_ha?: number;
    physical_depletion_enabled?: boolean;
    [key: string]: any;
  };
  fusion?: {
    evidence_level?: string;
    hydro_ecological_degradation_area_ha?: number;
    hydro_ecological_recovery_area_ha?: number;
    [key: string]: any;
  };
  physical_depletion_assessment?: string;
  spatial?: {
    crs?: string;
    dimensions?: [number, number];
    resolution?: [number, number];
    overlap_percentage?: number;
    [key: string]: any;
  };
  metrics: {
    area?: AreaMetrics;
    degradation?: DegradationMetrics;
    transition_matrix?: TransitionMatrixMetrics;
    water?: WaterMetrics;
    cva?: CvaMetrics;
    wetland_boundary_used?: boolean;
    normalized_degradation_ratio?: number;
    severity_weighted_degradation_area_ha?: number;
    [key: string]: any;
  };
  validation: {
    overall_status?: string;
    status?: string;
    qa_flags?: QaFlags;
    warnings?: string[];
    olofsson_status?: string;
    hydro_status?: string;
    [key: string]: any;
  };
  warnings: string[];
  errors: string[];
  metadata: Record<string, any>;
}
async function handleResponseError(res: Response, defaultMessage: string): Promise<never> {
  let detailMessage = defaultMessage;
  try {
    const data = await res.json();
    if (data?.detail) {
      detailMessage = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } else if (data?.error?.message) {
      detailMessage = data.error.message;
    }
  } catch {
    // Keep default message if response body is not JSON
  }
  throw new Error(detailMessage);
}

export async function fetchHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function createProject(name: string, description?: string): Promise<Project> {
  const res = await fetch(`${API_BASE}/projects`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, description, aoi_name: "Central Reserve" }),
  });
  if (!res.ok) await handleResponseError(res, "Failed to create project");
  return res.json();
}

export async function getProjects(): Promise<Project[]> {
  const res = await fetch(`${API_BASE}/projects`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch projects");
  return res.json();
}
export async function resetProject(projectId: string): Promise<Project> {
  const res = await fetch(
    `${API_BASE}/projects/${projectId}/reset`,
    {
      method: "POST",
    }
  );

  if (!res.ok) {
    await handleResponseError(
      res,
      "Failed to reset project session"
    );
  }

  return res.json();
}
export async function validateDatasets(
  projectId: string,
  analysisMode?: "vegetation" | "wetland"
): Promise<ValidationResult> {
  const url = analysisMode
    ? `${API_BASE}/projects/${projectId}/datasets/validate?analysis_mode=${analysisMode}`
    : `${API_BASE}/projects/${projectId}/datasets/validate`;
  const res = await fetch(url, {
    method: "POST",
  });
  if (!res.ok) await handleResponseError(res, "Failed to validate datasets");
  return res.json();
}

export async function uploadBandFile(
  projectId: string,
  timePoint: "t1" | "t2",
  band: "green" | "red" | "rededge" | "nir" | "boundary",
  file: File
): Promise<{ message: string; path: string }> {
  const formData = new FormData();
  formData.append("time_point", timePoint);
  formData.append("band", band);
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) await handleResponseError(res, `Failed to upload ${band} band`);
  return res.json();
}

export async function updateProjectDates(
  projectId: string,
  t1Date: string,
  t2Date: string,
) {
  const response = await fetch(
    `${API_BASE}/projects/${projectId}/dates`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        t1_date: t1Date,
        t2_date: t2Date,
      }),
    },
  );

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Failed to update observation dates");
  }

  return response.json();
}

export async function launchRun(projectId: string, config?: Record<string, any>): Promise<RunStatus> {
  const res = await fetch(`${API_BASE}/runs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_id: projectId, config }),
  });
  if (!res.ok) await handleResponseError(res, "Failed to launch analysis run");
  return res.json();
}

export async function getRunStatus(runId: string): Promise<RunStatus> {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch run status");
  return res.json();
}

export async function getRunResults(runId: string): Promise<AnalysisResults> {
  const res = await fetch(`${API_BASE}/runs/${runId}/results`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch analysis results");
  return res.json();
}

export async function getRunLayers(runId: string): Promise<MapLayer[]> {
  const res = await fetch(`${API_BASE}/runs/${runId}/layers`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch layer catalog");
  return res.json();
}

export function getLayerImageUrl(runId: string, layerId: string): string {
  return `${API_BASE}/runs/${runId}/layers/${layerId}/image`;
}

export function getPdfReportUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/report/pdf`;
}

export function getTechnicalPdfReportUrl(runId: string): string {
  return `${API_BASE}/runs/${runId}/report/pdf/technical`;
}

// ------------------------------------------------------------
// Preparation / Staging API
// ------------------------------------------------------------

export async function getPreparation(projectId: string): Promise<PreparationState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/preparation`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch preparation state");
  return res.json();
}

export async function savePreparationDraft(
  projectId: string,
  inclusion: Record<string, any> | null,
  manualExclusions: ManualExclusion[]
): Promise<PreparationState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/preparation`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      inclusion,
      manual_exclusions: manualExclusions,
    }),
  });
  if (!res.ok) await handleResponseError(res, "Failed to save preparation draft");
  return res.json();
}

export async function commitPreparation(projectId: string): Promise<PreparationState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/preparation/commit`, {
    method: "POST",
  });
  if (!res.ok) await handleResponseError(res, "Failed to commit and lock preparation");
  return res.json();
}

export async function unlockPreparation(projectId: string): Promise<PreparationState> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/preparation/unlock`, {
    method: "POST",
  });
  if (!res.ok) await handleResponseError(res, "Failed to unlock preparation");
  return res.json();
}

export async function getDatasetPreviewInfo(projectId: string): Promise<PreviewInfo> {
  const res = await fetch(`${API_BASE}/projects/${projectId}/datasets/preview-info`);
  if (!res.ok) await handleResponseError(res, "Failed to fetch dataset preview info");
  return res.json();
}

export function getDatasetPreviewImageUrl(projectId: string, timePoint: string = "t1"): string {
  return `${API_BASE}/projects/${projectId}/datasets/preview-image?time_point=${timePoint}`;
}

