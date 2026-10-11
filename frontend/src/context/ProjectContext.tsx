import React, { useState, useEffect, useCallback } from "react";
import type {
  Project,
  ValidationResult,
  RunStatus,
  MapLayer,
  AnalysisResults,
  PreparationState,
} from "../api/client";
import {
  getProjects,
  createProject,
  validateDatasets,
  uploadBandFile,
  launchRun,
  getRunStatus,
  getRunResults,
  getRunLayers,
  resetProject,
  getPreparation,
  savePreparationDraft,
  commitPreparation,
  unlockPreparation,
} from "../api/client";
import { ProjectContext } from "./projectContextObject";

export type ManualExclusion = {
  id: string;
  geometry: {
    type: "Polygon" | "MultiPolygon";
    coordinates: number[][][] | number[][][][];
  };
  category: string;
  comment: string;
};

export const ProjectProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [project, setProject] = useState<Project | null>(null);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [layers, setLayers] = useState<MapLayer[]>([]);
  const [activeLayerId, setActiveLayerId] = useState<string>("t2_health");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [manualExclusions, setManualExclusions] = useState<ManualExclusion[]>(
    []
  );
  const [preparation, setPreparation] = useState<PreparationState | null>(null);

  // Stable function: load run results and layers.
  // activeLayerId is intentionally excluded from deps to avoid a load loop.
  const loadRunData = useCallback(async (runId: string) => {
    try {
      const resData = await getRunResults(runId);
      setResults(resData);

      const layerList = await getRunLayers(runId);
      setLayers(layerList);

      setActiveLayerId((prev) => {
        if (layerList.length > 0 && !prev) {
          return layerList[0].id;
        }
        return prev;
      });
    } catch (err: unknown) {
      console.error("Failed to load run outputs:", err);
    }
  }, []);

  // Initialize or fetch default project on load.
  useEffect(() => {
    async function init() {
      try {
        setLoading(true);
        setError(null);

        const existing = await getProjects();

        let currentProj: Project;

        if (existing.length > 0) {
          currentProj = existing[0];
        } else {
          currentProj = await createProject(
            "Wetland Degradation Study",
            "Sentinel-2 5-channel temporal degradation analysis"
          );
        }

        setProject(currentProj);

        // Restore an existing run only if it still exists.
        if (currentProj.latest_run_id) {
          try {
            const existingRun = await getRunStatus(currentProj.latest_run_id);

            setRunStatus(existingRun);

            if (existingRun.status === "COMPLETED") {
              await loadRunData(existingRun.run_id);
            }
          } catch (err) {
            console.warn(
              `Ignoring stale latest_run_id ${currentProj.latest_run_id}:`,
              err
            );
          }
        }

        // Fetch current dataset validation status.
        try {
          const initialMode = (currentProj.analysis_mode as "vegetation" | "wetland") ?? "wetland";
          const val = await validateDatasets(currentProj.id, initialMode);
          setValidation(val);
          if (val.valid && (!currentProj.datasets_ready || currentProj.status === "INVALID")) {
            setProject((prev) =>
              prev
                ? {
                    ...prev,
                    datasets_ready: true,
                    status:
                      prev.preparation_status === "COMMITTED"
                        ? "PREPARATION_COMMITTED"
                        : "DATASETS_CONFIGURED",
                  }
                : null
            );
          }
        } catch {
          // Non-blocking initial validation check.
        }
      } catch (err: unknown) {
        const message =
          err instanceof Error ? err.message : "Unknown error";

        console.error("Failed to connect to backend API:", message);
        setError(`Backend connection error: ${message}`);
      } finally {
        setLoading(false);
      }
    }

    init();
  }, [loadRunData]);

  // Poll run status when active run is running.
  useEffect(() => {
    if (!runStatus || ["COMPLETED", "FAILED"].includes(runStatus.status)) {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const updated = await getRunStatus(runStatus.run_id);

        setRunStatus(updated);

        if (updated.status === "COMPLETED") {
          loadRunData(updated.run_id);
        }
      } catch (err) {
        console.error("Status poll error:", err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [runStatus, loadRunData]);

  const triggerValidation = async (mode?: "vegetation" | "wetland") => {
    if (!project) return;

    try {
      setLoading(true);
      setError(null);

      const resolvedMode = mode ?? (project.analysis_mode as "vegetation" | "wetland") ?? "wetland";
      const val = await validateDatasets(project.id, resolvedMode);

      setValidation(val);

      setProject((prev) =>
        prev
          ? {
              ...prev,
              analysis_mode: resolvedMode,
              datasets_ready: val.valid,
              status: val.valid ? "READY" : "INVALID",
            }
          : null
      );
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Validation failed"
      );
    } finally {
      setLoading(false);
    }
  };

  const uploadBand = async (
    timePoint: "t1" | "t2",
    band: "green" | "red" | "rededge" | "nir" | "boundary",
    file: File,
    mode?: "vegetation" | "wetland"
  ) => {
    if (!project) return;

    try {
      setLoading(true);
      setError(null);

      await uploadBandFile(project.id, timePoint, band, file);
      await triggerValidation(mode);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  // ------------------------------------------------------------
  // Manual exclusion management
  // ------------------------------------------------------------

  const addManualExclusion = (exclusion: ManualExclusion) => {
    setManualExclusions((current) => [...current, exclusion]);
  };

  const updateManualExclusion = (
    updatedExclusion: ManualExclusion
  ) => {
    setManualExclusions((current) =>
      current.map((exclusion) =>
        exclusion.id === updatedExclusion.id
          ? updatedExclusion
          : exclusion
      )
    );
  };

  const removeManualExclusion = (id: string) => {
    setManualExclusions((current) =>
      current.filter((exclusion) => exclusion.id !== id)
    );
  };

  const clearManualExclusions = () => {
    setManualExclusions([]);
  };

  // ------------------------------------------------------------
  // Preparation / staging workflow
  // ------------------------------------------------------------

  const fetchPreparation = useCallback(
    async (projectId?: string): Promise<PreparationState | undefined> => {
      const pid = projectId ?? project?.id;
      if (!pid) return undefined;
      try {
        const state = await getPreparation(pid);
        setPreparation(state);
        return state;
      } catch (err) {
        console.warn("fetchPreparation failed:", err);
        return undefined;
      }
    },
    [project?.id]
  );

  const saveDraft = useCallback(
    async (
      inclusion: Record<string, any> | null,
      exclusions: ManualExclusion[]
    ) => {
      if (!project) return;
      try {
        const state = await savePreparationDraft(
          project.id,
          inclusion,
          exclusions
        );
        setPreparation(state);
        setManualExclusions(exclusions);
      } catch (err: unknown) {
        setError(
          err instanceof Error ? err.message : "Failed to save draft"
        );
      }
    },
    [project]
  );

  const commitArea = useCallback(async () => {
    if (!project) return;
    try {
      const state = await commitPreparation(project.id);
      setPreparation(state);
      setProject((prev) =>
        prev ? { ...prev, preparation_status: "COMMITTED", is_locked: true } : null
      );
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to commit preparation"
      );
    }
  }, [project]);

  const unlockArea = useCallback(async () => {
    if (!project) return;
    try {
      const state = await unlockPreparation(project.id);
      setPreparation(state);
      setProject((prev) =>
        prev ? { ...prev, preparation_status: "DRAFT", is_locked: false } : null
      );
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : "Failed to unlock preparation"
      );
    }
  }, [project]);
  // ------------------------------------------------------------
  // Clear all analysis data and reset the current project session
  // ------------------------------------------------------------

  const clearAll = async () => {
    if (!project) return;

    try {
      setLoading(true);
      setError(null);

      // Reset the backend workspace while preserving the project.
      const reset = await resetProject(project.id);

      // Clear all frontend analysis state.
      setProject(reset);
      setValidation(null);
      setRunStatus(null);
      setResults(null);
      setLayers([]);
      setManualExclusions([]);
      setActiveLayerId("t2_health");
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to clear project session"
      );
    } finally {
      setLoading(false);
    }
  };
  // ------------------------------------------------------------
  // Launch analysis
  // ------------------------------------------------------------

  const triggerRun = async (config?: Record<string, any>) => {
    if (!project) return;

    try {
      setLoading(true);
      setError(null);

      // Clear results from any previous run.
      setResults(null);
      setLayers([]);
      setActiveLayerId("");

      // Preserve the existing analysis configuration and append
      // the manually excluded polygons and inclusion geometry.
      const runConfig = {
        ...(config || {}),
        manual_exclusions: manualExclusions,
        inclusion_geometry: preparation?.inclusion ?? null,
      };

      console.log(
        "[GIS DEBUG] manualExclusions before launch:",
        manualExclusions
      );

      console.log(
        "[GIS DEBUG] runConfig before launch:",
        runConfig
      );

      const initialRun = await launchRun(project.id, runConfig);

      // The backend-generated run_id is authoritative.
      setRunStatus(initialRun);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Run failed");
    } finally {
      setLoading(false);
    }
  };

  const refreshResults = async () => {
    if (runStatus?.run_id) {
      await loadRunData(runStatus.run_id);
    }
  };

  return (
    <ProjectContext.Provider
      value={{
        project,
        validation,
        runStatus,
        results,
        layers,
        activeLayerId,
        loading,
        error,

        manualExclusions,
        preparation,

        setActiveLayerId,
        triggerValidation,
        uploadBand,
        triggerRun,
        refreshResults,
        updateManualExclusion,
        addManualExclusion,
        removeManualExclusion,
        clearManualExclusions,
        clearAll,
        fetchPreparation,
        saveDraft,
        commitArea,
        unlockArea,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
};