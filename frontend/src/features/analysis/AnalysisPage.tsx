import { useState } from "react";
import AnalysisConfig from "./AnalysisConfig";
import AnalysisInputs from "./AnalysisInputs";
import { useProject } from "../../context/useProject";

type AnalysisMode = "vegetation" | "wetland";

type AnalysisPageProps = {
  analysisMode: AnalysisMode;
  onAnalysisModeChange: (mode: AnalysisMode) => void;
};

export default function AnalysisPage({
  analysisMode,
  onAnalysisModeChange,
}: AnalysisPageProps) {
  const { project, validation, runStatus, triggerRun } = useProject();

  const [config, setConfig] = useState({
    observationDatesEnabled: true,
    transitionEnabled: true,
    cvaEnabled: true,
    wetlandMaskEnabled: true,
    degradationEnabled: true,
    recoveryEnabled: true,
    areaStatisticsEnabled: true,
    inputQaEnabled: true,
    temporalQaEnabled: true,
    changeQaEnabled: true,
  });

  const toggleConfig = <K extends keyof typeof config>(key: K) => {
    setConfig((current) => ({
      ...current,
      [key]:
        typeof current[key] === "boolean"
          ? !current[key]
          : current[key],
    }));
  };

  const handleChange = (
    key:
      | "analysisMode"
      | "observationDatesEnabled"
      | "transitionEnabled"
      | "cvaEnabled"
      | "wetlandMaskEnabled"
      | "degradationEnabled"
      | "recoveryEnabled"
      | "areaStatisticsEnabled"
      | "inputQaEnabled"
      | "temporalQaEnabled"
      | "changeQaEnabled",
    value?: any
  ) => {
    if (key === "analysisMode") {
      const targetMode: AnalysisMode =
        value === "vegetation" || value === "wetland"
          ? value
          : analysisMode === "vegetation"
          ? "wetland"
          : "vegetation";
      onAnalysisModeChange(targetMode);
    } else {
      toggleConfig(key);
    }
  };

  const isRunning =
    runStatus && !["COMPLETED", "FAILED"].includes(runStatus.status);

  const isDatasetValid = Boolean(
    project?.datasets_ready || validation?.valid
  );

  const fullConfig = { ...config, analysisMode, analysis_mode: analysisMode };

  return (
    <div className="space-y-5">
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Analysis Pipeline Execution
        </div>

        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          {analysisMode === "vegetation"
            ? "Single-date vegetation health analysis — NDVI, GNDVI, NDRE classification."
            : "Two-date wetland degradation & depletion analysis — temporal change, CVA, hydrology, fusion."}
        </div>
      </section>

      <AnalysisInputs
        t1Date={project?.t1_date ?? null}
        t2Date={project?.t2_date ?? null}
        analysisMode={analysisMode}
      />

      <AnalysisConfig
        analysisMode={analysisMode}
        observationDatesEnabled={config.observationDatesEnabled}
        transitionEnabled={config.transitionEnabled}
        cvaEnabled={config.cvaEnabled}
        wetlandMaskEnabled={config.wetlandMaskEnabled}
        degradationEnabled={config.degradationEnabled}
        recoveryEnabled={config.recoveryEnabled}
        areaStatisticsEnabled={config.areaStatisticsEnabled}
        inputQaEnabled={config.inputQaEnabled}
        temporalQaEnabled={config.temporalQaEnabled}
        changeQaEnabled={config.changeQaEnabled}
        onModeChange={onAnalysisModeChange}
        onChange={handleChange}
      />

      {/* Real Step-by-Step Run Status */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Pipeline Execution Progress
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {runStatus
                ? runStatus.current_step
                : "Ready to launch backend execution."}
            </div>
          </div>

          <div className="rounded-md border border-[var(--border)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.12em] text-[var(--text)]">
            {runStatus?.status || "IDLE"}
          </div>
        </div>

        {/* Progress Bar */}
        {runStatus && (
          <div className="mt-4">
            <div className="mb-1 flex justify-between text-xs text-[var(--text-secondary)]">
              <span>{runStatus.current_step}</span>
              <span>{runStatus.progress}%</span>
            </div>

            <div className="h-2.5 w-full overflow-hidden rounded-full bg-[var(--border)]">
              <div
                className="h-full bg-[var(--accent)] transition-all duration-300"
                style={{ width: `${runStatus.progress}%` }}
              />
            </div>
          </div>
        )}

        {runStatus?.error && (
          <div className="mt-3 rounded-md border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-400">
            Execution Failure: {runStatus.error}
          </div>
        )}
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Run Pipeline
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {!isDatasetValid && !isRunning
                ? "Dataset validation required before launching analysis run."
                : analysisMode === "vegetation"
                ? "Launch single-date vegetation health inference (Green, Red, RedEdge, NIR → NDVI/GNDVI/NDRE)."
                : "Launch background execution of Sentinel-2 inference, 4×4 transitions, CVA, NDWI, and PDF cartography."}
            </div>
          </div>

          <button
            type="button"
            disabled={Boolean(isRunning) || !isDatasetValid}
            onClick={() => triggerRun(fullConfig)}
            className="rounded-md bg-[var(--accent)] px-5 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isRunning
              ? `Executing (${runStatus.progress}%)...`
              : !isDatasetValid
              ? "Validation Required"
              : "Run Analysis"}
          </button>
        </div>
      </section>
    </div>
  );
}
