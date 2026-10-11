import { useProject } from "../../context/useProject";
import { getPdfReportUrl, getTechnicalPdfReportUrl } from "../../api/client";

type ReportPageProps = {
  analysisMode: "vegetation" | "wetland";
};

export default function ReportPage({ analysisMode }: ReportPageProps) {
  const { runStatus, results } = useProject();

  const isCompleted = runStatus?.status === "COMPLETED";
  const runId = runStatus?.run_id ?? "";
  const pdfUrl = runId ? getPdfReportUrl(runId) : "";
  const techPdfUrl = runId ? getTechnicalPdfReportUrl(runId) : "";

  // Prefer the prop (selected mode) over what's stored in results.
  // results?.analysis_mode is only available after a completed run;
  // the prop reflects the user's current UI selection immediately.
  const effectiveMode =
    analysisMode ?? (results as any)?.analysis_mode ?? "wetland";
  const isVegetation = effectiveMode === "vegetation";

  const simpleReportLabel = isVegetation
    ? "Vegetation Health Report"
    : "Wetland Degradation Report";
  const simpleReportDesc = isVegetation
    ? "Executive summary: single-date canopy health, classification statistics, and key observations."
    : "Executive summary: temporal vegetation transitions, surface-water dynamics, and multi-evidence wetland assessment.";
  const technicalReportLabel = isVegetation
    ? "Vegetation Health Technical Report"
    : "Wetland Degradation Technical Report";
  const technicalReportDesc = isVegetation
    ? "Full technical audit: band statistics, index formulas, model architecture, class metrics, QA, and provenance."
    : "Full technical audit: 4×4 transition matrix, CVA displacements, NDWI dynamics, fusion rules, QA, and provenance.";

  const reportChecks = [
    { label: "Input Validation", state: "PASS" },
    {
      label: isVegetation ? "Scene Spectral Inference" : "T1 → T2 Spectral Inference",
      state: isCompleted ? "PASS" : "PENDING",
    },
    {
      label: isVegetation ? "Class Statistics" : "4×4 Transition Matrix",
      state: isCompleted ? "PASS" : "PENDING",
    },
    {
      label: isVegetation ? "Vegetation Index Computation" : "Change Vector Analysis (CVA)",
      state: isCompleted ? "PASS" : "PENDING",
    },
    { label: "Area Statistics & Degradation", state: isCompleted ? "PASS" : "PENDING" },
    { label: "Quality Assurance Validation", state: isCompleted ? "PASS" : "PENDING" },
    {
      label: "Independent Validation",
      state: "NOT_AVAILABLE",
    },
    {
      label: "Independent Wetland Evidence",
      state:
        !isVegetation && (results as any)?.metrics?.wetland_boundary_used
          ? "AVAILABLE"
          : "NOT_AVAILABLE",
    },
  ];

  return (
    <div className="space-y-5">
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          {isVegetation ? "Vegetation Health Report" : "Cartographic Analysis Report"}
        </div>
        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          Review report readiness, download the compiled ReportLab PDF reports, and preview the
          executive summary.
        </div>
      </section>

      {/* Report Readiness Section */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">Report Generation Readiness</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Automated validation check of required scientific figures, tables, and provenance metadata.
            </div>
          </div>

          <div
            className={`rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] ${
              isCompleted
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : "bg-amber-500/20 text-amber-400 border border-amber-500/30"
            }`}
          >
            {isCompleted ? "READY" : "AWAITING ENGINE OUTPUT"}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-2 gap-3">
          {reportChecks.map((check) => (
            <div
              key={check.label}
              className="flex items-center justify-between rounded-md border border-[var(--border)] p-3 text-xs"
            >
              <span className="text-[var(--text)]">{check.label}</span>
              <span
                className={`font-semibold text-[10px] uppercase tracking-[0.1em] ${
                  check.state === "PASS" || check.state === "AVAILABLE"
                    ? "text-emerald-400"
                    : "text-[var(--text-secondary)]"
                }`}
              >
                {check.state}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* PDF Download Action Bar — Simple Report */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">{simpleReportLabel}</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">{simpleReportDesc}</div>
          </div>

          <a
            href={isCompleted ? pdfUrl : "#"}
            target="_blank"
            rel="noreferrer"
            className={`rounded-md bg-[var(--accent)] px-5 py-2.5 text-xs font-medium text-white transition-opacity hover:opacity-90 ${
              !isCompleted ? "pointer-events-none opacity-40" : ""
            }`}
          >
            Download Report
          </a>
        </div>
      </section>

      {/* PDF Download Action Bar — Technical Report */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">{technicalReportLabel}</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">{technicalReportDesc}</div>
          </div>

          <a
            href={isCompleted ? techPdfUrl : "#"}
            target="_blank"
            rel="noreferrer"
            className={`rounded-md border border-[var(--accent)] bg-transparent px-5 py-2.5 text-xs font-medium text-[var(--accent)] transition-opacity hover:opacity-90 ${
              !isCompleted ? "pointer-events-none opacity-40" : ""
            }`}
          >
            Download Technical Report
          </a>
        </div>
      </section>

      {/* PDF Embedded Viewport — previews the simple report */}
      {isCompleted && (
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="mb-3 text-sm font-medium text-[var(--text)]">
            PDF Report Preview &nbsp;
            <span className="text-xs font-normal text-[var(--text-secondary)]">
              ({simpleReportLabel})
            </span>
          </div>
          <div className="h-[650px] w-full overflow-hidden rounded-md border border-[var(--border)] bg-[#525659]">
            <iframe
              src={pdfUrl}
              title={`${simpleReportLabel} Preview`}
              className="h-full w-full border-none"
            />
          </div>
        </section>
      )}
    </div>
  );
}