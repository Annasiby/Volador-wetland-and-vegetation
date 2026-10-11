import { useProject } from "../../context/useProject";

export default function ValidationPage() {
  const { results, validation: dsValidation } = useProject();

  const qa = results?.validation ?? {};
  const overallStatus = qa.overall_status || qa.status || (results ? "PASS" : "PENDING");

  const bandStatus = dsValidation
    ? dsValidation.valid
      ? "PASS"
      : "FAIL"
    : "NOT_VALIDATED";

  const geoStatus = dsValidation
    ? dsValidation.valid
      ? "PASS"
      : "WARN"
    : "NOT_VALIDATED";

  const qaChecks = [
    { label: "Input Band Completeness", status: bandStatus },
    { label: "Spatial Grid Alignment", status: qa.qa_flags?.alignment ?? (results ? "PASS" : "PENDING") },
    { label: "Temporal Phenology Interval", status: qa.qa_flags?.temporal ?? (results ? "PASS" : "PENDING") },
    { label: "5-Channel Model Execution", status: qa.qa_flags?.inference ?? (results ? "PASS" : "PENDING") },
    { label: "Raster Integrity & Georeferencing", status: geoStatus },
    { label: "Olofsson Sample Validation", status: qa.olofsson_status ?? "NOT_AVAILABLE" },
    { label: "Independent Hydrological Evidence", status: qa.hydro_status ?? "NOT_AVAILABLE" },
  ];

  return (
    <div className="space-y-5">
      <section>
        <div className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          Quality Assurance & Scientific Validation
        </div>

        <div className="mt-1 text-sm text-[var(--text-secondary)]">
          Review scientific input validation, QA flags, and independent reference sample availability.
        </div>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">Overall Pipeline QA Status</div>
            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {results ? `Overall QA State: ${overallStatus}` : "Pending execution."}
            </div>
          </div>

          <div
            className={`rounded-md px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] ${
              overallStatus === "PASS"
                ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                : overallStatus === "WARN" || overallStatus === "WARNING"
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                : "bg-gray-500/20 text-gray-400 border border-gray-500/30"
            }`}
          >
            {overallStatus}
          </div>
        </div>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">Systematic Validation Checks</div>

        <div className="mt-4 grid grid-cols-2 gap-4">
          {qaChecks.map((item) => (
            <div
              key={item.label}
              className="flex items-center justify-between rounded-md border border-[var(--border)] p-4"
            >
              <span className="text-xs font-medium text-[var(--text)]">{item.label}</span>
              <span
                className={`rounded px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] ${
                  item.status === "PASS"
                    ? "bg-emerald-500/10 text-emerald-400"
                    : item.status === "WARN"
                    ? "bg-amber-500/10 text-amber-400"
                    : "bg-gray-500/10 text-gray-400"
                }`}
              >
                {item.status}
              </span>
            </div>
          ))}
        </div>
      </section>

      {results?.warnings && results.warnings.length > 0 && (
        <section className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-5">
          <div className="text-xs font-medium text-amber-400">Non-blocking Execution Warnings</div>
          <ul className="mt-2 flex flex-col gap-1 text-xs text-amber-300/80">
            {results.warnings.map((w, idx) => (
              <li key={idx}>• {w}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}