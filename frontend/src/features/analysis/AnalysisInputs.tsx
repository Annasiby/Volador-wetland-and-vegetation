type AnalysisMode = "vegetation" | "wetland";

type AnalysisInputsProps = {
  t1Date: string | null;
  t2Date: string | null;
  analysisMode: AnalysisMode;
};

export default function AnalysisInputs({
  t1Date,
  t2Date,
  analysisMode,
}: AnalysisInputsProps) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="text-sm font-medium text-[var(--text)]">
          Analysis inputs
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          {analysisMode === "vegetation"
            ? "Single-date analysis — only one scene required."
            : "Two-date analysis — T1 baseline and T2 comparison required."}
        </div>
      </div>

      {analysisMode === "vegetation" ? (
        /* Vegetation mode: single date */
        <div className="p-5">
          <div className="rounded-md border border-[var(--accent)]/30 bg-[var(--accent)]/5 p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--accent)]">
              Vegetation Scene
            </div>

            <div className="mt-2 text-sm font-medium text-[var(--text)]">
              {t1Date ?? "No date set"}
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Single acquisition date — NDVI, GNDVI, NDRE classification
            </div>
          </div>
        </div>
      ) : (
        /* Wetland mode: two dates */
        <div className="grid grid-cols-2 gap-4 p-5">
          <div className="rounded-md border border-[var(--border)] p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
              T1 — Baseline
            </div>

            <div className="mt-2 text-sm font-medium text-[var(--text)]">
              {t1Date ?? "Not loaded"}
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Baseline observation
            </div>
          </div>

          <div className="rounded-md border border-[var(--border)] p-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
              T2 — Comparison
            </div>

            <div className="mt-2 text-sm font-medium text-[var(--text)]">
              {t2Date ?? "Not loaded"}
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Comparison observation
            </div>
          </div>
        </div>
      )}
    </section>
  );
}