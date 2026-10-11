export default function ReportPreview() {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-4">
        <div>
          <div className="text-sm font-medium text-[var(--text)]">
            Report preview
          </div>

          <div className="mt-1 text-xs text-[var(--text-secondary)]">
            Preview of the final report structure.
          </div>
        </div>

        <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
          Preview
        </div>
      </div>

      <div className="p-6">
        <div className="mx-auto max-w-3xl border border-[var(--border)] bg-white shadow-sm">
          <div className="border-b border-[var(--border)] px-10 py-8">
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">
              GIS ENGINE
            </div>

            <div className="mt-5 text-2xl font-semibold tracking-tight text-[var(--text)]">
              Wetland Change Analysis
            </div>

            <div className="mt-2 text-sm text-[var(--text-secondary)]">
              T1 → T2 comparative analysis
            </div>

            <div className="mt-6 grid grid-cols-2 gap-5 border-t border-[var(--border)] pt-5">
              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Baseline
                </div>

                <div className="mt-1 text-sm text-[var(--text)]">
                  T1 — Not available
                </div>
              </div>

              <div>
                <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Comparison
                </div>

                <div className="mt-1 text-sm text-[var(--text)]">
                  T2 — Not available
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-6 px-10 py-8">
            <div>
              <div className="text-sm font-semibold text-[var(--text)]">
                Executive Summary
              </div>

              <div className="mt-2 h-16 rounded bg-[var(--background)]" />
            </div>

            <div>
              <div className="text-sm font-semibold text-[var(--text)]">
                Analysis Results
              </div>

              <div className="mt-3 grid grid-cols-3 gap-3">
                <div className="h-16 rounded border border-[var(--border)]" />
                <div className="h-16 rounded border border-[var(--border)]" />
                <div className="h-16 rounded border border-[var(--border)]" />
              </div>
            </div>

            <div>
              <div className="text-sm font-semibold text-[var(--text)]">
                Validation
              </div>

              <div className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
                Validation results will be inserted from the completed
                engine run.
              </div>
            </div>

            <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
              <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                Limitation
              </div>

              <div className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
                Physical wetland depletion requires independent wetland
                spatial evidence and must not be inferred from vegetation
                change alone.
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}