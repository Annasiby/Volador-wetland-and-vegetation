export type AnalysisStatusState =
  | "READY"
  | "VALIDATING"
  | "RUNNING"
  | "COMPLETED"
  | "WARNING"
  | "FAILED";

type AnalysisStatusProps = {
  status: AnalysisStatusState;
};

const statusText: Record<AnalysisStatusState, string> = {
  READY: "Ready",
  VALIDATING: "Validating",
  RUNNING: "Running",
  COMPLETED: "Completed",
  WARNING: "Warning",
  FAILED: "Failed",
};

export default function AnalysisStatus({
  status,
}: AnalysisStatusProps) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="text-sm font-medium text-[var(--text)]">
          Processing status
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Analysis execution status reported by the engine.
        </div>
      </div>

      <div className="p-5">
        <div className="flex items-center gap-3">
          <span className="h-2 w-2 rounded-full bg-[var(--text-secondary)]" />

          <span className="text-sm font-medium text-[var(--text)]">
            {statusText[status]}
          </span>
        </div>

        {status === "READY" && (
          <div className="mt-3 text-xs text-[var(--text-secondary)]">
            Analysis has not been executed.
          </div>
        )}
      </div>
    </section>
  );
}