export type ReportCheckState =
  | "PASS"
  | "WARNING"
  | "NOT_AVAILABLE"
  | "NOT_VALIDATED";

type ReportReadinessProps = {
  checks: {
    label: string;
    state: ReportCheckState;
  }[];
};

const stateConfig: Record<
  ReportCheckState,
  {
    label: string;
    className: string;
  }
> = {
  PASS: {
    label: "Ready",
    className:
      "border-[#CFE0D3] bg-[#F1F6F2] text-[var(--success)]",
  },

  WARNING: {
    label: "Warning",
    className:
      "border-[#E7D4B8] bg-[#FAF4EA] text-[var(--warning)]",
  },

  NOT_AVAILABLE: {
    label: "Not available",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text-secondary)]",
  },

  NOT_VALIDATED: {
    label: "Not validated",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text-secondary)]",
  },
};

export default function ReportReadiness({
  checks,
}: ReportReadinessProps) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="text-sm font-medium text-[var(--text)]">
          Report readiness
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Required analysis, validation, and evidence checks before
          generating the final report.
        </div>
      </div>

      <div className="divide-y divide-[var(--border)]">
        {checks.map((check) => {
          const config = stateConfig[check.state];

          return (
            <div
              key={check.label}
              className="flex items-center justify-between px-5 py-3.5"
            >
              <span className="text-xs text-[var(--text)]">
                {check.label}
              </span>

              <span
                className={[
                  "rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em]",
                  config.className,
                ].join(" ")}
              >
                {config.label}
              </span>
            </div>
          );
        })}
      </div>
    </section>
  );
}