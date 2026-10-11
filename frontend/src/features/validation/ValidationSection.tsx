export type ValidationState =
  | "PASS"
  | "WARNING"
  | "FAIL"
  | "NOT_AVAILABLE"
  | "NOT_VALIDATED";

type ValidationSectionProps = {
  title: string;
  description: string;
  state: ValidationState;
  detail?: string;
};

const stateConfig: Record<
  ValidationState,
  {
    label: string;
    className: string;
  }
> = {
  PASS: {
    label: "Pass",
    className:
      "border-[#CFE0D3] bg-[#F1F6F2] text-[var(--success)]",
  },

  WARNING: {
    label: "Warning",
    className:
      "border-[#E7D4B8] bg-[#FAF4EA] text-[var(--warning)]",
  },

  FAIL: {
    label: "Fail",
    className:
      "border-[#E4C2BE] bg-[#FAEFED] text-[var(--danger)]",
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

export default function ValidationSection({
  title,
  description,
  state,
  detail,
}: ValidationSectionProps) {
  const config = stateConfig[state];

  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="flex items-start justify-between gap-6">
        <div className="min-w-0">
          <div className="text-sm font-medium text-[var(--text)]">
            {title}
          </div>

          <div className="mt-1 max-w-2xl text-xs leading-5 text-[var(--text-secondary)]">
            {description}
          </div>

          {detail && (
            <div className="mt-3 rounded-md bg-[var(--background)] p-3 text-xs leading-5 text-[var(--text-secondary)]">
              {detail}
            </div>
          )}
        </div>

        <span
          className={[
            "shrink-0 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em]",
            config.className,
          ].join(" ")}
        >
          {config.label}
        </span>
      </div>
    </div>
  );
}