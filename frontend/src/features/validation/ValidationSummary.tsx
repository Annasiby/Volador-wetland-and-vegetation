import type { ValidationState } from "./ValidationSection";

type ValidationSummaryProps = {
  overallState: ValidationState;
};

const summaryConfig: Record<
  ValidationState,
  {
    label: string;
    description: string;
    className: string;
  }
> = {
  PASS: {
    label: "Validation passed",
    description: "All required validation checks passed.",
    className:
      "border-[#CFE0D3] bg-[#F1F6F2] text-[var(--success)]",
  },

  WARNING: {
    label: "Validation warning",
    description:
      "The analysis contains one or more conditions that require review.",
    className:
      "border-[#E7D4B8] bg-[#FAF4EA] text-[var(--warning)]",
  },

  FAIL: {
    label: "Validation failed",
    description:
      "One or more required validation checks failed.",
    className:
      "border-[#E4C2BE] bg-[#FAEFED] text-[var(--danger)]",
  },

  NOT_AVAILABLE: {
    label: "Validation unavailable",
    description:
      "Required validation evidence is not available.",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text-secondary)]",
  },

  NOT_VALIDATED: {
    label: "Not validated",
    description:
      "The analysis has not completed the validation workflow.",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text-secondary)]",
  },
};

export default function ValidationSummary({
  overallState,
}: ValidationSummaryProps) {
  const config = summaryConfig[overallState];

  return (
    <section
      className={[
        "rounded-lg border p-5",
        config.className,
      ].join(" ")}
    >
      <div className="text-sm font-medium">
        {config.label}
      </div>

      <div className="mt-1 text-xs leading-5 opacity-80">
        {config.description}
      </div>
    </section>
  );
}