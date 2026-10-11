import type { DatasetValidationState } from "@/types/dataset";

type ValidationStatusProps = {
  status: DatasetValidationState;
};

const statusConfig: Record<
  DatasetValidationState,
  {
    label: string;
    className: string;
  }
> = {
  NOT_VALIDATED: {
    label: "Not validated",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text-secondary)]",
  },
  VALIDATING: {
    label: "Validating",
    className:
      "border-[var(--border)] bg-[var(--background)] text-[var(--text)]",
  },
  READY: {
    label: "Ready",
    className:
      "border-[#CFE0D3] bg-[#F1F6F2] text-[var(--success)]",
  },
  WARNING: {
    label: "Warning",
    className:
      "border-[#E7D4B8] bg-[#FAF4EA] text-[var(--warning)]",
  },
  INVALID: {
    label: "Invalid",
    className:
      "border-[#E4C2BE] bg-[#FAEFED] text-[var(--danger)]",
  },
};

export default function ValidationStatus({
  status,
}: ValidationStatusProps) {
  const config = statusConfig[status];

  return (
    <span
      className={[
        "inline-flex items-center rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em]",
        config.className,
      ].join(" ")}
    >
      {config.label}
    </span>
  );
}