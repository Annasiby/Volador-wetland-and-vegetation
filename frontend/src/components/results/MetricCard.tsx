type MetricCardProps = {
  label: string;
  value?: string;
  unit?: string;
  description?: string;
};

export default function MetricCard({
  label,
  value = "—",
  unit,
  description,
}: MetricCardProps) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
        {label}
      </div>

      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-2xl font-semibold tracking-tight text-[var(--text)]">
          {value}
        </span>

        {unit && (
          <span className="text-xs text-[var(--text-secondary)]">
            {unit}
          </span>
        )}
      </div>

      {description && (
        <div className="mt-2 text-xs leading-5 text-[var(--text-secondary)]">
          {description}
        </div>
      )}
    </div>
  );
}