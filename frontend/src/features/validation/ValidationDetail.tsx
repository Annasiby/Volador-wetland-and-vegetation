type ValidationDetailProps = {
  label: string;
  value: string;
  description?: string;
};

export default function ValidationDetail({
  label,
  value,
  description,
}: ValidationDetailProps) {
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-4">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
        {label}
      </div>

      <div className="mt-2 text-sm font-medium text-[var(--text)]">
        {value}
      </div>

      {description && (
        <div className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
          {description}
        </div>
      )}
    </div>
  );
}