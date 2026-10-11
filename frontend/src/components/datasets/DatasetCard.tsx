import type { Dataset } from "@/types/dataset";
import BandList from "./BandList";
import MetadataPanel from "./MetadataPanel";
import ValidationStatus from "./ValidationStatus";

type DatasetCardProps = {
  dataset: Dataset;
  onSelect: () => void;
};

export default function DatasetCard({
  dataset,
  onSelect,
}: DatasetCardProps) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="flex items-start justify-between border-b border-[var(--border)] px-5 py-4">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            {dataset.label}
          </div>

          <div className="mt-1 text-base font-medium text-[var(--text)]">
            {dataset.id ? dataset.id : "No dataset selected"}
          </div>
        </div>

        <ValidationStatus status={dataset.validation} />
      </div>

      <div className="space-y-6 p-5">
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Required Bands
          </div>

          <BandList bands={dataset.bands} />
        </div>

        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Metadata
          </div>

          <MetadataPanel metadata={dataset.metadata} />
        </div>

        <button
          type="button"
          onClick={onSelect}
          className="w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-4 py-2.5 text-xs font-medium text-[var(--text)] transition-colors hover:bg-[var(--background)]"
        >
          Select Dataset
        </button>
      </div>
    </section>
  );
}