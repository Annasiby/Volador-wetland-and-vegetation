import type { DatasetMetadata } from "@/types/dataset";

type MetadataPanelProps = {
  metadata: DatasetMetadata;
};

const metadataItems = [
  ["Date", "date"],
  ["CRS", "crs"],
  ["Dimensions", "dimensions"],
  ["Resolution", "resolution"],
  ["Transform", "transform"],
  ["NoData", "nodata"],
  ["Data type", "dtype"],
] as const;

export default function MetadataPanel({
  metadata,
}: MetadataPanelProps) {
  const values: Record<string, string | null> = {
    date: metadata.date,
    crs: metadata.crs,
    dimensions:
      metadata.width && metadata.height
        ? `${metadata.width} × ${metadata.height}`
        : null,
    resolution: metadata.resolution,
    transform: metadata.transform,
    nodata: metadata.nodata,
    dtype: metadata.dtype,
  };

  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-3">
      {metadataItems.map(([label, key]) => (
        <div key={key}>
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
            {label}
          </div>

          <div className="mt-1 truncate text-xs text-[var(--text)]">
            {values[key] ?? "—"}
          </div>
        </div>
      ))}
    </div>
  );
}