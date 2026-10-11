import type { BandName } from "@/types/dataset";

type BandListProps = {
  bands: Record<BandName, string | null>;
};

const bandLabels: Record<BandName, string> = {
  green: "Green",
  red: "Red",
  rededge: "RedEdge",
  nir: "NIR",
};

export default function BandList({ bands }: BandListProps) {
  return (
    <div className="space-y-2">
      {(Object.keys(bandLabels) as BandName[]).map((band) => (
        <div
          key={band}
          className="flex items-center justify-between border-b border-[var(--border)] py-2 last:border-b-0"
        >
          <span className="text-xs text-[var(--text-secondary)]">
            {bandLabels[band]}
          </span>

          <span
            className={[
              "max-w-[180px] truncate text-xs",
              bands[band]
                ? "text-[var(--text)]"
                : "text-[var(--text-secondary)]",
            ].join(" ")}
          >
            {bands[band] ?? "Not supplied"}
          </span>
        </div>
      ))}
    </div>
  );
}