import { useProject } from "../../context/useProject";

export default function ChangeStatistics() {
  const { results } = useProject();

  const cva = results?.cva || results?.metrics?.cva || {};
  const deg = results?.degradation || {};
  const rec = results?.recovery || {};
  const cats = results?.change?.category_summary || {};
  const wetland = results?.wetland || {};

  const fmtNum = (val?: any) => (typeof val === "number" ? val.toFixed(4) : "—");
  const fmtHa = (val?: any) => (typeof val === "number" ? `${val.toFixed(2)} ha` : "—");

  const canopyLossHa = cats["VEGETATION_TO_BACKGROUND"]?.area_ha;

  const deltaNdvi = cva.mean_delta_ndvi ?? cva.delta_ndvi?.mean;
  const deltaGndvi = cva.mean_delta_gndvi ?? cva.delta_gndvi?.mean;
  const deltaNdre = cva.mean_delta_ndre ?? cva.delta_ndre?.mean;
  const cvaMag = cva.cva_magnitude || results?.metrics?.cva_magnitude || {};
  const cvaMean = cva.mean_magnitude ?? cvaMag.mean;
  const cvaP95 = cva.p95_magnitude ?? cvaMag.p95;

  const spectralRows = [
    ["Mean ΔNDVI", fmtNum(deltaNdvi)],
    ["Mean ΔGNDVI", fmtNum(deltaGndvi)],
    ["Mean ΔNDRE", fmtNum(deltaNdre)],
    ["CVA Mean", fmtNum(cvaMean)],
    ["CVA P95", fmtNum(cvaP95)],
  ];

  const interpretationRows = [
    ["Degradation", fmtHa(deg.area_ha)],
    ["Recovery", fmtHa(rec.area_ha)],
    ["Canopy loss", fmtHa(canopyLossHa)],
    ["Physical Depletion Status", results?.physical_depletion_assessment || "NOT_ASSESSED"],
    ["Wetland Extent Area", fmtHa(wetland.area_ha)],
  ];

  return (
    <div className="grid grid-cols-2 gap-5">
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">
          Observed spectral change
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Continuous change derived from the multispectral indices and CVA.
        </div>

        <div className="mt-5 space-y-4">
          {spectralRows.map(([label, value]) => (
            <div
              key={label}
              className="flex items-center justify-between border-b border-[var(--border)] pb-3"
            >
              <span className="text-xs text-[var(--text-secondary)]">
                {label}
              </span>

              <span className="text-sm font-medium text-[var(--text)]">
                {value}
              </span>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="text-sm font-medium text-[var(--text)]">
          Change interpretation
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Transition-derived degradation and recovery indicators.
        </div>

        <div className="mt-5 space-y-4">
          {interpretationRows.map(([label, value]) => (
            <div
              key={label}
              className="flex items-center justify-between border-b border-[var(--border)] pb-3"
            >
              <span className="text-xs text-[var(--text-secondary)]">
                {label}
              </span>

              <span className="text-sm font-medium text-[var(--text)]">
                {value}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}