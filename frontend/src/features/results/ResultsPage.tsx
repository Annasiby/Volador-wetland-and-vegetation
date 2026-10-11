import ResultsSummary from "./ResultsSummary";
import TransitionMatrix from "./TransitionMatrix";
import ChangeStatistics from "./ChangeStatistics";
import { useProject } from "../../context/useProject";

export default function ResultsPage() {
  const { results, runStatus } = useProject();

  // ── Mode detection ────────────────────────────────────────────────────────
  const isVegetationMode = results?.analysis_mode === "vegetation";

  // ── Wetland / CVA metrics (only used in wetland mode) ────────────────────
  const cva = results?.cva || results?.metrics?.cva || {};
  const cvaMag = cva?.cva_magnitude || results?.metrics?.cva_magnitude || {};
  const hydro = results?.hydrology || results?.metrics?.water || {};

  const rawMeanMag = cva.mean_magnitude ?? cvaMag.mean;
  const meanMag =
    rawMeanMag !== undefined && rawMeanMag !== null
      ? typeof rawMeanMag === "number"
        ? rawMeanMag.toFixed(4)
        : String(rawMeanMag)
      : "—";

  const rawP95Mag = cva.p95_magnitude ?? cvaMag.p95;
  const p95Mag =
    rawP95Mag !== undefined && rawP95Mag !== null
      ? typeof rawP95Mag === "number"
        ? rawP95Mag.toFixed(4)
        : String(rawP95Mag)
      : "—";

  let waterNetStr = "—";
  if (typeof hydro.net_water_change_ha === "number") {
    waterNetStr = `${hydro.net_water_change_ha.toFixed(2)} ha`;
  } else if (
    hydro.water_gain_area_ha !== undefined ||
    hydro.water_loss_area_ha !== undefined
  ) {
    const gain =
      typeof hydro.water_gain_area_ha === "number" ? hydro.water_gain_area_ha : 0;
    const loss =
      typeof hydro.water_loss_area_ha === "number" ? hydro.water_loss_area_ha : 0;
    waterNetStr = `${(gain - loss).toFixed(2)} ha`;
  }

  const boundaryUsed = Boolean(
    results?.metrics?.wetland_boundary_used ||
      (results?.wetland?.status &&
        !["WETLAND_BOUNDARY_UNAVAILABLE", "UNAVAILABLE"].includes(
          results.wetland.status
        ))
  );

  // ── Vegetation metrics (only used in vegetation mode) ────────────────────
  const vegClasses =
    results?.vegetation?.classes ||
    results?.metrics?.class_distribution?.classes ||
    {};
  const healthyPct =
    typeof vegClasses["3"]?.percentage === "number"
      ? vegClasses["3"].percentage.toFixed(1)
      : "—";
  const moderatePct =
    typeof vegClasses["2"]?.percentage === "number"
      ? vegClasses["2"].percentage.toFixed(1)
      : "—";
  const severePct =
    typeof vegClasses["1"]?.percentage === "number"
      ? vegClasses["1"].percentage.toFixed(1)
      : "—";
  const backgroundPct =
    typeof vegClasses["0"]?.percentage === "number"
      ? vegClasses["0"].percentage.toFixed(1)
      : "—";

  const indices = results?.spectral_indices || {};
  const ndviMean =
    typeof indices.ndvi?.mean === "number"
      ? indices.ndvi.mean.toFixed(4)
      : "—";
  const gndviMean =
    typeof indices.gndvi?.mean === "number"
      ? indices.gndvi.mean.toFixed(4)
      : "—";
  const ndreMean =
    typeof indices.ndre?.mean === "number"
      ? indices.ndre.mean.toFixed(4)
      : "—";

  const meanConf =
    typeof results?.uncertainty?.mean_confidence === "number"
      ? `${(results.uncertainty.mean_confidence * 100).toFixed(1)}%`
      : typeof results?.metrics?.mean_confidence === "number"
      ? `${(results.metrics.mean_confidence * 100).toFixed(1)}%`
      : "—";

  const sceneDate = results?.temporal?.date || "—";

  return (
    <div className="space-y-5">
      {/* ── Header / top summary card ────────────────────────────────────── */}
      <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-[var(--text)]">
              Analysis Results Summary
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              {results
                ? `Run ID: ${results.run_id} · Mode: ${results.analysis_mode ?? "wetland"} · Status: ${results.status}`
                : "Results are currently awaiting engine output."}
            </div>
          </div>

          <div className="rounded-md border border-[var(--border)] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text)]">
            {runStatus?.status || "NOT PROCESSED"}
          </div>
        </div>

        {/* ── VEGETATION MODE metrics ───────────────────────────────────── */}
        {isVegetationMode ? (
          <div className="mt-5 space-y-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
              Vegetation health — {sceneDate}
            </div>

            {/* Health class distribution */}
            <div className="grid grid-cols-4 gap-4">
              <div className="rounded-md border border-emerald-500/20 bg-emerald-500/5 p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Healthy
                </div>
                <div className="mt-1 text-lg font-semibold text-emerald-400">
                  {healthyPct !== "—" ? `${healthyPct}%` : "—"}
                </div>
              </div>

              <div className="rounded-md border border-amber-500/20 bg-amber-500/5 p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Moderate Stress
                </div>
                <div className="mt-1 text-lg font-semibold text-amber-400">
                  {moderatePct !== "—" ? `${moderatePct}%` : "—"}
                </div>
              </div>

              <div className="rounded-md border border-red-500/20 bg-red-500/5 p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Severe Stress
                </div>
                <div className="mt-1 text-lg font-semibold text-red-400">
                  {severePct !== "—" ? `${severePct}%` : "—"}
                </div>
              </div>

              <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Background
                </div>
                <div className="mt-1 text-lg font-semibold text-[var(--text-secondary)]">
                  {backgroundPct !== "—" ? `${backgroundPct}%` : "—"}
                </div>
              </div>
            </div>

            {/* Spectral indices + confidence */}
            <div className="grid grid-cols-4 gap-4">
              <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  NDVI Mean
                </div>
                <div className="mt-1 text-sm font-medium text-[var(--text)]">
                  {ndviMean}
                </div>
              </div>

              <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  GNDVI Mean
                </div>
                <div className="mt-1 text-sm font-medium text-[var(--text)]">
                  {gndviMean}
                </div>
              </div>

              <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  NDRE Mean
                </div>
                <div className="mt-1 text-sm font-medium text-[var(--text)]">
                  {ndreMean}
                </div>
              </div>

              <div className="rounded-md border border-[var(--border)] bg-[var(--background)] p-4">
                <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                  Mean Confidence
                </div>
                <div className="mt-1 text-sm font-medium text-[var(--text)]">
                  {meanConf}
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* ── WETLAND MODE metrics ──────────────────────────────────────── */
          <div className="mt-5 grid grid-cols-3 gap-5">
            <div>
              <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                CVA Mean Magnitude
              </div>
              <div className="mt-1 text-sm text-[var(--text)] font-medium">
                {meanMag}
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                CVA P95 Threshold
              </div>
              <div className="mt-1 text-sm text-[var(--text)] font-medium">
                {p95Mag}
              </div>
            </div>

            <div>
              <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
                Surface Water Net Change
              </div>
              <div className="mt-1 text-sm text-[var(--text)] font-medium">
                {waterNetStr}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* ── WETLAND-ONLY sub-components ─────────────────────────────────────── */}
      {!isVegetationMode && (
        <>
          <ResultsSummary />

          <TransitionMatrix />

          <ChangeStatistics />

          <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
            <div className="text-sm font-medium text-[var(--text)]">
              Wetland Extent &amp; Hydrological Evidence Status
            </div>

            <div className="mt-1 text-xs text-[var(--text-secondary)]">
              Physical wetland depletion is claimed only when independent
              wetland spatial evidence is available.
            </div>

            <div className="mt-4 rounded-md border border-[var(--border)] bg-[var(--background)] p-4 text-xs text-[var(--text-secondary)]">
              {boundaryUsed
                ? "✓ Independent wetland spatial boundary active — restricted metrics evaluated."
                : "WETLAND_BOUNDARY_UNAVAILABLE — Metrics evaluated over full aligned scene bounding box."}
            </div>
          </section>
        </>
      )}

      {/* ── VEGETATION-ONLY: analysis scope notice ───────────────────────────── */}
      {isVegetationMode && results && (
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
          <div className="text-sm font-medium text-[var(--text)]">
            Single-Date Analysis Scope
          </div>
          <div className="mt-2 text-xs text-[var(--text-secondary)]">
            Vegetation mode performs single-date health classification (NDVI /
            GNDVI / NDRE) only. No temporal comparison, CVA, degradation,
            recovery, NDWI, or wetland depletion assessment is performed.
          </div>
        </section>
      )}
    </div>
  );
}