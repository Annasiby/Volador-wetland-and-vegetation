import { useProject } from "../../context/useProject";

const classes = ["Background", "Severe", "Moderate", "Healthy"];

export default function TransitionMatrix() {
  const { results } = useProject();

  const transitionData =
    results?.metrics?.transition_matrix || (results as any)?.transition_matrix;
  const matrixValues = transitionData?.matrix_m2 || transitionData?.matrix_pixels;

  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5">
      <div>
        <div className="text-sm font-medium text-[var(--text)]">
          Authoritative 4×4 Vegetation Health Transition Matrix
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          T1 (Baseline) class → T2 (Comparison) class transition area in hectares (ha).
        </div>
      </div>

      <div className="mt-5 overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="border-b border-[var(--border)] px-4 py-3 text-left font-medium text-[var(--text-secondary)]">
                T1 Baseline ↓ / T2 Target →
              </th>

              {classes.map((className) => (
                <th
                  key={className}
                  className="border-b border-[var(--border)] px-4 py-3 text-right font-medium text-[var(--text-secondary)]"
                >
                  {className}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {classes.map((rowClass, rowIndex) => (
              <tr key={rowClass}>
                <td className="border-b border-[var(--border)] px-4 py-3 font-medium text-[var(--text)]">
                  {rowClass}
                </td>

                {classes.map((colClass, colIndex) => {
                  let valStr = "—";
                  if (matrixValues && Array.isArray(matrixValues)) {
                    const rawVal = matrixValues[rowIndex]?.[colIndex];
                    if (rawVal !== undefined && rawVal !== null) {
                      valStr = (rawVal / 10000.0).toFixed(2) + " ha";
                    }
                  }
                  return (
                    <td
                      key={`${rowClass}-${colClass}`}
                      className={`border-b border-[var(--border)] px-4 py-3 text-right ${
                        rowIndex !== colIndex && valStr !== "—"
                          ? "font-semibold text-[var(--accent)]"
                          : "text-[var(--text-secondary)]"
                      }`}
                    >
                      {valStr}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 text-[11px] text-[var(--text-secondary)]">
        Diagonal cells indicate stable area; off-diagonal cells represent vegetation health transitions.
      </div>
    </section>
  );
}