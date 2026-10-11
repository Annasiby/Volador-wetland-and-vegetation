type AnalysisMode = "vegetation" | "wetland";

type AnalysisConfigProps = {
  analysisMode: AnalysisMode;
  observationDatesEnabled: boolean;
  transitionEnabled: boolean;
  cvaEnabled: boolean;
  wetlandMaskEnabled: boolean;
  degradationEnabled: boolean;
  recoveryEnabled: boolean;
  areaStatisticsEnabled: boolean;
  inputQaEnabled: boolean;
  temporalQaEnabled: boolean;
  changeQaEnabled: boolean;
  onModeChange?: (mode: AnalysisMode) => void;
  onChange: (
    key:
      | "analysisMode"
      | "observationDatesEnabled"
      | "transitionEnabled"
      | "cvaEnabled"
      | "wetlandMaskEnabled"
      | "degradationEnabled"
      | "recoveryEnabled"
      | "areaStatisticsEnabled"
      | "inputQaEnabled"
      | "temporalQaEnabled"
      | "changeQaEnabled",
    value?: any
  ) => void;
};

type LockedItemProps = {
  label: string;
};

function LockedItem({ label }: LockedItemProps) {
  return (
    <div className="flex items-center justify-between rounded-md border border-[var(--border)] px-4 py-3">
      <span className="text-xs text-[var(--text)]">{label}</span>

      <span className="rounded-full border border-[var(--border)] px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
        Locked
      </span>
    </div>
  );
}

function ConfigToggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-center justify-between rounded-md border border-[var(--border)] px-4 py-3">
      <span className="text-xs text-[var(--text)]">{label}</span>

      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="h-4 w-4 accent-[var(--accent)]"
      />
    </label>
  );
}

export default function AnalysisConfig({
  analysisMode,
  observationDatesEnabled,
  transitionEnabled,
  cvaEnabled,
  wetlandMaskEnabled,
  degradationEnabled,
  recoveryEnabled,
  areaStatisticsEnabled,
  inputQaEnabled,
  temporalQaEnabled,
  changeQaEnabled,
  onModeChange,
  onChange,
}: AnalysisConfigProps) {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="text-sm font-medium text-[var(--text)]">
          Analysis configuration
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Core V1 architecture is locked. Optional analysis outputs can be
          selected here.
        </div>
      </div>

      <div className="space-y-6 p-5">
        {/* Analysis mode */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Analysis mode
          </div>

          <div className="grid grid-cols-2 gap-2">
            <label
              className={`flex cursor-pointer items-center gap-3 rounded-md border px-4 py-3 transition-colors ${
                analysisMode === "vegetation"
                  ? "border-[var(--accent)] bg-[var(--accent)]/10"
                  : "border-[var(--border)]"
              }`}
            >
              <input
                type="radio"
                name="analysisMode"
                value="vegetation"
                checked={analysisMode === "vegetation"}
                onChange={() => {
                  if (onModeChange) {
                    onModeChange("vegetation");
                  } else {
                    onChange("analysisMode", "vegetation");
                  }
                }}
                className="h-4 w-4 accent-[var(--accent)]"
              />

              <div>
                <div className="text-xs font-medium text-[var(--text)]">
                  Vegetation
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-secondary)]">
                  Vegetation health and change
                </div>
              </div>
            </label>

            <label
              className={`flex cursor-pointer items-center gap-3 rounded-md border px-4 py-3 transition-colors ${
                analysisMode === "wetland"
                  ? "border-[var(--accent)] bg-[var(--accent)]/10"
                  : "border-[var(--border)]"
              }`}
            >
              <input
                type="radio"
                name="analysisMode"
                value="wetland"
                checked={analysisMode === "wetland"}
                onChange={() => {
                  if (onModeChange) {
                    onModeChange("wetland");
                  } else {
                    onChange("analysisMode", "wetland");
                  }
                }}
                className="h-4 w-4 accent-[var(--accent)]"
              />

              <div>
                <div className="text-xs font-medium text-[var(--text)]">
                  Wetland
                </div>
                <div className="mt-0.5 text-[10px] text-[var(--text-secondary)]">
                  Vegetation + wetland evidence
                </div>
              </div>
            </label>
          </div>
        </div>

        {/* Temporal */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Temporal
          </div>

          <div className="space-y-2">
            <ConfigToggle
              label="Use observation dates"
              checked={observationDatesEnabled}
              onChange={() => onChange("observationDatesEnabled")}
            />

            <LockedItem label="T1 ? T2" />
          </div>

          <div className="mt-2 text-[11px] text-[var(--text-secondary)]">
            {observationDatesEnabled
              ? "T1 and T2 observation dates are required for this run."
              : "Observation dates are optional. T1 ? T2 analysis will still run."}
          </div>
        </div>

        {/* Model */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Model
          </div>

          <div className="space-y-2">
            <LockedItem label="U-Net / ResNet34" />
            <LockedItem label="5-channel input" />
            <LockedItem label="NDVI" />
            <LockedItem label="GNDVI" />
            <LockedItem label="NDRE" />
            <LockedItem label="NIR" />
            <LockedItem label="RedEdge" />
          </div>
        </div>

        {/* Change detection */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Change detection
          </div>

          <div className="space-y-2">
            <ConfigToggle
              label="4×4 Transition Analysis"
              checked={transitionEnabled}
              onChange={() => onChange("transitionEnabled")}
            />

            <ConfigToggle
              label="CVA"
              checked={cvaEnabled}
              onChange={() => onChange("cvaEnabled")}
            />
          </div>
        </div>

        {/* Wetland */}
        {analysisMode === "wetland" && (
          <div>
            <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
              Wetland
            </div>

            <ConfigToggle
              label="Wetland Mask"
              checked={wetlandMaskEnabled}
              onChange={() => onChange("wetlandMaskEnabled")}
            />
          </div>
        )}

        {/* Metrics */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Metrics
          </div>

          <div className="space-y-2">
            <ConfigToggle
              label="Degradation"
              checked={degradationEnabled}
              onChange={() => onChange("degradationEnabled")}
            />

            <ConfigToggle
              label="Recovery"
              checked={recoveryEnabled}
              onChange={() => onChange("recoveryEnabled")}
            />

            <ConfigToggle
              label="Area Statistics"
              checked={areaStatisticsEnabled}
              onChange={() => onChange("areaStatisticsEnabled")}
            />
          </div>
        </div>

        {/* Validation */}
        <div>
          <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
            Validation
          </div>

          <div className="space-y-2">
            <ConfigToggle
              label="Input QA"
              checked={inputQaEnabled}
              onChange={() => onChange("inputQaEnabled")}
            />

            <ConfigToggle
              label="Temporal QA"
              checked={temporalQaEnabled}
              onChange={() => onChange("temporalQaEnabled")}
            />

            <ConfigToggle
              label="Change QA"
              checked={changeQaEnabled}
              onChange={() => onChange("changeQaEnabled")}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
