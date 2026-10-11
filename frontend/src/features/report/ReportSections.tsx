const reportSections = [
  {
    number: "01",
    title: "Executive Summary",
    description:
      "High-level summary of the T1 → T2 analysis and principal findings.",
  },
  {
    number: "02",
    title: "Data",
    description:
      "T1/T2 datasets, acquisition information, raster metadata, and input details.",
  },
  {
    number: "03",
    title: "Methods",
    description:
      "Model configuration, transition analysis, CVA, wetland masking, and area estimation.",
  },
  {
    number: "04",
    title: "Classification Results",
    description:
      "Health-class areas and T1/T2 classification outputs.",
  },
  {
    number: "05",
    title: "Transition Analysis",
    description:
      "Complete 4×4 T1 → T2 transition matrix and associated statistics.",
  },
  {
    number: "06",
    title: "Spectral Change",
    description:
      "Observed ΔNDVI, ΔGNDVI, ΔNDRE, and CVA statistics.",
  },
  {
    number: "07",
    title: "Degradation & Recovery",
    description:
      "Transition-derived degradation, canopy loss, recovery, and related metrics.",
  },
  {
    number: "08",
    title: "Wetland Extent",
    description:
      "Wetland extent evidence and distinction between vegetation degradation and physical depletion.",
  },
  {
    number: "09",
    title: "Validation & QA",
    description:
      "Input, temporal, change, model, and independent validation results.",
  },
  {
    number: "10",
    title: "Limitations",
    description:
      "Known limitations, unavailable evidence, uncertainty, and interpretation constraints.",
  },
  {
    number: "11",
    title: "Metadata",
    description:
      "Run information, configuration, provenance, and processing metadata.",
  },
];

export default function ReportSections() {
  return (
    <section className="rounded-lg border border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-5 py-4">
        <div className="text-sm font-medium text-[var(--text)]">
          Report contents
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Sections included in the final publication-ready report.
        </div>
      </div>

      <div className="divide-y divide-[var(--border)]">
        {reportSections.map((section) => (
          <div
            key={section.number}
            className="flex gap-4 px-5 py-4"
          >
            <div className="w-7 shrink-0 text-[10px] font-semibold text-[var(--text-secondary)]">
              {section.number}
            </div>

            <div>
              <div className="text-sm font-medium text-[var(--text)]">
                {section.title}
              </div>

              <div className="mt-1 text-xs leading-5 text-[var(--text-secondary)]">
                {section.description}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}