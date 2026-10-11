import { useState } from "react";
import { useProject } from "../../context/useProject";

type TopBarProps = {
  activeItem: string;
};

export default function TopBar({ activeItem }: TopBarProps) {
  const { project, validation, clearAll, loading } = useProject();
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  const isReady = project?.datasets_ready || validation?.valid;

  const handleClearAll = async () => {
    try {
      await clearAll();
      setShowClearConfirm(false);
    } catch (err) {
      console.error("Failed to clear session:", err);
    }
  };

  return (
    <>
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-7">
        <div>
          <div className="text-xs text-[var(--text-secondary)]">
            Wetland Change Analysis
          </div>

          <div className="mt-0.5 text-sm font-medium text-[var(--text)]">
            {activeItem}
          </div>
        </div>

        <div className="flex items-center gap-5">
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-[0.12em] text-[var(--text-secondary)]">
              Project
            </div>

            <div className="text-sm font-medium text-[var(--text)]">
              {project ? project.name : "No project loaded"}
            </div>
          </div>

          <div className="h-8 w-px bg-[var(--border)]" />

          <div className="flex items-center gap-2 text-xs text-[var(--text-secondary)]">
            <span
              className={`h-2 w-2 rounded-full ${
                isReady ? "bg-[var(--success)]" : "bg-amber-500"
              }`}
            />

            {project
              ? isReady
                ? "READY"
                : "CREATED"
              : "NO PROJECT"}
          </div>

          <div className="h-8 w-px bg-[var(--border)]" />

          <button
            type="button"
            onClick={() => setShowClearConfirm(true)}
            disabled={!project || loading}
            className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition hover:border-red-400 hover:text-red-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Clear All
          </button>
        </div>
      </header>

      {showClearConfirm && (
        <div className="fixed inset-0 z-[1000] flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-md rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 shadow-xl">
            <div className="text-base font-semibold text-[var(--text)]">
              Clear current session?
            </div>

            <p className="mt-3 text-sm leading-6 text-[var(--text-secondary)]">
              This will clear uploaded datasets, validation state,
              analysis runs and results, and manual exclusions.
              Your project will remain available.
            </p>

            <p className="mt-2 text-xs text-red-500">
              This action cannot be undone.
            </p>

            <div className="mt-6 flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setShowClearConfirm(false)}
                disabled={loading}
                className="rounded-md border border-[var(--border)] px-4 py-2 text-xs font-medium text-[var(--text-secondary)] hover:text-[var(--text)]"
              >
                Cancel
              </button>

              <button
                type="button"
                onClick={handleClearAll}
                disabled={loading}
                className="rounded-md bg-red-600 px-4 py-2 text-xs font-medium text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Clearing..." : "Clear All"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}