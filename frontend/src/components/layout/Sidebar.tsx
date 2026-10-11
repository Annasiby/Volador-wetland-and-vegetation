import type { ReactNode } from "react";

type NavigationItem = {
  label: string;
  icon: ReactNode;
};

type SidebarProps = {
  activeItem: string;
  onNavigate: (label: string) => void;
};

const navigationItems: NavigationItem[] = [
  { label: "Overview", icon: "⌂" },
  { label: "Datasets", icon: "▣" },
  { label: "Prepare", icon: "⬡" },
  { label: "Analysis", icon: "◇" },
  { label: "Results", icon: "▥" },
  { label: "Validation", icon: "✓" },
  { label: "Report", icon: "□" },
];

export default function Sidebar({
  activeItem,
  onNavigate,
}: SidebarProps) {
  return (
    <aside className="flex h-screen w-60 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-6 py-5">
        <div className="text-[15px] font-semibold tracking-[0.08em] text-[var(--text)]">
          GIS ENGINE
        </div>

        <div className="mt-1 text-xs text-[var(--text-secondary)]">
          Wetland analysis
        </div>
      </div>

      <div className="px-4 pt-6">
        <div className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-secondary)]">
          Workspace
        </div>

        <nav className="space-y-1">
          {navigationItems.map((item) => {
            const active = activeItem === item.label;

            return (
              <button
                key={item.label}
                type="button"
                onClick={() => onNavigate(item.label)}
                className={[
                  "flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors",
                  active
                    ? "bg-[var(--accent-soft)] font-medium text-[var(--accent)]"
                    : "text-[var(--text-secondary)] hover:bg-[var(--background)] hover:text-[var(--text)]",
                ].join(" ")}
              >
                <span
                  className={[
                    "flex h-5 w-5 items-center justify-center text-xs",
                    active
                      ? "text-[var(--accent)]"
                      : "text-[var(--text-secondary)]",
                  ].join(" ")}
                >
                  {item.icon}
                </span>

                {item.label}
              </button>
            );
          })}
        </nav>
      </div>

      <div className="mt-auto border-t border-[var(--border)] p-4">
        <div className="rounded-md bg-[var(--background)] p-3">
          <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--text-secondary)]">
            Engine
          </div>

          <div className="mt-1 flex items-center gap-2 text-xs text-[var(--text)]">
            <span className="h-2 w-2 rounded-full bg-[var(--success)]" />
            Ready
          </div>
        </div>
      </div>
    </aside>
  );
}