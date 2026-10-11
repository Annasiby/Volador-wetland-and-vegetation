/**
 * context/projectContextObject.ts
 *
 * Separated from ProjectContext.tsx to satisfy react-refresh/only-export-components:
 * Context definitions and types are kept in non-component files.
 */
import { createContext } from "react";
import type {
  Project,
  ValidationResult,
  RunStatus,
  MapLayer,
  AnalysisResults,
  PreparationState,
} from "../api/client";
import type { ManualExclusion } from "./ProjectContext";

export interface ProjectContextType {
  project: Project | null;
  validation: ValidationResult | null;
  runStatus: RunStatus | null;
  results: AnalysisResults | null;
  layers: MapLayer[];
  activeLayerId: string;
  loading: boolean;
  error: string | null;
  manualExclusions: ManualExclusion[];
  preparation: PreparationState | null;
  addManualExclusion: (exclusion: ManualExclusion) => void;
  updateManualExclusion: (exclusion: ManualExclusion) => void;
  removeManualExclusion: (id: string) => void;
  clearManualExclusions: () => void;
  clearAll: () => Promise<void>;
  setActiveLayerId: (id: string) => void;
  triggerValidation: (mode?: "vegetation" | "wetland") => Promise<void>;
  uploadBand: (
    timePoint: "t1" | "t2",
    band: "green" | "red" | "rededge" | "nir" | "boundary",
    file: File,
    mode?: "vegetation" | "wetland"
  ) => Promise<void>;
  triggerRun: (config?: Record<string, any>) => Promise<void>;
  refreshResults: () => Promise<void>;
  fetchPreparation: (projectId?: string) => Promise<PreparationState | undefined>;
  saveDraft: (inclusion: Record<string, any> | null, exclusions: ManualExclusion[]) => Promise<void>;
  commitArea: () => Promise<void>;
  unlockArea: () => Promise<void>;
}

export const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

