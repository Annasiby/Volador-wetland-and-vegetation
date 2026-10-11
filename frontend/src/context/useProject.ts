/**
 * context/useProject.ts
 *
 * Custom hook to consume ProjectContext.
 */
import { useContext } from "react";
import { ProjectContext } from "./projectContextObject";

export function useProject() {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error("useProject must be used within a ProjectProvider");
  }
  return context;
}
