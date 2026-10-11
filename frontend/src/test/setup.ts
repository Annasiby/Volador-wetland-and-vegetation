import "@testing-library/jest-dom";
import { vi } from "vitest";

// Mock canvas getContext
HTMLCanvasElement.prototype.getContext = vi.fn() as any;

// Mock URL.createObjectURL and revokeObjectURL
if (typeof window !== "undefined") {
  window.URL.createObjectURL = vi.fn(() => "blob:mock");
  window.URL.revokeObjectURL = vi.fn();
}

// Mock ResizeObserver
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
} as any;
