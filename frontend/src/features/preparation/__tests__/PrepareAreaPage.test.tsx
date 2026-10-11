import { render, screen, cleanup, act } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import PrepareAreaPage from "../PrepareAreaPage";
import { ProjectContext } from "../../../context/projectContextObject";
import type { ProjectContextType } from "../../../context/projectContextObject";

// ── Mock OpenLayers Classes ────────────────────────────────────────────────
vi.mock("ol/Map", () => {
  return {
    default: class MockMap {
      setTarget = vi.fn();
      dispose = vi.fn();
      getView = vi.fn().mockReturnValue({
        getProjection: vi.fn().mockReturnValue({ getCode: () => "EPSG:3857" }),
        fit: vi.fn(),
      });
      setView = vi.fn();
      addInteraction = vi.fn();
      removeInteraction = vi.fn();
      on = vi.fn();
    },
  };
});

vi.mock("ol/View", () => {
  return {
    default: class MockView {
      getProjection = vi.fn().mockReturnValue({ getCode: () => "EPSG:3857" });
      fit = vi.fn();
    },
  };
});

vi.mock("ol/layer/Vector", () => {
  return { default: class MockVectorLayer {} };
});

vi.mock("ol/source/Vector", () => {
  return {
    default: class MockVectorSource {
      clear = vi.fn();
      addFeature = vi.fn();
    },
  };
});

vi.mock("ol/layer/Image", () => {
  return {
    default: class MockImageLayer {
      setSource = vi.fn();
    },
  };
});

vi.mock("ol/source/ImageStatic", () => {
  return { default: class MockImageStatic {} };
});

vi.mock("ol/interaction/Draw", () => {
  return {
    default: class MockDraw {
      on = vi.fn();
    },
  };
});

vi.mock("ol/interaction/Modify", () => {
  return {
    default: class MockModify {
      on = vi.fn();
    },
  };
});

vi.mock("ol/proj", () => ({
  addProjection: vi.fn(),
  get: vi.fn().mockReturnValue(null),
}));

vi.mock("ol/proj/Projection", () => {
  return {
    default: class MockProjection {
      getCode() {
        return "mock-proj";
      }
    },
  };
});

vi.mock("ol/style", () => {
  return {
    Style: class MockStyle {},
    Fill: class MockFill {},
    Stroke: class MockStroke {},
    Circle: class MockCircle {},
  };
});

vi.mock("ol/format/GeoJSON", () => {
  return {
    default: class MockGeoJSON {
      writeGeometry = vi.fn().mockReturnValue('{"type":"Polygon","coordinates":[]}');
      readFeature = vi.fn().mockReturnValue({
        getGeometry: vi.fn(),
        set: vi.fn(),
      });
    },
  };
});

// ── Mock API client ────────────────────────────────────────────────────────
vi.mock("../../../api/client", () => ({
  getDatasetPreviewImageUrl: vi.fn(() => "http://localhost/preview.png"),
  getDatasetPreviewInfo: vi.fn().mockResolvedValue({ width: 100, height: 100 }),
  previewPreparationMetrics: vi.fn().mockResolvedValue(null),
}));

// ── Test helpers ───────────────────────────────────────────────────────────
const mockContextValue: ProjectContextType = {
  project: {
    id: "test-prj",
    name: "Test Project",
    aoi_name: "Test AOI",
    analysis_mode: "vegetation",
    status: "CREATED",
    datasets_ready: true,
    is_locked: false,
  } as any,
  validation: null,
  runStatus: null,
  results: null,
  layers: [],
  activeLayerId: "t1",
  loading: false,
  error: null,
  manualExclusions: [],
  preparation: {
    project_id: "test-prj",
    status: "DRAFT",
    is_locked: false,
    inclusion: null,
    manual_exclusions: [],
    metrics: {} as any,
    updated_at: new Date().toISOString(),
  },
  addManualExclusion: vi.fn(),
  updateManualExclusion: vi.fn(),
  removeManualExclusion: vi.fn(),
  clearManualExclusions: vi.fn(),
  clearAll: vi.fn().mockResolvedValue(undefined),
  setActiveLayerId: vi.fn(),
  triggerValidation: vi.fn().mockResolvedValue(undefined),
  uploadBand: vi.fn().mockResolvedValue(undefined),
  triggerRun: vi.fn().mockResolvedValue(undefined),
  refreshResults: vi.fn().mockResolvedValue(undefined),
  fetchPreparation: vi.fn().mockResolvedValue(undefined),
  saveDraft: vi.fn().mockResolvedValue(undefined),
  commitArea: vi.fn().mockResolvedValue(undefined),
  unlockArea: vi.fn().mockResolvedValue(undefined),
};

async function renderWithContext(ui: React.ReactElement, ctxOverrides: Partial<ProjectContextType> = {}) {
  const value = { ...mockContextValue, ...ctxOverrides };
  let result: any;
  await act(async () => {
    result = render(
      <ProjectContext.Provider value={value}>
        {ui}
      </ProjectContext.Provider>
    );
  });
  return result;
}

describe("PrepareAreaPage Mode Awareness", () => {
  afterEach(() => {
    cleanup();
  });

  // Test 1: Vegetation mode does not show T2 selector
  it("1. Vegetation: does not show T2 / temporal toggle", async () => {
    await renderWithContext(<PrepareAreaPage analysisMode="vegetation" />);
    expect(screen.queryByTestId("wetland-t1-t2-selector")).not.toBeInTheDocument();
  });

  // Test 2: Vegetation mode does not require wetland boundary
  it("2. Vegetation: does not require wetland boundary", async () => {
    await renderWithContext(<PrepareAreaPage analysisMode="vegetation" />);
    expect(screen.queryByText(/wetland boundary/i)).not.toBeInTheDocument();
  });

  // Test 3: Vegetation mode uses one scene
  it("3. Vegetation: uses one scene with Vegetation Scene label", async () => {
    await renderWithContext(<PrepareAreaPage analysisMode="vegetation" />);
    const label = screen.getByTestId("vegetation-scene-label");
    expect(label).toBeInTheDocument();
    expect(label).toHaveTextContent("Vegetation Scene");
  });

  // Test 4: Wetland mode shows T1/T2 selector
  it("4. Wetland: shows T1/T2 selector", async () => {
    await renderWithContext(<PrepareAreaPage analysisMode="wetland" />);
    expect(screen.getByTestId("wetland-t1-t2-selector")).toBeInTheDocument();
  });

  // Test 5: Wetland mode requires expected inputs (T1 and T2 buttons present)
  it("5. Wetland: provides both T1 and T2 base selection buttons", async () => {
    await renderWithContext(<PrepareAreaPage analysisMode="wetland" />);
    const selector = screen.getByTestId("wetland-t1-t2-selector");
    expect(selector).toBeInTheDocument();
    expect(selector).toHaveTextContent("T1");
    expect(selector).toHaveTextContent("T2");
  });

  // Test 6: Switching Wetland -> Vegetation removes T2 controls
  it("6. Switching Wetland -> Vegetation removes T2 controls", async () => {
    const { rerender } = await renderWithContext(<PrepareAreaPage analysisMode="wetland" />);
    expect(screen.getByTestId("wetland-t1-t2-selector")).toBeInTheDocument();
    expect(screen.queryByTestId("vegetation-scene-label")).not.toBeInTheDocument();

    await act(async () => {
      rerender(
        <ProjectContext.Provider value={mockContextValue}>
          <PrepareAreaPage analysisMode="vegetation" />
        </ProjectContext.Provider>
      );
    });

    expect(screen.queryByTestId("wetland-t1-t2-selector")).not.toBeInTheDocument();
    expect(screen.getByTestId("vegetation-scene-label")).toBeInTheDocument();
  });

  // Test 7: Switching Vegetation -> Wetland restores T2 controls
  it("7. Switching Vegetation -> Wetland restores T2 controls", async () => {
    const { rerender } = await renderWithContext(<PrepareAreaPage analysisMode="vegetation" />);
    expect(screen.queryByTestId("wetland-t1-t2-selector")).not.toBeInTheDocument();
    expect(screen.getByTestId("vegetation-scene-label")).toBeInTheDocument();

    await act(async () => {
      rerender(
        <ProjectContext.Provider value={mockContextValue}>
          <PrepareAreaPage analysisMode="wetland" />
        </ProjectContext.Provider>
      );
    });

    expect(screen.getByTestId("wetland-t1-t2-selector")).toBeInTheDocument();
    expect(screen.queryByTestId("vegetation-scene-label")).not.toBeInTheDocument();
  });

  // Test 8: Inclusion/exclusion functionality works in both modes
  it("8. Inclusion and exclusion drawing controls exist and work in both modes", async () => {
    // In vegetation mode
    const { unmount } = await renderWithContext(<PrepareAreaPage analysisMode="vegetation" />);
    expect(screen.getByRole("button", { name: /\+ Draw AOI/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /\+ Add Exclusion/i })).toBeInTheDocument();
    unmount();

    // In wetland mode
    await renderWithContext(<PrepareAreaPage analysisMode="wetland" />);
    expect(screen.getByRole("button", { name: /\+ Draw AOI/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /\+ Add Exclusion/i })).toBeInTheDocument();
  });
});

